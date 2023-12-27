import datetime
import json
import logging
import math
import os
import time

import numpy as np

import mindspore as ms
from mindspore import Parameter, ParameterTuple, Tensor, context, nn, ops
from mindspore.amp import DynamicLossScaler, LossScaler, StaticLossScaler, all_finite
from mindspore.nn.optim.optimizer import Optimizer, opt_init_args_register
from mindspore.train import Callback, Model, save_checkpoint
from mindspore.train.amp import _OutputTo16, _OutputTo32, custom_mixed_precision, get_black_list, get_white_list

try:
    import wandb
except ImportError:
    wandb = None

from .distributed import is_master
from .zero_shot import zero_shot_eval

LATEST_CHECKPOINT_NAME = "epoch_latest.ckpt"
_logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


_adam_opt = ops.MultitypeFuncGraph("adam_opt")


@_adam_opt.register(
    "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool"
)
def adam_opt(beta1, beta2, beta1_power, beta2_power, eps, lr, weight_decay, param, m, v, grad, decay_flag):
    """
    Update parameters.
    Args:
        beta1 (Tensor): The exponential decay rate for the 1st moment estimations. Should be in range (0.0, 1.0).
        beta2 (Tensor): The exponential decay rate for the 2nd moment estimations. Should be in range (0.0, 1.0).
        beta1_power (Tensor): beta1 ** t, where t is the optimization step.
        beta2_power (Tensor): beta2 ** t, where t is the optimization step.
        eps (Tensor): Term added to the denominator to improve numerical stability. Should be greater than 0.
        lr (Tensor): Learning rate.
        weight_decay (Tensor): Weight decay. Should be equal to or greater than 0.
        param (Tensor): Parameters.
        m (Tensor): m value of parameters.
        v (Tensor): v value of parameters.
        grad (Tensor): Gradient of parameters.
        decay_flag (bool): Applies weight decay or not.
    Returns:
        Tensor, the new value of parameter after updating.
    """
    success = True
    next_m = ops.mul(beta1, m) + ops.mul(1.0 - beta1, grad)
    next_v = ops.mul(beta2, v) + ops.mul(1.0 - beta2, ops.square(grad))
    regulate_m = next_m / (1.0 - beta1_power)
    regulate_v = next_v / (1.0 - beta2_power)

    update = regulate_m / (eps + ops.sqrt(regulate_v))
    if decay_flag:
        update = ops.mul(lr, update + ops.mul(weight_decay, param))
    else:
        update = ops.mul(lr, update)
    next_param = param - ops.reshape(update, ops.shape(param))

    success = ops.depend(success, ops.assign(param, next_param))
    success = ops.depend(success, ops.assign(m, next_m))
    success = ops.depend(success, ops.assign(v, next_v))
    return success


class AdamW(Optimizer):
    @opt_init_args_register
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, loss_scale=1.0):
        super().__init__(lr, params, weight_decay, loss_scale)
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        self.beta1 = Tensor(betas[0], ms.float32)
        self.beta2 = Tensor(betas[1], ms.float32)
        self.beta1_power = Parameter(Tensor(1.0, dtype=ms.float32), name="beta1_power")
        self.beta2_power = Parameter(Tensor(1.0, dtype=ms.float32), name="beta2_power")
        self.eps = Tensor(eps, ms.float32)
        self.moment1 = self._parameters.clone(prefix="adam_m", init="zeros")
        self.moment2 = self._parameters.clone(prefix="adam_v", init="zeros")
        assert self.use_parallel is False, "Parallel optimizer is not supported!"

    def construct(self, grads):
        grads = self.flatten_gradients(grads)
        grads = self.scale_grad(grads)
        lr = self.get_lr()
        weight_decay = self.get_weight_decay()
        self.assignadd(self.global_step, self.global_step_increase_tensor)

        params = self._parameters
        moment1 = self.moment1
        moment2 = self.moment2
        beta1_power = self.beta1_power * self.beta1
        self.beta1_power = beta1_power
        beta2_power = self.beta2_power * self.beta2
        self.beta2_power = beta2_power

        if self.is_group:
            if self.is_group_lr:
                success = self.hyper_map(
                    ops.partial(_adam_opt, self.beta1, self.beta2, beta1_power, beta2_power, self.eps),
                    lr,
                    weight_decay,
                    params,
                    moment1,
                    moment2,
                    grads,
                    self.decay_flags,
                )
            else:
                success = self.hyper_map(
                    ops.partial(_adam_opt, self.beta1, self.beta2, beta1_power, beta2_power, self.eps, lr),
                    weight_decay,
                    params,
                    moment1,
                    moment2,
                    grads,
                    self.decay_flags,
                )
        else:
            success = self.hyper_map(
                ops.partial(_adam_opt, self.beta1, self.beta2, beta1_power, beta2_power, self.eps, lr, weight_decay),
                params,
                moment1,
                moment2,
                grads,
                self.decay_flags,
            )
        return success


class TrainStep(nn.Cell):
    """Training step with loss scale.

    The steps of model optimization are performed in the following order:
        1. calculate grad
        2. allreduce grad
        3. clip grad [optional]
        4. call optimizer
    """

    def __init__(
        self,
        network: nn.Cell,
        criterion: nn.Cell,
        optimizer: nn.Optimizer,
        scaler: LossScaler,
        grad_clip_norm: float = None,
    ):
        super().__init__()
        self.network = network.set_grad()
        self.criterion = criterion.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.scaler = scaler
        if isinstance(self.scaler, StaticLossScaler):
            self.drop_overflow = False
        elif isinstance(self.scaler, DynamicLossScaler):
            self.drop_overflow = True
        else:
            raise NotImplementedError(f"Unsupported scaler: {type(self.scaler)}")
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode == context.ParallelMode.STAND_ALONE:
            self.grad_reducer = ops.identity
        elif self.parallel_mode in (context.ParallelMode.DATA_PARALLEL, context.ParallelMode.HYBRID_PARALLEL):
            self.grad_reducer = nn.DistributedGradReducer(self.weights)
        else:
            raise NotImplementedError(f"When creating reducer, Got Unsupported parallel mode: {self.parallel_mode}")
        if isinstance(network, nn.Cell) and network.jit_config_dict:
            self._jit_config_dict = network.jit_config_dict

        self.clip_grad = grad_clip_norm is not None
        self.clip_value = grad_clip_norm
        self.logit_scale = None
        for n, p in self.network.parameters_and_names():
            # TODO: _OutputTo16/32 will add disgusting prefix '_backbone' on param name, unwrap it before saving!
            if n == "logit_scale" or n == "_backbone.logit_scale":
                self.logit_scale = p
        assert self.logit_scale is not None, "Cannot fetch parameter `logit_scale` from network."

        def forward_fn(image, text):
            image_features, text_features, logit_scale = network(image, text)
            loss = criterion(image_features, text_features, logit_scale)
            loss = scaler.scale(loss)
            return loss, image_features, text_features, logit_scale

        self.grad_fn = ops.value_and_grad(forward_fn, grad_position=None, weights=self.weights, has_aux=True)

    def update(self, loss, grads):
        if self.clip_grad:
            loss = ops.depend(loss, self.optimizer(ops.clip_by_global_norm(grads, clip_norm=self.clip_value)))
        else:
            loss = ops.depend(loss, self.optimizer(grads))
        return loss

    def construct(self, *inputs):
        (loss, image_features, text_features, logit_scale), grads = self.grad_fn(*inputs)
        grads = self.grad_reducer(grads)
        loss = self.scaler.unscale(loss)
        grads = self.scaler.unscale(grads)

        if self.drop_overflow:
            status = all_finite(grads)
            if status:
                loss = self.update(loss, grads)
            loss = ops.depend(loss, self.scaler.adjust(status))
        else:
            loss = self.update(loss, grads)

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        loss = ops.depend(loss, ops.assign(self.logit_scale, ops.clamp(self.logit_scale, 0, 4.6052)))

        # if you want to get anything about training status, return it from here and logging it outside!
        return loss, image_features, text_features, logit_scale


def auto_mixed_precision(network, amp_level):
    if amp_level == "O0":
        network.to_float(ms.float32)
    elif amp_level == "O1":
        white_list = get_white_list()
        network = custom_mixed_precision(network, white_list=white_list)
    elif amp_level == "O2":
        black_list = get_black_list()
        black_list += [
            nn.GroupNorm,
            nn.SyncBatchNorm,
            nn.Softmax,
            nn.LogSoftmax,
            nn.LogSigmoid,
            nn.CrossEntropyLoss,
            nn.SoftmaxCrossEntropyWithLogits,
        ]
        network = custom_mixed_precision(network, black_list=black_list)
    elif amp_level == "O3":
        network.to_float(ms.float16)
        network = _OutputTo32(network)
    else:
        raise ValueError("The amp level {} is not supported".format(amp_level))

    return network


def build_trainer(
    network: nn.Cell,
    criterion: nn.Cell,
    optimizer: nn.Cell,
    amp_level: str = "O0",
    scaler: LossScaler = None,
    grad_clip_norm: float = None,
):
    """Build Trainer.

    Args:
        network: The backbone network to train, evaluate or predict.
        criterion: The function of calculating loss.
        optimizer: The optimizer for training.
        amp_level: The level of auto mixing precision training.
            'O0': single precision for all, 'O1': half precision for white list & single precision for others
            'O2': single precision for black list & half precision for others, 'O3': half precision for all.
        scaler: The manager helps perform the steps of gradient scaling conveniently.
        grad_clip_norm: The value at which to clip gradients. Disable if it's None.

    Returns:
        mindspore.Model

    """
    network = auto_mixed_precision(network, amp_level=amp_level)
    criterion = criterion.to_float(ms.float32)
    train_step_cell = TrainStep(
        network=network,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        grad_clip_norm=grad_clip_norm,
    ).set_train()
    trainer = Model(train_step_cell)
    return trainer


class CallbackForCLIP(Callback):
    def __init__(self, args, data, tokenizer, trainer, writer=None, start_epoch=0):
        self.args = args
        self.data = data
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.writer = writer
        self.start_epoch = start_epoch
        # initialize the following members to make linter happy
        self.step_ts = -1.0
        self.epoch_ts = -1.0
        self.num_batches_per_epoch = -1
        self.num_samples_per_epoch = -1
        self.sample_digits = -1
        self.losses_m = {}
        self.batch_time_m = AverageMeter()

    def _get_network_from_cbp(self, cb_params):
        network = cb_params.train_network if cb_params.mode == "train" else cb_params.eval_network
        if cb_params.dataset_sink_mode:  # train_network is connected to DatasetHelper when data_sink is enable.
            return network.network  # throw an error at the beginning of 1st epoch for helper is not connected. WTF!!!
        else:
            return network

    def _get_optimizer_from_cbp(self, cb_params):
        optimizer = cb_params.optimizer
        if optimizer is None:
            network = cb_params.train_network if cb_params.mode == "train" else cb_params.eval_network
            if cb_params.dataset_sink_mode:
                optimizer = network.network.optimizer
            else:
                optimizer = network.optimizer
        if optimizer is None or not isinstance(optimizer, nn.Optimizer):
            _logger.warning(f"Failed to get valid optimizer from callback, got {type(optimizer)}")
            optimizer = None
        return optimizer

    def _get_lr_from_cbp(self, cb_params):
        optimizer = self._get_optimizer_from_cbp(cb_params)
        if optimizer.global_step < 1:
            _logger.warning(
                "`global_step` of optimizer is less than 1. It seems to be a overflow at the first step. "
                "If you keep seeing this message, it means that the optimizer never actually called."
            )
            optim_step = Tensor((0,), ms.int32)
        else:  # if the optimizer is successfully called, the global_step will actually be the value of next step.
            optim_step = optimizer.global_step - 1
        if optimizer.dynamic_lr:
            if isinstance(optimizer.learning_rate, nn.CellList):
                # return the learning rates of the first parameter if dynamic_lr
                lr = optimizer.learning_rate[0](optim_step)[0]
            else:
                lr = optimizer.learning_rate(optim_step)[0]
        else:
            lr = optimizer.learning_rate
        return lr

    def on_train_epoch_begin(self, run_context):
        self.epoch_ts = time.time()
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num + self.start_epoch
        if is_master(self.args):
            _logger.info(f"Start epoch {cur_epoch - 1}")
        self.data["train"].set_epoch(cur_epoch - 1)  # set epoch in process safe manner via sampler or shared_epoch
        dataloader = self.data["train"].dataloader
        self.num_batches_per_epoch = dataloader.num_batches // self.args.accum_freq
        self.num_samples_per_epoch = dataloader.num_samples
        self.sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
        self.losses_m = {}
        self.batch_time_m = AverageMeter()

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        num_epochs = cb_params.epoch_num
        cur_epoch = cb_params.cur_epoch_num + self.start_epoch
        epoch, completed_epoch = cur_epoch - 1, cur_epoch
        train_time = time.time() - self.epoch_ts

        val_time = 0
        if any(v in self.data for v in ("val", "imagenet-val", "imagenet-v2")):
            val_time = time.time()
            model = self._get_network_from_cbp(cb_params).network  # TrainStep -> network(backbone, amp-ed)
            if hasattr(model, "_backbone"):  # _OutputTo32 will add a disgusting prefix '_backbone'
                model = model._backbone  # TrainStep -> network(backbone, amp-ed, last _outputTo32 unwrapped)
            evaluate(model, self.data, completed_epoch, self.args, tb_writer=self.writer, tokenizer=self.tokenizer)
            val_time = time.time() - val_time

        # Saving checkpoints.
        if self.args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": self.args.name,
            }
            if completed_epoch == self.args.epochs or (
                self.args.save_frequency > 0 and (completed_epoch % self.args.save_frequency) == 0
            ):
                save_checkpoint(  # TrainStep with network(backbone), criterion, optimizer, scaler, ema, accum_grad
                    self._get_network_from_cbp(cb_params),
                    os.path.join(self.args.checkpoint_path, f"epoch_{completed_epoch}.ckpt"),
                    append_dict=checkpoint_dict,
                )
            if self.args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(self.args.checkpoint_path, f"epoch_{completed_epoch - 1}.ckpt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)
            if self.args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(self.args.checkpoint_path, "tmp.ckpt")
                latest_save_path = os.path.join(self.args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                save_checkpoint(self._get_network_from_cbp(cb_params), tmp_save_path, append_dict=checkpoint_dict)
                os.replace(tmp_save_path, latest_save_path)

        if is_master(self.args):
            total_time = int(time.time() - self.epoch_ts)
            _logger.info(
                f"Total time since last epoch: {datetime.timedelta(seconds=total_time)}"
                f"(train: {train_time:.6f}s, val: {val_time:.6f}s), "
                f"ETA: {datetime.timedelta(seconds=(num_epochs - cur_epoch) * total_time)}"
            )

    def on_train_step_begin(self, run_context):
        self.step_ts = time.time()

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        num_epochs = cb_params.epoch_num
        num_batches = cb_params.batch_num
        num_steps = num_batches * num_epochs
        # cur_x start from 1, end at num_xs, range: [1, num_xs]
        cur_step = cb_params.cur_step_num + self.start_epoch * num_batches
        cur_epoch = cb_params.cur_epoch_num + self.start_epoch
        cur_batch = (cur_step - 1) % num_batches + 1

        i, epoch = cur_batch - 1, cur_epoch - 1
        i_accum = i // self.args.accum_freq
        step = self.num_batches_per_epoch * epoch + i_accum

        batch_time = time.time() - self.step_ts
        if cb_params.dataset_sink_mode:  # if data_sink is enable, this hook is actually invoked at end of epoch
            batch_time = batch_time / self.num_batches_per_epoch
        self.batch_time_m.update(batch_time)
        batch_count = i_accum + 1
        if is_master(self.args) and (
            i_accum % self.args.log_every_n_steps == 0 or batch_count == self.num_batches_per_epoch
        ):
            num_samples = batch_count * self.args.batch_size * self.args.accum_freq * self.args.world_size
            percent_complete = 100.0 * batch_count / self.num_batches_per_epoch

            outputs = cb_params.net_outputs  # todo: outputs is hardcode here
            # NOTE loss is coarsely sampled, just master node and per log update
            losses = {"loss": outputs[0]}
            for key, val in losses.items():
                if key not in self.losses_m:
                    self.losses_m[key] = AverageMeter()
                self.losses_m[key].update(val.numpy().item(), self.args.batch_size)

            logit_scale_scalar = outputs[3].numpy().item()
            lr_scalar = self._get_lr_from_cbp(cb_params).numpy().item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in self.losses_m.items()
                ]
            )
            samples_per_second = (
                self.args.accum_freq * self.args.batch_size * self.args.world_size / self.batch_time_m.val
            )
            samples_per_second_per_gpu = self.args.accum_freq * self.args.batch_size / self.batch_time_m.val
            _logger.info(
                f"Train Epoch: {epoch} [{num_samples:>{self.sample_digits}}/{self.num_samples_per_epoch} ({percent_complete:>3.0f}%)] "  # noqa: E501
                f"Batch (t): {self.batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "  # noqa: E501
                f"LR: {lr_scalar:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "batch_time": self.batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": lr_scalar,
            }
            log_data.update({name: val.val for name, val in self.losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if self.writer is not None:
                for name, val in log_data.items():
                    self.writer.add_value("scalar", name, Tensor(val))
                self.writer.record(step)

            if self.args.wandb:
                assert wandb is not None, "Please install wandb."
                log_data["step"] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            self.batch_time_m.reset()


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    # if not is_master(args):  # todo: let each rank do testing to avoid waiting
    #     return metrics
    model.set_train(False)
    model.phase = "eval"

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    if "val" in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data["val"].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        memory = (samples_per_val * 1024 * 2 + samples_per_val * samples_per_val) * 4 / 1024 / 1024 / 1024
        _logger.warning(f"Validation will use at least {memory:.6f}(GB) CPU memory.")
        # TODO: Offload to CPU as numpy.ndarray for logits calculations. How to optimize?
        for i, batch in enumerate(dataloader.create_tuple_iterator()):
            images, texts = batch
            image_features, text_features, logit_scale = model(images, texts)
            image_features, text_features, logit_scale = (
                image_features.to(ms.float32),
                text_features.to(ms.float32),
                logit_scale.to(ms.float32),
            )
            # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
            # however, system RAM is easily exceeded and compute time becomes problematic
            all_image_features.append(image_features.numpy())
            all_text_features.append(text_features.numpy())
            logit_scale = logit_scale.mean()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            batch_size = images.shape[0]
            labels = ops.arange(0, batch_size, dtype=ms.int64)
            total_loss = (ops.cross_entropy(logits_per_image, labels) + ops.cross_entropy(logits_per_text, labels)) / 2

            gen_loss = maybe_compute_generative_loss({})  # only coca gives "logits" and "labels" in model_out

            cumulative_loss += total_loss * batch_size
            if gen_loss is not None:
                cumulative_gen_loss += gen_loss * batch_size
            num_samples += batch_size
            # if is_master(args) and (i % 100) == 0:
            #     _logger.info(
            #         f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
            #         f"Clip Loss: {cumulative_loss.numpy().item() / num_samples:.6f}\t"
            #     )
            #     if gen_loss is not None:
            #         _logger.info(f"Generative Loss: {cumulative_gen_loss.numpy().item() / num_samples:.6f}\t")

        val_metrics = get_clip_metrics(
            image_features=np.concatenate(all_image_features),
            text_features=np.concatenate(all_text_features),
            logit_scale=logit_scale.numpy(),
        )
        loss = cumulative_loss / num_samples
        metrics.update(
            {**val_metrics, "clip_val_loss": loss.numpy().item(), "epoch": epoch, "num_samples": num_samples}
        )
        if gen_loss is not None:
            gen_loss = cumulative_gen_loss / num_samples
            metrics.update({"val_generative_loss": gen_loss.numpy().item()})

    if not metrics:
        return metrics

    _logger.info(f"Eval Epoch: {epoch} " + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()]))

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_value("scalar", name, Tensor(val))
            tb_writer.record(epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb and is_master(args):
        assert wandb is not None, "Please install wandb."
        if "train" in data:
            dataloader = data["train"].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data["epoch"] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = np.expand_dims(np.arange(0, len(text_features)), -1)

    for name, logit in logits.items():
        ranking = np.argsort(logit)[:, ::-1]
        preds = np.nonzero(ranking == ground_truth)[1]
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return ops.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
