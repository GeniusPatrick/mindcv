import datetime
import json
import logging
import math
import os
import time

import numpy as np

import mindspore as ms
from mindspore import Parameter, ParameterTuple, RowTensor, Tensor, context, nn, ops
from mindspore.boost.grad_accumulation import gradient_accumulation_op as _grad_accum_op
from mindspore.boost.grad_accumulation import gradient_clear_op as _grad_clear_op
from mindspore.train import Callback, Model, save_checkpoint
from mindspore.train.amp import _OutputTo16, _OutputTo32, custom_mixed_precision, get_black_list, get_white_list
from mindspore.train.loss_scale_manager import LossScaleManager

try:
    import wandb
except ImportError:
    wandb = None

from .distributed import is_master
from .zero_shot import zero_shot_eval

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"
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


_ema_op = ops.MultitypeFuncGraph("ema_op")
_grad_scale_op = ops.MultitypeFuncGraph("grad_scale_op")
reciprocal = ops.Reciprocal()


@_ema_op.register("Tensor", "Tensor", "Tensor")
def ema_op(factor, ema_weight, weight):
    return ops.assign(ema_weight, ema_weight * factor + weight * (1 - factor))


@_grad_scale_op.register("Tensor", "Tensor")
def grad_scale_op_tensor(scale, grad):
    return grad * ops.cast(reciprocal(scale), ops.dtype(grad))


@_grad_scale_op.register("Tensor", "RowTensor")
def grad_scale_op_row_tensor(scale, grad):
    return RowTensor(
        grad.indices,
        grad.values * ops.cast(reciprocal(scale), ops.dtype(grad.values)),
        grad.dense_shape,
    )


class TrainStep(nn.TrainOneStepWithLossScaleCell):
    """Training step with loss scale.

    The steps of model optimization are performed in the following order:
        1. calculate grad
        2. allreduce grad
        3. accumulate grad [optional]
        4. clip grad [optional]
        5. call optimizer
        6. ema weights [optional]
    """

    def __init__(
        self,
        network,
        optimizer,
        scale_sense=1.0,
        model_ema_decay: float = None,
        grad_clip_norm: float = None,
        gradient_accumulation_steps=1,
    ):
        super(TrainStep, self).__init__(network, optimizer, scale_sense)
        self.clip_grad = grad_clip_norm is not None
        self.clip_value = grad_clip_norm
        self.ema = model_ema_decay is not None
        self.ema_decay = model_ema_decay
        self.ema_step = Parameter(Tensor(0.0, ms.float32), name="ema_step")
        if model_ema_decay:
            self.parameters = ParameterTuple(network.get_parameters())  # same as model.state_dict in torch
            self.ema_weights = self.parameters.clone(prefix="ema", init="same")
        else:
            self.ema_weights = ParameterTuple(())  # placeholder
        self.accum_grad = gradient_accumulation_steps > 1
        self.accum_steps = gradient_accumulation_steps
        self.accum_step = Parameter(Tensor(0, dtype=ms.int32), name="accum_step")
        if self.accum_grad:
            self.accumulated_grads = self.weights.clone(prefix="accum_grad", init="zeros")
        else:
            self.accumulated_grads = ParameterTuple(())  # placeholder

    def update(self, loss, grads):
        if self.accum_grad:
            self.accum_step += 1
            loss = ops.depend(
                loss, self.hyper_map(ops.partial(_grad_accum_op, self.accum_steps), self.accumulated_grads, grads)
            )
            if self.accum_step % self.accum_steps == 0:
                if self.clip_grad:
                    loss = ops.depend(
                        loss, self.optimizer(ops.clip_by_global_norm(self.accumulated_grads, clip_norm=self.clip_value))
                    )
                else:
                    loss = ops.depend(loss, self.optimizer(self.accumulated_grads))
                loss = ops.depend(loss, self.hyper_map(ops.partial(_grad_clear_op), self.accumulated_grads))
            else:
                # update the learning rate, do not update the parameter
                # todo: get_lr does not update lr anymore since ms2.1
                loss = ops.depend(loss, self.optimizer.get_lr())
        else:
            if self.clip_grad:
                loss = ops.depend(loss, self.optimizer(ops.clip_by_global_norm(grads, clip_norm=self.clip_value)))
            else:
                loss = ops.depend(loss, self.optimizer(grads))

        if self.ema:
            self.ema_step += 1
            # ema factor is corrected by (1 - exp(-t/T)), where `t` means time and `T` means temperature.
            ema_decay = self.ema_decay * (1 - ops.exp(-self.ema_step / 2000))
            # update trainable parameters
            loss = ops.depend(loss, self.hyper_map(ops.partial(_ema_op, ema_decay), self.ema_weights, self.parameters))
        return loss

    def construct(self, *inputs):
        weights = self.weights
        loss, logit_scale = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = ops.ones_like(loss) * ops.cast(scaling_sens, ops.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(ops.partial(_grad_scale_op, scaling_sens), grads)
        # apply grad reducer on grads
        # When gradients in one bucket are all ready, the Reducer kicks off an asynchronous allreduce on that bucket
        # to calculate mean of gradients across all processes. When reduction is done,
        # averaged gradients are written to the param.grad field of all parameters.
        # refer to https://pytorch.org/docs/stable/notes/ddp.html
        grads = self.grad_reducer(grads)

        if self.loss_scaling_manager:  # scale_sense = update_cell: Cell --> TrainOneStepWithLossScaleCell.construct
            # get the overflow buffer
            cond = self.get_overflow_status(status, grads)
            overflow = self.process_loss_scale(cond)
            # if there is no overflow, do optimize
            if not overflow:
                loss = self.update(loss, grads)
        else:  # scale_sense = loss_scale: Tensor --> TrainOneStepCell.construct
            loss = self.update(loss, grads)

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        logit_scale_param = self.network.backbone_network.logit_scale
        loss = ops.depend(loss, ops.assign(logit_scale_param, ops.clamp(logit_scale_param, 0, 4.6052)))
        return loss, logit_scale


class WithLossCell(nn.WithLossCell):
    def construct(self, image, text):
        image_features, text_features, logit_scale = self._backbone(image, text)
        return self._loss_fn(image_features, text_features, logit_scale), logit_scale


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
    scaler: LossScaleManager = None,
    model_ema_decay: float = None,
    grad_clip_norm: float = None,
    gradient_accumulation_steps: int = 1,
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
        model_ema_decay: Decay factor for model weights exponential moving average. Disable if it's None.
        grad_clip_norm: The value at which to clip gradients. Disable if it's None.
        gradient_accumulation_steps: Accumulate the gradients of n batches before update.

    Returns:
        mindspore.Model

    """
    network = auto_mixed_precision(network, amp_level=amp_level)
    criterion = criterion.to_float(ms.float32)
    train_step_kwargs = dict(
        network=WithLossCell(network, criterion),
        optimizer=optimizer,
        model_ema_decay=model_ema_decay,
        grad_clip_norm=grad_clip_norm,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    update_cell = scaler.get_update_cell()
    # 1. loss_scale_type="static", drop_overflow_update=False
    # --> update_cell=None, TrainStep=TrainOneStepCell(scale_sense=loss_scale)
    # 2. loss_scale_type: static, drop_overflow_update: True
    # --> update_cell=FixedLossScaleUpdateCell, TrainStep=TrainOneStepWithLossScaleCell(scale_sense=update_cell)
    # 3. loss_scale_type: dynamic, drop_overflow_update: True
    # --> update_cell=DynamicLossScaleUpdateCell, TrainStep=TrainOneStepWithLossScaleCell(scale_sense=update_cell)
    if update_cell is None:
        train_step_kwargs["scale_sense"] = Tensor(scaler.get_loss_scale(), dtype=ms.float32)
    else:
        if not context.get_context("enable_ge") and context.get_context("device_target") == "CPU":
            raise ValueError(
                "Only `loss_scale_type` is `static` and `drop_overflow_update` is `False`"
                "are supported on device `CPU`."
            )
        train_step_kwargs["scale_sense"] = update_cell
    train_step_cell = TrainStep(**train_step_kwargs).set_train()
    model = Model(train_step_cell)
    return model


class CallbackForCLIP(Callback):
    def __init__(self, args, data, tokenizer, model, trainer, writer=None, start_epoch=0):
        self.args = args
        self.data = data
        self.tokenizer = tokenizer
        self.model = model
        self.trainer = trainer
        self.writer = writer
        self.start_epoch = start_epoch
        # initialize the following members to make linter happy
        self.step_ts = -1.
        self.epoch_ts = -1.
        self.num_batches_per_epoch = -1
        self.num_samples_per_epoch = -1
        self.sample_digits = -1
        self.losses_m = {}
        self.batch_time_m = AverageMeter()

    def _get_network_from_cbp(self, cb_params):
        network = cb_params.train_network if cb_params.mode == "train" else cb_params.eval_network
        if cb_params.dataset_sink_mode:  # train_network is connected to DatasetHelper when data_sink is enable.
            return network.network
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
            evaluate(self.model, self.data, completed_epoch, self.args, tb_writer=self.writer, tokenizer=self.tokenizer)
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
                save_checkpoint(  # TrainStep with backbone, loss, optimizer, scaler, ema, accum_grad
                    self._get_network_from_cbp(cb_params),
                    os.path.join(self.args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                    append_dict=checkpoint_dict,
                )
            if self.args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(self.args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)
            if self.args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(self.args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(self.args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                save_checkpoint(self._get_network_from_cbp(cb_params), tmp_save_path, append_dict=checkpoint_dict)
                os.replace(tmp_save_path, latest_save_path)

        total_time = time.time() - self.epoch_ts
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
                self.losses_m[key].update(val.item(), self.args.batch_size)

            logit_scale_scalar = outputs[1].item()
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
                f"Train Epoch: {epoch} [{num_samples:>{self.sample_digits}}/{self.num_samples_per_epoch} ({percent_complete:.0f}%)] "  # noqa: E501
                f"Batch (t): {self.batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "  # noqa: E501
                f"LR: {self._get_lr_from_cbp(cb_params):5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "batch_time": self.batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": self._get_lr_from_cbp(cb_params),
            }
            log_data.update({name: val.val for name, val in self.losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if self.writer is not None:
                for name, val in log_data.items():
                    self.writer.add_value("scalar", name, val)
                self.writer.record(step)

            if self.args.wandb:
                assert wandb is not None, "Please install wandb."
                log_data["step"] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            self.batch_time_m.reset()


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
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
        for i, batch in enumerate(dataloader):
            images, texts = batch

            image_features, text_features, logit_scale = model(images, texts)
            # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
            # however, system RAM is easily exceeded and compute time becomes problematic
            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())
            logit_scale = logit_scale.mean()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            batch_size = images.shape[0]
            labels = ops.arange(0, batch_size, dtype=ms.int64)
            total_loss = (ops.cross_entropy(logits_per_image, labels) + ops.cross_entropy(logits_per_text, labels)) / 2

            gen_loss = maybe_compute_generative_loss({})  # only coca gives "logits" and "labels" in model_out

            cumulative_loss += total_loss * batch_size
            num_samples += batch_size
            if is_master(args) and (i % 100) == 0:
                _logger.info(
                    f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                    f"Clip Loss: {cumulative_loss / num_samples:.6f}\t"
                )

                if gen_loss is not None:
                    cumulative_gen_loss += gen_loss * batch_size
                    _logger.info(f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

        val_metrics = get_clip_metrics(
            image_features=ops.cat(all_image_features),
            text_features=ops.cat(all_text_features),
            logit_scale=logit_scale.cpu(),
        )
        loss = cumulative_loss / num_samples
        metrics.update({**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples})
        if gen_loss is not None:
            gen_loss = cumulative_gen_loss / num_samples
            metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    _logger.info(f"Eval Epoch: {epoch} " + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()]))

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_value("scalar", name, val)
            tb_writer.record(epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
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
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = ops.arange(0, len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = ops.argsort(logit, descending=True)
        preds = ops.nonzero(ranking == ground_truth)[:, 1]
        preds = preds.numpy()
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
