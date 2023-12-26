import glob
import logging
import os
import random
import re
import sys
from datetime import datetime
from time import time

import numpy as np

import mindspore as ms
from mindspore import nn
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager

try:
    import wandb
except ImportError:
    wandb = None

try:
    from mindspore.train.summary import SummaryRecord
except ImportError:
    SummaryRecord = None

from open_clip import create_loss, create_model_and_transforms, get_tokenizer
from training.data import get_data
from training.distributed import broadcast_object, init_distributed_device, is_master
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import const_lr, const_lr_cooldown, cosine_lr
from training.train import LATEST_CHECKPOINT_NAME, CallbackForCLIP, build_trainer, evaluate

_logger = logging.getLogger(__name__)


def random_seed(seed=42, rank=0):
    ms.set_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See https://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def get_latest_checkpoint(path: str, remote: bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    checkpoints = glob.glob(path + "**/*.ckpt", recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):
    args = parse_args(args)
    assert args.accum_freq == 1, (
        "CLIP use the cached features from the other batches as negatives when --accum-freq is larger than 1, "
        "which is not implemented yet."
    )

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace("/", "-")
        timestamp = time()
        if args.distributed:
            # sync timestamp from master to all ranks
            timestamp = broadcast_object(args, timestamp)
        date_str = datetime.fromtimestamp(timestamp).strftime("%Y_%m_%d-%H_%M_%S")
        args.name = "-".join(
            [
                date_str,
                f"model_{model_name_safe}",
                f"lr_{args.lr}",
                f"b_{args.batch_size}",
                f"j_{args.workers}",
                f"p_{args.precision}",
            ]
        )

    resume_latest = args.resume == "latest"
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f"out-{args.rank}" if args.log_local else "out.log"
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print("Error. Experiment already exists. Use --name {} to specify a new experiment.")
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = "wandb" in args.report_to or "all" in args.report_to
    args.tensorboard = "tensorboard" in args.report_to or "all" in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ""
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ""

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                _logger.info(f"Found latest resume checkpoint at {resume_from}.")
            else:
                _logger.info(f"No latest resume checkpoint found in {checkpoint_path}.")
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    if args.distributed:
        _logger.info(
            f"Running in distributed mode with multiple processes. Device: {args.device}."
            f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
        )
    else:
        _logger.info(f"Running with a single process. Device {args.device}.")

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        # FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        # FIXME: support distillation with coca.
        assert "coca" not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    model_kwargs = {}
    if args.siglip:
        model_kwargs["init_logit_scale"] = np.log(10)  # different from CLIP
        model_kwargs["init_logit_bias"] = -10
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        **model_kwargs,
    )
    if args.distill:
        raise NotImplementedError("Distillation is not supported yet.")

    random_seed(args.seed, args.rank)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups, freeze_bn_stats=args.lock_image_freeze_bn_stats
        )
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers, freeze_layer_norm=args.lock_text_freeze_layer_norm
        )

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        _logger.info("Model:")
        _logger.info(f"{str(model)}")
        _logger.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                _logger.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    # optionally resume epoch from a checkpoint
    start_epoch = 0
    checkpoint = None
    if args.resume is not None:
        _logger.warning("Resuming from checkpoints does not work well for now!")
        checkpoint = ms.load_checkpoint(args.resume)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]

    # initialize datasets
    tokenizer = get_tokenizer(args.model)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    assert len(data), "At least one train or eval dataset must be specified."

    # create scheduler if train
    scheduler = None
    if "train" in data:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert (
                args.epochs_cooldown is not None
            ), "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                args.lr, args.warmup, total_steps, cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end
            )
        else:
            _logger.error(
                f"Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown."
            )
            exit(1)

    # create optimizer and scaler
    optimizer = None
    scaler = None
    if args.train_data or args.dataset_type == "synthetic":
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n  # noqa: E731
        include = lambda n, p: not exclude(n, p)  # noqa: E731

        named_parameters = list(model.parameters_and_names())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = nn.AdamWeightDecay(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            learning_rate=scheduler,
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.eps,
        )

        loss_scale_type, loss_scale_value = "static", 65536.0
        if loss_scale_type.lower() == "static":
            scaler = FixedLossScaleManager(loss_scale=loss_scale_value, drop_overflow_update=False)
        elif loss_scale_type.lower() == "dynamic":
            scaler = DynamicLossScaleManager(init_loss_scale=loss_scale_value, scale_factor=2, scale_window=2000)
        else:
            raise ValueError(f"Loss scale type only support ['static', 'dynamic'], but got{loss_scale_type}.")

    # optionally resume from loaded checkpoint
    if args.resume is not None:
        if "epoch" in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith("module"):
                sd = {k[len("module.") :]: v for k, v in sd.items()}
            ms.load_param_into_net(model, sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            _logger.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            ms.load_param_into_net(model, checkpoint)
            _logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != "none" and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert SummaryRecord is not None, "Please install mindinsight as tensorboard."
        writer = SummaryRecord(args.tensorboard_path)
    if args.wandb and is_master(args):
        assert wandb is None, (
            "WandB uploads data to public networks, which can lead to information security issues. "
            "If you are an external user, you can safely delete this line"
        )
        assert wandb is not None, "Please install wandb."
        _logger.debug("Starting wandb.")
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume="auto" if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log="all")
        wandb.save(params_file)
        _logger.debug("Finished loading wandb.")

    if "train" not in data:
        # Evaluate.
        evaluate(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer)
        return

    loss = create_loss(args)
    trainer = build_trainer(
        model, loss, optimizer, amp_level=args.amp_opt_level, scaler=scaler, grad_clip_norm=args.grad_clip_norm
    )
    callbacks = [CallbackForCLIP]
    trainer.train(args.epochs - start_epoch, data["train"].dataloader, callbacks=callbacks, dataset_sink_mode=True)

    if args.wandb and is_master(args):
        wandb.finish()
    if args.save_logs and args.tensorboard:
        writer.close()


def copy_codebase(args):
    from shutil import copytree, ignore_patterns

    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment.")
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns("log", "logs", "wandb"))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
