import json
import logging
import os
import re
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import mindspore as ms
from mindspore import ops

from .loss import ClipLoss, DistillClipLoss
from .model import (
    CLIP,
    CustomTextCLIP,
    convert_to_custom_text_state_dict,
    convert_weights_to_lp,
    get_cast_dtype,
    resize_pos_embed,
    resize_text_pos_embed,
    set_model_preprocess_cfg,
)
from .openai import load_openai_model
from .pretrained import download_pretrained, get_pretrained_cfg, list_pretrained_tags_by_model
from .tokenizer import DEFAULT_CONTEXT_LENGTH, SimpleTokenizer
from .transform import (
    AugmentationCfg,
    PreprocessCfg,
    image_transform_v2,
    merge_preprocess_dict,
    merge_preprocess_kwargs,
)

_MODEL_CONFIG_PATHS = [Path(__file__).parent / "model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = (".json",)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        with open(cf, "r") as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ("embed_dim", "vision_cfg", "text_cfg")):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def list_models():
    """enumerate available model architectures based on config files"""
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """add model config path or file and update registry"""
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def get_tokenizer(model_name: str = "", context_length: Optional[int] = None, **kwargs):
    config = get_model_config(model_name)
    assert config is not None, f"No valid model config found for {model_name}."
    text_config = config.get("text_cfg", {})
    if "tokenizer_kwargs" in text_config:
        tokenizer_kwargs = dict(text_config["tokenizer_kwargs"], **kwargs)
    else:
        tokenizer_kwargs = kwargs
    if context_length is None:
        context_length = text_config.get("context_length", DEFAULT_CONTEXT_LENGTH)

    tokenizer = SimpleTokenizer(
        context_length=context_length,
        **tokenizer_kwargs,
    )

    return tokenizer


def load_state_dict(checkpoint_path: str):
    checkpoint = ms.load_checkpoint(checkpoint_path)
    state_dict = checkpoint
    # TODO: We may should do unwrap stuffs like cleaning the augly "_backbone" prefix when saving checkpoints.
    if "optimizer.global_step" in state_dict:  # need to unwrap trainer-format checkpoint
        optimizer_state_dict = {}
        model_state_dict = {}
        mics_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("optimizer."):
                k = k[len("optimizer.") :]
                if k.startswith("adam") or k == "global_step" or "learning_rate" in k:
                    optimizer_state_dict[k] = v
                else:
                    model_state_dict[k] = v
            elif k.startswith("network."):
                k = k[len("network.") :]
                model_state_dict[k.replace("_backbone.", "")] = v
            else:
                mics_state_dict[k] = v
        state_dict = model_state_dict
    return state_dict


def load_checkpoint(model, checkpoint_path, strict=False):
    state_dict = load_state_dict(checkpoint_path)
    # detect old format and make compatible with new format
    if "positional_embedding" in state_dict and not hasattr(model, "positional_embedding"):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    # If loading a non-SigLIP model for SigLIP training. See https://github.com/mlfoundations/open_clip/issues/712
    if "logit_bias" not in state_dict and model.logit_bias is not None:
        state_dict["logit_bias"] = ops.zeros_like(state_dict["logit_scale"])
    # Certain text transformers no longer expect position_ids after transformers==4.31
    position_id_key = "text.transformer.embeddings.position_ids"
    if position_id_key in state_dict and not hasattr(model, position_id_key):
        del state_dict[position_id_key]
    resize_pos_embed(state_dict, model)
    resize_text_pos_embed(state_dict, model)
    incompatible_keys = ms.load_param_into_net(model, state_dict, strict_load=strict)
    return incompatible_keys


def create_model(
    model_name: str,
    pretrained: Optional[str] = None,
    precision: str = "fp32",
    force_quick_gelu: bool = False,
    force_custom_text: bool = False,
    force_patch_dropout: Optional[float] = None,
    force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
    force_preprocess_cfg: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    require_pretrained: bool = False,
    **model_kwargs,
):
    """
    device, jit & output_dict will never be supported.
    pretrained_image & pretrained_hf are not supported for now.
    """
    force_preprocess_cfg = force_preprocess_cfg or {}
    preprocess_cfg = asdict(PreprocessCfg())
    model_name = model_name.replace("/", "-")  # for callers using old naming with / in ViT names
    model_cfg = None

    if pretrained and pretrained.lower() == "openai":
        logging.info(f"Loading pretrained {model_name} from OpenAI.")
        model = load_openai_model(
            model_name,
            precision=precision,
            cache_dir=cache_dir,
        )
    else:
        model_cfg = model_cfg or get_model_config(model_name)
        if model_cfg is not None:
            logging.info(f"Loaded {model_name} model config.")
        else:
            logging.error(f"Model config for {model_name} not found; available models {list_models()}.")
            raise RuntimeError(f"Model config for {model_name} not found.")

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if force_patch_dropout is not None:
            # override the default patch dropout value
            model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

        if force_image_size is not None:
            # override model config's image size
            model_cfg["vision_cfg"]["image_size"] = force_image_size

        # cast_dtype set for fp16 and bf16 (manual mixed-precision), not set for 'amp' or 'pure' modes
        cast_dtype = get_cast_dtype(precision)
        custom_text = model_cfg.pop("custom_text", False) or force_custom_text
        model_cfg = dict(model_cfg, **model_kwargs)  # merge cfg dict w/ kwargs (kwargs overrides cfg)
        if custom_text:
            if "multimodal_cfg" in model_name:
                raise ImportError("COCA model have not been supported yet.")
            else:
                model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
        else:
            model = CLIP(**model_cfg, cast_dtype=cast_dtype)

        if precision in ("fp16", "bf16"):
            dtype = ms.float16 if "fp16" in precision else ms.bfloat16
            # manual mixed precision that matches original OpenAI behaviour
            convert_weights_to_lp(model, dtype=dtype)
        elif precision in ("pure_fp16", "pure_bf16"):
            dtype = ms.float16 if "fp16" in precision else ms.bfloat16
            for _, param in model.parameters_and_names():
                param.set_dtype(dtype)

        pretrained_loaded = False
        if pretrained:
            checkpoint_path = ""
            pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
            if pretrained_cfg:
                checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=cache_dir)
                preprocess_cfg = merge_preprocess_dict(preprocess_cfg, pretrained_cfg)
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained

            if checkpoint_path:
                logging.info(f"Loading pretrained {model_name} weights ({pretrained}).")
                load_checkpoint(model, checkpoint_path)
            else:
                error_str = (
                    f"Pretrained weights ({pretrained}) not found for model {model_name}."
                    f"Available pretrained tags ({list_pretrained_tags_by_model(model_name)}."
                )
                logging.warning(error_str)
                raise RuntimeError(error_str)
            pretrained_loaded = True

        if require_pretrained and not pretrained_loaded:
            # callers of create_model_from_pretrained always expect pretrained weights
            raise RuntimeError(
                f"Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded."
            )

    # set image preprocessing configuration in model attributes for convenience
    if getattr(model.visual, "image_size", None) is not None:
        # use image_size set on model creation (via config or force_image_size arg)
        force_preprocess_cfg["size"] = model.visual.image_size
    set_model_preprocess_cfg(model, merge_preprocess_dict(preprocess_cfg, force_preprocess_cfg))

    return model


def create_loss(args):
    if args.distill:
        return DistillClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=False,
            rank=args.rank,
            world_size=args.world_size,
        )
    return ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=False,
        rank=args.rank,
        world_size=args.world_size,
    )


def create_model_and_transforms(
    model_name: str,
    pretrained: Optional[str] = None,
    precision: str = "fp32",
    force_quick_gelu: bool = False,
    force_custom_text: bool = False,
    force_patch_dropout: Optional[float] = None,
    force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
    image_mean: Optional[Tuple[float, ...]] = None,
    image_std: Optional[Tuple[float, ...]] = None,
    image_interpolation: Optional[str] = None,
    image_resize_mode: Optional[str] = None,  # only effective for inference
    aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
    cache_dir: Optional[str] = None,
    **model_kwargs,
):
    """
    device, jit & output_dict will never be supported.
    pretrained_image & pretrained_hf are not supported for now.
    """
    force_preprocess_cfg = merge_preprocess_kwargs(
        {}, mean=image_mean, std=image_std, interpolation=image_interpolation, resize_mode=image_resize_mode
    )
    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_patch_dropout=force_patch_dropout,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        cache_dir=cache_dir,
        **model_kwargs,
    )

    pp_cfg = PreprocessCfg(**model.visual.preprocess_cfg)

    preprocess_train = image_transform_v2(
        pp_cfg,
        is_train=True,
        aug_cfg=aug_cfg,
    )
    preprocess_val = image_transform_v2(
        pp_cfg,
        is_train=False,
    )

    return model, preprocess_train, preprocess_val


def create_model_from_pretrained(
    model_name: str,
    pretrained: Optional[str] = None,
    precision: str = "fp32",
    force_quick_gelu: bool = False,
    force_custom_text: bool = False,
    force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
    image_mean: Optional[Tuple[float, ...]] = None,
    image_std: Optional[Tuple[float, ...]] = None,
    image_interpolation: Optional[str] = None,
    image_resize_mode: Optional[str] = None,  # only effective for inference
    return_transform: bool = True,
    cache_dir: Optional[str] = None,
    **model_kwargs,
):
    force_preprocess_cfg = merge_preprocess_kwargs(
        {}, mean=image_mean, std=image_std, interpolation=image_interpolation, resize_mode=image_resize_mode
    )

    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        cache_dir=cache_dir,
        require_pretrained=True,
        **model_kwargs,
    )

    if not return_transform:
        return model

    preprocess = image_transform_v2(
        PreprocessCfg(**model.visual.preprocess_cfg),
        is_train=False,
    )

    return model, preprocess
