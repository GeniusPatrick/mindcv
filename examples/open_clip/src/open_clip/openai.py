""" OpenAI pretrained model functions

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import os
from typing import List, Optional

import mindspore as ms

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .model import build_model_from_openai_state_dict, convert_weights_to_lp, get_cast_dtype
from .pretrained import download_pretrained_from_url, get_pretrained_url, list_pretrained_models_by_tag

__all__ = ["list_openai_models", "load_openai_model"]


def list_openai_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list_pretrained_models_by_tag("openai")


def load_openai_model(
    name: str,
    precision: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    precision: str
        Model precision, if None defaults to 'fp32'.
    cache_dir : Optional[str]
        The directory to cache the downloaded model weights

    Returns
    -------
    model : mindspore.nn.Cell
        The CLIP model
    preprocess : Callable[[PIL.Image], mindspore.Tensor]
        Transform operations that converts a PIL image into a tensor that the returned model can take as its input
    """
    if precision is None:
        precision = "fp32"

    if get_pretrained_url(name, "openai"):
        model_path = download_pretrained_from_url(get_pretrained_url(name, "openai"), cache_dir=cache_dir)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {list_openai_models()}")

    state_dict = ms.load_checkpoint(model_path)

    cast_dtype = get_cast_dtype(precision)
    model = build_model_from_openai_state_dict(state_dict, cast_dtype=cast_dtype)
    # FIXME support pure fp16/bf16 precision modes
    if precision != "fp16":
        for _, param in model.parameters_and_names():
            param.set_dtype(ms.float32)
        if precision == "bf16":
            # for bf16, convert back to low-precision
            convert_weights_to_lp(model, dtype=ms.bfloat16)

    # add mean / std attributes for consistency with OpenCLIP models
    model.visual.image_mean = OPENAI_DATASET_MEAN
    model.visual.image_std = OPENAI_DATASET_STD
    return model
