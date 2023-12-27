import collections.abc
import logging
import math
from itertools import repeat

from mindspore import Tensor, nn
from mindspore.common.initializer import (
    Constant,
    HeNormal,
    HeUniform,
    Normal,
    One,
    Uniform,
    XavierNormal,
    XavierUniform,
    Zero,
    _calculate_fan_in_and_fan_out,
    initializer,
)

_logger = logging.getLogger(__name__)


def freeze_batch_norm_2d(module, module_match={}, name=""):
    raise NotImplementedError


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0):
    scale = (b - a) / 2
    bias = (b + a) / 2
    tensor.set_data(initializer(Uniform(scale), tensor.shape, tensor.dtype) + bias)


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0):
    tensor.set_data(initializer(Normal(std, mean), tensor.shape, tensor.dtype))


def constant_(tensor: Tensor, val: float):
    tensor.set_data(initializer(Constant(val), tensor.shape, tensor.dtype))


def ones_(tensor: Tensor):
    tensor.set_data(initializer(One(), tensor.shape, tensor.dtype))


def zeros_(tensor: Tensor):
    tensor.set_data(initializer(Zero(), tensor.shape, tensor.dtype))


def xavier_uniform_(tensor: Tensor, gain: float = 1.0):
    tensor.set_data(initializer(XavierUniform(gain), tensor.shape, tensor.dtype))


def xavier_normal_(tensor: Tensor, gain: float = 1.0):
    tensor.set_data(initializer(XavierNormal(gain), tensor.shape, tensor.dtype))


def kaiming_uniform_(tensor: Tensor, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"):
    tensor.set_data(initializer(HeUniform(a, mode, nonlinearity), tensor.shape, tensor.dtype))


def kaiming_normal_(tensor: Tensor, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"):
    tensor.set_data(initializer(HeNormal(a, mode, nonlinearity), tensor.shape, tensor.dtype))


def reset_parameters_torch(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
        # we always get moving_mean/moving_variance, even use_batch_statistics is False
        zeros_(m.moving_mean)
        ones_(m.moving_variance)
        # we always get gamma/beta, even affine is False
        ones_(m.gamma)
        zeros_(m.beta)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
        ones_(m.gamma)
        zeros_(m.beta)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Conv1dTranspose, nn.Conv2dTranspose, nn.Conv3dTranspose)):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(m.weight.shape)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                uniform_(m.bias, -bound, bound)
    elif isinstance(m, nn.Dense):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(m.weight.shape)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            uniform_(m.bias, -bound, bound)
    else:
        # if m.parameters_dict():
        #     _logger.warning(f"Layer {m.__class__.__name__} with type {type(m).__name__} is not initialized")
        pass
