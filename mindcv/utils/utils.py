""" utils """
from mindspore import nn, ops
from mindspore.ops import ReduceOp


def check_batch_size(num_samples, ori_batch_size=32, refine=True):
    if num_samples % ori_batch_size == 0:
        return ori_batch_size
    else:
        # search a batch size that is divisible by num samples.
        for bs in range(ori_batch_size - 1, 0, -1):
            if num_samples % bs == 0:
                print(
                    f"WARNING: num eval samples {num_samples} can not be divided by "
                    f"the input batch size {ori_batch_size}. The batch size is refined to {bs}"
                )
                return bs


class AllReduceSum(nn.Cell):
    """Reduces the tensor data across all devices in such a way that all devices will get the same final result."""

    def __init__(self):
        super().__init__()
        self.all_reduce_sum = ops.AllReduce(ReduceOp.SUM)

    def construct(self, x):
        return self.all_reduce_sum(x)
