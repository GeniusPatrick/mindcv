import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.communication import get_group_size, get_local_rank, get_rank, init


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    if args.distributed:
        init()
        args.local_rank = get_local_rank()
        args.world_size = get_group_size()
        args.rank = get_rank()
        ms.context.set_auto_parallel_context(
            device_num=args.world_size,
            global_rank=args.rank,
            parallel_mode="data_parallel",
            gradients_mean=True,
        )

    device = f"{ms.get_context('device_target')}:{ms.get_context('device_id')}"
    args.device = device
    return device


class BroadCast(nn.Cell):
    def __init__(self, src):
        super().__init__()
        self.broadcast = ops.Broadcast(src)

    def construct(self, x):
        return self.broadcast(x)


def broadcast_object(args, obj, src=0):
    # broadcast a pickle-able python object from rank-0 to all ranks
    broadcast = BroadCast(src)
    objects = (Tensor(obj),)
    objects = broadcast(objects)
    return objects[0].numpy()


class AllGather(nn.Cell):
    def __init__(self):
        super().__init__()
        self.all_gather = ops.AllGather()

    def construct(self, x):
        return self.all_gather(x)


def all_gather_object(args, obj, dst=0):
    # gather a pickle-able python object across all ranks
    all_gather = AllGather()
    objects = all_gather(obj)
    return objects
