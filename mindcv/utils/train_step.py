"""Customized TrainOneStepCell.

Supported algorithms are list as follows:
    * Exponential Moving Average (EMA)
    * Gradient Clipping
    * Gradient Accumulation
"""

import mindspore as ms
from mindspore import Parameter, ParameterTuple, RowTensor, Tensor, nn, ops
from mindspore.boost.grad_accumulation import gradient_accumulation_op as _grad_accum_op
from mindspore.boost.grad_accumulation import gradient_clear_op as _grad_clear_op
from mindspore.ops import functional as F

__all__ = [
    "TrainStep",
]

_ema_op = ops.MultitypeFuncGraph("ema_op")
_grad_scale_op = ops.MultitypeFuncGraph("grad_scale_op")
reciprocal = ops.Reciprocal()


@_ema_op.register("Tensor", "Tensor", "Tensor")
def ema_op(factor, ema_weight, weight):
    return F.assign(ema_weight, ema_weight * factor + weight * (1 - factor))


@_grad_scale_op.register("Tensor", "Tensor")
def grad_scale_op_tensor(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale_op.register("Tensor", "RowTensor")
def grad_scale_op_row_tensor(scale, grad):
    return RowTensor(
        grad.indices,
        grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
        grad.dense_shape,
    )


class TrainStep(nn.TrainOneStepWithLossScaleCell):
    """Training step with loss scale.

    The customized trainOneStepCell also supported following algorithms:
        * Exponential Moving Average (EMA)
        * Gradient Clipping
        * Gradient Accumulation

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
        ema=False,
        ema_decay=0.9999,
        clip_grad=False,
        clip_value=15.0,
        gradient_accumulation_steps=1,
    ):
        super(TrainStep, self).__init__(network, optimizer, scale_sense)
        self.clip_grad = clip_grad
        self.clip_value = clip_value
        self.ema = ema
        self.ema_decay = ema_decay
        self.ema_step = Parameter(Tensor(0.0, ms.float32), name="ema_step")
        if self.ema:
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

    def optim_fn(self, loss, grads):
        if self.accum_grad:
            self.accum_step += 1
            loss = F.depend(
                loss, self.hyper_map(F.partial(_grad_accum_op, self.accum_steps), self.accumulated_grads, grads)
            )
            if self.accum_step % self.accum_steps == 0:
                if self.clip_grad:
                    loss = F.depend(
                        loss, self.optimizer(ops.clip_by_global_norm(self.accumulated_grads, clip_norm=self.clip_value))
                    )
                else:
                    loss = F.depend(loss, self.optimizer(self.accumulated_grads))
                loss = F.depend(loss, self.hyper_map(F.partial(_grad_clear_op), self.accumulated_grads))
            else:
                # update the learning rate, do not update the parameter
                loss = F.depend(loss, self.optimizer.get_lr())
        else:
            if self.clip_grad:
                loss = F.depend(loss, self.optimizer(ops.clip_by_global_norm(grads, clip_norm=self.clip_value)))
            else:
                loss = F.depend(loss, self.optimizer(grads))

        if self.ema:
            self.ema_step += 1
            # ema factor is corrected by (1 - exp(-t/T)), where `t` means time and `T` means temperature.
            ema_decay = self.ema_decay * (1 - F.exp(-self.ema_step / 2000))
            # update trainable parameters
            loss = F.depend(loss, self.hyper_map(F.partial(_ema_op, ema_decay), self.ema_weights, self.parameters))
        return loss

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = ops.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(_grad_scale_op, scaling_sens), grads)
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
                loss = self.optim_fn(loss, grads)
        else:  # scale_sense = loss_scale: Tensor --> TrainOneStepCell.construct
            loss = self.optim_fn(loss, grads)

        return loss
