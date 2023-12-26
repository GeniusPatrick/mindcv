import numpy as np


def lr_builder(adjuster, steps):
    return [adjuster(step) for step in range(steps)]


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def const_lr(base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        return lr

    return lr_builder(_lr_adjuster, steps)


def const_lr_cooldown(base_lr, warmup_length, steps, cooldown_steps, cooldown_power=1.0, cooldown_end_lr=0.0):
    def _lr_adjuster(step):
        start_cooldown_step = steps - cooldown_steps
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            if step < start_cooldown_step:
                lr = base_lr
            else:
                e = step - start_cooldown_step
                es = steps - start_cooldown_step
                # linear decay if power == 1; polynomial decay otherwise;
                decay = (1 - (e / es)) ** cooldown_power
                lr = decay * (base_lr - cooldown_end_lr) + cooldown_end_lr
        return lr

    return lr_builder(_lr_adjuster, steps)


def cosine_lr(base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_builder(_lr_adjuster, steps)
