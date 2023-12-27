import mindspore as ms
from mindspore import Tensor, nn, ops


class AllGather(nn.Cell):
    def __init__(self):
        super().__init__()
        self.all_gather = ops.AllGather()

    def construct(self, x):
        return self.all_gather(x)


class AllGatherWithGrad(nn.Cell):
    # https://zhuanlan.zhihu.com/p/615784842
    # https://www.aiuai.cn/aifarm1762.html
    def __init__(self):
        super().__init__()
        self.all_gather = ops.AllGather()
        self.bprop_debug = True  # this flag makes customized bprop work

    def construct(self, x):
        return self.all_gather(x)

    def bprop(self):
        raise NotImplementedError


def gather_features(
    image_features,
    text_features,
    all_gather,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
):
    # We gather tensors from all ranks
    if gather_with_grad:
        all_image_features = all_gather(image_features)  # w/ grad
        all_text_features = all_gather(text_features)  # w/ grad
    else:
        d = image_features.shape[1]
        gathered_image_features = all_gather(image_features).reshape(world_size, -1, d)  # w/o grad
        gathered_text_features = all_gather(text_features).reshape(world_size, -1, d)  # w/o grad
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = ops.reshape(gathered_image_features, (-1, d))
        all_text_features = ops.reshape(gathered_text_features, (-1, d))

    return all_image_features, all_text_features


class ClipLoss(nn.Cell):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        if world_size > 1:
            if gather_with_grad:
                self.all_gather = AllGatherWithGrad()
            else:
                self.all_gather = AllGather()
        else:
            self.all_gather = None

        # cache state
        assert cache_labels is False, "Cache labels is not supported!"
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, num_logits) -> Tensor:
        # calculated ground-truth and cache if enabled
        labels = ops.arange(num_logits, dtype=ms.int64)
        if self.world_size > 1 and self.local_loss:
            labels = labels + num_logits * self.rank
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.all_gather,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def construct(self, image_features, text_features, logit_scale):
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(logits_per_image.shape[0])

        total_loss = (ops.cross_entropy(logits_per_image, labels) + ops.cross_entropy(logits_per_text, labels)) / 2

        return total_loss


class DistillClipLoss(ClipLoss):
    def dist_loss(self, teacher_logits, student_logits):
        return -(ops.softmax(teacher_logits, axis=1) * ops.log_softmax(student_logits, axis=1)).sum(axis=1).mean(axis=0)

    def construct(
        self, image_features, text_features, logit_scale, dist_image_features, dist_text_features, dist_logit_scale
    ):
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = self.get_logits(
            dist_image_features, dist_text_features, dist_logit_scale
        )

        labels = self.get_ground_truth(logits_per_image.shape[0])

        contrastive_loss = (
            ops.cross_entropy(logits_per_image, labels) + ops.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image)
            + self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        return contrastive_loss, distill_loss
