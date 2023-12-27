import logging

from open_clip import (
    IMAGENET_CLASSNAMES,
    OPENAI_IMAGENET_TEMPLATES,
    build_zero_shot_classifier,
    get_tokenizer,
)

_logger = logging.getLogger(__name__)
_USE_TQDM = False


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.equal(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdims=True).numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    top1, top5, n = 0.0, 0.0, 0.0
    if _USE_TQDM:
        from tqdm import tqdm

        tuple_iterator = tqdm(dataloader.create_tuple_iterator(), unit_scale=args.batch_size, total=len(dataloader))
    else:
        tuple_iterator = dataloader.create_tuple_iterator()
    for images, target in tuple_iterator:
        # predict
        image_features = model.encode_image(images, normalize=True)
        logits = 100.0 * image_features @ classifier

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.shape[0]

    top1 = top1 / n
    top5 = top5 / n
    return top1, top5


def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    if "imagenet-val" not in data and "imagenet-v2" not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    _logger.info("Starting zero-shot imagenet.")
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    _logger.info("Building zero-shot classifier")
    classifier = build_zero_shot_classifier(
        model,
        tokenizer=tokenizer,
        classnames=IMAGENET_CLASSNAMES,
        templates=OPENAI_IMAGENET_TEMPLATES,
        num_classes_per_batch=10,
        use_tqdm=_USE_TQDM,
    )

    _logger.info("Using classifier")
    results = {}
    if "imagenet-val" in data:
        top1, top5 = run(model, classifier, data["imagenet-val"].dataloader, args)
        results["imagenet-zeroshot-val-top1"] = top1
        results["imagenet-zeroshot-val-top5"] = top5
    if "imagenet-v2" in data:
        top1, top5 = run(model, classifier, data["imagenet-v2"].dataloader, args)
        results["imagenetv2-zeroshot-val-top1"] = top1
        results["imagenetv2-zeroshot-val-top5"] = top5

    _logger.info("Finished zero-shot imagenet.")

    return results
