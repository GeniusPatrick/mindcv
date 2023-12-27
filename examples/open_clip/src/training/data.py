import ast
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from functools import partial
from itertools import islice
from multiprocessing import Value

import braceexpand
import numpy as np
import pandas as pd
import webdataset as wds
from PIL import Image
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, tar_file_expander, url_opener, valid_sample

import mindspore as ms
from mindspore.dataset import DistributedSampler, GeneratorDataset, ImageFolderDataset, SubsetRandomSampler, vision

_logger = logging.getLogger(__name__)


class CsvDataset:  # Dataset, RandomAccess
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        _logger.debug(f"Loading csv data from {input_filename}.")
        df = pd.read_csv(input_filename, sep=sep)

        self.image_root = os.path.join(os.path.dirname(input_filename), "images")
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        _logger.debug("Done loading data.")

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(os.path.join(self.image_root, str(self.images[idx]))))[0]
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: GeneratorDataset
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            # TODO: We can't re-shared samples for DistributedSampler in mindspore like torch does.
            #  Does Model.train make shuffling work properly? Probably not.
            #  Maybe we need to re-write Sampler for distributed training and explicitly invoke `set_epoch`.
            self.sampler.seed += epoch


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split("::")
        assert len(weights) == len(
            urllist
        ), f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, "sizes.json")
    len_filename = os.path.join(dir_path, "__len__")
    stats_filenames = [shard_name.replace(".tar", "_stats.json") for shard_name in shards_list]
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, "r"))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, "r").read())
    elif all([os.path.exists(stats_filename) for stats_filename in stats_filenames]):
        total_size = 0
        for stats_filename in stats_filenames:
            with open(stats_filename) as f:
                stats = json.load(f)
                total_size += stats["successes"]
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        raise NotImplementedError("ImagenetV2 is not implemented")
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        workers = args.workers if is_train else 1
        # TODO: Maybe we need reimplement ImageFolder in python and wrap it by GenerateDataset,
        #  which give exactly same image transform.
        dataset = ImageFolderDataset(data_path, num_parallel_workers=workers).map(
            operations=[vision.Decode(to_pil=True), preprocess_fn],
            input_columns="image",
        )

    if is_train:
        # for each class, randomly select 50 samples as subset
        dataset_targets = [v for k, v in dataset.get_class_indexing().items()]
        idxs = np.zeros(len(dataset_targets))
        target_array = np.array(dataset_targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype("int")
        sampler = SubsetRandomSampler(np.where(idxs)[0])
        dataset = dataset.use_sampler(sampler)
    else:
        sampler = None

    batch_size = args.batch_size  # if is_train else 16
    dataloader = dataset.batch(batch_size)

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader.create_tuple_iterator():
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = "txt" in sample
    has_image = "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    _logger.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def split_by_node(src, rank, world_size):  # maybe split_by_rank is a better name
    if world_size > 1:
        yield from islice(src, rank, None, world_size)
    else:
        yield from src


def split_by_worker(src, worker, num_workers):
    assert num_workers == 1, (
        "The number of workers must be 1. "
        "Reason: 1. We cannot get worker_id from MindData. 2. GeneratorDataset does not support sharding iterator."
    )
    if num_workers > 1:
        yield from islice(src, worker, None, num_workers)
    else:
        yield from src


def mindspore_worker_seed(args, increment=0):
    """get dataloader worker seed from pytorch"""
    # TODO: we actually cannot get seed of each dataloader worker(ms.dataset.get_seed()?), but only seed of each rank.
    #  the seed of each worker should be base_seed + rank * workers + worker_id.
    seed = (args.seed or 0) + args.rank
    # space out seed increments, so they can't overlap across workers in different iterations
    seed += increment * max(1, args.world_size * args.workers)
    return seed


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.detshuffle):
    def __init__(self, args, bufsize=1000, initial=100, seed=0, epoch=-1):
        super().__init__(bufsize, initial, seed, epoch)
        self.args = args

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = mindspore_worker_seed(self.args, epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(wds.SimpleShardList):  # IterableDataset
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        args,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__(urls)
        self.args = args
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(
                self.weights
            ), f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = mindspore_worker_seed(self.args, epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            _logger.info(f"Number of samples: {num_samples}, number of shards: {num_shards}")
            if not num_samples:
                raise RuntimeError(
                    "Currently, the number of dataset samples must be specified for the training dataset. "
                    "Please specify it via `--train-num-samples` if no dataset length info is present."
                )
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, (
            "--train_data_upsampling_factors is only supported "
            "when sampling with replacement (with --dataset-resampled)."
        )

    if is_train:
        if resampled:
            pipeline = [
                ResampledShards2(
                    args,
                    input_shards,
                    weights=args.train_data_upsampling_factors,
                    deterministic=True,
                    epoch=shared_epoch,
                )
            ]
        else:
            pipeline = [
                wds.SimpleShardList(input_shards),  # at this point we have an iterator over all the shards
                detshuffle2(
                    args,
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                partial(split_by_node, rank=args.rank, world_size=args.world_size),
                partial(split_by_worker, worker=0, num_workers=args.workers),
                # TODO: what if some worker is assigned the shards with much less samples than others?
            ]
        pipeline.extend(
            [
                # at this point, we have an iterator over the shards assigned to each worker at each node
                tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
                wds.shuffle(
                    bufsize=_SAMPLE_SHUFFLE_SIZE,
                    initial=_SAMPLE_SHUFFLE_INITIAL,
                ),
            ]
        )
    else:
        pipeline = [
            wds.SimpleShardList(input_shards),
            partial(split_by_worker, worker=0, num_workers=args.workers),
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ]
    pipeline.extend(
        [
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            wds.map_dict(image=lambda img: preprocess_img(img)[0], text=lambda text: tokenizer(text)[0]),
            wds.to_tuple("image", "text"),
            wds.batched(args.batch_size, partial=not is_train),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, "number of shards must be >= total workers"
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    workers = args.workers if is_train else 1
    dataloader = GeneratorDataset(dataset, column_names=["images", "texts"], num_parallel_workers=workers)

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(args.world_size, args.rank) if args.distributed and is_train else None
    shuffle = True if is_train and sampler is None else None

    workers = args.workers if is_train else 1
    batch_size = args.batch_size  # if is_train else 16
    dataloader = GeneratorDataset(
        dataset,
        column_names=["images", "texts"],
        shuffle=shuffle,
        num_parallel_workers=workers,
        sampler=sampler,
    ).batch(batch_size=batch_size, drop_remainder=is_train)
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset:
    def __init__(
        self,
        transform=None,
        image_size=(224, 224),
        caption="Dummy caption",
        dataset_size=100,
        tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new("RGB", image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        else:
            image = self.image
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(args.world_size, args.rank) if args.distributed and is_train else None
    shuffle = True if is_train and sampler is None else None

    workers = args.workers if is_train else 1
    batch_size = args.batch_size  # if is_train else 16
    dataloader = GeneratorDataset(
        dataset,
        column_names=["images", "texts"],
        shuffle=shuffle,
        num_parallel_workers=workers,
        sampler=sampler,
    ).batch(batch_size=batch_size, drop_remainder=is_train)
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split(".")[-1]
        if ext in ["csv", "tsv"]:
            return get_csv_dataset
        elif ext in ["tar"]:
            return get_wds_dataset
        else:
            raise ValueError(f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer
        )

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer
        )

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data


if __name__ == "__main__":

    class RandomDatasetMap:
        def __getitem__(self, index):
            data = np.random.randint(0, 1000, 3)
            data[0] = index
            return data

        def __len__(self):
            return 8

    def run_toy_map(rank):
        ms.dataset.set_seed(0)
        dataset = RandomDatasetMap()
        # sampler = DistributedSampler(num_shards=2, shard_id=0, shuffle=True)
        dataloader = GeneratorDataset(
            dataset,
            column_names=["data"],
            # sampler=sampler,
            shuffle=True,
            num_shards=2,
            shard_id=rank,
            num_parallel_workers=3,
        ).batch(batch_size=2)
        # dataloader = dataloader.create_tuple_iterator()
        for epoch in range(2):
            ms.dataset.set_seed(epoch + rank)
            # sampler.seed += epoch
            print(f"Epoch {epoch}")
            for batch in dataloader.create_tuple_iterator():
                print(batch[0].numpy())
            print("-" * 25)

    run_toy_map(0)
    run_toy_map(1)

    class RandomDatasetIter:
        def __iter__(self):
            for index in range(8):  # the length is 8
                data = np.random.randint(0, 1000, 3)
                data[0] = index
                yield data

    def run_toy_iter():
        ms.dataset.set_seed(0)
        dataset = RandomDatasetIter()  # does not support shard when using iterator
        dataloader = GeneratorDataset(
            dataset,
            column_names=["data"],
            shuffle=True,
            num_parallel_workers=3,
        ).batch(batch_size=2)
        for epoch in range(2):
            ms.dataset.set_seed(epoch)
            # sampler.seed += epoch
            print(f"Epoch {epoch}")
            for batch in dataloader.create_tuple_iterator():
                print(batch[0].numpy())
            print("-" * 25)

    # run_toy_iter()

    def run_wds():
        from dataclasses import dataclass

        @dataclass
        class WDSConfig:
            debug = False
            dataset_type = "auto"

            train_data = "/Users/wyf/Dataset/cc3m/cc3m/{00000..00001}.tar"
            train_data_upsampling_factors = None
            dataset_resampled = True
            train_num_samples = None
            val_data = "/Users/wyf/Dataset/cc3m/cc3m/{00000..00001}.tar"
            val_num_samples = None
            imagenet_val = None
            imagenet_v2 = None

            batch_size = 4
            workers = 1
            seed = 0
            distributed = False
            world_size = 1
            local_rank = 0
            rank = 0

        data = get_data(WDSConfig(), [lambda x: x, lambda x: x], 0, lambda x: x)
        for batch in data["train"].dataloader.create_tuple_iterator():
            print(batch[1])

    # run_wds()
