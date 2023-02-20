import math
import os
from collections import defaultdict
import pickle
import numpy as np
from typing import List, Tuple
from itertools import zip_longest, cycle
from functools import partial

import torch
from torch.utils.data import DataLoader

from megatron import mpu, print_rank_0
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.gpt2_dataset import GPT2Dataset
from megatron.data.samplers import DistributedBatchSampler



def make_data_loader(dataset, neox_args):
    """Build dataloader given an input dataset."""
    if dataset is None:
        return None
    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = neox_args.batch_size * world_size
    num_workers = neox_args.num_workers

    # Use a simple sampler with distributed batch sampler.
    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(
        sampler=sampler,
        batch_size=global_batch_size,
        drop_last=True,
        rank=rank,
        world_size=world_size,
    )
    # Torch dataloader.
    return torch.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=True
    )


def build_the_dataset(
    data_prefix,
    name,
    data_impl,
    num_samples,
    seq_length,
    seed,
    skip_warmup,
    build_index_mappings=True,
):
    """Build train/valid/test datasets."""

    indexed_dataset = make_indexed_dataset(data_prefix, data_impl, skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    print_rank_0("    {}:".format(name))
    print_rank_0("     no. of documents:{}".format(total_num_of_documents))
    dataset = None
    documents = np.arange(start=0, stop=total_num_of_documents, step=1, dtype=np.int32)
    dataset = GPT2Dataset(
        name,
        data_prefix,
        documents,
        indexed_dataset,
        num_samples,
        seq_length,
        seed,
        build_index_mappings=build_index_mappings,
    )
    return dataset


def build_train_valid_test_datasets(
    data_prefix,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    print(f'data_prefix{data_prefix}, data_impl:{data_impl}, skip_warmup:{skip_warmup}')
    # MMapIndexedDataset对象
    indexed_dataset = make_indexed_dataset(data_prefix, data_impl, skip_warmup)
    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(" > dataset split:")

    def print_split_stats(name, index):
        print_rank_0("    {}:".format(name))
        print_rank_0(
            "     document indices in [{}, {}) total of {} "
            "documents".format(
                splits[index], splits[index + 1], splits[index + 1] - splits[index]
            )
        )

    print_split_stats("train", 0)
    print_split_stats("validation", 1)
    print_split_stats("test", 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(
                start=splits[index], stop=splits[index + 1], step=1, dtype=np.int32
            )

            dataset = GPT2Dataset(
                name,
                data_prefix,
                documents,
                indexed_dataset,
                train_valid_test_num_samples[index],
                seq_length,
                seed,
            )
        return dataset

    train_dataset = build_dataset(0, "train")
    valid_dataset = build_dataset(1, "valid")
    test_dataset = build_dataset(2, "test")
    # GPT2Dataset对象
    return train_dataset, valid_dataset, test_dataset


def get_train_valid_test_split_(splits_string, size):
    """Get dataset splits from comma or '/' separated string list."""

    splits = []
    if splits_string.find(",") != -1:
        splits = [float(s) for s in splits_string.split(",")]
    elif splits_string.find("/") != -1:
        splits = [float(s) for s in splits_string.split("/")]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.0)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] + int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index


def get_normalized_weights_and_num_samples(
    weights: List[float], num_samples: int
) -> Tuple[List[float], List[int]]:
    # Normalize weights
    weight_sum = sum(weights)
    assert weight_sum > 0.0
    weights = [weight / weight_sum for weight in weights]
    # Add 0.5% (the 1.005 factor) so in case the blending dataset does
    # not uniformly distribute the number of samples, we still have
    # samples left to feed to the network.
    weighted_num_samples = []
    for weight in weights:
        weighted_num_samples.append(int(math.ceil(num_samples * weight * 1.005)))
    return weights, weighted_num_samples


def build_weighted_datasets(
    neox_args,
    train_num_samples,
    valid_num_samples,
    test_num_samples,
    train_weights,
    valid_weights,
    test_weights,
    build_index_mappings=True,
):
    # build individual datasets
    train_datasets, valid_datasets, test_datasets = [], [], []
    for i, (train_path, valid_path, test_path) in enumerate(
        zip_longest(
            neox_args.train_data_paths,
            neox_args.valid_data_paths,
            neox_args.test_data_paths,
        )
    ):
        if train_path:
            train_datasets.append(
                build_the_dataset(
                    data_prefix=train_path,
                    name=f"train_{i}",
                    data_impl=neox_args.data_impl,
                    num_samples=train_num_samples[i],
                    seq_length=neox_args.seq_length,
                    seed=neox_args.seed,
                    skip_warmup=(not neox_args.mmap_warmup),
                    build_index_mappings=build_index_mappings,
                )
            )

        if valid_path:
            valid_datasets.append(
                build_the_dataset(
                    data_prefix=valid_path,
                    name=f"valid_{i}",
                    data_impl=neox_args.data_impl,
                    num_samples=valid_num_samples[i],
                    seq_length=neox_args.seq_length,
                    seed=neox_args.seed,
                    skip_warmup=(not neox_args.mmap_warmup),
                    build_index_mappings=build_index_mappings,
                )
            )

        if test_path:
            test_datasets.append(
                build_the_dataset(
                    data_prefix=test_path,
                    name=f"test_{i}",
                    data_impl=neox_args.data_impl,
                    num_samples=test_num_samples[i],
                    seq_length=neox_args.seq_length,
                    seed=neox_args.seed,
                    skip_warmup=(not neox_args.mmap_warmup),
                    build_index_mappings=build_index_mappings,
                )
            )
    return train_datasets, valid_datasets, test_datasets


def weights_by_num_docs(l, alpha=0.3):
    """
    Builds weights from a multinomial distribution over groups of data according to the number of
    samples in each group.

    We sample from a group according to the probability p(L) ∝ |L| ** α,
    where p(L) is the probability of sampling from a given group,
          |L| is the number of examples in that datapoint,
          and α is a coefficient that acts to upsample data from underrepresented groups

    Hence α (`alpha`) allows us to control how much to 'boost' the probability of training on low-resource groups.

    See https://arxiv.org/abs/1911.02116 for more details
    """
    total_n_docs = sum(l)
    unbiased_sample_probs = [i / total_n_docs for i in l]

    probs = [i ** alpha for i in unbiased_sample_probs]

    # normalize
    total = sum(probs)
    probs = [i / total for i in probs]

    # weights should be the inverse of the number of samples
    unbiased_sample_probs_inverse = [1 - p for p in unbiased_sample_probs]
    weights = [p * p2 for p, p2 in zip(probs, unbiased_sample_probs_inverse)]

    # normalize
    total = sum(weights)
    weights = [i / total for i in weights]

    return weights


def build_train_valid_test_data_iterators(neox_args):
    """XXX"""

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0("> building train, validation, and test datasets ...")

    # Ensure only the first/last pipeline stages have data loaders
    if neox_args.is_pipe_parallel:
        is_first_stage = mpu.get_pipe_parallel_rank() == 0
        is_last_stage = (
            mpu.get_pipe_parallel_rank() == mpu.get_pipe_parallel_world_size() - 1
        )
        pipe_load = is_first_stage or is_last_stage
    else:
        pipe_load = True

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0 and pipe_load:
        # Number of train/valid/test samples.
        train_iters = neox_args.train_iters
        eval_iters = (train_iters // neox_args.eval_interval + 1) * neox_args.eval_iters
        test_iters = neox_args.eval_iters
        train_val_test_num_samples = [
            train_iters * neox_args.train_batch_size,
            eval_iters * neox_args.train_batch_size,
            test_iters * neox_args.train_batch_size,
        ]

        if neox_args.train_data_paths:
            # when individual train / valid / test data paths are provided
            # normalize weight values and get num samples for each dataset
            train_weights, train_num_samples = get_normalized_weights_and_num_samples(
                neox_args.train_data_weights, train_val_test_num_samples[0]
            )
            valid_weights, valid_num_samples = get_normalized_weights_and_num_samples(
                neox_args.valid_data_weights, train_val_test_num_samples[1]
            )
            test_weights, test_num_samples = get_normalized_weights_and_num_samples(
                neox_args.test_data_weights, train_val_test_num_samples[2]
            )

            # build individual datasets
            train_datasets, valid_datasets, test_datasets = build_weighted_datasets(
                neox_args,
                train_num_samples,
                valid_num_samples,
                test_num_samples,
                train_weights,
                valid_weights,
                test_weights,
                build_index_mappings=not neox_args.weight_by_num_documents,
            )

            if neox_args.weight_by_num_documents:

                # gets the number of documents in each datapath
                get_num_docs_list = lambda datasets: [
                    dataset.indexed_dataset.sizes.shape[0] for dataset in datasets
                ]
                train_num_docs, valid_num_docs, test_num_docs = (
                    get_num_docs_list(train_datasets),
                    get_num_docs_list(valid_datasets),
                    get_num_docs_list(test_datasets),
                )

                # builds weights according to alpha + the number of docs
                fn = partial(
                    weights_by_num_docs, alpha=neox_args.weighted_sampler_alpha
                )
                train_weights, valid_weights, test_weights = (
                    fn(train_num_docs),
                    fn(valid_num_docs),
                    fn(test_num_docs),
                )
                (
                    train_weights,
                    train_num_samples,
                ) = get_normalized_weights_and_num_samples(
                    train_weights, train_val_test_num_samples[0]
                )
                (
                    valid_weights,
                    valid_num_samples,
                ) = get_normalized_weights_and_num_samples(
                    valid_weights, train_val_test_num_samples[1]
                )
                test_weights, test_num_samples = get_normalized_weights_and_num_samples(
                    test_weights, train_val_test_num_samples[2]
                )

                # rebuild datasets weighted according to new weights
                train_datasets, valid_datasets, test_datasets = build_weighted_datasets(
                    neox_args,
                    train_num_samples,
                    valid_num_samples,
                    test_num_samples,
                    train_weights,
                    valid_weights,
                    test_weights,
                )

            if train_datasets:
                train_ds = BlendableDataset(train_datasets, train_weights)
            if valid_datasets:
                valid_ds = BlendableDataset(valid_datasets, valid_weights)
            if test_datasets:
                test_ds = BlendableDataset(test_datasets, test_weights)
        else:
            # when just data_path is provided
            # split dataset into train, valid and test from data_path
            print(f'neox_args.data_path: {neox_args.data_path}')
            train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
                data_prefix=neox_args.data_path,
                data_impl=neox_args.data_impl,
                splits_string=neox_args.split,
                train_valid_test_num_samples=train_val_test_num_samples,
                seq_length=neox_args.seq_length,
                seed=neox_args.seed,
                skip_warmup=(not neox_args.mmap_warmup),
            )

        # Build dataloders.
        train_dataloader = make_data_loader(train_ds, neox_args=neox_args)
        valid_dataloader = make_data_loader(valid_ds, neox_args=neox_args)
        test_dataloader = make_data_loader(test_ds, neox_args=neox_args)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and neox_args.train_iters > 0
        do_valid = valid_dataloader is not None and neox_args.eval_iters > 0
        do_test = test_dataloader is not None and neox_args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor([int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    if neox_args.is_pipe_parallel:
        # Only first/last pipeline stages have data loaders, so pipeline parallelism should
        # broadcast globally instead of just the model parallel group.
        torch.distributed.broadcast(flags, src=0)
    else:
        torch.distributed.broadcast(
            flags,
            mpu.get_model_parallel_src_rank(),
            group=mpu.get_model_parallel_group(),
        )
    neox_args.do_train = flags[0].item()
    neox_args.do_valid = flags[1].item()
    neox_args.do_test = flags[2].item()

    # Shift the start iterations.
    if train_dataloader is not None:
        train_dataloader.batch_sampler.start_iter = (
            neox_args.iteration * neox_args.gradient_accumulation_steps
        ) % len(train_dataloader)
        print_rank_0(
            "setting training data start iteration to {}".format(
                train_dataloader.batch_sampler.start_iter
            )
        )
    if valid_dataloader is not None:
        start_iter_val = (
            (neox_args.iteration * neox_args.gradient_accumulation_steps)
            // neox_args.eval_interval
        ) * neox_args.eval_iters
        valid_dataloader.batch_sampler.start_iter = start_iter_val % len(
            valid_dataloader
        )
        print_rank_0(
            "setting validation data start iteration to {}".format(
                valid_dataloader.batch_sampler.start_iter
            )
        )

    # Build iterators.
    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator


def compile_helper():
    """Compile helper function at runtime. Make sure this
    is invoked on a single process."""
    import os
    import subprocess

    path = os.path.abspath(os.path.dirname(__file__))
    ret = subprocess.run(["make", "-C", path])
    if ret.returncode != 0:
        print("Making C++ dataset helpers module failed, exiting.")
        import sys

        sys.exit(1)


def get_batch_data(data_dir, f, batch_size):
    abs_path = os.path.join(data_dir, f)
    data = pickle.load(open(abs_path, 'rb'))
    batch_data = defaultdict(list)
    for k, v in data.items():
        for i in range(0, len(v), batch_size):
            batch_data = {}
            for key in data:
                t = np.array(data[key][i: (i + 1) * batch_size])
                batch_data[key] = torch.from_numpy(t)
            yield batch_data


def metaicl_dataloader(neox_args):
    (train_dataloader, valid_dataloader, test1_dataloader, test2_dataloader) = (None, None, None, None)
    print_rank_0("> building train, validation, and test datasets ...")
    if neox_args.is_pipe_parallel:
        is_first_stage = mpu.get_pipe_parallel_rank() == 0
        is_last_stage = (
            mpu.get_pipe_parallel_rank() == mpu.get_pipe_parallel_world_size() - 1
        )
        pipe_load = is_first_stage or is_last_stage
    else:
        pipe_load = True

    if mpu.get_model_parallel_rank() == 0 and pipe_load:
        batch_size = neox_args.batch_size
        files = os.listdir(neox_args.data_path)
        if neox_args.only_eval:
            train_file_path = []
        else:
            train_file_path = [f for f in files if 'train' in f]
        test1_file_path = [f for f in files if 'test1' in f]
        test2_file_path = [f for f in files if 'test2' in f]
        dev_file_path = [f for f in files if 'dev' in f]

        print(f'train_file_path: {train_file_path} || dev_file_path: {dev_file_path} || test1_file_path: {test1_file_path} || test2_file_path: {test2_file_path}')
        train_dataloader = [d for f in train_file_path for d in get_batch_data(neox_args.data_path, f, batch_size)]
        valid_dataloader = [d for f in dev_file_path for d in get_batch_data(neox_args.data_path, f, batch_size)]
        test1_dataloader = [d for f in test1_file_path for d in get_batch_data(neox_args.data_path, f, batch_size)]
        test2_dataloader = [d for f in test2_file_path for d in get_batch_data(neox_args.data_path, f, batch_size)]

    else:
        pass
    
    if mpu.get_model_parallel_rank() == 0 and pipe_load:
        do_train = train_dataloader is not None and neox_args.train_iters > 0
        do_valid = valid_dataloader is not None and neox_args.eval_iters > 0
        do_test = test1_dataloader is not None and neox_args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor([int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    if neox_args.is_pipe_parallel:
        torch.distributed.broadcast(flags, src=0)
    else:
        torch.distributed.broadcast(
            flags,
            mpu.get_model_parallel_src_rank(),
            group=mpu.get_model_parallel_group(),
        )
    neox_args.do_train = flags[0].item()
    neox_args.do_valid = flags[1].item()
    neox_args.do_test = flags[2].item()

    # Build iterators.
    if train_dataloader is not None:
        train_data_iterator = iter(cycle(train_dataloader))
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(cycle(valid_dataloader))
    else:
        valid_data_iterator = None

    if test1_dataloader is not None:
        test1_data_iterator = iter(cycle(test1_dataloader))
    else:
        test1_data_iterator = None
    if test2_dataloader is not None:
        test2_data_iterator = iter(cycle(test2_dataloader))
    else:
        test2_data_iterator = None

    return train_data_iterator, valid_data_iterator, (test1_data_iterator, test2_data_iterator)
