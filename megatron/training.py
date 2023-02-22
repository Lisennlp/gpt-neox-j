# Copyright (c) 2021, EleutherAI contributors
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified from its original version
#

"""Pretrain utilities."""
from datetime import datetime
from functools import partial

import math
import sys
import os

import torch
import deepspeed
import numpy as np

from megatron.utils import (
    Timers,
    init_wandb,
    get_ltor_masks_and_position_ids,
    reduce_losses,
    get_ltor_masks_and_position_ids_,
)


from megatron import print_rank_0, mpu
from megatron.model import (
    GPT2ModelPipe,
    SoftEmbedding,
    get_params_for_weight_decay_optimization,
)
from megatron.checkpointing import load_checkpoint, save_checkpoint
from megatron.data.data_utils import build_train_valid_test_data_iterators, metaicl_dataloader
from megatron.initialize import initialize_megatron
from megatron.learning_rates import AnnealingLR
from megatron.logging import tb_wandb_log, training_log
from megatron.utils import (
    OverflowMonitor,
    get_noise_scale_logger,
    get_total_params,
    CharCounter,
)
from megatron.model.gpt2_model import cross_entropy
from eval_tasks import run_eval_harness


def pretrain(neox_args):
    """Main training program.

    This function will run the following in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the model.

    Arguments:
        neox_args: an instance of NeoXArgs containing the configuration for pretrain

    """
    neox_args.pred_results_dir = None
    if neox_args.only_eval:
        pred_results_dir = os.path.join(neox_args.data_path, f'pred_results')
        if not os.path.exists(pred_results_dir):
            os.mkdir(pred_results_dir)
        neox_args.pred_results_dir = pred_results_dir

    # setup logging and timers
    init_wandb(neox_args=neox_args)
    timers = Timers(
        use_wandb=neox_args.use_wandb, tensorboard_writer=neox_args.tensorboard_writer
    )

    # Initialize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(neox_args=neox_args) # 设置每个model的seed，并初始化topology（可以获取每个rank编号，stage id等等）
    print_rank_0(f'iters nums: {neox_args.train_iters}')

    # Model, optimizer, and learning rate.
    timers("model and optimizer").start()
    # 将model和optimizer分在每个卡上
    model, optimizer, lr_scheduler = setup_model_and_optimizer(
        neox_args=neox_args, use_cache=False
    )
   
    for k, v in model.named_parameters():
        print_rank_0(k, v.shape, v.sum().item(), v.dtype, rank=0)
    #     print_rank_0(k, v.shape, v.sum().item(), rank=7)

    timers("model and optimizer").stop()

    # Data stuff.
    timers("train/valid/test data iterators").start()

    if neox_args.icl_or_neo == 'icl':
        # @lsp-data====================
        (
            train_data_iterator,
            valid_data_iterator,
            test_data_iterator,
        ) = metaicl_dataloader(neox_args=neox_args)
        #@lsp-data====================
    else:
        (
            train_data_iterator,
            valid_data_iterator,
            test_data_iterator,
        ) = build_train_valid_test_data_iterators(neox_args=neox_args)
        # # 单train，valid，test文件时
        # neox_args.data_path = neox_args.train_data_path
        # (train_data_iterator, _, _,) = build_train_valid_test_data_iterators(neox_args=neox_args)
        # neox_args.data_path = neox_args.valid_data_path
        # (valid_data_iterator, _, _,) = build_train_valid_test_data_iterators(neox_args=neox_args)
        # neox_args.data_path = neox_args.test_data_path
        # (test_data_iterator, _, _,) = build_train_valid_test_data_iterators(neox_args=neox_args)

    timers("train/valid/test data iterators").stop()

    # Print setup timing.
    print_rank_0("done with setups ...")
    timers.log(["model and optimizer", "train/valid/test data iterators"])

    iteration = 0
    print_rank_0("training ...")
    if neox_args.do_train and neox_args.train_iters > 0:
        iteration = train(
            neox_args=neox_args,
            timers=timers,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_data_iterator=train_data_iterator,
            valid_data_iterator=valid_data_iterator,
            test_data_iterator=test_data_iterator,
        )

    # if neox_args.do_valid:
    #     prefix = "the end of training for val data"
    #     evaluate_and_print_results(
    #         neox_args=neox_args,
    #         prefix=prefix,
    #         forward_step_func=forward_step,
    #         data_iterator=valid_data_iterator,
    #         model=model,
    #         iteration=iteration,
    #         verbose=False,
    #         timers=timers,
    #     )

    # if neox_args.save and iteration != 0:
    #     save_checkpoint(
    #         neox_args=neox_args,
    #         iteration=iteration,
    #         model=model,
    #         optimizer=optimizer,
    #         lr_scheduler=lr_scheduler,
    #     )

    # if neox_args.do_test:
    #     # Run on test data.
    #     prefix = "the end of training for test data"
    #     evaluate_and_print_results(
    #         neox_args=neox_args,
    #         prefix=prefix,
    #         forward_step_func=forward_step,
    #         data_iterator=test_data_iterator,
    #         model=model,
    #         iteration=0,  # iteration 0 in order to always use full test data
    #         verbose=True,
    #         timers=timers,
    #     )


def _get_batch(neox_args, tokenizer, keys, data, datatype):
    """Support function for get_batch / get_batch pipe (to avoid code repetition)"""
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b["text"].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=neox_args.tokenizer.eod,
        eod_mask_loss=neox_args.eod_mask_loss,
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


def get_batch(neox_args, data_iterator):
    """Generate a batch"""

    # Items and their type.
    keys = ["text"]
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    return _get_batch(
        neox_args=neox_args,
        tokenizer=neox_args.tokenizer,
        keys=keys,
        data=data,
        datatype=datatype,
    )


def _get_batch_icl(neox_args, tokenizer, keys, data, datatype):
    """Support function for get_batch / get_batch pipe (to avoid code repetition)"""
    data_b = mpu.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens_ = data_b["input_ids"].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # loss_mask = data_b["token_type_ids"][:, 1:].bool()
    if "token_type_ids" in data_b:
        loss_mask = data_b["token_type_ids"][:, :-1].bool()
    else:
        loss_mask = torch.zeros_like(tokens)

    if "attention_mask" in data_b:
        attention_mask = data_b["attention_mask"][:, :-1].bool()
    else:
        attention_mask = torch.ones_like(tokens)

    # Get the masks and position ids.
    position_ids = get_ltor_masks_and_position_ids_(
        data=tokens,
        eod_token=0,
        eod_mask_loss=neox_args.eod_mask_loss,
    )
    print(f'tokens.shape[1]: {tokens.shape[1]}')
    print(f'labels.shape[1]: {labels.shape[1]}')
    print(f'loss_mask.shape[1]: {loss_mask.shape[1]}')
    print(f'attention_mask.shape[1]: {attention_mask.shape[1]}')
    print(f'position_ids.shape[1]: {position_ids.shape[1]}')

    return tokens, labels, loss_mask, attention_mask, position_ids


def get_batch_pipe(data, neox_args):
    """A modification of get_batch() to work with the latest batch instead of an iterator."""
    # Items and their type.
    datatype = torch.int64
    # @lsp
    if neox_args.icl_or_neo == 'icl':
        keys = list(data.keys())
        # keys = ["input_ids", 'attention_mask', 'token_type_ids']
        tokens, labels, loss_mask, attention_mask, position_ids = _get_batch_icl(
        neox_args, neox_args.tokenizer, keys, data, datatype)
        # loss_mask[:, :attention_mask.sum()] = 1
        attention_mask = torch.triu(attention_mask.new_ones(attention_mask.shape[1], attention_mask.shape[1]),
                                       diagonal=1).bool()[None,None,:, :]
    else:
        keys = ['text']
        tokens, labels, loss_mask, attention_mask, position_ids = _get_batch(
            neox_args, neox_args.tokenizer, keys, data, datatype
        )
    return (tokens, position_ids, attention_mask), (labels, loss_mask)
    # /Users/lishengping/codes/others/DeepSpeed/deepspeed/runtime/pipe/engine.py line 789


def forward_step(data_iterator, model, neox_args, timers, return_logits=False):
    """Forward step."""
    if neox_args.is_pipe_parallel:
        return model.eval_batch(data_iterator, return_logits=return_logits)

    # Get the batch.
    if timers is not None:
        timers("batch generator").start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        neox_args=neox_args, data_iterator=data_iterator
    )
    if timers is not None:
        timers("batch generator").stop()

    outputs = model((tokens, position_ids, attention_mask))
    loss = cross_entropy(
        outputs, (labels, loss_mask), _fp16=neox_args.fp16_lm_cross_entropy
    )
    if return_logits:
        return loss, outputs
    return loss


def get_model(neox_args, use_cache=False):
    """Build the model."""

    print_rank_0("building GPT2 model ...")

    # Build model on cpu. 将model变成pipe model
    model = GPT2ModelPipe(
        neox_args=neox_args,
        num_tokentypes=0,
        parallel_output=True,
        topology=mpu.get_topology(),
        use_cache=use_cache,
    )

    ### soft prompt tuning stuff ###
    if neox_args.soft_prompt_tuning is not None and neox_args.soft_prompt_tuning.get(
        "enabled", False
    ):
        soft_prompt = SoftEmbedding(
            neox_args,
            wte=getattr(model, "0").word_embeddings,
            n_tokens=neox_args.soft_prompt_tuning.get("n_tokens", 10),
            init_string=neox_args.soft_prompt_tuning.get("init_string", ""),
            init_range=neox_args.soft_prompt_tuning.get("init_range", 0.5),
        )
        model.insert_layers(
            layers=soft_prompt, idx=1
        )  # insert the soft prompt layer directly after the word embeddings

        # freeze everything but the soft prompt
        for name, param in model.named_parameters():
            if not "soft_embedding" in name:
                param.requires_grad = False

    if not neox_args.is_pipe_parallel:
        # Export PipeParallel model to nn.Sequential model to avoid the overhead of deepspeed's pipe parallel training
        model = model.to_sequential()

    if neox_args.deepspeed:
        # DeepSpeed handles CUDA, FP16, and DDP components.
        return model
    else:
        raise ValueError("Must be using deepspeed to run neox")


def get_optimizer(model, neox_args):
    """Set up the optimizer."""
    if neox_args.no_load_optim:
        return None, None
    # Build parameter groups (weight decay and non-decay). list 分配参数的decay
    param_groups = get_params_for_weight_decay_optimization(model, neox_args)
    print_rank_0(
        f'Configuring Optimizer type: {neox_args.optimizer_type} with params: {neox_args.optimizer["params"]}'
    )

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group["params"]:
            if not hasattr(param, "model_parallel"):
                param.model_parallel = False

    # Filter out params that don't require a grad (for soft prompt tuning, etc.)
    _param_groups = []
    for param_group in param_groups:
        trainable_params = [p for p in param_group["params"] if p.requires_grad]
        param_group["params"] = trainable_params
        _param_groups.append(param_group)
    param_groups = _param_groups

    if neox_args.optimizer_type.lower() in ["cpu_adam", "cpu_torch_adam"]:
        if neox_args.optimizer == "cpu_torch_adam":
            print(f'优化器类型: cpu_torch_adam')
            cpu_adam_optimizer = torch.optim.Adam
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(
            param_groups,
            weight_decay=neox_args.weight_decay,
            **neox_args.optimizer["params"],
        )
    elif neox_args.optimizer_type.lower() == "onebitadam":
        assert neox_args.deepspeed
        optimizer = None
        # onebitadam needs to be instantiated within the deepspeed engine to work :|
    elif neox_args.optimizer_type.lower() == "sm3":
        from .optimizers import SM3

        optimizer = SM3(param_groups, **neox_args.optimizer["params"])
    elif neox_args.optimizer_type.lower() == "madgrad_wd":
        from .optimizers import madgrad_wd

        optimizer = madgrad_wd(
            param_groups,
            weight_decay=neox_args.weight_decay,
            **neox_args.optimizer["params"],
        )
    # 走这
    elif neox_args.optimizer_type.lower() == "adam":
        # Use Adam
        if neox_args.use_bnb_optimizer:
            try:
                import bitsandbytes as bnb

                adam_optimizer = bnb.optim.Adam8bit
            except ModuleNotFoundError:
                print(
                    "Please install bitsandbytes following https://github.com/facebookresearch/bitsandbytes."
                )
                raise Exception
        else:
            try:
                # default to apex as it's slightly faster，使用FuseAdam
                from apex.optimizers import FusedAdam as Adam
                print(f'优化器：apex的FuseAdam')
            except ImportError:
                # if apex isn't installed, use deepspeed's FusedAdam
                print(
                    "WARNING: APEX not installed - defaulting to deepspeed's fused adam"
                )
                from deepspeed.ops.adam import FusedAdam as Adam
                print(f'优化器：deepspeed的FuseAdam')
            adam_optimizer = Adam
        optimizer = adam_optimizer(
            param_groups,
            weight_decay=neox_args.weight_decay,
            **neox_args.optimizer["params"], # 获取lr，beta，eps等
        )
    else:
        raise ValueError(f"Optimizer type {neox_args.optimizer_type} not recognized")

    if neox_args.deepspeed:
        # fp16 wrapper is not required for DeepSpeed.
        return optimizer, param_groups
    else:
        raise ValueError("Must be using deepspeed to run neox")


def get_learning_rate_scheduler(optimizer, neox_args):
    """Build the learning rate scheduler."""
    if neox_args.no_load_optim:
        # TODO: this should be configured as a separate arg
        return None
    if neox_args.deepspeed and neox_args.optimizer_type.lower() == "onebitadam":
        print_rank_0(
            "WARNING: onebitadam requires the lr scheduler be built by deepspeed - "
            "Make sure one is added to your deepspeed config"
        )
        return None

    # Add linear learning rate scheduler.
    if neox_args.lr_decay_iters is not None:
        num_iters = neox_args.lr_decay_iters
    else:
        num_iters = neox_args.train_iters
    num_iters = max(1, num_iters)
    init_step = 0
    warmup_iter = neox_args.warmup * num_iters
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=neox_args.lr,
        warmup_iter=warmup_iter,
        total_iters=num_iters,
        decay_style=neox_args.lr_decay_style,
        last_iter=init_step,
        min_lr=neox_args.min_lr,
        use_checkpoint_lr_scheduler=neox_args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=neox_args.override_lr_scheduler,
    )

    return lr_scheduler


def setup_model_and_optimizer(neox_args, use_cache=False, iteration=None):
    """Setup model and optimizer."""
    # pipeline model，已经分层了
    model = get_model(neox_args=neox_args, use_cache=use_cache)
    for k, v in model.named_parameters():
        print_rank_0(k, v.shape, v.sum().item(), v.dtype, rank=7)
    # 当加载预训练模型时
    model.load_state_dir(neox_args.load)
    # model.half()
    # 每个卡的model拥有自己的optimizer
    optimizer, param_groups = get_optimizer(model=model, neox_args=neox_args)
    lr_scheduler = get_learning_rate_scheduler(optimizer=optimizer, neox_args=neox_args)
    if neox_args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")
        if neox_args.no_load_optim:
            assert optimizer is None
            _model_params = None
            _lr_scheduler = None
        else:
            _model_params = param_groups if optimizer is None else None
            _lr_scheduler = lr_scheduler
        
        # 初始化为pipeline engine 
        # /nas2/kf/miniconda3/envs/pytorch1.10/lib/python3.8/site-packages/deepspeed/__init__.py
        # __import__('ipdb').set_trace()
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=neox_args,
            lr_scheduler=_lr_scheduler,
            dist_init_required=False,
            model_parameters=_model_params,
            config_params=neox_args.deepspeed_config,
            mpu=mpu if not neox_args.is_pipe_parallel else None,
        )
        print(f'-------------')
        model.total_params = get_total_params(model.module)
        print_rank_0(f' > total params: {"{:,}".format(model.total_params)}')

        if neox_args.is_pipe_parallel:
            model.set_has_attention_mask(True)
            model.set_batch_fn(partial(get_batch_pipe, neox_args=neox_args))
    else:
        raise ValueError("Must be using deepspeed to run neox")

    # 当保存了中间步骤时应该走这
    # if neox_args.load is not None:
    #     neox_args.iteration = load_checkpoint(
    #         neox_args=neox_args,
    #         model=model,
    #         optimizer=optimizer,
    #         lr_scheduler=lr_scheduler,
    #         iteration=iteration,
    #     )
    #     print_rank_0(
    #         f"Loading checkpoint and starting from iteration {neox_args.iteration}"
    #     )
    # else:
    #     neox_args.iteration = 0
    return model, optimizer, lr_scheduler


def backward_step(neox_args, timers, optimizer, model, loss):
    """Backward step."""

    # Backward pass.
    timers("backward-backward").start()
    if neox_args.deepspeed:
        model.backward(loss)
    else:
        raise ValueError("Must be using deepspeed to run neox")
    timers("backward-backward").stop()

    if neox_args.deepspeed:
        # DeepSpeed backward propagation already addressed all reduce communication.
        # Reset the timer to avoid breaking timer logs below.
        timers("backward-allreduce").reset()
    else:
        raise ValueError("Must be using deepspeed to run neox")


def train_step(neox_args, timers, data_iterator, model, optimizer, lr_scheduler):
    """Single training step."""

    # Pipeline parallelism schedules forward/backward/step
    if neox_args.is_pipe_parallel:
        reduced_loss = train_step_pipe(
            neox_args=neox_args, timers=timers, model=model, data_iterator=data_iterator
        )
    else:
        losses = []
        for _ in range(neox_args.gradient_accumulation_steps):
            # Forward model for one step.
            timers("forward").start()
            loss = forward_step(
                neox_args=neox_args,
                timers=timers,
                data_iterator=data_iterator,
                model=model,
            )
            timers("forward").stop()
            losses.append(loss)
            # Calculate gradients, reduce across processes, and clip.
            timers("backward").start()
            backward_step(
                neox_args=neox_args,
                timers=timers,
                optimizer=optimizer,
                model=model,
                loss=loss,
            )
            timers("backward").stop()
            # Update parameters.
            timers("optimizer").start()
            if neox_args.deepspeed:
                model.step()
            else:
                raise ValueError("Must be using deepspeed to run neox")
            timers("optimizer").stop()
        reduced_loss = {
            "lm_loss": reduce_losses(losses).mean()
        }  # reduces losses across machines for logging

    if neox_args.precision == "fp16" and model.optimizer.overflow:
        skipped_iter = 1
    else:
        skipped_iter = 0

    return reduced_loss, skipped_iter


def train_step_pipe(neox_args, timers, model, data_iterator):
    """Single training step with DeepSpeed's pipeline parallel engine."""

    assert neox_args.deepspeed
    # /Users/lishengping/codes/others/DeepSpeed/deepspeed/runtime/pipe/schedule.py控制每一步的forward顺序
    loss = model.train_batch(data_iter=data_iterator)
    # for k, v in model.module.named_parameters():
    #     if v.grad is not None:
    #         print_rank_0(f'name: {k} grad: {v.grad.data} shape: {v.grad.data.shape}')
    #     else:
    #         print(f'nograd name: {k}')
            # print_rank_0(k, v.grad.data, v.grad.data.shape)
    # print_rank_0(f'train loss: {loss.item()}')
    loss_dict = {"lm_loss": loss}
    # print_rank_0(f'loss: {loss.item()}')
    # Don't break Megatron's timers because we changed code paths.
    for t in [
        "forward",
        "backward",
        "allreduce",
        "optimizer",
        "batch generator",
        "data loader",
    ]:
        timers(t).reset()
    return loss_dict


def train(
    neox_args,
    timers,
    model,
    optimizer,
    lr_scheduler,
    train_data_iterator,
    valid_data_iterator,
    test_data_iterator,
):
    """Train the model function."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = neox_args.iteration

    timers("interval time").start()
    report_memory_flag = True

    # get noise scale logger (if neox_args.log_gradient_noise_scale is True)
    noise_scale_logger = get_noise_scale_logger(neox_args)

    # to monitor if we've skipped many iterations in a row and trigger an early exit
    overflow_monitor = OverflowMonitor(optimizer)
    iteration = 0


    prefix = "iteration {}".format(iteration)
    # 起始评测
    valid_iters = [152, 163, 112] #  dev:   bench: 1298, meta: 891
    valid_iters = [12, 13, 12] #  dev:   bench: 1298, meta: 891

    valid_data_loaders = [valid_data_iterator, *test_data_iterator]
    valid_types = ['meta_seen_dev', 'big_bench_test1', 'meta_unseen_test2']
    for i in range(1):
        neox_args.eval_iters = valid_iters[i]
        generate_and_print_results(neox_args, model, valid_data_loaders[-1])
        # evaluate_and_print_results(
        #             neox_args=neox_args,
        #             prefix=prefix,
        #             forward_step_func=forward_step,
        #             data_iterator=valid_data_loaders[i],
        #             model=model,
        #             iteration=iteration,
        #             verbose=False,
        #             timers=timers,
        #             valid_type=valid_types[i]
        #         )
        exit(0)
    if neox_args.only_eval:
        exit(0)
    while iteration < neox_args.train_iters:
        loss_dict, skipped_iter = train_step(
            neox_args=neox_args,
            timers=timers,
            data_iterator=train_data_iterator,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        iteration += 1
        overflow_monitor.check(skipped_iter)  # check for repeated overflow
        if neox_args.log_gradient_noise_scale:  # log noise scale if applicable
            noise_scale_logger.update()

        # get learning rate (if present) - if doing soft prompt tuning + pipe parallel, you
        # may have no tunable parameters on a specific rank
        if optimizer.param_groups:
            lr = optimizer.param_groups[0].get("lr", 0)
        else:
            lr = 0

        # Logging.
        report_memory_flag = training_log(
            neox_args=neox_args,
            timers=timers,
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=lr,
            iteration=iteration,
            loss_scale=optimizer.cur_scale if neox_args.precision == "fp16" else None,
            report_memory_flag=report_memory_flag,
            skipped_iter=skipped_iter,
            model=model,
            optimizer=optimizer,
            noise_scale_logger=noise_scale_logger,
        )

        # Checkpointing
        if (
            neox_args.save
            and neox_args.save_interval
            and iteration % neox_args.save_interval == 0
        ):
            save_checkpoint(
                neox_args=neox_args,
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )
            last_model_save_path = os.path.join(neox_args.save, f'global_step{iteration - 5000}')
            if torch.distributed.get_rank() == 0:
                if os.path.exists(last_model_save_path):
                    last_model_files = os.listdir(last_model_save_path)
                    for last_model_file in last_model_files:
                        if 'zero' in last_model_file:
                            abs_path = os.path.join(last_model_save_path, last_model_file)
                            if os.path.exists(abs_path):
                                os.remove(abs_path)
                                print(f'del file: ’{abs_path}‘ successful !!!')
                else:
                    print(f'file dir ‘{last_model_save_path}’ is not exists !!!')

        # Evaluation
        if (
            neox_args.eval_interval
            and iteration % neox_args.eval_interval == 0
            and neox_args.do_valid
        ):
            prefix = "iteration {}".format(iteration)
            for i in range(3):
                neox_args.eval_iters = valid_iters[i]
                evaluate_and_print_results(
                            neox_args=neox_args,
                            prefix=prefix,
                            forward_step_func=forward_step,
                            data_iterator=valid_data_loaders[i],
                            model=model,
                            iteration=iteration,
                            verbose=False,
                            timers=timers,
                            valid_type=valid_types[i]
                        )

        if neox_args.exit_interval and iteration % neox_args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rank = torch.distributed.get_rank()
            print_rank_0(
                "rank: {} | time: {} | exiting the program at iteration {}".format(
                    rank, time_str, iteration
                )
            )
            sys.exit()

    return iteration

def evaluate(
    neox_args, forward_step_fn, data_iterator, model, verbose=False, timers=None
):
    """Evaluation.
    neox_args: NeoX Arguments
    forward_step_fn: function with args `neox_args, timers,
                    data_iterator & model that will run a forward pass on the model
    data_iterator: Iterator that iterates over batches of data. Should return data in the form:
                    {'text': np.array([tokens], dtype=np.int64)}
                    where the size of the array is the model's context size + 1
                    (`get_batch` transforms it into inputs / labels)
    """
    # Turn on evaluation mode which disables dropout.
    model.eval()
    losses = []
    if neox_args.char_level_ppl:
        data_iterator = CharCounter(data_iterator, neox_args.tokenizer)

    with torch.no_grad():
        iteration = 0
        while iteration < neox_args.eval_iters:
            iteration += 1
            if verbose and iteration % neox_args.log_interval == 0:
                print_rank_0(
                    "Evaluating iter {}/{}".format(iteration, neox_args.eval_iters)
                )

            # although we're not accumulating gradients here, we count one iter as train_batch_size_per_gpu * g.a.s
            # to be consistent with deepspeed's pipe parallel engine
            # since pipe parallel already takes gas into account - default to 1 here if pipe parallel is true
            for _ in range(
                1
                if neox_args.is_pipe_parallel
                else neox_args.gradient_accumulation_steps
            ):
                # Forward evaluation
                loss = forward_step_fn(
                    model=model,
                    data_iterator=data_iterator,
                    neox_args=neox_args,
                    timers=timers,
                )
                losses.append(loss)

            # When contiguous memory optimizations are enabled, the buffers
            # allocated by the optimizations are deallocated during backward pass
            # in the absence of backward pass the buffers should be reset after each
            # forward pass
            if neox_args.deepspeed and neox_args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()

    # reduces losses across processes for logging & run eval harness tasks
    eval_results = {"lm_loss": reduce_losses(losses).mean().item()}
    eval_results["lm_loss_ppl"] = math.exp(eval_results["lm_loss"])

    if neox_args.char_level_ppl:
        # calculate character level perplexity, if specified
        # if neox_args.char_level_ppl:
        # unwrap the data_iterator
        tokens_per_char = data_iterator.tokens_per_char()
        print_rank_0(f"Counting chars took {data_iterator.total_time} seconds")

        data_iterator = data_iterator.data_iterator
        eval_results["lm_loss_char_lvl_ppl"] = math.exp(
            eval_results["lm_loss"] * tokens_per_char
        )

    if neox_args.eval_tasks:
        eval_results.update(
            run_eval_harness(
                model, forward_step_fn, neox_args, eval_tasks=neox_args.eval_tasks
            ).get("results")
        )
    # Move model back to the train mode.
    model.train()
    return eval_results


def evaluate_and_print_results(
    neox_args,
    prefix,
    forward_step_func,
    data_iterator,
    model,
    iteration,
    verbose=False,
    timers=None,
    valid_type='valid'
):
    """Helper function to evaluate and dump results on screen."""
    total_loss_dict = evaluate(
        neox_args=neox_args,
        forward_step_fn=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        verbose=verbose,
        timers=timers,
    )
    string = f" {valid_type} results at {prefix} | "
    for k, v in total_loss_dict.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                k3 = "_".join([k, k2])
                string += f"{k3} value: {v2:.6E} | "
                tb_wandb_log(
                    f"{valid_type}/{k3}",
                    v2,
                    iteration,
                    use_wandb=neox_args.use_wandb,
                    tensorboard_writer=neox_args.tensorboard_writer,
                )
        else:
            string += f"{k} value: {v:.6E} | "
            tb_wandb_log(
                f"{valid_type}/{k}",
                v,
                iteration,
                use_wandb=neox_args.use_wandb,
                tensorboard_writer=neox_args.tensorboard_writer,
            )

    length = len(string) + 1
    print_rank_0("-" * length)
    print_rank_0(string)
    print_rank_0("-" * length)


def generate_and_print_results(neox_args, model, data):
    model.eval()
    iteration = 0
    src_rank = model.grid.stage_to_global(model.num_stages - 1)
    while iteration < 1:
        if data is not None:
            cur_batch = data[iteration]
            input_ids = cur_batch['input_ids']
            loss_mask = cur_batch['token_type_ids']
            attention_mask = cur_batch['attention_mask']
            mask_loc = torch.where(loss_mask == 1)[1]
            mask_start = mask_loc[0] + 1 # 因为之后get_batch_icl还要去掉最后一个token
            mask_end = mask_loc[-1] + 1
            labels = input_ids[:, mask_start: mask_end]
            # print(f'labels: {labels}')
            input_id = input_ids[:, :mask_start + 1]
            loss_mask = loss_mask[:, :mask_start + 1]
            attention_mask = attention_mask[:, :mask_start + 1]
            d = {"input_ids": input_id, "attention_mask": attention_mask, "token_type_ids": loss_mask}
            # d = {"input_ids": input_id}
            mask_start = torch.cuda.LongTensor([mask_start])
        else:
            mask_start = torch.cuda.LongTensor([1000])

        torch.distributed.broadcast(tensor=mask_start, src=src_rank, group=mpu.get_pipe_parallel_group())
        count = 0
        generated_tokens = None
        while mask_start < neox_args.seq_length:
            if data is not None:
                if generated_tokens is not None:
                    d['input_ids'] = torch.cat([d['input_ids'], generated_tokens.view(1, 1).cpu()], dim=1)
                    d['attention_mask'] = torch.cat([d['attention_mask'], torch.tensor([[1]])], dim=1)
                    d['token_type_ids'] = torch.cat([d['token_type_ids'], torch.tensor([[1]])], dim=1)
                model_inputs = iter([d])
            else:
                model_inputs = None
            with torch.no_grad():
                loss, logits = model.eval_batch(model_inputs, return_logits=True)
            if logits is not None:  # if pipe parallel, not all ranks return logits
                generated_token_logits = logits[:, -1].contiguous()  # [bs, seq, vocab_size] -> [bs, vocab_size]
                generated_tokens = torch.argmax(generated_token_logits, dim=-1).view(-1)
                generated_tokens = torch.cuda.LongTensor(generated_tokens)
            else:
                generated_tokens = torch.cuda.LongTensor([0])
            torch.distributed.broadcast(tensor=generated_tokens, src=src_rank, group=mpu.get_pipe_parallel_group())
            print(f'generated_tokens: {generated_tokens}')
            if generated_tokens.item() == 628 or generated_tokens.item() == 198 or count > 20:
                break
            mask_start += 1
            count += 1
        iteration += 1