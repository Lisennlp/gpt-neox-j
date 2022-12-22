
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


import torch

from .initialize import get_model_parallel_group
from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_world_size
from .utils import VocabUtility


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target):
        loss_mask = target[1]
        target = target[0]
        # vocab_parallel_logits: b x len x vocab_size
        # Maximum value along vocab dimension across all GPUs.
        # @lsp
        logits_max, preds = torch.max(vocab_parallel_logits, dim=-1)
        # indices: b x len
        torch.distributed.all_reduce(
            preds,
            op=torch.distributed.ReduceOp.MAX,
            group=get_model_parallel_group(),
        )
        # =======
        torch.distributed.all_reduce(
            logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=get_model_parallel_group(),
        )
        # Subtract the maximum value.即 vocab_parallel_logits - logits_max，每个token减去最大的logit，
        # 这样一来，logits均为负值，为什么都要变为负值
        vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

        # Get the partition's vocab indices
        # 每个rank都有自己的词表索引范围，不在范围内的就被mask掉，因此最后在做softmax的时候，分母需要all_reduce
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_model_parallel_rank()
        world_size = get_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, rank, world_size
        )

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        # mask掉不在词表内的id，1表示mask掉，0表示保留
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0 # 0表示mask，1表示保留，和target_mask相反

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        # logits将batch和len维合并在一起了，因此之后的target也需要变成1维
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1) # 变成1维
        arange_1d = torch.arange(
            start=0, end=logits_2d.size()[0], device=logits_2d.device
        )
        # masked_target_1d: b x 1024; logits_2d: (b * 1024) x vocab_size
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        # target： b x 1024，将predicted_logits_1d恢复到2维， predicted_logits： b x 1024
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        # 从其他GPU获取logits，如果是pp的话其实其他gpu没有，mp需要allreduce
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_model_parallel_group(),
        )

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_model_parallel_group(),
        )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        # sum_exp_logits表示softmax分母, 即exp（all_logits），而分子为exp(predicted_logits)，但因为loss = -logsoftmax(logit)
        # 即loss = -log(exp(predicted_logits) / sum_exp_logits) = -(predicted_logits - logsum_exp_logits)
        # = logsum_exp_logits - predicted_logits
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Store softmax, target-mask and masked-target for backward pass.
        # 这里不太明白
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)
        # if loss_mask is not None:
        #     mask_loss = loss.masked_select(loss_mask.bool())
        #     print(f'train mask loss : {mask_loss.view(-1)}')
        # else:
        #     print('loss mask is none')
        return loss, preds

    @staticmethod
    def backward(ctx, grad_output, preds=None):

        # Retrieve tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as their gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).float()

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None


def vocab_parallel_cross_entropy(vocab_parallel_logits, target, loss_mask=None):
    """Helper function for the cross entropy."""
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, [target, loss_mask])