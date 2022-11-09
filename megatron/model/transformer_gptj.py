import math
import torch
import torch.nn.functional as F
import torch.nn as nn

from megatron import mpu
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.activations import get_activation
from megatron.model.utils import exists, get_fusion_type
from megatron.model.positional_embeddings import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_torch,
    AliBi,
)
from megatron.utils import print_rank_0

from megatron.model.fused_bias_dropout import (
    get_bias_dropout_add,
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
)
from megatron.model.utils import configure_sparse_attention

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

from megatron.model.positional_embeddings import SinusoidalPositionalEmbedding

from transformers.models.gptj.modeling_gptj import GPTJBlock 


class NormPipe(nn.Module):
    """Just a helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def __init__(self, norm_class, hidden_size, eps):
        super().__init__()
        self.ln_f = norm_class(hidden_size, eps=eps) # @lsp

    def forward(self, args):
        assert not isinstance(
            args, tuple
        ), "NormPipe should only receive a single tensor as input"
        return self.ln_f(args) # @lsp

        
class Linear(nn.Module):
    """
    A Parallel Linear Layer transforming the transformer outputs from hidden_size -> vocab_size
    """

    def __init__(
        self,
        gptj_args,
        parallel_output=True,
        init_method=nn.init.xavier_normal_,
    ):
        super().__init__()
        # final_linear -> lm_head
        self.lm_head = nn.Linear(
            in_features=gptj_args.hidden_size,
            out_features=gptj_args.padded_vocab_size,
            bias=True, # @lsp True   
        )
    def forward(self, hidden_states):
        return self.lm_head(hidden_states) # final_linear -> lm_head

class LinearPipe(Linear):
    """Another helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def forward(self, args):
        assert isinstance(
            args, torch.Tensor
        ), "LinearPipe expects a single argument - hidden_states"
        hidden_state = args
        # logits, bias = super().forward(hidden_state)
        logits = super().forward(hidden_state)    # gxh
        return logits


class Embedding(nn.Module):
    """Language model embeddings.
    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(
        self,
        gptj_args,
        hidden_size,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        init_method,
        num_tokentypes=0,
        use_pos_emb=True,
    ):
        # self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        super(Embedding, self).__init__()

        self.hidden_size = hidden_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes

        # Word embeddings (parallel).
        # self.word_embeddings = mpu.VocabParallelEmbedding(  # gxh
        #     gptj_args=gptj_args,
        #     num_embeddings=vocab_size,
        #     embedding_dim=self.hidden_size,
        #     init_method=self.init_method,
        # )
        self.word_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        self._word_embeddings_key = "word_embeddings"

        if gptj_args.use_bnb_optimizer:
            try:
                import bitsandbytes as bnb

                self.embedding_module = bnb.nn.StableEmbedding
            except ModuleNotFoundError:
                print(
                    "Please install bitsandbytes following https://github.com/facebookresearch/bitsandbytes."
                )
                raise Exception
        else:
            self.embedding_module = torch.nn.Embedding

        # Position embedding (serial).
        self.use_pos_emb = use_pos_emb
        if self.use_pos_emb:
            self.embedding_type = gptj_args.pos_emb
            if self.embedding_type == "learned":
                self.position_embeddings = self.embedding_module(
                    max_sequence_length, self.hidden_size
                )
                self._position_embeddings_key = "position_embeddings"
                # Initialize the position embeddings.
                self.init_method(self.position_embeddings.weight)
            elif self.embedding_type == "sinusoidal":
                self.position_embeddings = SinusoidalPositionalEmbedding(
                    self.hidden_size
                )

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = "tokentype_embeddings"
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = self.embedding_module(
                self.num_tokentypes, self.hidden_size
            )
            # Initialize the token-type embeddings.
            self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)
        self.opt_pos_emb_offset = gptj_args.opt_pos_emb_offset

        # For ticking position ids forward
        self.layer_past = None

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception("tokentype embeddings is already initialized")
        if torch.distributed.get_rank() == 0:
            print(
                "adding embedding for {} tokentypes".format(num_tokentypes), flush=True
            )
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = self.embedding_module(
            num_tokentypes, self.hidden_size
        )
        # Initialize the token-type embeddings.
        self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids, position_ids, tokentype_ids=None):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        if self.use_pos_emb and self.embedding_type in ["learned", "sinusoidal"]:
            if self.layer_past is not None:
                position_ids = position_ids + self.layer_past + 1

            self.layer_past = position_ids[:, -1]

            # OPT always adds 2 for some reason, according to the HF implementation
            if self.opt_pos_emb_offset:
                position_ids = position_ids + self.opt_pos_emb_offset
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = words_embeddings + position_embeddings
        else:
            embeddings = words_embeddings
        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None

        # Dropout.
        embeddings = self.embedding_dropout(embeddings)
        return embeddings


class EmbeddingPipe(Embedding):
    """Extends Embedding to forward attention_mask through the pipeline."""

    @property    
    def word_embeddings_weight(self):
        """Easy accessory for the pipeline engine to tie embeddings across stages."""
        return self.word_embeddings.weight

    def forward(self, args):
        assert (
            len(args) == 3
        ), f"Expected 3 arguments (input_ids, position_ids, attention_mask), but got {len(args)}."
        input_ids = args[0]
        position_ids = args[1]
        attention_mask = args[2]
        # rotary位置向量没在这块添加
        embeddings = super().forward(input_ids, position_ids)
        return embeddings, attention_mask

class TransformerLayerPipe(GPTJBlock):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline."""

    def forward(self, args):
        assert (
            len(args) == 2
        ), "ParallelTransformerLayerPipe expects 2 arguments - hidden_states and attention_mask"
        hidden_states, attention_mask = args
        # we are returning just [hidden_states, mask]
        # return super().forward(hidden_states, attention_mask), attention_mask
        # hidden_states.shape : newo： len x bsz x hidden_size， gptJ：bsz x len x hidden_size
        # print(f'hidden_states:\n{hidden_states}')
        # print(f'hidden_states_shape:\n{hidden_states.shape} sum: {hidden_states.sum()}')
        return super().forward(hidden_states, attention_mask=None)[0], attention_mask # lsp
