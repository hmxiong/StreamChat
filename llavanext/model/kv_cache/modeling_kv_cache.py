import math
from typing import List, Optional, Tuple, Union
from types import MethodType

import pdb
import torch
from torch import nn
import torch.utils.checkpoint

import torch.nn.functional as F

# from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    # apply_rotary_pos_emb,
    LlamaRotaryEmbedding,
    # apply_rotary_pos_emb,
)
import sys
sys.path.append('/13390024681/llama/EfficientVideo/Ours/llavanext/model/kv_cache')
from .modeling_llama import LlamaForCausalLM
import types
from .modeling_llama import LlamaRotaryEmbedding, LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding, LlamaAttention, LlamaDecoderLayer, LlamaModel, logger
from transformers.modeling_outputs import BaseModelOutputWithPast

__all__ = ["H2OLlamaForCausalLM", "H2OLlamaAttention",
            'H2OLlamaAttention_streaming', 'H2OLlamaForCausalLM_streaming']


from transformers.configuration_utils import PretrainedConfig

LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

class LlamaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
            Llama 2 up to 4096, CodeLlama up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import LlamaModel, LlamaConfig

    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = LlamaConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# the same as the original code 
def _make_causal_mask(
    bsz: int, tgt_len: int, past_key_values_length: int, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

class PivotKVCache_LayerWise:
    def __init__(
        self,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        hh_ratio=None,
        recent_ratio=None
    ):
        print(f"PIVOTKVCache-LayerWise: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.threshold = None
        # print(f"H2OKVCache-LayerWise: {hh_ratio}, {recent_ratio}")

    def __call__(self, past_key_values, attn_score_cache):

        if self.hh_ratio is not None and attn_score_cache.shape[-2]>1:
            self.hh_size = int(attn_score_cache.shape[-1] * self.hh_ratio)
            self.recent_size = int(attn_score_cache.shape[-1] * self.recent_ratio)
            self.cache_size = self.hh_size + self.recent_size
            self.image_save_ratio = self.hh_ratio

        # print(attn_score_cache.shape)
        self._update_hh_score(attn_score_cache)
        if past_key_values is None:
            return None
        seq_len = past_key_values[0].size(self.k_seq_dim)

        if seq_len <= self.cache_size:
            return past_key_values

        # hh-selection
        
        bsz, num_heads, _, head_dim = past_key_values[0].shape # [1, 32, seq_len, 128]
        #################################before-code#####################################
        select_hh_scores = self.hh_score[:, :seq_len - self.recent_size] # prune from the oldest tokens
        # merge_weights = nn.functional.softmax(self.hh_score, dim=-1)
        # merge_weights = self.hh_score
        
        # breakpoint()
        select_hh_scores[:, 0:4] = torch.full((attn_score_cache.shape[1], 4), float('inf'))
        sort_indices = torch.argsort(select_hh_scores, dim=-1, descending=True)
        keep_topk = sort_indices[:, 0:self.hh_size] # [32, self.hh_size]
        
        keep_topk = keep_topk.sort().values
        keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

        mask = torch.zeros(self.hh_score.shape, dtype=torch.bool).to(past_key_values[0].device)
        mask = mask.scatter(-1, keep_idx, 1)

        k_hh_recent = past_key_values[0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = past_key_values[1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        self.hh_score = self.hh_score[mask].view(num_heads, self.cache_size)
        ###############################before-code########################################
        ###############################before-merge#######################################
        # applying merge here
        # breakpoint()
        k_hh_pruned = past_key_values[0].squeeze()[~mask].view(bsz, num_heads, -1, head_dim)
        similarity = (k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ ((k_hh_recent / (torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128))).transpose(-1, -2)) # cosin
        max_values, max_indices = similarity.max(dim=-1)
        # # pivot merge
        merged_indices = max_indices.unsqueeze(-1).repeat(1, 1, 1, 128)
        k_hh_selected = torch.gather(input=k_hh_recent, dim=2, index=merged_indices)
        k_hh_merged = (k_hh_pruned + k_hh_selected)/2
        k_hh_recent = torch.scatter_reduce(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_merged, reduce='mean', include_self=True) # include_self=True seems decrease the performance
        v_hh_pruned = past_key_values[1].squeeze()[~mask].view(bsz, num_heads, -1, head_dim)
        v_hh_selected = torch.gather(input=v_hh_recent, dim=2, index=merged_indices)
        v_hh_merged = (v_hh_pruned + v_hh_selected)/2
        v_hh_recent = torch.scatter_reduce(input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_merged, reduce='mean', include_self=True)
        ###############################before-merge#######################################
       
        return (k_hh_recent, v_hh_recent)
    
    def _update_hh_score(self, attn_score_cache):
        
        num_new_tokens = attn_score_cache.shape[2]

        if self.hh_score is None:
            self.hh_score = attn_score_cache.sum(0).sum(1)
        else:
            # breakpoint() # check here 
            attn_score_cache = attn_score_cache.sum(0).sum(1)   
            attn_score_cache[..., :-num_new_tokens] = attn_score_cache[..., :-num_new_tokens] + self.hh_score
            self.hh_score = attn_score_cache

    def _clean_scores(self):
        self.hh_score = None
        self.seq_len = None


class TextAVGMergeKVCache_LayerWise:
    def __init__(
        self,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        hh_ratio=None,
        recent_ratio=None
    ):
        # print(f"H2OKVCache-LayerWise: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.threshold = None
        self.image_save_ratio = None
        # print(f"H2OKVCache-LayerWise: {recent_size}, {hh_size}")
        print(f"H2OKVCache-LayerWise:{recent_ratio}, {hh_ratio}")

    def __call__(self, past_key_values, attn_score_cache):

        if self.hh_ratio is not None and attn_score_cache.shape[-2]>1:
            self.hh_size = int(attn_score_cache.shape[-1] * self.hh_ratio)
            self.recent_size = int(attn_score_cache.shape[-1] * self.recent_ratio)
            self.cache_size = self.hh_size + self.recent_size
            self.image_save_ratio = self.hh_ratio

        # print(attn_score_cache.shape)
        self._update_hh_score(attn_score_cache)
        if past_key_values is None:
            return None
        seq_len = past_key_values[0].size(self.k_seq_dim)

        if seq_len <= self.cache_size:
            return past_key_values

        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values[0].shape # [1, 32, seq_len, 128]
        #################################only-image######################################
        image_position = self.image_position.clone()
        for i in range(self.image_position.shape[0]):
            image_position[i] = image_position[i] + 575 * i
        anti_image_mask = torch.full((self.hh_score.shape[0], self.hh_score.shape[1]), 0)
        for i in range(self.image_position.shape[0]):
            anti_image_mask[:, image_position[i]:image_position[i]+576] = -65516
        anti_image_mask = anti_image_mask.to(device=self.hh_score.device, dtype=self.hh_score.dtype)
        anti_image_mask[:, -self.recent_size:] = -65516
        self.hh_score = self.hh_score + anti_image_mask
        # image_hh_size = int(self.image_save_ratio * image_position.shape[-1] * 576)
        _, keep_topk = torch.topk(self.hh_score[:, :-self.recent_size], self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values
        # mask those keeping tok
        self.hh_score.scatter_(1, keep_topk, 0)
        self.hh_score[:, -self.recent_size:] = 0
        mask = self.hh_score >= 0
        expanded_mask = mask.unsqueeze(-1).expand_as(past_key_values[0])
        k_hh_recent = torch.masked_select(past_key_values[0].squeeze(), expanded_mask).view(bsz, num_heads, -1, head_dim)
        v_hh_recent = torch.masked_select(past_key_values[1].squeeze(), expanded_mask).view(bsz, num_heads, -1, head_dim)
        ##################only-image#################################
        ###############################only-image-merge###################################
        # applying merge here
        k_hh_pruned = torch.masked_select(past_key_values[0], ~expanded_mask).view(bsz, num_heads, -1, head_dim)
        # similarity = k_hh_pruned @ k_hh_recent.transpose(-1, -2) # dot product
        similarity = (k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ ((k_hh_recent / (torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128))).transpose(-1, -2)) # cosin
        max_values, max_indices = similarity.max(dim=-1)
        # ###############################open when ori merge open##################################
        if self.threshold == None:
            self.threshold = max_values.mean()
        else:
            self.threshold = (self.threshold + max_values.mean()) / 2
        filter_indices = (max_values.mean(1)>=self.threshold).squeeze(0)
        merged_indices = max_indices[..., filter_indices].unsqueeze(-1).repeat(1, 1, 1, 128)
        merge_weights = max_values[..., filter_indices].unsqueeze(-1).repeat(1, 1, 1, 128)
        k_hh_merged = k_hh_pruned[..., filter_indices, :]
        ############################################################################################
        # avg merge
        k_hh_recent = torch.scatter_reduce(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_pruned, reduce='mean', include_self=True)
        v_hh_pruned = past_key_values[1].squeeze()[~mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = torch.scatter_reduce(input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_pruned, reduce='mean', include_self=True) 
        ####################################only-image-merge#############################
        return (k_hh_recent, v_hh_recent)

    def _update_hh_score(self, attn_score_cache):
        
        num_new_tokens = attn_score_cache.shape[2]

        if self.hh_score is None:
            self.hh_score = attn_score_cache.sum(0).sum(1)
        else:
            # breakpoint() # check here 
            attn_score_cache = attn_score_cache.sum(0).sum(1)   
            attn_score_cache[..., :-num_new_tokens] = attn_score_cache[..., :-num_new_tokens] + self.hh_score
            self.hh_score = attn_score_cache

    def _clean_scores(self):
        self.hh_score = None
        self.seq_len = None

class AVGMergeKVCache_LayerWise:
    def __init__(
        self,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        hh_ratio=None,
        recent_ratio=None
    ):
        # print(f"H2OKVCache-LayerWise: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.threshold = None
        self.image_save_ratio = None
        # print(f"H2OKVCache-LayerWise: {recent_size}, {hh_size}")
        print(f"H2OKVCache-LayerWise:{recent_ratio}, {hh_ratio}")

    def __call__(self, past_key_values, attn_score_cache):

        if self.hh_ratio is not None and attn_score_cache.shape[-2]>1:
            self.hh_size = int(attn_score_cache.shape[-1] * self.hh_ratio)
            self.recent_size = int(attn_score_cache.shape[-1] * self.recent_ratio)
            self.cache_size = self.hh_size + self.recent_size
            self.image_save_ratio = self.hh_ratio

        # print(attn_score_cache.shape)
        self._update_hh_score(attn_score_cache)
        if past_key_values is None:
            return None
        seq_len = past_key_values[0].size(self.k_seq_dim)

        if seq_len <= self.cache_size:
            return past_key_values

        # hh-selection
        
        bsz, num_heads, _, head_dim = past_key_values[0].shape # [1, 32, seq_len, 128]
        #################################before-code#####################################
        select_hh_scores = self.hh_score[:, :seq_len - self.recent_size] # prune from the oldest tokens
        # merge_weights = nn.functional.softmax(self.hh_score, dim=-1)
        # merge_weights = self.hh_score
        
        # breakpoint()
        select_hh_scores[:, 0:4] = torch.full((attn_score_cache.shape[1], 4), float('inf'))
        sort_indices = torch.argsort(select_hh_scores, dim=-1, descending=True)
        keep_topk = sort_indices[:, 0:self.hh_size] # [32, self.hh_size]
        # pruned_indices = sort_indices[:, self.hh_size:] # [32, seq_len-self.recent_size-self.hh_size]
        
        keep_topk = keep_topk.sort().values
        keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

        mask = torch.zeros(self.hh_score.shape, dtype=torch.bool).to(past_key_values[0].device)
        mask = mask.scatter(-1, keep_idx, 1)

        k_hh_recent = past_key_values[0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = past_key_values[1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        self.hh_score = self.hh_score[mask].view(num_heads, self.cache_size)
        ###############################before-code########################################
        ###############################before-merge#######################################
        # applying merge here
        # breakpoint()
        k_hh_pruned = past_key_values[0].squeeze()[~mask].view(bsz, num_heads, -1, head_dim)
        # similarity = k_hh_pruned @ k_hh_recent.transpose(-1, -2) # dot product
        similarity = (k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ ((k_hh_recent / (torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128))).transpose(-1, -2)) # cosin
        max_values, max_indices = similarity.max(dim=-1)
        # breakpoint()   
        ###############################open when ori merge open##################################
        if self.threshold == None:
            self.threshold = max_values.mean()
        else:
            self.threshold = (self.threshold + max_values.mean()) / 2
        filter_indices = (max_values.mean(1)>=self.threshold).squeeze(0)
        merged_indices = max_indices[..., filter_indices].unsqueeze(-1).repeat(1, 1, 1, 128)
        merge_weights = max_values[..., filter_indices].unsqueeze(-1).repeat(1, 1, 1, 128)
        k_hh_merged = k_hh_pruned[..., filter_indices, :]
        ##########################################################################################     
        # avg merge
        k_hh_recent = torch.scatter_reduce(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_pruned, reduce='mean', include_self=True)
        v_hh_pruned = past_key_values[1].squeeze()[~mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = torch.scatter_reduce(input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_pruned, reduce='mean', include_self=True) 
        ###############################before-merge#######################################
        return (k_hh_recent, v_hh_recent)
    
    def _update_hh_score(self, attn_score_cache):
        
        num_new_tokens = attn_score_cache.shape[2]

        if self.hh_score is None:
            self.hh_score = attn_score_cache.sum(0).sum(1)
        else:
            # breakpoint() # check here 
            attn_score_cache = attn_score_cache.sum(0).sum(1)   
            attn_score_cache[..., :-num_new_tokens] = attn_score_cache[..., :-num_new_tokens] + self.hh_score
            self.hh_score = attn_score_cache

    def _clean_scores(self):
        self.hh_score = None
        self.seq_len = None

class WeightedMergeKVCache_LayerWise:
    def __init__(
        self,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        hh_ratio=None,
        recent_ratio=None
    ):
        # print(f"H2OKVCache-LayerWise: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.threshold = None
        self.image_save_ratio = None
        # print(f"H2OKVCache-LayerWise: {recent_size}, {hh_size}")
        print(f"H2OKVCache-LayerWise:{recent_ratio}, {hh_ratio}")

    def __call__(self, past_key_values, attn_score_cache):

        if self.hh_ratio is not None and attn_score_cache.shape[-2]>1:
            self.hh_size = int(attn_score_cache.shape[-1] * self.hh_ratio)
            self.recent_size = int(attn_score_cache.shape[-1] * self.recent_ratio)
            self.cache_size = self.hh_size + self.recent_size
            self.image_save_ratio = self.hh_ratio

        # print(attn_score_cache.shape)
        self._update_hh_score(attn_score_cache)
        if past_key_values is None:
            return None
        seq_len = past_key_values[0].size(self.k_seq_dim)

        if seq_len <= self.cache_size:
            return past_key_values

        # hh-selection
        
        bsz, num_heads, _, head_dim = past_key_values[0].shape # [1, 32, seq_len, 128]
        #################################before-code#####################################
        select_hh_scores = self.hh_score[:, :seq_len - self.recent_size] # prune from the oldest tokens
        # merge_weights = nn.functional.softmax(self.hh_score, dim=-1)
        # merge_weights = self.hh_score
        
        # breakpoint()
        select_hh_scores[:, 0:4] = torch.full((attn_score_cache.shape[1], 4), float('inf'))
        sort_indices = torch.argsort(select_hh_scores, dim=-1, descending=True)
        keep_topk = sort_indices[:, 0:self.hh_size] # [32, self.hh_size]
        # pruned_indices = sort_indices[:, self.hh_size:] # [32, seq_len-self.recent_size-self.hh_size]
        
        keep_topk = keep_topk.sort().values
        keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

        mask = torch.zeros(self.hh_score.shape, dtype=torch.bool).to(past_key_values[0].device)
        mask = mask.scatter(-1, keep_idx, 1)

        k_hh_recent = past_key_values[0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = past_key_values[1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        self.hh_score = self.hh_score[mask].view(num_heads, self.cache_size)
        ###############################before-code########################################
        ###############################before-merge#######################################
        # applying merge here
        # breakpoint()
        k_hh_pruned = past_key_values[0].squeeze()[~mask].view(bsz, num_heads, -1, head_dim)
        similarity = (k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ ((k_hh_recent / (torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128))).transpose(-1, -2)) # cosin
        max_values, max_indices = similarity.max(dim=-1)
        ###############################open when ori merge open##################################
        if self.threshold == None:
            self.threshold = max_values.mean()
        else:
            self.threshold = (self.threshold + max_values.mean()) / 2
        filter_indices = (max_values.mean(1)>=self.threshold).squeeze(0)
        merged_indices = max_indices[..., filter_indices].unsqueeze(-1).repeat(1, 1, 1, 128)
        merge_weights = max_values[..., filter_indices].unsqueeze(-1).repeat(1, 1, 1, 128)
        k_hh_merged = k_hh_pruned[..., filter_indices, :]
        #########################################################################################
        # weighted merge
        k_hh_recent = torch.scatter_reduce(input=k_hh_recent, dim=2, index=merged_indices, src=merge_weights*k_hh_merged, reduce='mean', include_self=True)
        v_hh_pruned = past_key_values[1].squeeze()[~mask].view(bsz, num_heads, -1, head_dim)
        v_hh_merged = v_hh_pruned[..., filter_indices, :]
        v_hh_recent = torch.scatter_reduce(input=v_hh_recent, dim=2, index=merged_indices, src=merge_weights*v_hh_merged, reduce='mean', include_self=True)
        ###############################before-merge#######################################
        return (k_hh_recent, v_hh_recent)
    
    def _update_hh_score(self, attn_score_cache):
        
        num_new_tokens = attn_score_cache.shape[2]

        if self.hh_score is None:
            self.hh_score = attn_score_cache.sum(0).sum(1)
        else:
            # breakpoint() # check here 
            attn_score_cache = attn_score_cache.sum(0).sum(1)   
            attn_score_cache[..., :-num_new_tokens] = attn_score_cache[..., :-num_new_tokens] + self.hh_score
            self.hh_score = attn_score_cache

    def _clean_scores(self):
        self.hh_score = None
        self.seq_len = None

class TextPivotMerge_LayerWise:
    def __init__(
        self,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        hh_ratio=None,
        recent_ratio=None
    ):
        # print(f"H2OKVCache-LayerWise: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.threshold = None
        self.image_save_ratio = None
        print(f"H2OKVCache-LayerWise: {recent_size}, {hh_size}")
        print(f"H2OKVCache-LayerWise:{recent_ratio}, {hh_ratio}")

    def __call__(self, past_key_values, attn_score_cache):

        if self.hh_ratio is not None and attn_score_cache.shape[-2]>1:
            self.hh_size = int(attn_score_cache.shape[-1] * self.hh_ratio)
            self.recent_size = int(attn_score_cache.shape[-1] * self.recent_ratio)
            self.cache_size = self.hh_size + self.recent_size
            self.image_save_ratio = self.hh_ratio

        # print(attn_score_cache.shape)
        self._update_hh_score(attn_score_cache)
        if past_key_values is None:
            return None
        seq_len = past_key_values[0].size(self.k_seq_dim)

        if seq_len <= self.cache_size:
            return past_key_values

        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values[0].shape # [1, 32, seq_len, 128]
        #################################only-image######################################
        image_position = self.image_position.clone()
        for i in range(self.image_position.shape[0]):
            image_position[i] = image_position[i] + 575 * i
        anti_image_mask = torch.full((self.hh_score.shape[0], self.hh_score.shape[1]), 0)
        for i in range(self.image_position.shape[0]):
            anti_image_mask[:, image_position[i]:image_position[i]+576] = -65516
        anti_image_mask = anti_image_mask.to(device=self.hh_score.device, dtype=self.hh_score.dtype)
        anti_image_mask[:, -self.recent_size:] = -65516
        self.hh_score = self.hh_score + anti_image_mask
        _, keep_topk = torch.topk(self.hh_score[:, :-self.recent_size], self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values
        # mask those keeping tok
        self.hh_score.scatter_(1, keep_topk, 0)
        self.hh_score[:, -self.recent_size:] = 0
        mask = self.hh_score >= 0
        expanded_mask = mask.unsqueeze(-1).expand_as(past_key_values[0])
        k_hh_recent = torch.masked_select(past_key_values[0].squeeze(), expanded_mask).view(bsz, num_heads, -1, head_dim)
        v_hh_recent = torch.masked_select(past_key_values[1].squeeze(), expanded_mask).view(bsz, num_heads, -1, head_dim)
        ##################only-image#################################
        ###############################only-image-merge###################################
        # applying merge here
        k_hh_pruned = torch.masked_select(past_key_values[0], ~expanded_mask).view(bsz, num_heads, -1, head_dim)
        similarity = (k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ ((k_hh_recent / (torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128))).transpose(-1, -2)) # cosin
        max_values, max_indices = similarity.max(dim=-1)
        # pivot merge
        merged_indices = max_indices.unsqueeze(-1).repeat(1, 1, 1, 128)
        k_hh_selected = torch.gather(input=k_hh_recent, dim=2, index=merged_indices)
        k_hh_merged = (k_hh_pruned + k_hh_selected)/2
        k_hh_recent = torch.scatter_reduce(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_merged, reduce='mean', include_self=True) # include_self=True seems decrease the performance
        v_hh_pruned = past_key_values[1].squeeze()[~mask].view(bsz, num_heads, -1, head_dim)
        v_hh_selected = torch.gather(input=v_hh_recent, dim=2, index=merged_indices)
        v_hh_merged = (v_hh_pruned + v_hh_selected)/2
        v_hh_recent = torch.scatter_reduce(input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_merged, reduce='mean', include_self=True)
        ####################################only-image-merge#############################
        return (k_hh_recent, v_hh_recent)
    
    def _update_hh_score(self, attn_score_cache):
        
        num_new_tokens = attn_score_cache.shape[2]

        if self.hh_score is None:
            self.hh_score = attn_score_cache.sum(0).sum(1)
        else:
            # breakpoint() # check here 
            attn_score_cache = attn_score_cache.sum(0).sum(1)   
            attn_score_cache[..., :-num_new_tokens] = attn_score_cache[..., :-num_new_tokens] + self.hh_score
            self.hh_score = attn_score_cache

    def _clean_scores(self):
        self.hh_score = None
        self.seq_len = None

class TextWeightedMerge_LayerWise:
    def __init__(
        self,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        hh_ratio=None,
        recent_ratio=None
    ):
        # print(f"H2OKVCache-LayerWise: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.threshold = None
        self.image_save_ratio = None
        # print(f"H2OKVCache-LayerWise: {recent_size}, {hh_size}")
        print(f"H2OKVCache-LayerWise:{recent_ratio}, {hh_ratio}")

    def __call__(self, past_key_values, attn_score_cache):

        if self.hh_ratio is not None and attn_score_cache.shape[-2]>1:
            self.hh_size = int(attn_score_cache.shape[-1] * self.hh_ratio)
            self.recent_size = int(attn_score_cache.shape[-1] * self.recent_ratio)
            self.cache_size = self.hh_size + self.recent_size
            self.image_save_ratio = self.hh_ratio

        # print(attn_score_cache.shape)
        self._update_hh_score(attn_score_cache)
        if past_key_values is None:
            return None
        seq_len = past_key_values[0].size(self.k_seq_dim)

        if seq_len <= self.cache_size:
            return past_key_values

        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values[0].shape # [1, 32, seq_len, 128]
        #################################only-image######################################
        image_position = self.image_position.clone()
        for i in range(self.image_position.shape[0]):
            image_position[i] = image_position[i] + 575 * i
        anti_image_mask = torch.full((self.hh_score.shape[0], self.hh_score.shape[1]), 0)
        for i in range(self.image_position.shape[0]):
            anti_image_mask[:, image_position[i]:image_position[i]+576] = -65516
        anti_image_mask = anti_image_mask.to(device=self.hh_score.device, dtype=self.hh_score.dtype)
        anti_image_mask[:, -self.recent_size:] = -65516
        self.hh_score = self.hh_score + anti_image_mask
        # image_hh_size = int(self.image_save_ratio * image_position.shape[-1] * 576)
        _, keep_topk = torch.topk(self.hh_score[:, :-self.recent_size], self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values
        # mask those keeping tok
        self.hh_score.scatter_(1, keep_topk, 0)
        self.hh_score[:, -self.recent_size:] = 0
        mask = self.hh_score >= 0
        expanded_mask = mask.unsqueeze(-1).expand_as(past_key_values[0])
        k_hh_recent = torch.masked_select(past_key_values[0].squeeze(), expanded_mask).view(bsz, num_heads, -1, head_dim)
        v_hh_recent = torch.masked_select(past_key_values[1].squeeze(), expanded_mask).view(bsz, num_heads, -1, head_dim)
        ##################only-image#################################
        ###############################only-image-merge###################################
        # applying merge here
        k_hh_pruned = torch.masked_select(past_key_values[0], ~expanded_mask).view(bsz, num_heads, -1, head_dim)
        # similarity = k_hh_pruned @ k_hh_recent.transpose(-1, -2) # dot product
        similarity = (k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ ((k_hh_recent / (torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128))).transpose(-1, -2)) # cosin
        max_values, max_indices = similarity.max(dim=-1)
        # ###############################open when ori merge open##################################
        if self.threshold == None:
            self.threshold = max_values.mean()
        else:
            self.threshold = (self.threshold + max_values.mean()) / 2
        filter_indices = (max_values.mean(1)>=self.threshold).squeeze(0)
        merged_indices = max_indices[..., filter_indices].unsqueeze(-1).repeat(1, 1, 1, 128)
        merge_weights = max_values[..., filter_indices].unsqueeze(-1).repeat(1, 1, 1, 128)
        k_hh_merged = k_hh_pruned[..., filter_indices, :]
        ############################################################################################
        # weighted merge
        k_hh_recent = torch.scatter_reduce(input=k_hh_recent, dim=2, index=merged_indices, src=merge_weights*k_hh_merged, reduce='mean', include_self=True)
        v_hh_pruned = k_hh_pruned = torch.masked_select(past_key_values[1], ~expanded_mask).view(bsz, num_heads, -1, head_dim)
        v_hh_merged = v_hh_pruned[..., filter_indices, :]
        v_hh_recent = torch.scatter_reduce(input=v_hh_recent, dim=2, index=merged_indices, src=merge_weights*v_hh_merged, reduce='mean', include_self=True)
        ####################################only-image-merge#############################
        return (k_hh_recent, v_hh_recent)
    
    def _update_hh_score(self, attn_score_cache):
        
        num_new_tokens = attn_score_cache.shape[2]

        if self.hh_score is None:
            self.hh_score = attn_score_cache.sum(0).sum(1)
        else:
            # breakpoint() # check here 
            attn_score_cache = attn_score_cache.sum(0).sum(1)   
            attn_score_cache[..., :-num_new_tokens] = attn_score_cache[..., :-num_new_tokens] + self.hh_score
            self.hh_score = attn_score_cache

    def _clean_scores(self):
        self.hh_score = None
        self.seq_len = None

class H2OKVCache_LayerWise:
    def __init__(
        self,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        hh_ratio=None,
        recent_ratio=None
    ):
        print(f"H2OKVCache-LayerWise: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        if recent_size and hh_size:
            self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.hh_score = None
        self.seq_len = None

    def __call__(self, past_key_values, attn_score_cache):
        if self.hh_ratio is not None:
            self.hh_size = int(attn_score_cache.shape[-1] * self.hh_ratio)
            self.recent_size = int(attn_score_cache.shape[-1] * self.recent_ratio)
            self.cache_size = self.hh_size + self.recent_size
            # breakpoint() # test here

        self._update_hh_score(attn_score_cache)
        if past_key_values is None:
            return None
        seq_len = past_key_values[0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        
        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values[0].shape
        ##################only-image#################################
        # image_position = self.image_position.clone()
        # for i in range(self.image_position.shape[0]):
        #     image_position[i] = image_position[i] + 575 * i
        # anti_image_mask = torch.full((self.hh_score.shape[0], self.hh_score.shape[1]), -65516)
        # for i in range(self.image_position.shape[0]):
        #     anti_image_mask[:, image_position[i]:image_position[i]+576] = 0
        # anti_image_mask[:, -self.recent_size:] = -65516
        # anti_image_mask = anti_image_mask.to(device=self.hh_score.device, dtype=self.hh_score.dtype)
        # self.hh_score = self.hh_score + anti_image_mask
        # self.image_save_ratio = 0.1
        # self.image_hh_size = 576 * self.image_save_ratio * image_position.shape[0]
        # _, keep_topk = torch.topk(self.hh_score, int(self.image_hh_size), dim=-1)
        # keep_topk = keep_topk.sort().values
        # # mask those keeping tok
        # self.hh_score.scatter_(1, keep_topk, -65516)      
        # mask = self.hh_score < 0
        # expanded_mask = mask.unsqueeze(-1).expand_as(past_key_values[0])
        # k_hh_recent = torch.masked_select(past_key_values[0].squeeze(), expanded_mask).view(bsz, num_heads, -1, head_dim)
        # v_hh_recent = torch.masked_select(past_key_values[1].squeeze(), expanded_mask).view(bsz, num_heads, -1, head_dim)
        ##################only-image#################################
        ##################before-code################################
        select_hh_scores = self.hh_score[:, :seq_len - self.recent_size]
        _, keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values

        # keep_recent = torch.arange(seq_len - self.recent_size, seq_len).expand(keep_topk.shape[0], 1).to(keep_topk.device)
        keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

        mask = torch.zeros(self.hh_score.shape, dtype=torch.bool).to(past_key_values[0].device)
        mask = mask.scatter(-1, keep_idx, 1)
        # print('mask',mask.shape)

        k_hh_recent = past_key_values[0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = past_key_values[1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        self.hh_score= self.hh_score[mask].view(num_heads, self.cache_size)
        ##################before-code################################
        # print('attention score', attn_score_cache.shape)

        # breakpoint()
        # print('test here')

        # self._clean_scores()
        
        # add token merge code
        """
        
        tensor([-7.0264e-01, -4.3286e-01, -1.2769e-01,  5.3833e-02,  5.5762e-01,
         1.4087e-01,  3.3765e-01, -5.3162e-02,  5.0201e-02,  2.9321e-01,
         1.6748e-01,  2.3950e-01, -8.3496e-02,  2.8760e-01,  1.4282e-01,
         3.0176e-01,  4.3152e-02,  3.8099e-04, -6.3330e-01,  4.0942e-01],
       device='cuda:0', dtype=torch.float16)
        
        """
        # breakpoint()
        
        # applying merge here
        # k_hh_pruned = past_key_values[0].squeeze()[~mask].view(bsz, num_heads, -1, head_dim)
        # similarity = k_hh_pruned @ k_hh_recent.transpose(-1, -2)
        # merged_indices = (similarity.max(dim=-1)[1]).unsqueeze(-1).repeat(1, 1, 1, 128)

        # # breakpoint()
        
        # k_hh_selected = torch.gather(input=k_hh_recent, dim=2, index=merged_indices)
        # # k_hh_merged = (k_hh_pruned + k_hh_selected)/2
        # k_hh_merged = k_hh_pruned * 0.1 + k_hh_selected * 0.9
        # k_hh_recent = torch.scatter(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_merged)
        
        # v_hh_pruned = past_key_values[1].squeeze()[~mask].view(bsz, num_heads, -1, head_dim)
        # v_hh_selected = torch.gather(input=v_hh_recent, dim=2, index=merged_indices)
        # v_hh_merged = v_hh_pruned * 0.1 + v_hh_selected * 0.9
        # v_hh_recent = torch.scatter(input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_merged)

        """"
        tensor([-7.0264e-01, -4.3286e-01, -1.2769e-01,  5.3833e-02,  5.5762e-01,
         1.4087e-01,  3.3765e-01, -5.3162e-02,  5.0201e-02,  2.9321e-01,
         1.6748e-01,  2.3950e-01, -8.3496e-02,  2.8760e-01,  1.4282e-01,
         3.0176e-01,  4.3152e-02,  3.8099e-04, -6.3330e-01,  4.0942e-01],
       device='cuda:0', dtype=torch.float16)
        
        """
        #######################only-image#############################
        # k_hh_pruned = torch.masked_select(past_key_values[0], expanded_mask).view(bsz, num_heads, -1, head_dim)
        # similarity = k_hh_pruned @ k_hh_recent.transpose(-1, -2)
        # merged_indices = (similarity.max(dim=-1)[1]).unsqueeze(-1).repeat(1, 1, 1, 128)

        # # breakpoint()
        
        # k_hh_selected = torch.gather(input=k_hh_recent, dim=2, index=merged_indices)
        # # k_hh_merged = (k_hh_pruned + k_hh_selected)/2
        # k_hh_merged = k_hh_pruned * 0.1 + k_hh_selected * 0.9
        # k_hh_recent = torch.scatter(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_merged)
        
        # v_hh_pruned = torch.masked_select(past_key_values[1], expanded_mask).view(bsz, num_heads, -1, head_dim)
        # v_hh_selected = torch.gather(input=v_hh_recent, dim=2, index=merged_indices)
        # v_hh_merged = v_hh_pruned * 0.1 + v_hh_selected * 0.9
        # v_hh_recent = torch.scatter(input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_merged)
        #######################only-image#############################

        return (k_hh_recent, v_hh_recent)

    def _update_hh_score(self, attn_score_cache):

        ############## stop here and find the bug ##############
        
        num_new_tokens = attn_score_cache.shape[2]

        if self.hh_score is None:
            
            self.hh_score = attn_score_cache.sum(0).sum(1)
        else:
            # breakpoint()
            # print('attn_score_cache', attn_score_cache.shape)
            try:
                attn_score_cache = attn_score_cache.sum(0).sum(1)   
                attn_score_cache[..., :-num_new_tokens] = attn_score_cache[..., :-num_new_tokens] + self.hh_score # 18
                self.hh_score = attn_score_cache
            except:
                breakpoint()

    def _clean_scores(self):
        self.hh_score = None

class MeanH2OKVCache_LayerWise:
    def __init__(
        self,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        hh_ratio=None,
        recent_ratio=None
    ):
        print(f"H2OKVCache-LayerWise: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        if recent_size and hh_size:
            self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.hh_score = None
        self.seq_len = None

    def __call__(self, past_key_values, attn_score_cache):
        if self.hh_ratio is not None:
            self.hh_size = int(attn_score_cache.shape[-1] * self.hh_ratio)
            self.recent_size = int(attn_score_cache.shape[-1] * self.recent_ratio)
            self.cache_size = self.hh_size + self.recent_size
            # breakpoint() # test here

        self._update_hh_score(attn_score_cache)
        if past_key_values is None:
            return None
        seq_len = past_key_values[0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        
        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values[0].shape
        ##################only-image#################################
        # image_position = self.image_position.clone()
        # for i in range(self.image_position.shape[0]):
        #     image_position[i] = image_position[i] + 575 * i
        # anti_image_mask = torch.full((self.hh_score.shape[0], self.hh_score.shape[1]), -65516)
        # for i in range(self.image_position.shape[0]):
        #     anti_image_mask[:, image_position[i]:image_position[i]+576] = 0
        # anti_image_mask[:, -self.recent_size:] = -65516
        # anti_image_mask = anti_image_mask.to(device=self.hh_score.device, dtype=self.hh_score.dtype)
        # self.hh_score = self.hh_score + anti_image_mask
        # self.image_save_ratio = 0.1
        # self.image_hh_size = 576 * self.image_save_ratio * image_position.shape[0]
        # _, keep_topk = torch.topk(self.hh_score, int(self.image_hh_size), dim=-1)
        # keep_topk = keep_topk.sort().values
        # # mask those keeping tok
        # self.hh_score.scatter_(1, keep_topk, -65516)      
        # mask = self.hh_score < 0
        # expanded_mask = mask.unsqueeze(-1).expand_as(past_key_values[0])
        # k_hh_recent = torch.masked_select(past_key_values[0].squeeze(), expanded_mask).view(bsz, num_heads, -1, head_dim)
        # v_hh_recent = torch.masked_select(past_key_values[1].squeeze(), expanded_mask).view(bsz, num_heads, -1, head_dim)
        ##################only-image#################################
        ##################before-code################################
        select_hh_scores = self.hh_score[:, :seq_len - self.recent_size]
        _, keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values

        # keep_recent = torch.arange(seq_len - self.recent_size, seq_len).expand(keep_topk.shape[0], 1).to(keep_topk.device)
        keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

        mask = torch.zeros(self.hh_score.shape, dtype=torch.bool).to(past_key_values[0].device)
        mask = mask.scatter(-1, keep_idx, 1)
        # print('mask',mask.shape)

        k_hh_recent = past_key_values[0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = past_key_values[1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        self.hh_score= self.hh_score[mask].view(num_heads, self.cache_size)
        ##################before-code################################
        # print('attention score', attn_score_cache.shape)

        # breakpoint()
        # print('test here')

        # self._clean_scores()
        
        # add token merge code
        """
        
        tensor([-7.0264e-01, -4.3286e-01, -1.2769e-01,  5.3833e-02,  5.5762e-01,
         1.4087e-01,  3.3765e-01, -5.3162e-02,  5.0201e-02,  2.9321e-01,
         1.6748e-01,  2.3950e-01, -8.3496e-02,  2.8760e-01,  1.4282e-01,
         3.0176e-01,  4.3152e-02,  3.8099e-04, -6.3330e-01,  4.0942e-01],
       device='cuda:0', dtype=torch.float16)
        
        """
        # breakpoint()
        
        # applying merge here
        # k_hh_pruned = past_key_values[0].squeeze()[~mask].view(bsz, num_heads, -1, head_dim)
        # similarity = k_hh_pruned @ k_hh_recent.transpose(-1, -2)
        # merged_indices = (similarity.max(dim=-1)[1]).unsqueeze(-1).repeat(1, 1, 1, 128)

        # # breakpoint()
        
        # k_hh_selected = torch.gather(input=k_hh_recent, dim=2, index=merged_indices)
        # # k_hh_merged = (k_hh_pruned + k_hh_selected)/2
        # k_hh_merged = k_hh_pruned * 0.1 + k_hh_selected * 0.9
        # k_hh_recent = torch.scatter(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_merged)
        
        # v_hh_pruned = past_key_values[1].squeeze()[~mask].view(bsz, num_heads, -1, head_dim)
        # v_hh_selected = torch.gather(input=v_hh_recent, dim=2, index=merged_indices)
        # v_hh_merged = v_hh_pruned * 0.1 + v_hh_selected * 0.9
        # v_hh_recent = torch.scatter(input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_merged)

        """"
        tensor([-7.0264e-01, -4.3286e-01, -1.2769e-01,  5.3833e-02,  5.5762e-01,
         1.4087e-01,  3.3765e-01, -5.3162e-02,  5.0201e-02,  2.9321e-01,
         1.6748e-01,  2.3950e-01, -8.3496e-02,  2.8760e-01,  1.4282e-01,
         3.0176e-01,  4.3152e-02,  3.8099e-04, -6.3330e-01,  4.0942e-01],
       device='cuda:0', dtype=torch.float16)
        
        """
        #######################only-image#############################
        # k_hh_pruned = torch.masked_select(past_key_values[0], expanded_mask).view(bsz, num_heads, -1, head_dim)
        # similarity = k_hh_pruned @ k_hh_recent.transpose(-1, -2)
        # merged_indices = (similarity.max(dim=-1)[1]).unsqueeze(-1).repeat(1, 1, 1, 128)

        # # breakpoint()
        
        # k_hh_selected = torch.gather(input=k_hh_recent, dim=2, index=merged_indices)
        # # k_hh_merged = (k_hh_pruned + k_hh_selected)/2
        # k_hh_merged = k_hh_pruned * 0.1 + k_hh_selected * 0.9
        # k_hh_recent = torch.scatter(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_merged)
        
        # v_hh_pruned = torch.masked_select(past_key_values[1], expanded_mask).view(bsz, num_heads, -1, head_dim)
        # v_hh_selected = torch.gather(input=v_hh_recent, dim=2, index=merged_indices)
        # v_hh_merged = v_hh_pruned * 0.1 + v_hh_selected * 0.9
        # v_hh_recent = torch.scatter(input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_merged)
        #######################only-image#############################

        return (k_hh_recent, v_hh_recent)

    def _update_hh_score(self, attn_score_cache):

        ############## stop here and find the bug ##############
        
        num_new_tokens = attn_score_cache.shape[2]

        if self.hh_score is None:
            
            self.hh_score = attn_score_cache.sum(0).mean(1)
        else:
            # breakpoint()
            # print('attn_score_cache', attn_score_cache.shape)
            try:
                attn_score_cache = attn_score_cache.sum(0).mean(1)   
                attn_score_cache[..., :-num_new_tokens] = attn_score_cache[..., :-num_new_tokens] + self.hh_score # 18
                self.hh_score = attn_score_cache
            except:
                breakpoint()

    def _clean_scores(self):
        self.hh_score = None


    def __init__(
        self,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        hh_ratio=None,
        recent_ratio=None
    ):
        print(f"H2OKVCache-LayerWise: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        if recent_size and hh_size:
            self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.hh_score = None
        self.seq_len = None

    def __call__(self, past_key_values, attn_score_cache):
        if self.hh_ratio is not None:
            self.hh_size = int(attn_score_cache.shape[-1] * self.hh_ratio)
            self.recent_size = int(attn_score_cache.shape[-1] * self.recent_ratio)
            self.cache_size = self.hh_size + self.recent_size
            # breakpoint() # test here

        self._update_hh_score(attn_score_cache)
        if past_key_values is None:
            return None
        seq_len = past_key_values[0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        
        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values[0].shape
        ##################only-image#################################
        image_position = self.image_position.clone()
        for i in range(self.image_position.shape[0]):
            image_position[i] = image_position[i] + 575 * i
        anti_image_mask = torch.full((self.hh_score.shape[0], self.hh_score.shape[1]), 0)
        for i in range(self.image_position.shape[0]):
            anti_image_mask[:, image_position[i]:image_position[i]+576] = -65516
        anti_image_mask = anti_image_mask.to(device=self.hh_score.device, dtype=self.hh_score.dtype)
        anti_image_mask[:, -self.recent_size:] = -65516
        self.hh_score = self.hh_score + anti_image_mask
        # image_hh_size = int(self.image_save_ratio * image_position.shape[-1] * 576)
        _, keep_topk = torch.topk(self.hh_score[:, :-self.recent_size], self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values
        # mask those keeping tok
        self.hh_score.scatter_(1, keep_topk, 0)
        self.hh_score[:, -self.recent_size:] = 0
        mask = self.hh_score >= 0
        expanded_mask = mask.unsqueeze(-1).expand_as(past_key_values[0])
        k_hh_recent = torch.masked_select(past_key_values[0].squeeze(), expanded_mask).view(bsz, num_heads, -1, head_dim)
        v_hh_recent = torch.masked_select(past_key_values[1].squeeze(), expanded_mask).view(bsz, num_heads, -1, head_dim)
        ##################only-image#################################
        return (k_hh_recent, v_hh_recent)

    def _update_hh_score(self, attn_score_cache):

        ############## stop here and find the bug ##############
        
        num_new_tokens = attn_score_cache.shape[2]

        if self.hh_score is None:
            
            self.hh_score = attn_score_cache.sum(0).sum(1)
        else:
            # breakpoint()
            # print('attn_score_cache', attn_score_cache.shape)
            try:
                attn_score_cache = attn_score_cache.sum(0).sum(1)   
                attn_score_cache[..., :-num_new_tokens] = attn_score_cache[..., :-num_new_tokens] + self.hh_score # 18
                self.hh_score = attn_score_cache
            except:
                breakpoint()

    def _clean_scores(self):
        self.hh_score = None

class H2OLlamaAttention_drop(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

        self.kv_cache = H2OKVCache_LayerWise(
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio
        )

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _clean_cache(self):
        self.kv_cache._clean_scores()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()
        # print('hidden_states', hidden_states.shape)
        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # remake causal mask
        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=past_key_value[0].shape[-2] if past_key_value is not None else 0,
            dtype=query_states.dtype,
            device=query_states.device,
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        position_length = kv_seq_len
        if not position_ids.nelement() > 1:
            if position_length < position_ids.item()+1:
                position_length = position_ids.item()+1

        cos, sin = self.rotary_emb(value_states, seq_len=position_length)
        ### Shift Pos: query pos is min(cache_size, idx)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        # ensure ratio prepared every rounds

        # self.kv_cache.image_position = torch.tensor([])
        if q_len != 1 and self.kv_cache.image_position.shape[0] > 0:
            past_key_value = self.kv_cache(past_key_value, attn_weights.detach().clone())
        
        # print('kv_cache shpae', past_key_value[0].shape)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

class MeanH2OLlamaAttention_drop(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

        self.kv_cache = MeanH2OKVCache_LayerWise(
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio
        )

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _clean_cache(self):
        self.kv_cache._clean_scores()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()
        # print('hidden_states', hidden_states.shape)
        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # remake causal mask
        # attention_mask = _make_causal_mask(
        #     bsz=bsz,
        #     tgt_len=q_len,
        #     past_key_values_length=past_key_value[0].shape[-2] if past_key_value is not None else 0,
        #     dtype=query_states.dtype,
        #     device=query_states.device,
        # )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        position_length = kv_seq_len
        if not position_ids.nelement() > 1:
            if position_length < position_ids.item()+1:
                position_length = position_ids.item()+1

        cos, sin = self.rotary_emb(value_states, seq_len=position_length)
        ### Shift Pos: query pos is min(cache_size, idx)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        # ensure ratio prepared every rounds

        # self.kv_cache.image_position = torch.tensor([])
        # if q_len != 1 and self.kv_cache.image_position.shape[0] > 0:
        #     past_key_value = self.kv_cache(past_key_value, attn_weights.detach().clone())
        
        # print('kv_cache shpae', past_key_value[0].shape)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.kv_cache = MixMerKVCache_LayerWise(
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio
        )

class WeightedLlamaAttention_drop(H2OLlamaAttention_drop):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.kv_cache = WeightedMergeKVCache_LayerWise(
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio
        )

class AVGMergeLlamaAttention_drop(H2OLlamaAttention_drop):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.kv_cache = AVGMergeKVCache_LayerWise(
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio
        )

class TextAVGMergeLlamaAttention_drop(H2OLlamaAttention_drop):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.kv_cache = TextAVGMergeKVCache_LayerWise(
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio
        )

class PivotMergeLlamaAttention_drop(H2OLlamaAttention_drop):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.kv_cache = PivotKVCache_LayerWise(
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio
        )

class TextH2OLlamaAttention_drop(H2OLlamaAttention_drop):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.kv_cache = TextH2OKVCache_LayerWise(
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio
        )

class TextWeightedLlamaAttention_drop(H2OLlamaAttention_drop):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.kv_cache = TextWeightedMerge_LayerWise(
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio
        )

class TextPivotLlamaAttention_drop(H2OLlamaAttention_drop):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.kv_cache = TextPivotMerge_LayerWise(
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio
        )
        
class H2OLlamaForCausalLM_drop(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            self.model.layers[layer_idx].self_attn = H2OLlamaAttention_drop(config)



## H2O KV Cache dropping with Position rolling
class H2OLlamaAttention_streaming(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

        self.kv_cache = H2OKVCache_LayerWise(
            hh_size=self.config.hh_size,
            recent_size=self.config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=self.config.heavy_ratio,
            recent_ratio=self.config.recent_ratio
        )

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _clean_cache(self):
        self.kv_cache._clean_scores()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # remake causal mask
        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=past_key_value[0].shape[-2] if past_key_value is not None else 0,
            dtype=query_states.dtype,
            device=query_states.device,
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if not position_ids.nelement() > 1:
            position_ids[0][0] = kv_seq_len - 1

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        ### Shift Pos: query pos is min(cache_size, idx)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        ###

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        ### Shift Pos: key pos is the pos in cache (Rolling KV Cache and using relative pos emb)
        key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)
        ###

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        # core code
        past_key_value = self.kv_cache(past_key_value, attn_weights.detach().clone()) 

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class H2OLlamaForCausalLM_streaming(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            self.model.layers[layer_idx].self_attn = H2OLlamaAttention_streaming(config)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class PixelPrunMergeAttentionLeft(LlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        keep_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()
        # if q_len != 1:
            # position_ids = torch.tensor(range(q_len), dtype=torch.int64).unsqueeze(0).to(hidden_states.device)
        # print('hidden_states', hidden_states.shape)
        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # remake causal mask
        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=past_key_value[0].shape[-2] if past_key_value is not None else 0,
            dtype=query_states.dtype,
            device=query_states.device,
        )

        ### Shift Pos: query pos is min(cache_size, idx)
        position_length = position_ids[0][-1]+1
        cos, sin = self.rotary_emb(value_states, seq_len=position_length)
        if key_states.shape[-2] > 1:
            cos = cos.repeat(1, self.num_heads, 1, 1)
            sin = sin.repeat(1, self.num_heads, 1, 1)
            cos = cos.squeeze(0)[keep_mask].view(bsz, self.num_heads, -1, self.head_dim).to(query_states.device)
            sin = sin.squeeze(0)[keep_mask].view(bsz, self.num_heads, -1, self.head_dim).to(query_states.device)
            query_states = (query_states * cos) + (rotate_half(query_states) * sin)
            key_states = (key_states * cos) + (rotate_half(key_states) * sin)
        else:
            # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        attn_weights = attn_weights + attention_mask
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        
        # print('kv_cache shpae', past_key_value[0].shape)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

class PixelPrunMergeDecoderLayerLeft(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.self_attn = PixelPrunMergeAttentionLeft(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        keep_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            keep_mask=keep_mask,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
class PixelPrunMergeAttention(LlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()
        # q_len = 
        # print('hidden_states', hidden_states.shape)
        if q_len > 1:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            query_backup = query_states.clone()
            key_backup = key_states.clone()
            value_states = self.v_proj(hidden_states)
            
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            query_backup = query_backup.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_backup = key_backup.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            # remake causal mask
            attention_mask = _make_causal_mask(
                bsz=bsz,
                tgt_len=q_len,
                past_key_values_length=past_key_value[0].shape[-2] if past_key_value is not None else 0,
                dtype=query_states.dtype,
                device=query_states.device,
            )
            kv_seq_len = key_states.shape[-2]
            position_length = kv_seq_len
            cos, sin = self.rotary_emb(value_states, seq_len=position_length)
            ### Shift Pos: query pos is min(cache_size, idx)
            # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)

            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
                self.head_dim
            )
            attn_weights = attn_weights + attention_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            hh_score = attn_weights.sum(0).sum(1)
            
            keep_ratio = 1/4
            img_len = 576
            img_st_pos = 35
            img_end_pos = 610
            keep_len = int(keep_ratio * (img_len))
            _, keep_topk = torch.topk(hh_score[:,img_st_pos:img_end_pos+1], keep_len, dim=-1)
            keep_topk = keep_topk.sort().values
            keep_topk = keep_topk + torch.tensor(img_st_pos)
            keep_len = q_len - (img_len-keep_len)
            keep_pre = torch.tensor(range(img_st_pos), dtype=torch.int64)
            keep_recent = torch.tensor(range(img_end_pos+1, q_len), dtype=torch.int64)
            keep_pre = keep_pre.unsqueeze(0).repeat(32,1).to(device=keep_topk.device)
            keep_recent = keep_recent.unsqueeze(0).repeat(32,1).to(device=keep_topk.device)
            keep_index = torch.concat((keep_pre, keep_topk, keep_recent), dim=-1)
            keep_mask = torch.zeros(q_len).unsqueeze(0).repeat(self.num_heads,1).to(torch.bool)
            keep_mask = keep_mask.to(dtype=torch.bool).to(device=keep_topk.device)
            keep_mask = keep_mask.scatter(dim=-1, index=keep_index, src=torch.ones(keep_len).unsqueeze(0).repeat(self.num_heads,1).to(keep_index.device).to(torch.bool))

            # pruned
            query_states = query_states.squeeze(0)[keep_mask].view(bsz, self.num_heads, -1, self.head_dim)
            key_states = key_states.squeeze(0)[keep_mask].view(bsz, self.num_heads, -1, self.head_dim)
            value_states = value_states.squeeze(0)[keep_mask].view(bsz, self.num_heads, -1, self.head_dim)

            # re RoPE re attention
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
                self.head_dim
            )
            attention_mask = _make_causal_mask(
                bsz=bsz,
                tgt_len=keep_len,
                past_key_values_length=past_key_value[0].shape[-2] if past_key_value is not None else 0,
                dtype=query_states.dtype,
                device=query_states.device,
            )
            attn_weights = attn_weights + attention_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()

            attn_output = attn_output.reshape(bsz, keep_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            if not output_attentions:
                attn_weights = None
            
            attn_output = attn_output
            self.keep_mask = keep_mask
            past_key_value = (key_states, value_states) if use_cache else None
            return attn_output, attn_weights, past_key_value, keep_mask, keep_len
        
        # cache branch    
        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # remake causal mask
        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=past_key_value[0].shape[-2] if past_key_value is not None else 0,
            dtype=query_states.dtype,
            device=query_states.device,
        )

        # kv cache branch
        position_length = position_ids[-1][-1]+1

        cos, sin = self.rotary_emb(value_states, seq_len=position_length)
        ### Shift Pos: query pos is min(cache_size, idx)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        attn_weights = attn_weights + attention_mask
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        
        # print('kv_cache shpae', past_key_value[0].shape)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class PixelPrunMergeDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.self_attn=PixelPrunMergeAttention(config=config)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        
        if hidden_states.shape[1] != 1:
            # Self Attention
            bsz, q_len, _ = hidden_states.size()
            hidden_states, self_attn_weights, present_key_value, keep_mask, keep_len = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            residual = residual.view(bsz, q_len, self.self_attn.num_heads, self.self_attn.head_dim).transpose(1, 2)
            residual = residual.squeeze(0)[keep_mask].view(bsz, self.self_attn.num_heads, -1, self.self_attn.head_dim)
            residual = residual.transpose(1, 2).reshape(bsz, keep_len, self.hidden_size)
            self.keep_mask = keep_mask
        else:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        outputs += (self.keep_mask,)
        return outputs


class PixelPrunMergeLlamaModel(LlamaModel):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                if idx == 21:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                    self.keep_mask = layer_outputs[2]
                elif idx > 21:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        keep_mask=self.keep_mask,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class PoolingWindows_LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

        self.kv_cache = H2OKVCache_LayerWise(
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio
        )

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _clean_cache(self):
        self.kv_cache._clean_scores()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()
        # print('hidden_states', hidden_states.shape)
        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # remake causal mask
        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=past_key_value[0].shape[-2] if past_key_value is not None else 0,
            dtype=query_states.dtype,
            device=query_states.device,
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        position_length = kv_seq_len
        if not position_ids.nelement() > 1:
            if position_length < position_ids.item()+1:
                position_length = position_ids.item()+1

        cos, sin = self.rotary_emb(value_states, seq_len=position_length)
        ### Shift Pos: query pos is min(cache_size, idx)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        ######################## new add here for the pooling windows kv ####################################
        
        past_key_value = self.kv_cache(past_key_value, query_states, attn_weights.detach().clone(), attention_mask)
        # print('kv_cache shpae', past_key_value[0].shape)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

class PoolingWindowLlamaAttention_drop(PoolingWindows_LlamaAttention):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)


        if not hasattr(self.config, 'kernel_size'):

            self.config.kernel_size = 5

        if not hasattr(self.config, 'pooling'):

            self.config.pooling = 'avgpool'
        
        self.kv_cache = PoolingKVCache_LayerWise(
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio,
            ####### add for pooling_window #######################

            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling

         )

class PoolingKVCache_LayerWise:

    def __init__(
        self,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        hh_ratio=None,
        recent_ratio=None,
        ######### add for pooling window ##############
        kernel_size=None,
        pooling=None,

    ):
        # print(f"H2OKVCache-LayerWise: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.threshold = None


        ############## add for pooling window #################

        # self.window_size = window_size
        # self.max_capacity_prompt = max_capacity_prompt
        # assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling


        print(f"H2OKVCache-LayerWise: {hh_ratio}, {recent_ratio}")

    def _clean_scores(self):
        self.hh_score = None

    def __call__(self, past_key_values=None, query_states=None, attn_score_cache=None, attention_mask=None):

        if attn_score_cache.shape[-2]>1:
            self.hh_size = int(attn_score_cache.shape[-1] * self.hh_ratio)
            self.recent_size = int(attn_score_cache.shape[-1] * self.recent_ratio)
            self.cache_size = self.hh_size + self.recent_size
            
            ####### add new here ###########################
 
            key_states = past_key_values[0]
            value_states = past_key_values[1]


            assert key_states.shape[-2] == query_states.shape[-2]

            bsz, num_heads, q_len, head_dim = query_states.shape

            if q_len < self.cache_size:

                return key_states, value_states

            else:

                attn_weights = torch.matmul(query_states[..., -self.recent_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)

                mask = torch.full((self.recent_size, self.recent_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
                mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
                mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
                mask = mask.to(attn_weights.device)
                attention_mask = mask[None, None, :, :]

                attn_weights[:, :, -self.recent_size:, -self.recent_size:] += attention_mask

                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, -self.recent_size:, : -self.recent_size].sum(dim = -2)
                if self.pooling == 'avgpool':
                    attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                elif self.pooling == 'maxpool':
                    attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                else:
                    raise ValueError('Pooling method not supported')
                indices = attn_cache.topk(self.cache_size - self.recent_size, dim=-1).indices  # here is important token rate hh_size
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                k_past_compress = key_states[:, :, :-self.recent_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-self.recent_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -self.recent_size:, :]
                v_cur = value_states[:, :, -self.recent_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)

                return (key_states, value_states)
        else:
            # directly return key and values without any operation
            return past_key_values