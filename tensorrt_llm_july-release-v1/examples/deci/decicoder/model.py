from collections import OrderedDict
import torch
import enum
import math
import numpy as np
import tensorrt as trt
from tensorrt_llm._common import default_net
from tensorrt_llm.functional import (RaggedTensor, Tensor, expand_dims, view, constant, slice,
                                     expand, concat, shape, transpose, permute, matmul, softmax,
                                     select, mul, einsum, cos, sin, expand_mask, gather_last_token_logits,
                                     assertion, sub, constant_to_tensor_, unsqueeze, split, gather, cast,
                                     identity)
from tensorrt_llm._utils import pad_vocab_size, str_dtype_to_trt, str_dtype_to_np
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.parameter import Parameter
from tensorrt_llm.layers.linear import ColumnLinear, RowLinear
from tensorrt_llm.layers import Embedding, RmsNorm, GatedMLP, AttentionMaskType

class RotaryEmbedding(Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, dtype='float32'):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.dtype = str_dtype_to_np(dtype)
        self.cos_cached, self.sin_cached = self.get_cos_sin_cache()

    def get_cos_sin_cache(self):
        inv_freq = 1.0 / (self.base ** (np.arange(0, self.dim, 2, dtype=self.dtype) / self.dim))
        t = np.arange(self.max_position_embeddings, dtype=self.dtype)
        freqs = np.einsum('i,j->ij', t, inv_freq)
        emb = np.concatenate((freqs, freqs), axis=-1)
        cos_emb = np.cos(emb)
        sin_emb = np.sin(emb)
        cos_cached = np.expand_dims(np.expand_dims(cos_emb, axis=0), axis=0)
        sin_cached = np.expand_dims(np.expand_dims(sin_emb, axis=0), axis=0)
        cos_cached = cos_cached.astype(self.dtype)
        sin_cached = sin_cached.astype(self.dtype)
        return cos_cached, sin_cached

    def forward(self, seq_start=None, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]    
        cos_cached = constant(self.cos_cached)
        sin_cached = constant(self.sin_cached)
        (s1, s2, s3, s4) = cos_cached.shape
        delta = sub(seq_len, seq_start)
        cos_value = slice(cos_cached, concat([0,0,seq_start,0]), concat([s1, s2, delta, s4]))
        sin_value = slice(sin_cached, concat([0,0,seq_start,0]), concat([s1, s2, delta, s4]))
        return (
            cos_value, sin_value
        )

def rotate_half(x, dtype):
    """Rotates half the hidden dims of the input."""
    (_, s2, _, s4) = x.shape
    s1 = shape(x, 0)
    s3 = shape(x, 2)
    x1 = slice(x, concat([0,0,0,0]), concat([s1, s2, s3, s4//2]))
    x2 = slice(x, concat([0,0,0,s4//2]), concat([s1, s2, s3, s4//2]))
    f1 = cast(constant_to_tensor_(-1.0), dtype)
    x = concat((mul(f1,x2), x1), x2.ndim()-1)
    return x    

def apply_rotary_pos_emb(x, cos_value, sin_value, dtype):
    x = (x * cos_value) + (rotate_half(x, dtype) * sin_value)
    return x

def repeat_kv(input_tensor, n_rep, batch, num_key_value_heads, slen, head_dim) -> Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    shape_tensor = concat([batch, num_key_value_heads, n_rep, slen, head_dim])
    output = []
    for hidden_states in input_tensor:
        hidden_states = expand_dims(hidden_states, [2])
        hidden_states = expand(hidden_states, shape_tensor)
        # hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        hidden_states = view(hidden_states, concat([batch, num_key_value_heads * n_rep, slen, head_dim]))
        output.append(hidden_states)
    return output
    # return hidden_states.view(batch, num_key_value_heads * n_rep, slen, head_dim)

class GQAttention(Module):
    def __init__(self, 
                 hidden_size, 
                 num_attention_heads,
                 num_key_value_heads,
                 max_position_embeddings, 
                 num_layers=1, 
                 attention_mask_type=AttentionMaskType.padding, 
                 bias=True, 
                 dtype=None, 
                 neox_rotary_style=False,
                 use_int8_kv_cache=False, 
                 tp_group=None, 
                 tp_size=1, 
                 multi_block_mode=False):
        super().__init__()

        self.attention_mask_type = attention_mask_type
        self.attention_head_size = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads // tp_size
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.num_attention_kv_heads = self.num_attention_heads
        self.hidden_size = hidden_size // tp_size
        self.max_position_embeddings = max_position_embeddings
        self.num_layers = num_layers
        self.neox_rotary_style = neox_rotary_style
        self.norm_factor = math.sqrt(self.attention_head_size)
        self.multi_block_mode = multi_block_mode
        self.dtype = dtype

        self.q_proj = ColumnLinear(hidden_size,
                                    hidden_size, 
                                    bias=False,
                                    dtype=dtype,
                                    gather_output=False)
        self.k_proj = ColumnLinear(hidden_size,
                                    self.num_key_value_heads * self.attention_head_size,
                                    bias=False,
                                    dtype=dtype,
                                    gather_output=False)
        self.v_proj = ColumnLinear(hidden_size,
                                    self.num_key_value_heads * self.attention_head_size,
                                    bias=False,
                                    dtype=dtype,
                                    gather_output=False)
        self.o_proj = ColumnLinear(self.num_attention_heads * self.attention_head_size, 
                                    self.hidden_size, 
                                    bias=False,
                                    dtype=dtype,
                                    gather_output=False)

        self.rotary_emb = RotaryEmbedding(self.attention_head_size, max_position_embeddings=self.max_position_embeddings, dtype=dtype)
        
    def forward(self,
                hidden_states: Tensor,
                position_ids=None,
                past_key_value=None,
                use_cache=True,
                attention_mask=None):

   
        bs = shape(hidden_states, 0)    
        q_len = shape(hidden_states, 1)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = view(query_states, concat([bs, q_len, self.num_attention_heads, self.attention_head_size])).transpose(1,2)
        key_states = view(key_states, concat([bs, q_len, self.num_key_value_heads, self.attention_head_size])).transpose(1,2)
        value_states = view(value_states, concat([bs, q_len, self.num_key_value_heads, self.attention_head_size])).transpose(1,2)

        kv_bs = shape(past_key_value, 0)
        num_kv_head = shape(past_key_value, 2)
        past_kv_len = shape(past_key_value, 3)
        kv_head_dim = shape(past_key_value, 4)
        past_key, past_value = past_key_value.split(1, 1) # (-1,2,4,-1,64) -> (-1,1,4,-1,64) * 2
        past_key = past_key.view(concat([kv_bs, num_kv_head, past_kv_len, kv_head_dim]), zero_is_placeholder=False) # (-1,1,4,-1,64) -> (-1,4,-1,64)
        past_value = past_value.view(concat([kv_bs, num_kv_head, past_kv_len, kv_head_dim]), zero_is_placeholder=False) # (-1,1,4,-1,64) -> (-1,4,-1,64)
        
        kv_len = q_len + past_kv_len
        cos_value, sin_value = self.rotary_emb(seq_start=past_kv_len, seq_len=kv_len)
        query_states = apply_rotary_pos_emb(query_states, cos_value, sin_value, dtype=self.dtype)
        key_states = apply_rotary_pos_emb(key_states, cos_value, sin_value, dtype=self.dtype)
        
        key_states = concat((past_key, key_states), dim=2)
        value_states = concat((past_value, value_states), dim=2)
        # return past_key_value with rope apply
        past_key_value = concat((unsqueeze(key_states, 1), unsqueeze(value_states, 1)), dim=1)
        past_key_value = identity(past_key_value)

        # create a mask that elements are zero
        starts = concat([0, 0, kv_len - q_len, 0])
        sizes = concat([bs, 32, q_len, kv_len])
        causal_mask = slice(attention_mask, starts, sizes)
        causal_mask = cast(causal_mask, self.dtype)

        # key_states = repeat_kv(key_states, n_rep=self.num_key_value_groups)
        # value_states = repeat_kv(value_states, n_rep=self.num_key_value_groups)
        key_states, value_states = repeat_kv((key_states, value_states), n_rep=self.num_key_value_groups, 
                                             batch=kv_bs, num_key_value_heads=num_kv_head, slen=kv_len, head_dim=kv_head_dim)

        key_states = permute(key_states, [0, 1, 3, 2])
        attention_scores = matmul(query_states, key_states)
        # attention_scores = query_states * key_states
        attention_scores = attention_scores / cast(constant_to_tensor_(self.norm_factor), self.dtype)
        attention_scores = attention_scores + causal_mask
        attention_probs = softmax(attention_scores, dim=-1)
        context = matmul(attention_probs, value_states)

        context = context.permute([0, 2, 1, 3])
        context = context.view(
                concat([bs, q_len, self.hidden_size]))

        attn_output = self.o_proj(context)

        attention_weight = None
        return attn_output, attention_weight, past_key_value


class DeciCoderDecoderLayer(Module):
    def __init__(self, layer_id, hidden_size, num_attention_heads, num_key_value_heads, 
                 max_position_embeddings, intermediate_size, dtype, hidden_act):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.self_attn = GQAttention(hidden_size=hidden_size, 
                               num_attention_heads=num_attention_heads, 
                               num_key_value_heads=num_key_value_heads,
                               max_position_embeddings=max_position_embeddings,
                               dtype=dtype) 
        self.mlp = GatedMLP(hidden_size=hidden_size,
                    ffn_hidden_size=intermediate_size,
                    hidden_act='silu',
                    dtype=dtype,
                    bias=False,
                    tp_group=None,)
        self.input_layernorm = RmsNorm(normalized_shape=hidden_size, dtype=dtype)
        self.post_attention_layernorm = RmsNorm(normalized_shape=hidden_size, dtype=dtype)


    def forward(self, hidden_states, past_key_value = None, output_attentions = False, use_cache = True, attention_mask=None):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn.forward(hidden_states=hidden_states, past_key_value=past_key_value, attention_mask=attention_mask)
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
    
class DeciModel(Module):
    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 max_position_embeddings,
                 dtype,
                 num_key_value_heads,
                 intermediate_size,
                 hidden_act='silu',
                 mlp_hidden_size=None,
                 neox_rotary_style=True,
                 tensor_parallel=1,
                 tensor_parallel_group=None,
                 multi_query_mode=False
                 ):
        super().__init__()
        self.vocab_embedding = Embedding(vocab_size, hidden_size, dtype=dtype)

        self.layers = ModuleList([
            DeciCoderDecoderLayer(layer_id=i,
                              hidden_size=hidden_size,
                              num_attention_heads=num_heads,
                              num_key_value_heads=num_key_value_heads,
                              max_position_embeddings=max_position_embeddings,
                              dtype=dtype,
                              hidden_act=hidden_act,
                              intermediate_size=intermediate_size,
                            )
            for i in range(num_layers)
        ])

        self.ln_f = RmsNorm(normalized_shape=hidden_size, dtype=dtype)

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                past_key_value=None,
                sequence_length=None,
                past_key_value_length=None,
                masked_tokens=None,
                use_cache=True,
                attention_mask=None,
                cache_indirection=None):

        hidden_states = self.vocab_embedding(input_ids)

        presents = []
        for layer, past in zip(self.layers, past_key_value):
            hidden_states = layer(hidden_states=hidden_states, past_key_value=past, attention_mask=attention_mask)

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states 

class DeciForCausalLM(DeciModel):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 max_position_embeddings,
                 dtype,
                 num_key_value_heads,
                 intermediate_size,
                 hidden_act='silu',
                 mlp_hidden_size=None,
                 neox_rotary_style=True,
                 tensor_parallel=1,
                 tensor_parallel_group=None,
                 multi_query_mode=False):
        if isinstance(dtype, str):
            self.kv_dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self.kv_dtype = dtype
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tensor_parallel = tensor_parallel
        self.max_position_embeddings = max_position_embeddings
        self.num_key_value_heads = num_key_value_heads

        # self._multi_query_mode = multi_query_mode
        super().__init__(num_layers, num_heads, hidden_size, vocab_size,
                         max_position_embeddings, dtype, num_key_value_heads, 
                         intermediate_size)
        vocab_size_padded = pad_vocab_size(vocab_size, tensor_parallel)
        self.lm_head = ColumnLinear(hidden_size,
                                    vocab_size_padded,
                                    bias=False,
                                    dtype=dtype,
                                    tp_group=tensor_parallel_group,
                                    tp_size=tensor_parallel,
                                    gather_output=True)

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                past_key_value=None,
                sequence_length=None,
                past_key_value_length=None,
                masked_tokens=None,
                use_cache=True,
                last_token_ids=None,
                attention_mask=None,
                cache_indirection=None
                ):
        bs = shape(input_ids, 0)
        seq_len = shape(input_ids, 1)
        self.max_input_length = constant_to_tensor_(self.max_position_embeddings)

        hidden_states = super().forward(input_ids, position_ids, past_key_value,
                                        sequence_length, past_key_value_length,
                                        masked_tokens, use_cache,
                                        attention_mask, cache_indirection)

        if use_cache:
            hidden_states, presents = hidden_states

        last_token_ids = seq_len * expand(constant_to_tensor_(1), unsqueeze(bs, 0))

        hidden_states = gather_last_token_logits(
            hidden_states, last_token_ids,
            default_net().plugin_config.remove_input_padding)

        # [batch_size, hidden_size] -> [batch_size, vocab_size]
        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output('logits', self.kv_dtype)

        if use_cache:
            for i, present in enumerate(presents):
                present.mark_output(f'present_key_value_{i}', self.kv_dtype)
            return (lm_logits, presents)

        return lm_logits

    def prepare_inputs(self, max_batch_size, max_input_len, max_new_tokens,
                       use_cache, max_beam_width):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        # Prepare inputs
        head_size = self.hidden_size // self.num_heads
        num_heads = self.num_heads // self.tensor_parallel
        max_position_embeddings = self.max_position_embeddings
        num_heads_kv = self.num_key_value_heads
        max_len = max_input_len + max_new_tokens
        bb_range = [
            1, (max_batch_size * max_beam_width + 1) // 2,
            max_batch_size * max_beam_width
        ]
        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        beam_width_range = [1, (max_beam_width + 1) // 2, max_beam_width]
        inlen_range = [1, 1, max_input_len]
        max_len_range = [0, (max_len + 1) // 2, max_len]
        num_tokens_range = [
            1, max_batch_size * max_beam_width,
            max(max_input_len * max_batch_size, max_beam_width * max_batch_size)
        ]

        past_key_value = []
        sequence_length = None
        past_key_value_length = None
        masked_tokens = None
        attention_mask = None
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        remove_input_padding = default_net().plugin_config.remove_input_padding

        input_ids = Tensor(name='input_ids',
                            dtype=trt.int32,
                            shape=[-1, -1],
                            dim_range=OrderedDict([
                                ('batch_size', [bb_range]),
                                ('num_tokens', [num_tokens_range]),
                            ]))
        position_ids = Tensor(name='position_ids',
                                dtype=trt.int32,
                                shape=[-1, -1],
                                dim_range=OrderedDict([
                                    ('batch_size', [bb_range]),
                                    ('num_tokens', [num_tokens_range]),
                                ]))


        for i in range(self.num_layers):
            kv_dim_range = OrderedDict([
                ('batch_size', [bb_range]),
                ('kv', [2]),
                ('num_heads', [num_heads_kv]),
                ('past_key_len', [max_len_range]),
                ('head_size', [head_size]),
            ])
            kv = Tensor(name=f'past_key_value_{i}',
                        dtype=self.kv_dtype,
                        shape=[-1, 2, num_heads_kv, -1, head_size],
                        dim_range=kv_dim_range)
            past_key_value.append(kv)

        if use_gpt_attention_plugin:
            sequence_length = Tensor(
                name='sequence_length',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size', [bb_range])]),
            )
            past_key_value_length = Tensor(
                name='past_key_value_length',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('past_key_value_length',
                                        [max_len_range])]),
            )
            masked_tokens = Tensor(name='masked_tokens',
                                   dtype=trt.int32,
                                   shape=[-1, -1],
                                   dim_range=OrderedDict([
                                       ('batch_size', [bb_range]),
                                       ('max_seq_len', [max_len_range]),
                                   ]))
        else:
            attention_mask = Tensor(name='attention_mask',
                                    dtype=trt.int32,
                                    shape=[-1, num_heads, max_position_embeddings, max_position_embeddings],
                                    dim_range=OrderedDict([
                                        ('batch_size', [bb_range]),
                                        ('mask_len', [num_heads]),
                                        ('q_len', [max_position_embeddings]),
                                        ('kv_len', [max_position_embeddings])
                                    ]))

        input_lengths = Tensor(name='input_lengths',
                               dtype=trt.int32,
                               shape=[-1],
                               dim_range=OrderedDict([('batch_size', [bb_range])
                                                      ]))

        max_input_length = Tensor(name='max_input_length',
                                  dtype=trt.int32,
                                  shape=[-1],
                                  dim_range=OrderedDict([('max_input_len',
                                                          [inlen_range])]))

        last_token_ids = Tensor(name='last_token_ids',
                                dtype=trt.int32,
                                shape=[-1],
                                dim_range=OrderedDict([
                                    ('batch_size', [bb_range]),
                                ]))

        cache_indirection = Tensor(name='cache_indirection',
                                   dtype=trt.int32,
                                   shape=[-1, -1, -1],
                                   dim_range=OrderedDict([
                                       ('batch_size', [bs_range]),
                                       ('beam_width', [beam_width_range]),
                                       ('max_seq_len', [max_len_range]),
                                   ]))

        return (input_ids, position_ids, past_key_value, sequence_length,
                past_key_value_length, masked_tokens, True, last_token_ids,
                attention_mask, cache_indirection)






                         



