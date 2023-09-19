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
                                     assertion, sub, constant_to_tensor_, unsqueeze, split, gather, cast)
from tensorrt_llm._utils import pad_vocab_size, str_dtype_to_trt
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.parameter import Parameter
from tensorrt_llm.layers.linear import ColumnLinear, RowLinear
from tensorrt_llm.layers import Embedding, RmsNorm, GatedMLP

class AttentionMaskType(enum.Enum):
    padding = 0
    causal = 1
    bidirectional = 2


class PositionEmbeddingType(enum.Enum):
    learned_absolute = enum.auto()
    rope = enum.auto()
    alibi = enum.auto()


class RotaryEmbedding(Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super(RotaryEmbedding, self).__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (self.base ** (np.arange(0, self.dim, 2, dtype=np.float32) / self.dim))

        # Build here to make `torch.jit.trace` work.
        # self._set_cos_sin_cache(
        #     seq_len=max_position_embeddings
        # )
    
    def get_cos_sin_cache(self, seq_len, inv_freq):
        self.max_seq_len_cached = seq_len
        t = constant(np.arange(self.max_seq_len_cached, dtype=np.float32))
        freqs = einsum('i,j->ij', (t, inv_freq))
        # freqs = einsum(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = concat((freqs, freqs), freqs.ndim()-1)
        cos_emb = cos(emb)
        sin_emb = sin(emb)
        cos_cached = cos_emb.view(concat([1,1,shape(cos_emb)]))
        sin_cached = sin_emb.view(concat([1,1,shape(sin_emb)]))
        return cos_cached, sin_cached
        # self.register_parameter("cos_cached", cos_cached)
        # self.register_parameter("sin_cached", sin_cached)

    def forward(self, seq_start=None, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # if seq_len > self.max_seq_len_cached:
        #     self._set_cos_sin_cache(seq_len=seq_len)
        
        inv_freq = constant(self.inv_freq)
        cos_cached, sin_cached = self.get_cos_sin_cache(seq_len=self.max_position_embeddings, inv_freq=inv_freq)
        # cos_cached = self.cos_cached
        # sin_cached = self.sin_cached
        (s1, s2, s3, s4) = cos_cached.shape
        delta = sub(seq_len, seq_start)
        cos_value1 = slice(cos_cached, concat([0,0,0,0]), concat([s1, s2, seq_len, s4]))
        sin_value1 = slice(sin_cached, concat([0,0,0,0]), concat([s1, s2, seq_len, s4]))
        cos_value2 = slice(cos_cached, concat([0,0,seq_start,0]), concat([s1, s2, delta, s4]))
        sin_value2 = slice(sin_cached, concat([0,0,seq_start,0]), concat([s1, s2, delta, s4]))

        # self.register_parameter("cos_value", cos_cached)
        # self.register_parameter("cos_value", sin_value)

        return (
            cos_value1, sin_value1, cos_value2, sin_value2
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    (_, s2, _, s4) = x.shape
    s1 = shape(x, 0)
    s3 = shape(x, 2)
    x1 = slice(x, concat([0,0,0,0]), concat([s1, s2, s3, s4//2]))
    x2 = slice(x, concat([0,0,0,s4//2]), concat([s1, s2, s3, s4//2]))
    x = concat((mul(-1.0,x2), x1), x2.ndim()-1)
    return x    

def apply_rotary_pos_emb(q, k, cos_sin):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    # cos = select(select(cos, 0, 0), 0, 0)
    # sin = select(select(sin, 0, 0), 0, 0)
    # cos = cos[position_ids]
    # (seq_len, dim) = cos.shape
    # (bs, _) = position_ids.shape
    # cos = cos.view([bs, 1, seq_len, dim])
    # sin = sin.view([bs, 1, seq_len, dim])
    # rotate_q_llm = rotate_half(q)
    # rotate_q_llm.mark_output('rotate_q', dtype=tensorrt_llm.str_dtype_to_trt('float32'))
    (cos_value1, sin_value1, cos_value2, sin_value2) = cos_sin
    q_embed = (q * cos_value2) + (rotate_half(q) * sin_value2)
    k_embed = (k * cos_value1) + (rotate_half(k) * sin_value1)
    # q_embed = mul(q, cos) + mul(rotate_half(q), sin)
    # k_embed = mul(k, cos) + mul(rotate_half(k), sin)
    return q_embed, k_embed


def repeat_kv(input_tensor, n_rep, batch, num_key_value_heads, slen, head_dim) -> Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    # _, num_key_value_heads, _, head_dim = hidden_states.shape
    # batch = shape(input_tensor[0], 0)
    # num_key_value_heads = shape(input_tensor[0], 1)
    # slen = shape(input_tensor[0], 2)
    # head_dim = shape(input_tensor[0], 3)

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
        # self.head_dim = max_position_embeddings // num_attention_heads
        self.num_layers = num_layers
        self.neox_rotary_style = neox_rotary_style
        self.norm_factor = math.sqrt(self.attention_head_size)

        self.multi_block_mode = multi_block_mode
        self.rotary_embedding_dim = 0
        # self.use_int8_kv_cache = use_int8_kv_cache
        # if self.use_int8_kv_cache:
        #     self.kv_orig_quant_scale = Parameter(shape=(1, ), dtype='float32')
        #     self.kv_quant_orig_scale = Parameter(shape=(1, ), dtype='float32')
        # else:
        #     self.register_parameter('kv_orig_quant_scale', None)
        #     self.register_parameter('kv_quant_orig_scale', None)
        
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

        self.rotary_emb = RotaryEmbedding(self.attention_head_size, max_position_embeddings=self.max_position_embeddings)
        
    def dot_product_attention(self, query_states, key_states, value_states, is_causal=False):
        # masked scaled_dot_product_attention
        key_states = permute(key_states, [0,1,3,2])
        attention_scores = matmul(query_states, key_states)
        attention_scores = attention_scores / self.norm_factor 

        if is_causal:
            # key_length = key_states.size(3)
            # query_length = query_states.size(2)
            key_length = shape(key_states, 3)
            query_length = shape(query_states, 2)
            starts = concat([0, 0, sub(key_length, query_length), 0])
            sizes = concat([1, 1, query_length, key_length])
            select_buf = np.expand_dims(
                np.tril(
                    np.ones((self.max_position_embeddings,
                                self.max_position_embeddings))).astype(bool),
                (0, 1))

            select_buf = np.logical_not(select_buf)
            mask_buf = np.zeros_like(select_buf, np.float32)
            mask_buf[select_buf] = float('-inf')
            buffer = constant(mask_buf)
            causal_mask = slice(buffer, starts, sizes)
            attention_scores = attention_scores + causal_mask

        attention_probs = softmax(attention_scores, dim=-1)
        context = matmul(attention_probs, value_states)
        return context

    def forward(self,
                hidden_states: Tensor,
                position_ids=None,
                past_key_value=None,
                use_cache=True,
                attention_mask=None):

        # (bs, _, _) = hidden_states.shape    
        bs = shape(hidden_states, 0)    
        q_len = shape(hidden_states, 1)
        # if past_key_value is None:
        #     # 1
        #     is_decode = False
        # else:
        #     is_decode = True

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = view(query_states, concat([bs, q_len, self.num_attention_heads, self.attention_head_size])).transpose(1,2)
        key_states = view(key_states, concat([bs, q_len, self.num_key_value_heads, self.attention_head_size])).transpose(1,2)
        value_states = view(value_states, concat([bs, q_len, self.num_key_value_heads, self.attention_head_size])).transpose(1,2)

        # kv_seq_len = shape(key_states, 2)
        # kv_seq_len = key_states.size(key_states.ndim()-2)

        # if is_decode:
        # if past_key_value is not None:
        # if past_key_value.size(3) != 0:
        # kv_seq_len += past_key_value[0].shape[-2]
        kv_bs = shape(past_key_value, 0)
        num_kv_head = shape(past_key_value, 2)
        past_kv_len = shape(past_key_value, 3)
        kv_head_dim = shape(past_key_value, 4)            
        past_key, past_value = past_key_value.split(1, 1) # (-1,2,4,-1,64) -> (-1,1,4,-1,64) * 2
        # past_key = select(past_key, 1, 0) #shuffle_36 (-1,1,4,-1,64) -> (-1,4,-1,64)
        # past_value = select(past_value, 1, 0) #shuffle_45 (-1,1,4,-1,64) -> (-1,4,-1,64)
        past_key = past_key.view(concat([kv_bs, num_kv_head, past_kv_len, kv_head_dim]), zero_is_placeholder=False)
        past_value = past_value.view(concat([kv_bs, num_kv_head, past_kv_len, kv_head_dim]), zero_is_placeholder=False)
        key_states = concat((past_key, key_states), dim=2)
        value_states = concat((past_value, value_states), dim=2)

        # return past_key_value without rope apply
        past_key_value = concat((unsqueeze(key_states, 1), unsqueeze(value_states, 1)), dim=1)

        # past_kv_len = shape(past_key, 2)
        kv_len = q_len + past_kv_len

        # # cos sin for k apply (0 -- kv_len)
        # cos1, sin1 = self.rotary_emb(seq_start=constant_to_tensor_(0), seq_len=kv_len)
        # # cos sin for q apply (past_kv_len -- kv_len)
        # cos2, sin2 = self.rotary_emb(seq_start=past_kv_len, seq_len=kv_len)

        position_embedding_value = self.rotary_emb(seq_start=past_kv_len, seq_len=kv_len)

        # query_states = (query_states * cos2) + (rotate_half(query_states) * sin2)
        # key_states = (key_states * cos1) + (rotate_half(key_states) * sin1)

        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, position_embedding_value)

        # if is_decode:
        # if past_key_value is not None:
        # if past_key_value.size(3) != 0:
        # key_states = concat((past_key, key_states), dim=2)
        # value_states = concat((past_value, value_states), dim=2)

        # if use_cache:
        # past_key_value = concat((unsqueeze(key_states, 1), unsqueeze(value_states, 1)), dim=1)
        # else:
        #     past_key_value = None

        # create a mask that elements are zero
        starts = concat([0, 0, kv_len - q_len, 0])
        # sizes = concat([1, 32, q_len, kv_len])
        # starts = concat([0, 0, 0, 0])
        sizes = concat([bs, 32, q_len, kv_len])

        # buffer = constant(mask_buf)
        causal_mask = slice(attention_mask, starts, sizes)
        causal_mask = cast(causal_mask, 'float32')

        # key_states = repeat_kv(key_states, n_rep=8)
        # value_states = repeat_kv(value_states, n_rep=8)

        key_states, value_states = repeat_kv((key_states, value_states), n_rep=8, batch=kv_bs, num_key_value_heads=num_kv_head, slen=kv_len, head_dim=kv_head_dim)

        key_states = permute(key_states, [0, 1, 3, 2])
        attention_scores = matmul(query_states, key_states)
        # attention_scores = query_states * key_states
        attention_scores = attention_scores / self.norm_factor
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
                 max_position_embeddings, intermediate_size, dtype):
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
                            #   hidden_act=hidden_act,
                              intermediate_size=intermediate_size,
                            #   neox_rotary_style=neox_rotary_style,
                            #   multi_query_mode=multi_query_mode,
                            #   tp_group=tensor_parallel_group,
                            #   tp_size=tensor_parallel
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

        # if past_key_value[0][0] is None:
        #     # past_key_value = tuple([None] * len(self.layers))
        #     is_decode = False
        #     # temp = np.arange(input_ids.size(input_ids.ndim()-1), dtype=np.int32)
        #     # position_ids = constant(np.arange(input_ids.size(input_ids.ndim()-1), dtype=np.int32))
        # else:
        #     is_decode = True
        #     # temp = np.arange(start=past_key_value[0][0].size(past_key_value[0][0].ndim()-2), stop=past_key_value[0][0].size(past_key_value[0][0].ndim()-2)+input_ids.size(input_ids.ndim()-1), dtype=np.int32)
        # #     position_ids = constant(temp)
        # #     # position_ids = constant(np.arange(start=input_ids.size(input_ids.ndim()-1), stop=(input_ids.size(input_ids.ndim()-1)+(past_key_value[0][0].size(past_key_value[0][0].ndim()-2))), dtype=np.int32))
        # # position_ids = unsqueeze(position_ids,0)


        if use_cache:
            presents = []

        # if attention_mask is not None:
        #     attention_mask = expand_mask(attention_mask,
        #                                  input_ids.size(input_ids.ndim()-1))

        # hidden_states = RaggedTensor.from_row_lengths(hidden_states,
        #                                               input_ids.row_lengths,
        #                                               input_ids.max_row_length)

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
                cache_indirection=None,
                mask_buf=None):
        bs = shape(input_ids, 0)
        seq_len = shape(input_ids, 1)
        self.max_input_length = constant_to_tensor_(self.max_position_embeddings)
        # self.register_parameter('max_input_length', self.max_input_length)
        # self.register_parameter('input_lengths', seq_len)
        # (bs, seq_len) = input_ids.shape

        # if past_key_value[0].size(3) == 0:
        #     is_decode=False
        # else:
        #     is_decode=True

        hidden_states = super().forward(input_ids, position_ids, past_key_value,
                                        sequence_length, past_key_value_length,
                                        masked_tokens, use_cache,
                                        attention_mask, cache_indirection)

        if use_cache:
            hidden_states, presents = hidden_states

        last_token_ids = seq_len * expand(constant_to_tensor_(1), unsqueeze(bs, 0))
        # self.register_parameter('last_token_ids', last_token_ids)

        # last_token_ids = seq_len * constant(np.ones(bs, dtype='int32'))
        # last_token_ids = constant(last_token_ids)

        hidden_states = gather_last_token_logits(
            hidden_states, last_token_ids,
            default_net().plugin_config.remove_input_padding)

        # [batch_size, hidden_size] -> [batch_size, vocab_size]
        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output('logits', self.kv_dtype)

        if use_cache:
            for i, present in enumerate(presents):
                present.mark_output(f'present_key_value_{i}', self.kv_dtype)
                # self.register_parameter(f'present_key_{i}', present[0])
                # self.register_parameter(f'present_key_{i}', present[1])
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
        num_heads_kv = 4
        max_len = max_input_len + max_new_tokens
        bb_range = [
            1, (max_batch_size * max_beam_width + 1) // 2,
            max_batch_size * max_beam_width
        ]
        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        beam_width_range = [1, (max_beam_width + 1) // 2, max_beam_width]
        inlen_range = [1, 1, max_input_len]
        max_len_range = [0, (max_len + 1) // 2, max_len]
        mask_len_range = [1, (max_len + 1) // 2 + 1, max_len + 1]
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
        select_buf = np.expand_dims(
                np.tril(
                    np.ones((self.max_position_embeddings,                  #  100可换为 self.max_position_embdeddings 此处只为给出size的最大
                                self.max_position_embeddings))).astype(bool),
                (0, 1))
        select_buf = np.logical_not(select_buf)
        mask_buf = np.zeros_like(select_buf, np.float32)
        # mask_buf[select_buf] = float('-inf')

        input_ids = Tensor(name='input_ids',
                            dtype=trt.int32,
                            shape=[-1, -1],
                            dim_range=OrderedDict([
                                ('batch_size', [[1,1,1]]),
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
                                    shape=[-1, 32, 2048, 2048],
                                    dim_range=OrderedDict([
                                        ('batch_size', [bb_range]),
                                        ('mask_len', [32]),
                                        ('q_len', [2048]),
                                        ('kv_len', [2048])
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






                         



