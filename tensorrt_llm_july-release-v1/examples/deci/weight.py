import time
from pathlib import Path

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy


def extract_layer_idx(name):
    ss = name.split('.')
    for s in ss:
        if s.isdigit():
            return s
    return None

def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx])
    else:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])


# def load_from_deci(tensor_llm_deci,
#                    deci,
#                    rank=0,
#                    tensor_parallel=1,
#                    dtype="float32"):
#     tensorrt_llm.logger.info('Loading weight from deci')
#     tik = time.time()

#     model_params = dict(deci.named_parameters())

#     # 可能不需要这个
#     for l in range(deci.config.num_hidden_layers):
#         prefix = f'model.layers.{l}.self_attn.'
#         q_weight = model_params[prefix + 'q_proj.weight']
#         k_weight = model_params[prefix + 'k_proj.weight']
#         v_weight = model_params[prefix + 'v_proj.weight']

#         # Todo
#     torch_dtype = str_dtype_to_torch(dtype)
#     # 贴合权重
#     for k, v in model_params.items():
#         pass


#     tok = time.time()
#     t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
#     tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
#     return
def load_weight(hf_model, llm_model, dtype='float32'):
    hf_params = dict(hf_model.named_parameters())
    torch_dtype = str_dtype_to_torch(dtype)
    for k, v in hf_params.items():
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(torch_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(torch_dtype).detach().cpu())
        if 'model.embed_tokens.weight' in k:
            llm_model.vocab_embedding.weight.value = v
        elif 'model.norm.weight' in k:
            llm_model.ln_f.weight.value = v
        elif 'lm_head.weight' in k:
            llm_model.lm_head.weight.value = v
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(layer_idx)
            if idx >= llm_model.num_layers:
                continue
            if 'input_layernorm.weight' in k:
                llm_model.layers[idx].input_layernorm.weight.value = v
            elif 'post_attention_layernorm.weight' in k:
                dst = llm_model.layers[idx].post_attention_layernorm.weight
                dst.value = v
            elif 'self_attn.q_proj.weight' in k:
                dst = llm_model.layers[idx].self_attn.q_proj.weight
                dst.value = v
            elif 'self_attn.k_proj.weight' in k:
                dst = llm_model.layers[idx].self_attn.k_proj.weight
                dst.value = v
            elif 'self_attn.v_proj.weight' in k:
                dst = llm_model.layers[idx].self_attn.v_proj.weight
                dst.value = v
            elif 'self_attn.o_proj.weight' in k:
                dst = llm_model.layers[idx].self_attn.o_proj.weight
                dst.value = v
            elif 'mlp.up_proj.weight' in k:
                dst = llm_model.layers[idx].mlp.gate.weight
                dst.value = v
            elif 'mlp.down_proj.weight' in k:
                dst = llm_model.layers[idx].mlp.proj.weight
                dst.value = v
            elif 'mlp.gate_proj.weight' in k:
                dst = llm_model.layers[idx].mlp.fc.weight
                dst.value = v