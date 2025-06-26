import json
import argparse
import os
from safetensors.torch import load_file
import torch

PARAM_MAPPING = {
    # embed tokens
    "model.embed_tokens.weight": "embed_tokens.weight",

    # fc layer
    "eagle_module.fc.bias": "fc.bias",
    "eagle_module.fc.weight": "fc.weight",

    # mid layer attention
    "eagle_module.layers.0.self_attn.k_proj.weight": "midlayer.self_attn.k_proj.weight",
    "eagle_module.layers.0.self_attn.o_proj.weight": "midlayer.self_attn.o_proj.weight",
    "eagle_module.layers.0.self_attn.q_proj.weight": "midlayer.self_attn.q_proj.weight",
    "eagle_module.layers.0.self_attn.v_proj.weight": "midlayer.self_attn.v_proj.weight",

    # mid layer mlp
    "eagle_module.layers.0.mlp.down_proj.weight": "midlayer.mlp.down_proj.weight",
    "eagle_module.layers.0.mlp.gate_proj.weight": "midlayer.mlp.gate_proj.weight",
    "eagle_module.layers.0.mlp.up_proj.weight": "midlayer.mlp.up_proj.weight",

    # mid layer norm
    "eagle_module.layers.0.input_layernorm.weight": "midlayer.input_layernorm.weight",
    "eagle_module.layers.0.post_attention_layernorm.weight": "midlayer.post_attention_layernorm.weight",

    # lm head
    "lm_head.weight": "lm_head.weight",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def convert_to_sglang_format(model_path, output_path):
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_file, "r") as f:
        index = json.load(f)

    # figure out the weight file and weight name in it   
    required_weight_files = dict()
    for key, value in index["weight_map"].items():
        if key in PARAM_MAPPING:
            if value not in required_weight_files:
                required_weight_files[value] = []
            required_weight_files[value].append(key)
    
    # download the weight files
    sgl_ckpt = dict()
    for weight_file, param_names in required_weight_files.items():
        weight_file_path = os.path.join(model_path, weight_file)
        print(weight_file_path)
        data = load_file(weight_file_path)
        for param_name in param_names:
            tensor = data[param_name]
            print(f"param_name: {param_name}, tensor: {tensor.shape}")
            sgl_name = PARAM_MAPPING[param_name]
            sgl_ckpt[sgl_name] = tensor

    # merge qkv
    q_weight = sgl_ckpt.pop("midlayer.self_attn.q_proj.weight")
    k_weight = sgl_ckpt.pop("midlayer.self_attn.k_proj.weight")
    v_weight = sgl_ckpt.pop("midlayer.self_attn.v_proj.weight")
    qkv_merged_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
    sgl_ckpt["midlayer.self_attn.qkv_proj.weight"] = qkv_merged_weight
    

    # # merge gate and up
    gate_weight = sgl_ckpt.pop("midlayer.mlp.gate_proj.weight")
    up_weight = sgl_ckpt.pop("midlayer.mlp.up_proj.weight")
    mlp_merged_weight = torch.cat([gate_weight, up_weight], dim=0)
    sgl_ckpt["midlayer.mlp.gate_up_proj.weight"] = mlp_merged_weight

    # save the sglang format checkpoint
    output_bin_file = os.path.join(output_path, "model.bin")
    torch.save(sgl_ckpt, output_bin_file)


def main():
    args = parse_args()
    convert_to_sglang_format(args.model_path, args.output_path)


if __name__ == "__main__":
    main()
