from src.engine import LLaDAEngine, Model, ModelArgs
import mlx.core as mx

# Initialize just the model structure
engine = LLaDAEngine(model_path="mlx-community/LLaDA2.0-mini-4bit")

# We want to see what keys the MODEL expects vs what keys are on disk
# 1. Create dummy model
# We need to manually do what load_model does to init the class
import json
import os
local_dir = engine.local_dir
config_path = f"{local_dir}/config.json"
with open(config_path, "r") as f:
    config = json.load(f)

args = ModelArgs(
    dim=config["hidden_size"],
    n_layers=1, # Just 1 layer to keep it brief
    head_dim=config["head_dim"],
    hidden_dim=config["intermediate_size"],
    n_heads=config["num_attention_heads"],
    n_kv_heads=config["num_key_value_heads"],
    norm_eps=config["rms_norm_eps"],
    vocab_size=config["vocab_size"],
    rope_theta=config["rope_theta"],
    num_experts=config.get("num_experts", 0),
    num_experts_per_tok=config.get("num_experts_per_tok", 0),
    num_shared_experts=config.get("num_shared_experts", 0),
    moe_intermediate_size=config.get("moe_intermediate_size", 0),
    quantization=config.get("quantization")
)

model = Model(args)

print("--- Model Parameter Keys (First 10) ---")
from mlx.utils import tree_flatten
outputs = tree_flatten(model.parameters())
# outputs is list of (key, array) tuples
keys = [k for k, v in outputs]
for k in keys[:10]:
    print(k)

print("\n--- Expected Disk Keys (from inspection) ---")
print("model.layers.0.attention.dense.weight")
print("model.layers.0.attention.dense.scales")
