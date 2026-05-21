import mlx.core as mx
import os

def inspect_keys():
    model_path = "models/LLaDA2.0-mini-4bit"
    print(f"Inspecting {model_path}...")
    
    file_path = f"{model_path}/model-00001-of-00002.safetensors"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    params = mx.load(file_path)
    keys = list(params.keys())
    keys.sort()
    print(f"Total keys in part 1: {len(keys)}")
    print("Example keys:")
    for k in keys[:50]:
        print(f"  {k}")

inspect_keys()
