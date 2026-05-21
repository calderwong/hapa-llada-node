from src.engine import LLaDAEngine
import sys

# Format manually
prompt = "Hello, how are you?"
formatted_prompt = f"<role>SYSTEM</role>You are a helpful assistant. You answer in English.<|role_end|><role>HUMAN</role>{prompt}<|role_end|><role>ASSISTANT</role>"

print(f"Testing with prompt: {repr(formatted_prompt)}")

engine = LLaDAEngine(model_path="mlx-community/LLaDA2.0-mini-4bit")
engine.load_model()
try:
    text = engine.generate_text(formatted_prompt, max_tokens=20, temp=0.0)
    print(f"Output: {text}")
except Exception as e:
    print(e)
