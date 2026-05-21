from src.engine import LLaDAEngine
import os

# Initialize engine
MODEL_PATH = "mlx-community/LLaDA2.0-mini-4bit"
engine = LLaDAEngine(model_path=MODEL_PATH)

print("Loading model...")
engine.load_model()

print("Generating text...")
try:
    text = engine.generate_text("Hello", max_tokens=10)
    print("Output:", text)
except Exception as e:
    import traceback
    traceback.print_exc()
