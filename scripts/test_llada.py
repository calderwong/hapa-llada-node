import sys
from mlx_lm import load, generate

def test_llada():
    model_path = "mlx-community/LLaDA2.0-mini-4bit"
    print(f"Loading model: {model_path}...")
    try:
        model, tokenizer = load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    prompt = "Explain quantum physics in one sentence."
    print(f"Generating for prompt: '{prompt}'")
    
    try:
        response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=100)
        print("\n--- Response ---")
        print(response)
        print("----------------")
    except Exception as e:
        print(f"Generation failed: {e}")

if __name__ == "__main__":
    test_llada()
