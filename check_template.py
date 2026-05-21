from transformers import AutoTokenizer
import os

model_path = "models/LLaDA2.0-mini-4bit"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.chat_template:
        print("--- Chat Template ---")
        print(tokenizer.chat_template)
    else:
        print("No chat template found.")
        
    print("\n--- Testing Template ---")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"}
    ]
    try:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"Formatted: {repr(formatted)}")
    except Exception as e:
        print(f"Error applying template: {e}")

except Exception as e:
    print(f"Error loading tokenizer: {e}")
