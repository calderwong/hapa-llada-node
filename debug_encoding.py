from transformers import AutoTokenizer

model_path = "models/LLaDA2.0-mini-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

prompt = "<role>SYSTEM</role>Hello<|role_end|>"
ids = tokenizer.encode(prompt, add_special_tokens=False)

print(f"Prompt: {prompt}")
print(f"IDs: {ids}")

# Check if single token
role_end_id = 156900 # From config
if role_end_id in ids:
    print("Found role_end token ID!")
else:
    print("Did NOT find role_end token ID. Tokenizer is splitting it.")
    for i in ids:
        print(f"{i} -> {tokenizer.decode([i])}")
