from transformers import AutoTokenizer
model_path = "models/LLaDA2.0-mini-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f"Mask Token: {tokenizer.mask_token}")
print(f"Mask Token ID: {tokenizer.mask_token_id}")
print(f"Pad Token ID: {tokenizer.pad_token_id}")
print(f"BOS Token ID: {tokenizer.bos_token_id}")
print(f"EOS Token ID: {tokenizer.eos_token_id}")

# Also check special tokens map
print(f"Special Tokens: {tokenizer.all_special_tokens}")
print(f"Special Ids: {tokenizer.all_special_ids}")
