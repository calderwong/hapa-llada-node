import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_config
from huggingface_hub import snapshot_download
import os
import json
from pathlib import Path
import numpy as np
import math
from .llada_mlx import Model, ModelArgs

class LLaDAEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.repo_root = Path(__file__).resolve().parents[1]
        self.local_dir = str(self.repo_root / "models" / model_path.split("/")[-1])
        self.load_error = None

    def load_model(self):
        if self.model is None:
            print(f"Loading model {self.model_path}...")
            
            if not os.path.exists(self.local_dir):
                print(f"Downloading snapshot to {self.local_dir}...")
                snapshot_download(repo_id=self.model_path, local_dir=self.local_dir)
            
            config_path = f"{self.local_dir}/config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
            
            args = ModelArgs(
                dim=config["hidden_size"],
                n_layers=config["num_hidden_layers"],
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
                moe_router_enable_expert_bias=config.get("moe_router_enable_expert_bias", False),
                n_group=config.get("n_group", 1),
                topk_group=config.get("topk_group", 1),
                norm_topk_prob=config.get("norm_topk_prob", True),
                score_function=config.get("score_function", "softmax"),
                router_dtype=config.get("router_dtype", "fp32"),
                routed_scaling_factor=config.get("routed_scaling_factor", 1.0),
                rotary_dim=config.get("rotary_dim", 0),
                quantization=config.get("quantization")
            )
            
            print("Initializing LLaDA Architecture in MLX...")
            self.model = Model(args)
            
            print("Loading weights...")
            weight_files = [f for f in os.listdir(self.local_dir) if f.endswith(".safetensors")]
            weights = {}
            for wf in weight_files:
                wf_path = os.path.join(self.local_dir, wf)
                w = mx.load(wf_path)
                weights.update(w)
            
            # Remap keys and Reshape 3D MoE Weights
            remapped_weights = {}
            for k, v in weights.items():
                # 1. Reshape MoE 3D weights to 2D
                if "switch_mlp" in k:
                    # v is mx.array
                    if v.ndim == 3:
                        # Case A: gate_proj / up_proj -> (Experts, OutDim, InPacked) -> (Experts*OutDim, InPacked)
                        if "gate_proj" in k or "up_proj" in k:
                            # Just flatten first two dims
                            file_shape = v.shape
                            new_shape = (file_shape[0] * file_shape[1], file_shape[2])
                            v = v.reshape(new_shape)
                            
                        # Case B: down_proj -> (Experts, OutDim, InPacked) -> (OutDim, Experts*InPacked)
                        elif "down_proj" in k:
                            # Needs transpose of Experts and OutDim first
                            # (E, D, P) -> (D, E, P) -> (D, E*P)
                            # However, 'scales' and 'biases' follow slightly different rules?
                            # Biases: (E, D) -> (D). Wait.
                            # QuantizedLinear bias is (OutDim).
                            # If down_proj bias is (Experts, Dim), and we effectively sum them?
                            # No, "Large Linear" for down_proj output is `Dim`.
                            # The bias should probably be Sum(Experts_Bias)? 
                            # Or does the Large Linear return (Dim)?
                            # Yes. LLaDASwitchMLP down_proj output is (Dim).
                            # So bias should be (Dim).
                            # If file has (Experts, Dim), we usually sum them or it is (Dim) broadcasted?
                            # Let's handle 'weight' and 'scales' first.
                            
                            if "weight" in k or "scales" in k or "biases" in k:
                                # Transpose 0 and 1
                                v = mx.transpose(v, (1, 0, 2))
                                # Flatten last two
                                s = v.shape
                                v = v.reshape(s[0], s[1] * s[2])
                    
                    # Handle Biases (2D usually for 3D weights) - Legacy check, now covered above
                    if v.ndim == 2 and "down_proj" in k and "biases" in k:
                         pass

                # Mapping Logic
                remapped_weights[k] = v
                
            weights = remapped_weights

            # Check critical weights
            if "model.lm_head.weight" in weights or "lm_head.weight" in weights:
                print("Confirmed: lm_head.weight found in checkpoint.")
            else:
                print("WARNING: lm_head.weight NOT found in checkpoint!")

            try:
                self.model.load_weights(list(weights.items()))
                print("Weights loaded successfully.")
            except Exception as e:
                self.load_error = str(e)
                print(f"Weight load warning: {e}")
            
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir)
            print("Model and Tokenizer ready.")

    def generate_text_strict(self, prompt: str, max_tokens: int = 50, temp: float = 0.6) -> str:
        out = self.generate_text(prompt, max_tokens=max_tokens, temp=temp)
        if isinstance(out, str) and (
            out.startswith("**Load Failure:**")
            or out.startswith("**System Error:**")
            or out.startswith("**Generation Error:**")
        ):
            raise RuntimeError(out)
        return out

    def generate_text(self, prompt: str, max_tokens: int = 50, temp: float = 0.6) -> str:
        if not self.model:
            try:
                self.load_model()
            except Exception as e:
                return f"**Load Failure:** {e}"

        # 1. Tokenize Prompt
        input_ids = self.tokenizer.encode(prompt)
        prompt_len = len(input_ids)
        
        # 2. Append [MASK] tokens
        # Verify mask_token_id is available
        mask_id = self.tokenizer.mask_token_id
        if mask_id is None:
             mask_id = 126081 # Fallback LLaDA default
        
        # Prepare input buffer
        full_ids = input_ids + [mask_id] * max_tokens
        
        # Convert to MLX
        # Shape: [1, seq_len]
        x = mx.array([full_ids])
        
        # 3. Diffusion Loop (Cosine Schedule)
        steps = 16 
        print(f"Starting diffusion generation: {max_tokens} tokens, {steps} steps (Cosine).")
        
        # Track masked status (local optimization)
        # 1 = kept/known, 0 = masked
        # Initially, only prompt is kept.
        # But we actually re-predict everything in the generated window every time.
        
        for step in range(steps):
            # Run Model
            logits = self.model(x) # [1, L, V]
            
            # Apply Temperature
            if temp > 0 and temp != 1.0:
                logits = logits / temp

            # Predict
            # For intermediate steps, we should use sampling if temp > 0?
            # Mask-Predict literature often sticks to Argmax but re-ranking by confidence.
            # Using sampling improves diversity.
            if temp > 0:
                sample_logits = logits.astype(mx.float32)
                pred_ids = mx.random.categorical(sample_logits, axis=-1)[0].tolist()
            else:
                pred_ids = mx.argmax(logits, axis=-1)[0].tolist()
            
            # Calculate Confidence 
            # (Use softmax probabilities of the *chosen* tokens)
            probs = mx.softmax(logits, axis=-1)
            
            # Gather probability of the selected token
            # mx.take is not super efficient for [L] indices on [L, V]
            # Simpler: just take max probability if argmax, or gather if sampled.
            # For speed, let's just use max prob as a proxy for "confidence" even if we sampled.
            # (Technically if we sampled a low-prob token, its confidence should be low).
            # Let's compute exact confidence of chosen tokens.
            
            # Convert pred_ids to array indices
            # logits shape [1, L, V]
            # We want probs[0, i, pred_ids[i]]
            # MLX doesn't have advanced indexing like numpy yet for this specific pattern easily?
            # Actually takes:
            # mx.take_along_axis(probs, pred_ids_reshaped, axis=-1)
            
            pred_indices = mx.array(pred_ids).reshape(1, -1, 1) # [1, L, 1]
            token_probs = mx.take_along_axis(probs, pred_indices, axis=-1).reshape(-1) # [L]
            conf = token_probs.tolist()

            # Schedule: How many to keep?
            # Cosine schedule
            # t goes 0 -> steps-1
            # ratio of MASKS should go 1 -> 0
            
            # t=0: ratio ~ 1.0 (all masked) -> keep ~ 0
            # t=T: ratio = 0.0 (none masked) -> keep all
            
            t = step / steps
            mask_ratio = math.cos(t * math.pi / 2)
            num_to_keep = int(max_tokens * (1 - mask_ratio))
            
            # Ensure we always keep at least one more than before? 
            # Or assume schedule handles it.
            
            # Select tokens to keep in the generated region
            gen_conf = conf[prompt_len:]
            gen_preds = pred_ids[prompt_len:]
            
            # Sort by confidence
            # argsort returns indices of sorted array (ascending)
            # We want descending (highest confidence first)
            sorted_local_indices = np.argsort(gen_conf)[::-1]
            
            # Indices (relative to generated part) to keep
            keep_local_indices = sorted_local_indices[:num_to_keep]
            
            # Reconstruct Input
            new_gen_part = [mask_id] * max_tokens
            
            for idx in keep_local_indices:
                new_gen_part[idx] = gen_preds[idx]
            
            # Update x
            final_ids = input_ids + new_gen_part
            x = mx.array([final_ids])
            
            # Last step: force fill everything
            if step == steps - 1:
                final_output_ids = input_ids + gen_preds
                return self.tokenizer.decode(final_output_ids[prompt_len:])

        return self.tokenizer.decode(gen_preds)
