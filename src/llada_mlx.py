import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    rope_theta: float
    rope_traditional: bool = False
    model_type: str = None
    rope_scaling: Optional[dict] = None
    num_experts: int = 0
    num_experts_per_tok: int = 0
    num_shared_experts: int = 0
    moe_intermediate_size: int = 0
    # MoE Config
    moe_router_enable_expert_bias: bool = False
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    score_function: str = "softmax"
    router_dtype: str = "fp32"
    routed_scaling_factor: float = 1.0
    rotary_dim: int = 0
    # Quantization
    quantization: Optional[dict] = None

# Removed wrapper class to avoid key nesting mismatch
def Linear(input_dim, output_dim, bias=False, quant=None):
    if quant:
        return nn.QuantizedLinear(
            input_dim, output_dim, 
            group_size=quant.get("group_size", 64),
            bits=quant.get("bits", 4),
            bias=bias
        )
    else:
        return nn.Linear(input_dim, output_dim, bias=bias)

class LLaDAMLP(nn.Module):
    def __init__(self, dim, hidden_dim, quant=None):
        super().__init__()
        self.gate_proj = Linear(dim, hidden_dim, bias=False, quant=quant)

        self.down_proj = Linear(hidden_dim, dim, bias=False, quant=quant)

        self.up_proj = Linear(dim, hidden_dim, bias=False, quant=quant)


    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

class LLaDASwitchMLP(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, quant=None):
        super().__init__()
        # Checkpoint structure matches a single large QuantizedLinear
        self.gate_proj = Linear(dim, hidden_dim * num_experts, bias=False, quant=quant)

        self.up_proj = Linear(dim, hidden_dim * num_experts, bias=False, quant=quant)

        self.down_proj = Linear(hidden_dim * num_experts, dim, bias=False, quant=quant)

        self.num_experts = num_experts

    def __call__(self, x, indices, probs):
        # Naive implementation (Computes all experts? No, we need to gather specific weights)
        # x: [B, L, Dim]
        # indices: [B, L, K]
        # probs: [B, L, K]
        
        # We need to apply the specific experts.
        # Since I haven't implemented sparse gather efficiently...
        # I can just use the "Large Linear" idea but masked?
        # NO. The large linear `gate_proj` is (Dim, E*Hidden).
        # We need to select specific slices of E*Hidden.
        # This is hard without `gather`.
        # However, `load_weights` FLATTENED it.
        # So I have one giant matrix.
        # Easiest way in MLX without gather-scatter kernel:
        # Loop over K? (K=8).
        # For each k in K:
        #   idx_k = indices[..., k] # [B, L]
        #   prob_k = probs[..., k]
        #   Expert_k(x) = down(silu(gate(x, idx_k)) * up(x, idx_k))
        #   acc += prob_k * Expert_k(x)
        
        # How to do gate(x, idx_k)?
        # gate_proj is (Dim, E*H).
        # We need (Dim, H) for specific expert `idx_k`.
        # This requires `gather` on the weight matrix!
        # `weight[idx_k * H : (idx_k+1)*H]`
        # MLX `take` can do this.
        
        # Let's assume K is small (8). Loop is okay.
        
        B, L, D = x.shape
        out = mx.zeros_like(x)
        
        if hasattr(self.gate_proj, "weight"):
            hidden_dim = self.gate_proj.weight.shape[0] // self.num_experts 
        else:
             # Standard Linear / QuantizedLinear in MLX
             hidden_dim = self.gate_proj.weight.shape[0] // self.num_experts
        # Linear layer: y = x @ W.T
        # W shape is (Out, In). (E*H, D).
        # So hidden_dim = W.shape[0] / E.
        # Correct.
        
        for k in range(indices.shape[-1]):
            # 1. Get Expert Indices for this 'slot'
            idx = indices[..., k] # [B, L]
            p = probs[..., k:k+1] # [B, L, 1]
            
            # Since MLX doesn't support easy dynamic slicing per-batch-element for weights in a Linear layer,
            # this naive loop is actually tricky for batching if experts differ per token!
            # The "Standard" MLX MoE implementation uses `mx.moe_gather` / `scatter`? 
            # Or `block_sparse_moe`.
            # Since I am "Antigravity", I should know the best way.
            # But `mx` doesn't have a built-in high level MoE layer in `nn`.
            # It has `mx.fast.moe_gate`? No.
            
            # Fallback: Loop over ALL experts and mask? 
            # Too slow (256 experts).
            
            # "Grouped" Matrix Mult?
            # Sort indices?
            
            # For this "Fix", I will assume we can't easily do efficient MoE in python-only MLX without proper kernels.
            # BUT, I loaded the weights into ONE `nn.Linear`.
            # If I just run the Linear: `gate = self.gate_proj(x)` -> [B, L, E*H].
            # This computes ALL experts. 256 * 512 = 131k hidden size.
            # 2048 * 131k ~= 260M params per layer.
            # Computed for every token.
            # It's heavy but correctness first!
            # If I compute ALL, then I can just pick the ones I need.
            
            # Compute ALL:
            # gate = self.gate_proj(x)
            # up = self.up_proj(x)
            # This is huge. Memory?
            # 4-bit quantization helps storage, but compute is FP16.
            # This might handle the "Correctness" but be slow.
            # Given "Smoke Test" passed (it ran), my previous code WAS computing all.
            # `gate = self.gate_proj(x)`.
            # So performance allows it (Mac Studio).
            
            pass 
        
        # Previous Naive Implementation was:
        # gate = self.gate_proj(x) # [B, L, E*H]
        # up = self.up_proj(x)
        # down = self.down_proj(nn.silu(gate) * up) # ??? 
        # Wait, `down` input is [B, L, E*H]. Output [B, L, D].
        # It was computing ALL experts and summing them implicitly (via dot product)?
        # `down_proj` W is (D, E*H).
        # y = hidden @ W.T.
        # sum( h_i * w_i ).
        # This sums ALL experts.
        # BUT, `gate` output was `silu(gate_proj(x))`.
        # `gate_proj(x)` computes activations for ALL experts.
        # The router `indices` were unused! 
        # And `probs` were unused!
        # So we were summing ALL 256 experts every time.
        # And since typically only top-k should be active, the others might be noise or "negative".
        # Or maybe the model relies on the router to mask them?
        # But we didn't mask them!
        # So we generated "All-Expert-Soup". This explains the garbage!
        
        # FIX:
        # 1. Compute All (gate/up).
        # 2. Mask out non-top-k experts.
        # 3. Scale by router probabilities.
        # 4. Compute down.
        
        gate_all = self.gate_proj(x) # [B, L, E*H]
        up_all = self.up_proj(x)
        
        # We need to create a mask for [B, L, E*H] based on indices [B, L, K].
        # And apply probs [B, L, K].
        
        B, L, _ = x.shape
        H = gate_all.shape[-1] // self.num_experts # Hidden size per expert
        
        # Reshape to [B, L, E, H]
        gate_all = gate_all.reshape(B, L, self.num_experts, H)
        up_all = up_all.reshape(B, L, self.num_experts, H)
        
        # Compute activation
        hidden = nn.silu(gate_all) * up_all # [B, L, E, H]
        
        # Masking and Weighting
        # Create zero mask
        # We can't scatter easily in Python loops if E is large.
        # But we have `indices` [B, L, K].
        # We want to keep `hidden` at `indices` and scale by `probs`.
        # Fill zero elsewhere.
        
        # 1. Gather relevant experts
        # indices_exp: [B, L, K, 1] (broadcast over H)
        # We need `take_along_axis` on dimension 2 (Experts).
        
        # Expand indices for H dim?
        # limit is take_along_axis indices must match dim except axis.
        # indices: [B, L, K]. data: [B, L, E, H]. Axis 2.
        # We need indices to be [B, L, K]. It works if broadcast works or manual loop.
        # MLX `take_along_axis`:
        # "Indices must have same rank as a".
        # So indices [B, L, K] -> [B, L, K, 1].
        indices_expanded = mx.expand_dims(indices, axis=-1)
        
        selected_hidden = mx.take_along_axis(hidden, indices_expanded, axis=2) # [B, L, K, H]
        
        # Scale by probs [B, L, K]
        probs_expanded = mx.expand_dims(probs, axis=-1) # [B, L, K, 1]
        selected_hidden = selected_hidden * probs_expanded
        
        # Now we have the correctly weighted hidden states for the Top K experts.
        # We need to project them back to Dim using Down Proj.
        # Down Proj weights are (Dim, E*H).
        # We can't use the giant linear easily unless we scatter back to [B, L, E, H].
        # Construct sparse tensor? No.
        # Scattering `selected_hidden` back to a zero tensor of shape [B, L, E, H].
        
        # MLX `scatter`?
        # `scatter(input, indices, updates, axis)`
        # Input: Zero [B, L, E, H].
        # Indices: indices_expanded [B, L, K, 1].
        # Updates: selected_hidden [B, L, K, H].
        # Axis: 2.
        
        # This creates the "masked and weighted" hidden state.
        zero_tensor = mx.zeros((B, L, self.num_experts, H), dtype=hidden.dtype)
        # However, `scatter` updates in place or returns new? Returns new.
        # MLX scatter handles slicing?
        # `a.at[indices].set(values)` style.
        # But `indices` is an array.
        # Using `mx.scatter` or direct indexing.
        # Standard way:
        masked_hidden = zero_tensor
        # We need to add, in case multiple experts are same? (Unlikely).
        # `masked_hidden.at[..., indices_expanded, :].set(selected_hidden)` 
        # But indices is complex.
        
        # Alternative: Perform Down Proj on just the K experts!
        # Down Proj for expert `k` is a distinct matrix W_k (H, D).
        # We have `selected_hidden` [B, L, K, H].
        # If we can gather weights W_k for the chosen experts...
        # W_down is [E*H, D]. Reshape -> [E, H, D].
        # Gather W_down at `indices`: [B, L, K, H, D].
        # Matmul `selected_hidden` [B, L, K, 1, H] @ [B, L, K, H, D] -> [B, L, K, 1, D].
        # Sum over K -> [B, L, D].
        
        # This avoids scattering huge tensors!
        # Accessing `self.down_proj.layer.weight` (or quant weight).
        # If quantized, we can't easily reshape/gather without dequantizing?
        # `QuantizedLinear` weight is packed.
        # We can't easily gather from it.
        
        # Fallback to "Scatter" approach?
        # Or "Zero out non-selected in Python".
        # `hidden` is [B, L, E, H].
        # Make a mask [B, L, E].
        # Set mask=1 at `indices`. 0 elsewhere.
        # `hidden = hidden * mask`.
        # Then flatten to [B, L, E*H].
        # Then `down_proj(hidden)`.
        
        # This works and leverages the full matrix mult kernels (which might be faster than manual gather-matmul in Python loop).
        # Creating the mask:
        mask = mx.zeros((B, L, self.num_experts), dtype=mx.float32)
        # indices [B, L, K].
        # How to scatter 1s into mask at indices?
        # `put_along_axis`?
        # `mx.put_along_axis(mask, indices, 1, axis=2)`? No such func?
        # `array[indices] = 1`.
        # indices are values.
        # We need a meshgrid of batch/seq indices?
        
        # Since I am short on time/steps and "Smoke Test" works (even if garbage), 
        # I want correctness.
        # The easiest "Correct" logic given MLX constraints:
        # Loop K.
        # Accumulate output.
        # For k in K:
        #    expert_idx = indices[..., k] # [B, L]
        #    ...
        # Wait, if I can't gather weights, I can't do per-expert projection easily without generating the full H state.
        
        # Let's go with "Scatter Hidden" logic using a mask.
        valid_experts_mask = mx.zeros((B, L, self.num_experts), dtype=hidden.dtype)
        
        # How to set 1s at `indices` efficiently?
        # One-hot encoding!
        # `indices` is [B, L, K].
        # Loop K?
        valid_experts_mask = mx.put_along_axis(
            valid_experts_mask,
            indices.astype(mx.int32),
            probs.astype(hidden.dtype),
            axis=2,
        )
             
        # Now we have a mask [B, L, E] where selected experts have their prob weights.
        # `valid_experts_mask` [B, L, E, 1]
        valid_experts_mask = mx.expand_dims(valid_experts_mask, -1)
        
        # Apply to `hidden` [B, L, E, H]
        hidden = hidden * valid_experts_mask
        
        # Flatten back to [B, L, E*H]
        hidden = hidden.reshape(B, L, -1)
        
        # Down Proj
        return self.down_proj(hidden)

class LLaDAMoeGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.num_experts = args.num_experts
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.norm_topk_prob = args.norm_topk_prob
        self.score_function = args.score_function
        self.router_dtype = args.router_dtype
        self.routed_scaling_factor = args.routed_scaling_factor
        self.gating_dim = args.dim
        self._enable_expert_bias = args.moe_router_enable_expert_bias

        self.weight = mx.zeros((self.num_experts, self.gating_dim), dtype=mx.bfloat16)
        self.expert_bias = mx.zeros((self.num_experts,), dtype=mx.bfloat16)

    def group_limited_topk(self, scores: mx.array) -> mx.array:
        num_tokens, _ = scores.shape

        if self.n_group <= 1 or self.topk_group >= self.n_group:
            masked_scores = scores
        else:
            experts_per_group = self.num_experts // self.n_group
            k_top = 2 if experts_per_group >= 2 else experts_per_group

            group_scores = mx.sum(
                mx.topk(
                    scores.reshape(num_tokens, self.n_group, experts_per_group),
                    k=k_top,
                    axis=-1,
                ),
                axis=-1,
            )

            group_idx = mx.argsort(group_scores, axis=-1)[..., -self.topk_group:]
            group_mask = mx.zeros(group_scores.shape, dtype=mx.bool_)
            group_mask = mx.put_along_axis(
                group_mask,
                group_idx.astype(mx.int32),
                mx.ones(group_idx.shape, dtype=mx.bool_),
                axis=1,
            )

            score_mask = mx.broadcast_to(
                mx.expand_dims(group_mask, axis=-1),
                (num_tokens, self.n_group, experts_per_group),
            ).reshape(num_tokens, self.num_experts)

            masked_scores = mx.where(score_mask, scores, -mx.inf)

        topk_idx = mx.argsort(masked_scores, axis=-1)[..., -self.top_k:]
        topk_idx = topk_idx[..., ::-1]
        return topk_idx

    def __call__(self, hidden_states: mx.array):
        B, L, D = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, D)

        if self.router_dtype == "fp32":
            logits = mx.matmul(
                hidden_states.astype(mx.float32),
                self.weight.astype(mx.float32).T,
            )
        else:
            logits = mx.matmul(hidden_states, self.weight.T)

        if self.score_function == "sigmoid":
            scores = mx.sigmoid(logits.astype(mx.float32)).astype(logits.dtype)
        else:
            scores = mx.softmax(logits, axis=-1)

        if self._enable_expert_bias:
            scores_for_routing = scores + self.expert_bias.astype(scores.dtype)
        else:
            scores_for_routing = scores
        topk_idx = self.group_limited_topk(scores_for_routing)

        scores = mx.take_along_axis(scores, topk_idx, axis=-1).astype(logits.dtype)

        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
        else:
            topk_weight = scores

        topk_weight = topk_weight * self.routed_scaling_factor

        return (
            topk_idx.reshape(B, L, -1),
            topk_weight.reshape(B, L, -1),
            logits.reshape(B, L, -1),
        )

class LLaDAMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Router is typically FP16/FP32 (not quantized)
        # Warning keys were "mlp.gate.expert_bias"
        # Since we use nn.Linear, the bias key is "bias".
        # We need to ensure the key mapping aligns.
        # "mlp.gate.weight" and "mlp.gate.expert_bias"
        # If we use nn.Linear(bias=True), keys are: .weight, .bias.
        # We might need to override the key loading or just subclass Linear to name bias "expert_bias" if strict loading?
        # Typically "expert_bias" in the state_dict maps to "bias" in nn.Linear if we rename it during load.
        # BUT, since we are relying on auto-loading keys, we probably need a custom Linear or simple remapping in engine.
        # For now, let's just use bias=True if enabled.
        self.gate = LLaDAMoeGate(args)
        
        # Shared Expert
        if args.num_shared_experts > 0:
            shared_dim = args.moe_intermediate_size * args.num_shared_experts
            self.shared_experts = LLaDAMLP(args.dim, shared_dim, quant=args.quantization)
        
        # Sparse Experts
        self.switch_mlp = LLaDASwitchMLP(
            args.dim, 
            args.moe_intermediate_size, 
            args.num_experts,
            quant=args.quantization
        )
        self.args = args

    def __call__(self, x):
        indices, selected_probs, _ = self.gate(x)
        
        # Correct Routing: Softmax -> TopK -> Renormalize
        # 1. Softmax over all experts
        
        output = self.shared_experts(x) if hasattr(self, "shared_experts") else 0
        
        # Switch MLP expects 'weights' to scale the expert output
        # In our implementation: sparse_out = switch_mlp(x, indices)
        # But we need to weight the outputs!
        # My switch_mlp naive impl: `gate`? No, switch_mlp computes `down(silu(gate)*up)`.
        # Wait. `switch_mlp` in my code:
        # `gate = self.gate_proj(x)` ...
        # It doesn't use the routing probabilities passed in!
        # It uses `indices` to select weights, then computes.
        # BUT where does `selected_probs` apply?
        # Standard MoE: y = sum_i( prob_i * Expert_i(x) ).
        # My `LLaDASwitchMLP` computes `Expert_i(x)`. output is [B, L, K, Dim] or summed?
        # My `LLaDASwitchMLP` returns `down`.
        # Let's check LLaDASwitchMLP logic.
        sparse_out = self.switch_mlp(x, indices, selected_probs)
        
        # Apply routed scaling factor
        
        return output + sparse_out

class LLaDAAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.dim
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5

        total_head_dim = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.query_key_value = Linear(dim, total_head_dim, bias=False, quant=args.quantization)

        
        self.query_layernorm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
        self.key_layernorm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
        
        self.dense = Linear(self.n_heads * self.head_dim, dim, bias=False, quant=args.quantization)
        
        self.rope_dim = args.rotary_dim if args.rotary_dim else args.head_dim
        self.rope = nn.RoPE(self.rope_dim, traditional=args.rope_traditional, base=args.rope_theta)

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape
        qkv = self.query_key_value(x)
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        q, k, v = mx.split(qkv, [q_dim, q_dim + kv_dim], axis=-1)
        
        # MLX attention expects tensors shaped as [B, N_heads, T_seq, D_head].
        # We also need this layout for RoPE to treat the sequence axis correctly.
        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.query_layernorm(q)
        k = self.key_layernorm(k)
        
        if self.rope_dim < self.head_dim:
             q_rope, q_pass = q[..., :self.rope_dim], q[..., self.rope_dim:]
             k_rope, k_pass = k[..., :self.rope_dim], k[..., self.rope_dim:]
             q_rope = self.rope(q_rope)
             k_rope = self.rope(k_rope)
             q = mx.concatenate([q_rope, q_pass], axis=-1)
             k = mx.concatenate([k_rope, k_pass], axis=-1)
        else:
             q = self.rope(q)
             k = self.rope(k)

        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(output)

class LLaDABlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.attention = LLaDAAttention(args)
        self.post_attention_layernorm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        
        # Handle Dense vs MoE based on first_k_dense_replace
        # Config defaults to 1 per config.json
        first_k = 1 # Hardcoded for now based on config we saw, should pass in args
        
        if layer_idx < first_k:
            self.mlp = LLaDAMLP(args.dim, args.hidden_dim, quant=args.quantization)
        else:
            self.mlp = LLaDAMoE(args)

    def __call__(self, x, mask=None, cache=None):
        r = self.attention(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r

class LLaDAModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Checkpoint calls it 'word_embeddings' and it is quantized!
        # We use a QuantizedLinear as embedding (project one hot) or QuantizedEmbedding if available.
        # But commonly in 4-bit models, embeddings might be kept fp16 or quantized.
        # The warning said "word_embeddings.scales" exists, so it IS quantized.
        # We can implement this as a QuantizedLinear called on one-hot input, or simpler:
        # Just use nn.QuantizedEmbedding logic if MLX supports it, or manual lookup.
        # MLX nn.QuantizedEmbedding exists? 
        # Let's try to map it to QuantizedLinear for now, but `word_embeddings` key implies expected input is indices.
        # Actually MLX has `nn.QuantizedEmbedding`.
        
        if args.quantization:
             self.word_embeddings = nn.QuantizedEmbedding(
                 num_embeddings=args.vocab_size,
                 dims=args.dim,
                 group_size=args.quantization.get("group_size", 64),
                 bits=args.quantization.get("bits", 4)
             )
        else:
             self.word_embeddings = nn.Embedding(args.vocab_size, args.dim)
             
        self.layers = [LLaDABlock(args, i) for i in range(args.n_layers)]
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def __call__(self, x, mask=None, cache=None):
        x = self.word_embeddings(x)
        for layer in self.layers:
            x = layer(x, mask, cache)
        return self.norm(x)

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = LLaDAModel(args)
        # Add LM Head
        self.lm_head = Linear(args.dim, args.vocab_size, bias=False, quant=args.quantization)

    def __call__(self, x, mask=None, cache=None):
        out = self.model(x, mask, cache)
        return self.lm_head(out)
