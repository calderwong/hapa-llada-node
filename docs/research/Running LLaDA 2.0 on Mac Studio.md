# Running LLaDA 2.0 on Mac Studio

**Model Architecture:** LLaDA 2.0 (Large Language Diffusion with mAsking)
**Type:** Diffusion-based Large Language Model (dLLM)
**Hardware Target:** Mac Studio (Apple Silicon M1/M2/M3 Max/Ultra)
**Framework:** MLX (Apple's Machine Learning framework)

## Overview
LLaDA 2.0 marks a departure from traditional autoregressive (token-by-token) models. It uses a **diffusion process** to refine a masked sequence into a coherent response in parallel rounds. This allows for:
- **Parallel Generation:** Potential for higher throughput.
- **Global Reasoning:** Bidirectional context understanding.
- **Iterative Refinement:** "Denoising" text to reduce hallucinations.

## Requirements
- **Hardware:** Apple Silicon Mac (Metal acceleration). Mac Studio with high RAM (32GB+) is recommended for 8-bit/4-bit quantization of larger models.
- **Software:**
    - Python 3.9+
    - `mlx`
    - `mlx-lm` (MLX Language Model tools)
    - `huggingface_hub`

## Availability
The `mlx-community` on Hugging Face has converted LLaDA 2.0 models to MLX format (quantized).
- **Mini (16B):** `mlx-community/LLaDA2.0-mini-4bit`, `6bit`, `8bit`
- **Flash (100B+):** `mlx-community/LLaDA2.0-flash-4bit` (Requires significant RAM)

## Implementation Strategy
We will build a "Hapa Node" wrapper around the model.
1.  **Engine:** Use `mlx_lm` to load and run the model.
    *   *Note:* Ensure `mlx_lm` supports the diffusion sampling capabilities of LLaDA, or implement a custom sampler using `mlx.core` if the standard generation loop is autoregressive-only. (Recent updates to `mlx-examples` often include LLaDA specific scripts).
2.  **Interface:** Expose via FastAPI to integrate with the Hapa ecosystem.
3.  **Philosophy:**
    *   **Sovereign:** Runs entirely locally/offline.
    *   **No Lazy Truths:** If `mlx_lm.generate` doesn't support diffusion masking, we will write the sampling loop.

## Node Runtime Notes

### Defaults
- **Port:** `8085`
- **Temperature:** `0.6`
- **Model:** `mlx-community/LLaDA2.0-mini-4bit`

### Running the node
The repo includes a canonical CLI entrypoint:
- `./hapa-llada-node` (delegates to `src/cli.py`)

Run via startup script:
```bash
./start.sh
```

Or run via CLI:
```bash
./hapa-llada-node serve --host 0.0.0.0 --port 8085
```

### Token workflow
- The node generates a token if `HAPA_LLADA_NODE_TOKEN` is not set.
- The generated token is persisted to `.node_token`.
- CLI token lookup priority:
  1. `--token`
  2. `HAPA_LLADA_NODE_TOKEN`
  3. `.node_token`

### Prompt templating
The API expects a raw `prompt` string; the UI/CLI apply the LLaDA chat template on the client side. If your prompt already contains `<role>`, the CLI will not re-template it.
