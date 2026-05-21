# Hapa LLaDA Node Implementation Plan

## Objective
Create a standalone, high-performance node for the Mac Studio to run the LLaDA 2.0 diffusion model using MLX, exposing it via a local API.

## Philosophy
- **Sovereign:** Local-first, no external dependencies after download.
- **Deep Truth:** Validate MLX integration for Diffusion models (vs standard Autoregressive) before building the API.
- **Utility:** Expose via standard interfaces (HTTP/JSON) for easy Hapa AG consumption.

## Steps

### Phase 1: Foundation & "Deep Truth" Verification
1.  [ ] **Environment Setup:**
    *   Initialize `hapa-llada-node` structure.
    *   Create `requirements.txt` (`mlx`, `mlx-lm`, `fastapi`, `uvicorn`, `huggingface_hub`).
    *   Setup Python Virtual Environment.
2.  [ ] **Prototype & Validate:**
    *   **Task:** Verify `mlx_lm` compatibility with LLaDA's diffusion architecture.
    *   **Action:** Create `scripts/test_llada.py`.
    *   **Test:** Download `mlx-community/LLaDA2.0-mini-4bit` (smaller for testing) and attempt generation.
    *   **Validation:** Does it generate coherent text? Does it use the diffusion mechanism?
    *   *Correction Loop:* If standard `mlx_lm.generate` fails, implementation of the specific LLaDA sampler will be required (referencing `mlx-examples` or model card code).

### Phase 2: Node Application Structure
3.  [ ] **Core Engine (`src/engine.py`):**
    *   Class `LLaDANodeEngine` to handle model loading and inference.
    *   Implement "keep-alive" to avoid reloading weights.
4.  [ ] **API Layer (`src/server.py`):**
    *   FastAPI application.
    *   Endpoint: `POST /generate` (Custom params for diffusion steps).
    *   Endpoint: `POST /v1/chat/completions` (OpenAI-compatible wrapper mostly for convenience, assuming chat template support).
5.  [ ] **Configuration:**
    *   Environment variables for MODEL_PATH, PORT, QUANTIZATION.

### Phase 3: Operations & Docs
6.  [ ] **Scripts:**
    *   `install.sh`: Setup venv and install deps.
    *   `start.sh`: Run the server.
7.  [ ] **Documentation:**
    *   `README.md`: Setup guide, API reference.
    *   Update `docs/research/Running LLaDA 2.0 on Mac Studio.md` with findings.

## Execution
*Status: Ready to start Phase 1.*
