# Hapa LLaDA Node

Local-first LLaDA 2.0 inference service for the Hapa node ecosystem.

## Verified role

Hapa LLaDA Node is a Python/FastAPI node that wraps a local Apple Silicon MLX implementation of LLaDA 2.0 and exposes it through:

- a browser UI served from `static/`
- unauthenticated health and UI generation endpoints
- bearer-token authenticated node/API endpoints
- a local CLI entrypoint: `./hapa-llada-node`

This repository is intended to act as Hapa's sovereign/local inference surface for LLaDA-class diffusion language models. That ecosystem role is verified from repository code and docs, not from a live runtime guarantee in this commit.

## Inferred Hapa role

Within Hapa, this node should be treated as a Local Compute / Inference node: a capability provider that other Hapa tools, Phamiliars, Overwatch processes, or UI surfaces can call when they need local text generation without sending prompts to a hosted model provider.

Related global wiki note:

- `../Hapa_Worldbuilding_Wiki/Nodes/Existing/hapa-llada-node.md`
- Obsidian link: `[[Nodes/Existing/hapa-llada-node|hapa-llada-node]]`

## What it runs

Default model:

- `MODEL_PATH=mlx-community/LLaDA2.0-mini-4bit`

Key implementation files:

- `src/server.py` — FastAPI app, static UI, health, capabilities, completions, screenshot endpoint
- `src/engine.py` — MLX/Hugging Face snapshot loading and LLaDA diffusion generation loop
- `src/llada_mlx.py` — local MLX model architecture implementation
- `src/cli.py` — lifecycle, status, config, health, capabilities, completion, screenshot, and self-test commands
- `src/node_common.py` — runtime and token file helpers
- `src/auth.py` — bearer/query-token auth helper
- `scripts/test_api.py` — simple HTTP API test client

## Installation

Recommended local setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Apple Silicon with MLX support is expected. The model snapshot is downloaded under `models/` on first load if it is not already present.

## Configuration

Create `.env` from `.env.example` if desired:

```bash
MODEL_PATH=mlx-community/LLaDA2.0-mini-4bit
PORT=8085
HAPA_LLADA_NODE_TOKEN=devtoken
```

Environment variables:

- `MODEL_PATH` — Hugging Face repo ID or local model path. Default: `mlx-community/LLaDA2.0-mini-4bit`.
- `PORT` — server port. Default: `8085`.
- `HAPA_LLADA_NODE_HOST` — host for CLI-managed start. Default: `127.0.0.1`.
- `HAPA_LLADA_NODE_TOKEN` — bearer token for authenticated endpoints. If absent, the node generates one and stores it in `.node_token`.
- `HAPA_LLADA_NODE_BASE_URL` — CLI/client override for API calls. Default: `http://127.0.0.1:8085` or the runtime file when present.

Do not commit `.env`, `.node_token`, downloaded models, or runtime artifacts.

## Run commands

Foreground server:

```bash
./hapa-llada-node serve --host 127.0.0.1 --port 8085
```

Legacy startup script:

```bash
./start.sh
```

Background lifecycle CLI:

```bash
./hapa-llada-node start --host 127.0.0.1 --port 8085
./hapa-llada-node status
./hapa-llada-node stop
```

Show local configuration without revealing the token:

```bash
./hapa-llada-node config
```

## Ports and auth

Default local URL:

- `http://127.0.0.1:8085`

Unauthenticated endpoints:

- `GET /` — web UI
- `GET /health` — service status
- `POST /generate` — UI-oriented text generation endpoint

Authenticated endpoints require `Authorization: Bearer <token>` or `?token=<token>` where supported:

- `GET /capabilities`
- `GET /v1/models`
- `POST /v1/completions`
- `GET /v1/screenshot`

Security notes:

- Default binding is localhost. Bind to `0.0.0.0` only when intentionally exposing the node on a trusted LAN.
- Treat `.node_token` and `HAPA_LLADA_NODE_TOKEN` like passwords.
- The current `/generate` endpoint is unauthenticated because it backs the local UI; do not expose it publicly without adding an access-control layer.

## Data inputs and outputs

Inputs:

- Prompt text from the UI, CLI, or API.
- Model files downloaded from Hugging Face or supplied locally under `models/`.
- Optional runtime/configuration from `.env`, `.node_token`, and environment variables.

Outputs:

- JSON health/capability/completion responses.
- Browser-rendered UI assets from `static/`.
- Optional screenshot bytes from `/v1/screenshot`.
- Runtime metadata in `artifacts/runtime/hapa_llada_node_runtime.json` while managed server mode is active.
- Downloaded model snapshots under `models/`.

## API examples

Health:

```bash
./hapa-llada-node health --base-url http://127.0.0.1:8085
```

Capabilities:

```bash
export HAPA_LLADA_NODE_TOKEN="devtoken"
./hapa-llada-node capabilities --base-url http://127.0.0.1:8085 --token "$HAPA_LLADA_NODE_TOKEN"
```

Completion:

```bash
./hapa-llada-node complete \
  --base-url http://127.0.0.1:8085 \
  --token "$HAPA_LLADA_NODE_TOKEN" \
  --prompt "Explain diffusion language models in one sentence." \
  --max-tokens 64
```

Smoke test against a running node:

```bash
./hapa-llada-node smoke-test --base-url http://127.0.0.1:8085 --token "$HAPA_LLADA_NODE_TOKEN"
```

Self-test can optionally start the node, but it may trigger model loading/download and is therefore heavier than syntax checks:

```bash
./hapa-llada-node self-test --start --port 8085
```

## Verification notes for this sweep

Cheap static verification should pass with:

```bash
python3 -m compileall src scripts
```

A full runtime smoke test was not assumed in documentation unless it is explicitly run in the current working session, because it can require Apple Silicon MLX support, model availability, RAM, and download time.

## License and Bananas attribution

Project-level code and documentation in this repository are licensed under the MIT License by Hapa.ai / Calder Wong. See `LICENSE`.

Contributors may also opt into Bananas work-contribution tracking for attribution. Bananas is an attribution/accounting layer for recognizing who contributed work; it is optional and does not replace the MIT License. Third-party dependency and model licenses remain governed by their own upstream terms and notices.
