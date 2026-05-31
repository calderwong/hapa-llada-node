# Hapa LLaDA Node Agent Guide

## Node Role

`hapa-llada-node` is Hapa's local LLaDA/MLX inference surface. It provides a browser UI, CLI, and FastAPI endpoints for local text generation while keeping model-backed execution under operator control.

## Source Of Truth

- `README.md` defines verified runtime behavior, auth, model paths, API, and cheap checks.
- `src/server.py` owns the FastAPI app and static UI routes.
- `src/engine.py` and `src/llada_mlx.py` own model loading and generation behavior.
- `src/cli.py` owns lifecycle, health, capabilities, completions, screenshots, and self-test commands.
- `scripts/` contains supporting API test clients.
- `SECURITY.md` defines publication secret checks.

## Safe Edit Boundaries

- Keep the default host local and treat `/generate` as unsafe for public exposure because it is unauthenticated for the UI.
- Do not commit `.env`, `.node_token`, downloaded model snapshots, runtime files, screenshots with private prompts, generated prompt logs, or model caches.
- Do not trigger model downloads or long-running inference during doc-only work.
- Preserve token redaction in config/status output.
- Keep hosted-provider assumptions out of this repo; it is a local inference node.

## Hapa Connectivity

- Reads prompt text, model config, local MLX/Hugging Face model files, and bearer tokens.
- Produces completion responses, screenshots, capability data, and runtime metadata.
- Related nodes: `hapa-chat-app`, `hapa_second_brain`, `hapa-lance-node`, `hapa-telemetry-node`, Hapa wiki, and Overwatch operations.
- Model files and generated outputs belong in `hapa-vault` or local model caches, not Git.

## Verification

```bash
python3 -m compileall src scripts
./hapa-llada-node config
```

Run full smoke/self-test only when Apple Silicon MLX support, model availability, RAM, and download time are acceptable.
