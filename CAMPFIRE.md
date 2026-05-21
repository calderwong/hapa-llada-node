# CAMPFIRE — Hapa LLaDA Node

## Campfire role

Hapa LLaDA Node is the local inference campfire for LLaDA 2.0 experiments inside the Hapa ecosystem. It keeps the flame close: prompt in, local MLX diffusion model out, with no hosted inference provider required after model files are present.

## Verified facts from this repository

- Language/runtime: Python 3.9+.
- Web service: FastAPI app in `src/server.py`.
- Default port: `8085`.
- Default host: `127.0.0.1`.
- Default model: `mlx-community/LLaDA2.0-mini-4bit`.
- Main CLI: `./hapa-llada-node` delegates to `src/cli.py`.
- Static UI: `static/` served by the FastAPI app.
- Token handling: `HAPA_LLADA_NODE_TOKEN` or generated `.node_token` for authenticated endpoints.
- Runtime state: `artifacts/runtime/hapa_llada_node_runtime.json` while the managed server is active.

## Inferred ecosystem fit

This node is best classified as Local Compute / Inference. It can serve Phamiliars, Hapa UI surfaces, Overwatch checks, and future node routers that need a sovereign local text-generation backend. Treat current runtime health as a thing to verify per machine/session, not as a permanent claim.

## Run ritual

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
./hapa-llada-node serve --host 127.0.0.1 --port 8085
```

Then open:

```text
http://127.0.0.1:8085
```

## API ritual

```bash
export HAPA_LLADA_NODE_TOKEN="devtoken"
./hapa-llada-node health --base-url http://127.0.0.1:8085
./hapa-llada-node capabilities --base-url http://127.0.0.1:8085 --token "$HAPA_LLADA_NODE_TOKEN"
./hapa-llada-node complete --base-url http://127.0.0.1:8085 --token "$HAPA_LLADA_NODE_TOKEN" --prompt "Hello"
```

## Safety constraints

- Do not commit `.node_token`, `.env`, downloaded `models/`, `.venv/`, or runtime artifacts.
- Do not expose `0.0.0.0` to untrusted networks.
- Treat `/generate` as local/UI-only unless an auth layer is added.
- Preserve upstream model/dependency license terms separately from this repository's project-level MIT license.

## Wiki anchor

- `[[Nodes/Existing/hapa-llada-node|hapa-llada-node]]`

## Bananas attribution

Contributors may opt into Bananas work-contribution tracking. Bananas is optional attribution/accounting for contribution recognition; it does not change the MIT project license or third-party license obligations.
