#!/bin/bash
# Hapa LLaDA Node Startup Script

# Act as sovereign node
source .venv/bin/activate

# Defaults
export MODEL_PATH=${MODEL_PATH:-"mlx-community/LLaDA2.0-mini-4bit"}
# Changed default port to 8085 to avoid ComfyUI (8188) and common FastAPI defaults (8000)
export PORT=${PORT:-8085}

echo "Starting Hapa LLaDA Node on port $PORT with model $MODEL_PATH..."
uvicorn src.server:app --host 127.0.0.1 --port $PORT
