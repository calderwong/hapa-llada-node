from datetime import datetime, timezone
import threading
import uuid

import anyio
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from src.engine import LLaDAEngine
from src.auth import verify_request_token
from src.node_common import repo_root, resolve_token, token_file_path, write_runtime, remove_runtime
import os
from contextlib import asynccontextmanager

# Optional Playwright Import
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

load_dotenv()

API_VERSION = "v1"
SERVICE_NAME = "hapa-llada-node"
ROOT = repo_root()
STATIC_DIR = ROOT / "static"

MODEL_PATH = os.getenv("MODEL_PATH", "mlx-community/LLaDA2.0-mini-4bit")
engine = LLaDAEngine(model_path=MODEL_PATH)

NODE_TOKEN = resolve_token(ROOT, token=os.environ.get("HAPA_LLADA_NODE_TOKEN"))
os.environ["HAPA_LLADA_NODE_TOKEN"] = NODE_TOKEN

_ENGINE_LOCK = threading.Lock()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_auth(request: Request) -> None:
    verify_request_token(request, NODE_TOKEN)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Node starting up...")
    try:
        engine.load_model()
    except Exception as e:
        print(f"Warning: Could not preload model: {e}")

    try:
        host = os.environ.get("HAPA_LLADA_NODE_HOST") or "127.0.0.1"
        port = int(os.environ.get("PORT") or 8085)
        write_runtime(root=ROOT, host=host, port=port, token_path=token_file_path(ROOT))
    except Exception as e:
        print(f"Warning: Could not write runtime file: {e}")

    yield
    print("Node shutting down...")
    remove_runtime(ROOT)

app = FastAPI(title="Hapa LLaDA Node", lifespan=lifespan)

# Mount Static Files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# CSS/JS convenience routes
@app.get("/css/{path:path}")
async def css(path: str):
    return FileResponse(str(STATIC_DIR / "css" / path))

@app.get("/js/{path:path}")
async def js(path: str):
    return FileResponse(str(STATIC_DIR / "js" / path))

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.6

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=200, ge=1)
    temperature: float = Field(default=0.6, ge=0.0)

    model: str = Field(default=MODEL_PATH)

@app.post("/generate")
async def generate(req: GenerateRequest):
    try:
        text = engine.generate_text(req.prompt, req.max_tokens, req.temperature)
        return {"text": text}
    except Exception as e:
        # Detailed error for UI
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/capabilities", dependencies=[Depends(_require_auth)])
async def capabilities():
    return {
        "api_version": API_VERSION,
        "time": utc_now_iso(),
        "service": SERVICE_NAME,
        "modalities": {
            "text": {
                "engines": ["llada_mlx"],
                "models": [MODEL_PATH],
                "features": ["completions", "temperature", "max_tokens"],
            },
            "visual": {
                "features": ["screenshot"] if PLAYWRIGHT_AVAILABLE else []
            }
        },
    }

@app.get("/v1/models", dependencies=[Depends(_require_auth)])
async def models():
    return {
        "api_version": API_VERSION,
        "time": utc_now_iso(),
        "object": "list",
        "data": [
            {
                "id": MODEL_PATH,
                "object": "model",
                "loaded": engine.model is not None,
                "load_error": engine.load_error,
            }
        ],
    }

@app.post("/v1/completions", dependencies=[Depends(_require_auth)])
async def completions(req: CompletionRequest):
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    if req.model and req.model != MODEL_PATH:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {req.model}")

    def _run() -> str:
        # We perform raw generation. Client is responsible for templating.
        print(f"Generating with prompt: {prompt[:50]}...")
        with _ENGINE_LOCK:
            return engine.generate_text_strict(prompt, max_tokens=req.max_tokens, temp=req.temperature)

    try:
        text = await anyio.to_thread.run_sync(_run)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "api_version": API_VERSION,
        "time": utc_now_iso(),
        "id": f"cmpl_{uuid.uuid4().hex}",
        "model": MODEL_PATH,
        "choices": [
            {
                "index": 0,
                "text": text,
            }
        ],
    }

@app.get("/v1/screenshot", dependencies=[Depends(_require_auth)])
async def screenshot():
    if not PLAYWRIGHT_AVAILABLE:
        raise HTTPException(status_code=501, detail="Server screenshot capability not available (playwright missing)")
    
    port = os.getenv("PORT", "8085")
    # Use 127.0.0.1 for local and pass the current runtime token to the UI.
    local_url = f"http://127.0.0.1:{port}/?token={NODE_TOKEN}"
    
    try:

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Viewport
            await page.set_viewport_size({"width": 1280, "height": 800})
            
            # Navigate
            await page.goto(local_url)
            
            # Wait for App
            # Wait for status badge to be present, or 'Online' text
            try:
                await page.wait_for_selector(".status-badge", timeout=5000)
                # Wait a bit for health check to possibly resolve
                await page.wait_for_timeout(2000) 
            except:
                pass 
                
            screenshot_bytes = await page.screenshot(type="png")
            await browser.close()
            
            return Response(content=screenshot_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Screenshot failed: {e}")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "api_version": API_VERSION,
        "time": utc_now_iso(),
        "model": MODEL_PATH,
        "model_loaded": engine.model is not None,
    }

@app.get("/")
async def read_index():
    return FileResponse(str(STATIC_DIR / "index.html"))
