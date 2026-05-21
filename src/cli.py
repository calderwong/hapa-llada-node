import argparse
import errno
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from src.node_common import (
    read_runtime,
    read_token,
    repo_root,
    runtime_file_path,
    remove_runtime,
)

load_dotenv()
ROOT = repo_root()


def _read_text_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return text or None
    except Exception:
        return None


def _node_token_file_paths() -> list[str]:
    paths: list[str] = []
    paths.append(os.path.join(os.getcwd(), ".node_token"))
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    paths.append(os.path.join(repo_root, ".node_token"))

    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _read_node_token_file() -> Optional[str]:
    for path in _node_token_file_paths():
        tok = _read_text_file(path)
        if tok:
            return tok
    return None


def _get_token(arg_token: Optional[str]) -> Optional[str]:
    return read_token(ROOT, token=arg_token)


def _apply_llada_chat_template(prompt: str) -> str:
    if "<role>" in prompt:
        return prompt
    system_prompt = "You are a helpful assistant."
    return (
        f"<role>SYSTEM</role>{system_prompt}<|role_end|>"
        f"<role>HUMAN</role>{prompt}<|role_end|>"
        f"<role>ASSISTANT</role>"
    )


def _http_json(method: str, url: str, *, token: Optional[str], payload: Optional[dict[str, Any]]) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}

    if token:
        headers["Authorization"] = f"Bearer {token}"

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, method=method, data=data, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            body = res.read()
            return json.loads(body.decode("utf-8")) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read() if hasattr(exc, "read") else b""
        text = body.decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {text}")


def _http_bytes(method: str, url: str, *, token: Optional[str]) -> bytes:
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url=url, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            return res.read()
    except urllib.error.HTTPError as exc:
        body = exc.read() if hasattr(exc, "read") else b""
        text = body.decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {text}")


def _default_base_url() -> str:
    env = os.environ.get("HAPA_LLADA_NODE_BASE_URL")
    if env:
        return env.rstrip("/")
    
    rt = read_runtime(ROOT)
    if rt:
        return rt.base_url.rstrip("/")
    
    return "http://127.0.0.1:8085"


def _require_token(token: Optional[str]) -> str:
    if not token:
        raise RuntimeError("Missing token (set HAPA_LLADA_NODE_TOKEN, create .node_token, or pass --token)")
    return token


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError as e:
        if e.errno == errno.ESRCH:
            return False
        return True
    return True


def _health(base_url: str) -> tuple[bool, Optional[dict[str, Any]]]:
    url = base_url.rstrip("/") + "/health"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return False, None
    
    try:
        obj = json.loads(raw)
    except Exception:
        return False, None
    
    if isinstance(obj, dict) and obj.get("status") == "ok" and obj.get("service") == "hapa-llada-node":
        return True, obj
    
    if isinstance(obj, dict):
        return False, obj
    
    return False, None


def _ps_command(pid: int) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, check=False
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    cmd = proc.stdout.strip()
    return cmd or None


def _wait_dead(pid: int, timeout_s: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if not _pid_alive(pid):
            return True
        time.sleep(0.05)
    return not _pid_alive(pid)


def cmd_serve(args: argparse.Namespace) -> int:
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path
    if args.token:
        os.environ["HAPA_LLADA_NODE_TOKEN"] = args.token

    if not os.environ.get("HAPA_LLADA_NODE_TOKEN"):
        tok = _read_node_token_file()
        if tok:
            os.environ["HAPA_LLADA_NODE_TOKEN"] = tok

    host = args.host or os.environ.get("HAPA_LLADA_NODE_HOST") or "127.0.0.1"
    port = int(args.port or os.environ.get("PORT") or 8085)
    os.environ["PORT"] = str(port)

    base_url = f"http://{host}:{port}"
    print(f"[hapa-llada-node] baseUrl={base_url}")
    if os.environ.get("HAPA_LLADA_NODE_TOKEN"):
        print(f"[hapa-llada-node] token={os.environ['HAPA_LLADA_NODE_TOKEN']}")
    else:
        print("[hapa-llada-node] token=(generated on startup)")

    import uvicorn

    uvicorn.run(
        "src.server:app",
        host=host,
        port=port,
        reload=bool(args.reload),
    )
    return 0


def cmd_screenshot(args: argparse.Namespace) -> int:
    base_url = str(args.base_url).rstrip("/")
    token = _require_token(_get_token(args.token))
    img = _http_bytes("GET", base_url + "/v1/screenshot", token=token)

    out_path = str(args.output or "screenshot.png")
    with open(out_path, "wb") as f:
        f.write(img)
    print(out_path)
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    base_url = str(args.base_url).rstrip("/")
    data = _http_json("GET", base_url + "/health", token=None, payload=None)
    print(json.dumps(data, indent=2))
    return 0


def cmd_capabilities(args: argparse.Namespace) -> int:
    base_url = str(args.base_url).rstrip("/")
    token = _require_token(_get_token(args.token))
    data = _http_json("GET", base_url + "/capabilities", token=token, payload=None)
    print(json.dumps(data, indent=2))
    return 0


def cmd_complete(args: argparse.Namespace) -> int:
    base_url = str(args.base_url).rstrip("/")
    token = _require_token(_get_token(args.token))

    payload: dict[str, Any] = {
        "prompt": _apply_llada_chat_template(args.prompt),
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
    }

    if args.model:
        payload["model"] = args.model

    data = _http_json("POST", base_url + "/v1/completions", token=token, payload=payload)

    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict) and isinstance(choices[0].get("text"), str):
        print(choices[0]["text"])
    else:
        print(json.dumps(data, indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    runtime_path = runtime_file_path(ROOT)
    rt = read_runtime(ROOT)
    if not rt:
        print(_json_dumps({"status": "NOT_RUNNING", "runtime_path": str(runtime_path)}))
        return 1
    
    pid_alive = _pid_alive(rt.pid)
    health_ok, health = _health(rt.base_url)
    
    cmdline = _ps_command(rt.pid) if pid_alive else None
    cmdline_ok = bool(cmdline and "src.server:app" in cmdline)
    
    if pid_alive and health_ok:
        status = "RUNNING"
    elif pid_alive and cmdline is not None and not cmdline_ok:
        status = "STALE_PID_MISMATCH"
    else:
        status = "STALE"
    
    out: dict[str, Any] = {
        "status": status,
        "runtime_path": str(runtime_path),
        "runtime": rt.__dict__,
        "pid_alive": pid_alive,
        "health_ok": health_ok,
        "cmdline_ok": cmdline_ok,
    }
    if cmdline is not None:
        out["cmdline"] = cmdline
    if health is not None:
        out["health"] = health
    
    print(_json_dumps(out))
    return 0 if status == "RUNNING" else 1


def cmd_stop(args: argparse.Namespace) -> int:
    runtime_path = runtime_file_path(ROOT)
    rt = read_runtime(ROOT)
    if not rt:
        print(_json_dumps({"status": "NOT_RUNNING", "runtime_path": str(runtime_path)}))
        return 0
    
    pid_alive = _pid_alive(rt.pid)
    health_ok, _ = _health(rt.base_url)
    cmdline = _ps_command(rt.pid) if pid_alive else None
    cmdline_ok = bool(cmdline and "src.server:app" in cmdline)
    
    if not pid_alive:
        try:
            runtime_path.unlink(missing_ok=True)
        except Exception:
            pass
        print(_json_dumps({"status": "OK", "action": "stale_runtime_removed", "runtime": rt.__dict__}))
        return 0
    
    if cmdline is not None and not cmdline_ok:
        try:
            runtime_path.unlink(missing_ok=True)
        except Exception:
            pass
        print(_json_dumps({
            "status": "OK",
            "action": "stale_runtime_removed_pid_mismatch",
            "runtime": rt.__dict__,
            "cmdline": cmdline,
        }))
        return 0
    
    if not args.force and not health_ok and not cmdline_ok:
        out: dict[str, Any] = {
            "status": "REFUSED",
            "message": "Refusing to stop PID without verification. Use --force to stop anyway.",
            "runtime": rt.__dict__,
            "pid_alive": pid_alive,
            "health_ok": health_ok,
            "cmdline": cmdline,
        }
        print(_json_dumps(out))
        return 2
    
    try:
        os.kill(rt.pid, signal.SIGTERM)
    except Exception as e:
        print(_json_dumps({"status": "ERROR", "message": f"failed to SIGTERM: {e}", "runtime": rt.__dict__}))
        return 1
    
    dead = _wait_dead(rt.pid, timeout_s=3.0)
    action = "SIGTERM"
    
    if not dead:
        try:
            os.kill(rt.pid, signal.SIGKILL)
        except Exception as e:
            print(_json_dumps({"status": "ERROR", "message": f"failed to SIGKILL: {e}", "runtime": rt.__dict__}))
            return 1
        dead = _wait_dead(rt.pid, timeout_s=2.0)
        action = "SIGKILL"
    
    if dead:
        try:
            runtime_path.unlink(missing_ok=True)
        except Exception:
            pass
        print(_json_dumps({"status": "OK", "action": action, "runtime": rt.__dict__}))
        return 0
    
    print(_json_dumps({"status": "ERROR", "message": "process still alive", "runtime": rt.__dict__}))
    return 1


def cmd_start(args: argparse.Namespace) -> int:
    rt = read_runtime(ROOT)
    if rt:
        pid_alive = _pid_alive(rt.pid)
        health_ok, _ = _health(rt.base_url)
        if pid_alive and health_ok:
            print(_json_dumps({
                "status": "ALREADY_RUNNING",
                "runtime": rt.__dict__,
                "message": "Node is already running. Use 'stop' first if you want to restart."
            }))
            return 1
    
    # Set up environment
    env = os.environ.copy()
    if args.model_path:
        env["MODEL_PATH"] = args.model_path
    if args.token:
        env["HAPA_LLADA_NODE_TOKEN"] = args.token
    
    host = args.host or env.get("HAPA_LLADA_NODE_HOST") or "127.0.0.1"
    port = str(args.port or env.get("PORT") or 8085)
    env["PORT"] = port
    env["HAPA_LLADA_NODE_HOST"] = host
    
    # Start the server in background
    cmd = [sys.executable, "-m", "uvicorn", "src.server:app", "--host", host, "--port", port]
    
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except Exception as e:
        print(_json_dumps({"status": "ERROR", "message": f"Failed to start server: {e}"}))
        return 1
    
    # Wait for startup
    base_url = f"http://{host}:{port}"
    started = False
    for _ in range(30):  # 3 seconds
        time.sleep(0.1)
        health_ok, _ = _health(base_url)
        if health_ok:
            started = True
            break
    
    if started:
        # Read the runtime file that should have been written
        rt = read_runtime(ROOT)
        if rt:
            print(_json_dumps({"status": "STARTED", "runtime": rt.__dict__, "pid": proc.pid}))
        else:
            print(_json_dumps({"status": "STARTED", "pid": proc.pid, "base_url": base_url}))
        return 0
    else:
        # Try to clean up
        try:
            proc.terminate()
        except:
            pass
        print(_json_dumps({"status": "ERROR", "message": "Server failed to start within 3 seconds"}))
        return 1


def cmd_config(args: argparse.Namespace) -> int:
    rt = read_runtime(ROOT)
    token = read_token(ROOT)
    
    config = {
        "runtime_present": rt is not None,
        "token_present": token is not None,
        "repo_root": str(ROOT),
        "runtime_path": str(runtime_file_path(ROOT)),
        "token_path": str(ROOT / ".node_token"),
    }
    
    if rt:
        config["runtime"] = rt.__dict__
    
    if args.show_token and token:
        config["token"] = token
    
    print(_json_dumps(config))
    return 0


def cmd_smoke_test(args: argparse.Namespace) -> int:
    base_url = str(args.base_url).rstrip("/")
    token = _require_token(_get_token(args.token))

    health = _http_json("GET", base_url + "/health", token=None, payload=None)
    print(json.dumps({"health": health}, indent=2))

    caps = _http_json("GET", base_url + "/capabilities", token=token, payload=None)
    print(json.dumps({"capabilities": caps}, indent=2))

    payload: dict[str, Any] = {
        "prompt": _apply_llada_chat_template(args.prompt),
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
    }

    completion = _http_json("POST", base_url + "/v1/completions", token=token, payload=payload)
    print(json.dumps({"completion": completion}, indent=2))

    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Smoke test failed: missing completion.choices")

    text = choices[0].get("text") if isinstance(choices[0], dict) else None
    if not text or not isinstance(text, str):
        raise RuntimeError("Smoke test failed: missing completion.choices[0].text")

    return 0


def cmd_self_test(args: argparse.Namespace) -> int:
    """Run automated self-test of the node."""
    import tempfile
    
    report = {
        "test_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tests": {},
        "ok": False
    }
    
    # Check if node is running
    rt = read_runtime(ROOT)
    if rt:
        # Node is already running, use it
        base_url = rt.base_url
        token = read_token(ROOT)
        report["tests"]["node_already_running"] = True
        managed = False
    elif args.start:
        # Start the node for testing
        print("Starting node for self-test...")
        env = os.environ.copy()
        host = "127.0.0.1"
        port = str(args.port or 8085)
        env["PORT"] = port
        env["HAPA_LLADA_NODE_HOST"] = host
        
        cmd = [sys.executable, "-m", "uvicorn", "src.server:app", "--host", host, "--port", port]
        
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
            managed = True
        except Exception as e:
            report["tests"]["start_node"] = {"ok": False, "error": str(e)}
            print(_json_dumps(report))
            return 1
        
        # Wait for startup
        base_url = f"http://{host}:{port}"
        started = False
        for _ in range(50):  # 5 seconds
            time.sleep(0.1)
            health_ok, _ = _health(base_url)
            if health_ok:
                started = True
                break
        
        if not started:
            proc.terminate()
            report["tests"]["start_node"] = {"ok": False, "error": "Failed to start within 5 seconds"}
            print(_json_dumps(report))
            return 1
        
        # Read token that should have been created
        token = read_token(ROOT)
        report["tests"]["start_node"] = {"ok": True, "pid": proc.pid}
    else:
        report["tests"]["node_running"] = {"ok": False, "error": "Node not running. Use --start to auto-start."}
        print(_json_dumps(report))
        return 1
    
    # Test health endpoint
    try:
        health = _http_json("GET", base_url + "/health", token=None, payload=None)
        report["tests"]["health"] = {
            "ok": health.get("status") == "ok",
            "response": health
        }
    except Exception as e:
        report["tests"]["health"] = {"ok": False, "error": str(e)}
    
    # Test capabilities (authenticated)
    try:
        caps = _http_json("GET", base_url + "/capabilities", token=token, payload=None)
        report["tests"]["capabilities"] = {
            "ok": "modalities" in caps,
            "has_text": "text" in caps.get("modalities", {})
        }
    except Exception as e:
        report["tests"]["capabilities"] = {"ok": False, "error": str(e)}
    
    # Test models endpoint
    try:
        models = _http_json("GET", base_url + "/v1/models", token=token, payload=None)
        report["tests"]["models"] = {
            "ok": "data" in models and len(models["data"]) > 0,
            "count": len(models.get("data", [])),
            "loaded": models["data"][0].get("loaded") if models.get("data") else None
        }
    except Exception as e:
        report["tests"]["models"] = {"ok": False, "error": str(e)}
    
    # Test inference
    try:
        payload = {
            "prompt": _apply_llada_chat_template("Hello, please respond with exactly: TEST_OK"),
            "max_tokens": 32,
            "temperature": 0.1,
        }
        completion = _http_json("POST", base_url + "/v1/completions", token=token, payload=payload)
        choices = completion.get("choices", [])
        text = choices[0].get("text", "") if choices else ""
        report["tests"]["inference"] = {
            "ok": bool(text),
            "response_length": len(text),
            "sample": text[:100] if text else None
        }
    except Exception as e:
        report["tests"]["inference"] = {"ok": False, "error": str(e)}
    
    # Check runtime file
    rt_check = read_runtime(ROOT)
    report["tests"]["runtime_file"] = {
        "ok": rt_check is not None,
        "path": str(runtime_file_path(ROOT))
    }
    
    # Overall status
    all_ok = all(
        test.get("ok", False) 
        for name, test in report["tests"].items() 
        if name != "node_already_running"
    )
    report["ok"] = all_ok
    
    # Clean up if we started the node
    if managed and args.start:
        try:
            proc.terminate()
            _wait_dead(proc.pid, timeout_s=2.0)
            remove_runtime(ROOT)
            report["cleanup"] = "stopped"
        except:
            report["cleanup"] = "failed"
    
    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.output}")
    else:
        print(_json_dumps(report))
    
    return 0 if all_ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hapa-llada-node")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Lifecycle commands
    start = sub.add_parser("start", help="Start the node server in background")
    start.add_argument("--host")
    start.add_argument("--port")
    start.add_argument("--model-path")
    start.add_argument("--token")
    start.set_defaults(func=cmd_start)

    stop = sub.add_parser("stop", help="Stop the running node server")
    stop.add_argument("--force", action="store_true", help="Force stop even without verification")
    stop.set_defaults(func=cmd_stop)

    status = sub.add_parser("status", help="Check node server status")
    status.set_defaults(func=cmd_status)

    config = sub.add_parser("config", help="Show node configuration")
    config.add_argument("--show-token", action="store_true", help="Include token in output")
    config.set_defaults(func=cmd_config)

    # Legacy serve command (foreground)
    serve = sub.add_parser("serve", help="Run server in foreground (legacy)")
    serve.add_argument("--host")
    serve.add_argument("--port")
    serve.add_argument("--model-path")
    serve.add_argument("--token")
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    health = sub.add_parser("health")
    health.add_argument("--base-url", default=_default_base_url())
    health.set_defaults(func=cmd_health)

    caps = sub.add_parser("capabilities")
    caps.add_argument("--base-url", default=_default_base_url())
    caps.add_argument("--token")
    caps.set_defaults(func=cmd_capabilities)

    complete = sub.add_parser("complete")
    complete.add_argument("--base-url", default=_default_base_url())
    complete.add_argument("--token")
    complete.add_argument("--prompt", default="Explain quantum physics in one sentence.")
    complete.add_argument("--max-tokens", dest="max_tokens", type=int, default=200)
    complete.add_argument("--temperature", type=float, default=0.6)
    complete.add_argument("--model")
    complete.set_defaults(func=cmd_complete)

    smoke = sub.add_parser("smoke-test")
    smoke.add_argument("--base-url", default=_default_base_url())
    smoke.add_argument("--token")
    smoke.add_argument("--prompt", default="Explain quantum physics in one sentence.")
    smoke.add_argument("--max-tokens", dest="max_tokens", type=int, default=64)
    smoke.add_argument("--temperature", type=float, default=0.6)
    smoke.set_defaults(func=cmd_smoke_test)

    # Self-test command
    self_test = sub.add_parser("self-test", help="Run automated self-test")
    self_test.add_argument("--start", action="store_true", help="Auto-start node if not running")
    self_test.add_argument("--port", type=int, help="Port to use if auto-starting")
    self_test.add_argument("--output", help="Save report to file")
    self_test.set_defaults(func=cmd_self_test)

    screenshot = sub.add_parser("screenshot")
    screenshot.add_argument("--base-url", default=_default_base_url())
    screenshot.add_argument("--token")
    screenshot.add_argument("--output")
    screenshot.set_defaults(func=cmd_screenshot)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
