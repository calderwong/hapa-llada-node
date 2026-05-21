from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class NodeRuntime:
    base_url: str
    host: str
    port: int
    token_path: str
    started_at: str
    pid: int


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def token_file_path(root: Optional[Path] = None) -> Path:
    r = root or repo_root()
    return r / ".node_token"


def runtime_file_path(root: Optional[Path] = None) -> Path:
    r = root or repo_root()
    return r / "artifacts" / "runtime" / "hapa_llada_node_runtime.json"


def read_token(root: Optional[Path] = None, token: Optional[str] = None) -> Optional[str]:
    tok = token.strip() if token else ""

    if not tok:
        env_token = os.environ.get("HAPA_LLADA_NODE_TOKEN")
        if env_token:
            tok = env_token.strip()

    if not tok:
        p = token_file_path(root)
        if p.exists():
            tok = p.read_text(encoding="utf-8").strip()

    return tok or None


def ensure_token(root: Optional[Path] = None, token: Optional[str] = None) -> str:
    tok = read_token(root=root, token=token)
    if not tok:
        tok = secrets.token_urlsafe(32)

    p = token_file_path(root)
    p.write_text(tok + "\n", encoding="utf-8")
    return tok


def resolve_token(root: Optional[Path] = None, token: Optional[str] = None) -> str:
    return ensure_token(root=root, token=token)


def read_runtime(root: Optional[Path] = None) -> Optional[NodeRuntime]:
    p = runtime_file_path(root)
    if not p.exists():
        return None

    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(doc, dict):
        return None

    base_url = doc.get("base_url")
    host = doc.get("host")
    port = doc.get("port")
    token_path = doc.get("token_path")
    started_at = doc.get("started_at")
    pid = doc.get("pid")

    if not isinstance(base_url, str):
        return None
    if not isinstance(host, str):
        return None
    if not isinstance(port, int):
        return None
    if not isinstance(token_path, str):
        return None
    if not isinstance(started_at, str):
        return None
    if not isinstance(pid, int):
        return None

    return NodeRuntime(
        base_url=base_url,
        host=host,
        port=port,
        token_path=token_path,
        started_at=started_at,
        pid=pid,
    )


def write_runtime(root: Path, host: str, port: int, token_path: Path) -> NodeRuntime:
    base_url = f"http://{host}:{port}"
    try:
        token_path_str = str(token_path.relative_to(root))
    except Exception:
        token_path_str = str(token_path)

    rt = NodeRuntime(
        base_url=base_url,
        host=host,
        port=port,
        token_path=token_path_str,
        started_at=_now(),
        pid=os.getpid(),
    )

    out = {
        "base_url": rt.base_url,
        "host": rt.host,
        "port": rt.port,
        "token_path": rt.token_path,
        "started_at": rt.started_at,
        "pid": rt.pid,
    }

    p = runtime_file_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rt


def remove_runtime(root: Optional[Path] = None) -> None:
    p = runtime_file_path(root)
    try:
        p.unlink()
    except FileNotFoundError:
        return
    except Exception:
        return
