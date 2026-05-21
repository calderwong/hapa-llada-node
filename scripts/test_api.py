import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Optional


def _http_json(method: str, url: str, *, token: Optional[str], payload: Optional[dict]) -> dict:
    data = None
    headers: dict[str, str] = {}

    if token:
        headers["Authorization"] = f"Bearer {token}"

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, method=method, data=data, headers=headers)

    try:
        with urllib.request.urlopen(req) as res:
            body = res.read()
            return json.loads(body.decode("utf-8")) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read() if hasattr(exc, "read") else b""
        text = body.decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {text}")


def _read_text_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return text or None
    except Exception:
        return None


def _get_token(arg_token: Optional[str]) -> Optional[str]:
    if arg_token:
        return arg_token
    env_tok = os.environ.get("HAPA_LLADA_NODE_TOKEN")
    if env_tok:
        return env_tok
    return _read_text_file(".node_token")


def _apply_llada_chat_template(prompt: str) -> str:
    if "<role>" in prompt:
        return prompt
    system_prompt = "You are a helpful assistant."
    return (
        f"<role>SYSTEM</role>{system_prompt}<|role_end|>"
        f"<role>HUMAN</role>{prompt}<|role_end|>"
        f"<role>ASSISTANT</role>"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="test_api")
    p.add_argument(
        "--base-url",
        default=os.environ.get("HAPA_LLADA_NODE_BASE_URL", "http://127.0.0.1:8085"),
    )
    p.add_argument("--token")
    p.add_argument("--capabilities", action="store_true")
    p.add_argument("--health", action="store_true")
    p.add_argument("--prompt", default="Explain quantum physics in one sentence.")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.6)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    base_url = str(args.base_url).rstrip("/")
    token = _get_token(args.token)

    if args.health:
        data = _http_json("GET", base_url + "/health", token=None, payload=None)
        print(json.dumps(data, indent=2))
        return 0

    if args.capabilities:
        if not token:
            raise RuntimeError("Missing token (set HAPA_LLADA_NODE_TOKEN or pass --token)")
        data = _http_json("GET", base_url + "/capabilities", token=token, payload=None)
        print(json.dumps(data, indent=2))
        return 0

    if not token:
        raise RuntimeError("Missing token (set HAPA_LLADA_NODE_TOKEN or pass --token)")

    payload = {
        "prompt": _apply_llada_chat_template(args.prompt),
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
    }

    data = _http_json("POST", base_url + "/v1/completions", token=token, payload=payload)
    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
