"""Execute Python code on a remote Kaggle Jupyter kernel via WebSocket.

Kaggle exposes notebook kernels through a Jupyter proxy URL. When the kernel
is running, we can open a WebSocket to its channels endpoint and send standard
Jupyter wire-protocol `execute_request` messages. This script wraps that flow
so arbitrary code can be run from the local shell and outputs streamed back.

Usage:
    python -m src.tools.kaggle_exec --base-url <proxy_url> --code "print('hi')"
    python -m src.tools.kaggle_exec --base-url <proxy_url> --file script.py
    python -m src.tools.kaggle_exec --base-url <proxy_url> --stdin < script.py

The base URL is the full proxy URL ending in `/proxy` — the same link Kaggle
gives when you click "Share" on a notebook session. This tool also reads
`KAGGLE_PROXY_URL` from the environment if `--base-url` is omitted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
import websocket


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_header(msg_type: str, session: str) -> dict:
    return {
        "msg_id": uuid.uuid4().hex,
        "username": "claude",
        "session": session,
        "msg_type": msg_type,
        "version": "5.3",
        "date": iso_now(),
    }


def build_execute_request(code: str, session: str) -> dict:
    return {
        "header": make_header("execute_request", session),
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        "channel": "shell",
    }


def fetch_kernel(base_url: str) -> dict:
    r = requests.get(f"{base_url}/api/kernels", timeout=15)
    r.raise_for_status()
    kernels = r.json()
    if not kernels:
        raise RuntimeError("No active kernel on this Kaggle session")
    return kernels[0]


def make_ws_url(base_url: str, kernel_id: str) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}{parsed.path}/api/kernels/{kernel_id}/channels"


def run_code(
    base_url: str,
    code: str,
    timeout: int = 600,
    quiet: bool = False,
) -> dict:
    kernel = fetch_kernel(base_url)
    kernel_id = kernel["id"]
    if not quiet:
        print(f"[kaggle_exec] kernel: {kernel_id} (state={kernel['execution_state']})",
              file=sys.stderr)

    ws_url = make_ws_url(base_url, kernel_id)
    session = uuid.uuid4().hex
    ws = websocket.create_connection(ws_url, timeout=timeout)

    try:
        msg = build_execute_request(code, session)
        ws.send(json.dumps(msg))
        msg_id = msg["header"]["msg_id"]

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        result_parts: list[str] = []
        error_info: dict | None = None
        execution_count: int | None = None
        done = False

        while not done:
            raw = ws.recv()
            if not raw:
                continue
            m = json.loads(raw)
            parent = m.get("parent_header") or {}
            if parent.get("msg_id") != msg_id:
                continue
            mtype = m.get("msg_type")
            content = m.get("content", {})

            if mtype == "stream":
                text = content.get("text", "")
                if content.get("name") == "stderr":
                    stderr_parts.append(text)
                    if not quiet:
                        sys.stderr.write(text)
                        sys.stderr.flush()
                else:
                    stdout_parts.append(text)
                    if not quiet:
                        sys.stdout.write(text)
                        sys.stdout.flush()
            elif mtype == "execute_result":
                data = content.get("data", {})
                text = data.get("text/plain", "")
                result_parts.append(text)
                if not quiet:
                    print(text)
            elif mtype == "display_data":
                data = content.get("data", {})
                text = data.get("text/plain", "")
                if text:
                    result_parts.append(text)
                    if not quiet:
                        print(text)
            elif mtype == "error":
                error_info = {
                    "ename": content.get("ename"),
                    "evalue": content.get("evalue"),
                    "traceback": content.get("traceback", []),
                }
                if not quiet:
                    for line in error_info["traceback"]:
                        print(line, file=sys.stderr)
            elif mtype == "execute_reply":
                execution_count = content.get("execution_count")
                status = content.get("status")
                if status == "error" and error_info is None:
                    error_info = {
                        "ename": content.get("ename"),
                        "evalue": content.get("evalue"),
                        "traceback": content.get("traceback", []),
                    }
                done = True
    finally:
        ws.close()

    return {
        "kernel_id": kernel_id,
        "execution_count": execution_count,
        "stdout": "".join(stdout_parts),
        "stderr": "".join(stderr_parts),
        "result": "\n".join(result_parts),
        "error": error_info,
        "ok": error_info is None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=os.environ.get("KAGGLE_PROXY_URL"),
                    help="Full Kaggle Jupyter proxy URL ending in /proxy")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--code", help="Inline Python code string")
    grp.add_argument("--file", help="Read code from file")
    grp.add_argument("--stdin", action="store_true",
                     help="Read code from stdin")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--json", action="store_true",
                    help="Emit full response as JSON instead of live stream")
    args = ap.parse_args()

    if not args.base_url:
        sys.exit("Missing --base-url (or KAGGLE_PROXY_URL env var)")

    if args.code:
        code = args.code
    elif args.file:
        code = open(args.file).read()
    else:
        code = sys.stdin.read()

    result = run_code(args.base_url, code, timeout=args.timeout,
                      quiet=args.json)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result["error"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
