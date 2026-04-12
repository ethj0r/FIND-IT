"""Sync files and directories from a remote Kaggle Jupyter kernel to local disk.

Kaggle sessions are ephemeral — when a session expires, everything in
/kaggle/working is lost. This tool pulls files back to the local repo so the
results survive the session (and are available to git).

Uses the Jupyter kernel execution channel (via kaggle_exec) to run short
Python helpers on the remote side and streams their output back. Binary files
are transferred as base64 over the kernel's stdout stream; directories are
packaged as tar.gz first.

Usage:
    python -m src.tools.kaggle_sync ls /kaggle/working
    python -m src.tools.kaggle_sync pull-file /kaggle/working/best.pt models/best.pt
    python -m src.tools.kaggle_sync pull-dir /kaggle/working/runs/xxx runs/xxx
    python -m src.tools.kaggle_sync watch-log /kaggle/working/train_full_baseline.log

Requires KAGGLE_PROXY_URL env var (the Kaggle Jupyter proxy URL).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import sys
import tarfile
from io import BytesIO
from pathlib import Path

from src.tools.kaggle_exec import run_code


MAX_FILE_MB = 200  # refuse to stream files larger than this
CHUNK_START = "===CHUNK_START==="
CHUNK_END = "===CHUNK_END==="


def _require_base_url() -> str:
    url = os.environ.get("KAGGLE_PROXY_URL")
    if not url:
        sys.exit("KAGGLE_PROXY_URL env var is required")
    return url


def _run(code: str, timeout: int = 600) -> str:
    base = _require_base_url()
    result = run_code(base, code, timeout=timeout, quiet=True)
    if result["error"]:
        err = result["error"]
        sys.stderr.write(f"[kaggle error] {err['ename']}: {err['evalue']}\n")
        for line in err.get("traceback") or []:
            sys.stderr.write(line + "\n")
        sys.exit(1)
    return result["stdout"]


def cmd_ls(path: str, max_depth: int = 2) -> None:
    code = f"""
import os
ROOT = {path!r}
MAX = {max_depth}
def walk(p, depth=0):
    try:
        entries = sorted(os.listdir(p))
    except Exception as e:
        print('  ' * depth + f'[err {{e}}]')
        return
    for e in entries:
        full = os.path.join(p, e)
        try:
            is_dir = os.path.isdir(full)
        except OSError:
            continue
        if is_dir:
            print('  ' * depth + e + '/')
            if depth < MAX:
                walk(full, depth + 1)
        else:
            try:
                size = os.path.getsize(full)
                size_mb = size / 1024**2
                tag = f'{{size_mb:.2f}} MB' if size_mb >= 0.1 else f'{{size}} B'
                print('  ' * depth + f'{{e}}  ({{tag}})')
            except OSError:
                print('  ' * depth + e)
walk(ROOT)
"""
    out = _run(code)
    sys.stdout.write(out)


def _pull_bytes(remote_path: str) -> bytes:
    """Read a remote file and return its raw bytes locally."""
    code = f"""
import os, base64, hashlib, sys
P = {remote_path!r}
if not os.path.exists(P):
    raise FileNotFoundError(P)
size = os.path.getsize(P)
if size > {MAX_FILE_MB} * 1024 * 1024:
    raise RuntimeError(f'File too large ({{size/1024**2:.1f}} MB > {MAX_FILE_MB} MB)')
with open(P, 'rb') as f:
    data = f.read()
h = hashlib.sha256(data).hexdigest()
b64 = base64.b64encode(data).decode('ascii')
print({CHUNK_START!r})
print(f'SIZE={{size}}')
print(f'SHA256={{h}}')
print('B64_BEGIN')
# Stream in ~64KB lines to keep ws messages reasonable
CHUNK = 64 * 1024
for i in range(0, len(b64), CHUNK):
    print(b64[i:i+CHUNK])
print('B64_END')
print({CHUNK_END!r})
sys.stdout.flush()
"""
    stdout = _run(code, timeout=900)
    # Parse the payload between CHUNK_START and CHUNK_END
    if CHUNK_START not in stdout or CHUNK_END not in stdout:
        sys.exit("Transfer markers not found in kernel output")
    block = stdout.split(CHUNK_START, 1)[1].split(CHUNK_END, 1)[0]
    lines = [l.strip() for l in block.splitlines() if l.strip()]
    meta: dict[str, str] = {}
    b64_lines: list[str] = []
    in_b64 = False
    for line in lines:
        if line == "B64_BEGIN":
            in_b64 = True
            continue
        if line == "B64_END":
            in_b64 = False
            continue
        if in_b64:
            b64_lines.append(line)
        elif "=" in line:
            k, v = line.split("=", 1)
            meta[k] = v

    payload = base64.b64decode("".join(b64_lines))
    if "SIZE" in meta and len(payload) != int(meta["SIZE"]):
        sys.exit(f"Size mismatch: expected {meta['SIZE']}, got {len(payload)}")
    if "SHA256" in meta:
        h = hashlib.sha256(payload).hexdigest()
        if h != meta["SHA256"]:
            sys.exit(f"SHA256 mismatch: expected {meta['SHA256']}, got {h}")
    return payload


def cmd_pull_file(remote: str, local: str) -> None:
    sys.stderr.write(f"[pull-file] {remote} -> {local}\n")
    data = _pull_bytes(remote)
    local_path = Path(local)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)
    sys.stderr.write(f"[pull-file] wrote {len(data)/1024**2:.2f} MB\n")


def cmd_pull_dir(remote: str, local: str) -> None:
    sys.stderr.write(f"[pull-dir] {remote} -> {local}\n")
    code = f"""
import os, base64, hashlib, tarfile, io, sys
P = {remote!r}
if not os.path.isdir(P):
    raise NotADirectoryError(P)
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode='w:gz') as tf:
    tf.add(P, arcname=os.path.basename(P.rstrip('/')))
data = buf.getvalue()
if len(data) > {MAX_FILE_MB} * 1024 * 1024:
    raise RuntimeError(f'Archive too large ({{len(data)/1024**2:.1f}} MB > {MAX_FILE_MB} MB)')
h = hashlib.sha256(data).hexdigest()
b64 = base64.b64encode(data).decode('ascii')
print({CHUNK_START!r})
print(f'SIZE={{len(data)}}')
print(f'SHA256={{h}}')
print('B64_BEGIN')
CHUNK = 64 * 1024
for i in range(0, len(b64), CHUNK):
    print(b64[i:i+CHUNK])
print('B64_END')
print({CHUNK_END!r})
sys.stdout.flush()
"""
    stdout = _run(code, timeout=1200)
    if CHUNK_START not in stdout or CHUNK_END not in stdout:
        sys.exit("Transfer markers not found")
    block = stdout.split(CHUNK_START, 1)[1].split(CHUNK_END, 1)[0]
    lines = [l.strip() for l in block.splitlines() if l.strip()]
    meta: dict[str, str] = {}
    b64_lines: list[str] = []
    in_b64 = False
    for line in lines:
        if line == "B64_BEGIN":
            in_b64 = True
            continue
        if line == "B64_END":
            in_b64 = False
            continue
        if in_b64:
            b64_lines.append(line)
        elif "=" in line:
            k, v = line.split("=", 1)
            meta[k] = v

    payload = base64.b64decode("".join(b64_lines))
    if "SHA256" in meta:
        h = hashlib.sha256(payload).hexdigest()
        if h != meta["SHA256"]:
            sys.exit(f"SHA256 mismatch")
    local_path = Path(local)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as tf:
        tf.extractall(local_path.parent)
    sys.stderr.write(f"[pull-dir] extracted {len(payload)/1024**2:.2f} MB to {local_path.parent}\n")


def cmd_watch_log(remote: str, tail: int = 60) -> None:
    """Print last N lines of a remote log file, ANSI-stripped."""
    code = f"""
import os, re
P = {remote!r}
if not os.path.exists(P):
    print('MISSING:', P); raise SystemExit(0)
content = open(P, 'r', errors='replace').read()
clean = re.sub(r'\\x1b\\[[0-9;]*m', '', content)
clean = clean.replace('\\r', '\\n')
lines = [l for l in clean.split('\\n') if l.strip()]
for l in lines[-{tail}:]:
    print(l[:200])
print('---')
print(f'total_lines={{len(lines)}}  size_bytes={{os.path.getsize(P)}}')
"""
    out = _run(code)
    sys.stdout.write(out)


def cmd_ps() -> None:
    """Show the training PID liveness and basic GPU status."""
    code = """
import os, subprocess
pid_file = '/kaggle/working/train_full_baseline.pid'
if os.path.exists(pid_file):
    pid = int(open(pid_file).read().strip())
    alive = os.path.exists(f'/proc/{pid}')
    print(f'training pid={pid} alive={alive}')
else:
    print('no pid file')
try:
    out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
    print('gpu:', out.stdout.strip())
except Exception as e:
    print('nvidia-smi err:', e)
"""
    out = _run(code)
    sys.stdout.write(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ls = sub.add_parser("ls", help="List remote directory tree")
    p_ls.add_argument("path")
    p_ls.add_argument("--depth", type=int, default=2)

    p_pf = sub.add_parser("pull-file", help="Pull a single remote file to local")
    p_pf.add_argument("remote")
    p_pf.add_argument("local")

    p_pd = sub.add_parser("pull-dir", help="Pull a remote directory as tar.gz")
    p_pd.add_argument("remote")
    p_pd.add_argument("local")

    p_wl = sub.add_parser("watch-log", help="Tail a remote log file (ANSI-stripped)")
    p_wl.add_argument("remote")
    p_wl.add_argument("--tail", type=int, default=60)

    sub.add_parser("ps", help="Check training process liveness and GPU")

    args = ap.parse_args()
    if args.cmd == "ls":
        cmd_ls(args.path, max_depth=args.depth)
    elif args.cmd == "pull-file":
        cmd_pull_file(args.remote, args.local)
    elif args.cmd == "pull-dir":
        cmd_pull_dir(args.remote, args.local)
    elif args.cmd == "watch-log":
        cmd_watch_log(args.remote, tail=args.tail)
    elif args.cmd == "ps":
        cmd_ps()


if __name__ == "__main__":
    main()
