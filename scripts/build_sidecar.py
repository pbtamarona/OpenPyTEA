#!/usr/bin/env python3
"""Build the standalone OpenPyTEA backend binary with PyInstaller.

Invokes PyInstaller against backend/openpytea-backend.spec to produce a
self-contained backend at dist/openpytea-backend/. Runs from any working
directory; resolves paths relative to the repo root.

Usage:
    python scripts/build_sidecar.py            # full rebuild
    python scripts/build_sidecar.py --noclean  # incremental
    python scripts/build_sidecar.py --skip-smoke-test
"""
from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

# Windows consoles default to cp1252, which can't encode the →/✓ glyphs
# printed below and raises UnicodeEncodeError. Force UTF-8 so the script
# behaves identically on every platform (including GitHub-hosted runners).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND = REPO_ROOT / "backend"
SPEC = BACKEND / "openpytea-backend.spec"
DIST = REPO_ROOT / "dist" / "openpytea-backend"
BIN_NAME = "openpytea-backend.exe" if os.name == "nt" else "openpytea-backend"


def run_pyinstaller(clean: bool) -> None:
    cmd = [sys.executable, "-m", "PyInstaller",
           str(SPEC),
           "--noconfirm",
           "--distpath", str(REPO_ROOT / "dist"),
           "--workpath", str(REPO_ROOT / "build")]
    if clean:
        cmd.append("--clean")
    print("→", " ".join(cmd))
    subprocess.check_call(cmd, cwd=BACKEND)


def smoke_test(timeout: int = 90) -> None:
    """Launch the built binary, hit /api/health, verify HTTP 200.

    First launches can take ~60s due to matplotlib font-cache build on
    fresh systems, so the default timeout is generous.
    """
    binary = DIST / BIN_NAME
    if not binary.exists():
        raise SystemExit(f"binary not found at {binary}")

    print(f"→ smoke-testing {binary}")
    proc = subprocess.Popen(
        [str(binary), "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Read stdout in a background thread so we can poll proc.poll() in the
    # main loop without blocking on readline() when the child dies.
    lines: "queue.Queue[str]" = queue.Queue()

    def _drain() -> None:
        for line in iter(proc.stdout.readline, ""):
            lines.put(line)
        lines.put("")  # sentinel: stream closed

    threading.Thread(target=_drain, daemon=True).start()

    port = None
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            try:
                line = lines.get(timeout=0.5)
            except queue.Empty:
                if proc.poll() is not None:
                    raise SystemExit(f"binary exited (code {proc.returncode}) before printing port marker")
                continue
            if line == "":
                raise SystemExit("binary closed stdout before printing port marker")
            print("  binary:", line.rstrip())
            if line.startswith("OPENPYTEA_BACKEND_PORT="):
                port = int(line.split("=", 1)[1].strip())
                break
        if port is None:
            raise SystemExit(f"timeout ({timeout}s) waiting for port marker")

        # Wait for uvicorn to actually accept connections (poll, don't sleep).
        url = f"http://127.0.0.1:{port}/api/health"
        accept_deadline = time.time() + 15
        last_err = None
        while time.time() < accept_deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        print(f"✓ /api/health HTTP 200 on port {port}")
                        return
                    raise SystemExit(f"/api/health returned HTTP {r.status}")
            except (urllib.error.URLError, ConnectionError) as e:
                last_err = e
                time.sleep(0.3)
        raise SystemExit(f"could not reach /api/health within 15s: {last_err}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--noclean", action="store_true",
                   help="Skip --clean (faster incremental builds)")
    p.add_argument("--skip-smoke-test", action="store_true",
                   help="Skip the post-build health-check")
    args = p.parse_args()

    if not SPEC.exists():
        raise SystemExit(f"spec not found: {SPEC}")

    # Wipe previous output to avoid stale files being kept.
    if not args.noclean and DIST.exists():
        print(f"→ removing previous {DIST}")
        shutil.rmtree(DIST)

    run_pyinstaller(clean=not args.noclean)

    if not args.skip_smoke_test:
        smoke_test()

    size_mb = sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file()) / 1024 / 1024
    print(f"✓ built {DIST} ({size_mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
