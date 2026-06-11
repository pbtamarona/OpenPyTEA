"""PyInstaller entrypoint for the OpenPyTEA backend.

Runs the FastAPI app under uvicorn, bound to 127.0.0.1 only. When `--port 0`
is passed, the OS picks a free port and the chosen value is printed on
stdout as a single line:

    OPENPYTEA_BACKEND_PORT=<port>

A Tauri (or any other) parent process can capture that line to know where
to send API requests. The marker line is also printed for fixed `--port N`
so the parent has a uniform signal.

This file is the PyInstaller entry script. It must live at the top of the
backend tree (alongside the `app/` package) so the bundled binary can
import `from app.main import app` cleanly.
"""
from __future__ import annotations

import argparse
import socket
import sys

import uvicorn

from app.main import app


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenPyTEA backend server")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default: 127.0.0.1, localhost only)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind (default: 8000; pass 0 for OS-assigned)")
    parser.add_argument("--log-level", default="warning",
                        choices=["critical", "error", "warning", "info", "debug", "trace"])
    args = parser.parse_args(argv)

    port = args.port if args.port != 0 else _pick_free_port(args.host)
    print(f"OPENPYTEA_BACKEND_PORT={port}", flush=True)

    uvicorn.run(app, host=args.host, port=port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
