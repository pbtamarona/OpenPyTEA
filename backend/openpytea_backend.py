"""PyInstaller entrypoint for the OpenPyTEA backend.

Runs the FastAPI app under uvicorn, bound to 127.0.0.1 only. When `--port 0`
is passed, the OS picks a free port and the chosen value is printed on
stdout as a single line:

    OPENPYTEA_BACKEND_PORT=<port>

The marker is printed *after* uvicorn has fully started listening, so a
parent process (e.g. the Tauri shell) can connect immediately on seeing
it without racing the socket bind.

This file is the PyInstaller entry script. It must live at the top of the
backend tree (alongside the `app/` package) so the bundled binary can
import `from app.main import app` cleanly.
"""
from __future__ import annotations

import multiprocessing

# Frozen-binary support for multiprocessing. When a worker re-invokes the
# bundled executable to spawn a helper, freeze_support() detects the helper
# invocation, runs the worker code, and exits — preventing our argparse from
# choking on the helper's own argv (e.g. `-B -S -I -c "from
# multiprocessing.resource_tracker import main; main(N)"`). Must run before
# argparse touches sys.argv, hence before everything else.
multiprocessing.freeze_support()

import argparse
import socket
import sys
import threading

import uvicorn

from app.main import app


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _parent_watchdog(server: uvicorn.Server) -> None:
    """Block until stdin hits EOF, then ask uvicorn to shut down.

    The parent (Tauri shell) spawns us with a piped stdin it never writes
    to. The OS closes that pipe when the parent exits for *any* reason —
    including crashes and force-kills that skip the parent's own cleanup —
    so stdin-EOF is a reliable orphan-prevention tether on every platform.
    """
    try:
        sys.stdin.buffer.read()
    except Exception:
        pass
    server.should_exit = True


class _AnnouncingServer(uvicorn.Server):
    """uvicorn.Server that prints the port marker after startup completes."""

    def __init__(self, config: uvicorn.Config, announce_port: int) -> None:
        super().__init__(config)
        self._announce_port = announce_port

    async def startup(self, sockets=None):  # type: ignore[override]
        await super().startup(sockets=sockets)
        # By the time we get here uvicorn has bound the socket and the
        # FastAPI app's startup events have run — safe for clients to
        # connect immediately on seeing the marker.
        print(f"OPENPYTEA_BACKEND_PORT={self._announce_port}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenPyTEA backend server")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default: 127.0.0.1, localhost only)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind (default: 8000; pass 0 for OS-assigned)")
    parser.add_argument("--log-level", default="warning",
                        choices=["critical", "error", "warning", "info", "debug", "trace"])
    parser.add_argument("--exit-on-parent-close", action="store_true",
                        help="Shut down when stdin reaches EOF, i.e. when the "
                             "parent process that spawned us with a piped stdin "
                             "has exited (cleanly or not). Opt-in so running "
                             "standalone from a terminal is unaffected.")
    args = parser.parse_args(argv)

    port = args.port if args.port != 0 else _pick_free_port(args.host)

    config = uvicorn.Config(app, host=args.host, port=port, log_level=args.log_level)
    server = _AnnouncingServer(config, port)
    if args.exit_on_parent_close:
        threading.Thread(target=_parent_watchdog, args=(server,), daemon=True).start()
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
