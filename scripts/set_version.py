#!/usr/bin/env python3
"""Stamp a version number into every file that declares one.

Used by CI when a v* tag triggers a release build, so the tag is the single
source of truth: `git tag v0.2.0` produces installers that are actually
versioned 0.2.0 instead of whatever tauri.conf.json last said.

    python scripts/set_version.py 0.2.0

Updates:
  - frontend/src-tauri/tauri.conf.json  (drives bundle names + app version)
  - frontend/src-tauri/Cargo.toml       (kept in sync for consistency)
  - frontend/package.json               (kept in sync for consistency)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TAURI_CONF = REPO_ROOT / "frontend/src-tauri/tauri.conf.json"
CARGO_TOML = REPO_ROOT / "frontend/src-tauri/Cargo.toml"
PACKAGE_JSON = REPO_ROOT / "frontend/package.json"

# Tauri requires semver; enforce it here so a malformed tag fails the build
# at this step with a clear message instead of deep inside the bundler.
SEMVER = re.compile(r"^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$")


def set_json_version(path: Path, version: str) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["version"] = version
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"✓ {path.relative_to(REPO_ROOT)} → {version}")


def set_cargo_version(path: Path, version: str) -> None:
    text = path.read_text(encoding="utf-8")
    # Only the first `version = "…"` — that's the [package] entry; dependency
    # tables in this manifest use inline `tauri = { version = … }` syntax.
    new_text, n = re.subn(
        r'^version\s*=\s*"[^"]*"',
        f'version = "{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise SystemExit(f"no package version line found in {path}")
    path.write_text(new_text, encoding="utf-8")
    print(f"✓ {path.relative_to(REPO_ROOT)} → {version}")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <version>  (e.g. 0.2.0)")
    version = sys.argv[1].lstrip("v")
    if not SEMVER.match(version):
        raise SystemExit(f"not a semver version: {version!r} (expected e.g. 0.2.0)")

    set_json_version(TAURI_CONF, version)
    set_json_version(PACKAGE_JSON, version)
    set_cargo_version(CARGO_TOML, version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
