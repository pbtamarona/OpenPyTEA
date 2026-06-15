#!/usr/bin/env python3
"""Patch the macOS .app Info.plist to register a custom UTI for .openpytea
files, then re-sign the bundle.

Tauri's fileAssociations config only generates a CFBundleDocumentTypes
entry that *consumes* a system UTI (public.json) — that's enough for the
app to appear in Finder's "Open With" submenu in theory, but in practice
macOS won't bind the extension strongly unless some app exports an
owning UTI. Without UTExportedTypeDeclarations, .openpytea files end up
with a dyn.* synthetic UTI and our claim is ignored.

This script:
  1. Adds UTExportedTypeDeclarations declaring `org.openpytea.project`
     (conforms to public.json, owns the .openpytea extension).
  2. Rewrites CFBundleDocumentTypes' LSItemContentTypes to refer to the
     custom UTI instead of public.json, and bumps LSHandlerRank to
     "Owner" so we're the canonical handler.
  3. Re-runs `codesign --force --deep --sign -` so the ad-hoc signature
     covers the modified Info.plist.

Run after `npm run tauri build`. Idempotent.
"""
from __future__ import annotations

import plistlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
APP = REPO_ROOT / "frontend/src-tauri/target/release/bundle/macos/OpenPyTEA.app"
PLIST = APP / "Contents/Info.plist"

UTI = "org.openpytea.project"
EXT = "openpytea"


def patch_info_plist() -> bool:
    if not PLIST.exists():
        raise SystemExit(f"Info.plist not found at {PLIST} — run `npm run tauri build` first")

    with open(PLIST, "rb") as f:
        data = plistlib.load(f)

    # 1. UTExportedTypeDeclarations
    exported = data.setdefault("UTExportedTypeDeclarations", [])
    if not any(t.get("UTTypeIdentifier") == UTI for t in exported):
        exported.append({
            "UTTypeIdentifier": UTI,
            "UTTypeDescription": "OpenPyTEA Project",
            "UTTypeConformsTo": ["public.json", "public.text"],
            "UTTypeTagSpecification": {
                "public.filename-extension": [EXT],
                "public.mime-type": ["application/x-openpytea-project"],
            },
        })

    # 2. CFBundleDocumentTypes -> point at our custom UTI, claim ownership
    for doc_type in data.get("CFBundleDocumentTypes", []):
        if EXT in doc_type.get("CFBundleTypeExtensions", []):
            doc_type["LSItemContentTypes"] = [UTI]
            doc_type["LSHandlerRank"] = "Owner"

    with open(PLIST, "wb") as f:
        plistlib.dump(data, f)
    print(f"✓ patched {PLIST.relative_to(REPO_ROOT)}")
    return True


def resign() -> None:
    # The ad-hoc signature on the bundle must be recomputed after any change
    # to Info.plist or any nested binary.
    subprocess.check_call([
        "codesign", "--force", "--deep", "--sign", "-",
        "--options", "runtime",
        str(APP),
    ])
    print(f"✓ re-signed {APP.relative_to(REPO_ROOT)}")


def rebuild_dmg() -> None:
    """Repackage the patched .app into a fresh DMG using hdiutil."""
    dmg_out = APP.parent.parent / "dmg" / "OpenPyTEA_0.1.0_aarch64.dmg"
    dmg_out.parent.mkdir(parents=True, exist_ok=True)

    stage = REPO_ROOT / "build" / "dmg-stage"
    if stage.exists():
        subprocess.check_call(["rm", "-rf", str(stage)])
    stage.mkdir(parents=True)

    # Copy the .app and add the standard Applications symlink for drag-install.
    subprocess.check_call(["cp", "-R", str(APP), str(stage / "OpenPyTEA.app")])
    (stage / "Applications").symlink_to("/Applications")

    # Remove any existing output so hdiutil doesn't refuse.
    if dmg_out.exists():
        dmg_out.unlink()
    subprocess.check_call([
        "hdiutil", "create",
        "-volname", "OpenPyTEA",
        "-srcfolder", str(stage),
        "-ov",
        "-format", "UDZO",
        str(dmg_out),
    ])
    print(f"✓ rebuilt {dmg_out.relative_to(REPO_ROOT)}")


def main() -> int:
    patch_info_plist()
    resign()
    rebuild_dmg()
    return 0


if __name__ == "__main__":
    sys.exit(main())
