# Standalone OpenPyTEA app (macOS)

This repository builds a single double-clickable **OpenPyTEA.app** (and a
matching `.dmg` installer) that bundles the GUI, the FastAPI backend, and
the Python runtime — recipients don't need Python, Node, or any other
tooling on their machine.

The build runs on macOS (Apple Silicon for now). Windows and Linux would
need to repeat the same build on a matching OS; cross-compilation is not
supported.

---

## For end users — installing & running

Open the `.dmg` you received, drag **OpenPyTEA** to your **Applications**
folder, eject the disk image, then double-click the app to launch.

**First launch.** The app is unsigned (no Apple Developer ID), so macOS
will refuse to open it with *"OpenPyTEA can't be opened…"* / *"is from an
unidentified developer"*. Clear Apple's "downloaded from the internet"
flag once, in Terminal:

```bash
xattr -dr com.apple.quarantine /Applications/OpenPyTEA.app
```

After that, double-click the app normally. The first launch takes
~30–45 s because Python's matplotlib builds its font cache; subsequent
launches are fast.

**Opening `.openpytea` files from Finder.** Once installed, double-clicking
any `.openpytea` project file in Finder will launch the app and load the
project automatically. If "OpenPyTEA" doesn't show up in *Open With*,
force-refresh Launch Services once:

```bash
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f /Applications/OpenPyTEA.app
killall Finder
```

### Using the app

The header has a **File** menu (native macOS menu bar at the top of the
screen): **New / Open / Save / Save As**, with the usual `⌘N / ⌘O / ⌘S /
⌘⇧S` shortcuts. The **Examples** dropdown in the window header loads
built-in case-study presets.

Typical workflow:

1. **Examples → pick a preset** (or build a project from scratch in the
   Equipment + Plant Config tabs)
2. **Results tab** — Calculate (auto-runs after loading)
3. **Add to comparison** with a name
4. Repeat with another example to stack up comparisons
5. **Analysis / Monte Carlo / Tornado** tabs — overlay the compared
   plants by ticking them in "Compare with"
6. **File → Save As → `mywork.openpytea`** to persist everything,
   including the comparison list

Quitting with unsaved work pops a *Save / Don't Save / Cancel* dialog.

---

## For developers — building a fresh installer

### Prerequisites (one-time per machine)

- **Python ≥3.10** — recommend Homebrew: `brew install python@3.13`
- **Node.js** — `brew install node`
- **Rust toolchain** — `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Git LFS** — required because the repo has LFS-tracked Aspen files:
  `brew install git-lfs && git lfs install`

### Build

```bash
# 1. Bootstrap the Python venv (one-time)
./start.sh   # runs once to make .venv and install deps, then Ctrl+C

# 2. Build the bundled backend (PyInstaller onedir, ~152 MB)
.venv/bin/python scripts/build_sidecar.py --skip-smoke-test

# 3. Build the Tauri shell + .dmg
cd frontend
npm install        # first time only
npm run tauri build

# 4. Patch the macOS Info.plist with a custom UTI for .openpytea files
#    and re-sign the bundle. Also regenerates the .dmg from the patched
#    .app.
cd ..
.venv/bin/python scripts/patch_macos_plist.py
```

Outputs:

- **`.app` bundle**: `frontend/src-tauri/target/release/bundle/macos/OpenPyTEA.app`
- **`.dmg` installer**: `frontend/src-tauri/target/release/bundle/dmg/OpenPyTEA_<version>_aarch64.dmg`

The `.dmg` is what you share with users. ~70–82 MB depending on
compression.

### Build limits

- **Apple Silicon only.** The Rust build produces an arm64 binary. To
  ship to Intel Macs, run the same sequence on an Intel Mac.
- **No cross-build.** A Windows `.msi` needs to be built on Windows; a
  Linux `.deb` / `.AppImage` on Linux. Same `scripts/build_sidecar.py`
  + `npm run tauri build` work, but `patch_macos_plist.py` is macOS-
  specific and isn't needed on Windows/Linux.
- **Unsigned.** Recipients have to clear the `com.apple.quarantine`
  attribute manually (see above). Signing requires an Apple Developer
  ID ($99/yr) and is not configured.

---

## Layout, briefly

```
backend/
  app/                     # FastAPI app
  openpytea_backend.py     # PyInstaller entrypoint
  openpytea-backend.spec   # PyInstaller spec

frontend/
  src/                     # React app
  src-tauri/               # Tauri (Rust) shell — bundles the .app
    src/lib.rs             # spawns backend sidecar, native menu, file
                           # associations, close-confirm
    tauri.conf.json        # bundle config (resources, fileAssociations,
                           # ad-hoc signing)

scripts/
  build_sidecar.py         # PyInstaller wrapper
  patch_macos_plist.py     # adds UTExportedTypeDeclarations + re-signs
                           # + rebuilds .dmg

dist/openpytea-backend/    # PyInstaller output (~152 MB)
```
