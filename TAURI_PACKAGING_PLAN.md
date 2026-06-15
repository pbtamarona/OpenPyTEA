# OpenPyTEA standalone-app packaging plan (Tauri)

**Goal:** ship OpenPyTEA as a single double-clickable application — backend + GUI bundled together, one icon, native installers for macOS / Windows / Linux. End users should not need Python, Node, or any setup.

This document is a working plan / decision log, not a commitment. Re-read it standalone; it does not assume chat context.

---

## Progress (latest first)

Work happens on branch `standalone-package` (see https://github.com/pbtamarona/OpenPyTEA/tree/standalone-package). The GUI work for users stays on `GUI-beta`; packaging experiments are isolated here so the working GUI never breaks.

### Status: a shareable Mac `.dmg` exists and works end-to-end.

Locally produced via `npm run tauri build` (frontend dir), 71 MB compressed installer, 163 MB `.app` on disk. Arm64 Apple Silicon only. Verified by an external tester on a different machine. **Apple Silicon Macs**: not yet on **Intel** or **Windows** — those require running the same build on a matching OS (Step 4 → CI handles this).

### Commits, latest first

- `559a1b0` **Ad-hoc sign the macOS bundle.** `bundle.macOS.signingIdentity = "-"` in `tauri.conf.json` so Tauri runs a proper deep ad-hoc codesign over the assembled `.app`. Fixes the "code has no resources but signature indicates they must be present" mismatch that was making downloaded copies show "*OpenPyTEA is damaged and can't be opened*" on recipients' Macs. Still not Developer-ID signed (would need $99/yr) — recipients still see the milder "from unidentified developer" warning and need `xattr -dr com.apple.quarantine /Applications/OpenPyTEA.app` (or right-click → Open).
- `ae1a49a` **`multiprocessing.freeze_support()`** at the top of the PyInstaller entrypoint. Monte Carlo's parallel workers re-invoke the bundled exe with Python-helper argv (`-B -S -I -c "from multiprocessing.resource_tracker import main; main(N)"`); without freeze_support our argparse rejected them and every spawn leaked a dead helper. MC still produced correct numbers but loudly.
- `f501592` **The "make the packaged .app talk to its backend" fix** (three production-only bugs in one commit):
  - `backend/app/main.py` CORS allowlist now includes `tauri://localhost` (macOS/Linux) and `https://tauri.localhost` (Windows). Without this the packaged webview's fetches were CORS-blocked and surfaced as "Load failed" after a few seconds.
  - `frontend/src/api/client.ts` switched from hand-rolled `"__TAURI_INTERNALS__" in window` to the official `isTauri()` from `@tauri-apps/api/core`. The hand-rolled check was unreliable across production webview load orders.
  - `frontend/src-tauri/src/lib.rs` enabled `tauri-plugin-log` unconditionally (was gated on `cfg!(debug_assertions)`), with both Stdout and LogDir targets. Production logs now appear in the terminal when launched there and persist to `~/Library/Logs/org.openpytea.app/` on macOS.
- `0b93cb1` **Multi-plant tornado.** Sensitivity + MC already supported it via `extra_plants`; tornado was the odd one out. Backend returns `{plants: [{name, factors, labels, lows, highs, base_value}…], plus_minus_value, metric, xlabel}`. Frontend renders grouped low/high bars per plant per parameter; single-plant keeps the original absolute-value x-axis trick, multi-plant switches to a pure Δ axis. Cherry-picked to `GUI-beta` as `9bce3fd`.
- `eb4c109` **Preset price fix.** Four preset JSONs (`h2_smr`, `h2_electrolysis`, `h2_methane_pyrolysis`, `lh2_smr_best`) were missing the `price` field on their hydrogen/liquid_h2 products → revenue silently calculated to 0 and tornado crashed with `KeyError("price")`. Added 5 US$/kg (H2) and 6 US$/kg (LH2) with std/min/max from the walkthrough notebook. Cherry-picked to `GUI-beta` as `ba607b5`.
- `cab872b` **Drain backend stderr.** Tauri shell piped stderr but never read it; uvicorn's access logs filled the 64 KB buffer and uvicorn blocked on its next stderr write. GUI got a few responses, then hung — the actual root cause of "examples don't load" earlier in the session. Added a second reader thread that drains stderr to the Rust logger.
- `9dd97de` **Sidecar timing race.** The entrypoint printed `OPENPYTEA_BACKEND_PORT=…` *before* uvicorn finished binding. Subclassed `uvicorn.Server` so the marker prints inside `startup()` — after the socket is bound and FastAPI startup events have run.
- `291743c` Step 3 — sidecar wiring (initial). Tauri spawns the PyInstaller binary, reads the port from stdout, exposes it via the `get_api_base` IPC command, kills the child on app exit.
- `95fe5cd` Step 2 — Tauri scaffold (`frontend/src-tauri/`, Tauri 2.11, identifier `org.openpytea.app`, 1400×900 default).
- `7bda17c` Step 1 — backend as PyInstaller binary (`dist/openpytea-backend/`, 152 MB onedir, `--port 0` + marker entrypoint, built by `python scripts/build_sidecar.py`).

### Distribution recipe today (Mac arm64)

```bash
# rebuild backend (if openpytea/backend code changed)
python scripts/build_sidecar.py --skip-smoke-test
# rebuild the .app + .dmg
cd frontend && npm run tauri build
# installer:
# frontend/src-tauri/target/release/bundle/dmg/OpenPyTEA_<version>_aarch64.dmg
```

Recipients on macOS need to clear Apple's "downloaded file" quarantine attribute once, after dragging the .app into Applications:

```bash
xattr -dr com.apple.quarantine /Applications/OpenPyTEA.app
```

First launch is 30–60 s while matplotlib builds its font cache; subsequent launches are fast.

### Next steps

- **Step 4 — CI release pipeline** (GitHub Actions matrix: macOS arm64 + Intel, Windows, Linux). Each runner needs Python, Node, Rust, then runs `scripts/build_sidecar.py` followed by `npm run tauri build`. Produces `.dmg` / `.msi` / `.deb` / `.AppImage` as Release assets on tag push. This is what unlocks Windows builds without owning a Windows machine.
- **Step 5 — polish.** Real icon set (currently Tauri's placeholder cup), the inner binary name (it's still `app` because Cargo crate is named `app` — `productName` only renames the .app bundle), optional auto-updater, signed installer (Apple Developer ID + Windows EV cert if/when budget allows).

### Known follow-ups, parked

- **Bundle size 152 MB.** `openpytea/__init__.py` eagerly imports `plotting.py` → forces `matplotlib` + `scienceplots` into the backend even though the backend never plots. Lazy-importing `openpytea.plotting` (a small library-side PR) would shave ~30–50 MB.
- **Matplotlib font cache rebuilds on every launch** — PyInstaller's runtime hook resets MPLCONFIGDIR. Adds 30–45 s to every cold start. Workaround: set `MPLCONFIGDIR` to a persistent user-dir path in the entrypoint.
- **Plant-name UX:** when a user adds a comparison plant in the GUI and gives it a custom name, that name flows through to Tornado/Sensitivity/MC legends now (`_rehydrate_extras` override added in `0b93cb1`). But the *active* plant's name still comes from `plant_config.plant_name` in the preset JSON, so legend labels can mix preset-defined and user-defined names.
- **Devtools in release** for easier in-browser-console debugging by recipients — `tauri = { features = ["devtools"] }` adds right-click-inspect, only worth it if we hit another opaque webview-side bug.

---

## Why Tauri (vs. the alternatives)

| Option | Bundle | Startup | Effort | Notes |
|---|---|---|---|---|
| **Tauri** (this plan) | ~5–15 MB shell + sidecar | <1s | medium-high | Native webview (WebView2 / WebKit / WebKitGTK). True native app. |
| Electron | 100–200 MB | 2–3s | medium | Ships its own Chromium. Mature ecosystem, large bundles. |
| pywebview | small | fast | low-medium | Pure Python; uses OS webview. Smaller community. |
| PyInstaller + browser | ~250 MB | 1–2s | low | Not a real desktop app — opens default browser at localhost. Simplest. |

Tauri wins on UX (real native window, fast startup, small shell). The cost is that it adds Rust to the toolchain and the Python sidecar is the hardest sub-problem.

If at any point Tauri proves too painful, the PyInstaller-as-launcher fallback gets us ~80% of the UX for 20% of the effort — see "Fallback plan" at the end.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  OpenPyTEA.app  (one icon, one process tree)             │
│                                                          │
│  ┌────────────────────────┐    ┌──────────────────────┐  │
│  │ Tauri shell (Rust)     │    │ Python sidecar       │  │
│  │  • native window       │◄──►│  • uvicorn          │  │
│  │  • native webview      │ http│  • FastAPI app      │  │
│  │  • lifecycle / IPC     │127.0│  • openpytea lib    │  │
│  │  • bundles React build │ .0.1│  • bundled by       │  │
│  │    in app resources    │     │    PyInstaller      │  │
│  └────────────────────────┘    └──────────────────────┘  │
│         ▲                                                │
│         │ loads index.html from disk                     │
│         │ (no Vite dev server, no localhost:5173)        │
└──────────────────────────────────────────────────────────┘
```

Lifecycle:

1. User double-clicks the app icon.
2. Tauri starts and spawns the Python sidecar on a kernel-assigned port (`--port 0`).
3. Sidecar prints its port on stdout; Tauri captures it.
4. Tauri opens the native webview pointing at the bundled `index.html`, with a small inline script setting `window.__OPENPYTEA_API__ = "http://127.0.0.1:<port>/api"`.
5. React mounts and starts making API calls to that port.
6. When the window closes, Tauri sends SIGTERM to the sidecar and waits for clean exit.

The sidecar binds to `127.0.0.1` only — no LAN exposure, no firewall prompts on macOS/Windows.

---

## What changes in the repo

**Stays the same** (no rewrites):
- `src/openpytea/` — the library
- `backend/app/` — FastAPI routers, schemas, business logic
- `frontend/src/` — all React components

**Modified:**
- `frontend/src/api/client.ts` — `BASE` reads from `window.__OPENPYTEA_API__` at runtime (port is dynamic) with the current `VITE_API_BASE_URL` env var as a dev-mode fallback. Means the same React build runs in dev (`npm run dev` + manual backend) and packaged.
- `backend/app/main.py` — accept `--port 0`, print the bound port on stdout once ready, bind only to `127.0.0.1`.

**New:**
- `src-tauri/` — Rust crate created via `npm create tauri-app`. Contains `tauri.conf.json` (window size, icons, bundle config, sidecar declaration), `Cargo.toml`, `src/main.rs` (sidecar-spawn logic).
- `scripts/build_sidecar.py` — invokes PyInstaller to produce a per-OS backend binary in `src-tauri/binaries/openpytea-backend-<target-triple>`.
- `.github/workflows/release.yml` — three-OS matrix build on tag push.

**Removed:**
- `start.sh` — not needed for end users. Keep it for developer convenience or delete; either is fine.

---

## The hard part: the Python sidecar

Bundling Python + numpy + scipy + pandas + matplotlib into a redistributable binary is the riskiest engineering work in this plan.

| Tool | Bundle size | Maturity | Notes |
|---|---|---|---|
| **PyInstaller** (recommended) | ~150–250 MB per OS | very mature | One binary per OS. Some hidden-import config needed for scipy/matplotlib. Standard choice. |
| PyOxidizer | ~100 MB | shaky | Embeds CPython directly. Painful with scipy native libs on macOS arm64. |
| Nuitka | ~150 MB | mature | Compiles Python to C; faster startup, harder to debug. |

**Plan: use PyInstaller in `--onefile` mode.** Boring, working, well-documented. Estimated final app bundle: **180–300 MB** depending on OS and how aggressive we get about excluding unused matplotlib backends.

Known gotchas to budget for:
- scipy's `_lib` and `_special` C extensions often need `--collect-binaries scipy` and `--collect-submodules scipy`.
- matplotlib drags in Tk/Qt/wx backends by default — explicitly exclude them.
- pandas pulls in `numexpr` and `bottleneck` opportunistically — fine to leave in.
- macOS arm64 vs x86_64 wheels — build matrix needs both, or ship a universal2 wheel.

---

## Per-OS build matrix

You cannot cross-build a real binary (no producing a working `.exe` from macOS, no `.dmg` from Windows). Need a runner per target.

| Target | CI runner | Output | Code signing |
|---|---|---|---|
| macOS Intel (x86_64) | `macos-13` | `.dmg` | Apple Developer ID ($99/yr) |
| macOS Apple Silicon (arm64) | `macos-14` | `.dmg` | same cert |
| Windows | `windows-latest` | `.msi` | EV cert (~$300/yr) or unsigned |
| Linux | `ubuntu-latest` | `.deb` + `.AppImage` | not required |

Without signing:
- **macOS**: first-run users see "OpenPyTEA can't be opened because it is from an unidentified developer" and have to right-click → Open. Acceptable for academic distribution; annoying for general public.
- **Windows**: Microsoft Defender SmartScreen warning. "More info → Run anyway."
- **Linux**: no signing infrastructure expected; fine.

---

## Sequenced plan (estimated effort)

Each step produces a checkpoint you can stop at if priorities change.

### Step 1 — Backend as a PyInstaller binary ✅ *done*
Produce `openpytea-backend` (or `.exe`) that runs the FastAPI app standalone with no Python on the host. Validate the hardest assumption first.

**Deliverable**: A binary you can double-click that starts uvicorn and serves the existing API. Useful on its own — could even ship as v0.1 of a standalone via the "PyInstaller fallback" path.

**Shipped:** commit `7bda17c`. 152 MB onedir at `dist/openpytea-backend/`. Built via `python scripts/build_sidecar.py`.

### Step 2 — Tauri shell loading the frontend ✅ *done*
Initialize `src-tauri/`, configure it to load the static `frontend/dist/` build. No backend yet. Confirms icon, window, build pipeline.

**Deliverable**: A window opens with the GUI but the API calls fail (no backend running). The plumbing is real.

**Shipped:** commit `95fe5cd`. Tauri 2.11, identifier `org.openpytea.app`. Verified `npm run tauri dev` opens the native window.

### Step 3 — Sidecar wiring ✅ *done (macOS arm64), shareable .dmg produced*
Tauri spawns the PyInstaller binary as a sidecar process. Captures the port from stdout. Injects it into the webview. Cleanly shuts down on window close. Handles the case where the sidecar dies unexpectedly (show an error dialog instead of a blank window).

**Deliverable**: Full app works end-to-end on your dev machine. Demo-able.

**Shipped:** initial wiring `291743c`, then a string of production fixes the moment the bundled `.app` was actually installed and tested:
- `9dd97de` sidecar timing race (announce port after uvicorn binds)
- `cab872b` drain backend stderr (uvicorn was blocking on a full pipe)
- `eb4c109` preset price fix (tornado was crashing on missing field)
- `0b93cb1` multi-plant tornado (parity with sensitivity / MC)
- `f501592` CORS for `tauri://localhost` + `isTauri()` detection + release-mode logging
- `ae1a49a` `multiprocessing.freeze_support()` so MC's worker spawns don't log errors
- `559a1b0` ad-hoc codesign the whole bundle (fixes "App is damaged" on recipients)

A `.dmg` produced via `npm run tauri build` has been confirmed installable and runnable on a second machine. Not yet verified on Windows / Linux (Step 4 territory). Graceful "sidecar died" UI is still on the polish list.

### Step 4 — CI release pipeline  *(2–3 days)*
GitHub Actions matrix building the three OSes on tag push. Outputs `.dmg`, `.msi`, `.deb`, `.AppImage` as release assets. This is fiddly — caching the PyInstaller venvs, matplotlib backend exclusion lists, signing the macOS bundle if certs exist.

**Deliverable**: `git tag v0.x.0 && git push --tags` produces a GitHub Release with installers for all three OSes.

### Step 5 — Polish  *(2–4 days)*
Icons in all required sizes/formats (`.icns` for macOS, `.ico` for Windows, `.png` set for Linux), app name strings, About dialog, optional auto-updater (Tauri has one), optional first-launch onboarding tweaks.

**Deliverable**: Looks and feels like a real product, not a developer-made app.

**Total: ~1.5–2 weeks of focused work** if nothing surprises us. Steps 3 and 4 are where surprises live.

---

## Decisions to make before we start

These shape the implementation; each has real cost/UX tradeoffs.

### 1. Code signing
- **(a) Ship signed** — buy Apple Developer ID ($99/yr) and a Windows EV cert (~$300/yr). Users get a smooth first-run. ~$400/yr ongoing.
- **(b) Ship unsigned with warning** — users right-click → Open on macOS, "More info → Run anyway" on Windows. Acceptable for v1, especially for an academic tool. Can add signing later.

### 2. Auto-updates
- **(a) Built-in updater** — Tauri has one. App checks a JSON manifest on a server (or GitHub Releases), prompts to install. Adds release-signing complexity.
- **(b) Manual redownload** — users grab the new installer from GitHub Releases when notified (release announcement, changelog). Simpler for v1.

### 3. User data persistence
Today the backend session is in-memory. Standalone-app users will expect saved projects to persist between launches.
- **(a) Per-OS app-data directory** — `~/Library/Application Support/OpenPyTEA/` (mac), `%APPDATA%/OpenPyTEA/` (win), `~/.local/share/openpytea/` (linux). Auto-load last session on startup.
- **(b) Rely on existing Save/Load JSON** — explicit user action, no implicit state. Matches the current model.

### 4. Branding
- App display name (OS shows this for the window, dock/taskbar, "About"): **OpenPyTEA**?
- Bundle identifier (must be reverse-DNS): e.g. `io.tudelft.openpytea` or `org.openpytea.app`?
- Icon — the existing logo upscaled, or commission a higher-res variant?

### 5. Bundle size budget
- **(a) Default** — ~250 MB installer per OS. Easiest.
- **(b) Aggressive trimming** — strip matplotlib backends we don't use, lazy-import pandas/scipy where possible. Maybe gets it down to ~150 MB. Hours-to-days of fiddling.

### 6. Distribution channel
- GitHub Releases is free and works. Suitable for v1.
- Later: macOS App Store (requires entitlements rework — not Tauri-friendly), Microsoft Store (possible but rare for research tools), Homebrew Cask (post-launch chore).

---

## Risks and unknowns

- **scipy native libs on macOS arm64** under PyInstaller — known to occasionally need extra `--collect-all scipy` flags. Budget half a day if it bites.
- **Sidecar cleanup on hard quit** (force-quit, OS shutdown) — Tauri's `on_window_event` should catch it, but worth testing on each OS.
- **CORS** — when running locally as a sidecar, CORS isn't an issue (same-origin via 127.0.0.1) but the backend currently allows everything; tighten to only `tauri://localhost` and `http://127.0.0.1` for the packaged build.
- **Antivirus false positives on Windows** — unsigned PyInstaller binaries occasionally get flagged. Signing dramatically reduces this.
- **Library updates** — every time we bump a Python dep (e.g. numpy 3.0), need to re-test the PyInstaller bundle. Adds release friction.

---

## Fallback plan: PyInstaller-as-launcher

If Tauri proves too painful or the sidecar wiring keeps misbehaving, the escape hatch is to skip Tauri entirely:

1. Use PyInstaller to package the FastAPI app *and* the pre-built React static files (FastAPI serves them).
2. On launch, the binary starts the server and runs `webbrowser.open("http://localhost:<port>")`.
3. App runs in the user's default browser. Not a real window, but a real installer.

Same bundle size, same "no Python needed" UX, no Rust toolchain, no Tauri sidecar machinery. We lose: native window chrome, dock icon behavior, system menu integration. We keep: working installers on all three OSes from a single Python codebase.

This is what we'd ship in week 1 of an emergency timeline. Worth keeping in our back pocket.

---

## Suggested next action

Step 1 (backend as PyInstaller binary) validates the hardest assumption and is useful on its own — both for the full Tauri path and the fallback. Doing that first is low-risk: if it works, both paths are open; if it doesn't, we learn the scipy/matplotlib bundling pain before investing in Tauri-specific work.
