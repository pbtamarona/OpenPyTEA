# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the OpenPyTEA standalone backend.

Build (onedir mode, default):

    pyinstaller backend/openpytea-backend.spec --clean --noconfirm

Output: dist/openpytea-backend/   (a directory; entry binary inside it.)

The spec is the source of truth for what gets bundled. CLI flags are
intentionally avoided so the build is reproducible across machines.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# --- data files -----------------------------------------------------------
# Non-Python files PyInstaller does not pick up by default.
datas = []
datas += collect_data_files("openpytea")        # cost_correlations.csv, cepci_values.csv
datas += collect_data_files("scienceplots")     # matplotlib style files used at import time
datas += [("app/presets", "app/presets")]       # example preset JSONs

# --- hidden imports -------------------------------------------------------
# Modules dynamically imported by name; PyInstaller's static analyser
# does not see these and will skip them without explicit listing.
hiddenimports = []
hiddenimports += collect_submodules("scipy")
hiddenimports += collect_submodules("openpytea")
hiddenimports += [
    # uvicorn picks its loop/protocol modules at runtime via auto-detection;
    # PyInstaller misses them without the explicit list.
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.loops.uvloop",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
]

# --- excludes -------------------------------------------------------------
# Heavy or irrelevant packages we know we don't ship. Stripping these
# matters for bundle size on the order of tens of MB each.
excludes = [
    "tkinter",
    "PyQt5", "PyQt6", "PySide2", "PySide6",
    "IPython", "jupyter", "notebook", "ipykernel",
    "matplotlib.tests", "numpy.tests", "scipy.tests", "pandas.tests",
    "sphinx", "pytest",
]


a = Analysis(
    ["openpytea_backend.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="openpytea-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="openpytea-backend",
)
