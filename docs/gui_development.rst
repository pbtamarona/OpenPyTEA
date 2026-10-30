GUI Development
===============

This page is for working on the GUI itself: how it is put together, and
how the desktop installers are built and released. For using the GUI, see
:doc:`gui`, which also explains how to run it from source.

Architecture
------------

The GUI is a client-server application around the OpenPyTEA library. A
FastAPI backend exposes the library as a REST API; a React frontend renders
a tabbed single-page app. The desktop app wraps both in a Tauri window.

Stack
^^^^^

- **Backend:** FastAPI (Python), port 8000 in development.
- **Frontend:** React 19 + TypeScript + Vite, port 5173 in development.
- **Charts:** Recharts for the interactive view; downloads are matplotlib
  figures rendered by the backend (``/api/plots/*``) with the library's own
  ``plotting.py``, identical to the Jupyter output.
- **Styling:** plain CSS in ``App.css`` (light and dark mode).
- **State:** in memory on the backend (module-level singletons in
  ``state.py``). No database; one user per backend.
- **Desktop shell:** Tauri v2 (Rust). It starts the backend as a bundled
  PyInstaller executable (the *sidecar*), which picks a free port and
  prints ``OPENPYTEA_BACKEND_PORT=<port>``; the shell passes that port to
  the frontend. The shell also provides the native menu, the ``.openpytea``
  file association, the unsaved-work prompt on close and the docs window.

Directory layout
^^^^^^^^^^^^^^^^

.. code-block:: text

   backend/
     app/
       main.py            FastAPI app, CORS, router mounting
       state.py           session state: equipment_list, plant, plant_config, results, MC runs
       project_store.py   multi-plant bookkeeping around the single-plant session
       plant_factory.py   builds Plant objects from saved JSON or session state
       schemas.py         Pydantic request/response models for every endpoint
       util.py            to_jsonable(): numpy → JSON-native converter
       presets/           example projects (one JSON file each)
       routers/           equipment, plant, analysis, plots, io, project
     openpytea_backend.py     PyInstaller entry point (the sidecar)
     openpytea-backend.spec   PyInstaller spec

   frontend/
     src/
       api/client.ts      typed fetch wrapper, one function per endpoint
       types/index.ts     TypeScript mirrors of the backend schemas
       components/        DownloadableChart, HelpTip, FactorChip, RampEditor,
                          UncertaintyEditor, PlantPicker, ComparePlants, ...
       pages/             one component per tab
       help/              (?) texts: help.ts + docHelp.json (generated)
       format.ts          number and label formatting
       App.tsx            tabs, header, plant switcher, file handling
     src-tauri/           Tauri shell (src/lib.rs, tauri.conf.json, icons)

   scripts/
     build_sidecar.py     PyInstaller build + smoke test of the backend
     build_docs.py        builds these docs into frontend/public/docs
     patch_macos_plist.py macOS file-type registration + re-sign
     set_version.py       stamps a release tag's version into the app

Data flow
^^^^^^^^^

1. **Equipment** is created on the backend through the equipment
   endpoints. Each item is built immediately as an ``Equipment`` or
   ``CompositeEquipment``, so the cost correlation and inflation
   adjustment run before the response. Each object keeps its sanitized
   input on ``_input_spec``; saving writes those specs, so loading rebuilds
   from the user's inputs (resolved ``num_units`` or adjusted quotes are
   never saved as inputs).
2. **Plant config** is a JSON dict in ``state.plant_config``, set through
   ``PUT /api/plant/config``. Keys without a form are passed through to
   the library unchanged.
3. **Calculate** (``POST /api/plant/calculate``) builds ``Plant(config)``
   from the config and equipment list, runs ``calculate_all()``, stores the
   plant in ``state.plant`` and returns the results.
4. **Analysis** endpoints run the library's analysis functions on the
   stored plant, and on other plants of the project rebuilt from their
   snapshots (``extra_plants``).
5. **Monte Carlo** results are histogrammed on the backend (about 80 bins
   plus summary statistics, about 10 KB instead of about 40 MB); the raw
   samples of the last run stay in memory for the figure downloads.

Key design decisions
^^^^^^^^^^^^^^^^^^^^

1. **Recharts on screen, matplotlib on download.** The download button of
   a chart fetches ``/api/plots/...``, where the backend runs the library's
   plotting function (Agg backend, one figure at a time behind a lock) and
   returns a 300-dpi PNG. Charts without a library counterpart (revenue)
   download a rendering of the screen chart.
2. **Multi-plant project, single open plant.** The session fields always
   hold the *open* plant, so the plant-level routers don't know about the
   project. ``project_store`` parks the open plant (input specs, config,
   results) before switching and rebuilds the target. Save files are
   format version 2 (``project``, ``plants``, ``active_plant``) and also
   mirror the open plant at the top level, so older builds still open
   them; version 1 files load as one-plant projects.
3. **Validated responses.** Every endpoint declares a ``response_model``,
   so FastAPI checks each response against the Pydantic schemas.
4. **No silent errors.** Every frontend API call reports failures in the
   error bar at the top of the page.
5. **Examples are data.** Each preset in ``backend/app/presets/`` is a
   complete project; adding a ``.json`` file there adds an example, no
   code change needed.

API
---

The backend serves an interactive API reference, generated from the
routes and schemas, at http://localhost:8000/docs while it runs. The
routers:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Prefix
     - Covers
   * - ``/api/equipment``
     - Equipment list (add, update, delete, import/export of the equipment
       input file) and the cost database, process types and materials.
   * - ``/api/plant``
     - Plant config, calculate, import/export of the plant input file,
       location factors and process types.
   * - ``/api/analysis``
     - Sensitivity (and its parameter list), tornado, Monte Carlo, and
       import of the analysis input file.
   * - ``/api/plots``
     - Library-rendered figures for the downloads (stacked bars, Compare
       charts, sensitivity, tornado, Monte Carlo).
   * - ``/api/project``
     - Project details and plants (add, open, duplicate, rename, delete,
       snapshot), save/load, ``run_tea`` result files, and examples.
   * - ``/api/health``
     - Health check.

Mutating project endpoints return the new project overview plus the open
plant's results, so the Results tab follows a plant switch.

Tabs in detail
--------------

Project
   Project details (name, user, date created, description), saved on blur,
   and the plants table with metrics and Open / Rename / Duplicate /
   Delete. The header shows a plant switcher when the project has several
   plants, except on tabs that have their own picker or show every plant.

Equipment
   The equipment table with totals and an add/edit dialog: category and
   type from the cost database (each type pinned by its correlation
   ``key`` sent as ``cost_func``, since several correlations can share a
   category and type), a second size input for two-variable correlations,
   material ("Auto" lets the library choose), number of units (blank lets
   the library parallelize; the user's value is kept separately as
   ``num_units_input``), and a direct-cost mode. *Composite equipment*
   builds a ``CompositeEquipment`` from components, with the installation
   rule and an optional vendor quote; the backend accepts nesting up to
   three levels from JSON. Below the table, the direct cost breakdown chart
   shows each equipment or each category (``direct_costs_data`` with
   ``by_category``).

Plant Config
   Plant picker, then cards for general settings, financial parameters,
   labor, additional capital costs (amount and year), products, variable
   OPEX and parameter dependencies. Products and variable OPEX items are
   renamed by clicking their name. Every price and quantity has a Monte
   Carlo uncertainty button opening ``UncertaintyEditor``, which offers each
   ``dist_id`` family with only the fields it reads (truncation bounds only
   for Normal, Lognormal and Bernoulli; a Lognormal μ must be explicit).
   Uncertainties are saved in the 3.0 sub-dicts (``price_uncertainty``,
   ``consumption_uncertainty``, ...). The dependencies card edits rules
   ``dependent = Σ weight × parent + offset`` stored in the plant config,
   so every analysis honours them; a dependent's uncertainty is only
   zero-centred ``noise``, and absolute bounds are removed from it.

Results
   Metric cards; capital cost, fixed OPEX, variable OPEX and revenue tables
   and charts. Cost factors (f_c, OSBL, D&E, contingency, working capital,
   the fixed OPEX factors, f_p) are click-to-edit, with (i) tips showing
   each equation; amounts can be set directly, overriding their factor
   through the library's component overrides. The cash flow card has the
   construction and start-up schedule editor (``capex_ramp``,
   ``production_ramp``).

Analysis
   Sensitivity panels (parameter, metric, ±variation, x axis as % change or
   actual value) and the tornado (``category``: economic, process, or
   both). On the value axis, a plant without the parameter is a flat line
   at its baseline, as in ``plot_sensitivity``. Plant checkboxes choose the
   plants to run and show; colours follow a plant's place in the project.

Monte Carlo
   Samples, batch size and plant checkboxes; statistics table, histograms
   and the input distributions.

Compare
   Every calculated plant of the project, plus plants imported from saved
   files (kept in the project as ``compared_plants``); checkboxes choose
   what the table and charts show.

All breakdown charts have a Value / % toggle, and charts with a grid have a
grid toggle. Names typed with ``_2`` or ``^3`` show as subscripts and
superscripts, as in the library's plots.

Documentation and (?) help
--------------------------

- ``scripts/build_docs.py`` builds these Sphinx docs into
  ``frontend/public/docs/`` (git-ignored), which Vite serves at ``/docs/``
  and the app bundles. MathJax and DataTables are downloaded into the build
  so the docs work offline in the app.
- In the desktop app, **OpenPyTEA** ▸ **OpenPyTEA Documentation** opens a
  separate docs window (external links open in the browser); in browser
  development the header logo has the same item. Both go through
  ``openDocs(path)`` in ``src/docs.ts``.
- The **(?)** tips (``HelpTip.tsx``) take their text from
  ``src/help/help.ts``. Entries keyed after config keys
  (``plant.interest_rate``, ``equipment.param``, ``metric.NPV``, ...) are
  extracted from the user guide's reference tables into
  ``src/help/docHelp.json`` by ``python scripts/build_docs.py --help-only``
  (committed; rerun it after editing those tables). Texts for GUI-only
  inputs are written in ``help.ts``.

Example presets
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - ID
     - Title
     - Source
   * - ``h2_smr``
     - Hydrogen – Steam Methane Reforming
     - Case study 1 (14 equipment)
   * - ``h2_methane_pyrolysis``
     - Hydrogen – Methane Pyrolysis
     - Case study 1 (13 equipment, carbon co-product)
   * - ``h2_electrolysis``
     - Hydrogen – Water Electrolysis
     - Case study 1 (3 equipment, direct costs)
   * - ``lh2_smr_best``
     - Liquid H2 – SMR Best Case
     - Case study 2 (11 equipment)
   * - ``geothermal_heat_pump``
     - Geothermal Heat Pump
     - Case study 3 (8 equipment)
   * - ``geothermal_power_plant``
     - Geothermal Power Plant
     - Case study 3 (7 equipment)

Known limitations
-----------------

- No editor for the depreciation method (the config key is kept).
- Monte Carlo runs synchronously, so large sample counts block the API
  until they finish.
- Form inputs are validated by the backend only (errors show in the error
  bar).
- The cash flow table shows the first scenario only.

Building the installers
-----------------------

Releases (GitHub Actions)
^^^^^^^^^^^^^^^^^^^^^^^^^

``.github/workflows/build-desktop.yml`` builds the Windows and macOS
installers and attaches them to a GitHub release:

- **Release tags** ``vX.Y.Z`` and ``vX.Y.ZrcN``. The app takes the
  library's version: ``scripts/set_version.py`` stamps the tag into the
  app (``3.2.2rc1`` becomes ``3.2.2-rc.1``, the form Tauri requires).
  Release candidates build only the ``.exe`` on Windows, because the
  ``.msi`` accepts numeric pre-release versions only.
- **Publishing a release on GitHub** creates the tag, which starts the
  build; about 10 minutes later the installers are added to that release.
  The same release also starts the PyPI upload
  (``python-publish.yml``): pre-releases go to TestPyPI, releases to
  PyPI. The uploaded version is the one in ``pyproject.toml``, so set it
  (``3.2.2rc1``, ``3.2.2``) before publishing.
- **Pushing a tag without a release** leaves a draft release with the
  installers; publishing that draft yourself still starts the PyPI upload.
- **desktop-latest**: running the workflow by hand (Actions ▸ Build
  desktop apps ▸ Run workflow) builds the current ``main`` into the
  rolling ``desktop-latest`` pre-release. Builds on every push to ``main``
  are switched off so release builds don't queue behind them; the
  workflow file shows how to switch them back on.

Building locally
^^^^^^^^^^^^^^^^

Installers can only be built on their own platform (Windows installers on
Windows, macOS on macOS). Prerequisites: Python 3.10+, Node.js, and the
Rust toolchain with Tauri's
`system prerequisites <https://v2.tauri.app/start/prerequisites/>`_.
From the repository root:

.. code-block:: bash

   # 1. Python environment with the library, backend and PyInstaller
   python -m venv .venv
   # activate it: source .venv/bin/activate (macOS) or .venv\Scripts\activate (Windows)
   pip install -e . -r backend/requirements.txt pyinstaller

   # 2. The backend sidecar (PyInstaller, then a smoke test against /api/health)
   python scripts/build_sidecar.py

   # 3. The bundled docs
   pip install -r docs/requirements.txt
   python scripts/build_docs.py

   # 4. The Tauri app and installers
   cd frontend
   npm ci
   npm run tauri build

   # 5. macOS only: register the .openpytea file type, re-sign, rebuild the .dmg
   cd ..
   python scripts/patch_macos_plist.py

The installers end up in ``frontend/src-tauri/target/release/bundle/``
(``nsis/`` and ``msi/`` on Windows, ``dmg/`` and ``macos/`` on macOS). The
apps are not code-signed: signing needs an Apple Developer ID and a
Windows code-signing certificate, neither of which is set up.
