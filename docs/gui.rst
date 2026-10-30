Graphical User Interface
========================

OpenPyTEA includes a graphical user interface (GUI) that lets you perform
a complete techno-economic assessment — equipment costing, plant
configuration, results, sensitivity/tornado analysis, Monte Carlo, and
multi-plant comparison — without writing any Python code. Every chart can
be downloaded as the same figure the library draws in Jupyter.

Installing
----------

The desktop app is self-contained: no Python or Node.js required.
Download the installer for your platform from the
`Releases page <https://github.com/pbtamarona/OpenPyTEA/releases>`_
(each OpenPyTEA release from 3.2.2 on comes with installers of the same
version):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Platform
     - File
   * - Windows (x64)
     - ``OpenPyTEA_<version>_x64-setup.exe`` (or the ``.msi``)
   * - macOS (Apple Silicon)
     - ``OpenPyTEA_<version>_aarch64.dmg``

The app isn't code-signed yet, so the operating system warns on first
launch:

- **Windows:** run the ``.exe`` and follow the setup wizard. If SmartScreen
  shows *"Windows protected your PC"*, click **More info** → **Run anyway**.
- **macOS:** open the ``.dmg`` and drag OpenPyTEA into **Applications**.
  Before the first launch, clear the "downloaded from the internet" flag
  once in Terminal:

  .. code-block:: bash

     xattr -dr com.apple.quarantine /Applications/OpenPyTEA.app

The first launch takes 30–45 s while matplotlib builds its font cache;
later launches are fast.

Double-clicking a ``.openpytea`` project file opens it in the app. On
macOS, if OpenPyTEA doesn't show up under *Open With*, refresh Launch
Services once:

.. code-block:: bash

   /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f /Applications/OpenPyTEA.app
   killall Finder

Using the app
-------------

The tabs follow the TEA workflow:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Tab
     - What it does
   * - Project
     - Project details and the list of plants (open, rename, duplicate,
       delete, add).
   * - Plant Config
     - Location, financial parameters, labor, additional capital costs,
       products, variable OPEX, Monte Carlo uncertainties and parameter
       dependencies.
   * - Equipment
     - Equipment list with cost-database lookup, composite equipment, and a
       direct cost breakdown by equipment or by category.
   * - Results
     - Metrics, capital and OPEX breakdowns with editable cost factors and
       amounts, revenue, and the cash flow with an editable construction
       and production ramp-up.
   * - Analysis
     - Sensitivity plots (several panels, several plants) and tornado
       diagrams.
   * - Monte Carlo
     - Uncertainty analysis with histograms and summary statistics, for one
       or several plants.
   * - Compare
     - Side-by-side capital cost, OPEX, levelized cost, NPV and IRR of the
       project's plants.

A typical session:

1. **Examples** → pick a case study (or build a plant in the Equipment and
   Plant Config tabs).
2. **Results** → **Run Calculations**.
3. Add more plants (**Examples** again, or **+ Add Plant** on the Project
   tab).
4. **Analysis** / **Monte Carlo** → tick the plants to overlay, then run.
5. **Compare** → see all calculated plants side by side.
6. **File** ▸ **Save As…** to keep the whole project in a ``.openpytea``
   file.

The **File** menu (``Ctrl`` on Windows, ``⌘`` on macOS):

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Item
     - Shortcut
     - Does
   * - New Project
     - ``Ctrl+N``
     - Start an empty project.
   * - Open Project…
     - ``Ctrl+O``
     - Open a ``.openpytea`` file.
   * - Save / Save As…
     - ``Ctrl+S`` / ``Ctrl+Shift+S``
     - Save the project.
   * - Save as JSON Files…
     - ``Ctrl+Shift+E``
     - Write the three ``run_tea`` result files of the open plant.

Closing the app with unsaved work asks whether to save. The **(?)** next to
an input explains it and links to the matching section of these docs;
**OpenPyTEA** ▸ **OpenPyTEA Documentation** opens the docs in their own
window, offline.

Library JSON files per tab
^^^^^^^^^^^^^^^^^^^^^^^^^^

The library's file workflow (:doc:`user_guide/io_workflow`: equipment,
plant and analysis input files in, three result files out) is available
tab by tab:

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Tab
     - Import
     - Save
   * - Equipment
     - Equipment input (asks Replace / Add when the plant has equipment)
     - Equipment input, ``<plant>_equipment_results.json``
   * - Plant Config
     - Plant input (replaces the open plant's settings after a confirm;
       equipment untouched)
     - Plant input
   * - Results
     - —
     - ``<plant>_plant_results.json``
   * - Analysis
     - Analysis input → sensitivity panels and tornado form
     - ``<plant>_analysis_results.json``
   * - Monte Carlo
     - Analysis input → samples and batch size
     - ``<plant>_analysis_results.json``

Imports only fill in settings; nothing is run. Only ``"run": true`` blocks
are read, as ``run_tea`` does. What the GUI can't follow (for example
``random_seed``, out-of-range values, or parameters the plant doesn't have)
is listed in the notice bar. Plant keys without a form (``loc_factor``, …)
and equipment installation-factor overrides are kept through edits, save
and export.

Running from source
-------------------

To run the GUI from a clone of the repository (Python 3.10+ and Node.js
needed), start the backend and the frontend in two terminals from the
repository root:

.. code-block:: bash

   # Terminal 1: backend
   pip install -e . -r backend/requirements.txt
   cd backend
   PYTHONPATH=../src python -m uvicorn app.main:app --reload --port 8000

   # Terminal 2: frontend
   cd frontend
   npm install
   npm run dev

On Windows PowerShell, set the path with ``$env:PYTHONPATH="..\src";``
before ``python -m uvicorn ...``. Then open http://localhost:5173. On macOS
and Linux, ``./start.sh`` does all of this in one go (it creates ``.venv``
and installs everything on the first run).

How the GUI is built, and how to build the installers, is described in
:doc:`gui_development`.
