Installation
============

Requirements
------------

OpenPyTEA requires **Python 3.10 or later**.

Install from PyPI
-----------------

The recommended way to install OpenPyTEA is via ``pip``:

.. code-block:: bash

   pip install OpenPyTEA

Clone from source
-----------------

To work with the notebooks or contribute to development, clone the repository.
We recommend `uv <https://github.com/astral-sh/uv>`_ for this workflow — it
locks the full environment so the virtual environment can be exactly reproduced:

.. code-block:: bash

   git clone https://github.com/PBTamarona/OpenPyTEA.git
   cd OpenPyTEA
   uv sync

``uv sync`` reads the lockfile and creates a ``.venv`` with all dependencies
pinned, making it easy to share and replicate the environment across machines.

If you prefer plain pip instead:

.. code-block:: bash

   git clone https://github.com/PBTamarona/OpenPyTEA.git
   cd OpenPyTEA
   pip install -e .

Dependencies
------------

OpenPyTEA automatically installs all required dependencies:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Package
     - Purpose
   * - ``numpy >= 1.24``
     - Numerical computations and array operations
   * - ``pandas >= 1.5``
     - Tabular data handling (cost databases, results)
   * - ``matplotlib >= 3.8``
     - Plotting and visualization
   * - ``scienceplots >= 2.2``
     - Publication-quality figure styling
   * - ``scipy >= 1.10``
     - Optimization (IRR solver) and statistical distributions
   * - ``seaborn >= 0.12``
     - Statistical visualization (Monte Carlo plots)
   * - ``tqdm >= 4.64``
     - Progress bars for Monte Carlo simulations
   * - ``jinja2 >= 3.1``
     - Template rendering for result reports

Optional extras
---------------

.. code-block:: bash

   # Jupyter / IPython kernel support
   pip install "OpenPyTEA[ipython]"

   # Development / testing
   pip install "OpenPyTEA[test]"

Verify installation
-------------------

.. code-block:: python

   import openpytea
   print(openpytea.__version__)

Web GUI (Work in progress)
------------------

The interactive web GUI requires additional setup. See :doc:`gui` for full
instructions. **Please note, the GUI is still in active development and not 
all features are available yet.**

.. code-block:: bash

   # Backend (FastAPI)
   cd backend
   pip install -r requirements.txt
   uvicorn app.main:app --reload

   # Frontend (React + Vite) — in a separate terminal
   cd frontend
   npm install
   npm run dev
