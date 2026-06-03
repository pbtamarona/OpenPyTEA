Installation
============

Requirements
------------

OpenPyTEA requires **Python 3.10 or later**.

Install from PyPI
-----------------

The recommended way to install OpenPyTEA is from PyPI:

.. tab-set::

   .. tab-item:: pip

      .. code-block:: bash

         pip install OpenPyTEA

   .. tab-item:: uv (recommended)

      `uv <https://github.com/astral-sh/uv>`_ is a fast Python package manager.
      Install it first, then:

      .. code-block:: bash

         uv pip install OpenPyTEA

      Or add it as a project dependency:

      .. code-block:: bash

         uv add OpenPyTEA

Install from source
-------------------

To get the latest development version:

.. code-block:: bash

   git clone https://github.com/PBTamarona/OpenPyTEA.git
   cd OpenPyTEA
   pip install -e .

Or with uv:

.. code-block:: bash

   git clone https://github.com/PBTamarona/OpenPyTEA.git
   cd OpenPyTEA
   uv sync

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

Web GUI (optional)
------------------

The interactive web GUI requires additional setup. See :doc:`gui` for full
instructions.

.. code-block:: bash

   # Backend (FastAPI)
   cd backend
   pip install -r requirements.txt
   uvicorn app.main:app --reload

   # Frontend (React + Vite) — in a separate terminal
   cd frontend
   npm install
   npm run dev
