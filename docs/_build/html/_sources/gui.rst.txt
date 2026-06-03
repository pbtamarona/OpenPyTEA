Web GUI
=======

OpenPyTEA includes a browser-based graphical user interface (GUI) that
lets you perform a complete techno-economic assessment without writing any
Python code. The GUI consists of a **FastAPI** backend and a **React +
TypeScript** frontend.

Architecture
------------

.. code-block:: text

   Browser (React/Vite on port 5173)
         ↕  REST API (JSON)
   FastAPI backend (port 8000)
         ↕  Python objects
   openpytea library

Starting the GUI
----------------

**Step 1 — Start the backend:**

.. code-block:: bash

   cd backend
   pip install -r requirements.txt
   uvicorn app.main:app --reload

The API will be available at ``http://localhost:8000``.
Interactive API docs (Swagger UI) are at ``http://localhost:8000/docs``.

**Step 2 — Start the frontend** (new terminal):

.. code-block:: bash

   cd frontend
   npm install
   npm run dev

Open ``http://localhost:5173`` in your browser.

Tabs and workflow
-----------------

The GUI is organized into five tabs that follow the TEA workflow:

Equipment tab
~~~~~~~~~~~~~

* Add equipment items using dropdowns for category, type, and material
* Enter the size/capacity parameter or supply a direct cost override
* View purchased and installed (direct) costs in a live table
* Delete or edit any item

Plant Config tab
~~~~~~~~~~~~~~~~~

* Set general parameters: plant name, process type, location, currency
* Configure financial assumptions: interest rate, lifetime, tax rate
* Add products and variable OPEX inputs (with optional uncertainty bounds
  for Monte Carlo)
* Configure labor parameters

Results tab
~~~~~~~~~~~

* Click **Run Calculations** to compute all costs and financial metrics
* View metric cards: Levelized Cost, NPV, IRR, ROI, Payback Time
* Inspect interactive stacked-bar charts for CAPEX, fixed OPEX, variable
  OPEX, and revenue
* Browse the year-by-year cash flow table
* Download any chart as a PNG

Analysis tab
~~~~~~~~~~~~~

* **Sensitivity**: select a parameter, choose a metric, set ±variation →
  generates a line chart
* **Tornado**: rank all parameters by ±impact on the selected metric

Monte Carlo tab
~~~~~~~~~~~~~~~

* Set the number of samples and batch size
* Run the simulation and view summary statistics (mean, std, percentiles)
* Download histograms for each financial metric

Saving and loading projects
---------------------------

Use the **Save** and **Load** buttons in the header to export and import
the complete project state as a JSON file. The format is identical to the
:doc:`user_guide/io_workflow` JSON files, so you can load a GUI-saved file
in Python or vice versa.

Example presets
---------------

The **Examples** dropdown loads one of the built-in case studies (hydrogen
production pathways, liquefaction, geothermal) to demonstrate the GUI
workflow with realistic data.

REST API reference
------------------

The FastAPI backend exposes a documented REST API. With the backend running,
visit ``http://localhost:8000/docs`` for the interactive Swagger UI or
``http://localhost:8000/redoc`` for the ReDoc interface.

Key endpoint groups:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Prefix
     - Description
   * - ``/api/equipment/``
     - CRUD operations on the equipment list
   * - ``/api/plant/``
     - Plant configuration and calculation
   * - ``/api/analysis/``
     - Sensitivity, tornado, and Monte Carlo
   * - ``/api/project/``
     - Save, load, and example presets
   * - ``GET /api/health``
     - Health check
