Examples & Case Studies
========================

OpenPyTEA ships with three fully worked case studies in the ``examples/``
directory of the repository. Each is a self-contained Jupyter notebook that
demonstrates the library's capabilities on a real-world engineering scenario.

Running the examples
--------------------

.. code-block:: bash

   git clone https://github.com/PBTamarona/OpenPyTEA.git
   cd OpenPyTEA
   pip install "OpenPyTEA[ipython]"
   jupyter notebook examples/

Case Study 1 — Hydrogen Production Pathways
--------------------------------------------

**File**: ``examples/case_study_1.ipynb``

Compares the techno-economics of three hydrogen production routes:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Scenario
     - Technology
   * - SMR
     - Steam Methane Reforming (incumbent)
   * - Pyrolysis
     - Methane Pyrolysis (solid carbon by-product)
   * - Electrolysis
     - Water Electrolysis (green hydrogen)

Location: **US Gulf Coast**. The notebook covers equipment selection, CAPEX
and OPEX breakdowns, sensitivity to natural gas and electricity prices, and a
side-by-side Monte Carlo comparison of the levelized cost of hydrogen (LCOH)
across all three pathways.

Key topics demonstrated
~~~~~~~~~~~~~~~~~~~~~~~

* Creating multiple :class:`~openpytea.equipment.Equipment` objects per scenario
* Running the same analysis on three :class:`~openpytea.plant.Plant` instances
* Using :func:`~openpytea.plotting.plot_multiple_monte_carlo` for cross-scenario comparison
* Interpreting tornado diagrams to identify cost drivers

Case Study 2 — Hydrogen Liquefaction Precooling
-------------------------------------------------

**File**: ``examples/case_study_2.ipynb``

Techno-economic assessment of a **hydrogen liquefaction** plant with
precooling train in the **Netherlands**, featuring 11 process equipment items.

Key topics demonstrated
~~~~~~~~~~~~~~~~~~~~~~~

* Handling large equipment lists with mixed categories (heat exchangers,
  compressors, expanders, vessels)
* European location factor application
* Multi-year CAPEX ramp and production ramp-up
* Sensitivity of the LCOH to the compressor electricity price

Case Study 3 — Geothermal Energy Systems
-----------------------------------------

**File**: ``examples/case_study_3.ipynb``

Compares two geothermal applications:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Scenario
     - System
   * - District heating
     - Heat pump for residential heating
   * - Power generation
     - Organic Rankine Cycle (ORC)

Key features: 30-year project lifetime, full MACRS depreciation, and
Monte Carlo uncertainty analysis for both scenarios.

Key topics demonstrated
~~~~~~~~~~~~~~~~~~~~~~~

* Long-lifetime project economics (30-year cash flows)
* MACRS depreciation configuration
* Levelized cost of heat (LCOH) vs. levelized cost of electricity (LCOE)
* Geothermal resource uncertainty via Monte Carlo

Walkthrough notebook
---------------------

**File**: ``walkthrough.ipynb`` (root of repository)

A comprehensive introductory notebook covering every main feature of
OpenPyTEA step by step — ideal for first-time users.

Tutorial videos
---------------

Video walkthroughs covering the Python API, the web GUI, and the JSON
workflow are available on YouTube. See the `README
<https://github.com/PBTamarona/OpenPyTEA#tutorials>`_ for links.
