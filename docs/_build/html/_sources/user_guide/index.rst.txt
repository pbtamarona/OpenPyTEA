User Guide
==========

This section provides in-depth documentation for every feature of OpenPyTEA.
Follow the pages in order for a structured introduction, or jump directly to
the topic you need.

.. toctree::
   :maxdepth: 1

   equipment
   plant
   analysis
   plotting
   io_workflow

Overview of the workflow
------------------------

A typical OpenPyTEA study follows five steps:

.. code-block:: text

   1. Equipment costing  →  Equipment objects with purchased/direct costs
   2. Plant configuration →  Costs, financial parameters, products, OPEX
   3. Calculate           →  NPV, IRR, ROI, levelized cost, cash flows
   4. Analyse             →  Sensitivity, tornado, Monte Carlo
   5. Export              →  JSON results, publication-quality figures

Each step maps to a module:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Module
     - Key objects
     - Description
   * - :mod:`openpytea.equipment`
     - :class:`~openpytea.equipment.Equipment`
     - Equipment-level cost estimation with CEPCI inflation adjustment
   * - :mod:`openpytea.plant`
     - :class:`~openpytea.plant.Plant`
     - Plant-level CAPEX, OPEX, and financial analysis
   * - :mod:`openpytea.analysis`
     - Functions
     - Sensitivity, tornado, and Monte Carlo analysis
   * - :mod:`openpytea.plotting`
     - Functions
     - Matplotlib-based publication-quality visualizations
   * - :mod:`openpytea.io`
     - Functions
     - JSON configuration loading and result export
