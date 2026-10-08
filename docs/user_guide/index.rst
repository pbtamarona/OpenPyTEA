:orphan:

User Guide
==========

This section provides in-depth documentation for every feature of OpenPyTEA.
Follow the pages in order for a structured introduction, or jump directly to
the topic you need.

Walkthrough notebooks
---------------------

The `OpenPyTEA walkthrough <https://github.com/pbtamarona/OpenPyTEA/tree/main/walkthrough>`_ is a
series of five Jupyter notebooks that covers the full workflow end-to-end —
equipment costing, plant configuration, financial calculations, cost
breakdowns and sensitivity analysis, Monte Carlo uncertainty analysis, and
the JSON-based workflow — with live code, outputs, and explanatory text.
It is the recommended starting point before reading the individual guide pages.
Each guide page below links to the part of the walkthrough that shows its
examples in action.

.. toctree::
   :maxdepth: 1

   equipment
   plant
   analysis
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
