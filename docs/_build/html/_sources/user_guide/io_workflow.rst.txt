JSON Workflow
=============

OpenPyTEA supports a fully declarative, JSON-based workflow that makes studies
**reproducible**, **shareable**, and easy to version-control.
The :mod:`openpytea.io` module handles loading configuration files and
exporting results.

Why use JSON configs?
---------------------

* Separate configuration from code — rerun the same study with different
  parameters without editing Python
* Store equipment databases, plant assumptions, and analysis settings in
  human-readable files
* Compatible with the web GUI's save/load feature

File structure
--------------

A complete TEA study uses three JSON files:

.. code-block:: text

   project/
   ├── equipment.json      # List of equipment items
   ├── plant.json          # Plant configuration and financial assumptions
   └── analysis.json       # Sensitivity/Monte Carlo settings

``equipment.json``
-------------------

.. code-block:: json

   {
     "metadata": {
       "version": "2.1.0",
       "description": "Ammonia plant equipment"
     },
     "equipment": [
       {
         "name": "COMP-01",
         "process_type": "Fluids",
         "category": "Compressors (Centrifugal)",
         "type": "Centrifugal",
         "material": "Carbon steel",
         "param": 5000
       },
       {
         "name": "HX-01",
         "process_type": "Fluids",
         "category": "Heat exchangers (Shell and tube)",
         "type": "Fixed head",
         "material": "Stainless steel 316",
         "param": 250
       }
     ]
   }

``plant.json``
---------------

.. code-block:: json

   {
     "metadata": {
       "version": "2.1.0"
     },
     "plant": {
       "plant_name": "Green Ammonia Plant",
       "process_type": "Fluids",
       "country": "Netherlands",
       "interest_rate": 0.09,
       "project_lifetime": 20,
       "tax_rate": 0.25,
       "capex_ramp": [0.3, 0.6, 0.1],
       "production_ramp": [0, 0, 0, 0.5, 0.8],
       "plant_products": {
         "ammonia": {"production": 125000, "price": 500}
       },
       "variable_opex_inputs": {
         "electricity": {"consumption": 110, "price": 75}
       }
     }
   }

``analysis.json``
------------------

.. code-block:: json

   {
     "sensitivity": {
       "parameters": ["electricity", "interest_rate", "plant_utilization"],
       "metric": "levelized_cost",
       "plus_minus_value": 0.5,
       "n_points": 30
     },
     "tornado": {
       "metric": "npv",
       "plus_minus_value": 0.2
     },
     "monte_carlo": {
       "n_samples": 50000,
       "batch_size": 5000,
       "uncertainties": {
         "fixed_capital_factor": {"std": 0.30, "min": 0.25, "max": 1.75},
         "fixed_opex_factor": {"std": 0.25, "min": 0.30, "max": 1.70},
         "interest_rate": {"std": 0.03, "min": 0.03, "max": 0.20}
       }
     }
   }

Running a study
---------------

Load all three files with the high-level ``run_tea`` function:

.. code-block:: python

   from openpytea import run_tea

   results = run_tea(
       equipment_json="project/equipment.json",
       plant_json="project/plant.json",
       analysis_json="project/analysis.json",
   )

   print(results["metrics"])          # NPV, IRR, ROI, …
   print(results["sensitivity"])      # sensitivity curves
   print(results["monte_carlo"])      # MC summary statistics

Individual loaders
------------------

Use individual functions when building workflows step by step:

.. code-block:: python

   from openpytea.io import (
       load_equipment_config,
       load_plant_config,
       run_equipment,
       run_plant,
   )

   # Load and compute equipment costs
   equipment_list = run_equipment("project/equipment.json")

   # Load plant config and attach equipment
   plant = run_plant("project/plant.json", equipment=equipment_list)
   plant.calculate_all(print_results=True)

Exporting results
-----------------

.. code-block:: python

   from openpytea.io import export_plant_results, export_equipment_results

   # Save equipment cost summary
   export_equipment_results(equipment_list, "results/equipment_results.json")

   # Save full plant TEA results
   export_plant_results(plant, "results/plant_results.json")

The exported JSON includes timestamps, version metadata, cost breakdowns,
and all computed financial metrics — suitable for archiving and sharing.

See also
--------

* :mod:`openpytea.io` — full API reference
* :doc:`../gui` — web GUI save/load feature (same JSON format)
