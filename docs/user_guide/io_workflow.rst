Config Files & CLI
===================

OpenPyTEA supports a fully declarative, JSON-based workflow that makes studies
**reproducible**, **shareable**, and easy to version-control.
The :mod:`openpytea.io` module handles loading configuration files and
exporting results.

Four high-level functions drive the workflow:

* ``run_equipment()`` — reads an equipment configuration file and returns a
  list of :class:`~openpytea.equipment.Equipment` objects with all costs
  estimated, writing the results to an output JSON file.
* ``run_plant()`` — reads a plant configuration file, combines it with the
  equipment list, runs all calculations, and writes the plant results to an
  output JSON file.
* ``run_tea()`` — executes the full TEA pipeline (cost breakdowns, sensitivity
  analysis, Monte Carlo simulation) from three input files, writing results and
  optional plots to an output directory.
* ``run_openpytea()`` — the single-file counterpart to ``run_tea()``: same
  pipeline, but driven by one combined JSON file rather than three. This is
  the entry point used by the CLI (see :ref:`single-file-cli-workflow`).

To see all examples below in action, refer to the
`walkthrough notebook <https://github.com/pbtamarona/OpenPyTEA/blob/main/walkthrough.ipynb>`_
and the
`case study JSON notebook <https://github.com/pbtamarona/OpenPyTEA/blob/main/examples/case_study_1_with_JSON.ipynb>`_.

.. code-block:: python

   from openpytea import (
       run_equipment, run_plant, run_tea, run_openpytea, load_results
   )

File structure
--------------

A complete TEA study uses three JSON input files:

.. code-block:: text

   project/
   ├── equipment.json      # List of equipment items
   ├── plant.json          # Plant configuration and financial assumptions
   └── analysis.json       # Analysis settings and output options

For single-file / CLI use, the same three blocks (``equipment``, ``plant``,
``analysis``) can instead live under one combined file — see
:ref:`single-file-cli-workflow` below.

``equipment.json``
------------------

Each entry requires ``name``, ``process_type``, ``category``, and either
``param`` (size/capacity parameter) or ``purchased_cost`` (manual override).
All other fields are optional.

.. code-block:: json

   {
     "equipment": [
       {
         "name": "COMP-1",
         "param": 945,
         "process_type": "Fluids",
         "category": "Compressors, fans, & blowers",
         "type": "Compressor, centrifugal",
         "material": "Carbon steel"
       },
       {
         "name": "HX-1",
         "param": 31.87,
         "process_type": "Fluids",
         "category": "Heat Exchangers",
         "type": "U-tube shell & tube",
         "material": "Carbon steel"
       }
     ]
   }

``plant.json``
--------------

The ``plant`` object mirrors the configuration dict accepted by
:class:`~openpytea.plant.Plant`. Monte Carlo uncertainty fields (``std``,
``min``, ``max``) can be embedded directly in ``variable_opex_inputs``,
``plant_products``, and ``operator_hourly_rate`` — they are ignored when
running plant-only calculations and activated automatically when
``run_tea()`` executes a Monte Carlo block. Uncertainty on project-level
factors (``fixed_capital_factor``, ``fixed_opex_factor``,
``project_lifetime``, ``interest_rate``, ``plant_utilization``,
``tax_rate``) is instead configured via the top-level
``project_uncertainties`` key; omitting a parameter there falls back to its
built-in default distribution, and setting ``"std": 0`` disables sampling
for it.

.. code-block:: json

   {
     "plant": {
       "plant_name": "Steam Reforming",
       "process_type": "Fluids",
       "country": "United States",
       "region": "Gulf Coast",
       "interest_rate": 0.09,
       "project_lifetime": 30,
       "plant_utilization": 0.95,

       "project_uncertainties": {
         "fixed_capital_factor": { "std": 0.3, "min": 0.25, "max": 1.75 },
         "interest_rate": { "std": 0.02 },
         "plant_utilization": { "std": 0.05 }
       },

       "operator_hourly_rate": {
         "rate": 38.11,
         "std": 10,
         "min": 25,
         "max": 50
       },

       "plant_products": {
         "hydrogen": {
           "production": 50000,
           "price": 5.5
         }
       },

       "variable_opex_inputs": {
         "electricity": {
           "consumption": 38293.44,
           "price": 0.05,
           "std": 0.03,
           "min": 0.01,
           "max": 3
         },
         "methane_feed": {
           "consumption": 219936,
           "price": 0.4,
           "std": 0.25,
           "min": 0,
           "max": 5
         }
       }
     }
   }

``analysis.json``
-----------------

The ``analysis`` block contains one sub-object per analysis type. Each must
have ``"run": true`` to be executed, plus an ``"args"`` dict that is passed
directly to the corresponding Python function. The ``output`` block controls
whether results and plots are saved, and in what format.

.. code-block:: json

   {
     "analysis": {
       "direct_costs":    { "run": true, "args": { "pct": false } },
       "fixed_capital":   { "run": true, "args": { "additional_capex": false, "pct": false } },
       "fixed_opex":      { "run": true, "args": { "pct": true } },
       "variable_opex":   { "run": true, "args": { "pct": false } },
       "levelized_cost":  { "run": true, "args": { "pct": false } },
       "cash_flow":       { "run": true },

       "sensitivity": {
         "run": true,
         "cases": [
           {
             "name": "methane_feed_lcop",
             "args": { "parameter": "methane_feed", "plus_minus_value": 1.0, "metric": "LCOP" }
           },
           {
             "name": "electricity_npv",
             "args": { "parameter": "electricity", "plus_minus_value": 0.5, "metric": "NPV" }
           }
         ]
       },

       "tornado": {
         "run": true,
         "args": { "plus_minus_value": 0.5, "metric": "NPV" }
       },

       "monte_carlo": {
         "run": true,
         "args": { "num_samples": 1000000, "batch_size": 10000 },
         "metric": ["LCOP", "NPV"],
         "plot_inputs": true
       }
     },

     "output": {
       "save_json": true,
       "save_plots": true,
       "plot_format": "pdf",
       "dpi": 600
     }
   }

The ``"metric"`` list under ``monte_carlo`` controls which metrics are
rendered as histogram plots when ``save_plots`` is ``true``. Setting
``"plot_inputs": true`` additionally renders a grid of histograms (via
:func:`~openpytea.plotting.plot_monte_carlo_inputs`) showing the sampled
distribution of every uncertain input (project-level factors from
``plant.project_uncertainties``, plus any priced ``variable_opex_inputs``,
``plant_products``, and ``operator_hourly_rate``), saved as
``{plant_name}_monte_carlo_inputs.{plot_format}``. The ``args`` dict maps
directly to :func:`~openpytea.analysis.monte_carlo` keyword arguments.

Running a study
---------------

Equipment I/O
~~~~~~~~~~~~~

:func:`~openpytea.io.run_equipment` reads the equipment file, builds the
:class:`~openpytea.equipment.Equipment` objects, and writes a results JSON:

.. code-block:: python

   equipment_list = run_equipment(
       input_path="project/equipment.json",
       output_path="outputs/equipment_results.json",
   )

   print(equipment_list[0])   # inspect a single Equipment object

Plant I/O
~~~~~~~~~

:func:`~openpytea.io.run_plant` loads the plant configuration, attaches the
equipment, runs all calculations, and writes a plant results JSON:

.. code-block:: python

   plant = run_plant(
       plant_input_path="project/plant.json",
       plant_output_path="outputs/plant_results.json",
       equipment_input_path="project/equipment.json",
   )

If you already have an ``equipment_list`` from a previous ``run_equipment``
call, pass it directly instead:

.. code-block:: python

   plant = run_plant(
       plant_input_path="project/plant.json",
       plant_output_path="outputs/plant_results.json",
       equipment_list=equipment_list,
   )

Full TEA pipeline
~~~~~~~~~~~~~~~~~

:func:`~openpytea.io.run_tea` orchestrates the complete workflow from the
three input files and writes all results and plots to ``output_dir``:

.. code-block:: python

   results = run_tea(
       equipment_input_path="project/equipment.json",
       plant_input_path="project/plant.json",
       analysis_input_path="project/analysis.json",
       output_dir="outputs/tea_results",
   )

The function returns a dict with keys for each analysis that was run:

.. code-block:: text

   results["direct_costs"]    # equipment-level cost breakdown
   results["fixed_capital"]   # CAPEX breakdown
   results["fixed_opex"]      # fixed OPEX breakdown
   results["variable_opex"]   # variable OPEX breakdown
   results["levelized_cost"]  # LCOP breakdown (CAPEX, OPEX, side revenue)
   results["cash_flow"]       # cumulative cash flow diagram data
   results["sensitivity"]     # dict of sensitivity cases
   results["tornado"]         # tornado data
   results["monte_carlo"]     # Monte Carlo results (metrics + sampled inputs)

.. _single-file-cli-workflow:

Single-file / CLI workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~openpytea.io.run_openpytea` runs the exact same pipeline as
``run_tea()``, but from a single combined configuration file instead of
three. This is convenient for CLI use and for sharing one self-contained
study file. The file simply merges the top-level keys that would otherwise
live in ``equipment.json``, ``plant.json``, and ``analysis.json``:

.. code-block:: json

   {
     "equipment": [ { "name": "COMP-1", "param": 945, "...": "..." } ],
     "plant": { "plant_name": "Steam Reforming", "...": "..." },
     "analysis": { "direct_costs": { "run": true }, "...": "..." },
     "output": { "save_json": true, "save_plots": true }
   }

.. code-block:: python

   results = run_openpytea(
       config_path="project/config.json",
       output_dir="outputs/tea_results",
   )

It returns the same results dict, and writes the same
``{plant_name}_equipment_results.json``,
``{plant_name}_plant_results.json``, and
``{plant_name}_analysis_results.json`` files (plus optional plots) as
``run_tea()`` — so :func:`~openpytea.io.load_results` and everything under
:ref:`Comparing multiple scenarios <comparing-multiple-scenarios>` work
unchanged. Use :func:`~openpytea.io.load_openpytea_config` directly if you
only need the parsed config dict (e.g. to validate a file before running
it). See ``examples/input_configs/smr_combined.json`` for a full example.

The ``openpytea`` command
^^^^^^^^^^^^^^^^^^^^^^^^^^

Installing the package also installs an ``openpytea`` console command
(:mod:`openpytea.cli`), a thin wrapper around the four functions above.
Running the combined config from the previous example needs no Python at
all:

.. code-block:: console

   $ openpytea run project/config.json --output-dir outputs/tea_results
   Ran: direct_costs, fixed_capital, fixed_opex, variable_opex, ...
   Results written to: outputs/tea_results

``--output-dir``/``-o`` is optional — omit it to fall back to the config's
own ``output.directory`` (or ``"results"`` if that isn't set either).

The other three functions are available as their own subcommands, for
running a single stage of the pipeline:

.. code-block:: console

   $ openpytea equipment project/equipment.json outputs/equipment_results.json
   Wrote 14 equipment item(s) to outputs/equipment_results.json

   $ openpytea plant project/plant.json outputs/plant_results.json \
         --equipment project/equipment.json
   Wrote plant 'Steam Reforming' results to outputs/plant_results.json

   $ openpytea tea --equipment project/equipment.json \
         --plant project/plant.json \
         --analysis project/analysis.json \
         --output-dir outputs/tea_results
   Ran: direct_costs, fixed_capital, ...
   Results written to: outputs/tea_results

Run ``openpytea --help`` or ``openpytea <command> --help`` for the full
list of options, and ``openpytea --version`` to print the installed
version.

Loading saved results
---------------------

Use :func:`~openpytea.io.load_results` to reload a previously saved analysis
results file for further analysis or visualization. The output file is named
``{plant_name}_analysis_results.json`` inside ``output_dir``:

.. code-block:: python

   results = load_results(
       filepath="outputs/tea_results/Steam Reforming_analysis_results.json"
   )

   mc = results["monte_carlo"]

   from openpytea.plotting import plot_monte_carlo
   fig, ax = plot_monte_carlo(mc, metric="LCOP", bins=30)

.. _comparing-multiple-scenarios:

Comparing multiple scenarios
-----------------------------

Run ``run_tea()`` for each scenario and reload the Monte Carlo results to
compare them on the same figure:

.. code-block:: python

   run_tea(
       equipment_input_path="project/smr_equipment.json",
       plant_input_path="project/smr_plant.json",
       analysis_input_path="project/analysis.json",
       output_dir="outputs/smr_results",
   )

   run_tea(
       equipment_input_path="project/aec_equipment.json",
       plant_input_path="project/aec_plant.json",
       analysis_input_path="project/analysis.json",
       output_dir="outputs/aec_results",
   )

   smr_mc = load_results("outputs/smr_results/Steam Reforming_analysis_results.json")["monte_carlo"]
   aec_mc = load_results("outputs/aec_results/Electrolysis_analysis_results.json")["monte_carlo"]

   from openpytea.plotting import plot_multiple_monte_carlo
   plot_multiple_monte_carlo(data_list=[smr_mc, aec_mc], metric="LCOP", bins=30)

See also
--------

* :mod:`openpytea.io` — full API reference
* :doc:`analysis` — analysis functions used inside ``run_tea()``
* :doc:`plotting` — visualization functions for loaded results
* `Walkthrough notebook <https://github.com/pbtamarona/OpenPyTEA/blob/main/walkthrough.ipynb>`_ — end-to-end worked example
