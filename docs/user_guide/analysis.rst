Analysis
========

The :mod:`openpytea.analysis` module provides tools for understanding cost
structure and how uncertain inputs affect financial outcomes:

* **Cost breakdowns** — prepare equipment-level and plant-level CAPEX/OPEX data
* **One-way sensitivity** — vary one parameter across a range and observe the metric
* **Tornado diagram** — rank all parameters by their ±impact on a single metric
* **Monte Carlo simulation** — propagate all uncertainties simultaneously

All analysis functions accept a configured and calculated
:class:`~openpytea.plant.Plant` object. Visualization of the results is
handled separately by :doc:`plotting`.

To see the outputs of all code examples below, refer to the
`walkthrough notebook <https://github.com/pbtamarona/OpenPyTEA/blob/main/walkthrough.ipynb>`_.

.. code-block:: python

   from openpytea.analysis import (
       direct_costs_data, fixed_capital_data,
       fixed_opex_data, variable_opex_data,
       sensitivity_data, tornado_data, monte_carlo,
   )

CAPEX and OPEX breakdowns
--------------------------

Cost breakdowns are produced in two steps: the analysis functions prepare
structured data, and the plotting functions render it. This separation lets
you reuse the data in custom visualizations or export it directly.

The four data-preparation functions and their outputs:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Output
   * - ``direct_costs_data(plants)``
     - Equipment-level purchased and direct costs.
   * - ``fixed_capital_data(plants)``
     - ISBL, OSBL, D&E, contingency (and optional additional CAPEX).
   * - ``fixed_opex_data(plants)``
     - Each fixed OPEX component (absolute or as % of total).
   * - ``variable_opex_data(plants)``
     - Each variable OPEX item.

Basic usage (single plant):

.. code-block:: python

   # Equipment-level CAPEX
   direct_costs = direct_costs_data(plants=plant)

   # Fixed capital breakdown (include additional CAPEX events)
   fixed_capital = fixed_capital_data(plants=plant, additional_capex=True)

   # Fixed OPEX as percentage of total
   fixed_opex = fixed_opex_data(plants=plant, pct=True)

   # Variable OPEX by item
   variable_opex = variable_opex_data(plants=plant)

Pass the returned data to ``plot_stacked_bar()`` to visualize it — see
:doc:`plotting` for details.

Comparing multiple plants
~~~~~~~~~~~~~~~~~~~~~~~~~

Pass a list of :class:`~openpytea.plant.Plant` objects to compare two or
more configurations side-by-side:

.. code-block:: python

   from copy import deepcopy

   plant_b = deepcopy(plant)
   plant_b.update_configuration({
       "plant_name": "Scenario B",
       "variable_opex_inputs": {
           "electricity": {"consumption": 0.9e6, "price": 0.05},
       },
   })
   plant_b.calculate_all()

   variable_opex = variable_opex_data(plants=[plant, plant_b])
   # pass to plot_stacked_bar() for a side-by-side chart

One-way sensitivity analysis
-----------------------------

:func:`~openpytea.analysis.sensitivity_data` varies a single parameter over
a symmetric range while holding everything else constant, then records the
selected metric at each point.

.. code-block:: python

   # Default metric is LCOP; vary electricity price ±50 %
   sens = sensitivity_data(plants=plant, parameter="electricity", plus_minus_value=0.5)

   # Specify metric and label explicitly
   npv_sens = sensitivity_data(
       plants=plant,
       parameter="methanol",       # product price
       metric="NPV",
       plus_minus_value=0.5,
       label="Project A — NPV [USD]",
   )

``parameter`` can be any of:

* A key from ``variable_opex_inputs`` — varies that item's *price*
* A key from ``plant_products`` — varies that product's *price*
* ``"fixed_capital"`` — scales total installed CAPEX
* ``"fixed_opex"`` — scales total fixed OPEX
* ``"interest_rate"`` — discount rate
* ``"project_lifetime"`` — project duration
* ``"operator_hourly_rate"`` — labor wage

Supported ``metric`` values:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Description
   * - ``"LCOP"``
     - Levelized cost of the primary product (default).
   * - ``"NPV"``
     - Net Present Value.
   * - ``"IRR"``
     - Internal Rate of Return.
   * - ``"ROI"``
     - Return on Investment.
   * - ``"PBT"``
     - Simple payback time in years.

For metrics that depend on revenue (NPV, ROI, IRR, PBT), product prices are
included in the evaluation automatically.

Comparing multiple plants:

.. code-block:: python

   pbt_comparison = sensitivity_data(
       plants=[plant, plant_b],
       parameter="electricity",
       metric="PBT",
       plus_minus_value=0.5,
       additional_capex=True,   # account for mid-project CAPEX events
       n_points=50,
   )

Pass the result to ``plot_sensitivity()`` — see :doc:`plotting`.

Tornado diagram
----------------

A tornado diagram evaluates every variable-cost driver and financial
parameter independently at ±``plus_minus_value``, then ranks them by
impact on the chosen metric.

.. code-block:: python

   from openpytea.analysis import tornado_data

   # Default metric is LCOP
   td = tornado_data(plant=plant, plus_minus_value=0.5)

   # Profit-oriented metric — product prices are included automatically
   td_roi = tornado_data(plant=plant, plus_minus_value=0.5, metric="ROI")

Pass ``td`` to ``plot_tornado()`` — see :doc:`plotting`.

Monte Carlo simulation
-----------------------

Monte Carlo assigns probability distributions to all uncertain inputs and
evaluates the plant thousands or millions of times, producing a distribution
of outcomes for each financial metric.

Configuring input uncertainties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Variable OPEX and product price uncertainties** are defined inline in the
existing ``variable_opex_inputs`` and ``plant_products`` configuration keys
by adding ``std``, ``min``, and ``max`` fields to each item:

.. code-block:: python

   plant.update_configuration({
       "plant_products": {
           "methanol": {
               "production": 150_000,
               "price": 1.75,
               "std": 0.25,    # standard deviation
               "min": 1.25,    # lower truncation bound
               "max": 2.25,    # upper truncation bound
           },
       },
       "operator_hourly_rate": {
           "rate": 38.11,
           "std": 10.0,
           "min": 20.0,
           "max": 60.0,
       },
       "variable_opex_inputs": {
           "electricity": {
               "consumption": 1.4e6,
               "price": 0.10,
               "std": 0.035,
               "min": 0.025,
               "max": 0.175,
           },
           "natural_gas": {
               "consumption": 1.0e5,
               "price": 0.05,
               "std": 0.03,
               "min": 0.001,
               "max": 0.10,
           },
       },
   })

**Project-level financial uncertainties** are set through the
``project_uncertainties`` key:

.. code-block:: python

   plant.update_configuration({
       "project_uncertainties": {
           "fixed_capital_factor": {"std": 0.30, "min": 0.25, "max": 1.75},
           "fixed_opex_factor":    {"std": 0.30, "min": 0.25, "max": 1.75},
           "project_lifetime":     {"std": 5},     # min/max auto-derived
           "interest_rate":        {"std": 0.03},  # min/max auto-derived
           "plant_utilization":    {"std": 0.05},  # opt-in; default std=0
           "tax_rate":             {"std": 0.10},  # opt-in; default std=0
       }
   })

The first four keys are **active by default**. ``plant_utilization`` and
``tax_rate`` require an explicit ``std > 0`` to be sampled. When ``min``/
``max`` are omitted they are derived as ±2 × std around the plant baseline.
Set ``std=0`` for any key to disable it.

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Key
     - Description
     - Default std
   * - ``fixed_capital_factor``
     - Multiplicative factor on total installed CAPEX.
     - 0.30 (30%)
   * - ``fixed_opex_factor``
     - Multiplicative factor on annual fixed OPEX.
     - 0.30 (30%)
   * - ``project_lifetime``
     - Economic project life (years).
     - 5 years
   * - ``interest_rate``
     - Discount / financing rate.
     - 0.03 (3 pp)
   * - ``plant_utilization``
     - Yearly fraction of operating time.
     - 0 (opt-in)
   * - ``tax_rate``
     - Corporate tax rate.
     - 0 (opt-in)

Running the simulation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openpytea.analysis import monte_carlo

   mc_results = monte_carlo(
       plant,
       num_samples=1_000_000,   # increase for accuracy, decrease for speed
       batch_size=10_000,       # adjust to available memory
   )

Results are stored on the ``Plant`` object and returned as a dict keyed by
metric name. Each entry contains the raw sample array:

.. code-block:: python

   # Access results
   print(mc_results["LCOP"])    # array of LCOP samples
   print(mc_results["NPV"])     # array of NPV samples
   # Available keys: "LCOP", "NPV", "IRR", "ROI", "PBT"

Visualizing results
~~~~~~~~~~~~~~~~~~~

Pass the plant (or ``mc_results``) to the plotting functions:

.. code-block:: python

   from openpytea.plotting import plot_monte_carlo, plot_monte_carlo_inputs

   # Distribution of the LCOP
   plot_monte_carlo(plant, metric="LCOP", bins=30)

   # Verify input distributions (useful for checking std/min/max settings)
   plot_monte_carlo_inputs(mc_results, bins=40)

See :doc:`plotting` for full plotting options.

Comparing multiple plants under uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openpytea.plotting import plot_multiple_monte_carlo

   mc_b = monte_carlo(plant_b, num_samples=1_000_000, batch_size=10_000)

   plot_multiple_monte_carlo(
       data_list=[plant, plant_b],
       metric="LCOP",
       bins=30,
   )

See also
--------

* :mod:`openpytea.analysis` — full API reference
* :doc:`plotting` — visualization options
* `Walkthrough notebook <https://github.com/pbtamarona/OpenPyTEA/blob/main/walkthrough.ipynb>`_ — end-to-end worked example
