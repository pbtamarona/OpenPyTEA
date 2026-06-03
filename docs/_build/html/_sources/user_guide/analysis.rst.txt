Sensitivity & Uncertainty Analysis
====================================

The :mod:`openpytea.analysis` module provides three tools for understanding
how uncertain inputs affect financial outcomes:

* **Sensitivity analysis** — vary one parameter across a range and plot the metric
* **Tornado diagram** — rank parameters by their ±impact on a single metric
* **Monte Carlo simulation** — propagate all uncertainties simultaneously

All analysis functions accept a configured and calculated
:class:`~openpytea.plant.Plant` object.

One-way sensitivity analysis
-----------------------------

.. code-block:: python

   from openpytea import sensitivity_data, plot_sensitivity

   # Compute sensitivity of the levelized cost to the electricity price (±50 %)
   sens = sensitivity_data(
       plant,
       parameter="electricity",   # key in variable_opex_inputs
       metric="levelized_cost",
       plus_minus_value=0.5,       # ±50 %
       n_points=30,
   )

   fig, ax = plot_sensitivity(
       sens,
       xlabel="Electricity price (USD/MWh)",
       ylabel="LCOA (USD/t NH₃)",
   )

The ``parameter`` argument accepts:

* A key from ``variable_opex_inputs`` — varies that input's *price*
* A key from ``plant_products`` — varies that product's *price*
* Any numeric attribute of the :class:`~openpytea.plant.Plant` object
  (e.g., ``"interest_rate"``, ``"plant_utilization"``)

Supported ``metric`` values
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value
     - Description
   * - ``"levelized_cost"``
     - Levelized cost of the primary product
   * - ``"npv"``
     - Net Present Value
   * - ``"irr"``
     - Internal Rate of Return
   * - ``"roi"``
     - Return on Investment
   * - ``"payback_time"``
     - Simple payback time in years

Multiple parameters on one chart
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Call ``sensitivity_data`` repeatedly and overlay the results:

.. code-block:: python

   params = {
       "electricity": "Electricity price",
       "interest_rate": "Discount rate",
       "plant_utilization": "Capacity utilization",
   }

   all_sens = {
       label: sensitivity_data(plant, parameter=param, metric="npv")
       for param, label in params.items()
   }

   fig, ax = plot_sensitivity(all_sens, ylabel="NPV (M USD)")

Tornado diagram
---------------

A tornado diagram shows the asymmetric impact of each parameter varying
independently by a fixed percentage:

.. code-block:: python

   from openpytea import tornado_data, plot_tornado

   tornado = tornado_data(
       plant,
       metric="levelized_cost",
       plus_minus_value=0.2,    # ±20 %
   )

   fig, ax = plot_tornado(tornado, title="Tornado — Levelized Cost of Ammonia")

Parameters are automatically ranked from most to least impactful, producing
the characteristic "tornado" shape.

Monte Carlo simulation
-----------------------

Monte Carlo propagates uncertainty through the full economic model to yield
probability distributions for each financial metric.

Configuring uncertainties
~~~~~~~~~~~~~~~~~~~~~~~~~

Each uncertain input is described by a mean, standard deviation, and bounds:

.. code-block:: python

   plant.update_configuration({
       "project_uncertainties": {
           # Capital cost uncertainty (normal distribution, truncated)
           "fixed_capital_factor": {
               "std": 0.30, "min": 0.25, "max": 1.75
           },
           # Fixed OPEX uncertainty
           "fixed_opex_factor": {
               "std": 0.25, "min": 0.30, "max": 1.70
           },
           # Project lifetime uncertainty (years)
           "project_lifetime": {
               "std": 5, "min": 10, "max": 40
           },
           # Discount rate uncertainty
           "interest_rate": {
               "std": 0.03, "min": 0.03, "max": 0.20
           },
       }
   })

Running the simulation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openpytea import monte_carlo, plot_monte_carlo

   mc_results = monte_carlo(plant, n_samples=50_000, batch_size=5_000)

   # Inspect summary statistics
   print(mc_results["levelized_cost"]["summary"])

   # Plot distribution
   fig, ax = plot_monte_carlo(mc_results, metric="levelized_cost")

The ``mc_results`` dictionary contains one entry per metric:

.. code-block:: python

   {
       "levelized_cost": {
           "samples": np.ndarray,        # raw samples
           "summary": {
               "mean": ..., "std": ...,
               "p5": ..., "p25": ..., "median": ..., "p75": ..., "p95": ...,
               "min": ..., "max": ...,
           }
       },
       "npv": { ... },
       "irr": { ... },
       ...
   }

Comparing multiple plants
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openpytea import plot_multiple_monte_carlo

   mc_smr = monte_carlo(plant_smr, n_samples=50_000)
   mc_elec = monte_carlo(plant_electrolysis, n_samples=50_000)

   fig, ax = plot_multiple_monte_carlo(
       [mc_smr, mc_elec],
       labels=["SMR", "Electrolysis"],
       metric="levelized_cost",
   )

See also
--------

* :mod:`openpytea.analysis` — full API reference
* :doc:`plotting` — visualization options
