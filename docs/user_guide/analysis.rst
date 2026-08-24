Analysis
========

The :mod:`openpytea.analysis` module provides tools for understanding cost
structure and how uncertain inputs affect financial outcomes:

* **Cost breakdowns** — prepare equipment-level and plant-level CAPEX/OPEX data
* **Levelized cost breakdown** — split the LCOP into discounted CAPEX, OPEX, and side revenue
* **Cash flow diagram** — track a project's cumulative cash position over time
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
       fixed_opex_data, variable_opex_data, levelized_cost_data,
       cash_flow_data, sensitivity_data, tornado_data, monte_carlo,
   )

CAPEX and OPEX breakdowns
--------------------------

Cost breakdowns are produced in two steps: the analysis functions prepare
structured data, and the plotting functions render it. This separation lets
you reuse the data in custom visualizations or export it directly.

The five data-preparation functions and their outputs:

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
   * - ``levelized_cost_data(plants)``
     - Discounted CAPEX, OPEX, and side revenue per unit of main product.

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

Levelized cost breakdown
--------------------------

:func:`~openpytea.analysis.levelized_cost_data` follows the same ``data`` +
``plot_stacked_bar()`` pattern as the CAPEX/OPEX breakdowns above, but
mirrors the discounting logic in
:meth:`~openpytea.plant.Plant.calculate_levelized_cost`: capital cost, cash
cost, side-product revenue, and production are each discounted over the
project lifetime at the plant's interest rate, then divided by discounted
production to express every component per unit of main product.

.. code-block:: python

   from openpytea.analysis import levelized_cost_data
   from openpytea.plotting import plot_stacked_bar

   lcop = levelized_cost_data(plants=plant)
   fig, ax = plot_stacked_bar(lcop)

Side revenue is stored as a **negative** value (since it is subtracted from
the LCOP numerator), so the three components sum directly to the plant's
LCOP: ``CAPEX + OPEX + Side revenue = LCOP``. ``plot_stacked_bar()`` renders
it as a waterfall-style base below zero rather than stacking it like a
normal cost, so the top of the bar still reads as the true net LCOP — see
:doc:`plotting` for the rendering details.

As with the other breakdowns, pass a list of plants to compare their LCOP
composition side-by-side, and ``pct=True`` to express components as a
percentage of the total instead of absolute values. Only the scalar
(non-Monte Carlo) case is supported — each plant's ``project_lifetime`` and
``interest_rate`` must be a single value, not a sampled array.

Cash flow diagram
--------------------------

:func:`~openpytea.analysis.cash_flow_data` prepares the data behind the
classic project cash flow diagram: cumulative cash position vs. time,
including the dip into debt during construction/start-up, the point of
deepest ("maximum") investment, the break-even (pay-back) point where the
curve first crosses back above zero, and the eventual climb into profit.
It (re)runs each plant's
:meth:`~openpytea.plant.Plant.calculate_cash_flow` to ensure the underlying
annual cash flow array is up to date.

.. code-block:: python

   from openpytea.analysis import cash_flow_data
   from openpytea.plotting import plot_cash_flow

   cash_flow = cash_flow_data(plant)
   fig, ax = plot_cash_flow(cash_flow)

The returned dict has one entry per plant under ``"curves"``, each carrying
the cumulative curve itself (``"years"``, ``"cumulative"``) alongside the
derived figures ``"max_investment"``, ``"max_investment_year"``,
``"breakeven_year"`` (``None`` if the project never recovers), and its alias
``"payback_time"`` — useful for pulling numbers into a report without
re-deriving them from the curve:

.. code-block:: python

   curve = cash_flow["curves"][0]
   print(f"Max investment: {curve['max_investment']:,.0f} in year {curve['max_investment_year']:.0f}")
   print(f"Break-even: year {curve['breakeven_year']:.1f}")

Comparing multiple plants
~~~~~~~~~~~~~~~~~~~~~~~~~

Pass a list of plants to overlay their cumulative cash flow curves, each
with its own shaded debt region and break-even line:

.. code-block:: python

   cash_flow_multi = cash_flow_data([plant, plant_b])
   fig, ax = plot_cash_flow(cash_flow_multi, figsize=(4.5, 3))

Only the scalar (non-Monte Carlo) case is supported; if a plant's
``cash_flow`` has multiple rows (vectorised inputs), the first row is used.

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

**Consumption and production quantities** can be given their own uncertainty
too, independent of price. This is opt-in: nest a ``"consumption_uncertainty"``
dict inside a ``variable_opex_inputs`` item, or a ``"production_uncertainty"``
dict inside a ``plant_products`` item, using the same ``std``/``min``/``max``/
``dist_id`` fields as everywhere else. The baseline ``"consumption"`` /
``"production"`` value is used as the sampling mean unless overridden with
``loc``/``mean`` inside the sub-dict. Items without one of these sub-dicts
keep their consumption/production fixed at the baseline value, exactly as
before.

.. code-block:: python

   plant.update_configuration({
       "variable_opex_inputs": {
           "electricity": {
               "consumption": 1.4e6,
               "price": 0.10,
               "std": 0.035,          # price uncertainty
               "min": 0.025,
               "max": 0.175,
               "consumption_uncertainty": {
                   "std": 1.4e5,      # 10% of baseline consumption
                   "min": 1.0e6,
                   "max": 1.8e6,
               },
           },
       },
       "plant_products": {
           "methanol": {
               "production": 150_000,
               "price": 1.75,
               "std": 0.25,           # price uncertainty
               "min": 1.25,
               "max": 2.25,
               "production_uncertainty": {
                   "std": 15_000,     # 10% of baseline production
                   "min": 100_000,
                   "max": 200_000,
               },
           },
       },
   })

Sampled consumption/production values show up in the Monte Carlo results
under display names like ``"Electricity consumption"`` and ``"Methanol
production"``. Production uncertainty is applied even when product prices
aren't configured, since production also drives LCOP directly (not just
revenue).

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
``tax_rate`` require an explicit ``std > 0`` (or an explicit ``dist_id``) to
be sampled. For ``project_lifetime``, ``interest_rate``, ``plant_utilization``,
and ``tax_rate``, omitted ``min``/``max`` are derived as ±2 × std around the
plant's baseline value. For ``fixed_capital_factor`` and ``fixed_opex_factor``
the default bounds are a fixed ``[0.25, 1.75]`` regardless of ``std`` unless
you set ``min``/``max`` explicitly. Set ``std=0`` for any key to disable
sampling for it (the value collapses to its baseline).

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

Every uncertain input above defaults to a **Normal** distribution built from
its ``std`` (and, if given, ``min``/``max`` truncation bounds). Add a
``dist_id`` field to any uncertainty block — in ``variable_opex_inputs``,
``plant_products``, ``operator_hourly_rate``, or ``project_uncertainties`` —
to draw from a different family instead. Field names are reused across
families (``loc``/``mean``/``price``/``rate``, ``scale``/``std``, ``shape``,
``minimum``/``min``, ``maximum``/``max``); which ones apply depends on
``dist_id``.

Under the hood every family is a frozen `scipy.stats
<https://docs.scipy.org/doc/scipy/reference/stats.html>`_ distribution, so
the parameter meanings and shapes follow SciPy's conventions. The table
below links each family to its SciPy reference page for the full
mathematical definition:

.. list-table::
   :header-rows: 1
   :widths: 10 22 30 38

   * - ``dist_id``
     - Family
     - Parameters used
     - Notes
   * - 0 / 1
     - Fixed value
     - ``loc``
     - No randomness; every draw equals ``loc``.
   * - 2
     - `Lognormal <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html>`_
     - ``loc`` (μ), ``scale`` (σ)
     - Optional ``min``/``max`` truncate the drawn samples.
   * - 3
     - `Normal <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html>`_ *(default)*
     - ``loc`` (mean), ``scale`` (std)
     - Optional ``min``/``max`` truncate the drawn samples.
   * - 4
     - `Uniform <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html>`_
     - ``min``, ``max``
     -
   * - 5
     - `Triangular <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.triang.html>`_
     - ``loc`` (mode), ``min``, ``max``
     -
   * - 6
     - `Bernoulli <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html>`_
     - ``loc`` (probability *p*), ``scale`` (success value, default 1)
     - Draws are 0 or ``scale``. Optional ``min``/``max`` truncate.
   * - 7
     - `Discrete uniform <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.randint.html>`_
     - ``min``, ``max``
     - Integers, inclusive of ``max``.
   * - 8
     - `Weibull <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html>`_
     - ``loc`` (offset), ``scale`` (λ), ``shape`` (k)
     -
   * - 9
     - `Gamma <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html>`_
     - ``loc`` (offset), ``scale`` (θ), ``shape`` (k)
     -
   * - 10
     - `Beta <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html>`_
     - ``loc`` (α), ``shape`` (β), ``max`` (upper bound, default 1)
     -
   * - 11
     - `Generalized extreme value (GEV) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html>`_
     - ``loc`` (μ), ``scale`` (σ), ``shape`` (ξ)
     -
   * - 12
     - `Student's t <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html>`_
     - ``loc`` (median), ``scale``, ``shape`` (ν, degrees of freedom)
     -

.. code-block:: python

   plant.update_configuration({
       "plant_products": {
           "methanol": {
               "production": 150_000,
               "price": 1.75,
               "dist_id": 5,     # Triangular; "price" doubles as the mode
               "min": 1.25,
               "max": 2.50,
           },
       },
       "project_uncertainties": {
           "fixed_capital_factor": {
               "dist_id": 2,     # Lognormal
               "loc": 0.0,       # mu
               "std": 0.20,      # sigma
           },
       },
   })

Only Lognormal, Normal, and Bernoulli (``dist_id`` 2, 3, 6) apply ``min``/
``max`` as post-hoc truncation via rejection sampling. For Uniform,
Triangular, and Discrete uniform the bounds define the distribution itself.
Weibull, Gamma, GEV, and Student's t ignore ``min``/``max`` entirely (Beta
uses ``max`` as its upper scale bound instead).

For direct programmatic use outside of ``monte_carlo``, the same families
are available via :func:`~openpytea.analysis.make_distribution` (returns a
frozen ``scipy.stats`` distribution) and
:func:`~openpytea.analysis.sample_distribution` (draws an array of samples,
with optional truncation).

Running the simulation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openpytea.analysis import monte_carlo

   mc_results = monte_carlo(
       plant,
       num_samples=1_000_000,   # increase for accuracy, decrease for speed
       batch_size=1_000,        # samples evaluated per batch; default 1000
       random_seed=42,          # optional, for reproducible runs
   )

``monte_carlo`` returns a dict:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Description
   * - ``"metrics"``
     - Dict of sample arrays keyed by metric: ``"LCOP"``, ``"NPV"``,
       ``"ROI"``, ``"PBT"``.
   * - ``"inputs"``
     - Dict mapping each sampled input's display name to its sample array.
   * - ``"name"``
     - The plant's name.
   * - ``"num_samples"``, ``"additional_capex"``, ``"currency"``
     - Echo of the parameters the run was executed with.

``LCOP`` is always computed. ``NPV``, ``ROI``, and ``PBT`` are only
meaningful (otherwise they stay zero-filled) when **every** entry in
``plant_products`` has a ``"price"`` set. There is no ``"IRR"`` key here —
IRR is available for :func:`~openpytea.analysis.sensitivity_data` and
:func:`~openpytea.analysis.tornado_data`, but not for ``monte_carlo``. Pass
``additional_capex=True`` to account for mid-project CAPEX events in the
ROI/PBT calculation.

The same ``"metrics"`` and ``"inputs"`` dicts are also stored on the plant
as ``plant.monte_carlo_metrics`` and ``plant.monte_carlo_inputs``, which is
what the plotting functions fall back to when passed a ``Plant`` directly.

.. code-block:: python

   # Access results
   print(mc_results["metrics"]["LCOP"])   # array of LCOP samples
   print(mc_results["metrics"]["NPV"])    # array of NPV samples

Visualizing results
~~~~~~~~~~~~~~~~~~~

Pass the plant (or ``mc_results``) to the plotting functions:

.. code-block:: python

   from openpytea.plotting import plot_monte_carlo, plot_monte_carlo_inputs

   # Distribution of the LCOP
   fig, ax = plot_monte_carlo(plant, metric="LCOP", bins=30)

   # Verify input distributions (useful for checking std/min/max settings)
   fig, axes = plot_monte_carlo_inputs(mc_results, bins=40)

See :doc:`plotting` for full plotting options.

Comparing multiple plants under uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openpytea.plotting import plot_multiple_monte_carlo

   mc_b = monte_carlo(plant_b, num_samples=1_000_000, batch_size=10_000)

   fig, ax = plot_multiple_monte_carlo(
       data_list=[plant, plant_b],
       metric="LCOP",
       bins=30,
   )

See also
--------

* :mod:`openpytea.analysis` — full API reference
* :doc:`plotting` — visualization options
* `Walkthrough notebook <https://github.com/pbtamarona/OpenPyTEA/blob/main/walkthrough.ipynb>`_ — end-to-end worked example
