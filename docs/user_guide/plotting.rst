Plotting
========

The :mod:`openpytea.plotting` module wraps matplotlib to produce
publication-quality figures using the `SciencePlots
<https://github.com/garrettj403/SciencePlots>`_ style. All functions return
a ``(fig, ax)`` tuple so you can further customize them before saving.

Cost breakdown charts
---------------------

Stacked bar charts visualize cost structure data returned by the ``*_data``
helper functions in :mod:`openpytea.analysis`.

.. code-block:: python

   from openpytea import (
       fixed_capital_data,
       fixed_opex_data,
       variable_opex_data,
       plot_stacked_bar,
   )

   # Capital cost breakdown (ISBL, OSBL, D&E, Contingency)
   capex_data = fixed_capital_data(plant)
   fig, ax = plot_stacked_bar(capex_data, title="Capital Cost Breakdown")

   # Fixed OPEX breakdown
   fopex_data = fixed_opex_data(plant)
   fig, ax = plot_stacked_bar(fopex_data, title="Fixed OPEX Breakdown")

   # Variable OPEX breakdown
   vopex_data = variable_opex_data(plant)
   fig, ax = plot_stacked_bar(vopex_data, title="Variable OPEX Breakdown")

Equipment direct costs
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openpytea import direct_costs_data, plot_stacked_bar

   equip_data = direct_costs_data(plant)
   fig, ax = plot_stacked_bar(equip_data, title="Equipment Direct Costs")

Sensitivity plots
-----------------

.. code-block:: python

   from openpytea import sensitivity_data, plot_sensitivity

   sens = sensitivity_data(plant, parameter="electricity", metric="npv")
   fig, ax = plot_sensitivity(
       sens,
       xlabel="Electricity price variation",
       ylabel="NPV (USD)",
       figsize=(6, 4),
   )
   fig.savefig("sensitivity.pdf")

Multiple parameters on the same axes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass a dictionary mapping labels to ``sensitivity_data`` outputs:

.. code-block:: python

   sens_dict = {
       "Electricity": sensitivity_data(plant, "electricity", "npv"),
       "CapEx factor": sensitivity_data(plant, "fixed_capital_factor", "npv"),
   }
   fig, ax = plot_sensitivity(sens_dict, ylabel="NPV (USD)")

Tornado diagrams
----------------

.. code-block:: python

   from openpytea import tornado_data, plot_tornado

   t_data = tornado_data(plant, metric="levelized_cost", plus_minus_value=0.2)
   fig, ax = plot_tornado(t_data, title="Tornado — LCOA")
   fig.savefig("tornado.pdf")

Monte Carlo histograms
-----------------------

.. code-block:: python

   from openpytea import monte_carlo, plot_monte_carlo

   mc = monte_carlo(plant, n_samples=50_000)

   # Single metric
   fig, ax = plot_monte_carlo(mc, metric="levelized_cost")

   # Multiple metrics in a grid
   fig, axes = plot_monte_carlo(mc)   # defaults to all metrics

Comparing scenarios
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openpytea import plot_multiple_monte_carlo

   fig, ax = plot_multiple_monte_carlo(
       [mc_scenario_a, mc_scenario_b],
       labels=["Scenario A", "Scenario B"],
       metric="npv",
   )

Saving figures
--------------

All functions return a ``(fig, ax)`` tuple. Use standard matplotlib methods
to save:

.. code-block:: python

   fig, ax = plot_stacked_bar(capex_data)
   fig.savefig("capex.png", dpi=300, bbox_inches="tight")
   fig.savefig("capex.pdf")          # vector format for publications

Customizing style
-----------------

Figures use the IEEE SciencePlots style by default. You can override any
matplotlib settings after the call:

.. code-block:: python

   fig, ax = plot_sensitivity(sens)
   ax.set_title("Custom title", fontsize=14)
   ax.set_xlim(-0.6, 0.6)
   ax.legend(loc="upper left")

See also
--------

* :mod:`openpytea.plotting` — full API reference
* :mod:`openpytea.analysis` — data preparation functions
