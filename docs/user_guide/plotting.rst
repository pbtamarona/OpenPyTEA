Plotting
========

The :mod:`openpytea.plotting` module wraps matplotlib to produce
publication-quality figures using the `SciencePlots
<https://github.com/garrettj403/SciencePlots>`_ style. All functions return
a ``(fig, ax)`` tuple — a :class:`matplotlib.figure.Figure` and a
:class:`matplotlib.axes.Axes` — so you can further customize or save the
figure directly.

To see the outputs of all code examples below, refer to the
`walkthrough notebook <https://github.com/pbtamarona/OpenPyTEA/blob/main/walkthrough.ipynb>`_.

Cost breakdown charts
---------------------

Stacked bar charts visualize cost structure data returned by the ``*_data``
helper functions in :mod:`openpytea.analysis`.

.. code-block:: python

   from openpytea.analysis import (
       direct_costs_data,
       fixed_capital_data,
       fixed_opex_data,
       variable_opex_data,
   )
   from openpytea.plotting import plot_stacked_bar

   # Equipment-level direct costs
   equip_data = direct_costs_data(plants=plant)
   fig, ax = plot_stacked_bar(equip_data)

   # Capital cost breakdown (ISBL, OSBL, D&E, Contingency)
   capex_data = fixed_capital_data(plants=plant)
   fig, ax = plot_stacked_bar(capex_data)

   # Fixed OPEX breakdown
   fopex_data = fixed_opex_data(plants=plant)
   fig, ax = plot_stacked_bar(fopex_data)

   # Variable OPEX breakdown
   vopex_data = variable_opex_data(plants=plant)
   fig, ax = plot_stacked_bar(vopex_data)

Sensitivity plots
-----------------

.. code-block:: python

   from openpytea.analysis import sensitivity_data
   from openpytea.plotting import plot_sensitivity

   # Vary electricity price ±50 % and plot LCOP
   sens = sensitivity_data(plants=plant, parameter="electricity", plus_minus_value=0.5)
   fig, ax = plot_sensitivity(sens)

   fig.savefig("sensitivity.pdf")

Axis labels and the legend are set automatically from the data returned by
:func:`~openpytea.analysis.sensitivity_data`. Pass a custom ``figsize`` to
resize the chart:

.. code-block:: python

   fig, ax = plot_sensitivity(sens, figsize=(5, 3))

Comparing multiple plants
~~~~~~~~~~~~~~~~~~~~~~~~~

Pass a list of plants to :func:`~openpytea.analysis.sensitivity_data` to
plot all curves on the same axes:

.. code-block:: python

   sens_multi = sensitivity_data(
       plants=[plant, plant_b],
       parameter="electricity",
       metric="NPV",
       plus_minus_value=0.5,
   )
   fig, ax = plot_sensitivity(sens_multi)

Tornado diagrams
----------------

.. code-block:: python

   from openpytea.analysis import tornado_data
   from openpytea.plotting import plot_tornado

   # Default metric is LCOP
   td = tornado_data(plant=plant, plus_minus_value=0.5)
   fig, ax = plot_tornado(td)

   # Profit-oriented metric
   td_roi = tornado_data(plant=plant, plus_minus_value=0.5, metric="ROI")
   fig, ax = plot_tornado(td_roi)

   fig.savefig("tornado.pdf")

Monte Carlo histograms
-----------------------

.. code-block:: python

   from openpytea.analysis import monte_carlo
   from openpytea.plotting import plot_monte_carlo

   mc_results = monte_carlo(plant, num_samples=1_000_000, batch_size=10_000)

   # Distribution of the LCOP
   fig, ax = plot_monte_carlo(plant, metric="LCOP", bins=30)

   fig.savefig("monte_carlo_lcop.pdf")

Visualizing input distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :func:`~openpytea.plotting.plot_monte_carlo_inputs` to verify that the
``std``/``min``/``max`` settings produce the intended input distributions:

.. code-block:: python

   from openpytea.plotting import plot_monte_carlo_inputs

   fig, axes = plot_monte_carlo_inputs(mc_results, bins=40)

Comparing scenarios
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openpytea.plotting import plot_multiple_monte_carlo

   mc_b = monte_carlo(plant_b, num_samples=1_000_000, batch_size=10_000)

   fig, ax = plot_multiple_monte_carlo(
       data_list=[plant, plant_b],
       metric="LCOP",
       bins=30,
   )

Saving figures
--------------

All functions return a ``(fig, ax)`` tuple. Use ``fig`` directly to save:

.. code-block:: python

   fig, ax = plot_stacked_bar(capex_data)
   fig.savefig("capex.png", dpi=300, bbox_inches="tight")
   fig.savefig("capex.pdf")   # vector format for publications

Customizing axes
-----------------

You can modify the returned axes object with standard matplotlib calls:

.. code-block:: python

   fig, ax = plot_sensitivity(sens)
   ax.set_title("Custom title", fontsize=14)
   ax.set_xlim(-0.6, 0.6)
   ax.legend(loc="upper left")

See also
--------

* :mod:`openpytea.plotting` — full API reference
* :mod:`openpytea.analysis` — data preparation functions
* `Walkthrough notebook <https://github.com/pbtamarona/OpenPyTEA/blob/main/walkthrough.ipynb>`_ — end-to-end worked example
