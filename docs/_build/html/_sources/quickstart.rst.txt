Quick Start
===========

This guide walks through a complete techno-economic assessment in under
10 minutes using OpenPyTEA's Python API.

The scenario: a small **ammonia plant** in the Netherlands powered by
electrolysis, with a single compressor as the main equipment item.

Step 1 — Import the library
----------------------------

.. code-block:: python

   from openpytea import Equipment, Plant
   from openpytea import sensitivity_data, plot_sensitivity
   from openpytea import fixed_capital_data, plot_stacked_bar

Step 2 — Define equipment
--------------------------

Create an :class:`~openpytea.equipment.Equipment` object by specifying the
equipment category, type, size parameter, and material:

.. code-block:: python

   compressor = Equipment(
       name="COMP-01",
       param=5000,            # shaft power in kW
       process_type="Fluids",
       category="Compressors (Centrifugal)",
       type="Centrifugal",
       material="Carbon steel",
   )

   print(compressor)

The purchased cost and direct (installed) cost are computed automatically
from cost correlations and adjusted to 2024 USD using the Chemical
Engineering Plant Cost Index (CEPCI).

Step 3 — Configure the plant
-----------------------------

Pass equipment and financial parameters to :class:`~openpytea.plant.Plant`:

.. code-block:: python

   plant = Plant({
       "plant_name": "Ammonia Plant",
       "process_type": "Fluids",
       "country": "Netherlands",
       "equipment": [compressor],

       # Financial assumptions
       "interest_rate": 0.09,
       "project_lifetime": 20,
       "tax_rate": 0.25,

       # Product: ammonia production and selling price
       "plant_products": {
           "ammonia": {
               "production": 125_000,   # t/yr
               "price": 500,            # USD/t
           }
       },

       # Variable OPEX: electricity consumption and price
       "variable_opex_inputs": {
           "electricity": {
               "consumption": 110,      # GWh/yr
               "price": 75,             # USD/MWh
           }
       },
   })

Step 4 — Run the calculation
-----------------------------

.. code-block:: python

   plant.calculate_all(print_results=True)

This prints a summary table of all costs and financial metrics:

.. code-block:: text

   ╔══════════════════════════════════════════╗
   ║            PLANT TEA RESULTS             ║
   ╠══════════════════════════════════════════╣
   ║  Fixed Capital Investment  :  $ 12.3 M   ║
   ║  Variable OPEX (annual)    :  $ 8.3 M/yr ║
   ║  Fixed OPEX (annual)       :  $ 1.1 M/yr ║
   ║  Revenue (annual)          :  $ 62.5 M/yr║
   ╠══════════════════════════════════════════╣
   ║  NPV                       :  $ 412 M    ║
   ║  IRR                       :  62.3 %     ║
   ║  ROI                       :  47.8 %     ║
   ║  Payback Time              :  2.1 yr     ║
   ║  Levelized Cost (ammonia)  :  166 USD/t  ║
   ╚══════════════════════════════════════════╝

Step 5 — Visualise the CAPEX breakdown
----------------------------------------

.. code-block:: python

   capex = fixed_capital_data(plant)
   fig, ax = plot_stacked_bar(capex, title="Capital Cost Breakdown")
   fig.savefig("capex.png", dpi=150)

Step 6 — Sensitivity analysis
-------------------------------

See how the levelized cost of ammonia changes as the electricity price varies
±50 %:

.. code-block:: python

   sens = sensitivity_data(
       plant,
       parameter="electricity",   # variable OPEX item
       metric="levelized_cost",
       plus_minus_value=0.5,
       n_points=30,
   )
   fig, ax = plot_sensitivity(sens, ylabel="LCOA (USD/t)")

Step 7 — What's next?
----------------------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: User Guide
      :link: user_guide/index
      :link-type: doc

      Deep dives into every module — equipment costing, plant configuration,
      Monte Carlo simulation, JSON workflows, and more.

   .. grid-item-card:: Case Studies
      :link: examples
      :link-type: doc

      Real-world examples: hydrogen production, liquefaction, and
      geothermal energy.
