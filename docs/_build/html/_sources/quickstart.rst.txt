Quick Start
===========

This guide walks through a complete techno-economic assessment in under
10 minutes using OpenPyTEA's Python API.

Step 1 — Import the library
----------------------------

.. code-block:: python

   from openpytea import Equipment, Plant
   from openpytea import sensitivity_data, plot_sensitivity
   from openpytea import fixed_capital_data, plot_stacked_bar

Step 2 — Define equipment
--------------------------

Create an :class:`~openpytea.equipment.Equipment` object by specifying the
equipment category, type, size parameter, and material. To check which equipment 
categories and types are available, open the 
:download:`cost correlation database <../src/openpytea/data/cost_correlations.csv>`. 
Here is a snippet of how the database is structured:

.. csv-table::
   :header: "key", "category", "type", "units", "s_lower", "s_upper", "form", "cost_year", "source"
   :widths: 22, 18, 14, 10, 6, 6, 10, 6, 8
   :class: small-table

   "impeller_mixer_turton_2001", "Agitators, blenders, & mixers", "Impeller mixer", "power, kW", 5, 150, "quad log-log", 2001, "Turton et al. (2018)"
   "boilers_packaged_towler_2010", "Boilers, heaters, & furnaces", "Boiler, packaged", "steam, kg/h", 5000, 200000, "power-law", 2010, "Towler & Sinnott (2010)"
   "solid_bowl_centrifuge_turton_2001", "Centrifuges", "Solid bowl", "diameter, m", 0.3, 2, "quad log-log", 2001, "Turton et al. (2018)"
   "centrifugal_compressor_towler_2010", "Compressors, fans, & blowers", "Compressor, centrifugal", "driver power, kW", 75, 30000, "power-law", 2010, "Towler & Sinnott (2010)"
   "...", "...", "...", "...", "...", "...", "...", "...", "..."

For example, let's define a centrifugal compressor with a shaft power of 5000 kW, made of carbon steel:

.. code-block:: python

   compressor = Equipment(
       name="COMP-01",
       param=5000,            # driver power in kW
       process_type="Fluids",
       category="Compressors, fans, & blowers",
       type="Compressor, centrifugal",
       material="Carbon steel",
   )

   print(compressor)

The purchased cost and direct (installed) cost are computed automatically
from cost correlations and adjusted to 2024 USD using the Chemical
Engineering Plant Cost Index (:download:`CEPCI <../src/openpytea/data/cepci_values.csv>`).

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

To learn more about plant configuration inputs, see the :doc:`Plant user guide <user_guide/plant>`.

Step 4 — Run the calculation
-----------------------------

.. code-block:: python

   plant.calculate_all(print_results=True)

This prints a summary table of all costs and financial metrics:

.. code-block:: text

   Capital cost estimation
   ===================================
   ISBL: 9,874,560.00 USD
   OSBL: 987,456.00 USD
   Design and engineering: 1,184,947.20 USD
   Contingency: 1,184,947.20 USD
   ===================================
   Fixed capital investment: 13,231,910.40 USD
   Variable production costs estimation
   ===================================
     - Electricity: 8,250,000.00 USD per year
   ===================================
   Total Variable OPEX: 8,250,000.00 USD per year
   Fixed production costs estimation
   ===================================
   Operating labor costs: 220,000.00 USD per year
   Supervision costs: 44,000.00 USD per year
   ...
   ===================================
   Fixed OPEX: 1,123,456.00 USD per year
   Revenue estimation
   ===================================
     - Ammonia: 62,500,000.00 USD per year
   ===================================
   Total Revenue: 62,500,000.00 USD per year
   Year | Present Value [USD] | Cumulative NPV [USD]
   -------------------------------------------
      1 |     48,623,853.21 |      48,623,853.21
      2 |     44,609,498.36 |      93,233,351.57
      3 |     40,926,145.28 |     134,159,496.85
      ...
     20 |     17,284,565.91 |     411,847,293.42
   Levelized cost: 166.123 USD/unit
   Payback time: 2.09 years
   Return of investment: 47.83%
   Internal Rate of Return: 62.31%

Step 5 — Visualise the CAPEX breakdown
----------------------------------------

.. code-block:: python

   capex = fixed_capital_data(plant)
   fig, ax = plot_stacked_bar(capex)
   fig.savefig("capex.png", dpi=150)

For more visualisation options, see the :doc:`Plotting user guide <user_guide/plotting>`.

Step 6 — Sensitivity analysis
-------------------------------

See how the levelized cost of ammonia changes as the electricity price varies
±50 %:

.. code-block:: python

   sens = sensitivity_data(
       plant,
       parameter="electricity",   # variable OPEX item
       metric="LCOP", # levelized cost of product
       plus_minus_value=0.5,
       n_points=30,
       label="LCOA (USD/t)",
   )
   fig, ax = plot_sensitivity(sens)

For more sensitivity and uncertainty analysis options, see the :doc:`Analysis user guide <user_guide/analysis>`.

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
