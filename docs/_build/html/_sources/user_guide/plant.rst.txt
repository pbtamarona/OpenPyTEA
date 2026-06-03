Plant-Level TEA
===============

The :class:`~openpytea.plant.Plant` class is the core of OpenPyTEA. It takes
a list of equipment objects and financial parameters, and computes the full
capital and operating cost structure together with financial metrics.

Creating a ``Plant``
---------------------

.. code-block:: python

   from openpytea import Plant, Equipment

   # Equipment list (one or more items)
   pump = Equipment(
       name="P-01", param=50, process_type="Fluids",
       category="Pumps (Centrifugal)", type="Single stage",
       material="Carbon steel",
   )

   plant = Plant({
       "plant_name": "My Process Plant",
       "process_type": "Fluids",
       "country": "United States",
       "region": "Gulf Coast",
       "currency": "USD",
       "equipment": [pump],
       "interest_rate": 0.09,
       "project_lifetime": 20,
   })

   plant.calculate_all(print_results=True)

Configuration parameters
-------------------------

General
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``plant_name``
     - ``""``
     - Descriptive name of the plant
   * - ``process_type``
     - required
     - ``"Solids"``, ``"Fluids"``, or ``"Mixed"``
   * - ``country``
     - ``"United States"``
     - Country for location cost factor lookup
   * - ``region``
     - ``"Gulf Coast"``
     - Region within the country
   * - ``currency``
     - ``"USD"``
     - Currency label (display only)
   * - ``exchange_rate``
     - ``1.0``
     - Multiplier to convert USD results to local currency

Financial
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``interest_rate``
     - ``0.09``
     - Discount / hurdle rate for NPV calculations
   * - ``project_lifetime``
     - ``20``
     - Operating life in years (can be an array for scenario analysis)
   * - ``plant_utilization``
     - ``1.0``
     - Annual capacity utilization factor (0–1)
   * - ``tax_rate``
     - ``0``
     - Corporate income tax rate applied to cash flow
   * - ``working_capital``
     - ``None``
     - Working capital; auto-calculated as fraction of fixed capital if ``None``

Products and OPEX
~~~~~~~~~~~~~~~~~

.. code-block:: python

   plant.update_configuration({
       "plant_products": {
           "hydrogen": {
               "production": 10_000,    # t/yr
               "price": 3.0,            # USD/kg → set unit carefully
           },
       },
       "variable_opex_inputs": {
           "electricity": {
               "consumption": 55,       # GWh/yr
               "price": 60,             # USD/MWh
           },
           "natural gas": {
               "consumption": 200,      # GWh/yr (LHV)
               "price": 30,             # USD/MWh
           },
       },
   })

Capital cost structure
-----------------------

The fixed capital investment (FCI) is built up from the equipment direct
costs in four layers:

.. math::

   \text{ISBL} = \text{loc\_factor} \times \sum C_{\text{direct}}

.. math::

   \text{OSBL} = f_{\text{os}} \times \text{ISBL}

.. math::

   \text{D\&E} = f_{\text{de}} \times (\text{ISBL} + \text{OSBL})

.. math::

   \text{Contingency} = f_{\text{X}} \times (\text{ISBL} + \text{OSBL})

.. math::

   \text{FCI} = \text{ISBL} + \text{OSBL} + \text{D\&E} + \text{Contingency}

Default factors by process type:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20

   * - Factor
     - Solids
     - Fluids
     - Mixed
   * - OSBL (:math:`f_{\text{os}}`)
     - 0.40
     - 0.30
     - 0.40
   * - D&E (:math:`f_{\text{de}}`)
     - 0.20
     - 0.30
     - 0.25
   * - Contingency (:math:`f_{\text{X}}`)
     - 0.10
     - 0.10
     - 0.10

You can override any factor or component directly:

.. code-block:: python

   plant.update_configuration({
       "fixed_capital_factors": {"osbl": 0.25},        # override factor
       "fixed_capital_components": {"contingency": 5e6},  # set value directly
   })

Operating cost structure
-------------------------

Fixed OPEX
~~~~~~~~~~

Fixed operating costs are calculated from empirical factors applied to the
ISBL, FCI, and labor costs:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Component
     - Default basis
   * - Operating labor
     - Empirical correlation (solid/fluid process units)
   * - Supervision
     - 0.25 × operating labor
   * - Direct salary overhead
     - 0.50 × (labor + supervision)
   * - Laboratory charges
     - 0.10 × operating labor
   * - Maintenance
     - 0.05 × ISBL
   * - Taxes & insurance
     - 0.015 × ISBL
   * - Rent of land
     - 0.015 × (ISBL + OSBL)
   * - Environmental charges
     - 0.010 × (ISBL + OSBL)
   * - Operating supplies
     - 0.009 × ISBL
   * - General plant overhead
     - 0.65 × (labor + supervision + overhead)
   * - Patents & royalties
     - 0.020 × cash cost
   * - Distribution & selling
     - 0.020 × cash cost
   * - R&D
     - 0.030 × cash cost

Override individual fixed OPEX factors:

.. code-block:: python

   plant.update_configuration({
       "fixed_opex_factors": {
           "maintenance": 0.03,       # lower maintenance fraction
           "taxes_insurance": 0.02,
       }
   })

Variable OPEX
~~~~~~~~~~~~~

Variable costs are computed as:

.. math::

   C_{\text{var}} = \sum_i \text{consumption}_i \times \text{price}_i \times \text{utilization}

Labor modeling
~~~~~~~~~~~~~~

By default, operating labor is estimated from an empirical correlation based
on the number of solid and fluid handling process steps. You can override
this manually:

.. code-block:: python

   plant.update_configuration({
       "operators_per_shift": 5,
       "operators_hired": 25,
       "operator_hourly_rate": 35,       # USD/hr
       "working_weeks_per_year": 49,
       "working_shifts_per_week": 5,
       "operating_shifts_per_day": 3,
   })

Location factors
----------------

Location factors scale the ISBL to account for regional construction cost
differences relative to the US Gulf Coast (factor = 1.00):

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Country / Region
     - Factor range
     - Example
   * - United States (Gulf Coast)
     - 1.00
     - 1.00
   * - United States (West Coast)
     - 1.07
     - 1.07
   * - Netherlands
     - 1.20
     - —
   * - Japan
     - 1.26
     - —
   * - Australia
     - 1.30
     - —
   * - India
     - 0.85
     - —

Set a custom location factor if your country is not listed:

.. code-block:: python

   plant.update_configuration({"loc_factor": 1.15})

Cash flow and financial metrics
--------------------------------

CAPEX and production ramp-up
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``capex_ramp`` to distribute capital spending across construction years
and ``production_ramp`` to ramp up production:

.. code-block:: python

   plant.update_configuration({
       # 30 % in year 1, 60 % in year 2, 10 % in year 3
       "capex_ramp": [0.3, 0.6, 0.1],

       # 0 % in construction years 1–3, then 50 % → 80 % → 100 %
       "production_ramp": [0, 0, 0, 0.5, 0.8],
   })

Financial metrics
~~~~~~~~~~~~~~~~~

After calling :meth:`~openpytea.plant.Plant.calculate_all`, the following
metrics are available:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute
     - Description
   * - ``plant.npv``
     - Net Present Value (USD)
   * - ``plant.irr``
     - Internal Rate of Return (fraction)
   * - ``plant.roi``
     - Return on Investment (fraction)
   * - ``plant.payback_time``
     - Simple payback time (years)
   * - ``plant.levelized_cost``
     - Levelized cost per unit of primary product
   * - ``plant.cash_flow``
     - Year-by-year cash flow DataFrame

Depreciation
~~~~~~~~~~~~

Three depreciation methods are supported:

.. code-block:: python

   # Straight-line
   plant.update_configuration({
       "depreciation": {"method": "straight_line"}
   })

   # Declining balance
   plant.update_configuration({
       "depreciation": {"method": "declining_balance", "rate": 0.2}
   })

   # MACRS (US tax depreciation) — specify recovery period in years
   plant.update_configuration({
       "depreciation": {"method": "macrs", "class": 7}
   })

Scenario analysis
-----------------

Arrays can be passed for key financial parameters to evaluate multiple
scenarios at once:

.. code-block:: python

   import numpy as np

   plant.update_configuration({
       "interest_rate": np.linspace(0.05, 0.15, 11),
   })
   plant.calculate_all()
   # plant.npv, plant.irr, etc. are now arrays over the interest-rate sweep

See also
--------

* :class:`~openpytea.plant.Plant` — full API reference
* :doc:`analysis` — sensitivity and Monte Carlo analysis
