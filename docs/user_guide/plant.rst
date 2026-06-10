Plant-Level TEA
===============

The :class:`~openpytea.plant.Plant` class is the core of OpenPyTEA. It
aggregates equipment objects and financial assumptions into a full
techno-economic assessment, covering:

* **Fixed CAPEX** (ISBL / OSBL / D&E / Contingency) via process-type multipliers
* **Location-adjusted costs** via country/region factors
* **Variable OPEX** from itemized consumption and price data
* **Fixed OPEX** including automatic labor estimation, maintenance, taxes, overheads, and working capital
* **Revenue** for a main product and optional co-products
* **Cash flow, NPV, LCOP, payback time, ROI, IRR** over the project lifetime
* **Scenario arrays** — pass an array for any scalar parameter to evaluate multiple scenarios in one call

To see the outputs of all code examples below, refer to the
`walkthrough notebook <https://github.com/pbtamarona/OpenPyTEA/blob/main/walkthrough.ipynb>`_.

Creating a ``Plant``
---------------------

.. code-block:: python

   from openpytea import Plant, Equipment

   # Define equipment first (see Equipment guide)
   # hx, comp1, comp2 = Equipment(...)

   config = {
       "plant_name": "Demo Plant",                  # used in plots
       # Basic plant information
       "process_type": "Fluids",                    # "Solids" | "Fluids" | "Mixed"
       "country": "United States",                  # optional, defaults to "United States"
       "region": "Gulf Coast",                      # optional, defaults to "Gulf Coast"
       "currency": "USD",                           # optional, defaults to "USD"
                                                    # use \$ when using symbol to avoid syntax errors
       "exchange_rate": 1.0,                        # optional, defaults to 1.0
       "equipment": [hx, comp1, comp2],             # list of Equipment objects
       "interest_rate": 0.09,                       # optional, defaults to 0.09
       "project_lifetime": 30,                      # int ≥ 3, optional, defaults to 20
       "plant_utilization": 0.90,                   # 0–1, optional, defaults to 1
       "tax_rate": 0.25,                            # 0–1, not used in LCOP, defaults to 0

       # Operator labor
       "operator_hourly_rate": {"rate": 35},        # USD/hr, optional, defaults to $38.11/hr
       "working_weeks_per_year": 46,                # optional, defaults to 49
       "working_shifts_per_week": 5,                # optional, defaults to 5

       # Products — first entry is the main product for LCOP calculations
       "plant_products": {
           "methanol": {
               "production": 125_000,               # units/yr
               "price": 2.5,                        # USD/unit (not needed for LCOP)
           },
           "hydrogen": {                            # co-product
               "production": 100_000,
               "price": 2.0,
           },
       },

       # Variable OPEX — consumables and utilities
       "variable_opex_inputs": {
           "electricity":   {"consumption": 2.2e6, "price": 0.08},   # units/yr, USD/unit
           "cooling_water": {"consumption": 1.6e6, "price": 0.0007},
       },

       # Additional CAPEX and OPEX
       "working_capital": None,                     # USD; defaults to 15% of FCI
       "additional_capex_cost": [500_000, 200_000], # one-off CAPEX events during operation
       "additional_capex_years": [8, 15],           # years in which they occur

       # Depreciation (optional; defaults to straight-line if omitted)
       "depreciation": {
           "method": "macrs",                       # "straight_line" | "declining_balance" | "macrs"
           "macrs_class": 7,                        # 3, 5, 7, 10, 15, or 20
           "service_start_year": 2,                 # first operating year in the production ramp
       },
   }

   demo_plant = Plant(config)
   print(demo_plant)
   demo_plant.calculate_all(print_results=True)


Updating configuration
-----------------------

Any setting can be changed after construction without rebuilding the plant.
Nested dictionaries (e.g., ``variable_opex_inputs``, ``plant_products``) are
merged recursively, so unspecified sub-keys are preserved:

.. code-block:: python

   plant.update_configuration({
       "interest_rate": 0.08,
       "variable_opex_inputs": {
           "steam": {"consumption": 4.0e5, "price": 0.02},
       },
   })
   plant.calculate_all()

Configuration reference
------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 15 57

   * - Key
     - Default
     - Description
   * - ``plant_name``
     - ``""``
     - Name used in plots and reports.
   * - ``process_type``
     - required
     - ``"Solids"``, ``"Fluids"``, or ``"Mixed"`` — sets default cost factors.
   * - ``country``
     - ``"United States"``
     - Country for location factor lookup.
   * - ``region``
     - ``"Gulf Coast"``
     - Region within the country (required for countries with regional factors).
   * - ``loc_factor``
     - ``None``
     - Direct location factor override. Bypasses ``country``/``region`` lookup.
   * - ``currency``
     - ``"USD"``
     - Currency label for display. Use ``\$`` when using symbol to avoid RST syntax errors.
   * - ``exchange_rate``
     - ``1.0``
     - Conversion from USD to ``currency`` (1 USD = ``exchange_rate`` × currency unit).
   * - ``equipment``
     - required
     - List of :class:`~openpytea.equipment.Equipment` objects.
   * - ``interest_rate``
     - ``0.09``
     - Discount rate for NPV, capital recovery, and working-capital interest.
   * - ``project_lifetime``
     - ``20``
     - Project duration in years (integer ≥ 3). Accepts an array for scenario analysis.
   * - ``plant_utilization``
     - ``1.0``
     - Annual capacity utilization (0–1). Scales production and variable OPEX.
   * - ``tax_rate``
     - ``0``
     - Corporate income tax rate (0–1). Not applied to levelized cost calculations.
   * - ``working_capital``
     - ``None``
     - Working capital in USD. Defaults to 0.15 × FCI when ``None``.
   * - ``plant_products``
     - ``{}``
     - Products dict. First entry is the main product for LCOP; others are co-products.
   * - ``variable_opex_inputs``
     - ``{}``
     - Utilities and raw materials, each with ``consumption`` (units/yr) and ``price`` (USD/unit).
   * - ``operator_hourly_rate``
     - ``{"rate": 38.11}``
     - Operator wage in USD/hr.
   * - ``working_weeks_per_year``
     - ``49``
     - Annual working weeks per operator.
   * - ``working_shifts_per_week``
     - ``5``
     - Shifts per operator per week.
   * - ``operating_shifts_per_day``
     - ``3``
     - Daily operating shifts (continuous plants typically run 3).
   * - ``operators_hired``
     - auto
     - Total headcount. Computed automatically from process type; override to fix.
   * - ``operators_per_shift``
     - auto
     - Operators per shift. Computed automatically; override to fix.
   * - ``fixed_capital_factors``
     - ``{}``
     - Per-key overrides for OSBL, D&E, and contingency factors.
   * - ``fixed_capital_components``
     - ``{}``
     - Absolute cost overrides for any FCI component (takes precedence over factors).
   * - ``fixed_opex_factors``
     - ``{}``
     - Per-key factor overrides for fixed OPEX components.
   * - ``fixed_opex_components``
     - ``{}``
     - Absolute cost overrides for any fixed OPEX component.
   * - ``capex_ramp``
     - ``[0.3, 0.6, 0.1]``
     - CAPEX spending fractions by construction year (must sum to 1.0).
   * - ``production_ramp``
     - ``[0, 0, 0.4, 0.8]``
     - Capacity fractions by project year. Years beyond the list default to 1.0.
   * - ``depreciation``
     - ``{}``
     - Depreciation settings (method, life, etc.) — see :ref:`depreciation`.

Capital cost structure
-----------------------

The fixed capital investment (FCI) is assembled in four layers:

.. math::

   \text{FCI} = \text{ISBL} \times (1 + f_{\text{os}}) \times (1 + f_{\text{de}} + f_X) \times LF

where :math:`\text{ISBL}` is the sum of equipment direct costs, :math:`f_{\text{os}}` is the
OSBL factor, :math:`f_{\text{de}}` is the design & engineering factor, :math:`f_X` is
the contingency factor, and :math:`LF` is the location factor.

*Source: Towler & Sinnott (2022)*

Default factors by process type:

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 20

   * - Factor
     - Solids
     - Fluids
     - Mixed
   * - OSBL :math:`(f_{\text{os}})`
     - 0.40
     - 0.30
     - 0.40
   * - D&E :math:`(f_{\text{de}})`
     - 0.20
     - 0.30
     - 0.25
   * - Contingency :math:`(f_X)`
     - 0.10
     - 0.10
     - 0.10

*Source: Towler & Sinnott (2022)*

Override individual factors or set absolute component values:

.. code-block:: python

   plant.update_configuration({
       "fixed_capital_factors": {
           "osbl": 0.25,              # override factor (default 0.30 for Fluids)
           "de": 0.35,                # override factor (default 0.30)
       },
       "fixed_capital_components": {
           "contingency": 15_000_000, # absolute value, overrides factor
       },
   })
   plant.calculate_fixed_capital(print_results=True)

Access individual components after calculation:

.. code-block:: python

   print(f"ISBL        : ${plant.isbl:,.0f}")
   print(f"OSBL        : ${plant.osbl:,.0f}")
   print(f"D&E         : ${plant.dne:,.0f}")
   print(f"Contingency : ${plant.contingency:,.0f}")
   print(f"FCI         : ${plant.fci:,.0f}")

Location factors
~~~~~~~~~~~~~~~~

Location factors scale the ISBL to reflect regional construction cost
differences relative to the US Gulf Coast (LF = 1.00).

*Source: Towler & Sinnott (2022)*

.. list-table::
   :header-rows: 1
   :widths: 60 20

   * - Country / Region
     - Factor
   * - United States — Gulf Coast
     - 1.00
   * - United States — East Coast
     - 1.04
   * - United States — Midwest
     - 1.02
   * - United States — West Coast
     - 1.07
   * - Canada — Ontario
     - 1.00
   * - Canada — Fort McMurray
     - 1.60
   * - Mexico
     - 1.03
   * - Brazil
     - 1.14
   * - China — Imported
     - 1.12
   * - China — Indigenous
     - 0.61
   * - Japan
     - 1.26
   * - Southeast Asia
     - 1.12
   * - Australia
     - 1.21
   * - India
     - 1.02
   * - Middle East
     - 1.07
   * - France
     - 1.13
   * - Germany
     - 1.11
   * - Italy
     - 1.14
   * - Netherlands
     - 1.19
   * - Russia
     - 1.53
   * - United Kingdom
     - 1.02

To use a country not in the table, set ``loc_factor`` directly:

.. code-block:: python

   plant.update_configuration({"loc_factor": 1.15})

Fixed OPEX
-----------

Fixed operating costs do not vary with production rate. The table below
lists every component, its default calculation basis, and the
``fixed_opex_factors`` key used to override its multiplier.

*Source: Turton et al. (2018)*

.. list-table::
   :header-rows: 1
   :widths: 32 38 30

   * - Component
     - Default formula
     - ``fixed_opex_factors`` key
   * - Operating labor
     - From shift schedule × hourly rate
     - —
   * - Supervision
     - 0.25 × operating labor
     - ``"supervision"``
   * - Direct salary overhead
     - 0.50 × (labor + supervision)
     - ``"direct_salary_overhead"``
   * - Laboratory charges
     - 0.10 × operating labor
     - ``"laboratory_charges"``
   * - Maintenance
     - 0.05 × ISBL
     - ``"maintenance"``
   * - Taxes & insurance
     - 0.015 × ISBL
     - ``"taxes_insurance"``
   * - Rent of land
     - 0.015 × (ISBL + OSBL)
     - ``"rent_of_land"``
   * - Environmental charges
     - 0.010 × (ISBL + OSBL)
     - ``"environmental_charges"``
   * - Operating supplies
     - 0.009 × ISBL
     - ``"operating_supplies"``
   * - General plant overhead
     - 0.65 × (labor + supervision + overhead)
     - ``"general_plant_overhead"``
   * - Interest on working capital
     - working capital × interest rate
     - ``"working_capital"`` (default: 0.15 × FCI)
   * - Patents & royalties
     - 0.02 × cash cost of production\*
     - ``"patents_royalties"``
   * - Distribution & selling
     - 0.02 × cash cost of production\*
     - ``"distribution_selling"``
   * - R&D
     - 0.03 × cash cost of production\*
     - ``"rnd"``

\* Cash cost of production = (variable + fixed costs so far) / (1 − sum of the three rates above), ensuring these fractions are expressed consistently as a share of total cash cost.

Override factors or fix absolute component values:

.. code-block:: python

   plant.update_configuration({
       "fixed_opex_factors": {
           "maintenance": 0.06,        # 6% of ISBL instead of 5%
           "rent_of_land": 0.01,       # 1% instead of 1.5%
           "rnd": 0.0,                 # zero out R&D
       },
       "fixed_opex_components": {
           "supervision_costs": 100_000,  # fixed value, overrides factor
       },
   })
   plant.calculate_fixed_opex(print_results=True)

Access components after calculation:

.. code-block:: python

   print(f"Operating labor : ${plant.operating_labor_costs:,.0f}")
   print(f"Supervision     : ${plant.supervision_costs:,.0f}")
   print(f"Maintenance     : ${plant.maintenance_costs:,.0f}")
   # ... and so on

Labor modeling
~~~~~~~~~~~~~~

Operating labor cost is calculated as:

.. math::

   C_{\text{labor}} = N_{\text{hired}} \times H_{\text{year}} \times r

where :math:`H_{\text{year}} = W_{\text{weeks}} \times W_{\text{shifts}} \times (24 / S_{\text{day}})` is
the working hours per operator per year and :math:`r` is the hourly rate (default **$38.11/hr**).

**Operators per shift** is estimated from the equipment list using an empirical correlation:

.. math::

   N_{\text{shift}} = \sqrt{6.29 + 31.7 \cdot N_{\text{solid}}^2 + 0.23 \cdot N_{\text{fluid}}}

where :math:`N_{\text{solid}}` and :math:`N_{\text{fluid}}` are the numbers of solid-handling
and fluid-handling process steps respectively (:math:`N_{\text{solid}} \leq 2`).

**Total operators hired** accounts for the continuous plant schedule versus each operator's
working schedule:

.. math::

   N_{\text{hired}} = \left\lceil N_{\text{shift}} \times
   \frac{365 \times S_{\text{day}}}{W_{\text{weeks}} \times W_{\text{shifts}}} \right\rceil

*Source: Turton et al. (2018)*

Schedule parameters are set via ``working_weeks_per_year`` (default: 49),
``working_shifts_per_week`` (default: 5), and ``operating_shifts_per_day`` (default: 3).

Any part of the labor calculation can be bypassed:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Config key
     - Effect
   * - ``operator_hourly_rate: {"rate": r}``
     - Sets the hourly wage rate.
   * - ``operators_per_shift``
     - Skips the empirical formula; uses this value directly.
   * - ``operators_hired``
     - Skips both formula and hired count; uses this value directly. Set to ``None`` to revert to auto.
   * - ``working_weeks_per_year``, ``working_shifts_per_week``, ``operating_shifts_per_day``
     - Adjust the shift schedule for both hired count and annual hours.

.. code-block:: python

   # Print auto-calculated defaults first
   print(f"Operators per shift : {plant.calculate_operators_per_shift():.1f}")
   print(f"Operators hired     : {plant.calculate_operators_hired()}")

   # Override manually
   plant.update_configuration({
       "operators_hired": 12,
       "working_weeks_per_year": 46,
       "working_shifts_per_week": 5,
       "operating_shifts_per_day": 3,
       "operator_hourly_rate": {"rate": 42.0},
   })
   plant.calculate_fixed_opex()
   print(f"Operating labor costs : ${plant.operating_labor_costs:,.0f}/yr")

Variable OPEX
--------------

Variable costs scale with production and are calculated as:

.. math::

   C_{\text{var}} = \sum_i \text{consumption}_i \times \text{price}_i
   \times 365 \times \text{plant\_utilization}

Each entry in ``variable_opex_inputs`` needs a ``consumption`` (annual
quantity in any consistent unit) and a ``price`` (USD per unit):

.. code-block:: python

   plant.update_configuration({
       "variable_opex_inputs": {
           "electricity":   {"consumption": 1.4e6, "price": 0.075},
           "cooling_water": {"consumption": 1.6e6, "price": 0.0007},
           "steam":         {"consumption": 4.0e5, "price": 0.02},
           "natural_gas":   {"consumption": 1.0e5, "price": 0.035},
       },
   })
   plant.calculate_variable_opex(print_results=True)

Revenue
--------

The first product in ``plant_products`` is the main product used for LCOP
calculations. Additional products are treated as co-products and their
revenue is credited against the main product cost.
Annual revenue per product is:

.. math::

   R = \text{production} \times \text{price} \times 365 \times \text{plant\_utilization}

.. code-block:: python

   plant.update_configuration({
       "plant_products": {
           "hydrogen":  {"production": 10_000, "price": 3.0},   # main product
           "oxygen":    {"production": 80_000, "price": 0.05},  # co-product
       },
   })
   plant.calculate_revenue(print_results=True)

Cash flow
----------

The cash flow represents the net annual financial performance of the plant, combining revenues, operating costs, capital investments, taxes, and depreciation.
It is generated by calling:

.. code-block:: python

   plant.calculate_cash_flow(print_results=True)

The method returns a styled DataFrame and, when ``print_results=True``, displays the full year-by-year table. The table has the following columns (all monetary values in the plant's currency):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Description
   * - Year
     - Project year (1 = first construction year).
   * - Capital cost
     - CAPEX spending (positive during construction, negative when working capital is released at end of life).
   * - Revenue
     - Annual product revenue, scaled by the production ramp.
   * - Cash cost
     - Total annual OPEX (fixed + variable), scaled by the production ramp.
   * - Gross profit
     - Revenue − Cash cost.
   * - Depreciation
     - Annual depreciation charge for the selected method.
   * - Taxable income
     - Gross profit − Depreciation (from the previous year, due to one-year tax lag).
   * - Tax paid
     - ``tax_rate`` × previous year's taxable income.
   * - Cash flow
     - Gross profit − Tax paid − Capital cost.

A representative output for a 20-year project (values in USD, abbreviated):

.. code-block:: text

   Year  Capital cost      Revenue      Cash cost    Gross profit  Depreciation  Taxable income    Tax paid     Cash flow
      1   -15,000,000            0              0               0             0               0           0   -15,000,000
      2   -30,000,000            0              0               0             0               0           0   -30,000,000
      3    -5,000,000    8,000,000      4,500,000       3,500,000     2,500,000       1,000,000           0    -1,500,000
      4             0   16,000,000      4,500,000      11,500,000     2,500,000       3,500,000     250,000    11,250,000
      5             0   20,000,000      4,500,000      15,500,000     2,500,000      11,500,000     875,000    14,625,000
    ...          ...          ...            ...             ...           ...             ...         ...           ...
     20     5,000,000   20,000,000      4,500,000      15,500,000             0      15,500,000   3,875,000    16,625,000



CAPEX ramp
~~~~~~~~~~

Capital spending is spread across construction years. The default profile:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Year
     - CAPEX fraction
     - Description
   * - 0
     - 30%
     - Initial design and early procurement
   * - 1
     - 60%
     - Major equipment installation
   * - 2
     - 10%
     - Commissioning and start-up
   * - Final year
     - —
     - Working capital released (negative CAPEX)

Override with a list that sums to 1.0:

.. code-block:: python

   plant.update_configuration({
       "capex_ramp": [0.2, 0.5, 0.2, 0.1],  # 4-year build
   })

Production ramp
~~~~~~~~~~~~~~~

Plant output ramps up gradually from zero. The default profile:

.. list-table::
   :header-rows: 1
   :widths: 20 35

   * - Year
     - Production level
   * - 0–1
     - 0% (construction)
   * - 2
     - 40%
   * - 3
     - 80%
   * - 4+
     - 100% (steady state)

Annual production in year :math:`t`:

.. math::

   Q_t = \text{daily\_production} \times 365 \times \text{plant\_utilization}
   \times \text{ramp\_factor}_t

Override with a list of capacity fractions (0–1). Years beyond the list
default to 1.0:

.. code-block:: python

   plant.update_configuration({
       "production_ramp": [0, 0, 0, 0.3, 0.6, 0.9],  # slower 6-year ramp
   })

Cash flow formula
~~~~~~~~~~~~~~~~~

For each operating year :math:`t`:

.. math::

   \text{Cash Flow}_t =
   \underbrace{(R_t - \text{OPEX}_t)}_{\text{Gross Profit}}
   - \text{Tax}_t - \text{CAPEX}_t

where tax in year :math:`t` is based on taxable income from year
:math:`t-1` (one-year lag), and taxable income = Gross Profit − Depreciation.

.. _depreciation:

Depreciation
~~~~~~~~~~~~

Three methods are supported:

.. code-block:: python

   # Straight-line
   plant.update_configuration({
       "depreciation": {
           "method": "straight_line",
           "life": 12,
           "salvage_fraction": 0.05,
           "service_start_year": 2,
       }
   })

   # Declining balance (200% DDB)
   plant.update_configuration({
       "depreciation": {
           "method": "declining_balance",
           "life": 10,
           "db_factor": 2.0,           # 2.0 = 200% DDB, 1.5 = 150% DB
           "salvage_fraction": 0.10,
           "service_start_year": 2,
       }
   })

   # MACRS (US tax depreciation)
   plant.update_configuration({
       "depreciation": {
           "method": "macrs",
           "class": 7,                 # recovery period in years
       }
   })

Financial metrics
------------------

Call :meth:`~openpytea.plant.Plant.calculate_all` to compute everything
at once, or run each method individually after
:meth:`~openpytea.plant.Plant.calculate_cash_flow`.

Net Present Value (NPV)
~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   NPV = \sum_{t=1}^{t_p} \frac{\text{Cash Flow}_t}{(1 + i)^t}

.. code-block:: python

   plant.calculate_npv(print_results=True)
   print(plant.npv)          # scalar
   print(plant.npv_array)    # cumulative NPV by year

Levelized Cost of Product (LCOP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Break-even selling price of the main product that sets NPV = 0:

.. math::

   LCOP = \frac{
       \sum_{t=1}^{t_p} \dfrac{CAPEX_t + OPEX_t - R^{\text{side}}_t}{(1+i)^t}
   }{
       \sum_{t=1}^{t_p} \dfrac{Q_t}{(1+i)^t}
   }

where :math:`R^{\text{side}}_t` is co-product revenue and :math:`Q_t` is
main-product production in year :math:`t`.

.. code-block:: python

   plant.calculate_levelized_cost(print_results=True)
   print(plant.levelized_cost)

Payback time (PBT)
~~~~~~~~~~~~~~~~~~~

First year when cumulative undiscounted cash flow ≥ 0:

.. math::

   PBT = \frac{FCI}{\overline{CF}}

.. code-block:: python

   plant.calculate_payback_time(print_results=True)
   print(plant.payback_time)

Return on Investment (ROI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   ROI = \frac{\sum_{t=1}^{t_p} \text{Net Profit}_t}{t_p \times (FCI + WC)}

.. code-block:: python

   plant.calculate_roi(print_results=True)
   print(plant.roi)

Internal Rate of Return (IRR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Discount rate :math:`r` that sets NPV = 0:

.. math::

   0 = \sum_{t=1}^{t_p} \frac{CF_t}{(1 + r)^t}

.. code-block:: python

   plant.calculate_irr(print_results=True)
   print(plant.irr)

Scenario analysis
-----------------

Pass a NumPy array for any scalar parameter to evaluate multiple scenarios
simultaneously. All financial metrics become arrays of the same length:

.. code-block:: python

   import numpy as np

   plant.update_configuration({
       "interest_rate": np.linspace(0.05, 0.15, 11),
   })
   plant.calculate_all()
   # plant.npv, plant.irr, plant.levelized_cost, etc. are now length-11 arrays

See also
--------

* :class:`~openpytea.plant.Plant` — full API reference
* :doc:`analysis` — sensitivity and Monte Carlo analysis
* `Walkthrough notebook <https://github.com/pbtamarona/OpenPyTEA/blob/main/walkthrough.ipynb>`_ — end-to-end worked example

References
----------

* Towler, G.; Sinnott, R. *Chemical Engineering Design*, 3rd ed.;
  Elsevier, 2022. https://doi.org/10.1016/C2019-0-02025-0
* Turton, R.; Shaeiwitz, J. A.; Bhattacharyya, D.; Whiting, W. B.
  *Analysis, Synthesis, and Design of Chemical Processes*, 5th ed.;
  Prentice Hall, 2018.
