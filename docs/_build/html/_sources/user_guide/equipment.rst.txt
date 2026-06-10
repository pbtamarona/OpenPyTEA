Equipment Cost Estimation
=========================

The :mod:`openpytea.equipment` module estimates purchased and installed costs
for individual process units using published cost correlations, automatic CEPCI
inflation adjustment, and process/material installation factors.

How costs are estimated
-----------------------

Purchased cost correlations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two correlation forms are supported:

**Power-law**

.. math::

   C_p = a + b \cdot S^n

**Quadratic log-log**

.. math::

   \log_{10} C_p = K_1 + K_2 \log_{10} S + K_3 \left(\log_{10} S\right)^2

where :math:`S` is the equipment size parameter (e.g., shaft power in kW,
heat transfer area in m²) and :math:`C_p` is the purchased cost in the
correlation's reference year (USD).

All correlations and their coefficients are stored in
:download:`cost_correlations.csv <../../src/openpytea/data/cost_correlations.csv>`.

**OpenPyTEA does not require a database match.** You can bypass the built-in
correlations entirely and supply your own ``purchased_cost`` directly — for
vendor quotes, proprietary data, or equipment types not yet in the database.
See Example 3 below.

CEPCI inflation adjustment
~~~~~~~~~~~~~~~~~~~~~~~~~~

Purchased costs are inflated from the correlation's reference year to the
target year using the Chemical Engineering Plant Cost Index (CEPCI):

.. math::

   C_{\text{target}} = C_{\text{ref}} \times
   \frac{\text{CEPCI}_{\text{target}}}{\text{CEPCI}_{\text{ref}}}

Historical CEPCI values are bundled with the package in
:download:`cepci_values.csv <../../src/openpytea/data/cepci_values.csv>`.
The default target year is 2024.

The bundled values are sourced from the
`University of Manchester CEPCI table <https://www.training.itservices.manchester.ac.uk/public/gced/CEPCI.html?reactors/CEPCI/index.html>`_
(accessed 7 April 2026). **Users are encouraged to verify these values and,
if more recent or detailed data are available, replace them by editing
``cepci_values.csv`` directly before running their analysis.**

Direct (installed) cost
~~~~~~~~~~~~~~~~~~~~~~~

The direct cost adds installation contributions on top of the purchased cost:

.. math::

   C_D = C_p \left[
       (1 + f_p) \cdot f_m
       + \left( f_{er} + f_{el} + f_i + f_c + f_s + f_l \right)
   \right]

where :math:`f_m` is the material factor and :math:`f_p`, :math:`f_{er}`,
:math:`f_{el}`, :math:`f_i`, :math:`f_c`, :math:`f_s`, :math:`f_l` are the
piping, erection, electrical, instrumentation, civil, structural, and lagging
factors respectively. Default installation factor values depend on the ``process_type``
(see :ref:`process-factors` below). **All factors can be overridden per equipment
item via constructor keyword arguments** — see Example 8 for details.

*Source: Towler & Sinnott (2022)*

The ``Equipment`` class
-----------------------

.. code-block:: python

   from openpytea import Equipment

Each :class:`~openpytea.equipment.Equipment` object represents one piece of
process equipment. On construction it automatically:

1. Looks up the matching cost correlation from ``cost_correlations.csv``.
2. Computes the purchased cost, with automatic parallelization if the size
   parameter exceeds the correlation's upper bound.
3. Inflates the cost to the target year using CEPCI.
4. Applies process and material factors to produce the direct (installed) cost.

Constructor parameters
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Type
     - Description
   * - ``name``
     - str
     - Identifier for this equipment item (used in plots and reports).
   * - ``param``
     - float
     - Size/capacity parameter. Units depend on the equipment type —
       check ``cost_correlations.csv``. Ignored when ``purchased_cost``
       is given.
   * - ``process_type``
     - str
     - ``"Solids"``, ``"Fluids"``, ``"Mixed"``, or ``"Electrical"`` —
       controls default installation factor values.
   * - ``category``
     - str
     - Equipment category — must match a row in ``cost_correlations.csv``
       (case-insensitive).
   * - ``type``
     - str or None
     - Equipment sub-type within the category. Required when a category
       has multiple types.
   * - ``material``
     - str
     - Construction material — controls the material factor :math:`f_m`.
       Default: ``"Carbon steel"``.
   * - ``target_year``
     - int
     - Year to inflate costs to. Default: ``2024``.
   * - ``purchased_cost``
     - float or None
     - Supply your own purchased cost and bypass the correlation entirely.
   * - ``cost_year``
     - int or None
     - Reference year of a manually supplied ``purchased_cost``. If given,
       CEPCI inflation is applied from this year to ``target_year``.
   * - ``cost_func``
     - str or None
     - Explicit correlation key (the ``key`` column in the CSV). Use this
       to select a specific correlation when multiple exist for the same
       category/type.
   * - ``num_units``
     - int or None
     - Override the number of parallel units. By default this is set
       automatically by the parallelization logic.
   * - ``piping_factor``, ``erection_factor``, … ``lagging_factor``
     - float or None
     - Per-factor overrides. ``None`` uses the ``process_type`` default.
   * - ``material_factor``
     - float or None
     - Override the material factor. ``None`` uses the material table.

Usage examples
--------------

The examples below show the main ways to create ``Equipment`` objects.
To see the printed outputs of each code cell, refer to the
`walkthrough notebook <https://github.com/pbtamarona/OpenPyTEA/blob/main/walkthrough.ipynb>`_.

Example 1 — Standard usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a heat exchanger using a correlation from the database:

.. code-block:: python

   from openpytea import Equipment

   hx = Equipment(
       name="HX-101",
       param=250,                       # heat transfer area in m²
       process_type="Fluids",
       category="Heat exchangers",
       type="Floating head",
       material="316 stainless steel",
       target_year=2024,
   )

   print(hx)
   print(f"Purchased cost : ${hx.purchased_cost:,.0f}")
   print(f"Direct cost    : ${hx.direct_cost:,.0f}")

Example 2 — Selecting a specific correlation key
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When multiple correlations cover the same equipment type (e.g., compressors
from different studies), use ``cost_func`` to pin the exact database key:

.. code-block:: python

   comp = Equipment(
       name="COMP-01",
       param=1,                         # net electric power, MW
       process_type="Fluids",
       category="Compressors, fans, & blowers",
       type="Compressor, centrifugal",
       material="Carbon steel",
       cost_func="co2_compressor_manzolini_2011",
   )
   print(comp)

Example 3 — Manually specified purchased cost
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Skip the correlation entirely and supply your own cost. If you also provide
``cost_year``, CEPCI inflation to ``target_year`` is applied automatically:

.. code-block:: python

   dryer = Equipment(
       name="Rotary Dryer D-301",
       param=0,                         # ignored when purchased_cost is set
       process_type="Solids",
       category="Dryers",
       material="Carbon steel",
       purchased_cost=1_500_000,        # vendor quote in 2021 USD
       cost_year=2021,                  # inflated to target_year=2024
   )
   print(dryer)

Example 4 — Automatic parallelization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``param`` exceeds the correlation's upper capacity limit, the module
splits the load into the minimum number of equal parallel units:

.. code-block:: python

   # The centrifugal compressor correlation is valid up to 30 000 kW.
   # Requesting 50 000 kW triggers automatic splitting into 2 units.
   large_comp = Equipment(
       name="COMP-LARGE",
       param=50_000,                    # driver power in kW
       process_type="Fluids",
       category="Compressors, fans, & blowers",
       type="Compressor, centrifugal",
       material="Carbon steel",
   )
   print(large_comp)
   print(f"Number of parallel units: {large_comp.num_units}")

Example 5 — Inflation to a custom target year
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   hx_2020 = Equipment(
       name="HX-102",
       param=250,
       process_type="Fluids",
       category="Heat exchangers",
       type="Floating head",
       material="316 stainless steel",
       target_year=2020,                # inflate to 2020 instead of 2024
   )
   print(hx_2020)

Example 6 — Fixing the number of units manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   fridge = Equipment(
       name="Refrigerator R-201",
       param=180,
       process_type="Fluids",
       category="Utilities",
       type="Packaged mechanical refrigerator",
       num_units=3,                     # bypass auto-parallelization
   )
   print(fridge)

Example 7 — Comparing materials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The material factor :math:`f_m` multiplies the installed cost. Here the same
agitator is costed in carbon steel versus Hastelloy C:

.. code-block:: python

   mixer_cs = Equipment(
       name="Agitator M-101",
       param=100,
       process_type="Fluids",
       category="Agitators, blenders, & mixers",
       type="Propeller mixer",
       material="Carbon steel",         # fm = 1.00
   )

   mixer_alloy = Equipment(
       name="Agitator M-101 (Hastelloy)",
       param=100,
       process_type="Fluids",
       category="Agitators, blenders, & mixers",
       type="Propeller mixer",
       material="Hastelloy C",          # fm = 1.55
   )

   print(f"Carbon steel direct cost : ${mixer_cs.direct_cost:,.0f}")
   print(f"Hastelloy C direct cost  : ${mixer_alloy.direct_cost:,.0f}")

Example 8 — Overriding installation factors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Individual installation factors can be overridden without affecting the rest.
The class attributes ``process_factors`` and ``material_factors`` show all
defaults:

.. code-block:: python

   # Inspect defaults first
   print(Equipment.process_factors["Fluids"])
   print(Equipment.material_factors["316 stainless steel"])

   reactor = Equipment(
       name="Reactor R-101",
       param=50,
       process_type="Fluids",
       category="Reactors",
       type="Glass-lined agitated",
       material="316 stainless steel",
       piping_factor=0.95,              # override default 0.80
       material_factor=1.4,             # override default 1.30
   )
   print(f"piping_factor   : {reactor.piping_factor}")
   print(f"material_factor : {reactor.material_factor}")

Listing available equipment
---------------------------

Print all categories and types in the built-in database:

.. code-block:: python

   from openpytea.equipment import CostCorrelationDB, COST_DB_DF

   db = CostCorrelationDB(COST_DB_DF)

   # All unique categories
   print(db.df["category"].unique())

   # All types and metadata for a specific category
   mask = db.df["category"].str.lower() == "heat exchangers"
   print(db.df.loc[mask, ["type", "form", "cost_year", "source"]])

You can also download the full database:
:download:`cost_correlations.csv <../../src/openpytea/data/cost_correlations.csv>`

.. _materials:

Available materials
-------------------

The table below lists the valid ``material`` strings and their factors
:math:`f_m`. Costs are relative to carbon steel (= 1.0).

*Source: Towler & Sinnott (2022)*

.. list-table::
   :header-rows: 1
   :widths: 55 20

   * - Material
     - Factor :math:`f_m`
   * - ``"Carbon steel"``
     - 1.00
   * - ``"Aluminum"``
     - 1.07
   * - ``"Bronze"``
     - 1.07
   * - ``"Cast steel"``
     - 1.10
   * - ``"304 stainless steel"``
     - 1.30
   * - ``"316 stainless steel"``
     - 1.30
   * - ``"321 stainless steel"``
     - 1.50
   * - ``"Hastelloy C"``
     - 1.55
   * - ``"Monel"``
     - 1.65
   * - ``"Nickel"``
     - 1.70
   * - ``"Inconel"``
     - 1.70

.. _process-factors:

Process installation factors
-----------------------------

Default installation factors by ``process_type``. Any factor can be
overridden per equipment item via the corresponding constructor keyword
(e.g., ``piping_factor=0.95``).

*Source: Towler & Sinnott (2022)*

.. list-table::
   :header-rows: 1
   :widths: 32 14 14 14 14

   * - Factor
     - Solids
     - Fluids
     - Mixed
     - Electrical
   * - Erection :math:`(f_{er})`
     - 0.60
     - 0.30
     - 0.50
     - 0.40
   * - Piping :math:`(f_p)`
     - 0.20
     - 0.80
     - 0.60
     - 0.10
   * - Instrumentation :math:`(f_i)`
     - 0.20
     - 0.30
     - 0.30
     - 0.70
   * - Electrical :math:`(f_{el})`
     - 0.15
     - 0.20
     - 0.20
     - 0.70
   * - Civil :math:`(f_c)`
     - 0.20
     - 0.30
     - 0.30
     - 0.20
   * - Structural steel :math:`(f_s)`
     - 0.10
     - 0.20
     - 0.20
     - 0.10
   * - Lagging & painting :math:`(f_l)`
     - 0.05
     - 0.10
     - 0.10
     - 0.10

Standalone inflation adjustment
--------------------------------

The :func:`~openpytea.equipment.inflation_adjustment` function can be used
independently to convert any cost between years:

.. code-block:: python

   from openpytea import inflation_adjustment

   # Convert a $500 000 cost quoted in 2015 to 2024 USD
   adjusted = inflation_adjustment(500_000, cost_year=2015, target_year=2024)
   print(f"2015 cost: $500,000  →  2024 cost: ${adjusted:,.0f}")

See also
--------

* :class:`~openpytea.equipment.Equipment` — full API reference
* :class:`~openpytea.equipment.CostCorrelationDB` — database interface
* :func:`~openpytea.equipment.inflation_adjustment` — CEPCI utility
* `Walkthrough notebook <https://github.com/pbtamarona/OpenPyTEA/blob/main/walkthrough.ipynb>`_ — end-to-end worked example

.. _equip-references:

References
----------

* Towler, G.; Sinnott, R. *Chemical Engineering Design*, 3rd ed.;
  Elsevier, 2022. https://doi.org/10.1016/C2019-0-02025-0
