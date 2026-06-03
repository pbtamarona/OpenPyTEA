Equipment Cost Estimation
=========================

The :mod:`openpytea.equipment` module provides equipment-level capital cost
estimation based on published cost correlations and CEPCI inflation adjustment.

How costs are estimated
-----------------------

OpenPyTEA uses **power-law** and **quad log-log** correlations of the form:

.. math::

   C_p = a + b \cdot S^n \quad \text{(power-law)}

.. math::

   \log C_p = k_1 + k_2 \log S + k_3 (\log S)^2 \quad \text{(quad log-log)}

where :math:`S` is the equipment size parameter (e.g., power in kW, area in
m²) and :math:`C_p` is the purchased cost in the correlation's reference year.

Costs are then adjusted to the target year using the **Chemical Engineering
Plant Cost Index (CEPCI)**:

.. math::

   C_{\text{target}} = C_{\text{ref}} \times
   \frac{\text{CEPCI}_{\text{target}}}{\text{CEPCI}_{\text{ref}}}

The purchased cost is subsequently scaled by **process factors** (piping,
instrumentation, civil works, …) and a **material factor** to give the
installed *direct cost*.

The ``Equipment`` class
-----------------------

.. code-block:: python

   from openpytea import Equipment

   heat_exchanger = Equipment(
       name="HX-01",
       param=250,                          # heat transfer area in m²
       process_type="Fluids",
       category="Heat exchangers (Shell and tube)",
       type="Fixed head",
       material="Stainless steel 316",
       target_year=2024,
   )

   print(heat_exchanger)
   print(f"Purchased cost : ${heat_exchanger.purchased_cost:,.0f}")
   print(f"Direct cost    : ${heat_exchanger.direct_cost:,.0f}")

Key constructor arguments
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Type
     - Description
   * - ``name``
     - str
     - Identifier for the equipment item
   * - ``param``
     - float
     - Size/capacity parameter (units depend on equipment type)
   * - ``process_type``
     - str
     - ``"Solids"``, ``"Fluids"``, or ``"Mixed"`` — controls indirect cost factors
   * - ``category``
     - str
     - Equipment category (must match the cost database)
   * - ``type``
     - str
     - Equipment sub-type within the category
   * - ``material``
     - str
     - Construction material (affects the material factor)
   * - ``target_year``
     - int
     - Year to inflate costs to (default: 2024)
   * - ``cost``
     - float or None
     - Override: supply a direct purchased cost instead of using a correlation

Listing available equipment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openpytea.equipment import CostCorrelationDB, COST_DB_DF

   db = CostCorrelationDB(COST_DB_DF)

   # Show all categories
   print(db.df["category"].unique())

   # Show types within a category
   mask = db.df["category"] == "Compressors (Centrifugal)"
   print(db.df.loc[mask, "type"].unique())

Materials
~~~~~~~~~

The following materials are available. Each carries a cost multiplier
relative to carbon steel (= 1.0):

.. list-table::
   :header-rows: 1
   :widths: 50 20

   * - Material
     - Factor
   * - Carbon steel
     - 1.00
   * - Stainless steel 304
     - 1.30
   * - Stainless steel 316
     - 1.45
   * - Carpenter 20 CB-3
     - 1.55
   * - Nickel 200
     - 1.65
   * - Monel 400
     - 1.65
   * - Inconel 600
     - 1.70
   * - Incoloy 825
     - 1.70
   * - Titanium
     - 1.70

Equipment parallelization
--------------------------

When the size parameter exceeds the maximum capacity of a single unit, the
:class:`~openpytea.equipment.CostCorrelationDB` automatically determines the
minimum number of parallel units and the adjusted capacity per unit:

.. code-block:: python

   # A compressor needing 50 000 kW — may require multiple parallel units
   large_comp = Equipment(
       name="COMP-LARGE",
       param=50_000,
       process_type="Fluids",
       category="Compressors (Centrifugal)",
       type="Centrifugal",
       material="Carbon steel",
   )
   print(f"Number of units: {large_comp.num_units}")

Inflation adjustment utility
-----------------------------

You can use the standalone function for quick inflation calculations:

.. code-block:: python

   from openpytea import inflation_adjustment

   # Adjust a $500 000 cost from 2015 to 2024
   adjusted = inflation_adjustment(500_000, cost_year=2015, target_year=2024)
   print(f"Adjusted cost: ${adjusted:,.0f}")

Process factors
---------------

The direct (installed) cost is computed from the purchased cost using
additive process factors that depend on the ``process_type``:

.. math::

   C_{\text{direct}} = C_p \cdot f_m \cdot
   (1 + f_{\text{er}} + f_p + f_i + f_{\text{el}} + f_c + f_s + f_l)

where :math:`f_m` is the material factor and the :math:`f_*` terms are
erection, piping, instrumentation, electrical, civil, structural, and
lagging factors respectively. Default values differ between
``"Solids"``, ``"Fluids"``, and ``"Mixed"`` process types.

See also
--------

* :class:`~openpytea.equipment.Equipment` — full API reference
* :class:`~openpytea.equipment.CostCorrelationDB` — database interface
* :func:`~openpytea.equipment.inflation_adjustment` — CEPCI adjustment
