Equipment Cost Estimation
=========================

The :mod:`openpytea.equipment` module estimates purchased and installed costs
for individual process units using published cost correlations, automatic CEPCI
inflation adjustment, and process/material installation factors.

How costs are estimated
-----------------------

Purchased cost correlations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Six correlation forms are supported:

**Offset power-law**

.. math::

   C_p = a + b \cdot S^n

**Log-log quadratic**

.. math::

   \log_{10} C_p = K_1 + K_2 \log_{10} S + K_3 \left(\log_{10} S\right)^2
   + K_4 \left(\log_{10} S\right)^3 + K_5 \left(\log_{10} S\right)^4

**Ln-ln quadratic**

.. math::

   \ln C_p = K_1 + K_2 \ln S + K_3 \left(\ln S\right)^2
   + K_4 \left(\ln S\right)^3 + K_5 \left(\ln S\right)^4

Same form as log-log quadratic but with natural rather than base-10
logarithms. :math:`K_4` and :math:`K_5` are optional (default 0) and let
either log-based form fit a cubic or quartic trend when the underlying
data calls for it — the built-in correlations of both forms currently
only use :math:`K_1`–:math:`K_3`.

**Power-sizing**

.. math::

   C_p = C_0 \left( \frac{S}{S_0} \right)^f

**Exponential**

.. math::

   C_p = a \cdot \exp(b \cdot S)

**2-var power-law**

.. math::

   C_p = a + b \cdot S_1^{n} \cdot S_2^{n_2}

For equipment priced off two independent size parameters (e.g., a belt
conveyor's width and length). :math:`a` is typically 0 unless the
correlation has a genuine offset term. Evaluating this form requires
passing both sizes, e.g. ``Equipment(..., param=(S1, S2))`` — see
Example 10 below.

where :math:`S` (or :math:`S_1`, :math:`S_2`) is the equipment size
parameter (e.g., shaft power in kW, heat transfer area in m²) and
:math:`C_p` is the purchased cost in the correlation's reference year
(USD). For the power-sizing form, :math:`S_0` and :math:`C_0` are a
reference size and its corresponding cost.

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
item via constructor keyword arguments** — see Example 9 for details.

*Source: Towler & Sinnott (2022)*

Equipment Cost Correlations
----------------------------

``cost_correlations.csv`` bundles 419 correlations spanning 34 equipment
categories (agitators, compressors, heat exchangers, pressure vessels,
reactors, conveyors, and more), pulled from several published sources:

* Turton et al., *Analysis, Synthesis, and Design of Chemical Processes*
  (2018) — log-log quadratic correlations.
* Towler & Sinnott, *Chemical Engineering Design* (2010) — offset
  power-law correlations.
* Perry's Chemical Engineers' Handbook, Table 9-50 (1997) — power-sizing
  correlations, cost-escalated to 1996 via the Marshall & Swift index.
* Seider et al., *Product and Process Design Principles*, 4th ed.
  (2013) — ln-ln quadratic, offset/2-var power-law, and exponential
  correlations for solids-handling, size-enlargement, and separation
  equipment.
* Ulrich (2003), ESDU 97006 (1997), and several process-specific studies
  (Manzolini, Kreutz, Parkinson, Towler, Nexant, NREL) covering
  compressors, furnaces, gas separation, and CO\ :sub:`2` capture
  equipment.

Each row records the correlation's ``category``, ``type``, size
``units``, valid size range (``s_lower``–``s_upper``, plus
``s2_lower``–``s2_upper`` for two-parameter forms), correlation
``form``, reference ``cost_year``, its ``default material`` (the
construction material the quoted cost basis assumes, e.g. ``Carbon
steel`` or ``Stainless steel`` — override via the ``Equipment``
constructor's ``material`` argument), its ``source``
(clickable, linking to the DOI or reference where available), any
``Remarks`` (basis of the quoted cost, e.g. free-on-board vs. installed,
included/excluded motor, escalation notes), and the correlation ``key``.
Use the ``key`` value as the ``cost_func`` argument when you need to pin
a specific correlation (see Example 2 below), and the
``category``/``type`` values for the ``Equipment`` constructor.

**The correlation coefficients themselves are not shown in this summary
table** — download the full
:download:`cost_correlations.csv <../../src/openpytea/data/cost_correlations.csv>`
for those.

The table below is generated directly from ``cost_correlations.csv`` and
is searchable, sortable, and paginated — use the search box to filter by
category, type, form, or source, or click a column header to sort.
``Category`` and ``Type`` stay pinned while you scroll horizontally
through the remaining columns.

.. csv-table:: Built-in equipment cost correlations
   :file: ../_static/cost_correlations_table.csv
   :header-rows: 1
   :widths: 13, 15, 9, 6, 6, 9, 5, 9, 14, 17, 13
   :class: sphinx-datatable

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
     - float, or tuple/list of two floats
     - Size/capacity parameter. Units depend on the equipment type —
       check ``cost_correlations.csv``. Pass a 2-element ``(S1, S2)``
       tuple/list for two-parameter forms such as ``"2-var power-law"``
       (see Example 10). Ignored when ``purchased_cost`` is given.
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
     - str or None
     - Construction material. Default: ``None``, which uses the
       resolved correlation's ``default material`` (falling back to
       ``"Carbon steel"`` if there's no match). A material that matches
       this resolved default — whether auto-filled or passed explicitly
       — always gets :math:`f_m = 1.0`, since the correlation's cost
       already prices that material in. Passing a *different* material
       instead uses the *ratio* of the two materials' factors from the
       table below (target :math:`f_m` divided by default :math:`f_m`,
       using 1.0 for a default material not in the table, e.g. "Cast
       iron"), so the factor stays relative to the correlation's actual
       cost basis instead of double-counting it. Raises if the target
       material isn't found in the table.
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
     - Override the material factor. ``None`` uses 1.0 when
       ``material`` matches the resolved default, or the ratio of the
       two materials' table factors otherwise.

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

Example 7 — Automatic default material resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Leaving ``material`` unset resolves it from the matched correlation's own
``default material`` column, with :math:`f_m = 1.0` — the correlation's
quoted cost already prices in that material, so no extra multiplier is
applied. This batch centrifuge correlation defaults to "Stainless steel":

.. code-block:: python

   centrifuge = Equipment(
       name="Centrifuge C-101",
       param=30,                        # bowl diameter, in
       process_type="Fluids",
       category="Centrifuges",
       type="Batch, bottom-drive, vertical basket",
   )
   print(f"material        : {centrifuge.material}")         # Stainless steel
   print(f"material_factor : {centrifuge.material_factor}")  # 1.0

Passing a *different* material instead rescales the table's factor
relative to the correlation's own default rather than the usual carbon
steel baseline — requesting Inconel construction here applies the ratio
of Inconel's factor to Stainless steel's factor (1.70 / 1.30 ≈ 1.31),
not the raw 1.70, since the correlation's cost basis is already
stainless steel rather than carbon steel:

.. code-block:: python

   centrifuge_inconel = Equipment(
       name="Centrifuge C-101 (Inconel)",
       param=30,
       process_type="Fluids",
       category="Centrifuges",
       type="Batch, bottom-drive, vertical basket",
       material="Inconel",
   )
   print(f"material_factor : {centrifuge_inconel.material_factor:.3f}")  # 1.308

If the resolved default isn't recognized in ``material_factors`` (e.g.
"Cast iron", "Ceramic"), it's still reported as ``material`` with
:math:`f_m = 1.0`, and an explicit different material falls back to its
raw table factor (an unrecognized default has no factor to divide out).

Example 8 — Comparing materials
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

Example 9 — Overriding installation factors
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

Example 10 — Two-parameter correlations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Correlations with ``form = "2-var power-law"`` are priced off two
independent size parameters. Pass both as a ``(S1, S2)`` tuple/list in
``param`` — here, belt width (in) and belt length (ft):

.. code-block:: python

   belt = Equipment(
       name="Belt conveyor BC-101",
       param=(24, 150),                 # (width, in;  length, ft)
       process_type="Solids",
       category="Conveyors",
       cost_func="belt_conveyor_seider_2013",
   )
   print(belt)

``s2_lower``/``s2_upper`` in the CSV bound the second parameter the same
way ``s_lower``/``s_upper`` bound the first, except exceeding
``s2_upper`` always raises ``ValueError`` — it never triggers
parallelization the way ``s_upper`` does.

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
   * - ``"Stainless steel"``
     - 1.30
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

``"Stainless steel"`` (no grade specified) is not part of the original
Towler & Sinnott table — it's added here, set equal to
``"304 stainless steel"`` since that's the most common general-purpose
grade. This is also the value used in ``cost_correlations.csv`` for
rows whose ``default material`` is unspecified-grade stainless steel.

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
