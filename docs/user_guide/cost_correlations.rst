:hide-toc:

Cost Correlations Database
==========================

``cost_correlations.csv`` bundles 417 correlations spanning 34 equipment
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
constructor's ``material`` argument, see :ref:`materials`), its ``source``
(clickable, linking to the DOI or reference where available), any
``Remarks`` (basis of the quoted cost, e.g. free-on-board vs. installed,
included/excluded motor, escalation notes), and the correlation ``key``.
Use the ``key`` value as the ``cost_func`` argument when you need to pin
a specific correlation (see :ref:`example-select-key`), and the
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
   :widths: auto
   :class: sphinx-datatable

