Examples & Case Studies
========================

OpenPyTEA ships with three fully worked case studies in the
`examples/ <https://github.com/PBTamarona/OpenPyTEA/tree/main/examples>`_
directory of the repository. Each is a self-contained Jupyter notebook that
demonstrates the library's capabilities on a real-world engineering scenario.

Running the examples
--------------------

.. code-block:: bash

   git clone https://github.com/PBTamarona/OpenPyTEA.git
   cd OpenPyTEA
   pip install "OpenPyTEA[ipython]"
   jupyter notebook examples/

Case Study 1 — Hydrogen Production Pathways
--------------------------------------------

**File**: `examples/case_study_1.ipynb <https://github.com/PBTamarona/OpenPyTEA/blob/main/examples/case_study_1.ipynb>`_

Compares the techno-economics of three hydrogen production routes:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Scenario
     - Technology
   * - SMR
     - Steam Methane Reforming (incumbent)
   * - Pyrolysis
     - Methane Pyrolysis (solid carbon by-product)
   * - Electrolysis
     - Water Electrolysis (green hydrogen)

The notebook covers equipment selection, CAPEX 
and OPEX breakdowns, sensitivity to natural gas and electricity prices, and a
side-by-side Monte Carlo comparison of the levelized cost of hydrogen (LCOH)
across all three pathways.

Key topics demonstrated
~~~~~~~~~~~~~~~~~~~~~~~

* Creating multiple :class:`~openpytea.equipment.Equipment` objects per scenario
* Using both cost correlations and direct supply quotes for equipment purchased costs
* Modelling byproduct revenues and mid-project ``additional_capex`` for equipment replacement
* Running the same analysis on three :class:`~openpytea.plant.Plant` instances
* Using :func:`~openpytea.plotting.plot_multiple_monte_carlo` for cross-scenario comparison
* Interpreting tornado diagrams to identify cost drivers

Case Study 2 — Hydrogen Liquefaction Precooling
-------------------------------------------------

**File**: `examples/case_study_2.ipynb <https://github.com/PBTamarona/OpenPyTEA/blob/main/examples/case_study_2.ipynb>`_

Techno-economic assessment of a **precooling process of hydrogen liquefaction**. This case study builds
on the a multi-objective optimization study to minimize specific energy consumption and levelized cost 
in mixed-refrigerant systems.

Key topics demonstrated
~~~~~~~~~~~~~~~~~~~~~~~

* Integrating OpenPyTEA into an optimization workflow to evaluate techno-economic 
trade-offs across candidate plant configurations (a stepping stone toward full process-optimization coupling)
* Using breakdown charts to compare CAPEX and OPEX structure between plant configurations
* Evaluating the impact of process design choices on both specific energy consumption and levelized cost
* Copying and modifying an existing :class:`~openpytea.plant.Plant` instance to create a new scenario with different equipment and cost structure

Case Study 3 — Geothermal Energy Systems
-----------------------------------------

**File**: `examples/case_study_3.ipynb <https://github.com/PBTamarona/OpenPyTEA/blob/main/examples/case_study_3.ipynb>`_

Compares two geothermal applications:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Scenario
     - System
   * - District heating
     - Heat pump for residential heating
   * - Power generation
     - Organic Rankine Cycle (ORC)

Key features: 30-year project lifetime, full MACRS depreciation, and
Monte Carlo uncertainty analysis for both scenarios.

Key topics demonstrated
~~~~~~~~~~~~~~~~~~~~~~~

* Modelling pre-production CAPEX (geothermal site exploration and drilling) using the additional cost configuration
* Configuring MACRS depreciation for long-lifetime assets
* Comparing levelized cost of heat (LCOH) against levelized cost of electricity (LCOE) across scenarios

For a step-by-step walkthrough of every main feature, see the :doc:`tutorials` page.
