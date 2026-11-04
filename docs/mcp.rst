AI Assistants (MCP Server)
==========================

OpenPyTEA comes with a `Model Context Protocol <https://modelcontextprotocol.io>`_
(MCP) server, so an AI assistant such as Claude Desktop or Claude Code can
run techno-economic assessments with the library. You describe the plant
in plain words ("a 50 t/day PEM electrolyser at 5 ct/kWh, what's the LCOH
and what drives it?"); the assistant builds the equipment and plant
inputs, runs the calculation and analyses through the server, and reports
the numbers.

Installing
----------

The server is an optional extra (it needs Python 3.10 or later):

.. code-block:: bash

   pip install "openpytea[mcp]"

This installs the ``openpytea-mcp`` command, which serves MCP over stdio.

**Claude Code:**

.. code-block:: bash

   claude mcp add openpytea -- openpytea-mcp

**Claude Desktop** (or any other MCP client): add the server to the
client's configuration file (``claude_desktop_config.json`` for Claude
Desktop), using the full path to the command if it isn't on the client's
``PATH``:

.. code-block:: json

   {
     "mcpServers": {
       "openpytea": {"command": "/path/to/openpytea-mcp"}
     }
   }

Tools
-----

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Tool
     - What it does
   * - ``get_input_format``
     - The equipment and plant input format, with examples.
   * - ``list_examples`` / ``get_example``
     - The bundled example plants (the GUI's examples) and their complete
       inputs.
   * - ``search_cost_correlations``
     - Search the equipment cost correlations: category, type, size units
       and range, cost year, source; and the construction materials.
   * - ``cost_equipment``
     - Purchased and installed cost of a list of equipment.
   * - ``evaluate_plant`` / ``load_example``
     - Full TEA of a plant: capital costs, OPEX, revenue, cash flow, LCOP,
       NPV, IRR, ROI, payback.
   * - ``list_plants`` / ``get_plant_config``
     - The plants evaluated in the session, and a plant's inputs.
   * - ``list_sensitivity_parameters``
     - The parameters a sensitivity or tornado analysis can vary.
   * - ``run_sensitivity``
     - A metric against one parameter varied by ±x %.
   * - ``run_tornado``
     - Parameters ranked by their effect on a metric (economic, process,
       or both).
   * - ``run_monte_carlo``
     - Uncertainty analysis: mean, standard deviation and percentiles of
       LCOP, NPV, ROI and payback, and of the sampled inputs.

Plot tools
~~~~~~~~~~

These draw the library's own figures (the ones ``run_tea``, the
``plot_*`` functions and the GUI's downloads produce) and return them as
images the assistant can show you. Every plot tool also takes
``save_to``, a file path ending in ``.png``, ``.pdf``, ``.svg``, ``.jpg``
or ``.eps``, to save the figure (raster formats at 600 dpi), e.g. for a
report.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Tool
     - What it draws
   * - ``plot_costs``
     - Stacked-bar breakdown, one bar per plant: direct (equipment) costs,
       fixed capital, fixed OPEX, variable OPEX or the levelized cost
       build-up; amounts or shares in %.
   * - ``plot_cash_flow``
     - Cumulative cash flow over the project lifetime, one line per plant.
   * - ``plot_sensitivity``
     - A metric against one parameter, one curve per plant; x as % change
       or as the parameter's own value.
   * - ``plot_tornado``
     - The tornado chart of a plant.
   * - ``plot_monte_carlo``
     - Histogram of LCOP, NPV, ROI or payback, several plants overlaid.
       Reuses each plant's last ``run_monte_carlo`` result unless
       ``num_samples`` is given.
   * - ``plot_monte_carlo_inputs``
     - Histograms of the sampled inputs, one figure for the process and
       one for the economic parameters.

Inputs use the same layout as the :doc:`JSON workflow <user_guide/io_workflow>`
files, so a plant built by the assistant can be saved (``get_plant_config``)
and run with ``run_tea``, the ``openpytea`` command or the GUI's import
buttons, and the other way round.

Evaluated plants are kept in memory, by plant name, for as long as the
server runs (normally the length of the assistant's session). Evaluating a
plant with a name already in use replaces it.
