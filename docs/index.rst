.. role:: red-bold

Open-Source Techno-Economic Assessment Python-Toolkit
=========

.. image:: https://img.shields.io/pypi/v/OpenPyTEA.svg
   :target: https://pypi.org/project/OpenPyTEA/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://github.com/PBTamarona/OpenPyTEA/blob/main/LICENSE
   :alt: MIT License

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :alt: Python 3.10+

|

**OpenPyTEA:**  An open-source python toolkit for techno-economic assessment
of chemical process plants and energy systems with economic sensitivity
and uncertainty evaluation. It bridges process simulation tools —
which provide mass and energy balances — with rigorous economic evaluation,
covering capital expenditure (CAPEX), operating expenses (OPEX), and key
financial metrics.
Source code, issue tracker, and contributions:
`github.com/pbtamarona/OpenPyTEA <https://github.com/pbtamarona/OpenPyTEA>`_.

----

.. admonition:: 🎓 Workshop Recap — Open-Science TEA in Action
   :class: important

   **Open-Science Techno-Economic Assessment with OpenPyTEA: From Process Design to Economic Insights**

   On November 6, 2026, we hosted the first OpenPyTEA workshop at TU Delft. More than 70 participants
   joined from many countries and from across industry, academia, and research institutes.

   The day ran from process design and TEA fundamentals, through a hands-on OpenPyTEA session and an
   industry perspective, to a look at multi-agent systems for process design.

   Thank you to everyone who joined us in Delft, to our invited speakers Arnoud Higler and Artur
   Schweidtmann, to fellow organizers Thijs Vlugt and Mahinder Ramdin, and to the TU Delft Process &
   Product Technology Institute (Pro2Tech) and the TU Delft Institute for Computational Science and
   Engineering for making it possible.

   The workshop materials stay online at
   `github.com/pbtamarona/openpytea_workshop <https://github.com/pbtamarona/openpytea_workshop>`_.
   Want to try OpenPyTEA and get involved? Everything is open source on
   `GitHub <https://github.com/pbtamarona/OpenPyTEA>`_.

   Keep an eye out for the next one! 👀

   .. image:: _static/workshop-flyer.jpg
      :alt: OpenPyTEA Workshop Flyer
      :width: 600px
      :align: center

----

.. grid:: 2
   :gutter: 3
   :margin: 0

   .. grid-item-card:: :octicon:`download` Installation
      :link: installation
      :link-type: doc

      Set up OpenPyTEA using pip, uv, or from source.

   .. grid-item-card:: :octicon:`rocket` Quick Start
      :link: quickstart
      :link-type: doc

      Run your first techno-economic assessment in minutes.

   .. grid-item-card:: :octicon:`book` User Guide
      :link: user_guide/index
      :link-type: doc

      In-depth guides for equipment costing, plant TEA, analysis, and more.

   .. grid-item-card:: :octicon:`code` API Reference
      :link: api/index
      :link-type: doc

      Complete autodoc reference for all classes and functions.

   .. grid-item-card:: :octicon:`beaker` Examples
      :link: examples
      :link-type: doc

      Worked Jupyter notebooks covering hydrogen, geothermal, and other
      case studies with full TEA workflows.

   .. grid-item-card:: :octicon:`book` Tutorials
      :link: tutorials
      :link-type: doc

      Walkthrough notebooks covering the full OpenPyTEA workflow
      step by step, in five parts.

   .. grid-item-card:: :octicon:`browser` Graphical User Interface
      :link: gui
      :link-type: doc

      Run TEA interactively without writing code, in the desktop app for
      Windows and macOS.

Key features
------------

* **Modular architecture** — equipment costing, plant economics, and analysis are cleanly separated
* **Transparent methodology** — every formula is open and documented
* **Extensive financial metrics** — NPV, IRR, ROI, levelized cost, payback time, and full cash-flow tables
* **Multiple depreciation methods** — straight-line, declining-balance, MACRS
* **Sensitivity & uncertainty** — one-way sensitivity, tornado diagrams, Monte Carlo simulation
* **Graphical user interface** — installable desktop app (or run from source) for no-code analysis
* **Reproducible workflows** — JSON-based configuration files and result export
* **Research-oriented design** — easy integration with other frameworks (optimization, LCA, etc.)

Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   user_guide/equipment
   user_guide/cost_correlations
   user_guide/plant
   user_guide/analysis
   user_guide/io_workflow
   gui_development

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/equipment
   api/plant
   api/analysis
   api/plotting
   api/io

.. toctree::
   :maxdepth: 1
   :caption: More

   examples
   tutorials
   gui
   contributing
   citation
