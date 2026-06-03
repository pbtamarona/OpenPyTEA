OpenPyTEA
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

**OpenPyTEA** (Open-source Python Toolkit for Economic Assessment) is a
comprehensive Python library for techno-economic assessment (TEA) of chemical
and energy systems. It bridges process simulation tools — which provide mass
and energy balances — with rigorous economic evaluation, covering capital
expenditure (CAPEX), operating expenses (OPEX), and key financial metrics.

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

      Hydrogen, geothermal, and other case-study notebooks.

   .. grid-item-card:: :octicon:`browser` Web GUI
      :link: gui
      :link-type: doc

      Run TEA interactively without writing code.

Key features
------------

* **Modular architecture** — equipment costing, plant economics, and analysis are cleanly separated
* **Transparent methodology** — every formula is open and documented
* **Rich financial metrics** — NPV, IRR, ROI, levelized cost, payback time, and full cash-flow tables
* **Multiple depreciation methods** — straight-line, declining-balance, MACRS
* **Sensitivity & uncertainty** — one-way sensitivity, tornado diagrams, Monte Carlo simulation
* **Geographic flexibility** — location cost factors for 17 countries and regions
* **Reproducible workflows** — JSON-based configuration files and result export
* **Interactive web GUI** — React + FastAPI front-end for no-code analysis
* **Education-friendly** — designed for university TEA courses

Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   user_guide/equipment
   user_guide/plant
   user_guide/analysis
   user_guide/plotting
   user_guide/io_workflow

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/equipment
   api/plant
   api/analysis
   api/plotting
   api/io

.. toctree::
   :maxdepth: 1
   :caption: More

   examples
   gui
   contributing
