<p align="center">
  <img src="docs/logo-white.png" alt="OpenPyTEA" width="400"/>
</p>

**OpenPyTEA** is an open-source Python toolkit for performing **techno-economic assessment (TEA)** of chemical and energy systems. It was created to address a persistent gap in the TEA workflow: while process simulators model mass and energy balances, researchers often lack an equally transparent and flexible way to evaluate the **economic feasibility** of their designs. Commercial tools remain *black-box tools*, and many academic TEA implementations are process-specific, undocumented, or difficult to reproduce.

**OpenPyTEA** provides a fully open, modular, and traceable framework that brings TEA into the Python ecosystem. By integrating **equipment cost estimation**, **capital and operating expenditure modeling**, **cash-flow analysis**, **cost breakdowns**, **sensitivity evaluation**, and **Monte Carlo uncertainty propagation**, the toolkit enables users to perform end-to-end TEA with clarity and reproducibility.

Beyond its functionality, **OpenPyTEA is designed as a community-driven TEA platform**. Users can contribute new equipment cost correlations, improve economic models, report issues, and expand the toolkit’s capabilities over time. This collaborative approach helps build a shared, transparent, and continually improving TEA resource—similar to the open-source progress seen in the LCA community.

Whether used for early-stage process design, technology screening, or teaching, **OpenPyTEA** makes TEA more accessible, consistent, and aligned with FAIR research principles (Findable, Accessible, Interoperable, and Reusable).

**For a full walkthrough of the features and usage of OpenPyTEA, refer to the `walkthrough.ipynb` notebook**:  
https://github.com/pbtamarona/OpenPyTEA/blob/main/walkthrough.ipynb

**For the full documentation of the package, visit the ReadTheDocs page:**  
https://openpytea.readthedocs.io

**For some case-study examples, please check the `examples` folder:**
https://github.com/pbtamarona/OpenPyTEA/tree/main/examples

---
## 🎓 Workshop

We are hosting a one-day workshop on open-science techno-economic assessment using **OpenPyTEA**!

<p align="center">
  <img src="workshop-flyer.jpg" alt="OpenPyTEA" width="800"/>
</p>

**Open-Science Techno-Economic Assessment with OpenPyTEA: From Process Design to Economic Insights**

The workshop covers the full workflow from process design and simulation to economic evaluation, combining lectures, a hands-on session with OpenPyTEA, and an industry talk by Shell. It closes with a community discussion on establishing shared TEA standards, bringing together students, researchers, engineers, and policymakers across chemical, energy, and sustainability sciences.

- 📅 **Date:** November 6, 2026 | 09:30 – 18:00
- 📍 **Location:** Process & Energy, TU Delft, Delft, The Netherlands
- 📝 **Registration:** [aanmelder.nl/openpytea2026](https://aanmelder.nl/openpytea2026)

Lunch, snacks, coffee, and drinks will be provided!

---
## ✨ Key Features
- **Modular architecture:** clean separation of cost correlations, equipment objects, plant economics, and uncertainty analysis.  
- **Transparent and reproducible:** all algorithms, equations, and assumptions are openly available for full traceability.
- **Cost breakdown visualization:** built-in functions to plot stacked bar charts of equipment costs, fixed capital, operating costs, and levelized cost of production (LCOP).
- **Cash flow diagrams:** visualize a project's cumulative cash flow over time, including its maximum investment and pay-back point, with support for overlaying multiple plants.
- **Built-in uncertainty tools:** automatic generation of sensitivity plots and Monte Carlo simulations, covering process quantities (consumption and production rates) as well as prices and financial assumptions.
- **Parameter dependencies:** declare how quantities depend on one another — cooling water scaling with production, a byproduct's yield tracking the main product, capital cost scaling with capacity — and every analysis honours the same graph.
- **Workflow using JSON configuration files:** standardized input/output structure via `io.py` for reproducible analyses and multi-scenario evaluation.
- **Flexible analysis and visualization:** separation of data processing (`analysis.py`) and plotting (`plotting.py`) allows users to apply custom visualization tools.
- **Interoperable and extensible:** easy integration with process simulators, optimization frameworks, and LCA tools.  
- **Education-friendly:** ideal for teaching TEA and process design without reliance on proprietary software.  
- **Community-driven:** users can contribute new correlations, improve models, request features, and shape the evolution of the platform.  

---
## 📦 Installation

### 1. **Install from PyPI (recommended)**

```bash
pip install openpytea
```

### 2. **Install from GitHub (development version)**

```bash
pip install git+https://github.com/pbtamarona/OpenPyTEA
```

or with `uv`:

```bash
uv add git+https://github.com/pbtamarona/OpenPyTEA
```

**OpenPyTEA** requires **Python ≥ 3.10**.  
The main dependencies include:

- `matplotlib`
- `numpy`  
- `pandas`
- `scienceplots`  
- `scipy`  
- `seaborn`  
- `tqdm`  
- `jinja2` 

---
## ⚙️ Package (Repository) Structure
```
src/openpytea/
├── equipment.py            # Equipment-level costing and inflation correction
├── plant.py                # Plant-level TEA: CAPEX, OPEX, cash flows, financial metrics
├── analysis.py             # Sensitivity and uncertainty analysis (sensitivity plots, Monte Carlo)
├── plotting.py             # Visualization functions (plots and figures)
├── io.py                   # JSON-based workflow: load inputs and export results
├── helpers.py              # Helper functions for data handling and common operations
└── data/                   # Cost correlations database and CEPCI data
examples/                   # Example notebooks and case studies
walkthrough.ipynb           # Walkthrough of the package

backend/                    # FastAPI backend for the web GUI (standalone-package branch only)
├── app/
│   ├── main.py             # FastAPI app with CORS and router mounting
│   ├── state.py            # In-memory session state
│   ├── schemas.py          # Pydantic request/response models
│   ├── util.py             # JSON serialization utilities
│   ├── routers/            # API endpoints (equipment, plant, analysis, I/O)
│   └── presets/            # Example preset JSON files
└── requirements.txt

frontend/                   # React + TypeScript web GUI (standalone-package branch only)
├── src/
│   ├── api/client.ts       # Typed API client
│   ├── types/index.ts      # TypeScript interfaces
│   ├── pages/              # Equipment, Plant Config, Results, Analysis, Monte Carlo, Compare
│   ├── App.tsx             # Tab navigation + examples dropdown
│   └── App.css             # Styling
└── package.json

pyproject.toml
README.md
```

---
## 🏗️ Software Architecture

![OpenPyTEA Architecture](examples/img/architecture.png)

Software architecture and data flow of **OpenPyTEA**, illustrating the progression from user input to TEA output. Users provide economic assumptions, process simulation results, and equipment-sizing parameters. Equipment-sizing information is linked with cost correlations and CEPCI values stored in CSV databases to calculate inflation-adjusted purchased and direct costs. `Equipment` objects are aggregated into a `Plant` object, where CAPEX, OPEX, and financial performance metrics are evaluated. The `analysis.py` module subsequently operates on `Plant` objects to perform sensitivity and uncertainty analyses.

<!-- ## 🖥️ Web GUI (**work in progress**)

OpenPyTEA includes an optional web-based graphical interface for users who prefer a visual workflow over Python scripting. The GUI provides the full TEA workflow through a tabbed browser interface:

- **Equipment** — add, edit, and remove equipment with cost database lookup
- **Plant Config** — configure location, financial parameters, labor, products, and variable OPEX
- **Results** — run calculations and view metric cards, cost breakdown charts, and cash flow tables
- **Analysis** — sensitivity plots and tornado diagrams. Sensitivity supports a **multi-panel grid** (different parameter and metric per panel, e.g. NPV vs. interest rate, ROI vs. electricity price, all in one figure) and **multi-plant overlay** (curves for every plant added on the Compare tab share the same axes for direct comparison)
- **Monte Carlo** — uncertainty analysis with histogram distributions, fitted normal curves, and summary statistics. **Multi-plant overlay** shows distributions for several plants on the same axes, mirroring `plot_multiple_monte_carlo` from the library
- **Compare** — side-by-side comparison of saved plants (CAPEX/OPEX breakdown bars, key metric bars). Plants imported here are also reused as the overlay set on the Analysis and Monte Carlo tabs
- **Downloadable charts** — all plots include a download button to export as standalone PNG images with full axis labels
- **Examples** — built-in presets from the case study notebooks for quick demonstration

### Running the GUI

**Quick start** (requires Python 3.10+ and Node.js — macOS/Linux):

```bash
./start.sh
```

That's it. On first run the script creates a local `.venv`, installs the backend dependencies, runs `npm install`, then launches both servers and opens your browser at http://localhost:5173. Subsequent runs just start the servers. Press `Ctrl+C` to stop both.

<details>
<summary><strong>Manual steps</strong> (if you'd rather run backend and frontend yourself)</summary>

**Backend** (Python 3.10+):
```bash
pip install -e .          # install OpenPyTEA from repo root
cd backend
pip install -r requirements.txt
PYTHONPATH=../src python3 -m uvicorn app.main:app --reload --port 8000
```

**Frontend** (Node.js):
```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173.

</details>

Click **Examples** in the header to load a case study preset and explore.

For detailed architecture documentation, see `GUI_ARCHITECTURE.md`. -->

<!-- --- -->

---
## 🧠 Core Concepts

### 1. **Equipment-level costing**

Each process unit (e.g., compressor, heat exchanger, reactor) is represented by an `Equipment` object:

```python
from openpytea.equipment import Equipment

compressor = Equipment(
    name='COMP',
    param=5000,  # kW
    process_type='Fluids',
    category='Compressors, fans, & Blowers',
    type='Compressor, centrifugal',
    material='Carbon steel'
)

print(compressor.direct_cost)
```

Each equipment item retrieves its cost correlation from the internal database in `data/cost_correlations.csv` and adjusts the cost to the desired year using the Chemical Engineering Plant Cost Index (CEPCI).

### 2. **Plant-level techno-economic assessment**

Multiple equipment objects can be grouped into a `Plant` instance for full TEA

```python
from openpytea.plant import Plant

ammonia_plant = Plant({
    'plant_name': 'Ammonia Production Plant',
    'country': 'Netherlands',
    'process_type': 'Fluids',
    'equipment': [compressor],
    'interest_rate': 0.09,
    'plant_utilization': 0.95,
    'project_lifetime': 20,  # in years
    'plant_products': {  # Here we define the product(s) of the plant
        'ammonia': {
            'production': 125_000,  # Daily production in kg/day
        }
    },
    'variable_opex_inputs': {
        'electricity': {
            'consumption': 110,  # Daily consumption, in MWh
            'price': 75  # US$/MWh
        },
        'hydrogen': {
            'consumption': 22_000,  # Daily consumption, in kg/day
            'price': 2  # US$/kg
        },
    },
})

ammonia_plant.calculate_cash_flow(print_results=True)
ammonia_plant.calculate_levelized_cost()
```
Main outputs include:
- Capital expenditures (CAPEX): inside/outside battery limits, engineering, contingency, and location factors
- Operating expenditures (OPEX): variable and operating expenditures, including utilities, maintenance, labor, and overhead costs
- Financial metrics: Net Present Value (NPV), Internal Rate of Return (IRR), Return on Investment (ROI), Payback Time (PBT), and Levelized Cost of Product (LCOP)

### 3. **CAPEX and OPEX breakdown plots**

Following a `data` + `plot` pattern used throughout the package, OpenPyTEA includes convenience functions for visualizing the economic structure of one or more plants as stacked bar charts:

- `direct_costs_data()` + `plot_stacked_bar()`: direct equipment costs (per equipment item).  
- `fixed_capital_data()` + `plot_stacked_bar()`: fixed capital components (ISBL, OSBL, design & engineering, contingency).  
- `variable_opex_data()` + `plot_stacked_bar()`: variable operating costs by input mass and energy stream.  
- `fixed_opex_data()` + `plot_stacked_bar()`: fixed operating expenses, including labor, supervision, maintenance, overhead, R&D, and more.  
- `levelized_cost_data()` + `plot_stacked_bar()`: levelized cost of production (LCOP), broken down into discounted CAPEX, OPEX, and side-product revenue.

```python
from openpytea.analysis import direct_costs_data, levelized_cost_data
from openpytea.plotting import plot_stacked_bar

direct_costs = direct_costs_data(ammonia_plant)
fig, ax = plot_stacked_bar(direct_costs)

lcop = levelized_cost_data(ammonia_plant)
fig, ax = plot_stacked_bar(lcop)
```

Each `*_data()` function also accepts a **list of plants**, in which case `plot_stacked_bar` draws one bar per plant side-by-side for direct comparison. Separating data preparation (`analysis.py`) from plotting (`plotting.py`) means you can also feed the returned dictionaries into your own custom visualization code.

### 4. **Cash flow diagram**

`cash_flow_data()` and `plot_cash_flow()` visualize a project's cumulative cash flow over time: the dip into debt during construction, the point of maximum investment, the break-even (pay-back) point, and the eventual climb into profit.

```python
from openpytea.analysis import cash_flow_data
from openpytea.plotting import plot_cash_flow

cash_flow = cash_flow_data(ammonia_plant)
fig, ax = plot_cash_flow(cash_flow)
```

As with the cost breakdowns, passing a **list of plants** overlays their cumulative cash flow curves — each with its own shaded debt region and break-even line — for direct comparison. The returned dictionary also carries the underlying figures (`max_investment`, `max_investment_year`, `breakeven_year`/`payback_time`) for use outside the plot, e.g. in reports.

### 5. **Sensitivity and uncertainty analysis**

**OpenPyTEA** provides integrated tools for visual sensitivity and probabilistic analysis of cost and performance drivers.

One-Way Sensitivity Line Plot
```python
from openpytea.analysis import sensitivity_data
from openpytea.plotting import plot_sensitivity

results = sensitivity_data(
    ammonia_plant,
    parameter="electricity",
    plus_minus_value=0.5,
)
fig, ax = plot_sensitivity(results)
```
The `plants` input may also be a list of `Plant` objects to generate comparison plots.

Besides prices and financial assumptions, `parameter` also accepts a **process quantity** — `"electricity.consumption"` or `"ammonia.production"` — to sweep the physical side of the plant.

Tornado Plot (One-at-a-Time Sensitivity)
```python
from openpytea.analysis import tornado_data
from openpytea.plotting import plot_tornado

results = tornado_data(ammonia_plant, plus_minus_value=0.5)
fig, ax = plot_tornado(results)
```
Pass `include_process_params=True` to rank consumption and production quantities alongside the prices and financial assumptions.

Monte Carlo Simulation
```python
from openpytea.analysis import monte_carlo
from openpytea.plotting import plot_monte_carlo

results = monte_carlo(ammonia_plant, num_samples=1_000_000)
fig, ax = plot_monte_carlo(results)
```
Outputs include probability distributions and confidence intervals for LCOP, NPV, ROI, and payback time—supporting uncertainty-informed decision-making. With `plot_multiple_monte_carlo`, **OpenPyTEA** can also visualize Monte Carlo results for multiple plants to enable uncertainty comparisons. `plot_monte_carlo_inputs` shows the sampled inputs themselves, split into **process** and **economic** figures, to confirm each distribution came out as intended.

Uncertainty is configured per item: `price_uncertainty` on any `variable_opex_inputs` or `plant_products` entry, `consumption_uncertainty`/`production_uncertainty` for the quantities, and `project_uncertainties` for the project-level scalars.

Parameter Dependencies
```python
ammonia_plant.update_configuration({
    "variable_opex_inputs": {
        "hydrogen": {
            "consumption_dependency": {
                "depends_on": {"production:ammonia": 0.176},  # kg H2 per kg NH3
            },
        },
    },
})
```
Rather than varying independently, a quantity can be defined as a linear function of one or more others — `dependent = Σ wᵢ · parentᵢ + offset`. Here the hydrogen feed follows ammonia production instead of drifting away from it. Process and economic parameters can drive each other in either direction (a byproduct's yield tracking the main product, `fixed_capital_factor` scaling with capacity), chains and multi-parent nodes resolve automatically, and cycles raise an error. A dependent may carry its own additive `noise` on top of the implied mean.

Because dependencies live on the `Plant`, **all three analyses honour them**: Monte Carlo samples through the graph, while `sensitivity_data` and `tornado_data` propagate each perturbation through it and refuse to vary a parameter that a dependency already determines. See the Analysis user guide in the [documentation](https://openpytea.readthedocs.io) for the full configuration format.

### 6. **Workflow using JSON config files and command-line interface**

**OpenPyTEA** supports a workflow using structured JSON input files via the `io.py` module. This enables standardized, reproducible, and scalable TEA studies.

Key functionalities include:
- `run_equipment()`: evaluate equipment costs from JSON input
- `run_plant()`: construct and evaluate a plant configuration
- `run_tea()`: execute full TEA, including cost breakdowns, sensitivity, and uncertainty analysis
- `run_openpytea()`: single-file counterpart to `run_tea()` — runs the same pipeline from one combined JSON file (`equipment` + `plant` + `analysis` blocks), intended for CLI use

This workflow is demonstrated in `case_study_1_with_JSON.ipynb` in the example folder.

Installing **OpenPyTEA** also installs an `openpytea` command-line tool, so the same combined-file workflow can be run without writing any Python:

```bash
openpytea run project/config.json --output-dir outputs/tea_results
```

`openpytea equipment`, `openpytea plant`, and `openpytea tea` (the three-file variant of `run_tea()`) are also available — run `openpytea --help` for the full command list. See the [JSON Workflow guide](docs/user_guide/io_workflow.rst) for details.

---
## ▶️ Tutorials

Step-by-step tutorial videos covering the full OpenPyTEA workflow are available here:

**Tutorial 01 - Creating Equipment**

<a href="https://www.youtube.com/watch?v=z-hspQh_wVE" target="_blank">
  <img src="https://img.youtube.com/vi/z-hspQh_wVE/0.jpg" width="320" alt="Tutorial 01 - Creating Equipment">
</a>

**Tutorial 02 - Creating a Plant**

<a href="https://www.youtube.com/watch?v=eoooa2gjCwE" target="_blank">
  <img src="https://img.youtube.com/vi/eoooa2gjCwE/0.jpg" width="320" alt="Tutorial 02 - Creating a Plant">
</a>

**Tutorial 03 - Performing Analysis**

<a href="https://www.youtube.com/watch?v=o1zosMUZaDc" target="_blank">
  <img src="https://img.youtube.com/vi/o1zosMUZaDc/0.jpg" width="320" alt="Tutorial 03 - Performing Analysis">
</a>

The notebooks used in the tutorials and the raw video files are available in the [tutorial_videos folder](https://github.com/pbtamarona/OpenPyTEA/tree/main/tutorial_videos)

---
## 📘 Example Workflows

Example notebooks are available in the `examples/` folder, including:

- Comparison of hydrogen production pathwways 
- Hydrogen liquefaction precooling system
- Geothermal-based heating and power generation

Run any example via:
```bash
jupyter notebook examples/case_study_1.ipynb
```
Each notebook demonstrates:
- Input definition and equipment configuration
- Cash-flow and investment evaluation
- Sensitivity and uncertainty analysis
- Visualization of key economic indicators

---
## 🧑‍🏫 Educational Use

**OpenPyTEA** is suitable for chemical and process engineering education.
Students can perform full TEA using their simulation outputs—estimating capital, operating, and profitability metrics—without commercial software.
All algorithms are visible and modifiable, eliminating the “black-box” nature of most TEA tools.

---
## 🛠️ Contributing
We welcome community contributions!
You can help by:
- Adding or updating equipment cost correlations
- Improving the documentation or creating tutorials
- Extending the visualization or uncertainty modules

To contribute:
1. Fork the repository.
2. Create a new branch:
```bash
git checkout -b feature-new-equipment
```
3. Commit your changes and open a Pull Request.

Please follow PEP8 coding conventions and include a short description of your updates.

---
## 📖 Publication

**OpenPyTEA** is described in the following peer-reviewed paper published in *SoftwareX*:

> Tamarona, P.B., Vlugt, T.J.H., & Ramdin, M. (2026). *OpenPyTEA: An open-source python toolkit
> for techno-economic assessment of chemical process plants and energy systems with economic
> sensitivity and uncertainty evaluation.* SoftwareX, 35, 102816.
> https://doi.org/10.1016/j.softx.2026.102816

If you use **OpenPyTEA** in your research, please cite this paper (see [Citation](#-citation) below).

---
## 📚 Citation

If you use **OpenPyTEA** in your research, please cite the following paper:

> Tamarona, P.B., Vlugt, T.J.H., & Ramdin, M. (2026). *OpenPyTEA: An open-source python toolkit
> for techno-economic assessment of chemical process plants and energy systems with economic
> sensitivity and uncertainty evaluation.* SoftwareX, 35, 102816.
> https://doi.org/10.1016/j.softx.2026.102816

**BibTeX:**
```bibtex
@article{TAMARONA2026102816,
title = {OpenPyTEA: An open-source python toolkit for techno-economic assessment of chemical process plants and energy systems with economic sensitivity and uncertainty evaluation},
journal = {SoftwareX},
volume = {35},
pages = {102816},
year = {2026},
issn = {2352-7110},
doi = {https://doi.org/10.1016/j.softx.2026.102816},
url = {https://www.sciencedirect.com/science/article/pii/S2352711026003080},
author = {P.B. Tamarona and T.J.H. Vlugt and M. Ramdin},
keywords = {Techno-economic assessment, Process design, Process plant, Power plant, Chemical engineering},
` ` `
```

---
## 📄 License

**OpenPyTEA** is released under the MIT License.

You are free to use, modify, and distribute the code with proper attribution.

---
## 📬 Contact
Panji B. Tamarona

📧 P.B.Tamarona@tudelft.nl

Repository: https://github.com/pbtamarona/OpenPyTEA