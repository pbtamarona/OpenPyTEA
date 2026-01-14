# 🧩 OpenPyTEA

**OpenPyTEA** is an open-source Python toolkit for performing **techno-economic assessment (TEA)** of chemical and energy systems. It was created to address a persistent gap in the TEA workflow: while process simulators model mass and energy balances, researchers often lack an equally transparent and flexible way to evaluate the **economic feasibility** of their designs. Commercial tools remain *black boxes*, and many academic TEA implementations are process-specific, undocumented, or difficult to reproduce.

**OpenPyTEA** provides a fully open, modular, and traceable framework that brings TEA into the Python ecosystem. By integrating **equipment cost estimation**, **capital and operating expenditure modeling**, **cash-flow analysis**, **cost breakdown plots**, **sensitivity evaluation**, and **Monte Carlo uncertainty propagation**, the toolkit enables users to perform end-to-end TEA with clarity and reproducibility.

Beyond its functionality, **OpenPyTEA is designed as a community-driven TEA platform**. Users can contribute new equipment cost correlations, improve economic models, report issues, and expand the toolkit’s capabilities over time. This collaborative approach helps build a shared, transparent, and continually improving TEA resource—similar to the open-source progress seen in the LCA community.

Whether used for early-stage process design, technology screening, or teaching, **OpenPyTEA** makes TEA more accessible, consistent, and aligned with FAIR research principles.

**For a full walkthrough of the features and usage of OpenPyTEA, refer to the `walkthrough.ipynb` notebook**:  
https://github.com/pbtamarona/OpenPyTEA/blob/main/walkthrough.ipynb

---

## ✨ Key Features
- **Modular architecture:** clean separation of cost correlations, equipment objects, plant economics, and uncertainty analysis.  
- **Transparent and reproducible:** all algorithms, equations, and assumptions are openly available for full traceability.
- **Cost breakdown visualization:** built-in helpers to plot stacked bar charts of direct equipment costs, fixed capital, and operating costs.
- **Built-in uncertainty tools:** automatic generation of sensitivity plots and Monte Carlo simulations.  
- **Interoperable and extensible:** easy integration with process simulators, optimization frameworks, and LCA tools.  
- **Education-friendly:** ideal for teaching TEA and process design without reliance on proprietary software.  
- **Community-driven:** users can contribute new correlations, improve models, request features, and shape the evolution of the platform.  

---

## 📦 Installation

### 1. **Using `uv`**

```bash
uv add git+https://github.com/pbtamarona/OpenPyTEA
```

### 2. **Using `pip`**

```bash
pip install git+https://github.com/pbtamarona/OpenPyTEA
```

**OpenPyTEA** requires **Python ≥ 3.11**.  
The main dependencies include:

- `numpy`  
- `pandas`  
- `scipy`  
- `matplotlib`  
- `scienceplots`  
- `tqdm`  

---

## ⚙️ Package Structure
```
src/openpytea/
├── equipment.py            # Equipment-level costing and inflation correction
├── plant.py                # Plant-level TEA: CAPEX, OPEX, cash flows, financial metrics
├── analysis.py             # Sensitivity and uncertainty analysis (sensitivity plots, Monte Carlo)
└── data/                   # Cost correlations database and CEPCI data
examples/                   # Example notebooks and case studies
walkthrough.ipynb           # walkthrough of the package
README.md
```

---

## 🧠 Core Concepts

### 1. **Equipment-level costing**

Each process unit (e.g., compressor, heat exchanger, reactor) is represented by an `Equipment` object:

```python
from openpytea.equipment import Equipment

compressor = Equipment(
    name='COMP',
    param=5000,  # kW
    category='Compressors & Blowers',
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

ammonia_plant = Plant(
    'name':'Ammonia Production Plant', 'country':'Netherlands',
    'process_type':'Fluids', 'equipment'=[compressor],
    'interest_rate':0.09, 'plant_utilization':0.95, 'project_lifetime':20,  # in years
    'plant_products': {  # Here we define the product(s) of the plant
        'ammonia': {
            'production':125_000, # Daily production in kg/day,
        }
    },
    'variable_opex_inputs':{
        'electricity':{
            'consumption': 110,  # Daily consumption, in MWh 
            'price': 75  # US$/MWh
        }
        'hydrogen':{
            'consumption': 22_000,  # Daily consumption, in kg/day
            'price': 2  # US$/kg
        }
    }
)

plant.calculate_cash_flow(print_results=True)
plant.calculate_levelized_cost()
```
Main outputs include:
- Capital expenditures (CAPEX): inside/outside battery limits, engineering, contingency, and location factors
- Operating expenditures (OPEX): variable and operating expenditures, including utilities, maintenance, labor, and overhead costs
- Financial metrics: Net Present Value (NPV), Internal Rate of Return (IRR), Return on Investment (ROI), Payback Time (PBT), and Levelized Cost of Product (LCOP)

### 3. **CAPEX and OPEX breakdown plots**

OpenPyTEA includes convenience functions for visualizing the economic structure of a process plant using stacked bar plots:

- `plot_direct_costs_bar(plant)`: direct equipment costs (per equipment item).  
- `plot_fixed_capital_bar(plant)`: fixed capital components (ISBL, OSBL, design & engineering, contingency).  
- `plot_variable_opex_bar(plant)`: variable operating costs by input mass and energy stream.  
- `plot_fixed_opex_bar(plant)`: fixed operating expenses, including labor, supervision, maintenance, overhead, R&D, and more.

These plots provide a quick visual breakdown of the main CAPEX and OPEX contributors in a flowsheet.

### 3. **Sensitivity and uncertainty analysis**

**OpenPyTEA** provides integrated tools for visual sensitivity and probabilistic analysis of cost and performance drivers.

One-Way Sensitivity Line Plot
```python
from openpytea.analysis import sensitivity_plot

results = sensitivity_plot(
    plant, 
    parameter="electricity", 
    plus_minus_value =0.5
    )
```
The `plant` input may also be a list of `Plant` objects to generate comparison plots.

Tornado Plot (One-at-a-Time Sensitivity)
```python
from openpytea.analysis import tornado_plot

tornado_plot(
    plant,
    plus_minus_value = 0.5,
)
```

Monte Carlo Simulation
```python
from openpytea.analysis import monte_carlo

results = monte_carlo(
    plant,
    num_samples=1_000_000
)

```
Outputs include probability distributions and confidence intervals for LCOP or NPV—supporting uncertainty-informed decision-making. With `plot_multiple_monte_carlo`, **OpenPyTEA** can also visualize Monte Carlo results for multiple plants to enable uncertainty comparisons.

---

## 📘 Example Workflows

Example notebooks are available in the `examples/` folder, including:

- Hydrogen liquefaction  
- Hydrogen production  
- Geothermal heat and power  

Run any example via:
```bash
jupyter notebook examples/hydrogen_liquefaction.ipynb
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

## 📚 Citation

If you use **OpenPyTEA** in your research, please cite it using the automatic GitHub citation feature or the `CITATION.cff` file included in this repository.

On GitHub, click:
```
Repository page → "Cite this repository"
```
This will provide formatted citation export options (BibTeX, APA, MLA, etc.) based on the CITATION.cff metadata.

Or if you prefer to cite manually, you may use:

> Tamarona, P.B., Vlugt, T.J.H., & Ramdin, M. (2025). *OpenPyTEA: An open-source python toolkit for techno-economic assessment of process plants with economic sensitivity and uncertainty evaluation.* GitHub Repository. Available at: [https://github.com/pbtamarona/OpenPyTEA](https://github.com/pbtamarona/OpenPyTEA)

**BibTeX:**
```bibtex
@misc{tamarona2025openpytea,
  author       = {Panji B. Tamarona and Thijs J.H. Vlugt and Mahinder Ramdin},
  title        = {OpenPyTEA: An open-source python toolkit for techno-economic assessment of process plants with economic sensitivity and uncertainty evaluation},
  year         = {2025},
  url          = {\url{https://github.com/pbtamarona/OpenPyTEA}},
  version      = {1.2.0}
  note         = {Accessed: YYYY-MM-DD}
}
```

---

## 📄 License

**OpenPyTEA** is released under the MIT License.

You are free to use, modify, and distribute the code with proper attribution.

## 📬 Contact
Panji B. Tamarona

📧 P.B.Tamarona@tudelft.nl

Repository: https://github.com/pbtamarona