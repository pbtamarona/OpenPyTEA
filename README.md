# 🧩 OpenPyTEA
**OpenPyTEA** is an open-source Python toolkit for **techno-economic assessment (TEA)** of chemical and energy systems.  It integrates **equipment cost estimation**, **cash-flow analysis**, and **uncertainty evaluation** into a **transparent and reproducible** workflow.  **OpenPyTEA** bridges the gap between process modeling and economic evaluation—empowering both researchers and students to perform standardized TEA directly in Python.

---

## ✨ Key Features
- **Modular architecture:** distinct modules for cost correlations, equipment modeling, plant-level assessment, and uncertainty analysis.  
- **Transparent and reproducible:** all equations and assumptions are openly defined for full traceability.  
- **Built-in uncertainty tools:** automatic generation of tornado plots and Monte Carlo simulations.  
- **Extensible:** easy integration with life-cycle assessment or optimization frameworks.  
- **Educational use:** ideal for teaching process design and cost analysis without commercial software.

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
- `matplotlib`  
- `scipy`  
- `openpyxl`  
- `tqdm`  
- `jupyter`

---

## ⚙️ Package Structure
```
src/openpytea/
├── equipment.py            # Equipment-level costing and inflation correction
├── plant.py                # Plant-level TEA: CAPEX, OPEX, cash flows, financial metrics
├── analysis.py             # Sensitivity and uncertainty analysis (tornado plots, Monte Carlo)
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
    eq_type='compressor',
    sizing_var=5000,  # kW
    material='carbon_steel'
)

print(compressor.installed_cost)
```

Each equipment item retrieves its cost correlation from the internal database in `data/cost_correlations.csv` and adjusts the cost to the desired year using the Chemical Engineering Plant Cost Index (CEPCI).

### 2. **Plant-level techno-economic assessment**

Multiple equipment objects can be grouped into a `Plant` instance for full TEA

```python
from openpytea.plant import Plant

plant = Plant(
    name='Hydrogen_Liquefaction',
    equipment_list=[compressor],
    location='Netherlands',
    lifetime=20,
    interest_rate=0.08,
    tax_rate=0.21
)

plant.run_cashflow(product_price=3.5, production_rate=10000)
print(plant.results_summary())
```
Main outputs include:
- Capital expenditures (CAPEX): inside/outside battery limits, engineering, contingency, and location factors
- Operating expenditures (OPEX): utilities, maintenance, labor, taxes, and overheads
- Financial metrics: Net Present Value (NPV), Internal Rate of Return (IRR), Return on Investment (ROI), Payback Period (PBP), and Levelized Cost of Product (LCOP)

### 3. **Sensitivity and uncertainty analysis**

**OpenPyTEA** provides integrated tools for visual and probabilistic analysis of cost and performance drivers.

Tornado Plot (One-at-a-Time Sensitivity)
```python
from openpytea.analysis import tornado_plot

tornado_plot(
    plant,
    params=['CAPEX', 'fuel_price', 'interest_rate', 'product_price'],
    metric='LCOP'
)
```
Monte Carlo Simulation
```python
from openpytea.analysis import monte_carlo

results = monte_carlo(
    plant,
    n_iter=5000,
    variable_std={'CAPEX': 0.15, 'fuel_price': 0.10},
    metric='NPV'
)
```
Outputs include probability distributions and confidence intervals for LCOP or NPV—supporting uncertainty-informed decision-making.

---

## 📘 Example Workflows

Example notebooks are available in the `examples/` folder, including:

- Hydrogen liquefaction  
- Hydrogen production  
- Geothermal power  
- Distillation  

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
  version      = {1.0.0}
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