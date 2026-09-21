"""
Regenerate every figure in ``docs/_static/plotting`` used by the
"Analysis and Plotting" user guide page.

The plant configuration mirrors the walkthrough notebooks (Parts 3 and 4)
so the figures match what readers see when they run the notebooks.

Every single-panel figure is rendered at the same ``figsize`` and DPI so the
documentation page looks uniform. Run from the repository root:

    python docs/generate_figures.py
"""

from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from openpytea import Equipment, Plant  # noqa: E402
from openpytea.analysis import (  # noqa: E402
    cash_flow_data,
    direct_costs_data,
    fixed_capital_data,
    fixed_opex_data,
    levelized_cost_data,
    monte_carlo,
    sensitivity_data,
    tornado_data,
    variable_opex_data,
)
from openpytea.plotting import (  # noqa: E402
    plot_cash_flow,
    plot_monte_carlo,
    plot_monte_carlo_inputs,
    plot_multiple_monte_carlo,
    plot_sensitivity,
    plot_stacked_bar,
    plot_tornado,
)

OUT_DIR = Path(__file__).parent / "_static" / "plotting"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# One common size for every single-panel chart
FIGSIZE = (3.2, 2.2)
# Stacked bars: width is per bar (auto-scaled by plot_stacked_bar), height fixed
BAR_FIGSIZE = (1.3, 2.2)
DPI = 300

NUM_SAMPLES = 100_000
BATCH_SIZE = 10_000
SEED = 42


def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png")


# ----------------------------------------------------------------------
# Demo plant (walkthrough Parts 1-2)
# ----------------------------------------------------------------------
hx = Equipment(
    name="HX-101", param=900, process_type="Fluids",
    category="Heat Exchangers", type="U-tube shell & tube",
    material="316 stainless steel",
)
comp1 = Equipment(
    name="Comp-1", process_type="Fluids", material="Carbon steel", param=1,
    category="Compressors, fans, & Blowers", type="Compressor, centrifugal",
    cost_func="co2_compressor_manzolini_2011",
)
comp2 = Equipment(
    name="Air Compressor", param=50_000, process_type="Fluids",
    category="Compressors, fans, & blowers", type="Compressor, centrifugal",
)

config = {
    "plant_name": "Demo Plant",
    "process_type": "Fluids",
    "country": "United States",
    "region": "Gulf Coast",
    "loc_factor": 1.10,
    "currency": "USD",
    "exchange_rate": 1.0,
    "equipment": [hx, comp1, comp2],
    "interest_rate": 0.08,
    "project_lifetime": 20,
    "plant_utilization": 0.90,
    "tax_rate": 0.25,
    "fixed_capital_factors": {"osbl": 0.25, "de": 0.35},
    "fixed_capital_components": {"contingency": 15_000_000},
    "operators_hired": 12,
    "operator_hourly_rate": {"rate": 42.0},
    "working_weeks_per_year": 46,
    "working_shifts_per_week": 5,
    "operating_shifts_per_day": 3,
    "fixed_opex_factors": {"maintenance": 0.06, "rent_of_land": 0.01, "rnd": 0.0},
    "fixed_opex_components": {"supervision_costs": 100_000},
    "plant_products": {
        "methanol": {"production": 100_000, "price": 1.95},
        "hydrogen": {"production": 75_000, "price": 1.25},
    },
    "variable_opex_inputs": {
        "electricity":   {"consumption": 1.4e6, "price": 0.075},
        "cooling_water": {"consumption": 1.6e6, "price": 0.0007},
        "steam":         {"consumption": 4.0e5, "price": 0.02},
        "natural_gas":   {"consumption": 1.0e5, "price": 0.035},
    },
    "working_capital": None,
    "additional_capex_cost": [500_000, 200_000],
    "additional_capex_years": [8, 15],
    "capex_ramp": [0.2, 0.5, 0.2, 0.1],
    "production_ramp": [0, 0, 0, 0, 0.4, 0.8],
    "depreciation": {
        "method": "declining_balance", "life": 10, "db_factor": 2.0,
        "salvage_fraction": 0.1, "service_start_year": 2,
    },
}

demo_plant = Plant(config)
demo_plant.calculate_all()

copy_plant = deepcopy(demo_plant)
copy_plant.update_configuration({
    "plant_name": "Copied Plant",
    "variable_opex_inputs": {
        "electricity":   {"consumption": 0.9e6, "price": 0.05},
        "cooling_water": {"consumption": 2.0e6, "price": 0.0007},
        "steam":         {"consumption": 1.3e5, "price": 0.02},
        "natural_gas":   {"consumption": 1.1e6, "price": 0.015},
    },
})
copy_plant.calculate_all()

# ----------------------------------------------------------------------
# Cost breakdowns
# ----------------------------------------------------------------------
print("Cost breakdowns")
fig, _ = plot_stacked_bar(direct_costs_data(plants=demo_plant), figsize=BAR_FIGSIZE, show=False)
save(fig, "direct_costs")
fig, _ = plot_stacked_bar(fixed_capital_data(plants=demo_plant, additional_capex=True), figsize=BAR_FIGSIZE, show=False)
save(fig, "fixed_capital")
fig, _ = plot_stacked_bar(fixed_opex_data(plants=demo_plant, pct=True), figsize=BAR_FIGSIZE, show=False)
save(fig, "fixed_opex")
fig, _ = plot_stacked_bar(variable_opex_data(plants=demo_plant), figsize=BAR_FIGSIZE, show=False)
save(fig, "variable_opex")
fig, _ = plot_stacked_bar(variable_opex_data(plants=[demo_plant, copy_plant]), figsize=BAR_FIGSIZE, show=False)
save(fig, "variable_opex_multi")
fig, _ = plot_stacked_bar(levelized_cost_data(plants=demo_plant), figsize=BAR_FIGSIZE, show=False)
save(fig, "levelized_cost")

# ----------------------------------------------------------------------
# Cash flow
# ----------------------------------------------------------------------
print("Cash flow")
fig, _ = plot_cash_flow(cash_flow_data(demo_plant), figsize=FIGSIZE, show=False)
save(fig, "cash_flow")
fig, _ = plot_cash_flow(cash_flow_data([demo_plant, copy_plant]), figsize=FIGSIZE, show=False)
save(fig, "cash_flow_multi")

# ----------------------------------------------------------------------
# Sensitivity
# ----------------------------------------------------------------------
print("Sensitivity")
sens = sensitivity_data(plants=demo_plant, parameter="electricity", plus_minus_value=0.5)
fig, _ = plot_sensitivity(sens, figsize=FIGSIZE, show=False)
save(fig, "sensitivity")

npv_sens = sensitivity_data(
    plants=demo_plant, parameter="methanol", metric="NPV",
    plus_minus_value=0.5, label="Project A - NPV / [USD]",
)
fig, _ = plot_sensitivity(npv_sens, figsize=FIGSIZE, show=False)
save(fig, "sensitivity_npv")

sens_multi = sensitivity_data(
    plants=[demo_plant, copy_plant], parameter="electricity",
    metric="PBT", plus_minus_value=0.5, additional_capex=True, n_points=50,
)
fig, _ = plot_sensitivity(sens_multi, figsize=FIGSIZE, show=False)
save(fig, "sensitivity_multi")

production_sens = sensitivity_data(
    plants=demo_plant, parameter="methanol.production", plus_minus_value=0.5,
)
fig, ax = plot_sensitivity(production_sens, figsize=FIGSIZE, show=False)
ax.set_title("LCOP vs. methanol production")
ax.legend(loc="upper right")
ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(which="both", top=False, right=False)
save(fig, "sensitivity_custom_axes")

# ----------------------------------------------------------------------
# Tornado
# ----------------------------------------------------------------------
print("Tornado")
fig, _ = plot_tornado(tornado_data(plant=demo_plant, plus_minus_value=0.5), figsize=FIGSIZE, show=False)
save(fig, "tornado_lcop")
fig, _ = plot_tornado(tornado_data(plant=demo_plant, plus_minus_value=0.5, metric="ROI"), figsize=FIGSIZE, show=False)
save(fig, "tornado_roi")
fig, _ = plot_tornado(
    tornado_data(plant=demo_plant, plus_minus_value=0.5, include_process_params=True),
    figsize=(FIGSIZE[0], 3.4), show=False,
)
save(fig, "tornado_process")

# ----------------------------------------------------------------------
# Monte Carlo (walkthrough Part 4, independent inputs)
# ----------------------------------------------------------------------
print("Monte Carlo")
demo_plant.update_configuration({
    "plant_products": {
        "methanol": {
            "production": 150_000,
            "production_uncertainty": {"std": 15_000, "min": 100_000, "max": 200_000},
            "price": 1.75,
            "price_uncertainty": {"dist_id": 5, "min": 1.25, "max": 2.25},
        },
        "hydrogen": {
            "production": 50_000,
            "price": 1.25,
            "price_uncertainty": {"std": 0.1, "min": 0.75, "max": 1.75},
        },
    },
    "operator_hourly_rate": {
        "rate": 38.11,
        "rate_uncertainty": {"std": 10.0, "min": 20.0, "max": 60.0},
    },
    "variable_opex_inputs": {
        "electricity": {
            "consumption": 1.4e6,
            "consumption_uncertainty": {"std": 1.4e5, "min": 1.0e6, "max": 1.8e6},
            "price": 0.1,
            "price_uncertainty": {"std": 0.035, "min": 0.025, "max": 0.175},
        },
        "cooling_water": {
            "consumption": 1.6e6, "price": 0.0008,
            "price_uncertainty": {"dist_id": 9, "loc": 0.0, "scale": 0.0002, "shape": 4},
        },
        "steam": {
            "consumption": 4.0e5, "price": 0.04,
            "price_uncertainty": {"dist_id": 4, "min": 0.00, "max": 0.08},
        },
        "natural_gas": {
            "consumption": 1.0e5, "price": 0.05,
            "price_uncertainty": {"std": 0.03, "min": 0.001, "max": 0.10},
        },
    },
    "project_uncertainties": {
        "fixed_capital_factor": {"std": 0.3, "min": 0.25, "max": 1.75},
        "fixed_opex_factor":    {"std": 0.3, "min": 0.25, "max": 1.75},
        "project_lifetime":     {"std": 5},
        "interest_rate":        {"std": 0.03},
    },
})

mc_results = monte_carlo(demo_plant, num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE, random_seed=SEED)

fig, _ = plot_monte_carlo(demo_plant, metric="LCOP", bins=30, figsize=FIGSIZE, show=False)
save(fig, "monte_carlo_lcop")

fig_p, _, fig_e, _ = plot_monte_carlo_inputs(mc_results, bins=40, show=False)
save(fig_p, "monte_carlo_inputs_process")
save(fig_e, "monte_carlo_inputs_economic")

copy_plant.update_configuration({
    "plant_products": {
        "methanol": {"production": 150_000, "price": 1.75,
                     "price_uncertainty": {"dist_id": 5, "min": 1.25, "max": 2.25}},
        "hydrogen": {"production": 50_000, "price": 1.25,
                     "price_uncertainty": {"std": 0.1, "min": 0.75, "max": 1.75}},
    },
    "variable_opex_inputs": {
        "electricity":   {"price": 0.05, "price_uncertainty": {"std": 0.025, "min": 0.01, "max": 0.1}},
        "cooling_water": {"price": 0.0007, "price_uncertainty": {"dist_id": 5, "min": 0.0001, "max": 0.0014}},
        "steam":         {"price": 0.02, "price_uncertainty": {"dist_id": 2, "loc": -3.91, "scale": 0.3, "min": 0.00, "max": 0.08}},
        "natural_gas":   {"price": 0.015, "price_uncertainty": {"dist_id": 9, "loc": 0.0, "scale": 0.005, "shape": 3}},
    },
})
monte_carlo(copy_plant, num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE, random_seed=SEED)

fig, _ = plot_multiple_monte_carlo(data_list=[demo_plant, copy_plant], metric="LCOP", bins=30, figsize=FIGSIZE, show=False)
save(fig, "monte_carlo_multiple")

# ----------------------------------------------------------------------
# Dependencies (walkthrough Part 4)
# ----------------------------------------------------------------------
print("Dependencies")
demo_plant.update_configuration({
    "plant_products": {
        "hydrogen": {
            "production_dependency": {
                "depends_on": {"production:methanol": 50_000 / 150_000},
            }
        },
    },
    "variable_opex_inputs": {
        "cooling_water": {
            "consumption_dependency": {
                "depends_on": {"production:methanol": 1.6e6 / 150_000},
            }
        },
        "steam": {
            "consumption_dependency": {
                "depends_on": {"consumption:cooling_water": 4.0e5 / 1.6e6},
            }
        },
        "natural_gas": {
            "consumption_dependency": {
                "depends_on": {"production:methanol": 0.4667, "production:hydrogen": 0.6},
            }
        },
    },
    "project_uncertainties": {
        "fixed_capital_factor": {
            "dependency": {"depends_on": {"production:methanol": 2.0e-6}, "offset": 0.7},
        },
    },
})
demo_plant.update_configuration({
    "variable_opex_inputs": {
        "cooling_water": {"consumption_uncertainty": {"noise": 5.0e4}},
    },
    "project_uncertainties": {
        "fixed_capital_factor": {"noise": 0.1},
    },
})

mc_dep = monte_carlo(demo_plant, num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE, random_seed=SEED)

fig, _ = plot_monte_carlo_inputs(mc_dep, category="process", bins=40, show=False)
save(fig, "dependency_inputs_process")
fig, _ = plot_monte_carlo_inputs(mc_dep, category="economic", bins=40, show=False)
save(fig, "dependency_inputs_economic")

# Scatter plots: each dependent against its driver(s)
inputs = mc_dep["inputs"]
pairs = [
    ("Methanol production", "Cooling Water consumption",
     "Methanol production / [units/day]", "Cooling water consumption / [units/day]",
     "dependency_scatter_cooling_water"),
    ("Cooling Water consumption", "Steam consumption",
     "Cooling water consumption / [units/day]", "Steam consumption / [units/day]",
     "dependency_scatter_steam"),
    ("Methanol production", "Hydrogen production",
     "Methanol production / [units/day]", "Hydrogen production / [units/day]",
     "dependency_scatter_hydrogen"),
    ("Methanol production", "Fixed capital factor",
     "Methanol production / [units/day]", "Fixed capital factor / [-]",
     "dependency_scatter_fixed_capital"),
]
for xkey, ykey, xlabel, ylabel, name in pairs:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(inputs[xkey], inputs[ykey], s=1, alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save(fig, name)

ng_mean = 0.4667 * inputs["Methanol production"] + 0.6 * inputs["Hydrogen production"]
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.scatter(ng_mean, inputs["Natural Gas consumption"], s=1, alpha=0.3)
ax.set_xlabel(r"$0.4667 \times$ methanol $+ 0.6 \times$ hydrogen")
ax.set_ylabel("Natural gas consumption")
save(fig, "dependency_scatter_natural_gas")

# Deterministic analyses with vs. without dependencies
independent_plant = deepcopy(demo_plant)
independent_plant.update_configuration({"plant_name": "Demo Plant (no dependencies)"})
for props in independent_plant.variable_opex_inputs.values():
    props.pop("consumption_dependency", None)
for props in independent_plant.plant_products.values():
    props.pop("production_dependency", None)
independent_plant.project_uncertainties["fixed_capital_factor"].pop("dependency", None)

TORNADO_TALL = (FIGSIZE[0], 3.4)
fig, _ = plot_tornado(
    tornado_data(plant=independent_plant, plus_minus_value=0.5, metric="ROI", include_process_params=True),
    figsize=TORNADO_TALL, show=False,
)
save(fig, "dependency_tornado_independent")
fig, _ = plot_tornado(
    tornado_data(plant=demo_plant, plus_minus_value=0.5, metric="ROI", include_process_params=True),
    figsize=TORNADO_TALL, show=False,
)
save(fig, "dependency_tornado_dependent")

roi_comparison = sensitivity_data(
    plants=[demo_plant, independent_plant], parameter="methanol.production",
    metric="ROI", plus_minus_value=0.5,
)
fig, _ = plot_sensitivity(roi_comparison, figsize=FIGSIZE, show=False)
save(fig, "dependency_sensitivity")

print("Done.")
