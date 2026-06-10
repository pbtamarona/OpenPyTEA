"""OpenPyTEA: An open-source Python toolkit for techno-economic
assessment of chemical process plants and energy systems with
economic sensitivity and uncertainty evaluation.
"""

from .plant import Plant
from .equipment import Equipment
from .analysis import (direct_costs_data, fixed_capital_data,
                       fixed_opex_data, variable_opex_data,
                       sensitivity_data, tornado_data,
                       monte_carlo)
from .plotting import (plot_stacked_bar, plot_sensitivity,
                       plot_tornado, plot_monte_carlo,
                       plot_monte_carlo_inputs,
                       plot_multiple_monte_carlo)

from .io import (load_results, run_equipment, run_plant, run_tea)

__version__ = "2.1.0"
__author__ = "Panji B. Tamarona"
__email__ = "P.B.Tamarona@tudelft.nl"
__license__ = "MIT"
__description__ = (
    "OpenPyTEA: An open-source python toolkit for techno-economic"
    " assessment of chemical process plants and energy systems with"
    " economic sensitivity and uncertainty evaluation"
)
__url__ = "https://github.com/PBTamarona/OpenPyTEA"

__all__ = [
    "Plant",
    "Equipment",
    "direct_costs_data",
    "fixed_capital_data",
    "fixed_opex_data",
    "variable_opex_data",
    "sensitivity_data",
    "tornado_data",
    "monte_carlo",
    "plot_stacked_bar",
    "plot_sensitivity",
    "plot_tornado",
    "plot_monte_carlo",
    "plot_monte_carlo_inputs",
    "plot_multiple_monte_carlo",
    "load_results",
    "run_equipment",
    "run_plant",
    "run_tea"
]
