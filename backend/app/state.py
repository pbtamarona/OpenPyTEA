"""In-memory session state for a single OpenPyTEA project."""

from openpytea.equipment import Equipment
from openpytea.plant import Plant

equipment_list: list[Equipment] = []
plant: Plant | None = None
results: dict = {}
mc_results: dict | None = None
plant_config: dict = {}
