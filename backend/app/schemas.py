"""Pydantic models for request/response bodies."""

from pydantic import BaseModel, Field
from typing import Any


class EquipmentIn(BaseModel):
    name: str
    param: float | None = None
    process_type: str = "Fluids"
    category: str = ""
    type: str | None = None
    material: str = "Carbon steel"
    num_units: int | None = None
    purchased_cost: float | None = None
    cost_year: int | None = None
    cost_func: str | None = None
    target_year: int = 2024


class EquipmentOut(BaseModel):
    index: int
    name: str
    category: str
    type: str | None
    material: str
    process_type: str
    param: float | None
    num_units: int | None
    cost_year: int | None
    target_year: int
    purchased_cost: float
    direct_cost: float


class PlantConfigIn(BaseModel):
    plant_name: str = "My Plant"
    process_type: str = "Fluids"
    country: str = "United States"
    region: str = "Gulf Coast"
    currency: str = "USD"
    exchange_rate: float = 1.0
    interest_rate: float = 0.09
    project_lifetime: int = 20
    plant_utilization: float = 1.0
    tax_rate: float = 0.0
    working_capital: float | None = None
    depreciation: dict | None = None
    operators_per_shift: int | None = None
    operators_hired: int | None = None
    operator_hourly_rate: dict = Field(
        default_factory=lambda: {"rate": 38.11, "std": 10, "min": 10, "max": 100}
    )
    working_weeks_per_year: int = 49
    working_shifts_per_week: int = 5
    operating_shifts_per_day: int = 3
    variable_opex_inputs: dict[str, dict] = Field(default_factory=dict)
    plant_products: dict[str, dict] = Field(default_factory=dict)
    fc: float | None = None
    fp: float | None = None
    additional_capex_years: list[int] | None = None
    additional_capex_cost: list[float] | None = None


class CalculationResults(BaseModel):
    capital_costs: dict[str, Any]
    variable_opex: dict[str, Any]
    fixed_opex: dict[str, Any]
    revenue: dict[str, Any]
    cash_flow: dict[str, Any]
    metrics: dict[str, Any]


class SensitivityIn(BaseModel):
    parameter: str
    plus_minus_value: float = 0.2
    n_points: int = 21
    metric: str = "LCOP"
    additional_capex: bool = False


class TornadoIn(BaseModel):
    plus_minus_value: float = 0.2
    metric: str = "LCOP"
    additional_capex: bool = False


class MonteCarloIn(BaseModel):
    num_samples: int = 50000
    batch_size: int = 1000
    additional_capex: bool = False
