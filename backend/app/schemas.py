"""Pydantic models for request/response bodies."""

from pydantic import BaseModel, Field
from typing import Any


# ── Generic responses ──────────────────────────────────────────────


class OkResponse(BaseModel):
    ok: bool = True


class LoadResponse(BaseModel):
    ok: bool = True
    equipment_count: int


class LoadExampleResponse(BaseModel):
    ok: bool = True
    title: str | None = None
    equipment_count: int


# ── Equipment ──────────────────────────────────────────────────────


class EquipmentIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    param: float | None = Field(default=None, ge=0, le=1e12)
    process_type: str = "Fluids"
    category: str = Field(default="", max_length=255)
    type: str | None = Field(default=None, max_length=255)
    material: str = "Carbon steel"
    num_units: int | None = Field(default=None, ge=1, le=10000)
    purchased_cost: float | None = Field(default=None, ge=0, le=1e12)
    cost_year: int | None = Field(default=None, ge=1900, le=2100)
    cost_func: str | None = None
    target_year: int = Field(default=2024, ge=1900, le=2100)


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


class CostDBEntry(BaseModel):
    key: str
    type: str | None = None
    units: str = ""
    s_lower: float | None = None
    s_upper: float | None = None


class ExamplePreset(BaseModel):
    id: str
    title: str
    description: str = ""


# ── Plant configuration ───────────────────────────────────────────


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


# ── Calculation results ───────────────────────────────────────────


class CapitalCosts(BaseModel):
    purchased_cost: float
    isbl: float
    osbl: float
    design_and_engineering: float
    contingency: float
    fixed_capital: float
    working_capital: float | None = None


class VariableOpex(BaseModel):
    breakdown: dict[str, float]
    total: float


class FixedOpex(BaseModel):
    operating_labor: float
    supervision: float
    direct_salary_overhead: float
    laboratory_charges: float
    maintenance: float
    taxes_insurance: float
    rent_of_land: float
    environmental_charges: float
    operating_supplies: float
    general_plant_overhead: float
    interest_working_capital: float
    patents_royalties: float
    distribution_selling: float
    rnd: float
    total: float


class Revenue(BaseModel):
    breakdown: dict[str, float]
    total: float


class CashFlow(BaseModel):
    capital_cost_array: list[float]
    revenue_array: list[float]
    cash_cost_array: list[float]
    gross_profit_array: list[float]
    depreciation_array: list[float]
    taxable_income_array: list[float]
    tax_paid_array: list[float]
    cash_flow: list[float]
    production_array: list[float]


class Metrics(BaseModel):
    levelized_cost: float | None = None
    npv: float | None = None
    irr: float | None = None
    roi: float | None = None
    payback_time: float | None = None


class CalculationResults(BaseModel):
    capital_costs: CapitalCosts
    variable_opex: VariableOpex
    fixed_opex: FixedOpex
    revenue: Revenue
    cash_flow: CashFlow
    metrics: Metrics


# ── Analysis inputs ───────────────────────────────────────────────


class SensitivityIn(BaseModel):
    parameter: str
    plus_minus_value: float = Field(default=0.2, gt=0, le=10)
    n_points: int = Field(default=21, ge=3, le=1000)
    metric: str = "LCOP"
    additional_capex: bool = False


class TornadoIn(BaseModel):
    plus_minus_value: float = Field(default=0.2, gt=0, le=10)
    metric: str = "LCOP"
    additional_capex: bool = False


class MonteCarloIn(BaseModel):
    num_samples: int = Field(default=50000, ge=100, le=5_000_000)
    batch_size: int = Field(default=1000, ge=10, le=100_000)
    additional_capex: bool = False


# ── Analysis results ──────────────────────────────────────────────


class SensitivityCurve(BaseModel):
    plant: str
    x: list[float]
    y: list[float | None]
    baseline: float


class SensitivityResult(BaseModel):
    curves: list[SensitivityCurve]
    xlabel: str
    ylabel: str
    parameter: str
    metric: str


class TornadoResult(BaseModel):
    factors: list[str]
    lows: list[float]
    highs: list[float]
    base_value: float
    labels: list[str]
    plus_minus_value: float
    metric: str
    xlabel: str


class Histogram(BaseModel):
    bin_edges: list[float]
    counts: list[int]


class MCMetricStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    histogram: Histogram


class MCInputStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    histogram: Histogram


class MonteCarloResult(BaseModel):
    name: str
    num_samples: int
    currency: str
    metrics: dict[str, MCMetricStats]
    inputs: dict[str, MCInputStats]
