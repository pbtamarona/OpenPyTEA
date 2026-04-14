export interface EquipmentItem {
  index: number;
  name: string;
  category: string;
  type: string | null;
  material: string;
  process_type: string;
  param: number | null;
  num_units: number | null;
  cost_year: number | null;
  target_year: number;
  purchased_cost: number;
  direct_cost: number;
}

export interface EquipmentInput {
  name: string;
  param?: number | null;
  process_type: string;
  category: string;
  type?: string | null;
  material: string;
  num_units?: number | null;
  purchased_cost?: number | null;
  cost_year?: number | null;
  cost_func?: string | null;
  target_year: number;
}

export interface CostDBEntry {
  key: string;
  type: string | null;
  units: string;
  s_lower: number | null;
  s_upper: number | null;
}

export interface PlantConfig {
  plant_name: string;
  process_type: string;
  country: string;
  region: string;
  currency: string;
  exchange_rate: number;
  interest_rate: number;
  project_lifetime: number;
  plant_utilization: number;
  tax_rate: number;
  working_capital: number | null;
  depreciation: Record<string, unknown> | null;
  operators_per_shift: number | null;
  operators_hired: number | null;
  operator_hourly_rate: { rate: number; std: number; min: number; max: number };
  working_weeks_per_year: number;
  working_shifts_per_week: number;
  operating_shifts_per_day: number;
  variable_opex_inputs: Record<string, Record<string, number>>;
  plant_products: Record<string, Record<string, number>>;
  fc: number | null;
  fp: number | null;
  additional_capex_years: number[] | null;
  additional_capex_cost: number[] | null;
}

export interface CalculationResults {
  capital_costs: Record<string, number | null>;
  variable_opex: { breakdown: Record<string, number>; total: number };
  fixed_opex: Record<string, number>;
  revenue: { breakdown: Record<string, number>; total: number };
  cash_flow: {
    capital_cost_array: number[];
    revenue_array: number[];
    cash_cost_array: number[];
    gross_profit_array: number[];
    depreciation_array: number[];
    taxable_income_array: number[];
    tax_paid_array: number[];
    cash_flow: number[];
    production_array: number[];
  };
  metrics: {
    levelized_cost: number | null;
    npv: number | null;
    irr: number | null;
    roi: number | null;
    payback_time: number | null;
  };
}

export interface SensitivityResult {
  curves: { plant: string; x: number[]; y: number[]; baseline: number }[];
  xlabel: string;
  ylabel: string;
  parameter: string;
  metric: string;
}

export interface TornadoResult {
  factors: string[];
  lows: number[];
  highs: number[];
  base_value: number;
  labels: string[];
  plus_minus_value: number;
  metric: string;
  xlabel: string;
}

export interface MCHistogram {
  bin_edges: number[];
  counts: number[];
}

export interface MCMetricStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  p5: number;
  p25: number;
  p50: number;
  p75: number;
  p95: number;
  histogram: MCHistogram;
}

export interface MonteCarloResult {
  name: string;
  num_samples: number;
  currency: string;
  metrics: Record<string, MCMetricStats>;
  inputs: Record<string, { mean: number; std: number; min: number; max: number; histogram: MCHistogram }>;
}
