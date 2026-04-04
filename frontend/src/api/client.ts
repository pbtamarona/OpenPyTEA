const BASE = "http://localhost:8000/api";

async function request<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// Equipment
import type { EquipmentItem, EquipmentInput, CostDBEntry, PlantConfig, CalculationResults, SensitivityResult, TornadoResult, MonteCarloResult } from "../types";

export const getEquipment = () => request<EquipmentItem[]>("/equipment");
export const addEquipment = (eq: EquipmentInput) =>
  request<EquipmentItem>("/equipment", { method: "POST", body: JSON.stringify(eq) });
export const updateEquipment = (idx: number, eq: EquipmentInput) =>
  request<EquipmentItem>(`/equipment/${idx}`, { method: "PUT", body: JSON.stringify(eq) });
export const deleteEquipment = (idx: number) =>
  request<{ ok: boolean }>(`/equipment/${idx}`, { method: "DELETE" });

export const getCostDBCategories = () =>
  request<Record<string, CostDBEntry[]>>("/equipment/cost-db/categories");
export const getProcessTypes = () => request<string[]>("/equipment/process-types");
export const getMaterials = () => request<string[]>("/equipment/materials");

// Plant
export const getPlantConfig = () => request<PlantConfig>("/plant/config");
export const setPlantConfig = (cfg: PlantConfig) =>
  request<{ ok: boolean }>("/plant/config", { method: "PUT", body: JSON.stringify(cfg) });
export const getLocations = () => request<Record<string, unknown>>("/plant/locations");
export const runCalculations = () =>
  request<CalculationResults>("/plant/calculate", { method: "POST" });

// Analysis
export const getSensitivityParameters = () => request<string[]>("/analysis/sensitivity/parameters");
export const runSensitivity = (params: { parameter: string; plus_minus_value: number; n_points: number; metric: string; additional_capex: boolean }) =>
  request<SensitivityResult>("/analysis/sensitivity", { method: "POST", body: JSON.stringify(params) });
export const runTornado = (params: { plus_minus_value: number; metric: string; additional_capex: boolean }) =>
  request<TornadoResult>("/analysis/tornado", { method: "POST", body: JSON.stringify(params) });
export const runMonteCarlo = (params: { num_samples: number; batch_size: number; additional_capex: boolean }) =>
  request<MonteCarloResult>("/analysis/monte-carlo", { method: "POST", body: JSON.stringify(params) });

// Project I/O
export const saveProject = () => request<unknown>("/project/save", { method: "POST" });
export const loadProject = async (file: File) => {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/project/load`, { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
};
