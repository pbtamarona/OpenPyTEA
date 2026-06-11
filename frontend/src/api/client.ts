import type {
  EquipmentItem, EquipmentInput, CostDBEntry, PlantConfig,
  CalculationResults, SensitivityResult, TornadoResult,
  MonteCarloMultiResult, PlantInput,
} from "../types";

// Resolve the API base URL once per page load.
//
// In a Tauri shell the backend is spawned on a kernel-assigned port; we ask
// the Rust side for the URL via the `get_api_base` IPC command, polling
// until the port marker arrives on the backend's stdout. Outside Tauri
// (e.g. `start.sh` dev mode) we use VITE_API_BASE_URL / localhost:8000.
async function resolveBaseUrl(): Promise<string> {
  const tauri = "__TAURI_INTERNALS__" in window;
  if (tauri) {
    const { invoke } = await import("@tauri-apps/api/core");
    // Backend cold-start can take up to ~60s on first launch (matplotlib font
    // cache). Poll every 100ms.
    for (let i = 0; i < 700; i++) {
      const url = await invoke<string | null>("get_api_base");
      if (url) return url;
      await new Promise((r) => setTimeout(r, 100));
    }
    throw new Error("Backend did not start within 70s");
  }
  return import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";
}

let basePromise: Promise<string> | null = null;
const getBase = (): Promise<string> => (basePromise ??= resolveBaseUrl());

async function request<T>(path: string, opts?: RequestInit): Promise<T> {
  const base = await getBase();
  const res = await fetch(`${base}${path}`, {
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
export const runSensitivity = (params: {
  parameter: string; plus_minus_value: number; n_points: number; metric: string;
  additional_capex: boolean; extra_plants?: PlantInput[];
}) =>
  request<SensitivityResult>("/analysis/sensitivity", { method: "POST", body: JSON.stringify(params) });
export const runTornado = (params: { plus_minus_value: number; metric: string; additional_capex: boolean }) =>
  request<TornadoResult>("/analysis/tornado", { method: "POST", body: JSON.stringify(params) });
export const runMonteCarlo = (params: {
  num_samples: number; batch_size: number; additional_capex: boolean; extra_plants?: PlantInput[];
}) =>
  request<MonteCarloMultiResult>("/analysis/monte-carlo", { method: "POST", body: JSON.stringify(params) });

// Project I/O
export const saveProject = () => request<unknown>("/project/save", { method: "POST" });
export const loadProject = async (file: File) => {
  const base = await getBase();
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${base}/project/load`, { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
};

// Examples
export interface ExamplePreset {
  id: string;
  title: string;
  description: string;
}
export const getExamples = () => request<ExamplePreset[]>("/project/examples");
export const loadExample = (id: string) =>
  request<{ ok: boolean; title: string; equipment_count: number }>(`/project/examples/${id}`, { method: "POST" });
