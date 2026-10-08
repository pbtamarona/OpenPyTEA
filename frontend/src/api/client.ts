import type {
  EquipmentItem, EquipmentInput, CostDBEntry, PlantConfig,
  CalculationResults, SensitivityResult, TornadoResult,
  MonteCarloMultiResult, PlantInput, ProjectMeta, ProjectOverview, PlantSnapshot,
} from "../types";

// Resolve the API base URL once per page load.
//
// In a Tauri shell the backend is spawned on a kernel-assigned port; we ask
// the Rust side for the URL via the `get_api_base` IPC command, polling
// until the port marker arrives on the backend's stdout. Outside Tauri
// (e.g. `start.sh` dev mode) we use VITE_API_BASE_URL / localhost:8000.
async function resolveBaseUrl(): Promise<string> {
  // Use Tauri's official isTauri() rather than a hand-rolled global check —
  // production webviews can inject internals after our module loads, making
  // naive `"__TAURI_INTERNALS__" in window` checks return false too early.
  const core = await import("@tauri-apps/api/core");
  if (core.isTauri()) {
    // Backend cold-start can take up to ~60s on first launch (matplotlib font
    // cache), longer still when AV software scans the bundle. Poll every
    // 200ms for up to 120s — keep this in sync with the startup-banner
    // poller in App.tsx.
    for (let i = 0; i < 600; i++) {
      const url = await core.invoke<string | null>("get_api_base");
      if (url) return url;
      await new Promise((r) => setTimeout(r, 200));
    }
    throw new Error("Backend did not start within 120s");
  }
  return import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";
}

let basePromise: Promise<string> | null = null;
const getBase = (): Promise<string> =>
  (basePromise ??= resolveBaseUrl().catch((e) => {
    // Never cache a rejection: a timed-out resolution (slow cold start)
    // must be retried on the next API call, not fail instantly forever.
    basePromise = null;
    throw e;
  }));

// Fetch a matplotlib-rendered figure (PNG) from the backend — the exact
// rendering the library produces in Jupyter. GET when no body is given.
export async function fetchPlotPng(path: string, body?: unknown): Promise<Blob> {
  const base = await getBase();
  const res = await fetch(`${base}${path}`, body === undefined ? undefined : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `HTTP ${res.status}`);
  }
  return res.blob();
}

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
export const runTornado = (params: {
  plus_minus_value: number; metric: string; additional_capex: boolean;
  extra_plants?: PlantInput[];
}) =>
  request<TornadoResult>("/analysis/tornado", { method: "POST", body: JSON.stringify(params) });
export const runMonteCarlo = (params: {
  num_samples: number; batch_size: number; additional_capex: boolean; extra_plants?: PlantInput[];
}) =>
  request<MonteCarloMultiResult>("/analysis/monte-carlo", { method: "POST", body: JSON.stringify(params) });

// Project (metadata + plant list). Every mutation returns the new overview,
// including the open plant's results.
export const getProject = () => request<ProjectOverview>("/project");
export const setProjectMeta = (meta: ProjectMeta) =>
  request<ProjectOverview>("/project/meta", { method: "PUT", body: JSON.stringify(meta) });
export const addPlant = (name?: string) =>
  request<ProjectOverview>("/project/plants", { method: "POST", body: JSON.stringify({ name: name ?? null }) });
export const activatePlant = (id: string) =>
  request<ProjectOverview>(`/project/plants/${id}/activate`, { method: "POST" });
export const duplicatePlant = (id: string) =>
  request<ProjectOverview>(`/project/plants/${id}/duplicate`, { method: "POST" });
export const renamePlant = (id: string, name: string) =>
  request<ProjectOverview>(`/project/plants/${id}`, { method: "PATCH", body: JSON.stringify({ name }) });
export const deletePlant = (id: string) =>
  request<ProjectOverview>(`/project/plants/${id}`, { method: "DELETE" });
export const getPlantSnapshot = (id: string) =>
  request<PlantSnapshot>(`/project/plants/${id}/snapshot`);

// Project I/O
export const newProject = () =>
  request<{ ok: boolean }>("/project/new", { method: "POST" });

export const saveProject = () => request<unknown>("/project/save", { method: "POST" });

/** The three run_tea result files (<plant>_equipment/plant/analysis_results.json)
 *  for the active plant, as file name + JSON text. */
export const exportJsonResults = () =>
  request<{ files: { name: string; content: string }[] }>("/project/export-json", { method: "POST" });

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

/** Load a project from raw JSON text (Tauri path: text comes from the
 *  read_project_text IPC command).
 *  Sends the parsed JSON directly to /project/load_json — bypasses the
 *  Blob → File → FormData multipart round-trip, which is unreliable inside
 *  Tauri's WebKit webview. */
export const loadProjectFromText = async (text: string) => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error("File is not valid JSON");
  }
  return request<{ ok: boolean; equipment_count: number }>(
    "/project/load_json",
    { method: "POST", body: JSON.stringify(parsed) },
  );
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
