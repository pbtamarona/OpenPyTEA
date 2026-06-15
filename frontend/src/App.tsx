import { useState, useRef, useEffect, useCallback } from "react";
import EquipmentPage from "./pages/EquipmentPage";
import PlantConfigPage from "./pages/PlantConfigPage";
import ResultsPage from "./pages/ResultsPage";
import AnalysisPage from "./pages/AnalysisPage";
import MonteCarloPage from "./pages/MonteCarloPage";
import {
  saveProject, loadProject, loadProjectFromText,
  newProject, getExamples, loadExample,
} from "./api/client";
import type { ExamplePreset } from "./api/client";
import ComparePage from "./pages/ComparePage";
import WelcomePage from "./pages/WelcomePage";
import type { CalculationResults, ComparedPlant, PlantInput } from "./types";
import "./App.css";

const TABS = ["Equipment", "Plant Config", "Results", "Analysis", "Monte Carlo", "Compare"] as const;
const PROJECT_EXT = "openpytea";

/** Detect Tauri at runtime. Cached on first call. */
let _isTauriCache: boolean | null = null;
async function detectTauri(): Promise<boolean> {
  if (_isTauriCache !== null) return _isTauriCache;
  try {
    const { isTauri } = await import("@tauri-apps/api/core");
    _isTauriCache = isTauri();
  } catch {
    _isTauriCache = false;
  }
  return _isTauriCache;
}

const basename = (path: string | null): string => {
  if (!path) return "Untitled";
  const last = path.split("/").pop() ?? path;
  return last.replace(new RegExp(`\\.${PROJECT_EXT}$`), "");
};

function App() {
  const [tab, setTab] = useState<(typeof TABS)[number]>("Equipment");
  const [results, setResults] = useState<CalculationResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [examples, setExamples] = useState<ExamplePreset[]>([]);
  const [fileMenuOpen, setFileMenuOpen] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [dark, setDark] = useState(() => {
    const saved = localStorage.getItem("openpytea-theme");
    if (saved) return saved === "dark";
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  });
  const [comparedPlants, setComparedPlants] = useState<ComparedPlant[]>([]);
  const [showWelcome, setShowWelcome] = useState(true);

  // Project file state
  const [currentPath, setCurrentPath] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const markDirty = useCallback(() => setDirty(true), []);

  const fileRef = useRef<HTMLInputElement>(null);

  const addToComparison = async (name: string, currency: string, r: CalculationResults) => {
    let source: PlantInput | undefined;
    try {
      const data = (await saveProject()) as { equipment?: unknown[]; plant?: unknown };
      if (Array.isArray(data.equipment) && data.equipment.length > 0 && data.plant) {
        source = {
          name,
          equipment: data.equipment as Record<string, unknown>[],
          plant: data.plant as Record<string, unknown>,
        };
      }
    } catch {
      // non-fatal — plant just won't be available for analysis overlays
    }
    setComparedPlants((prev) => [
      ...prev,
      { id: crypto.randomUUID(), name, currency, results: r, source },
    ]);
  };

  const removeFromComparison = (id: string) => {
    setComparedPlants((prev) => prev.filter((p) => p.id !== id));
  };

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
    localStorage.setItem("openpytea-theme", dark ? "dark" : "light");
  }, [dark]);

  useEffect(() => {
    if (showWelcome) return;
    getExamples().then(setExamples).catch((e: unknown) => {
      // non-critical — examples just won't appear in the dropdown
      console.warn("Examples fetch failed:", e);
    });
  }, [showWelcome]);

  // Window title reflects current project + dirty state.
  useEffect(() => {
    const title = `OpenPyTEA — ${basename(currentPath)}${dirty ? " •" : ""}`;
    document.title = title;
    (async () => {
      if (!(await detectTauri())) return;
      try {
        const { getCurrentWindow } = await import("@tauri-apps/api/window");
        await getCurrentWindow().setTitle(title);
      } catch {
        // ignore — title is cosmetic
      }
    })();
  }, [currentPath, dirty]);

  // ── Project file operations ───────────────────────────────────────────

  const confirmIfDirty = (verb: string): boolean =>
    !dirty || window.confirm(`You have unsaved changes. ${verb} anyway?`);

  const handleNew = useCallback(async () => {
    setFileMenuOpen(false);
    if (!confirmIfDirty("Start a new project")) return;
    try {
      await newProject();
      setCurrentPath(null);
      setDirty(false);
      setResults(null);
      setError(null);
      setRefreshKey((k) => k + 1);
      setTab("Equipment");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "New project failed");
    }
  }, [dirty]);

  const handleOpen = useCallback(async () => {
    setFileMenuOpen(false);
    if (!confirmIfDirty("Open another project")) return;
    try {
      if (await detectTauri()) {
        const [{ open }, { readTextFile }] = await Promise.all([
          import("@tauri-apps/plugin-dialog"),
          import("@tauri-apps/plugin-fs"),
        ]);
        const selected = await open({
          multiple: false,
          filters: [{ name: "OpenPyTEA Project", extensions: [PROJECT_EXT, "json"] }],
        });
        if (!selected || typeof selected !== "string") return; // user cancelled
        const text = await readTextFile(selected);
        await loadProjectFromText(text);
        setCurrentPath(selected);
      } else {
        // Browser fallback: hidden <input type=file>
        fileRef.current?.click();
        return; // handleBrowserLoad takes over
      }
      setDirty(false);
      setResults(null);
      setError(null);
      setRefreshKey((k) => k + 1);
      setTab("Equipment");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Open failed");
    }
  }, [dirty]);

  const handleBrowserLoad = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      await loadProject(file);
      setCurrentPath(file.name); // browser only gives us a basename
      setDirty(false);
      setResults(null);
      setError(null);
      setRefreshKey((k) => k + 1);
      setTab("Equipment");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Load failed");
    }
    e.target.value = "";
  };

  const writeProjectTo = async (path: string) => {
    const data = await saveProject();
    const text = JSON.stringify(data, null, 2);
    const { writeTextFile } = await import("@tauri-apps/plugin-fs");
    await writeTextFile(path, text);
  };

  const handleSaveAs = useCallback(async () => {
    setFileMenuOpen(false);
    try {
      if (await detectTauri()) {
        const { save } = await import("@tauri-apps/plugin-dialog");
        const path = await save({
          defaultPath: currentPath ?? `untitled.${PROJECT_EXT}`,
          filters: [{ name: "OpenPyTEA Project", extensions: [PROJECT_EXT] }],
        });
        if (!path) return; // user cancelled
        await writeProjectTo(path);
        setCurrentPath(path);
        setDirty(false);
      } else {
        // Browser fallback: blob download
        const data = await saveProject();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${basename(currentPath)}.${PROJECT_EXT}`;
        a.click();
        URL.revokeObjectURL(url);
        setDirty(false);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Save failed");
    }
  }, [currentPath]);

  const handleSave = useCallback(async () => {
    setFileMenuOpen(false);
    // Without a known path (or outside Tauri), Save behaves as Save As.
    if (!currentPath || !(await detectTauri())) {
      return handleSaveAs();
    }
    try {
      await writeProjectTo(currentPath);
      setDirty(false);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Save failed");
    }
  }, [currentPath, handleSaveAs]);

  const handleLoadExample = async (id: string) => {
    setFileMenuOpen(false);
    if (!confirmIfDirty("Load an example")) return;
    try {
      await loadExample(id);
      setCurrentPath(null);
      setDirty(false);
      setResults(null);
      setError(null);
      setRefreshKey((k) => k + 1);
      setTab("Equipment");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load example");
    }
  };

  // Keyboard shortcuts (Cmd / Ctrl + N / O / S / Shift+S).
  // Use a ref so the listener always calls the freshest handler closures.
  const handlersRef = useRef({ handleNew, handleOpen, handleSave, handleSaveAs });
  handlersRef.current = { handleNew, handleOpen, handleSave, handleSaveAs };
  useEffect(() => {
    if (showWelcome) return;
    const onKey = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey)) return;
      const k = e.key.toLowerCase();
      const h = handlersRef.current;
      if (k === "n" && !e.shiftKey) { e.preventDefault(); h.handleNew(); }
      else if (k === "o" && !e.shiftKey) { e.preventDefault(); h.handleOpen(); }
      else if (k === "s" && !e.shiftKey) { e.preventDefault(); h.handleSave(); }
      else if (k === "s" && e.shiftKey) { e.preventDefault(); h.handleSaveAs(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [showWelcome]);

  if (showWelcome) {
    return <WelcomePage onContinue={() => setShowWelcome(false)} />;
  }

  const cmdKey = navigator.platform.includes("Mac") ? "⌘" : "Ctrl";

  return (
    <div className="app">
      <header className="header">
        <img src="/logo.png" alt="OpenPyTEA" className="brand-logo" />
        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t} className={tab === t ? "tab active" : "tab"} onClick={() => setTab(t)}>
              {t}
            </button>
          ))}
        </nav>
        <div className="header-actions">
          <div className="examples-dropdown">
            <button className="btn-examples" onClick={() => setFileMenuOpen(!fileMenuOpen)}>
              File ▾
            </button>
            {fileMenuOpen && (
              <>
                <div className="dropdown-backdrop" onClick={() => setFileMenuOpen(false)} />
                <div className="dropdown-menu file-menu">
                  <button className="dropdown-item file-item" onClick={handleNew}>
                    <span>New Project</span><span className="shortcut">{cmdKey}N</span>
                  </button>
                  <button className="dropdown-item file-item" onClick={handleOpen}>
                    <span>Open…</span><span className="shortcut">{cmdKey}O</span>
                  </button>
                  <button className="dropdown-item file-item" onClick={handleSave}>
                    <span>Save</span><span className="shortcut">{cmdKey}S</span>
                  </button>
                  <button className="dropdown-item file-item" onClick={handleSaveAs}>
                    <span>Save As…</span><span className="shortcut">{cmdKey}⇧S</span>
                  </button>
                  <div className="dropdown-divider" />
                  <div className="dropdown-section">Examples</div>
                  {examples.map((ex) => (
                    <button key={ex.id} className="dropdown-item" onClick={() => handleLoadExample(ex.id)}>
                      <span className="dropdown-item-title">{ex.title}</span>
                      <span className="dropdown-item-desc">{ex.description}</span>
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
          <input ref={fileRef} type="file" accept=".openpytea,.json" hidden onChange={handleBrowserLoad} />
          <button className="btn-theme" onClick={() => setDark((d) => !d)} title={dark ? "Switch to light mode" : "Switch to dark mode"}>
            {dark ? "☀️" : "🌙"}
          </button>
        </div>
      </header>
      {error && <div className="error-bar">{error}<button onClick={() => setError(null)}>&times;</button></div>}
      <main className="main">
        {tab === "Equipment" && <EquipmentPage key={refreshKey} setError={setError} markDirty={markDirty} />}
        {tab === "Plant Config" && <PlantConfigPage key={refreshKey} setError={setError} markDirty={markDirty} />}
        {tab === "Results" && <ResultsPage results={results} setResults={setResults} setError={setError} onAddToComparison={addToComparison} markDirty={markDirty} />}
        {tab === "Analysis" && <AnalysisPage setError={setError} comparedPlants={comparedPlants} />}
        {tab === "Monte Carlo" && <MonteCarloPage setError={setError} comparedPlants={comparedPlants} />}
        {tab === "Compare" && (
          <ComparePage
            plants={comparedPlants}
            onRemove={removeFromComparison}
            onImport={(plant) => setComparedPlants((prev) => [...prev, plant])}
            setError={setError}
          />
        )}
      </main>
    </div>
  );
}

export default App;
