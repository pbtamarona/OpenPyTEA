import { useState, useRef, useEffect, useCallback } from "react";
import EquipmentPage from "./pages/EquipmentPage";
import PlantConfigPage from "./pages/PlantConfigPage";
import ResultsPage from "./pages/ResultsPage";
import AnalysisPage from "./pages/AnalysisPage";
import MonteCarloPage from "./pages/MonteCarloPage";
import {
  saveProject, loadProject, loadProjectFromText,
  newProject, getExamples, loadExample, runCalculations,
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
  const [examplesOpen, setExamplesOpen] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [dark, setDark] = useState(() => {
    const saved = localStorage.getItem("openpytea-theme");
    if (saved) return saved === "dark";
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  });
  const [comparedPlants, setComparedPlants] = useState<ComparedPlant[]>([]);
  const [showWelcome, setShowWelcome] = useState(true);
  const [inTauri, setInTauri] = useState(false);

  // Project file state.
  //   dirty           = user edited inputs (equipment / plant config)
  //                     since the last save / load. Used for the
  //                     "Load an example anyway?" switch-prompt.
  //   comparisonDirty = user added/removed compared plants since the
  //                     last save / load. NOT used for switch-prompts
  //                     (would block the typical multi-example
  //                     comparison flow) but IS checked on app close.
  const [currentPath, setCurrentPath] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [comparisonDirty, setComparisonDirty] = useState(false);
  const markDirty = useCallback(() => setDirty(true), []);

  // Close-confirmation modal state (Tauri-only: triggered when the
  // user tries to quit while there's unsaved work).
  const [closeConfirmOpen, setCloseConfirmOpen] = useState(false);

  // Loading banner: non-null message shows a fixed top progress bar.
  const [loadingMsg, setLoadingMsg] = useState<string | null>(null);

  // Refs for state values that listeners (close-requested, menu, etc.)
  // need to read freshly. Reassigning a ref in render is safe — these
  // are not part of React's render output.
  const dirtyRef = useRef(dirty);
  const comparisonDirtyRef = useRef(comparisonDirty);
  const comparedPlantsRef = useRef(comparedPlants);
  const currentPathRef = useRef(currentPath);
  dirtyRef.current = dirty;
  comparisonDirtyRef.current = comparisonDirty;
  comparedPlantsRef.current = comparedPlants;
  currentPathRef.current = currentPath;

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
    setComparisonDirty(true);
    // We deliberately don't set the input-`dirty` flag — that one drives
    // the load-example/open prompts and would otherwise block the typical
    // "load A, add, load B, add, …" flow. comparisonDirty is only checked
    // on app close.
  };

  const removeFromComparison = (id: string) => {
    setComparedPlants((prev) => prev.filter((p) => p.id !== id));
    setComparisonDirty(true);
  };

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
    localStorage.setItem("openpytea-theme", dark ? "dark" : "light");
  }, [dark]);

  useEffect(() => {
    detectTauri().then(setInTauri);
  }, []);

  // Backend-ready poller for Tauri builds: shows a "Starting backend…" banner
  // while uvicorn boots inside the .app (first launch is slow because of
  // matplotlib's font cache), clears it once the port marker arrives.
  useEffect(() => {
    let canceled = false;
    let delayTimer: ReturnType<typeof setTimeout> | undefined;
    (async () => {
      if (!(await detectTauri())) return;
      // Only show the banner if startup actually takes a noticeable moment,
      // so instant-ready cases don't flash a useless banner.
      delayTimer = setTimeout(() => {
        if (!canceled) setLoadingMsg("Starting backend… (first launch may take ~30–45 s)");
      }, 800);
      const { invoke } = await import("@tauri-apps/api/core");
      for (let i = 0; i < 700; i++) {
        if (canceled) return;
        const url = await invoke<string | null>("get_api_base");
        if (url) {
          if (delayTimer) clearTimeout(delayTimer);
          setLoadingMsg(null);
          return;
        }
        await new Promise((r) => setTimeout(r, 200));
      }
      if (!canceled) setLoadingMsg("Backend did not start. Try restarting the app.");
    })();
    return () => {
      canceled = true;
      if (delayTimer) clearTimeout(delayTimer);
    };
  }, []);

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

  // window.confirm is suppressed in Tauri's WebKit webview, so use the
  // native ask dialog when running inside Tauri and fall back to
  // window.confirm in browser-dev.
  const confirmIfDirty = async (verb: string): Promise<boolean> => {
    if (!dirty) return true;
    const message = `You have unsaved changes. ${verb} anyway?`;
    if (await detectTauri()) {
      const { ask } = await import("@tauri-apps/plugin-dialog");
      return await ask(message, {
        title: "Unsaved Changes",
        okLabel: "Continue",
        cancelLabel: "Cancel",
      });
    }
    return window.confirm(message);
  };

  const handleNew = useCallback(async () => {
    if (!(await confirmIfDirty("Start a new project"))) return;
    try {
      setLoadingMsg("Creating new project…");
      await newProject();
      setCurrentPath(null);
      setComparedPlants([]);
      setComparisonDirty(false);
      setDirty(false);
      setResults(null);
      setError(null);
      setRefreshKey((k) => k + 1);
      setTab("Equipment");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "New project failed");
    } finally {
      setLoadingMsg(null);
    }
  }, [dirty]);

  /** Restore compared plants from a parsed save-file. UUIDs are
      regenerated; everything else is taken verbatim. */
  const restoreComparedPlants = (parsed: unknown): ComparedPlant[] => {
    const arr = (parsed as { compared_plants?: unknown }).compared_plants;
    if (!Array.isArray(arr)) return [];
    return arr.map((p: unknown) => {
      const obj = p as {
        name?: string; currency?: string;
        results?: CalculationResults; source?: PlantInput;
      };
      return {
        id: crypto.randomUUID(),
        name: obj.name ?? "Plant",
        currency: obj.currency ?? "$",
        results: obj.results as CalculationResults,
        source: obj.source,
      };
    }).filter((p) => p.results != null);
  };

  /** Load a project file at a known path (Tauri only). Shared between
      File ▸ Open and the OS file-association open-file event. */
  const loadFromPath = useCallback(async (path: string) => {
    try {
      setLoadingMsg("Reading project file…");
      const { readTextFile } = await import("@tauri-apps/plugin-fs");
      const text = await readTextFile(path);
      const parsed = JSON.parse(text);

      setLoadingMsg("Loading project…");
      await loadProjectFromText(text);

      setComparedPlants(restoreComparedPlants(parsed));
      setComparisonDirty(false);
      setCurrentPath(path);
      setDirty(false);
      setError(null);
      setRefreshKey((k) => k + 1);
      setTab("Equipment");
      // If the user opens a file from Finder while the welcome screen is up,
      // skip straight to the main app.
      setShowWelcome(false);

      // Auto-recompute so the Results tab is populated and analysis pages
      // (which need the backend's Plant object) work without a manual
      // Calculate click.
      setLoadingMsg("Calculating…");
      try {
        const r = await runCalculations();
        setResults(r);
      } catch {
        setResults(null);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Open failed");
    } finally {
      setLoadingMsg(null);
    }
  }, []);

  const handleOpen = useCallback(async () => {
    if (!(await confirmIfDirty("Open another project"))) return;
    try {
      if (await detectTauri()) {
        const { open } = await import("@tauri-apps/plugin-dialog");
        const selected = await open({
          multiple: false,
          filters: [{ name: "OpenPyTEA Project", extensions: [PROJECT_EXT, "json"] }],
        });
        if (!selected || typeof selected !== "string") return; // user cancelled
        await loadFromPath(selected);
      } else {
        // Browser fallback: hidden <input type=file>
        fileRef.current?.click();
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Open failed");
    }
  }, [dirty, loadFromPath]);

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
    const backendData = await saveProject() as Record<string, unknown>;
    // Read compared plants via ref so we always serialize the latest list,
    // even when this function is called from a memoized handler whose
    // closure captured an older `comparedPlants` value.
    const cps = comparedPlantsRef.current;
    const full = {
      ...backendData,
      compared_plants: cps.map((p) => ({
        name: p.name,
        currency: p.currency,
        results: p.results,
        source: p.source,
      })),
    };
    const text = JSON.stringify(full, null, 2);
    const { writeTextFile } = await import("@tauri-apps/plugin-fs");
    await writeTextFile(path, text);
  };

  const handleSaveAs = useCallback(async (): Promise<boolean> => {
    try {
      if (await detectTauri()) {
        const { save } = await import("@tauri-apps/plugin-dialog");
        const path = await save({
          defaultPath: currentPath ?? `untitled.${PROJECT_EXT}`,
          filters: [{ name: "OpenPyTEA Project", extensions: [PROJECT_EXT] }],
        });
        if (!path) return false; // user cancelled
        await writeProjectTo(path);
        setCurrentPath(path);
        setDirty(false);
        setComparisonDirty(false);
        return true;
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
        setComparisonDirty(false);
        return true;
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Save failed");
      return false;
    }
  }, [currentPath]);

  const handleSave = useCallback(async (): Promise<boolean> => {
    // Without a known path (or outside Tauri), Save behaves as Save As.
    if (!currentPath || !(await detectTauri())) {
      return handleSaveAs();
    }
    try {
      await writeProjectTo(currentPath);
      setDirty(false);
      setComparisonDirty(false);
      return true;
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Save failed");
      return false;
    }
  }, [currentPath, handleSaveAs]);

  const handleLoadExample = async (id: string) => {
    setExamplesOpen(false);
    if (!(await confirmIfDirty("Load an example"))) return;
    try {
      setLoadingMsg("Loading example…");
      await loadExample(id);
      // Intentionally NOT clearing comparedPlants — the typical multi-plant
      // comparison workflow is: load A → calculate → Add to comparison →
      // load B → calculate → Add to comparison → overlay both on the
      // analysis tabs. Clearing would break that.
      setCurrentPath(null);
      setDirty(false);
      // Comparison list is unchanged by example loads; leave comparisonDirty
      // alone so building a comparison still counts as unsaved work.
      setError(null);
      setRefreshKey((k) => k + 1);
      setTab("Equipment");

      setLoadingMsg("Calculating…");
      try {
        const r = await runCalculations();
        setResults(r);
      } catch {
        setResults(null);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load example");
    } finally {
      setLoadingMsg(null);
    }
  };

  // File-action dispatch — used by both the native menu (Tauri) and the
  // browser-dev keydown listener. Ref-stored so listeners always see the
  // freshest closure without re-attaching on every render.
  const handlersRef = useRef({ handleNew, handleOpen, handleSave, handleSaveAs });
  handlersRef.current = { handleNew, handleOpen, handleSave, handleSaveAs };

  // Native menu events (Tauri): the Rust shell emits "menu" with the item
  // id (e.g. "menu:new") whenever the user clicks File ▸ New or hits ⌘N.
  useEffect(() => {
    if (showWelcome) return;
    let unlisten: (() => void) | undefined;
    (async () => {
      if (!(await detectTauri())) return;
      const { listen } = await import("@tauri-apps/api/event");
      unlisten = await listen<string>("menu", ({ payload }) => {
        const h = handlersRef.current;
        if (payload === "menu:new") h.handleNew();
        else if (payload === "menu:open") h.handleOpen();
        else if (payload === "menu:save") h.handleSave();
        else if (payload === "menu:save-as") h.handleSaveAs();
      });
    })();
    return () => { unlisten?.(); };
  }, [showWelcome]);

  // OS file-open events (Tauri only): fires when the user double-clicks a
  // .openpytea file in Finder or chooses Open With → OpenPyTEA. The Rust
  // shell forwards the file path via the `open-file` event.
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    (async () => {
      if (!(await detectTauri())) return;
      const { listen } = await import("@tauri-apps/api/event");
      unlisten = await listen<string>("open-file", ({ payload }) => {
        if (typeof payload === "string" && payload) {
          loadFromPath(payload);
        }
      });
    })();
    return () => { unlisten?.(); };
  }, [loadFromPath]);

  // Drain any file paths the OS asked us to open before the frontend was
  // ready (cold-start file double-click case: macOS fires RunEvent::Opened
  // during app launch, well before this useEffect attached the open-file
  // listener below, so the live emit is lost).
  useEffect(() => {
    (async () => {
      if (!(await detectTauri())) return;
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        const paths = await invoke<string[]>("take_pending_open_files");
        for (const p of paths) {
          await loadFromPath(p);
        }
      } catch (e) {
        console.warn("take_pending_open_files failed", e);
      }
    })();
  }, [loadFromPath]);

  // Quit requests come from Rust (both red-close-button and Cmd+Q go
  // through the same `request-close` event now). We pop the modal here if
  // there's unsaved work; otherwise we let the quit go through immediately.
  const requestForceQuit = async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("force_quit");
  };

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    (async () => {
      if (!(await detectTauri())) return;
      const { listen } = await import("@tauri-apps/api/event");
      unlisten = await listen("request-close", () => {
        const d = dirtyRef.current, cd = comparisonDirtyRef.current;
        console.log("[close] request-close received; dirty=", d, "comparisonDirty=", cd);
        if (d || cd) {
          setCloseConfirmOpen(true);
        } else {
          // No unsaved work — just quit.
          requestForceQuit();
        }
      });
      console.log("[close] request-close listener attached");
    })();
    return () => { unlisten?.(); };
  }, []);

  const handleCloseSave = async () => {
    const ok = await handleSave();
    if (!ok) {
      // Save failed or user cancelled the file dialog: keep the window open.
      setCloseConfirmOpen(false);
      return;
    }
    setCloseConfirmOpen(false);
    await requestForceQuit();
  };

  const handleCloseDiscard = async () => {
    setCloseConfirmOpen(false);
    await requestForceQuit();
  };

  const handleCloseCancel = () => setCloseConfirmOpen(false);

  // Browser-dev keyboard shortcuts. Skipped in Tauri, where the native
  // menu's accelerators (CmdOrCtrl+N/O/S/⇧S) handle the same keys via
  // the OS menu system and we don't want to double-fire.
  useEffect(() => {
    if (showWelcome) return;
    let canceled = false;
    let attached = false;
    (async () => {
      if (await detectTauri()) return; // native menu handles shortcuts
      if (canceled) return;
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
      attached = true;
      // store on a closure-captured slot for cleanup
      (handlersRef.current as unknown as { _keyHandler?: typeof onKey })._keyHandler = onKey;
    })();
    return () => {
      canceled = true;
      if (attached) {
        const onKey = (handlersRef.current as unknown as { _keyHandler?: (e: KeyboardEvent) => void })._keyHandler;
        if (onKey) window.removeEventListener("keydown", onKey);
      }
    };
  }, [showWelcome]);

  const loadingBanner = loadingMsg ? (
    <div className="loading-banner" role="status" aria-live="polite">
      <span className="loading-banner-msg">{loadingMsg}</span>
      <div className="loading-banner-track">
        <div className="loading-banner-bar" />
      </div>
    </div>
  ) : null;

  const closeConfirmModal = closeConfirmOpen ? (
    <div className="modal-overlay">
      <div className="modal" style={{ maxWidth: 440 }}>
        <h2>Unsaved Changes</h2>
        <p style={{ marginTop: 0, color: "var(--text-secondary)", fontSize: 14, lineHeight: 1.5 }}>
          You have unsaved work
          {dirty && comparisonDirty
            ? " (edited inputs and an unsaved comparison)"
            : dirty
              ? " (edited inputs)"
              : " (unsaved comparison)"}
          . Do you want to save before quitting?
        </p>
        <div className="modal-actions">
          <button className="btn-secondary" onClick={handleCloseCancel}>Cancel</button>
          <button className="btn-secondary" onClick={handleCloseDiscard}>Don't Save</button>
          <button className="btn-primary" onClick={handleCloseSave}>Save</button>
        </div>
      </div>
    </div>
  ) : null;

  if (showWelcome) {
    return (
      <>
        {loadingBanner}
        {closeConfirmModal}
        <WelcomePage onContinue={() => setShowWelcome(false)} />
      </>
    );
  }

  const cmdKey = navigator.platform.includes("Mac") ? "⌘" : "Ctrl";

  return (
    <div className="app">
      {loadingBanner}
      {closeConfirmModal}
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
            <button className="btn-examples" onClick={() => setExamplesOpen(!examplesOpen)}>
              Examples
            </button>
            {examplesOpen && (
              <>
                <div className="dropdown-backdrop" onClick={() => setExamplesOpen(false)} />
                <div className="dropdown-menu">
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
          {/* In Tauri the system menu bar handles File ▸ New / Open / Save /
              Save As. In browser-dev mode (start.sh) there's no system menu,
              so we show inline buttons + keyboard shortcuts. */}
          {!inTauri && (
            <>
              <button className="btn-secondary" onClick={handleNew} title={`New project (${cmdKey}N)`}>New</button>
              <button className="btn-secondary" onClick={handleOpen} title={`Open (${cmdKey}O)`}>Open</button>
              <button className="btn-secondary" onClick={handleSave} title={`Save (${cmdKey}S)`}>Save</button>
              <input ref={fileRef} type="file" accept=".openpytea,.json" hidden onChange={handleBrowserLoad} />
            </>
          )}
          <button className="btn-theme" onClick={() => setDark((d) => !d)} title={dark ? "Switch to light mode" : "Switch to dark mode"}>
            {dark ? "☀️" : "🌙"}
          </button>
        </div>
      </header>
      {error && <div className="error-bar">{error}<button onClick={() => setError(null)}>&times;</button></div>}
      <main className="main">
        {tab === "Equipment" && <EquipmentPage key={refreshKey} setError={setError} markDirty={markDirty} />}
        {tab === "Plant Config" && <PlantConfigPage key={refreshKey} setError={setError} markDirty={markDirty} />}
        {tab === "Results" && <ResultsPage results={results} setResults={setResults} setError={setError} onAddToComparison={addToComparison} />}
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
