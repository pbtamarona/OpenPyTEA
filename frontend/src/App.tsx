import { useState, useRef, useEffect } from "react";
import EquipmentPage from "./pages/EquipmentPage";
import PlantConfigPage from "./pages/PlantConfigPage";
import ResultsPage from "./pages/ResultsPage";
import AnalysisPage from "./pages/AnalysisPage";
import MonteCarloPage from "./pages/MonteCarloPage";
import { saveProject, loadProject, getExamples, loadExample } from "./api/client";
import type { ExamplePreset } from "./api/client";
import type { CalculationResults } from "./types";
import "./App.css";

const TABS = ["Equipment", "Plant Config", "Results", "Analysis", "Monte Carlo"] as const;

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
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
    localStorage.setItem("openpytea-theme", dark ? "dark" : "light");
  }, [dark]);

  useEffect(() => {
    getExamples().then(setExamples).catch((e: unknown) => {
      setError(e instanceof Error ? e.message : "Failed to load examples");
    });
  }, []);

  const handleSave = async () => {
    try {
      const data = await saveProject();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "openpytea_project.json";
      a.click();
      URL.revokeObjectURL(url);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Save failed");
    }
  };

  const handleLoad = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      await loadProject(file);
      setResults(null);
      setError(null);
      setRefreshKey((k) => k + 1);
      setTab("Equipment");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Load failed");
    }
    e.target.value = "";
  };

  const handleLoadExample = async (id: string) => {
    setExamplesOpen(false);
    try {
      await loadExample(id);
      setResults(null);
      setError(null);
      setRefreshKey((k) => k + 1);
      setTab("Equipment");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load example");
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>OpenPyTEA</h1>
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
          <button className="btn-secondary" onClick={handleSave}>Save</button>
          <button className="btn-secondary" onClick={() => fileRef.current?.click()}>Load</button>
          <input ref={fileRef} type="file" accept=".json" hidden onChange={handleLoad} />
          <button className="btn-theme" onClick={() => setDark((d) => !d)} title={dark ? "Switch to light mode" : "Switch to dark mode"}>
            {dark ? "\u2600\uFE0F" : "\uD83C\uDF19"}
          </button>
        </div>
      </header>
      {error && <div className="error-bar">{error}<button onClick={() => setError(null)}>&times;</button></div>}
      <main className="main">
        {tab === "Equipment" && <EquipmentPage key={refreshKey} setError={setError} />}
        {tab === "Plant Config" && <PlantConfigPage key={refreshKey} setError={setError} />}
        {tab === "Results" && <ResultsPage results={results} setResults={setResults} setError={setError} />}
        {tab === "Analysis" && <AnalysisPage setError={setError} />}
        {tab === "Monte Carlo" && <MonteCarloPage setError={setError} />}
      </main>
    </div>
  );
}

export default App;
