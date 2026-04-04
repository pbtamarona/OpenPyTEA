import { useState, useRef } from "react";
import EquipmentPage from "./pages/EquipmentPage";
import PlantConfigPage from "./pages/PlantConfigPage";
import ResultsPage from "./pages/ResultsPage";
import AnalysisPage from "./pages/AnalysisPage";
import MonteCarloPage from "./pages/MonteCarloPage";
import { saveProject, loadProject } from "./api/client";
import type { CalculationResults } from "./types";
import "./App.css";

const TABS = ["Equipment", "Plant Config", "Results", "Analysis", "Monte Carlo"] as const;

function App() {
  const [tab, setTab] = useState<(typeof TABS)[number]>("Equipment");
  const [results, setResults] = useState<CalculationResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

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
      setTab("Equipment");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Load failed");
    }
    e.target.value = "";
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
          <button className="btn-secondary" onClick={handleSave}>Save</button>
          <button className="btn-secondary" onClick={() => fileRef.current?.click()}>Load</button>
          <input ref={fileRef} type="file" accept=".json" hidden onChange={handleLoad} />
        </div>
      </header>
      {error && <div className="error-bar">{error}<button onClick={() => setError(null)}>&times;</button></div>}
      <main className="main">
        {tab === "Equipment" && <EquipmentPage />}
        {tab === "Plant Config" && <PlantConfigPage />}
        {tab === "Results" && <ResultsPage results={results} setResults={setResults} setError={setError} />}
        {tab === "Analysis" && <AnalysisPage setError={setError} />}
        {tab === "Monte Carlo" && <MonteCarloPage setError={setError} />}
      </main>
    </div>
  );
}

export default App;
