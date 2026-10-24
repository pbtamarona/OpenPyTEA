import { exportJsonResults, importAnalysisConfig } from "../api/client";
import type { AnalysisImport } from "../types";
import { openJsonFile, saveResultsFile } from "../jsonFiles";

interface Props {
  setError: (e: string | null) => void;
  setNotice: (m: string | null) => void;
  /** Apply the file's settings to the page; returns the notice to show */
  onImported: (settings: AnalysisImport, fileName: string) => string;
}

/** Import an analysis input file (load_analysis_config format) into the
    page's settings, or save <plant>_analysis_results.json as run_tea does.
    Shared by the Analysis and Monte Carlo tabs. */
export default function AnalysisFileBar({ setError, setNotice, onImported }: Props) {
  const importFile = async () => {
    setError(null);
    try {
      const file = await openJsonFile("Import analysis input JSON");
      if (!file) return;
      setNotice(onImported(await importAnalysisConfig(file.data), file.name));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Import failed");
    }
  };

  const saveResults = async () => {
    setError(null);
    try {
      const where = await saveResultsFile(() => exportJsonResults("analysis"));
      if (where) setNotice(`Saved ${where}`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Save failed");
    }
  };

  return (
    <div className="card plant-picker">
      <label>Analysis file</label>
      <button className="btn-secondary" onClick={importFile} title="Read the settings of an analysis input file (the format openpytea.io.load_analysis_config reads); nothing is run">
        Import JSON…
      </button>
      <button className="btn-secondary" onClick={saveResults} title="Write <plant>_analysis_results.json, as run_tea does, with the analyses run for the open plant">
        Save results JSON…
      </button>
      <span className="plant-picker-hint">settings from run_tea's analysis input file · results as run_tea writes them</span>
    </div>
  );
}
