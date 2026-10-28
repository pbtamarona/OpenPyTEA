import { useEffect, useState } from "react";
import {
  getSensitivityParameters, runSensitivity, runTornado, fetchPlotPng,
} from "../api/client";
import type { AnalysisImport, SensitivityResult, TornadoCategory, TornadoResult, ProjectOverview } from "../types";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  BarChart, Bar, ReferenceLine, Cell, Legend,
} from "recharts";
import DownloadableChart from "../components/DownloadableChart";
import { fmtTick, fmtValue, nameLabel, niceTicks, texToText } from "../format";
import HelpTip from "../components/HelpTip";
import ComparePlants from "../components/ComparePlants";
import AnalysisFileBar from "../components/AnalysisFileBar";
import { usePlantSelection, type PlantsRequest } from "../plantSources";

const METRICS = ["LCOP", "NPV", "IRR", "ROI", "PBT"];
const COLORS = ["#4361ee", "#e63946", "#06d6a0", "#f77f00", "#7209b7", "#4cc9f0", "#d62828", "#2a9d8f", "#e9c46a", "#264653"];

/** Dropdown text of a sensitivity key, named as the library's plots name it */
const formatParam = (p: string) => {
  const [first, item, field] = p.split(".");
  return item === undefined ? nameLabel(first) : `${nameLabel(item)} ${field ?? "price"}`;
};

type XAxisMode = "percent" | "value";

/** Inputs a sensitivity result was computed with */
interface SensRun {
  parameter: string;
  plus_minus_value: number;
  n_points: number;
  metric: string;
}

interface Panel {
  id: string;
  parameter: string;
  metric: string;
  plus_minus_value: number;
  result: SensitivityResult | null;
  /** Plant id of each result curve */
  ids: string[];
  ran: SensRun | null;
  loading: boolean;
  /** x axis: % change, or the parameter's actual value (display only) */
  x_axis: XAxisMode;
}

const NoneTicked = () => (
  <p className="compare-plants-empty">No plants ticked. Tick a plant above to show its results.</p>
);

interface Props {
  setError: (e: string | null) => void;
  setNotice: (m: string | null) => void;
  project: ProjectOverview | null;
  /** The page stays mounted (results kept) while hidden on other tabs;
      false while hidden. */
  active: boolean;
}

const newPanel = (parameter = "", metric = "LCOP", plus_minus_value = 0.2, x_axis: XAxisMode = "percent"): Panel => ({
  id: crypto.randomUUID(),
  parameter,
  metric,
  plus_minus_value,
  result: null,
  ids: [],
  ran: null,
  loading: false,
  x_axis,
});

export default function AnalysisPage({ setError, setNotice, project, active }: Props) {
  const [parameters, setParameters] = useState<string[]>([]);
  const [sensPoints, setSensPoints] = useState(21);
  const [panels, setPanels] = useState<Panel[]>([newPanel()]);
  const sensPlants = usePlantSelection(project);

  const [tornPM, setTornPM] = useState(0.2);
  const [tornMetric, setTornMetric] = useState("LCOP");
  const [tornCategory, setTornCategory] = useState<TornadoCategory>("economic");
  const [tornResult, setTornResult] = useState<TornadoResult | null>(null);
  // Plant id of each tornado result, and the inputs of the run
  const [tornIds, setTornIds] = useState<string[]>([]);
  const [tornRan, setTornRan] = useState<
    { plus_minus_value: number; metric: string; category: TornadoCategory } | null
  >(null);
  const [tornLoading, setTornLoading] = useState(false);
  const tornSelection = usePlantSelection(project);

  // Refetched every time the tab is shown: the configuration may have
  // gained or lost items while the user was on another tab
  useEffect(() => {
    if (!active) return;
    getSensitivityParameters().then((p) => {
      setParameters(p);
      setPanels((prev) =>
        prev.map((panel) => (panel.parameter === "" && p.length > 0 ? { ...panel, parameter: p[0] } : panel)),
      );
    }).catch((e: unknown) => {
      setError(e instanceof Error ? e.message : "Failed to load parameters");
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  const updatePanel = (id: string, patch: Partial<Panel>) =>
    setPanels((prev) => prev.map((p) => (p.id === id ? { ...p, ...patch } : p)));

  const removePanel = (id: string) =>
    setPanels((prev) => (prev.length > 1 ? prev.filter((p) => p.id !== id) : prev));

  const addPanel = () => {
    const last = panels[panels.length - 1];
    setPanels((prev) => [
      ...prev,
      newPanel(last?.parameter || parameters[0] || "", last?.metric || "LCOP", last?.plus_minus_value ?? 0.2, last?.x_axis),
    ]);
  };

  const runAll = async () => {
    setError(null);
    let plantsReq: PlantsRequest;
    try {
      plantsReq = await sensPlants.request();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Could not load the selected plants");
      return;
    }
    setPanels((prev) => prev.map((p) => ({ ...p, loading: true })));

    const tasks = panels.map(async (panel) => {
      const ran: SensRun = {
        parameter: panel.parameter,
        plus_minus_value: panel.plus_minus_value,
        n_points: sensPoints,
        metric: panel.metric,
      };
      try {
        const r = await runSensitivity({ ...ran, additional_capex: false, ...plantsReq.body });
        return { id: panel.id, result: r as SensitivityResult | null, ran, error: null as string | null };
      } catch (e: unknown) {
        return { id: panel.id, result: null, ran, error: e instanceof Error ? e.message : "Sensitivity failed" };
      }
    });

    const outcomes = await Promise.all(tasks);
    const errors = outcomes.filter((o) => o.error).map((o) => o.error!);
    if (errors.length) setError(errors.join(" • "));

    setPanels((prev) =>
      prev.map((p) => {
        const out = outcomes.find((o) => o.id === p.id);
        if (!out) return { ...p, loading: false };
        return out.result
          ? { ...p, loading: false, result: out.result, ids: plantsReq.ids, ran: out.ran }
          : { ...p, loading: false };
      }),
    );
  };

  /** A plant ticked after a run: rerun each panel's plants plus this one
      with the inputs of that run (a plant alone may not have the varied
      parameter, which the library rejects). */
  const addSensPlant = async (plantId: string) => {
    const targets = panels.filter((p) => p.result && p.ran && !p.ids.includes(plantId));
    if (targets.length === 0) return;
    const ids = new Set(targets.map((t) => t.id));
    setPanels((prev) => prev.map((p) => (ids.has(p.id) ? { ...p, loading: true } : p)));
    const outcomes = await Promise.all(targets.map(async (panel) => {
      try {
        const req = await sensPlants.requestFor([...panel.ids, plantId]);
        const r = await runSensitivity({ ...panel.ran!, additional_capex: false, ...req.body });
        return { id: panel.id, result: r as SensitivityResult | null, ids: req.ids, error: null as string | null };
      } catch (e: unknown) {
        return { id: panel.id, result: null, ids: panel.ids, error: e instanceof Error ? e.message : "Sensitivity failed" };
      }
    }));
    const errors = outcomes.filter((o) => o.error).map((o) => o.error!);
    if (errors.length) setError(errors.join(" • "));
    setPanels((prev) =>
      prev.map((p) => {
        const out = outcomes.find((o) => o.id === p.id);
        if (!out) return p;
        return out.result ? { ...p, loading: false, result: out.result, ids: out.ids } : { ...p, loading: false };
      }),
    );
  };

  const toggleSensPlant = (plantId: string) => {
    const adding = !sensPlants.isTicked(plantId);
    sensPlants.toggle(plantId);
    if (adding) addSensPlant(plantId);
  };

  const doTornado = async () => {
    setTornLoading(true);
    setError(null);
    try {
      const ran = { plus_minus_value: tornPM, metric: tornMetric, category: tornCategory };
      const req = await tornSelection.request();
      const r = await runTornado({ ...ran, additional_capex: false, ...req.body });
      setTornResult(r);
      setTornIds(req.ids);
      setTornRan(ran);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Tornado failed");
    } finally {
      setTornLoading(false);
    }
  };

  /** A plant ticked after a run: compute its tornado alone and add it. */
  const addTornPlant = async (plantId: string) => {
    if (!tornResult || !tornRan || tornIds.includes(plantId)) return;
    setTornLoading(true);
    try {
      const req = await tornSelection.requestFor([plantId]);
      const r = await runTornado({ ...tornRan, additional_capex: false, ...req.body });
      setTornResult((prev) => (prev ? { ...prev, plants: [...prev.plants, ...r.plants] } : prev));
      setTornIds((prev) => [...prev, ...req.ids]);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Tornado failed");
    } finally {
      setTornLoading(false);
    }
  };

  const toggleTornPlant = (plantId: string) => {
    const adding = !tornSelection.isTicked(plantId);
    tornSelection.toggle(plantId);
    if (adding) addTornPlant(plantId);
  };

  /** An analysis input file's sensitivity cases become the panels and its
      tornado settings fill the tornado form; nothing is run. */
  const applyImport = (r: AnalysisImport, fileName: string): string => {
    const notes = [...r.notes];
    const parts: string[] = [];
    if (r.sensitivity) {
      // A parameter set by a dependency can't be varied (not offered here)
      const usable = r.sensitivity.filter((c) => {
        if (parameters.includes(c.parameter)) return true;
        notes.push(`Sensitivity '${c.name}' left out: ${formatParam(c.parameter)} is set by a dependency.`);
        return false;
      });
      if (usable.length > 0) {
        setPanels(usable.map((c) => newPanel(c.parameter, c.metric, c.plus_minus_value, c.x_axis)));
        setSensPoints(usable[0].n_points);
        if (usable.some((c) => c.n_points !== usable[0].n_points)) {
          notes.push(`Every panel uses ${usable[0].n_points} points (one setting here).`);
        }
        parts.push(`${usable.length} sensitivity case${usable.length === 1 ? "" : "s"}`);
      }
    }
    if (r.tornado) {
      setTornPM(r.tornado.plus_minus_value);
      setTornMetric(r.tornado.metric);
      setTornCategory(r.tornado.category);
      parts.push("the tornado settings");
    }
    if (r.monte_carlo) notes.push("Its Monte Carlo settings are imported on the Monte Carlo tab.");
    const head = parts.length > 0
      ? `Imported ${parts.join(" and ")} from ${fileName}; run them with the buttons below.`
      : `${fileName} has no sensitivity or tornado analysis to import.`;
    return [head, ...notes].join(" ");
  };

  const anyLoading = panels.some((p) => p.loading);
  const isGrid = panels.length > 1;

  // Tornado chart data — bars are deltas from base_value. With a single plant
  // the x-axis tick formatter adds the base back to display actual metric
  // values; with multiple plants the bases differ, so we fall back to a pure
  // Δ display.
  // Only the ticked plants of the run are shown
  const tornPlants = (tornResult?.plants ?? [])
    .map((p, i) => ({ ...p, pid: tornIds[i] }))
    .filter((p) => tornSelection.isTicked(p.pid));
  const isMultiTorn = tornPlants.length > 1;
  const tornColor = (pid: string, pi: number) => COLORS[tornSelection.colorIndex(pid, pi) % COLORS.length];
  const shownPM = tornRan?.plus_minus_value ?? tornPM;
  const multiPlant = (project?.plants.length ?? 0) > 1;
  const shownMetric = tornRan?.metric ?? tornMetric;

  const tornMaxDelta = tornPlants.reduce((acc, p) => {
    for (let i = 0; i < p.lows.length; i++) {
      acc = Math.max(acc, Math.abs(p.lows[i] - p.base_value), Math.abs(p.highs[i] - p.base_value));
    }
    return acc;
  }, 0);
  const tornMaxBase = tornPlants.reduce((acc, p) => Math.max(acc, Math.abs(p.base_value)), 0);
  const tornScale = tornMaxBase >= 1e6 || tornMaxDelta >= 1e6 ? 1e6 : 1;

  type TornRow = { label: string; range: number } & Record<string, number | undefined | string>;
  const tornChartData: TornRow[] = (() => {
    const rows = new Map<string, TornRow>();
    tornPlants.forEach((p, pi) => {
      for (let i = 0; i < p.labels.length; i++) {
        const label = texToText(p.labels[i]);
        let row = rows.get(label);
        if (!row) {
          row = { label, range: 0 };
          rows.set(label, row);
        }
        row[`p${pi}_low`] = (p.lows[i] - p.base_value) / tornScale;
        row[`p${pi}_high`] = (p.highs[i] - p.base_value) / tornScale;
        (row.range as number) += Math.abs(p.highs[i] - p.lows[i]);
      }
    });
    return Array.from(rows.values()).sort((a, b) => (b.range as number) - (a.range as number));
  })();

  // For single-plant we keep the old behaviour of showing absolute metric
  // values on the axis (delta + base). For multi-plant the bars stay as Δ.
  const tornBase = isMultiTorn ? 0 : (tornPlants[0]?.base_value ?? 0) / tornScale;

  const tornXLabel = tornResult
    ? (() => {
        const cleaned = texToText(tornResult.xlabel);
        const scaled = tornScale === 1e6 ? cleaned.replace("/ [", "/ [million ") : cleaned;
        return isMultiTorn ? `Δ ${scaled}` : scaled;
      })()
    : "";

  return (
    <div>
      <AnalysisFileBar setError={setError} setNotice={setNotice} onImported={applyImport} />

      {/* Sensitivity */}
      <div className="card">
        <h2 style={{ marginTop: 0 }}>Sensitivity Analysis</h2>

        <ComparePlants project={project} isTicked={sensPlants.isTicked} onToggle={toggleSensPlant} />

        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 12, marginBottom: 16 }}>
          <button className="btn-primary" onClick={runAll} disabled={anyLoading}>
            {anyLoading && <span className="spinner" />}Run Sensitivity{panels.length > 1 ? ` (${panels.length} panels)` : ""}
          </button>
          <button
            onClick={addPanel}
            title="Add another panel with its own parameter & metric"
            style={{
              padding: "8px 16px",
              border: "1px solid var(--accent)",
              background: "transparent",
              color: "var(--accent)",
              borderRadius: 6,
              cursor: "pointer",
              fontSize: 14,
              fontWeight: 500,
            }}
          >
            + Add panel
          </button>
          <label style={{ fontSize: 13, color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 6, marginLeft: "auto" }}>
            Points<HelpTip id="analysis.points" />
            <input
              type="number"
              value={sensPoints}
              onChange={(e) => setSensPoints(+e.target.value)}
              style={{ width: 70 }}
            />
          </label>
        </div>

        <div
          style={
            isGrid
              ? { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(420px, 1fr))", gap: 16 }
              : undefined
          }
        >
          {panels.map((panel, idx) => (
            <SensitivityPanel
              key={panel.id}
              panel={panel}
              parameters={parameters}
              showRemove={panels.length > 1}
              compact={isGrid}
              label={isGrid ? `(${String.fromCharCode(97 + idx)})` : undefined}
              isTicked={sensPlants.isTicked}
              colorIndex={sensPlants.colorIndex}
              showLegend={multiPlant}
              serverPlot={async () => {
                // The figure of the result on screen: its run, ticked plants only
                const req = await sensPlants.requestFor(panel.ids.filter(sensPlants.isTicked));
                return fetchPlotPng("/plots/sensitivity", {
                  ...panel.ran!, additional_capex: false, x_axis: panel.x_axis, ...req.body,
                });
              }}
              onChange={(patch) => updatePanel(panel.id, patch)}
              onRemove={() => removePanel(panel.id)}
            />
          ))}
        </div>
      </div>

      {/* Tornado */}
      <div className="card">
        <h2>Tornado Analysis</h2>

        <ComparePlants project={project} isTicked={tornSelection.isTicked} onToggle={toggleTornPlant} />

        <div className="form-grid" style={{ marginBottom: 16 }}>
          <div className="form-group">
            <label>+/- Variation<HelpTip id="analysis.tornado_plus_minus" /></label>
            <input type="number" step="0.05" value={tornPM} onChange={(e) => setTornPM(+e.target.value)} />
          </div>
          <div className="form-group">
            <label>Metric<HelpTip id="metric" /></label>
            <select value={tornMetric} onChange={(e) => setTornMetric(e.target.value)}>
              {METRICS.map((m) => <option key={m}>{m}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label>Parameters<HelpTip id="analysis.tornado_category" /></label>
            <select value={tornCategory} onChange={(e) => setTornCategory(e.target.value as TornadoCategory)}>
              <option value="economic">Economic</option>
              <option value="process">Process</option>
              <option value="both">Both</option>
            </select>
          </div>
        </div>
        <button className="btn-primary" onClick={doTornado} disabled={tornLoading}>
          {tornLoading && <span className="spinner" />}Run Tornado{isMultiTorn ? ` (${tornPlants.length} plants)` : ""}
        </button>

        {tornResult && tornPlants.length === 0 && <NoneTicked />}
        {tornResult && tornChartData.length > 0 && (
          <DownloadableChart
            filename="tornado"
            maxWidth={800}
            serverPlot={async () => {
              const req = await tornSelection.requestFor(tornPlants.map((p) => p.pid));
              return fetchPlotPng("/plots/tornado", { ...tornRan!, additional_capex: false, ...req.body });
            }}
            height={Math.max(
              300,
              tornChartData.length * (isMultiTorn ? Math.max(40, tornPlants.length * 18) : 30) + 120,
            )}
            style={{ marginTop: 20 }}
          >
            <ResponsiveContainer>
              <BarChart data={tornChartData} layout="vertical" margin={{ left: 160, bottom: 52 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  tickFormatter={(v: number) => fmtTick(v + tornBase)}
                  label={{ value: tornXLabel, position: "insideBottom", offset: -34, style: { fontWeight: "bold", fontSize: 14, fill: "#666" } }}
                />
                <YAxis type="category" dataKey="label" width={150} tick={{ fontSize: 12, fontWeight: "bold" }} />
                <Tooltip
                  formatter={(v) => {
                    const unit = tornXLabel.match(/\[(.+)\]/)?.[1] ?? "";
                    const actual = typeof v === "number" ? v + tornBase : v;
                    const val = typeof actual === "number" ? fmtValue(actual, shownMetric === "IRR" ? 3 : 2) : actual;
                    return (unit && tornMetric !== "IRR") ? `${val} ${unit}` : val;
                  }}
                  labelStyle={{ fontWeight: "bold", color: "var(--text)" }}
                />
                <Legend
                  verticalAlign="bottom"
                  align="right"
                  content={() => (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 16, justifyContent: "flex-end", fontSize: 12, color: "var(--text-secondary)" }}>
                      {isMultiTorn
                        ? tornPlants.map((p, pi) => (
                            <span key={p.pid} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                              <span style={{ width: 12, height: 12, background: tornColor(p.pid, pi), display: "inline-block" }} />
                              {p.name}
                            </span>
                          ))
                        : [
                            // Name the plant when the project has others
                            ...(multiPlant && tornPlants[0] ? [{ color: "", label: tornPlants[0].name }] : []),
                            { color: "#4361ee", label: `-${shownPM * 100}%` },
                            { color: "#e63946", label: `+${shownPM * 100}%` },
                          ].map(({ color, label }) => (
                            <span key={label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                              {color && <span style={{ width: 12, height: 12, background: color, display: "inline-block" }} />}
                              {color ? label : <strong>{label}</strong>}
                            </span>
                          ))}
                    </div>
                  )}
                />
                <ReferenceLine x={0} stroke="#333" strokeDasharray="3 3" />
                {/* Render all "low" bars first, then all "high" bars, so Recharts
                    groups them as [p0_low p1_low ... p0_high p1_high ...] at each
                    parameter row — each plant's low and high sit on opposite sides
                    of zero with the same color. */}
                {tornPlants.map((p, pi) => {
                  const color = isMultiTorn ? tornColor(p.pid, pi) : "#4361ee";
                  return (
                    <Bar key={`p${pi}_low`} dataKey={`p${pi}_low`} fill={color}
                         name={isMultiTorn ? `${p.name} −${shownPM * 100}%` : `−${shownPM * 100}%`}>
                      {tornChartData.map((_, i) => (
                        <Cell key={i} fill={color} fillOpacity={isMultiTorn ? 0.55 : 1} />
                      ))}
                    </Bar>
                  );
                })}
                {tornPlants.map((p, pi) => {
                  const color = isMultiTorn ? tornColor(p.pid, pi) : "#e63946";
                  return (
                    <Bar key={`p${pi}_high`} dataKey={`p${pi}_high`} fill={color}
                         name={isMultiTorn ? `${p.name} +${shownPM * 100}%` : `+${shownPM * 100}%`}>
                      {tornChartData.map((_, i) => <Cell key={i} fill={color} />)}
                    </Bar>
                  );
                })}
              </BarChart>
            </ResponsiveContainer>
          </DownloadableChart>
        )}
      </div>
    </div>
  );
}

interface PanelProps {
  panel: Panel;
  parameters: string[];
  showRemove: boolean;
  compact: boolean;
  label?: string;
  isTicked: (plantId: string) => boolean;
  colorIndex: (plantId: string, fallback: number) => number;
  /** Name the plants even when one is shown (the project has several) */
  showLegend: boolean;
  serverPlot: () => Promise<Blob>;
  onChange: (patch: Partial<Panel>) => void;
  onRemove: () => void;
}

function SensitivityPanel({ panel, parameters, showRemove, compact, label, isTicked, colorIndex, showLegend, serverPlot, onChange, onRemove }: PanelProps) {
  const result = panel.result;
  // Only the ticked plants of the run are shown
  const ticked = (result?.curves ?? [])
    .map((c, i) => ({ ...c, pid: panel.ids[i] }))
    .filter((c) => isTicked(c.pid));
  // A plant without the parameter has no values to put on the value axis
  const byValue = panel.x_axis === "value";
  const curves = byValue ? ticked.filter((c) => c.x_value) : ticked;
  const leftOut = ticked.filter((c) => !c.x_value).map((c) => c.plant);

  const allY = curves.flatMap((c) => c.y).filter((v): v is number => typeof v === "number");
  const maxAbs = allY.length ? Math.max(...allY.map(Math.abs)) : 0;
  const scale = maxAbs >= 1e6 ? 1e6 : 1;
  const allX = curves.flatMap((c) => c.x_value ?? []);
  const xScale = byValue && allX.length && Math.max(...allX.map(Math.abs)) >= 1e6 ? 1e6 : 1;

  const chartData = (() => {
    if (curves.length === 0) return [];
    if (byValue) {
      // Each plant varies around its own base value: one row per point,
      // sorted along the axis (connectNulls joins each plant's points)
      const rows: Record<string, number>[] = [];
      curves.forEach((c) => c.x_value!.forEach((xv, i) => {
        const y = c.y[i];
        if (typeof y === "number") rows.push({ x: xv / xScale, [c.pid]: y / scale });
      }));
      return rows.sort((a, b) => a.x - b.x);
    }
    return curves[0].x.map((x, i) => {
      const row: Record<string, number> = { x };
      curves.forEach((c) => {
        const y = c.y[i];
        if (typeof y === "number") row[c.pid] = y / scale;
      });
      return row;
    });
  })();
  // Room for the widest separated tick label (e.g. "-1,250.5")
  const yAxisWidth = (() => {
    let longest = 4;
    chartData.forEach((row) => Object.entries(row).forEach(([k, v]) => {
      if (k !== "x") longest = Math.max(longest, fmtTick(v).length);
    }));
    return Math.max(60, longest * 7 + 16);
  })();

  const inMillions = (label: string, s: number) => (s === 1e6 ? label.replace("/ [", "/ [million ") : label);
  const yLabel = result ? inMillions(texToText(result.ylabel), scale) : "";
  const xLabel = result ? inMillions(texToText(byValue ? result.xlabel_value : result.xlabel), xScale) : "";
  const xUnit = xLabel.match(/\[(.+)\]/)?.[1] ?? "";
  const chartHeight = compact ? 280 : 380;

  return (
    <div style={{ border: compact ? "1px solid var(--border)" : undefined, borderRadius: compact ? 6 : 0, padding: compact ? 12 : 0, marginBottom: compact ? 0 : 12 }}>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "end", gap: 12, marginBottom: 12 }}>
        {label && <span style={{ fontWeight: 600, color: "var(--text-secondary)" }}>{label}</span>}
        <div className="form-group" style={{ flex: "1 1 200px", marginBottom: 0 }}>
          <label style={{ fontSize: 12 }}>Parameter<HelpTip id="analysis.parameter" /></label>
          <select value={panel.parameter} onChange={(e) => onChange({ parameter: e.target.value })}>
            {parameters.map((p) => <option key={p} value={p}>{formatParam(p)}</option>)}
          </select>
        </div>
        <div className="form-group" style={{ width: 100, marginBottom: 0 }}>
          <label style={{ fontSize: 12 }}>Metric<HelpTip id="metric" /></label>
          <select value={panel.metric} onChange={(e) => onChange({ metric: e.target.value })}>
            {METRICS.map((m) => <option key={m}>{m}</option>)}
          </select>
        </div>
        <div className="form-group" style={{ width: 90, marginBottom: 0 }}>
          <label style={{ fontSize: 12 }}>+/- Var<HelpTip id="analysis.plus_minus" /></label>
          <input
            type="number"
            step="0.05"
            value={panel.plus_minus_value}
            onChange={(e) => onChange({ plus_minus_value: +e.target.value })}
          />
        </div>
        <div className="form-group" style={{ width: 110, marginBottom: 0 }}>
          <label style={{ fontSize: 12 }}>X axis<HelpTip id="analysis.x_axis" /></label>
          <select value={panel.x_axis} onChange={(e) => onChange({ x_axis: e.target.value as XAxisMode })}>
            <option value="percent">% change</option>
            <option value="value">Value</option>
          </select>
        </div>
        {showRemove && (
          <button className="btn-danger" onClick={onRemove} style={{ marginBottom: 0 }}>Remove</button>
        )}
      </div>

      {result && ticked.length === 0 && <NoneTicked />}
      {result && byValue && ticked.length > 0 && leftOut.length > 0 && (
        <p className="compare-plants-empty">
          {leftOut.join(", ")} {leftOut.length > 1 ? "have" : "has"} no {formatParam(panel.ran?.parameter ?? panel.parameter)}, so {leftOut.length > 1 ? "they are" : "it is"} left out on the value axis.
        </p>
      )}
      {result && chartData.length > 0 && (
        <DownloadableChart filename={`sensitivity_${panel.parameter}_${panel.metric}`} height={chartHeight} serverPlot={serverPlot} style={{ marginTop: 4 }}>
          <div style={{ position: "absolute", left: 0, top: 0, bottom: 50, width: 20, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <span style={{ transform: "rotate(-90deg)", whiteSpace: "nowrap", fontWeight: "bold", fontSize: compact ? 12 : 14, color: "#666" }}>{yLabel}</span>
          </div>
          <div style={{ position: "absolute", left: 24, right: 0, top: 0, bottom: 0 }}>
            <ResponsiveContainer>
              <LineChart data={chartData} margin={{ left: 10, bottom: 52, top: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="x"
                  tickFormatter={fmtTick}
                  {...(chartData.length ? {
                    type: "number" as const,
                    domain: ["dataMin", "dataMax"],
                    ticks: niceTicks(chartData[0].x, chartData[chartData.length - 1].x),
                  } : {})}
                  label={{ value: xLabel, position: "insideBottom", offset: -34, style: { fontWeight: "bold", fontSize: compact ? 13 : 14, fill: "#666" } }} />
                <YAxis width={yAxisWidth} tickFormatter={fmtTick} domain={["auto", "auto"]} />
                <Tooltip
                  labelFormatter={(v) => {
                    const x = typeof v === "number" ? v : parseFloat(String(v));
                    return byValue ? `x: ${fmtTick(x)}${xUnit ? ` ${xUnit}` : ""}` : `x: ${x.toFixed(2)} %`;
                  }}
                  formatter={(v, name) => {
                    const unit = yLabel.match(/\[(.+)\]/)?.[1] ?? "";
                    const val = typeof v === "number" ? fmtValue(v, panel.metric === "IRR" ? 3 : 2) : v;
                    return [`${val}${unit ? ` ${unit}` : ""}`, name];
                  }}
                  labelStyle={{ color: "var(--text)" }}
                />
                {(curves.length > 1 || showLegend) && <Legend verticalAlign="top" height={28} />}
                {curves.map((c, i) => (
                  <Line key={c.pid} type="monotone" dataKey={c.pid} name={c.plant} stroke={COLORS[colorIndex(c.pid, i) % COLORS.length]} strokeWidth={2} dot={false} connectNulls />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </DownloadableChart>
      )}
    </div>
  );
}
