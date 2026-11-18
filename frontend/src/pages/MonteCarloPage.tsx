import { useEffect, useRef, useState } from "react";
import { runMonteCarlo, fetchPlotPng, importAnalysisConfig, type ExampleAutoRun } from "../api/client";
import type { AnalysisImport, MonteCarloMultiResult, MonteCarloResult, ProjectOverview } from "../types";
import {
  ComposedChart, Area, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  ReferenceLine, ReferenceDot,
} from "recharts";
import DownloadableChart from "../components/DownloadableChart";
import { fmtTick, fmtValue, niceTicks, texToText } from "../format";
import { saveFigure } from "../save";
import HelpTip from "../components/HelpTip";
import ComparePlants from "../components/ComparePlants";
import AnalysisFileBar from "../components/AnalysisFileBar";
import { usePlantSelection } from "../plantSources";

/** Name and unit of an MC output metric — the same wording and units as
    the library's matplotlib figures (helpers._default_metric_label). */
function metricInfo(metric: string, cur: string): { name: string; unit: string | null } {
  switch (metric.toUpperCase()) {
    case "LCOP": return { name: "Levelized cost", unit: `${cur}/unit` };
    case "NPV": return { name: "Net present value", unit: cur };
    case "ROI": return { name: "Return on investment", unit: "%" };
    case "PBT": case "PAYBACK": case "PAYBACK_TIME": return { name: "Payback time", unit: "years" };
    case "SIMPLE_PBT": case "SIMPLE_PAYBACK": case "SIMPLE_PAYBACK_TIME":
      return { name: "Simple payback time", unit: "years" };
    case "IRR": return { name: "Internal rate of return", unit: "-" };
    default: return { name: metric, unit: null };
  }
}

/** Unit of a sampled MC input, from the library's input naming: prices
    are per unit of the item, consumption/production quantities per day
    (the library annualizes them × 365 × utilization), factors and rates
    are fractions. */
function inputUnit(name: string, cur: string): string | null {
  if (/ price$/i.test(name)) return `${cur}/unit`;
  if (/ (consumption|production)$/i.test(name)) return "unit/day";
  switch (name.toLowerCase()) {
    case "operator hourly rate": return `${cur}/h`;
    case "project lifetime": return "years";
    case "interest rate": case "fixed capital factor": case "fixed opex factor": return "-";
    default: return null;
  }
}

const withUnit = (label: string, unit: string | null) => (unit ? `${label} / [${unit}]` : label);

const COLORS = ["#4361ee", "#e63946", "#06d6a0", "#f77f00", "#7209b7", "#4cc9f0", "#d62828", "#2a9d8f", "#e9c46a", "#264653"];

interface Props {
  setError: (e: string | null) => void;
  setNotice: (m: string | null) => void;
  project: ProjectOverview | null;
  /** Monte Carlo settings of a just-loaded example, run once when the page
      mounts (it mounts while its plant is the open one) */
  autoRun?: ExampleAutoRun;
}

const SQRT_2PI = Math.sqrt(2 * Math.PI);
const normalPdf = (x: number, mu: number, sigma: number) => {
  if (sigma <= 0) return 0;
  const z = (x - mu) / sigma;
  return Math.exp(-(z * z) / 2) / (sigma * SQRT_2PI);
};

/** A plant of the run, with its project id and its position in the run */
type ShownPlant = MonteCarloResult & { pid: string; pos: number };

interface OverlayRow {
  x: number;
  [series: string]: number;
}

// Empirical density rows for one input's histogram — no fitted curve: the
// bins themselves show whatever distribution family was sampled (uniform,
// triangular, gamma, ...).
function inputHistRows(hist: { bin_edges: number[]; counts: number[] }, n: number) {
  const rows: { x: number; density: number }[] = [];
  for (let i = 0; i < hist.counts.length; i++) {
    const lo = hist.bin_edges[i];
    const hi = hist.bin_edges[i + 1];
    const w = hi - lo;
    rows.push({ x: (lo + hi) / 2, density: w > 0 ? hist.counts[i] / (n * w) : 0 });
  }
  return rows;
}

function buildOverlayData(plants: ShownPlant[], metric: string, showFit: boolean, pdfPoints = 200): OverlayRow[] {
  const series = plants
    .map((p) => ({ name: p.pid, num_samples: p.num_samples, stats: p.metrics[metric] }))
    .filter((s) => s.stats != null);
  if (series.length === 0) return [];

  let xMin = Infinity, xMax = -Infinity;
  series.forEach((s) => {
    const edges = s.stats.histogram.bin_edges;
    if (edges[0] < xMin) xMin = edges[0];
    if (edges[edges.length - 1] > xMax) xMax = edges[edges.length - 1];
    // Extend a bit beyond histogram so the fitted curve has tails.
    if (showFit) {
      xMin = Math.min(xMin, s.stats.mean - 4 * s.stats.std);
      xMax = Math.max(xMax, s.stats.mean + 4 * s.stats.std);
    }
  });

  // Collect x points: bin centers (per plant) + uniform PDF grid
  const xSet = new Set<number>();
  for (let i = 0; i <= pdfPoints; i++) xSet.add(xMin + ((xMax - xMin) * i) / pdfPoints);
  series.forEach((s) => {
    const edges = s.stats.histogram.bin_edges;
    for (let i = 0; i < s.stats.histogram.counts.length; i++) {
      xSet.add((edges[i] + edges[i + 1]) / 2);
    }
  });
  const xs = Array.from(xSet).sort((a, b) => a - b);

  return xs.map((x) => {
    const row: OverlayRow = { x };
    series.forEach((s) => {
      // Histogram density at x: the height of the bin x falls inside. Every
      // x within the histogram's range gets a value (not just bin centres),
      // otherwise the step Area has no adjacent points to join and the
      // bars never render. Outside the range stays undefined (no bars).
      const edges = s.stats.histogram.bin_edges;
      const counts = s.stats.histogram.counts;
      const last = counts.length - 1;
      for (let i = 0; i <= last; i++) {
        const lo = edges[i];
        const hi = edges[i + 1];
        if (x >= lo && (x < hi || (i === last && x <= hi))) {
          const w = hi - lo;
          if (w > 0) row[`${s.name}__hist`] = counts[i] / (s.num_samples * w);
          break;
        }
      }
      if (showFit) row[`${s.name}__pdf`] = normalPdf(x, s.stats.mean, s.stats.std);
    });
    return row;
  });
}

/** Value at percentile `p` (0-100) from the run's 1001-point quantile
    table (0.1% steps), interpolated between steps. */
function quantileAt(q: number[], p: number): number {
  const pos = (p / 100) * (q.length - 1);
  const i = Math.floor(pos);
  return i >= q.length - 1 ? q[q.length - 1] : q[i] + (pos - i) * (q[i + 1] - q[i]);
}

/** Power of ten (a multiple of 3) the percentile values are shown in, as
    in the library's legend: 0 while the largest is in [0.001, 9999]. */
function pctExponent(values: number[]): number {
  const largest = Math.max(0, ...values.map(Math.abs));
  if (largest === 0) return 0;
  const e = Math.floor(Math.log10(largest));
  return e >= -3 && e < 4 ? 0 : 3 * Math.floor(e / 3);
}

/** 3 significant figures, never in e-notation (the library's _fmt3) */
function fmt3(x: number): string {
  if (x === 0) return "0";
  return x.toFixed(Math.max(0, 2 - Math.floor(Math.log10(Math.abs(x)))));
}

/** Cumulative curves of several plants on one x axis: rows of x and each
    plant's cumulative probability at its own quantile points (the other
    plants' keys stay undefined, bridged with connectNulls). */
function buildCumulativeData(plants: ShownPlant[], metric: string): OverlayRow[] {
  const rows: OverlayRow[] = [];
  plants.forEach((p) => {
    const q = p.metrics[metric]?.quantiles;
    q?.forEach((x, i) => rows.push({ x, [`${p.pid}__cdf`]: i / (q.length - 1) }));
  });
  return rows.sort((a, b) => a.x - b.x);
}

type MCView = "histogram" | "cumulative";
const DEFAULT_PCTS = ["10", "50", "90"];
const validPct = (t: string) => t.trim() !== "" && Number.isFinite(Number(t)) && +t >= 0 && +t <= 100;

export default function MonteCarloPage({ setError, setNotice, project, autoRun }: Props) {
  const [numSamples, setNumSamples] = useState(50000);
  // Fitted-normal overlay on the metric histograms (screen and downloads);
  // off by default, as in the library; remembered across sessions
  const [showFit, setShowFit] = useState(() => {
    try { return localStorage.getItem("openpytea-mc-fit") === "on"; } catch { return false; }
  });
  const toggleFit = (on: boolean) => {
    setShowFit(on);
    try { localStorage.setItem("openpytea-mc-fit", on ? "on" : "off"); } catch { /* private mode */ }
  };
  // Percentile markers (the library's percentiles, P10/P50/P90 by default):
  // three editable levels; a blank or out-of-range box is left out
  const [showPct, setShowPct] = useState(() => {
    try { return localStorage.getItem("openpytea-mc-pct") !== "off"; } catch { return true; }
  });
  const [pctText, setPctText] = useState<string[]>(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("openpytea-mc-pct-levels") ?? "null");
      if (Array.isArray(saved) && saved.length === 3) return saved.map(String);
    } catch { /* missing or private mode */ }
    return DEFAULT_PCTS;
  });
  const togglePct = (on: boolean) => {
    setShowPct(on);
    try { localStorage.setItem("openpytea-mc-pct", on ? "on" : "off"); } catch { /* private mode */ }
  };
  const setPctLevel = (i: number, text: string) => {
    const next = pctText.map((t, j) => (j === i ? text : t));
    setPctText(next);
    try { localStorage.setItem("openpytea-mc-pct-levels", JSON.stringify(next)); } catch { /* private mode */ }
  };
  const percentiles = showPct ? pctText.filter(validPct).map(Number) : [];
  // Histogram or cumulative distribution, per metric card
  const [viewOf, setViewOf] = useState<Record<string, MCView>>({});
  const [batchSize, setBatchSize] = useState(1000);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<MonteCarloMultiResult | null>(null);
  // Plant id of each plant in the result
  const [resultIds, setResultIds] = useState<string[]>([]);
  const selection = usePlantSelection(project);
  const [runCount, setRunCount] = useState(1);

  /** Run the ticked plants. The arguments let the example auto-run pass the
      settings and plants it has just set, before React applies them. */
  const run = async (samples: number = numSamples, batch: number = batchSize, plantIds?: string[]) => {
    setLoading(true);
    setError(null);
    try {
      const req = plantIds ? await selection.requestFor(plantIds) : await selection.request();
      setRunCount(req.ids.length);
      const r = await runMonteCarlo({
        num_samples: samples, batch_size: batch, additional_capex: false,
        ...req.body,
      });
      setResult(r);
      setResultIds(req.ids);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Monte Carlo failed");
    } finally {
      setLoading(false);
    }
  };

  // An example's Monte Carlo settings fill the form and are run straight
  // away, once
  const autoRan = useRef(false);
  useEffect(() => {
    if (!autoRun || autoRan.current || autoRun.monte_carlo_plants.length === 0) return;
    autoRan.current = true;
    (async () => {
      try {
        const r = await importAnalysisConfig({ analysis: autoRun.analysis });
        if (!r.monte_carlo) return;
        setNumSamples(r.monte_carlo.num_samples);
        setBatchSize(r.monte_carlo.batch_size);
        selection.setTicked(autoRun.monte_carlo_plants);
        await run(r.monte_carlo.num_samples, r.monte_carlo.batch_size, autoRun.monte_carlo_plants);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Running the example's Monte Carlo failed");
      }
    })();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRun]);

  const fmt = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 2 });
  // Only the ticked plants of the run are shown
  const plants: ShownPlant[] = (result?.plants ?? [])
    .map((p, i) => ({ ...p, pid: resultIds[i], pos: i }))
    .filter((p) => selection.isTicked(p.pid));
  const colorOf = (p: ShownPlant) => COLORS[selection.colorIndex(p.pid, p.pos) % COLORS.length];
  // Ticked plants that weren't in the run (Monte Carlo is too slow to add
  // them on the fly)
  const missing = result
    ? selection.tickedIds().filter((id) => !resultIds.includes(id)).map(selection.nameOf).filter(Boolean)
    : [];

  // Collect all metric names that appeared on at least one plant
  const metricNames = Array.from(new Set(plants.flatMap((p) => Object.keys(p.metrics))));

  /** An analysis input file's Monte Carlo settings fill the form; nothing is run. */
  const applyImport = (r: AnalysisImport, fileName: string): string => {
    const notes = [...r.notes];
    if (r.sensitivity?.length || r.tornado) notes.push("Its sensitivity and tornado settings are imported on the Analysis tab.");
    if (!r.monte_carlo) return [`${fileName} has no Monte Carlo analysis to import.`, ...notes].join(" ");
    setNumSamples(r.monte_carlo.num_samples);
    setBatchSize(r.monte_carlo.batch_size);
    return [
      `Imported the Monte Carlo settings from ${fileName} (${r.monte_carlo.num_samples.toLocaleString("en-US")} samples); run it with the button below.`,
      ...notes,
    ].join(" ");
  };

  return (
    <div>
      <AnalysisFileBar setError={setError} setNotice={setNotice} onImported={applyImport} />

      <div className="card">
        <h2>Monte Carlo Uncertainty Analysis</h2>
        <div className="form-grid" style={{ marginBottom: 16 }}>
          <div className="form-group">
            <label>Number of Samples<HelpTip id="mc.num_samples" /></label>
            <input type="number" value={numSamples} onChange={(e) => setNumSamples(+e.target.value)} />
          </div>
          <div className="form-group">
            <label>Batch Size<HelpTip id="mc.batch_size" /></label>
            <input type="number" value={batchSize} onChange={(e) => setBatchSize(+e.target.value)} />
          </div>
        </div>

        <ComparePlants
          project={project}
          isTicked={selection.isTicked}
          onToggle={selection.toggle}
          note="Each ticked plant runs its own Monte Carlo simulation with the same number of samples."
        />

        <button className="btn-primary" onClick={() => run()} disabled={loading}>
          {loading && <span className="spinner" />}
          {loading ? "Running Monte Carlo..." : "Run Monte Carlo"}
        </button>
        {!loading && missing.length > 0 && (
          <p className="compare-plants-note" style={{ marginTop: 8 }}>
            Run again to include {missing.join(", ")}.
          </p>
        )}
        {loading && (
          <p style={{ color: "#868e96", fontSize: 13, marginTop: 8 }}>
            This may take a while with {numSamples.toLocaleString()} samples
            {runCount > 1 ? ` × ${runCount} plants` : ""}...
          </p>
        )}
      </div>

      {result && plants.length === 0 && (
        <div className="card"><p className="compare-plants-empty">No plants ticked. Tick a plant above to show its results.</p></div>
      )}

      {plants.length > 0 && (
        <>
          {/* Summary statistics — one row per (plant, metric) */}
          <div className="card">
            <h2>Results Summary ({plants[0].num_samples.toLocaleString()} samples)</h2>
            <div style={{ overflowX: "auto" }}>
              <table>
                <thead>
                  <tr>
                    <th>Plant</th><th>Metric</th><th>Unit</th><th>Mean</th><th>Std</th>
                    <th>P5</th><th>P25</th><th>Median</th><th>P75</th><th>P95</th>
                    <th>Min</th><th>Max</th>
                  </tr>
                </thead>
                <tbody>
                  {plants.flatMap((p) =>
                    Object.entries(p.metrics).map(([name, stats]) => (
                      <tr key={`${p.pid}-${name}`}>
                        <td>
                          <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: colorOf(p), marginRight: 8, verticalAlign: "middle" }} />
                          <strong>{p.name}</strong>
                        </td>
                        <td>{name}</td>
                        <td>{metricInfo(name, p.currency).unit ?? ""}</td>
                        <td className="number">{fmt(stats.mean)}</td>
                        <td className="number">{fmt(stats.std)}</td>
                        <td className="number">{fmt(stats.p5)}</td>
                        <td className="number">{fmt(stats.p25)}</td>
                        <td className="number">{fmt(stats.p50)}</td>
                        <td className="number">{fmt(stats.p75)}</td>
                        <td className="number">{fmt(stats.p95)}</td>
                        <td className="number">{fmt(stats.min)}</td>
                        <td className="number">{fmt(stats.max)}</td>
                      </tr>
                    )),
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div className="mc-plot-options">
            <label>
              <input type="checkbox" checked={showFit} onChange={(e) => toggleFit(e.target.checked)} />
              Show fitted normal distribution
            </label>
            <label>
              <input type="checkbox" checked={showPct} onChange={(e) => togglePct(e.target.checked)} />
              Show percentiles
            </label>
            {pctText.map((t, i) => (
              <span key={i} className="mc-pct-level">
                P
                <input
                  type="number" min={0} max={100} step="any" value={t}
                  disabled={!showPct}
                  className={validPct(t) ? undefined : "invalid"}
                  title="Percentile, 0 to 100 (P90: 90% of the samples are at or below it). Leave blank to drop it."
                  onChange={(e) => setPctLevel(i, e.target.value)}
                />
              </span>
            ))}
          </div>

          {/* Histograms or cumulative curves for each metric — overlaid across plants */}
          {metricNames.map((metric) => {
            const view: MCView = viewOf[metric] ?? "histogram";
            const cumulative = view === "cumulative";
            const data = cumulative ? buildCumulativeData(plants, metric) : buildOverlayData(plants, metric, showFit);
            if (data.length === 0) return null;
            const seriesPlants = plants.filter((p) => p.metrics[metric] != null);
            const multi = seriesPlants.length > 1;
            // Overlaid plants share one axis: name the currency only if they agree
            const currencies = [...new Set(seriesPlants.map((p) => p.currency))];
            const info = metricInfo(metric, currencies.length === 1 ? currencies[0] : "currency");
            const xLabel = withUnit(info.name, info.unit);
            // Percentile values per plant, in one shared power of ten (as the figure's legend)
            const pctValues = new Map(seriesPlants.map((p) => [p.pid, percentiles.map((pc) => quantileAt(p.metrics[metric]!.quantiles, pc))]));
            const exp = pctExponent([...pctValues.values()].flat());
            const pctHeader = percentiles.map((pc) => `P${pc}`).join(", ");
            const xMin = data[0].x;
            const xMax = data[data.length - 1].x;

            return (
              <div key={metric} className="card">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
                  <h2>{metric} Distribution</h2>
                  <div className="seg-toggle" role="group" aria-label="Histogram or cumulative distribution">
                    <button className={cumulative ? "" : "active"} onClick={() => setViewOf({ ...viewOf, [metric]: "histogram" })}>Histogram</button>
                    <button className={cumulative ? "active" : ""} onClick={() => setViewOf({ ...viewOf, [metric]: "cumulative" })}>Cumulative</button>
                  </div>
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 18, fontSize: 13, color: "var(--text-secondary)", marginBottom: 8 }}>
                  {seriesPlants.map((p) => {
                    const s = p.metrics[metric]!;
                    const parts = [
                      !cumulative && showFit ? `μ=${fmt(s.mean)}, σ=${fmt(s.std)}` : null,
                      percentiles.length ? `${pctHeader} = ${pctValues.get(p.pid)!.map((v) => fmt3(v / 10 ** exp)).join(", ")}` : null,
                    ].filter(Boolean);
                    return (
                      <span key={p.pid} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <span style={{ width: 12, height: 12, background: colorOf(p), display: "inline-block" }} />
                        <strong>{p.name}</strong> {parts.join("; ")}
                        {percentiles.length > 0 && exp !== 0 && <span>(×10<sup>{exp}</sup>)</span>}
                      </span>
                    );
                  })}
                </div>
                <DownloadableChart
                  filename={`mc_${metric}${cumulative ? "_cumulative" : ""}`}
                  height={430}
                  serverPlot={(opts) => fetchPlotPng("/plots/monte-carlo", {
                    metric,
                    kind: view,
                    show_fit: showFit,
                    percentiles: percentiles.length ? percentiles : null,
                    plants: seriesPlants.map((p) => p.pos),
                  }, opts)}
                >
                  {cumulative ? (
                  <ResponsiveContainer>
                    <ComposedChart data={data} margin={{ bottom: 46, left: 12, top: 10, right: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="x"
                        type="number"
                        domain={["dataMin", "dataMax"]}
                        ticks={niceTicks(xMin, xMax)}
                        tickFormatter={fmtTick}
                        label={{ value: xLabel, position: "insideBottom", offset: -30, style: { fontSize: 14, fill: "#666" } }}
                      />
                      <YAxis width={74} tick={{ fontSize: 11 }} domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} label={{ value: "Cumulative probability", angle: -90, position: "insideLeft", offset: 0, style: { fontSize: 14, fill: "#666", textAnchor: "middle" } }} />
                      <Tooltip
                        labelFormatter={(v) => `${metric}: ${fmtValue(Number(v), 3)}${info.unit && info.unit !== "-" ? ` ${info.unit}` : ""}`}
                        formatter={(v, name) => {
                          const pid = String(name).replace(/__cdf$/, "");
                          const plantName = plants.find((p) => p.pid === pid)?.name ?? pid;
                          return [`${(Number(v) * 100).toFixed(1)}% at or below`, plantName];
                        }}
                      />
                      {/* Several plants: dashed levels, a dot where each curve crosses them */}
                      {multi && percentiles.map((pc) => (
                        <ReferenceLine key={`lvl-${pc}`} y={pc / 100} stroke="#999" strokeDasharray="4 3" />
                      ))}
                      {seriesPlants.map((p) => (
                        <Line
                          key={`${p.pid}-cdf`}
                          type="linear"
                          dataKey={`${p.pid}__cdf`}
                          stroke={colorOf(p)}
                          strokeWidth={2}
                          dot={false}
                          connectNulls
                          isAnimationActive={false}
                        />
                      ))}
                      {multi && seriesPlants.flatMap((p) => percentiles.map((pc, i) => (
                        <ReferenceDot key={`${p.pid}-dot-${pc}`} x={pctValues.get(p.pid)![i]} y={pc / 100} r={3.5} fill={colorOf(p)} stroke="none" />
                      )))}
                      {/* One plant: guide lines from the y-axis to the curve and down */}
                      {!multi && percentiles.flatMap((pc, i) => {
                        const v = pctValues.get(seriesPlants[0].pid)![i];
                        return [
                          <ReferenceLine key={`h-${pc}`} segment={[{ x: xMin, y: pc / 100 }, { x: v, y: pc / 100 }]} stroke="#888" strokeDasharray="4 3" />,
                          <ReferenceLine key={`v-${pc}`} segment={[{ x: v, y: 0 }, { x: v, y: pc / 100 }]} stroke="#888" strokeDasharray="4 3" />,
                        ];
                      })}
                    </ComposedChart>
                  </ResponsiveContainer>
                  ) : (
                  <ResponsiveContainer>
                    <ComposedChart data={data} margin={{ bottom: 46, left: 12, top: 10, right: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="x"
                        type="number"
                        domain={["dataMin", "dataMax"]}
                        ticks={niceTicks(data[0].x, data[data.length - 1].x)}
                        tickFormatter={fmtTick}
                        label={{ value: xLabel, position: "insideBottom", offset: -30, style: { fontSize: 14, fill: "#666" } }}
                      />
                      <YAxis width={74} tick={{ fontSize: 11 }} tickFormatter={(v: number) => (v === 0 ? "0" : Number(v).toExponential(1))} label={{ value: "Probability density", angle: -90, position: "insideLeft", offset: 0, style: { fontSize: 14, fill: "#666", textAnchor: "middle" } }} />
                      <Tooltip
                        labelFormatter={(v) => `${metric}: ${fmtValue(Number(v), 3)}${info.unit && info.unit !== "-" ? ` ${info.unit}` : ""}`}
                        formatter={(v, name) => {
                          const label = String(name);
                          const isHist = label.endsWith("__hist");
                          const isPdf = label.endsWith("__pdf");
                          const pid = label.replace(/__(hist|pdf)$/, "");
                          const plantName = plants.find((p) => p.pid === pid)?.name ?? pid;
                          const tag = isHist ? "histogram" : isPdf ? "fitted normal" : "";
                          return [Number(v).toExponential(2), `${plantName}${tag ? ` (${tag})` : ""}`];
                        }}
                      />
                      {seriesPlants.map((p) => {
                        const color = colorOf(p);
                        return [
                          <Area
                            key={`${p.pid}-hist`}
                            type="stepAfter"
                            dataKey={`${p.pid}__hist`}
                            stroke={color}
                            strokeOpacity={0.5}
                            fill={color}
                            fillOpacity={0.25}
                            isAnimationActive={false}
                            connectNulls={false}
                          />,
                          showFit && <Line
                            key={`${p.pid}-pdf`}
                            type="monotone"
                            dataKey={`${p.pid}__pdf`}
                            stroke={color}
                            strokeWidth={2}
                            dot={false}
                            isAnimationActive={false}
                          />,
                        ];
                      })}
                      {/* Percentile lines, dashed in each plant's colour */}
                      {seriesPlants.flatMap((p) => pctValues.get(p.pid)!.map((v, i) => (
                        <ReferenceLine key={`${p.pid}-p-${i}`} x={v} stroke={multi ? colorOf(p) : "#888"} strokeDasharray="4 3" />
                      )))}
                    </ComposedChart>
                  </ResponsiveContainer>
                  )}
                </DownloadableChart>
              </div>
            );
          })}

          {/* Input distributions — the first plant shown */}
          {plants[0] && Object.keys(plants[0].inputs).length > 0 && (
            <div className="card">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
                <h2>Input Parameter Distributions ({plants[0].name})</h2>
                <div style={{ display: "flex", gap: 8 }}>
                  {(["process", "economic"] as const).map((cat) => (
                    <button
                      key={cat}
                      className="btn-secondary" style={{ padding: "4px 10px", fontSize: 12 }}
                      title="Download the library-rendered figure, exactly as plot_monte_carlo_inputs produces in Jupyter"
                      onClick={async () => {
                        try {
                          const blob = await fetchPlotPng(`/plots/monte-carlo/inputs?category=${cat}&plant=${plants[0].pos}`);
                          await saveFigure(blob, `mc_inputs_${cat}.png`);
                        } catch (e: unknown) {
                          setError(e instanceof Error ? e.message : "Figure download failed");
                        }
                      }}
                    >
                      ⬇ {cat} figure
                    </button>
                  ))}
                </div>
              </div>
              <table>
                <thead>
                  <tr><th>Input</th><th>Unit</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
                </thead>
                <tbody>
                  {Object.entries(plants[0].inputs).map(([name, stats]) => (
                    <tr key={name}>
                      <td>{texToText(stats.label ?? name)}</td>
                      <td>{inputUnit(name, plants[0].currency) ?? ""}</td>
                      <td className="number">{fmt(stats.mean)}</td>
                      <td className="number">{fmt(stats.std)}</td>
                      <td className="number">{fmt(stats.min)}</td>
                      <td className="number">{fmt(stats.max)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* One small histogram per sampled input — the empirical bins
                  show the configured distribution family directly (a uniform
                  price reads flat, a triangular one peaks at its mode, ...).
                  Constant inputs (std = 0) are in the table but not plotted. */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: 16, marginTop: 20 }}>
                {Object.entries(plants[0].inputs)
                  .filter(([, stats]) => stats.std > 0)
                  .map(([name, stats], i) => {
                    const rows = inputHistRows(stats.histogram, plants[0].num_samples);
                    if (rows.length === 0) return null;
                    const color = COLORS[i % COLORS.length];
                    const unit = inputUnit(name, plants[0].currency);
                    return (
                      <div key={name}>
                        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4, color: "var(--text-secondary)" }}>{texToText(stats.label ?? name)}</div>
                        <DownloadableChart filename={`mc_input_${name.replace(/\W+/g, "_")}`} height={unit ? 196 : 180}>
                          <ResponsiveContainer>
                            <ComposedChart data={rows} margin={{ bottom: unit ? 22 : 8, left: 4, top: 4, right: 8 }}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis
                                dataKey="x"
                                type="number"
                                domain={["dataMin", "dataMax"]}
                                tick={{ fontSize: 11 }}
                                ticks={niceTicks(rows[0].x, rows[rows.length - 1].x, 4)}
                                tickFormatter={fmtTick}
                                label={unit ? { value: `[${unit}]`, position: "insideBottom", offset: -16, style: { fontSize: 11, fill: "#666" } } : undefined}
                              />
                              <YAxis tick={{ fontSize: 11 }} width={48} tickFormatter={(v: number) => (v === 0 ? "0" : Number(v).toExponential(0))} />
                              <Tooltip
                                labelFormatter={(v) => `${name}: ${fmtTick(Number(v))}${unit && unit !== "-" ? ` ${unit}` : ""}`}
                                formatter={(v) => [Number(v).toExponential(2), "density"]}
                              />
                              <Area
                                type="stepAfter"
                                dataKey="density"
                                stroke={color}
                                strokeOpacity={0.6}
                                fill={color}
                                fillOpacity={0.3}
                                isAnimationActive={false}
                              />
                            </ComposedChart>
                          </ResponsiveContainer>
                        </DownloadableChart>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
