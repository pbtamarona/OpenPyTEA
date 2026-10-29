import { useRef, useState } from "react";
import type { ComparedPlant, CalculationResults } from "../types";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, Cell,
  type TooltipContentProps,
} from "recharts";
import type { NameType, ValueType } from "recharts/types/component/DefaultTooltipContent";
import DownloadableChart from "../components/DownloadableChart";
import { fetchPlotPng } from "../api/client";
import { fixedOpexLabel, fmtPct, nameLabel } from "../format";
import PctToggle from "../components/PctToggle";

const COLORS = ["#4361ee", "#e63946", "#06d6a0", "#f77f00", "#7209b7", "#4cc9f0", "#d62828", "#2a9d8f", "#e9c46a", "#264653"];

/** Native warning in Tauri (window.alert is suppressed in its webview). */
async function warn(text: string, title: string) {
  try {
    const { isTauri } = await import("@tauri-apps/api/core");
    if (isTauri()) {
      const { message } = await import("@tauri-apps/plugin-dialog");
      await message(text, { title, kind: "warning" });
      return;
    }
  } catch {
    // fall through to the browser dialog
  }
  window.alert(text);
}

interface Props {
  /** The project's calculated plants, always compared */
  projectPlants: ComparedPlant[];
  /** Names of project plants without results yet */
  notCalculated: string[];
  /** Plants imported from saved files */
  plants: ComparedPlant[];
  /** Plants unticked (not shown) */
  hidden: Set<string>;
  onToggle: (id: string) => void;
  /** Colour slot of a plant, the same as on Analysis and Monte Carlo */
  colorIndex: (id: string, fallback: number) => number;
  onRemove: (id: string) => void;
  onImport: (plant: ComparedPlant) => void;
  setError: (e: string | null) => void;
}

export default function ComparePage({
  projectPlants, notCalculated, plants: imported, hidden, onToggle, colorIndex, onRemove, onImport, setError,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  // Breakdown charts shown as % of each plant's total, by chart
  const [pctOn, setPctOn] = useState<Record<string, boolean>>({});
  const pctSet = (k: string) => (on: boolean) => setPctOn((prev) => ({ ...prev, [k]: on }));
  const all = [...projectPlants, ...imported];
  // Only the ticked plants are compared
  const plants = all.filter((p) => !hidden.has(p.id));
  const colorOf = (p: ComparedPlant) => COLORS[colorIndex(p.id, all.indexOf(p)) % COLORS.length];
  const importedIds = new Set(imported.map((p) => p.id));
  const pendingNote = notCalculated.length > 0 && (
    <p className="compare-plants-note">
      Not calculated yet (open on the Results tab to include): {notCalculated.join(", ")}
    </p>
  );

  const fmt = (n: number | null | undefined) =>
    n != null ? n.toLocaleString("en-US", { maximumFractionDigits: 2 }) : "-";

  const fmtCompact = (n: number | null | undefined) => {
    if (n == null) return "-";
    const abs = Math.abs(n);
    if (abs >= 1e9) return (n / 1e9).toFixed(2) + "B";
    if (abs >= 1e6) return (n / 1e6).toFixed(2) + "M";
    if (abs >= 1e3) return (n / 1e3).toFixed(2) + "k";
    return n.toFixed(2);
  };

  const pct = (n: number | null | undefined) =>
    n != null ? (n * 100).toFixed(2) + "%" : "-";

  // Deduplicate display names
  const displayNames = new Map<string, string>();
  const nameCounts = new Map<string, number>();
  plants.forEach((p) => nameCounts.set(p.name, (nameCounts.get(p.name) || 0) + 1));
  const nameIdx = new Map<string, number>();
  plants.forEach((p) => {
    if ((nameCounts.get(p.name) || 0) > 1) {
      const idx = (nameIdx.get(p.name) || 0) + 1;
      nameIdx.set(p.name, idx);
      displayNames.set(p.id, `${p.name} (${idx})`);
    } else {
      displayNames.set(p.id, p.name);
    }
  });

  const dn = (p: ComparedPlant) => displayNames.get(p.id) || p.name;

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(reader.result as string);
        if (!data.results || !data.results.metrics) {
          setError("JSON file does not contain calculation results. Run calculations before saving.");
          return;
        }
        const name = data.plant?.plant_name || file.name.replace(/\.json$/, "");
        const hasSource = Array.isArray(data.equipment) && data.equipment.length > 0 && data.plant;
        onImport({
          id: crypto.randomUUID(),
          name,
          currency: data.plant?.currency || "USD",
          results: data.results as CalculationResults,
          source: hasSource ? { name, equipment: data.equipment, plant: data.plant } : undefined,
        });
      } catch {
        setError("Failed to parse JSON file");
      }
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  // Mixed currencies?
  const currencies = new Set(plants.map((p) => p.currency));
  const currencyLabel = currencies.size === 1 ? plants[0]?.currency || "USD" : "mixed currencies";

  // Build grouped chart data
  function buildGrouped(
    extractor: (r: CalculationResults) => Record<string, number>,
    label: (key: string) => string = nameLabel,
  ): { category: string; [k: string]: string | number }[] {
    const allKeys = new Set<string>();
    plants.forEach((p) => {
      Object.keys(extractor(p.results)).forEach((k) => allKeys.add(k));
    });
    return Array.from(allKeys)
      .map((key) => {
        const row: Record<string, string | number> = { category: label(key) };
        let total = 0;
        plants.forEach((p) => {
          const val = extractor(p.results)[key] ?? 0;
          row[dn(p)] = val;
          total += Math.abs(val);
        });
        return { ...row, _total: total };
      })
      .filter((r) => (r._total as number) > 0)
      .sort((a, b) => (b._total as number) - (a._total as number))
      .map(({ _total: _, ...rest }) => rest as { category: string; [k: string]: string | number });
  }

  const capexData = buildGrouped((r) => ({
    ISBL: r.capital_costs.isbl ?? 0,
    OSBL: r.capital_costs.osbl ?? 0,
    "D&E": r.capital_costs.design_and_engineering ?? 0,
    Contingency: r.capital_costs.contingency ?? 0,
  }));

  const varOpexData = buildGrouped((r) => r.variable_opex.breakdown);
  const fixedOpexData = buildGrouped(
    (r) => Object.fromEntries(Object.entries(r.fixed_opex).filter(([k]) => k !== "total")),
    fixedOpexLabel,
  );

  // Grouped rows as % of each plant's own total (the library's pct=True)
  const asPlantShare = (rows: { category: string; [k: string]: string | number }[]) => {
    const totals = new Map(plants.map((p) => [dn(p), rows.reduce((t, r) => t + Number(r[dn(p)] ?? 0), 0)]));
    return rows.map((r) => {
      const out = { ...r };
      totals.forEach((t, k) => { out[k] = t ? (Number(r[k] ?? 0) / t) * 100 : 0; });
      return out;
    });
  };

  // Single-metric comparison data
  function buildMetricBars(
    extractor: (r: CalculationResults) => number | null,
  ) {
    return plants.map((p) => ({
      name: dn(p),
      value: extractor(p.results) ?? 0,
      fill: colorOf(p),
    }));
  }

  // Library-rendered figure of the ticked plants; only when every one has
  // its inputs to rebuild from (results-only imports get the screen render)
  const libraryPlot = (kind: string, pct = false) =>
    plants.every((p) => p.source)
      ? async () => {
          if (currencies.size > 1 && !pct) {
            await warn(
              `These plants use mixed currencies (${[...currencies].join(", ")}). ` +
              `The figure's axis label uses the first plant's currency (${plants[0].currency}).`,
              "Mixed currencies",
            );
          }
          return fetchPlotPng(`/plots/compare/${kind}`, {
            extra_plants: plants.map((p) => ({ ...p.source!, name: dn(p) })),
            pct,
          });
        }
      : undefined;

  // The bars are coloured per plant (Cell), which the default tooltip can't
  // see, so write the value in the hovered bar's colour
  const metricTooltip = (show: (v: number) => string) =>
    ({ active, payload, label }: TooltipContentProps<ValueType, NameType>) => {
      const item = payload?.[0];
      if (!active || !item) return null;
      return (
        <div className="recharts-default-tooltip" style={{ padding: 10, border: "1px solid", whiteSpace: "nowrap" }}>
          <p style={{ margin: 0 }}>{label}</p>
          <p style={{ margin: 0, paddingTop: 4, color: item.payload?.fill }}>value : {show(Number(item.value))}</p>
        </div>
      );
    };

  const lcopBars = buildMetricBars((r) => r.metrics.levelized_cost);
  const npvBars = buildMetricBars((r) => r.metrics.npv);
  const irrBars = buildMetricBars((r) => r.metrics.irr);

  if (all.length === 0) {
    return (
      <div className="card" style={{ textAlign: "center", padding: 60 }}>
        <p style={{ color: "var(--text-muted)", marginBottom: 16 }}>
          Every calculated plant of the project is compared here. Calculate a
          plant on the Results tab, or import a saved project JSON.
        </p>
        {pendingNote}
        <button className="btn-primary" onClick={() => fileRef.current?.click()}>
          Import from JSON
        </button>
        <input ref={fileRef} type="file" accept=".json" hidden onChange={handleImport} />
      </div>
    );
  }

  const toolbar = (
    <>
      <div style={{ display: "flex", gap: 8, marginBottom: 20, alignItems: "center" }}>
        <button className="btn-primary" onClick={() => fileRef.current?.click()}>
          Import from JSON
        </button>
        <input ref={fileRef} type="file" accept=".json" hidden onChange={handleImport} />
        <span style={{ color: "var(--text-muted)", fontSize: 13 }}>
          {plants.length} of {all.length} plant{all.length !== 1 ? "s" : ""} in comparison
        </span>
      </div>
      <div className="card compare-plants">
        <span className="compare-plants-label">Plants</span>
        <div className="compare-plants-list">
          {all.map((p) => (
            <label key={p.id}>
              <input type="checkbox" checked={!hidden.has(p.id)} onChange={() => onToggle(p.id)} />
              <span className="compare-plants-swatch" style={{ background: colorOf(p) }} />
              {dn(p)}
              {imported.includes(p) && <span className="compare-plants-tag">imported</span>}
            </label>
          ))}
        </div>
        {pendingNote}
      </div>
    </>
  );

  if (plants.length === 0) {
    return (
      <div>
        {toolbar}
        <div className="card">
          <p className="compare-plants-empty">No plants ticked. Tick a plant above to compare it.</p>
        </div>
      </div>
    );
  }

  return (
    <div>
      {toolbar}

      {/* Metrics table */}
      <div className="card">
        <h2>Metrics Overview</h2>
        <div style={{ overflowX: "auto" }}>
          <table>
            <thead>
              <tr>
                <th>Plant</th><th>Currency</th><th>Fixed Capital</th>
                <th>Fixed OPEX</th><th>Variable OPEX</th><th>Revenue</th>
                <th>LCOP</th><th>NPV</th><th>IRR</th><th>ROI</th><th>PBT</th><th></th>
              </tr>
            </thead>
            <tbody>
              {plants.map((p) => (
                <tr key={p.id}>
                  <td>
                    <span style={{ display: "inline-block", width: 12, height: 12, borderRadius: "50%", background: colorOf(p), marginRight: 8, verticalAlign: "middle" }} />
                    {dn(p)}
                  </td>
                  <td>{p.currency}</td>
                  <td className="number">{fmtCompact(p.results.capital_costs.fixed_capital)}</td>
                  <td className="number">{fmtCompact(p.results.fixed_opex.total)}</td>
                  <td className="number">{fmtCompact(p.results.variable_opex.total)}</td>
                  <td className="number">{fmtCompact(p.results.revenue.total)}</td>
                  <td className="number">{fmt(p.results.metrics.levelized_cost)}</td>
                  <td className="number">{fmtCompact(p.results.metrics.npv)}</td>
                  <td className="number">{pct(p.results.metrics.irr)}</td>
                  <td className="number">{p.results.metrics.roi != null ? p.results.metrics.roi.toFixed(2) + "%" : "-"}</td>
                  <td className="number">{fmt(p.results.metrics.payback_time)}</td>
                  <td>
                    {importedIds.has(p.id) && (
                      <button className="btn-danger" onClick={() => onRemove(p.id)} title="Remove this imported plant">Remove</button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* CAPEX breakdown */}
      {capexData.length > 0 && (
        <div className="card">
          <div className="card-head">
            <h2>Capital Cost ({currencyLabel})</h2>
            <PctToggle pct={!!pctOn.capital} onChange={pctSet("capital")} />
          </div>
          <DownloadableChart filename="compare_capex" height={350} serverPlot={libraryPlot("capital", !!pctOn.capital)}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={pctOn.capital ? asPlantShare(capexData) : capexData} margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={{ fontSize: 12 }} />
                <YAxis tickFormatter={(v: number) => (pctOn.capital ? fmtPct(v) : fmtCompact(v))} />
                <Tooltip formatter={(v) => (pctOn.capital ? fmtPct(Number(v)) : fmt(Number(v)))} />
                <Legend verticalAlign="top" align="right" wrapperStyle={{ paddingBottom: 8 }} />
                {plants.map((p) => (
                  <Bar key={p.id} dataKey={dn(p)} fill={colorOf(p)} radius={[4, 4, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </DownloadableChart>
        </div>
      )}

      {/* Variable and fixed OPEX side by side (stacked on narrow windows) */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(420px, 1fr))", gap: 20 }}>
      {/* Variable OPEX breakdown */}
      {varOpexData.length > 0 && (
        <div className="card">
          <div className="card-head">
            <h2>Annual Variable OPEX ({currencyLabel})</h2>
            <PctToggle pct={!!pctOn.variable_opex} onChange={pctSet("variable_opex")} />
          </div>
          <DownloadableChart filename="compare_variable_opex" height={400} serverPlot={libraryPlot("variable_opex", !!pctOn.variable_opex)}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={pctOn.variable_opex ? asPlantShare(varOpexData) : varOpexData} margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={{ fontSize: 11 }} angle={-35} textAnchor="end" height={70} interval={0} />
                <YAxis tickFormatter={(v: number) => (pctOn.variable_opex ? fmtPct(v) : fmtCompact(v))} />
                <Tooltip formatter={(v) => (pctOn.variable_opex ? fmtPct(Number(v)) : fmt(Number(v)))} />
                <Legend verticalAlign="top" align="right" wrapperStyle={{ paddingBottom: 8 }} />
                {plants.map((p) => (
                  <Bar key={p.id} dataKey={dn(p)} fill={colorOf(p)} radius={[4, 4, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </DownloadableChart>
        </div>
      )}

      {/* Fixed OPEX breakdown */}
      {fixedOpexData.length > 0 && (
        <div className="card">
          <div className="card-head">
            <h2>Annual Fixed OPEX ({currencyLabel})</h2>
            <PctToggle pct={!!pctOn.fixed_opex} onChange={pctSet("fixed_opex")} />
          </div>
          <DownloadableChart filename="compare_fixed_opex" height={400} serverPlot={libraryPlot("fixed_opex", !!pctOn.fixed_opex)}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={pctOn.fixed_opex ? asPlantShare(fixedOpexData) : fixedOpexData} margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={{ fontSize: 11 }} angle={-35} textAnchor="end" height={70} interval={0} />
                <YAxis tickFormatter={(v: number) => (pctOn.fixed_opex ? fmtPct(v) : fmtCompact(v))} />
                <Tooltip formatter={(v) => (pctOn.fixed_opex ? fmtPct(Number(v)) : fmt(Number(v)))} />
                <Legend verticalAlign="top" align="right" wrapperStyle={{ paddingBottom: 8 }} />
                {plants.map((p) => (
                  <Bar key={p.id} dataKey={dn(p)} fill={colorOf(p)} radius={[4, 4, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </DownloadableChart>
        </div>
      )}

      </div>

      {/* Key metrics comparison */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 20 }}>
        {/* Levelized Cost */}
        <div className="card">
          <h2>Levelized Cost</h2>
          <DownloadableChart filename="compare_lcop" height={250} serverPlot={libraryPlot("levelized_cost")}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={lcopBars} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tickFormatter={(v: number) => v.toFixed(2)} />
                <Tooltip content={metricTooltip(fmt)} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {lcopBars.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </DownloadableChart>
        </div>

        {/* NPV */}
        <div className="card">
          <h2>NPV ({currencyLabel})</h2>
          <DownloadableChart filename="compare_npv" height={250}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={npvBars} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tickFormatter={(v: number) => fmtCompact(v)} />
                <Tooltip content={metricTooltip(fmt)} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {npvBars.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </DownloadableChart>
        </div>

        {/* IRR */}
        <div className="card">
          <h2>IRR</h2>
          <DownloadableChart filename="compare_irr" height={250}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={irrBars} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tickFormatter={(v: number) => (v * 100).toFixed(1) + "%"} />
                <Tooltip content={metricTooltip(pct)} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {irrBars.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </DownloadableChart>
        </div>
      </div>
    </div>
  );
}
