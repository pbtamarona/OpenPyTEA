import { useRef } from "react";
import type { ComparedPlant, CalculationResults } from "../types";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, Cell,
} from "recharts";
import DownloadableChart from "../components/DownloadableChart";

const COLORS = ["#4361ee", "#e63946", "#06d6a0", "#f77f00", "#7209b7", "#4cc9f0", "#d62828", "#2a9d8f", "#e9c46a", "#264653"];

interface Props {
  plants: ComparedPlant[];
  onRemove: (id: string) => void;
  onImport: (plant: ComparedPlant) => void;
  setError: (e: string | null) => void;
}

export default function ComparePage({ plants, onRemove, onImport, setError }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);

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
        onImport({
          id: crypto.randomUUID(),
          name: data.plant?.plant_name || file.name.replace(/\.json$/, ""),
          currency: data.plant?.currency || "USD",
          results: data.results as CalculationResults,
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
  ): { category: string; [k: string]: string | number }[] {
    const allKeys = new Set<string>();
    plants.forEach((p) => {
      Object.keys(extractor(p.results)).forEach((k) => allKeys.add(k));
    });
    return Array.from(allKeys)
      .map((key) => {
        const row: Record<string, string | number> = { category: key.replace(/_/g, " ") };
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

  // Single-metric comparison data
  function buildMetricBars(
    extractor: (r: CalculationResults) => number | null,
  ) {
    return plants.map((p, i) => ({
      name: dn(p),
      value: extractor(p.results) ?? 0,
      fill: COLORS[i % COLORS.length],
    }));
  }

  const lcopBars = buildMetricBars((r) => r.metrics.levelized_cost);
  const npvBars = buildMetricBars((r) => r.metrics.npv);
  const irrBars = buildMetricBars((r) => r.metrics.irr);

  if (plants.length === 0) {
    return (
      <div className="card" style={{ textAlign: "center", padding: 60 }}>
        <p style={{ color: "var(--text-muted)", marginBottom: 16 }}>
          No plants added yet. Run calculations on the Results tab and click
          "Add to Comparison", or import a saved project JSON.
        </p>
        <button className="btn-primary" onClick={() => fileRef.current?.click()}>
          Import from JSON
        </button>
        <input ref={fileRef} type="file" accept=".json" hidden onChange={handleImport} />
      </div>
    );
  }

  return (
    <div>
      {/* Toolbar */}
      <div style={{ display: "flex", gap: 8, marginBottom: 20, alignItems: "center" }}>
        <button className="btn-primary" onClick={() => fileRef.current?.click()}>
          Import from JSON
        </button>
        <input ref={fileRef} type="file" accept=".json" hidden onChange={handleImport} />
        <span style={{ color: "var(--text-muted)", fontSize: 13 }}>
          {plants.length} plant{plants.length !== 1 ? "s" : ""} in comparison
        </span>
      </div>

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
              {plants.map((p, i) => (
                <tr key={p.id}>
                  <td>
                    <span style={{ display: "inline-block", width: 12, height: 12, borderRadius: "50%", background: COLORS[i % COLORS.length], marginRight: 8, verticalAlign: "middle" }} />
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
                  <td><button className="btn-danger" onClick={() => onRemove(p.id)}>Remove</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* CAPEX breakdown */}
      {capexData.length > 0 && (
        <div className="card">
          <h2>Capital Cost Breakdown ({currencyLabel})</h2>
          <DownloadableChart filename="compare_capex" height={350}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={capexData} margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={{ fontSize: 12 }} />
                <YAxis tickFormatter={(v: number) => fmtCompact(v)} />
                <Tooltip formatter={(v) => fmt(Number(v))} />
                <Legend />
                {plants.map((p, i) => (
                  <Bar key={p.id} dataKey={dn(p)} fill={COLORS[i % COLORS.length]} radius={[4, 4, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </DownloadableChart>
        </div>
      )}

      {/* Variable OPEX breakdown */}
      {varOpexData.length > 0 && (
        <div className="card">
          <h2>Variable OPEX Breakdown ({currencyLabel})</h2>
          <DownloadableChart filename="compare_variable_opex" height={350}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={varOpexData} margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={{ fontSize: 12 }} />
                <YAxis tickFormatter={(v: number) => fmtCompact(v)} />
                <Tooltip formatter={(v) => fmt(Number(v))} />
                <Legend />
                {plants.map((p, i) => (
                  <Bar key={p.id} dataKey={dn(p)} fill={COLORS[i % COLORS.length]} radius={[4, 4, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </DownloadableChart>
        </div>
      )}

      {/* Key metrics comparison */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 20 }}>
        {/* Levelized Cost */}
        <div className="card">
          <h2>Levelized Cost</h2>
          <DownloadableChart filename="compare_lcop" height={250}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={lcopBars} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tickFormatter={(v: number) => v.toFixed(2)} />
                <Tooltip formatter={(v) => fmt(Number(v))} />
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
                <Tooltip formatter={(v) => fmt(Number(v))} />
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
                <Tooltip formatter={(v) => pct(Number(v))} />
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
