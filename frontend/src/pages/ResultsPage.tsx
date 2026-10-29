import { useState, useEffect } from "react";
import { runCalculations, getPlantConfig, setPlantConfig as savePlantConfig, fetchPlotPng, exportJsonResults } from "../api/client";
import type { CalculationResults, PlantConfig, ProjectOverview } from "../types";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from "recharts";
import DownloadableChart from "../components/DownloadableChart";
import PlantPicker from "../components/PlantPicker";
import RampEditor from "../components/RampEditor";
import FactorChip from "../components/FactorChip";
import HelpTip from "../components/HelpTip";
import { asShare, fixedOpexLabel, fmtFactor, fmtPct, nameLabel } from "../format";
import PctToggle from "../components/PctToggle";
import { saveResultsFile } from "../jsonFiles";

/** Where a factor is stored: a plant setting (fc, fp) or a factor dict */
type FactorGroup = "plant" | "fixed_capital_factors" | "fixed_opex_factors"
  | "fixed_capital_components" | "fixed_opex_components";

interface FactorRow {
  name: string;
  value: number;
  group: FactorGroup;
  key: string;
  /** null: the item has no factor (e.g. operating labor) */
  factor: number | null;
  custom: boolean;
  /** What the factor multiplies */
  basis: string;
  /** Why the factor can't be edited */
  locked?: string;
  /** Equation shown in the (i) tip */
  eq?: string;
  /** Set the amount directly (null = compute it again); absent = not editable */
  setValue?: (v: number | null) => void;
  /** The amount is set directly */
  valueCustom?: boolean;
}

const OVERRIDDEN = "Value set directly, so this factor isn't used. Clear the value to use the factor again.";

// Fixed OPEX items: result key, factor key, component key (fixed_opex_components),
// library default and what the factor multiplies
const FIXED_OPEX_ITEMS: [string, string, string, number, string][] = [
  ["supervision", "supervision", "supervision_costs", 0.25, "Operating labor"],
  ["direct_salary_overhead", "direct_salary_overhead", "direct_salary_overhead", 0.5, "(Labor + Supervision)"],
  ["laboratory_charges", "laboratory_charges", "laboratory_charges", 0.1, "Operating labor"],
  ["maintenance", "maintenance", "maintenance_costs", 0.05, "ISBL"],
  ["taxes_insurance", "taxes_insurance", "taxes_insurance_costs", 0.015, "ISBL"],
  ["rent_of_land", "rent_of_land", "rent_of_land_costs", 0.015, "(ISBL + OSBL)"],
  ["environmental_charges", "environmental_charges", "environmental_charges", 0.01, "(ISBL + OSBL)"],
  ["operating_supplies", "operating_supplies", "operating_supplies", 0.009, "ISBL"],
  ["general_plant_overhead", "general_plant_overhead", "general_plant_overhead", 0.65, "(Labor + Supervision + Salary overhead)"],
  ["interest_working_capital", "working_capital_interest", "interest_working_capital", 0, "Working capital"],
  ["patents_royalties", "patents_royalties", "patents_royalties", 0.02, "CCOP"],
  ["distribution_selling", "distribution_selling", "distribution_selling_costs", 0.02, "CCOP"],
  ["rnd", "rnd", "RnD_costs", 0.03, "CCOP"],
];

interface Props {
  results: CalculationResults | null;
  setResults: (r: CalculationResults | null) => void;
  setError: (e: string | null) => void;
  setNotice: (m: string | null) => void;
  markDirty: () => void;
  project: ProjectOverview | null;
  onSwitchPlant: (id: string) => void;
}

export default function ResultsPage({ results, setResults, setError, setNotice, markDirty, project, onSwitchPlant }: Props) {
  const [loading, setLoading] = useState(false);
  const [plantConfig, setPlantConfig] = useState<PlantConfig | null>(null);
  // Breakdown charts shown as % of their total, by chart
  const [pctOn, setPctOn] = useState<Record<string, boolean>>({});
  const pctSet = (k: string) => (on: boolean) => setPctOn((prev) => ({ ...prev, [k]: on }));

  useEffect(() => {
    getPlantConfig().then(setPlantConfig).catch(() => {});
  }, []);

  const calculate = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await runCalculations();
      setResults(r);
      // Calculate is derived from current inputs — no markDirty: the user
      // should be able to compute, add to comparison, and load another
      // example without a "save your changes?" prompt blocking the flow.
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Calculation failed");
    } finally {
      setLoading(false);
    }
  };

  // Save a changed plant configuration and recalculate
  const applyConfig = async (next: PlantConfig) => {
    setError(null);
    try {
      await savePlantConfig(next);
      setPlantConfig(next);
      markDirty();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Saving the plant settings failed");
      return;
    }
    await calculate();
  };

  // CAPEX / production schedule (null = library default, so the key is
  // left out of saved files)
  const applyRamps = (capex: number[] | null, production: number[] | null) => {
    if (!plantConfig) return;
    const next: PlantConfig = { ...plantConfig, capex_ramp: capex, production_ramp: production };
    if (capex === null) delete next.capex_ramp;
    if (production === null) delete next.production_ramp;
    applyConfig(next);
  };

  // A cost factor (null = back to the default): f_c and f_p are plant
  // settings, the others sit in fixed_capital_factors / fixed_opex_factors
  // (the working capital factor with the fixed OPEX ones, as in the library)
  const applyFactor = (group: FactorGroup, key: string, v: number | null) => {
    if (!plantConfig) return;
    if (group === "plant") {
      applyConfig({ ...plantConfig, [key]: v });
      return;
    }
    const out = { ...(plantConfig[group] ?? {}) };
    if (v === null) delete out[key]; else out[key] = v;
    applyConfig({ ...plantConfig, [group]: out });
  };

  // The amount, editable in place when the library lets it be set directly
  const valueCell = (d: FactorRow) => (
    <td className="number" style={{ whiteSpace: "nowrap" }}>
      {d.setValue
        ? <FactorChip variant="amount" value={d.value} custom={!!d.valueCustom} disabled={loading} onCommit={d.setValue} />
        : fmt(d.value)}
    </td>
  );

  // Factor chip and what it multiplies, as two table cells
  const factorCells = (d: FactorRow) => (
    <td style={{ whiteSpace: "nowrap" }}>
      {d.factor !== null && (
        <FactorChip value={d.factor} custom={d.custom} locked={d.locked} disabled={loading} onCommit={(v) => applyFactor(d.group, d.key, v)} />
      )}
      <span style={{ color: "var(--text-muted)", fontSize: 12, marginLeft: 6 }}>{d.basis}</span>
    </td>
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

  const picker = (
    <PlantPicker project={project} onSwitch={onSwitchPlant} hint="results are for the selected plant" />
  );

  if (!results) {
    return (
      <div>
      {picker}
      <div className="card" style={{ textAlign: "center", padding: 40 }}>
        <p style={{ color: "#868e96", marginBottom: 16 }}>Configure equipment and plant, then run calculations.</p>
        <button className="btn-primary" style={{ padding: "12px 32px", fontSize: 16 }} onClick={calculate} disabled={loading}>
          {loading && <span className="spinner" />}
          {loading ? "Calculating..." : "Run Calculations"}
        </button>
      </div>
      </div>
    );
  }

  const m = results.metrics;

  const byDesc = (a: { value: number }, b: { value: number }) => b.value - a.value;

  // CAPEX breakdown, each component with the factor it was computed with
  // (read back from the results, so it is what the library used)
  const capex = results.capital_costs;
  const isbl = capex.isbl ?? 0;
  const osbl = capex.osbl ?? 0;
  const fixedCapital = capex.fixed_capital ?? 0;
  const ratio = (a: number | null | undefined, b: number) => (b ? (a ?? 0) / b : 0);
  const fcf = plantConfig?.fixed_capital_factors ?? {};
  const fcc = plantConfig?.fixed_capital_components ?? {};
  // ISBL has no direct override: an amount sets the f_c that gives it
  const fcNow = plantConfig?.fc ?? 1;
  const setIsbl = (v: number | null) => applyFactor("plant", "fc", v === null || !isbl ? null : (v / isbl) * fcNow);
  const setCapital = (key: string) => (v: number | null) => applyFactor("fixed_capital_components", key, v);
  const capexData = ([
    { name: "ISBL", value: isbl, group: "plant", key: "fc", factor: plantConfig?.fc ?? 1, custom: plantConfig?.fc != null, basis: "× Σ direct cost × location", setValue: setIsbl, valueCustom: plantConfig?.fc != null },
    { name: "OSBL", value: osbl, group: "fixed_capital_factors", key: "osbl", factor: ratio(osbl, isbl), custom: fcf.osbl != null, basis: "× ISBL", locked: fcc.osbl != null ? OVERRIDDEN : undefined, setValue: setCapital("osbl"), valueCustom: fcc.osbl != null },
    { name: "D&E", value: capex.design_and_engineering ?? 0, group: "fixed_capital_factors", key: "de", factor: ratio(capex.design_and_engineering, isbl + osbl), custom: fcf.de != null, basis: "× (ISBL + OSBL)", locked: fcc.dne != null ? OVERRIDDEN : undefined, setValue: setCapital("dne"), valueCustom: fcc.dne != null },
    { name: "Contingency", value: capex.contingency ?? 0, group: "fixed_capital_factors", key: "contingency", factor: ratio(capex.contingency, isbl + osbl), custom: fcf.contingency != null, basis: "× (ISBL + OSBL)", locked: fcc.contingency != null ? OVERRIDDEN : undefined, setValue: setCapital("contingency"), valueCustom: fcc.contingency != null },
  ] satisfies FactorRow[]).filter((d) => d.value > 0).sort(byDesc);
  const factorOf = (k: string) => fmtFactor(capexData.find((d) => d.key === k)?.factor ?? 0);
  const wcFactor = ratio(capex.working_capital, fixedCapital);
  const wcManual = plantConfig?.working_capital != null;
  const cur = plantConfig?.currency || "USD";
  const fixedCapitalEq = [
    "Fixed capital = ISBL + OSBL + D&E + Contingency",
    "",
    `ISBL = Σ equipment direct cost × location factor × f_c (f_c = ${factorOf("fc")})`,
    `OSBL = f_OSBL × ISBL = ${factorOf("osbl")} × ${fmt(isbl)}`,
    `D&E = f_D&E × (ISBL + OSBL) = ${factorOf("de")} × ${fmt(isbl + osbl)}`,
    `Contingency = f_cont × (ISBL + OSBL) = ${factorOf("contingency")} × ${fmt(isbl + osbl)}`,
    "",
    `Fixed capital = ${fmt(fixedCapital)} ${cur}`,
    "Click a factor to change it; clear it to go back to the default.",
  ].join("\n");
  const workingCapitalEq = wcManual
    ? "Working capital is set as a fixed amount (here or on the Plant Config tab). Clear the amount to compute it as f_WC × fixed capital again."
    : [
        "Working capital = f_WC × Fixed capital",
        `= ${fmtFactor(wcFactor)} × ${fmt(fixedCapital)} = ${fmt(capex.working_capital)} ${cur}`,
        "",
        "It is drawn in the last construction year and recovered in the final project year.",
      ].join("\n");

  // Fixed OPEX breakdown, each item with its factor (read back from the
  // results) and equation
  const fixedOpex = results.fixed_opex;
  const opexLabel = fixedOpexLabel;
  const labor = fixedOpex.operating_labor ?? 0;
  const supervision = fixedOpex.supervision ?? 0;
  const salaryOverhead = fixedOpex.direct_salary_overhead ?? 0;
  const itemsSum = Object.entries(fixedOpex).reduce((t, [k, v]) => (k === "total" ? t : t + (v ?? 0)), 0);
  // Cash cost of production: patents, distribution and R&D are shares of
  // it, so it equals variable OPEX plus every fixed OPEX item
  const ccop = results.variable_opex.total + itemsSum;
  const bases: Record<string, number> = {
    "Operating labor": labor,
    "(Labor + Supervision)": labor + supervision,
    "ISBL": isbl,
    "(ISBL + OSBL)": isbl + osbl,
    "(Labor + Supervision + Salary overhead)": labor + supervision + salaryOverhead,
    "Working capital": capex.working_capital ?? 0,
    "CCOP": ccop,
  };
  const baseDefs: Record<string, string> = {
    "(Labor + Supervision)": `Labor + Supervision = ${fmt(labor)} + ${fmt(supervision)}`,
    "(ISBL + OSBL)": `ISBL + OSBL = ${fmt(isbl)} + ${fmt(osbl)}`,
    "(Labor + Supervision + Salary overhead)": `Labor + Supervision + Salary overhead = ${fmt(labor)} + ${fmt(supervision)} + ${fmt(salaryOverhead)}`,
    "CCOP": `CCOP (cash cost of production) = Variable OPEX + all fixed OPEX items = ${fmt(results.variable_opex.total)} + ${fmt(itemsSum)}`,
  };
  const foFactors = plantConfig?.fixed_opex_factors ?? {};
  const foComponents = plantConfig?.fixed_opex_components ?? {};
  const rate = plantConfig?.operator_hourly_rate?.rate ?? 38.11;
  const hours = plantConfig
    ? plantConfig.working_weeks_per_year * plantConfig.working_shifts_per_week * (24 / plantConfig.operating_shifts_per_day)
    : 0;
  const fixedOpexData = ([
    {
      name: opexLabel("operating_labor"), value: labor, group: "plant", key: "", factor: null, custom: false,
      basis: "operators × hours × rate",
      eq: [
        "Operating labor = Operators hired × Working hours per year × Hourly rate",
        hours && rate ? `= ${Number((labor / (hours * rate)).toFixed(2))} × ${fmt(hours)} h × ${fmt(rate)} ${cur}/h = ${fmt(labor)}` : "",
        "",
        "Working hours = weeks per year × shifts per week × 24 h / shifts per day.",
        "Set the operators and the hourly rate on the Plant Config tab.",
      ].join("\n"),
    },
    ...FIXED_OPEX_ITEMS.map(([k, fk, ck, dflt, basis]): FactorRow => {
      const value = fixedOpex[k] ?? 0;
      const base = bases[basis];
      const factor = base ? value / base : (foFactors[fk] ?? dflt);
      const name = opexLabel(k);
      return {
        name, value, group: "fixed_opex_factors", key: fk, factor,
        custom: foFactors[fk] != null,
        basis: `× ${basis}`,
        locked: foComponents[ck] != null ? OVERRIDDEN : undefined,
        setValue: (v) => applyFactor("fixed_opex_components", ck, v),
        valueCustom: foComponents[ck] != null,
        eq: foComponents[ck] != null ? [
          `${name} is set directly to ${fmt(value)} ${cur}.`,
          "",
          `Clear the amount to compute it as f × ${basis} again.`,
        ].join("\n") : [
          `${name} = f × ${basis}`,
          `= ${fmtFactor(factor)} × ${fmt(base)} = ${fmt(value)}`,
          ...(baseDefs[basis] ? ["", baseDefs[basis]] : []),
          ...(k === "interest_working_capital" ? ["", "0 (default): working capital is an investment in the cash flow (drawn at the end of construction, recovered at the end). Above 0 it is debt-funded instead: this interest is paid every year and the draw/release is dropped."] : []),
          ...(basis === "CCOP" ? ["", "Patents & Royalties, Distribution & Selling and R&D are shares of the cash cost of production, which itself includes them."] : []),
        ].join("\n"),
      };
    }),
  ] satisfies FactorRow[]).sort(byDesc);
  const fp = plantConfig?.fp ?? 1;
  const fixedOpexEq = [
    "Annual fixed OPEX = f_p × (sum of the items above)",
    `= ${fmtFactor(fp)} × ${fmt(itemsSum)} = ${fmt(fixedOpex.total)} ${cur}`,
    "",
    "f_p scales the whole fixed OPEX (1 = as estimated). Each item's (i) shows its own equation.",
  ].join("\n");

  // Variable OPEX breakdown
  const varOpexData = Object.entries(results.variable_opex.breakdown)
    .filter(([, v]) => typeof v === "number" && (v as number) > 0)
    .map(([k, v]) => ({ name: nameLabel(k), value: v as number }))
    .sort(byDesc);

  // Revenue breakdown
  const revenueData = Object.entries(results.revenue.breakdown)
    .map(([k, v]) => ({ name: nameLabel(k), value: v as number }))
    .sort(byDesc);

  // Chart rows, as values or as % of the chart's total
  const capexChart = pctOn.capital ? asShare(capexData) : capexData;
  const fixedOpexPos = fixedOpexData.filter((d) => d.value > 0);
  const fixedOpexChart = pctOn.fixed_opex ? asShare(fixedOpexPos) : fixedOpexPos;
  const varOpexChart = pctOn.variable_opex ? asShare(varOpexData) : varOpexData;
  const revenueChart = pctOn.revenue ? asShare(revenueData) : revenueData;
  const pctQuery = (k: string) => (pctOn[k] ? "?pct=true" : "");
  const currency = plantConfig?.currency || "USD";
  const lcUnit = `${currency}/Unit`;

  const discountRate = plantConfig?.interest_rate ?? 0;
  const cumulativeNPV = results.cash_flow.cash_flow.reduce<number[]>((acc, cf, i) => {
    const discounted = cf / Math.pow(1 + discountRate, i + 1);
    acc.push((acc[i - 1] ?? 0) + discounted);
    return acc;
  }, []);

  return (
    <div>
      {picker}
      <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginBottom: 16 }}>
        <button
          className="btn-secondary"
          title="Write <plant>_plant_results.json, as run_tea does"
          onClick={async () => {
            setError(null);
            try {
              const where = await saveResultsFile(() => exportJsonResults("plant"));
              if (where) setNotice(`Saved ${where}`);
            } catch (e: unknown) {
              setError(e instanceof Error ? e.message : "Save failed");
            }
          }}
        >Save results JSON…</button>
        <button className="btn-primary" onClick={calculate} disabled={loading}>
          {loading && <span className="spinner" />}Recalculate
        </button>
      </div>

      {/* Key metrics */}
      <div className="metrics-row">
        <div className="metric-card">
          <div className="label">Levelized Cost (in {lcUnit})</div>
          <div className="value">{fmt(m.levelized_cost)}</div>
        </div>
        <div className="metric-card">
          <div className="label">NPV (in {currency})</div>
          <div className="value">{fmtCompact(m.npv)}</div>
        </div>
        <div className="metric-card">
          <div className="label">IRR</div>
          <div className="value">{pct(m.irr)}</div>
        </div>
        <div className="metric-card">
          <div className="label">ROI</div>
          <div className="value">{m.roi != null ? m.roi.toFixed(2) + "%" : "-"}</div>
        </div>
        <div className="metric-card">
          <div className="label">Payback Time</div>
          <div className="value">{fmt(m.payback_time)} yr</div>
        </div>
      </div>

      {/* CAPEX */}
      <div className="card">
        <div className="card-head">
          <h2>Capital Costs (in {currency})</h2>
          <PctToggle pct={!!pctOn.capital} onChange={pctSet("capital")} />
        </div>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 0 }}>
          <table style={{ flexShrink: 0, width: "50%", fontSize: 13 }}>
            <tbody>
              {capexData.map((d) => (
                <tr key={d.name}>
                  <td style={{ paddingRight: 12, whiteSpace: "nowrap" }}>{d.name}</td>
                  {factorCells(d)}
                  {valueCell(d)}
                </tr>
              ))}
              <tr style={{ fontWeight: 700 }}>
                <td style={{ paddingRight: 12, whiteSpace: "nowrap" }}>Fixed Capital<HelpTip icon="i" text={fixedCapitalEq} /></td>
                <td />
                <td className="number" style={{ whiteSpace: "nowrap" }}>{fmt(capex.fixed_capital)}</td>
              </tr>
              <tr>
                <td style={{ paddingRight: 12, whiteSpace: "nowrap" }}>Working Capital<HelpTip icon="i" text={workingCapitalEq} /></td>
                <td style={{ whiteSpace: "nowrap" }}>
                  <FactorChip
                    value={wcFactor}
                    custom={plantConfig?.fixed_opex_factors?.working_capital != null}
                    locked={wcManual ? OVERRIDDEN : undefined}
                    disabled={loading}
                    onCommit={(v) => applyFactor("fixed_opex_factors", "working_capital", v)}
                  />
                  <span style={{ color: "var(--text-muted)", fontSize: 12, marginLeft: 6 }}>× Fixed capital</span>
                </td>
                {valueCell({
                  name: "Working Capital", value: capex.working_capital ?? 0, group: "plant", key: "working_capital", factor: null, custom: false, basis: "",
                  setValue: (v) => applyFactor("plant", "working_capital", v), valueCustom: wcManual,
                })}
              </tr>
            </tbody>
          </table>
          {capexData.length > 0 && (
            <DownloadableChart filename="capital_costs" serverPlot={() => fetchPlotPng(`/plots/stacked/capital${pctQuery("capital")}`)} height={capexData.length * 40 + 60} style={{ flex: 1, minWidth: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={capexChart} layout="vertical" barCategoryGap={0} margin={{ top: 4, right: 16, bottom: 42, left: 8 }}>
                  <XAxis type="number" tickFormatter={pctOn.capital ? fmtPct : (v: number) => (v / 1e6).toFixed(0) + "M"} tick={{ fontSize: 12 }} domain={[0, Math.max(...capexChart.map((d) => d.value)) * 1.1]} label={{ value: pctOn.capital ? "Share of total (%)" : `Cost (${currency})`, position: "insideBottom", offset: -28, style: { fontSize: 14, fill: "#666" } }} />
                  <YAxis type="category" dataKey="name" width={100} tick={{ fontSize: 12 }} />
                  <Tooltip formatter={(v) => (pctOn.capital ? fmtPct(Number(v)) : fmt(Number(v)))} />
                  <Bar dataKey="value" fill="#4361ee" radius={[0, 4, 4, 0]} barSize={14} />
                </BarChart>
              </ResponsiveContainer>
            </DownloadableChart>
          )}
        </div>
      </div>

      {/* Fixed OPEX */}
      <div className="card">
        <div className="card-head">
          <h2>Annual fixed OPEX (in {currency})</h2>
          <PctToggle pct={!!pctOn.fixed_opex} onChange={pctSet("fixed_opex")} />
        </div>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 0 }}>
          <table style={{ flexShrink: 0, width: "50%", fontSize: 13 }}>
            <tbody>
              {fixedOpexData.map((d) => (
                <tr key={d.name}>
                  <td style={{ paddingRight: 12, whiteSpace: "nowrap" }}>{d.name}{d.eq && <HelpTip icon="i" text={d.eq} />}</td>
                  {factorCells(d)}
                  {valueCell(d)}
                </tr>
              ))}
              <tr style={{ fontWeight: 700 }}>
                <td style={{ paddingRight: 12, whiteSpace: "nowrap" }}>Total<HelpTip icon="i" text={fixedOpexEq} /></td>
                {(() => {
                  // No direct override for the total: an amount sets the f_p that gives it
                  const total: FactorRow = {
                    name: "Total", value: fixedOpex.total, group: "plant", key: "fp", factor: fp, custom: plantConfig?.fp != null, basis: "× Σ items",
                    setValue: (v) => applyFactor("plant", "fp", v === null || !itemsSum ? null : v / itemsSum), valueCustom: plantConfig?.fp != null,
                  };
                  return <>{factorCells(total)}{valueCell(total)}</>;
                })()}
              </tr>
            </tbody>
          </table>
          {fixedOpexData.some((d) => d.value > 0) && (
            <DownloadableChart filename="fixed_opex" serverPlot={() => fetchPlotPng(`/plots/stacked/fixed_opex${pctQuery("fixed_opex")}`)} height={fixedOpexData.filter((d) => d.value > 0).length * 30 + 60} style={{ flex: 1, minWidth: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={fixedOpexChart} layout="vertical" barCategoryGap={0} margin={{ top: 4, right: 16, bottom: 42, left: 8 }}>
                  <XAxis type="number" tickFormatter={pctOn.fixed_opex ? fmtPct : (v: number) => (v / 1e3).toFixed(0) + "k"} tick={{ fontSize: 12 }} domain={[0, Math.max(...fixedOpexChart.map((d) => d.value)) * 1.1]} label={{ value: pctOn.fixed_opex ? "Share of total (%)" : `Annual cost (${currency})`, position: "insideBottom", offset: -28, style: { fontSize: 14, fill: "#666" } }} />
                  <YAxis type="category" dataKey="name" width={160} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v) => (pctOn.fixed_opex ? fmtPct(Number(v)) : fmt(Number(v)))} />
                  <Bar dataKey="value" fill="#f72585" radius={[0, 4, 4, 0]} barSize={14} />
                </BarChart>
              </ResponsiveContainer>
            </DownloadableChart>
          )}
        </div>
      </div>

      {/* Variable OPEX */}
      {varOpexData.length > 0 && (
        <div className="card">
          <div className="card-head">
            <h2>Annual variable OPEX (in {currency})</h2>
            <PctToggle pct={!!pctOn.variable_opex} onChange={pctSet("variable_opex")} />
          </div>
          <div style={{ display: "flex", alignItems: "flex-start", gap: 0 }}>
            <table style={{ flexShrink: 0, width: "50%", fontSize: 13 }}>
              <tbody>
                {varOpexData.map((d) => (
                  <tr key={d.name}><td style={{ paddingRight: 12, whiteSpace: "nowrap" }}>{d.name}</td><td className="number" style={{ whiteSpace: "nowrap" }}>{fmt(d.value)}</td></tr>
                ))}
                <tr style={{ fontWeight: 700 }}><td style={{ paddingRight: 12 }}>Total</td><td className="number">{fmt(results.variable_opex.total)}</td></tr>
              </tbody>
            </table>
            <DownloadableChart filename="variable_opex" serverPlot={() => fetchPlotPng(`/plots/stacked/variable_opex${pctQuery("variable_opex")}`)} height={varOpexData.length * 40 + 60} style={{ flex: 1, minWidth: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={varOpexChart} layout="vertical" barCategoryGap={0} margin={{ top: 4, right: 16, bottom: 42, left: 8 }}>
                  <XAxis type="number" tickFormatter={pctOn.variable_opex ? fmtPct : (v: number) => (v / 1e6).toFixed(1) + "M"} tick={{ fontSize: 12 }} domain={[0, Math.max(...varOpexChart.map((d) => d.value)) * 1.1]} label={{ value: pctOn.variable_opex ? "Share of total (%)" : `Annual cost (${currency})`, position: "insideBottom", offset: -28, style: { fontSize: 14, fill: "#666" } }} />
                  <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 12 }} />
                  <Tooltip formatter={(v) => (pctOn.variable_opex ? fmtPct(Number(v)) : fmt(Number(v)))} />
                  <Bar dataKey="value" fill="#7209b7" radius={[0, 4, 4, 0]} barSize={14} />
                </BarChart>
              </ResponsiveContainer>
            </DownloadableChart>
          </div>
        </div>
      )}

      {/* Revenue */}
      {results.revenue.total > 0 && (
        <div className="card">
          <div className="card-head">
            <h2>Annual revenue (in {currency})</h2>
            <PctToggle pct={!!pctOn.revenue} onChange={pctSet("revenue")} />
          </div>
          <div style={{ display: "flex", alignItems: "flex-start", gap: 0 }}>
            <table style={{ flexShrink: 0, width: "50%", fontSize: 13 }}>
              <tbody>
                {revenueData.map((d) => (
                  <tr key={d.name}><td style={{ paddingRight: 12, whiteSpace: "nowrap" }}>{d.name}</td><td className="number" style={{ whiteSpace: "nowrap" }}>{fmt(d.value)}</td></tr>
                ))}
                <tr style={{ fontWeight: 700 }}><td style={{ paddingRight: 12 }}>Total</td><td className="number">{fmt(results.revenue.total)}</td></tr>
              </tbody>
            </table>
            {revenueData.length > 0 && (
              <DownloadableChart filename="revenue" height={revenueData.length * 40 + 60} style={{ flex: 1, minWidth: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={revenueChart} layout="vertical" barCategoryGap={0} margin={{ top: 4, right: 16, bottom: 42, left: 8 }}>
                    <XAxis type="number" tickFormatter={pctOn.revenue ? fmtPct : (v: number) => (v / 1e6).toFixed(1) + "M"} tick={{ fontSize: 12 }} domain={[0, Math.max(...revenueChart.map((d) => d.value)) * 1.1]} label={{ value: pctOn.revenue ? "Share of total (%)" : `Annual revenue (${currency})`, position: "insideBottom", offset: -28, style: { fontSize: 14, fill: "#666" } }} />
                    <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 12 }} />
                    <Tooltip formatter={(v) => (pctOn.revenue ? fmtPct(Number(v)) : fmt(Number(v)))} />
                    <Bar dataKey="value" fill="#4cc9f0" radius={[0, 4, 4, 0]} barSize={14} />
                  </BarChart>
                </ResponsiveContainer>
              </DownloadableChart>
            )}
          </div>
        </div>
      )}

      {/* Cash Flow Table */}
      {results.cash_flow.cash_flow && results.cash_flow.cash_flow.length > 0 && (
        <div className="card">
          <h2>Cash Flow (in {currency})</h2>
          {plantConfig && (
            <RampEditor
              capexRamp={plantConfig.capex_ramp}
              productionRamp={plantConfig.production_ramp}
              lifetime={plantConfig.project_lifetime}
              busy={loading}
              onApply={applyRamps}
            />
          )}
          <div style={{ overflowX: "auto" }}>
            <table style={{ textAlign: "center" }}>
              <thead>
                <tr>
                  <th>Year</th><th>CAPEX</th><th>Revenue</th><th>Operating Costs</th>
                  <th>Depreciation</th><th>Gross Profit</th><th>Tax</th><th>Cash Flow</th><th>NPV</th>
                </tr>
              </thead>
              <tbody>
                {results.cash_flow.cash_flow.map((_val, yearIdx) => (
                  <tr key={yearIdx}>
                    <td>{yearIdx + 1}</td>
                    <td>{fmt(results.cash_flow.capital_cost_array[yearIdx] ?? 0)}</td>
                    <td>{fmt(results.cash_flow.revenue_array[yearIdx] ?? 0)}</td>
                    <td>{fmt(results.cash_flow.cash_cost_array[yearIdx] ?? 0)}</td>
                    <td>{fmt(results.cash_flow.depreciation_array[yearIdx] ?? 0)}</td>
                    <td>{fmt(results.cash_flow.gross_profit_array[yearIdx] ?? 0)}</td>
                    <td>{fmt(results.cash_flow.tax_paid_array[yearIdx] ?? 0)}</td>
                    <td>{fmt(results.cash_flow.cash_flow[yearIdx] ?? 0)}</td>
                    <td>{fmt(cumulativeNPV[yearIdx])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
