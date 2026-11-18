import { useState } from "react";
import HelpTip from "./HelpTip";

// The library's defaults (Plant.calculate_cash_flow), in %
const DEFAULT_CAPEX = [30, 60, 10];
const DEFAULT_PRODUCTION = [0, 0, 40, 80];

const toPct = (r: number[] | null | undefined, fallback: number[]) =>
  r && r.length ? r.map((v) => Math.round(v * 1e6) / 1e4) : fallback;
// OPEX ramps: null = the library default (derived from production)
const toPctOrNull = (r: number[] | null | undefined) => (r && r.length ? toPct(r, []) : null);
const same = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => v === b[i]);
const sameOrNull = (a: number[] | null, b: number[] | null) => (a && b ? same(a, b) : a === b);

/** Fractions (0–1, OPEX may exceed 1) to save; null = back to the library default */
export interface Ramps {
  capex_ramp: number[] | null;
  production_ramp: number[] | null;
  fixed_opex_ramp: number[] | null;
  variable_opex_ramp: number[] | null;
}

interface Props {
  capexRamp: number[] | null | undefined;
  productionRamp: number[] | null | undefined;
  fixedOpexRamp: number[] | null | undefined;
  variableOpexRamp: number[] | null | undefined;
  lifetime: number;
  busy: boolean;
  onApply: (ramps: Ramps) => void;
}

/** Edit how CAPEX is spent over the construction years and how production
    ramps up; OPEX follows production as in the library unless a row is
    switched to custom. */
export default function RampEditor({ capexRamp, productionRamp, fixedOpexRamp, variableOpexRamp, lifetime, busy, onApply }: Props) {
  const savedCapex = toPct(capexRamp, DEFAULT_CAPEX);
  const savedProd = toPct(productionRamp, DEFAULT_PRODUCTION);
  const savedFixed = toPctOrNull(fixedOpexRamp);
  const savedVar = toPctOrNull(variableOpexRamp);
  const [capex, setCapex] = useState(savedCapex);
  const [prod, setProd] = useState(savedProd);
  const [fixed, setFixed] = useState(savedFixed);
  const [vari, setVari] = useState(savedVar);

  const total = capex.reduce((s, v) => s + v, 0);
  const totalOk = Math.abs(total - 100) < 1e-4;
  const prodOk = prod.every((v) => v >= 0 && v <= 100);
  const opexOk = [fixed, vari].every((r) => !r || r.every((v) => v >= 0));
  const changed = !same(capex, savedCapex) || !same(prod, savedProd)
    || !sameOrNull(fixed, savedFixed) || !sameOrNull(vari, savedVar);
  const isDefault = same(savedCapex, DEFAULT_CAPEX) && same(savedProd, DEFAULT_PRODUCTION)
    && savedFixed === null && savedVar === null;

  // One column per year, plus a last "Year n+" column at 100%
  const nCols = Math.max(capex.length, prod.length, fixed?.length ?? 0, vari?.length ?? 0) + 1;
  const years = Array.from({ length: nCols }, (_, i) => i);

  const resize = (arr: number[], n: number, fill: number) =>
    n > arr.length ? [...arr, ...Array(n - arr.length).fill(fill)] : arr.slice(0, n);
  const setAt = (arr: number[], i: number, v: number) => arr.map((x, j) => (j === i ? v : x));
  // Custom OPEX rows span the construction and ramp-up columns
  const setSpan = (c: number[], p: number[]) => {
    const n = Math.max(c.length, p.length);
    setCapex(c);
    setProd(p);
    setFixed((r) => r && resize(r, n, 100));
    setVari((r) => r && resize(r, n, 100));
  };

  // Library defaults: variable OPEX follows production, fixed OPEX is
  // charged in full from the first year with any production on
  const startUp = prod.findIndex((v) => v > 0);
  const prodAt = (y: number) => (y < prod.length ? prod[y] : 100);
  const fixedOn = (y: number) => ((startUp === -1 ? y >= prod.length : y >= startUp) ? 100 : 0);

  const input = (value: number, onChange: (v: number) => void, bad = false, max: number | undefined = 100) => (
    <input
      type="number" min={0} max={max} step={5} value={value}
      onChange={(e) => onChange(e.target.value === "" ? 0 : +e.target.value)}
      style={{ width: 64, textAlign: "right", borderColor: bad ? "var(--error-text)" : undefined }}
    />
  );
  const stepper = (label: string, n: number, set: (n: number) => void, min: number, max: number) => (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 13 }}>
      {label}
      <button className="btn-secondary" style={{ padding: "2px 9px" }} disabled={n <= min} onClick={() => set(n - 1)}>−</button>
      <strong style={{ minWidth: 14, textAlign: "center" }}>{n}</strong>
      <button className="btn-secondary" style={{ padding: "2px 9px" }} disabled={n >= max} onClick={() => set(n + 1)}>+</button>
    </span>
  );
  const muted = { color: "var(--text-muted)" };
  const rowHead = { textAlign: "left" as const, whiteSpace: "nowrap" as const, paddingRight: 12 };

  const opexRow = (
    label: string, help: string, defaultNote: string, row: number[] | null,
    set: (r: number[] | null) => void, derived: (y: number) => number,
  ) => (
    <tr style={row ? undefined : muted}>
      <td style={rowHead}>
        <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <input
            type="checkbox" checked={row !== null} title="Custom schedule"
            onChange={(e) => set(e.target.checked ? years.slice(0, -1).map(derived) : null)}
          />
          {label}
        </label>
        {row ? <HelpTip id={help} /> : <em> {defaultNote}</em>}
      </td>
      {years.map((y) => (
        <td key={y} style={{ textAlign: "center" }}>
          {row && y < row.length
            ? input(row[y], (v) => set(setAt(row, y, v)), row[y] < 0, undefined)
            : <span style={muted}>{row ? 100 : derived(y)}</span>}
        </td>
      ))}
    </tr>
  );

  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap", marginBottom: 6 }}>
        <h3 style={{ margin: 0, fontSize: 15 }}>Construction &amp; start-up schedule</h3>
        <span className="compare-plants-tag">{isDefault ? "default" : "custom"}</span>
      </div>
      <p style={{ ...muted, fontSize: 13, margin: "0 0 10px" }}>
        How the fixed capital is spent over the construction years, and how fast production reaches full capacity.
        By default variable OPEX and revenue follow production and fixed OPEX is paid in full from the first year with production;
        tick an OPEX row to set its own schedule (% of the full annual cost, above 100% for start-up overspend).
      </p>

      <div style={{ display: "flex", gap: 24, flexWrap: "wrap", marginBottom: 10 }}>
        {stepper("Construction years", capex.length, (n) => setSpan(resize(capex, n, 0), prod), 1, lifetime - 1)}
        {stepper("Ramp-up years", prod.length, (n) => setSpan(capex, resize(prod, n, 100)), 1, lifetime - 1)}
      </div>

      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "auto", fontSize: 13 }}>
          <thead>
            <tr>
              <th style={rowHead}>Year</th>
              {years.map((y) => <th key={y} style={{ textAlign: "center" }}>{y === nCols - 1 ? `${y + 1}+` : y + 1}</th>)}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={rowHead}>CAPEX spent (%)<HelpTip id="plant.capex_ramp" /></td>
              {years.map((y) => (
                <td key={y} style={{ textAlign: "center" }}>
                  {y < capex.length ? input(capex[y], (v) => setCapex(setAt(capex, y, v)), !totalOk) : <span style={muted}>–</span>}
                </td>
              ))}
              <td className="ramp-total" style={{ whiteSpace: "nowrap", paddingLeft: 12, color: totalOk ? "var(--success-text)" : "var(--error-text)" }}>
                {totalOk
                  ? <HelpTip text="Total 100%" icon="✓" />
                  : `Total ${Math.round(total * 100) / 100}% (must be 100%)`}
              </td>
            </tr>
            <tr>
              <td style={rowHead}>Production (% of capacity)<HelpTip id="plant.production_ramp" /></td>
              {years.map((y) => (
                <td key={y} style={{ textAlign: "center" }}>
                  {y < prod.length ? input(prod[y], (v) => setProd(setAt(prod, y, v)), prod[y] < 0 || prod[y] > 100) : <span style={muted}>100</span>}
                </td>
              ))}
            </tr>
            {opexRow("Variable OPEX (%)", "plant.variable_opex_ramp", "follows production", vari, setVari, prodAt)}
            {opexRow("Fixed OPEX (%)", "plant.fixed_opex_ramp", "from start-up", fixed, setFixed, fixedOn)}
          </tbody>
        </table>
      </div>

      <div style={{ display: "flex", gap: 8, marginTop: 10, alignItems: "center" }}>
        <button
          className="btn-primary"
          disabled={busy || !changed || !totalOk || !prodOk || !opexOk}
          onClick={() => {
            const frac = (r: number[] | null) => r && r.map((v) => v / 100);
            onApply({
              capex_ramp: frac(capex), production_ramp: frac(prod),
              fixed_opex_ramp: frac(fixed), variable_opex_ramp: frac(vari),
            });
          }}
        >
          {busy && <span className="spinner" />}Apply &amp; recalculate
        </button>
        <button
          className="btn-secondary"
          disabled={busy || (isDefault && !changed)}
          onClick={() => {
            setCapex(DEFAULT_CAPEX); setProd(DEFAULT_PRODUCTION); setFixed(null); setVari(null);
            onApply({ capex_ramp: null, production_ramp: null, fixed_opex_ramp: null, variable_opex_ramp: null });
          }}
        >
          Reset to defaults
        </button>
        {changed && <span style={{ ...muted, fontSize: 12 }}>Unapplied changes</span>}
      </div>
    </div>
  );
}
