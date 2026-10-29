import { useState } from "react";
import HelpTip from "./HelpTip";

// The library's defaults (Plant.calculate_cash_flow), in %
const DEFAULT_CAPEX = [30, 60, 10];
const DEFAULT_PRODUCTION = [0, 0, 40, 80];

const toPct = (r: number[] | null | undefined, fallback: number[]) =>
  r && r.length ? r.map((v) => Math.round(v * 1e6) / 1e4) : fallback;
const same = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => v === b[i]);

interface Props {
  capexRamp: number[] | null | undefined;
  productionRamp: number[] | null | undefined;
  lifetime: number;
  busy: boolean;
  /** Fractions (0–1) to save, or null to go back to the library default */
  onApply: (capex: number[] | null, production: number[] | null) => void;
}

/** Edit how CAPEX is spent over the construction years and how production
    ramps up; OPEX follows production, as in the library. */
export default function RampEditor({ capexRamp, productionRamp, lifetime, busy, onApply }: Props) {
  const savedCapex = toPct(capexRamp, DEFAULT_CAPEX);
  const savedProd = toPct(productionRamp, DEFAULT_PRODUCTION);
  const [capex, setCapex] = useState(savedCapex);
  const [prod, setProd] = useState(savedProd);

  const total = capex.reduce((s, v) => s + v, 0);
  const totalOk = Math.abs(total - 100) < 1e-4;
  const prodOk = prod.every((v) => v >= 0 && v <= 100);
  const changed = !same(capex, savedCapex) || !same(prod, savedProd);
  const isDefault = same(savedCapex, DEFAULT_CAPEX) && same(savedProd, DEFAULT_PRODUCTION);

  // One column per year, plus a last "Year n+" column at full capacity
  const nCols = Math.max(capex.length, prod.length) + 1;
  const years = Array.from({ length: nCols }, (_, i) => i);

  const resize = (arr: number[], n: number, fill: number) =>
    n > arr.length ? [...arr, ...Array(n - arr.length).fill(fill)] : arr.slice(0, n);
  const setAt = (arr: number[], i: number, v: number) => arr.map((x, j) => (j === i ? v : x));

  // Fixed OPEX is charged from the first year with any production on
  const startUp = prod.findIndex((v) => v > 0);
  const prodAt = (y: number) => (y < prod.length ? prod[y] : 100);
  const fixedOn = (y: number) => (startUp === -1 ? y >= prod.length : y >= startUp);

  const input = (value: number, onChange: (v: number) => void, bad = false) => (
    <input
      type="number" min={0} max={100} step={5} value={value}
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

  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap", marginBottom: 6 }}>
        <h3 style={{ margin: 0, fontSize: 15 }}>Construction &amp; start-up schedule</h3>
        <span className="compare-plants-tag">{isDefault ? "default" : "custom"}</span>
      </div>
      <p style={{ ...muted, fontSize: 13, margin: "0 0 10px" }}>
        How the fixed capital is spent over the construction years, and how fast production reaches full capacity.
        Variable OPEX and revenue follow production; fixed OPEX is paid in full from the first year with production.
      </p>

      <div style={{ display: "flex", gap: 24, flexWrap: "wrap", marginBottom: 10 }}>
        {stepper("Construction years", capex.length, (n) => setCapex(resize(capex, n, 0)), 1, lifetime - 1)}
        {stepper("Ramp-up years", prod.length, (n) => setProd(resize(prod, n, 100)), 1, lifetime - 1)}
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
              <td style={{ whiteSpace: "nowrap", paddingLeft: 12, color: totalOk ? "var(--success-text)" : "var(--error-text)" }}>
                Total {Math.round(total * 100) / 100}%{totalOk ? " ✓" : " (must be 100%)"}
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
            <tr style={muted}>
              <td style={rowHead}>Variable OPEX (%) <em>follows production</em></td>
              {years.map((y) => <td key={y} style={{ textAlign: "center" }}>{prodAt(y)}</td>)}
            </tr>
            <tr style={muted}>
              <td style={rowHead}>Fixed OPEX (%) <em>from start-up</em></td>
              {years.map((y) => <td key={y} style={{ textAlign: "center" }}>{fixedOn(y) ? 100 : 0}</td>)}
            </tr>
          </tbody>
        </table>
      </div>

      <div style={{ display: "flex", gap: 8, marginTop: 10, alignItems: "center" }}>
        <button
          className="btn-primary"
          disabled={busy || !changed || !totalOk || !prodOk}
          onClick={() => onApply(capex.map((v) => v / 100), prod.map((v) => v / 100))}
        >
          {busy && <span className="spinner" />}Apply &amp; recalculate
        </button>
        <button
          className="btn-secondary"
          disabled={busy || (isDefault && !changed)}
          onClick={() => { setCapex(DEFAULT_CAPEX); setProd(DEFAULT_PRODUCTION); onApply(null, null); }}
        >
          Reset to defaults
        </button>
        {changed && <span style={{ ...muted, fontSize: 12 }}>Unapplied changes</span>}
      </div>
    </div>
  );
}
