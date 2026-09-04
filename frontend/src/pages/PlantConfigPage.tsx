import { useEffect, useState } from "react";
import { getPlantConfig, setPlantConfig, getLocations } from "../api/client";
import type { PlantConfig, DependencyBlock } from "../types";

// ── Parameter dependencies (3.0 dependency DAG) ────────────────────
// A dependency ties one parameter to others: dependent = Σ weight·parent
// + offset. It lives in the plant config (consumption_dependency /
// production_dependency / dependency blocks) and is honoured by Monte
// Carlo, sensitivity, and tornado alike.

const PROJECT_PARAMS = [
  "fixed_capital_factor", "fixed_opex_factor", "project_lifetime",
  "interest_rate", "plant_utilization", "tax_rate", "operator_hourly_rate",
];

interface DepRule {
  target: string;                              // "consumption:x" | "production:y" | "project:p"
  parents: { ref: string; weight: number }[];
  offset: number;
  noise: number | null;                        // MC-only additive noise on the DAG-implied mean
}

const refLabel = (ref: string) => {
  const [kind, name] = ref.split(":");
  return `${name} (${kind})`;
};

const allNodeRefs = (config: PlantConfig): string[] => [
  ...Object.keys(config.variable_opex_inputs).map((k) => `consumption:${k}`),
  ...Object.keys(config.plant_products).map((k) => `production:${k}`),
  ...PROJECT_PARAMS.map((p) => `project:${p}`),
];

const rulesFromConfig = (config: PlantConfig): DepRule[] => {
  const rules: DepRule[] = [];
  const push = (target: string, block: DependencyBlock | null | undefined, noise: number | null) => {
    if (!block || !block.depends_on) return;
    rules.push({
      target,
      parents: Object.entries(block.depends_on).map(([ref, weight]) => ({ ref, weight })),
      offset: block.offset ?? 0,
      noise,
    });
  };
  for (const [k, v] of Object.entries(config.variable_opex_inputs)) {
    push(`consumption:${k}`, v.consumption_dependency, v.consumption_uncertainty?.noise ?? null);
  }
  for (const [k, v] of Object.entries(config.plant_products)) {
    push(`production:${k}`, v.production_dependency, v.production_uncertainty?.noise ?? null);
  }
  push("project:operator_hourly_rate", config.operator_hourly_rate.dependency,
    config.operator_hourly_rate.noise ?? null);
  for (const [p, v] of Object.entries(config.project_uncertainties ?? {})) {
    if (p === "operator_hourly_rate") continue;
    push(`project:${p}`, v.dependency as DependencyBlock | null, (v.noise as number | undefined) ?? null);
  }
  return rules;
};

// Remove the dependency (and its noise) stored at `target`, restoring the
// shape an independent parameter needs.
const clearDep = (config: PlantConfig, target: string): PlantConfig => {
  const [kind, name] = target.split(":");
  const next = structuredClone(config);
  if (kind === "consumption" && next.variable_opex_inputs[name]) {
    delete next.variable_opex_inputs[name].consumption_dependency;
    delete next.variable_opex_inputs[name].consumption_uncertainty;
  } else if (kind === "production" && next.plant_products[name]) {
    delete next.plant_products[name].production_dependency;
    delete next.plant_products[name].production_uncertainty;
  } else if (target === "project:operator_hourly_rate") {
    delete next.operator_hourly_rate.dependency;
    delete next.operator_hourly_rate.noise;
    // A dependent op-rate must not carry std; give it back on release
    if (next.operator_hourly_rate.std == null) next.operator_hourly_rate.std = 0;
  } else if (kind === "project") {
    const pu = next.project_uncertainties ?? {};
    if (pu[name]) {
      delete pu[name].dependency;
      delete pu[name].noise;
      if (Object.keys(pu[name]).length === 0) delete pu[name];
    }
    next.project_uncertainties = Object.keys(pu).length ? pu : null;
  }
  return next;
};

// Write `rule` into the config (clearing whatever was at `prevTarget` first
// when the dependent changed).
const writeDep = (config: PlantConfig, prevTarget: string | null, rule: DepRule): PlantConfig => {
  let next = prevTarget && prevTarget !== rule.target ? clearDep(config, prevTarget) : structuredClone(config);
  next = prevTarget === rule.target ? clearDep(next, rule.target) : next;
  const block: DependencyBlock = {
    depends_on: Object.fromEntries(rule.parents.map((p) => [p.ref, p.weight])),
    offset: rule.offset,
  };
  const [kind, name] = rule.target.split(":");
  if (kind === "consumption" && next.variable_opex_inputs[name]) {
    next.variable_opex_inputs[name].consumption_dependency = block;
    if (rule.noise != null) next.variable_opex_inputs[name].consumption_uncertainty = { noise: rule.noise };
  } else if (kind === "production" && next.plant_products[name]) {
    next.plant_products[name].production_dependency = block;
    if (rule.noise != null) next.plant_products[name].production_uncertainty = { noise: rule.noise };
  } else if (rule.target === "project:operator_hourly_rate") {
    next.operator_hourly_rate.dependency = block;
    // A dependent's value is set by the graph: the library rejects "std"
    // (only zero-centered "noise" applies), and absolute min/max bounds
    // would truncate that noise distribution to nothing — drop them all.
    delete next.operator_hourly_rate.std;
    delete next.operator_hourly_rate.min;
    delete next.operator_hourly_rate.max;
    if (rule.noise != null) next.operator_hourly_rate.noise = rule.noise;
  } else if (kind === "project") {
    const pu = next.project_uncertainties ?? {};
    // Replace, don't merge: leftover std/min/max from an independent
    // uncertainty block would clash with dependent-noise semantics
    pu[name] = { dependency: block };
    if (rule.noise != null) pu[name].noise = rule.noise;
    next.project_uncertainties = pu;
  }
  return next;
};

const defaultConfig: PlantConfig = {
  plant_name: "My Plant", process_type: "Fluids", country: "United States",
  region: "Gulf Coast", currency: "USD", exchange_rate: 1.0,
  interest_rate: 0.09, project_lifetime: 20, plant_utilization: 1.0,
  tax_rate: 0.0, working_capital: null, depreciation: null,
  operators_per_shift: null, operators_hired: null,
  operator_hourly_rate: { rate: 38.11, std: 10, min: 10, max: 100 },
  working_weeks_per_year: 49, working_shifts_per_week: 5,
  operating_shifts_per_day: 3, variable_opex_inputs: {},
  plant_products: {}, fc: null, fp: null,
  additional_capex_years: null, additional_capex_cost: null,
};

interface Props {
  setError: (e: string | null) => void;
  markDirty: () => void;
}

export default function PlantConfigPage({ setError, markDirty }: Props) {
  const [config, setConfig] = useState<PlantConfig>({ ...defaultConfig });
  const [locations, setLocations] = useState<Record<string, unknown>>({});
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    getPlantConfig().then((c) => {
      if (c && Object.keys(c).length > 0) setConfig(c as PlantConfig);
    }).catch((e: unknown) => {
      setError(e instanceof Error ? e.message : "Failed to load plant config");
    });
    getLocations().then(setLocations).catch((e: unknown) => {
      setError(e instanceof Error ? e.message : "Failed to load locations");
    });
  }, []);

  const regions = (() => {
    const loc = locations[config.country];
    if (!loc) return [];
    if (typeof loc === "number") return [];
    return Object.keys(loc as Record<string, number>);
  })();

  const [saveError, setSaveError] = useState<string | null>(null);
  const [editingVarOpexKey, setEditingVarOpexKey] = useState<string | null>(null);
  const [editingVarOpexName, setEditingVarOpexName] = useState("");
  const [editingProductKey, setEditingProductKey] = useState<string | null>(null);
  const [editingProductName, setEditingProductName] = useState("");

  const save = async () => {
    setSaveError(null);
    try {
      await setPlantConfig(config);
      markDirty();
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e: unknown) {
      setSaveError(e instanceof Error ? e.message : "Save failed");
    }
  };

  const u = (field: keyof PlantConfig, value: unknown) =>
    setConfig((prev) => ({ ...prev, [field]: value }));

  // Variable OPEX helpers
  const addVarOpex = () => {
    const name = prompt("Variable OPEX item name (e.g., electricity):");
    if (!name) return;
    setConfig((prev) => ({
      ...prev,
      variable_opex_inputs: { ...prev.variable_opex_inputs, [name]: { consumption: 0, price: 0, std: 0, min: 0, max: 99999 } },
    }));
  };

  const removeVarOpex = (key: string) => {
    setConfig((prev) => {
      const copy = { ...prev.variable_opex_inputs };
      delete copy[key];
      return { ...prev, variable_opex_inputs: copy };
    });
  };

  const renameVarOpex = (oldKey: string, newKey: string) => {
    const trimmed = newKey.trim();
    if (!trimmed || trimmed === oldKey) { setEditingVarOpexKey(null); return; }
    setConfig((prev) => {
      const renamed = Object.fromEntries(
        Object.entries(prev.variable_opex_inputs).map(([k, v]) => (k === oldKey ? [trimmed, v] : [k, v]))
      );
      return { ...prev, variable_opex_inputs: renamed };
    });
    setEditingVarOpexKey(null);
  };

  const updateVarOpex = (key: string, field: string, value: number) => {
    setConfig((prev) => ({
      ...prev,
      variable_opex_inputs: {
        ...prev.variable_opex_inputs,
        [key]: { ...prev.variable_opex_inputs[key], [field]: value },
      },
    }));
  };

  // Products helpers
  const addProduct = () => {
    const name = prompt("Product name (e.g., hydrogen):");
    if (!name) return;
    setConfig((prev) => ({
      ...prev,
      plant_products: { ...prev.plant_products, [name]: { production: 0, price: 0, std: 0, min: 0, max: 99999 } },
    }));
  };

  const removeProduct = (key: string) => {
    setConfig((prev) => {
      const copy = { ...prev.plant_products };
      delete copy[key];
      return { ...prev, plant_products: copy };
    });
  };

  const renameProduct = (oldKey: string, newKey: string) => {
    const trimmed = newKey.trim();
    if (!trimmed || trimmed === oldKey) { setEditingProductKey(null); return; }
    setConfig((prev) => {
      const renamed = Object.fromEntries(
        Object.entries(prev.plant_products).map(([k, v]) => (k === oldKey ? [trimmed, v] : [k, v]))
      );
      return { ...prev, plant_products: renamed };
    });
    setEditingProductKey(null);
  };

  const updateProduct = (key: string, field: string, value: number) => {
    setConfig((prev) => ({
      ...prev,
      plant_products: {
        ...prev.plant_products,
        [key]: { ...prev.plant_products[key], [field]: value },
      },
    }));
  };

  // Dependency rules are derived from the config on every render; edits
  // write straight back into the config via writeDep/clearDep.
  const rules = rulesFromConfig(config);
  const nodeRefs = allNodeRefs(config);
  const availableTargets = (current: string | null) =>
    nodeRefs.filter((r) => r === current || !rules.some((rl) => rl.target === r));

  return (
    <div>
      {/* General */}
      <div className="card">
        <h2>General</h2>
        <div className="form-grid">
          <div className="form-group">
            <label>Plant Name</label>
            <input value={config.plant_name} onChange={(e) => u("plant_name", e.target.value)} />
          </div>
          <div className="form-group">
            <label>Process Type</label>
            <select value={config.process_type} onChange={(e) => u("process_type", e.target.value)}>
              <option>Solids</option><option>Fluids</option><option>Mixed</option>
            </select>
          </div>
          <div className="form-group">
            <label>Country</label>
            <select value={config.country} onChange={(e) => { u("country", e.target.value); u("region", ""); }}>
              {Object.keys(locations).map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>
          {regions.length > 0 && (
            <div className="form-group">
              <label>Region</label>
              <select value={config.region} onChange={(e) => u("region", e.target.value)}>
                {regions.map((r) => <option key={r}>{r}</option>)}
              </select>
            </div>
          )}
          <div className="form-group">
            <label>Currency</label>
            <input value={config.currency} onChange={(e) => u("currency", e.target.value)} />
          </div>
          <div className="form-group">
            <label>Exchange Rate</label>
            <input type="number" step="0.01" value={config.exchange_rate} onChange={(e) => u("exchange_rate", +e.target.value)} />
          </div>
        </div>
      </div>

      {/* Financial */}
      <div className="card">
        <h2>Financial Parameters</h2>
        <div className="form-grid">
          <div className="form-group">
            <label>Interest Rate</label>
            <input type="number" step="0.01" value={config.interest_rate} onChange={(e) => u("interest_rate", +e.target.value)} />
          </div>
          <div className="form-group">
            <label>Project Lifetime (years)</label>
            <input type="number" value={config.project_lifetime} onChange={(e) => u("project_lifetime", +e.target.value)} />
          </div>
          <div className="form-group">
            <label>Plant Utilization (0-1)</label>
            <input type="number" step="0.01" value={config.plant_utilization} onChange={(e) => u("plant_utilization", +e.target.value)} />
          </div>
          <div className="form-group">
            <label>Tax Rate</label>
            <input type="number" step="0.01" value={config.tax_rate} onChange={(e) => u("tax_rate", +e.target.value)} />
          </div>
          <div className="form-group">
            <label>Working Capital (blank=auto)</label>
            <input type="number" value={config.working_capital ?? ""} onChange={(e) => u("working_capital", e.target.value ? +e.target.value : null)} />
          </div>
        </div>
      </div>

      {/* Labor */}
      <div className="card">
        <h2>Labor & Operations</h2>
        <div className="form-grid">
          <div className="form-group">
            <label>Operator Hourly Rate ($)</label>
            <input type="number" step="0.01" value={config.operator_hourly_rate.rate}
              onChange={(e) => setConfig((p) => ({ ...p, operator_hourly_rate: { ...p.operator_hourly_rate, rate: +e.target.value } }))} />
          </div>
          <div className="form-group">
            <label>Rate Std Dev (MC)</label>
            <input type="number" step="0.1" value={config.operator_hourly_rate.std ?? 0}
              disabled={config.operator_hourly_rate.dependency != null}
              title={config.operator_hourly_rate.dependency != null ? "Set by a parameter dependency — its spread comes from the parents (plus optional noise)" : undefined}
              onChange={(e) => setConfig((p) => ({ ...p, operator_hourly_rate: { ...p.operator_hourly_rate, std: +e.target.value } }))} />
          </div>
          <div className="form-group">
            <label>Operators/Shift (blank=auto)</label>
            <input type="number" value={config.operators_per_shift ?? ""} onChange={(e) => u("operators_per_shift", e.target.value ? +e.target.value : null)} />
          </div>
          <div className="form-group">
            <label>Working Weeks/Year</label>
            <input type="number" value={config.working_weeks_per_year} onChange={(e) => u("working_weeks_per_year", +e.target.value)} />
          </div>
          <div className="form-group">
            <label>Working Shifts/Week</label>
            <input type="number" value={config.working_shifts_per_week} onChange={(e) => u("working_shifts_per_week", +e.target.value)} />
          </div>
          <div className="form-group">
            <label>Operating Shifts/Day</label>
            <input type="number" value={config.operating_shifts_per_day} onChange={(e) => u("operating_shifts_per_day", +e.target.value)} />
          </div>
        </div>
      </div>

      {/* Products */}
      <div className="card">
        <h2>Plant Products</h2>
        {Object.keys(config.plant_products).length === 0 ? (
          <p style={{ color: "#868e96", fontSize: 13 }}>No products defined. First product is the main product.</p>
        ) : (
          <table>
            <thead><tr><th>Product</th><th>Production (daily)</th><th>Price ($/unit)</th><th>Std</th><th>Min</th><th>Max</th><th></th></tr></thead>
            <tbody>
              {Object.entries(config.plant_products).map(([key, val]) => (
                <tr key={key}>
                  <td>
                    {editingProductKey === key ? (
                      <input
                        value={editingProductName}
                        onChange={(e) => setEditingProductName(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") renameProduct(key, editingProductName);
                          if (e.key === "Escape") setEditingProductKey(null);
                        }}
                        autoFocus
                        style={{ width: 130 }}
                      />
                    ) : (
                      <strong>{key}</strong>
                    )}
                  </td>
                  <td><input type="number" value={val.production ?? 0} onChange={(e) => updateProduct(key, "production", +e.target.value)} style={{ width: 100 }} /></td>
                  <td><input type="number" value={val.price ?? 0} onChange={(e) => updateProduct(key, "price", +e.target.value)} style={{ width: 80 }} /></td>
                  <td><input type="number" value={val.std ?? 0} onChange={(e) => updateProduct(key, "std", +e.target.value)} style={{ width: 60 }} /></td>
                  <td><input type="number" value={val.min ?? 0} onChange={(e) => updateProduct(key, "min", +e.target.value)} style={{ width: 60 }} /></td>
                  <td><input type="number" value={val.max ?? 99999} onChange={(e) => updateProduct(key, "max", +e.target.value)} style={{ width: 70 }} /></td>
                  <td style={{ display: "flex", gap: 6 }}>
                    {editingProductKey === key ? (
                      <button className="btn-primary" onClick={() => renameProduct(key, editingProductName)}>Done</button>
                    ) : (
                      <button className="btn-primary" onClick={() => { setEditingProductKey(key); setEditingProductName(key); }}>Edit</button>
                    )}
                    <button className="btn-danger" onClick={() => removeProduct(key)}>Remove</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <button className="btn-primary" style={{ marginTop: 12 }} onClick={addProduct}>+ Add Product</button>
      </div>

      {/* Variable OPEX */}
      <div className="card">
        <h2>Variable OPEX Inputs</h2>
        {Object.keys(config.variable_opex_inputs).length === 0 ? (
          <p style={{ color: "#868e96", fontSize: 13 }}>No variable costs defined yet.</p>
        ) : (
          <table>
            <thead><tr><th>Item</th><th>Consumption (daily)</th><th>Price ($/unit)</th><th>Std</th><th>Min</th><th>Max</th><th></th></tr></thead>
            <tbody>
              {Object.entries(config.variable_opex_inputs).map(([key, val]) => (
                <tr key={key}>
                  <td>
                    {editingVarOpexKey === key ? (
                      <input
                        value={editingVarOpexName}
                        onChange={(e) => setEditingVarOpexName(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") renameVarOpex(key, editingVarOpexName);
                          if (e.key === "Escape") setEditingVarOpexKey(null);
                        }}
                        autoFocus
                        style={{ width: 130 }}
                      />
                    ) : (
                      <strong>{key}</strong>
                    )}
                  </td>
                  <td><input type="number" value={val.consumption ?? 0} onChange={(e) => updateVarOpex(key, "consumption", +e.target.value)} style={{ width: 100 }} /></td>
                  <td><input type="number" value={val.price ?? 0} onChange={(e) => updateVarOpex(key, "price", +e.target.value)} style={{ width: 80 }} /></td>
                  <td><input type="number" value={val.std ?? 0} onChange={(e) => updateVarOpex(key, "std", +e.target.value)} style={{ width: 60 }} /></td>
                  <td><input type="number" value={val.min ?? 0} onChange={(e) => updateVarOpex(key, "min", +e.target.value)} style={{ width: 60 }} /></td>
                  <td><input type="number" value={val.max ?? 99999} onChange={(e) => updateVarOpex(key, "max", +e.target.value)} style={{ width: 70 }} /></td>
                  <td style={{ display: "flex", gap: 6 }}>
                    {editingVarOpexKey === key ? (
                      <button className="btn-primary" onClick={() => renameVarOpex(key, editingVarOpexName)}>Done</button>
                    ) : (
                      <button className="btn-primary" onClick={() => { setEditingVarOpexKey(key); setEditingVarOpexName(key); }}>Edit</button>
                    )}
                    <button className="btn-danger" onClick={() => removeVarOpex(key)}>Remove</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <button className="btn-primary" style={{ marginTop: 12 }} onClick={addVarOpex}>+ Add Item</button>
      </div>

      {/* Parameter dependencies (3.0 dependency DAG) */}
      <div className="card">
        <h2>Parameter Dependencies</h2>
        <p style={{ color: "#868e96", fontSize: 13, marginBottom: 12 }}>
          Tie a parameter to others: <em>dependent = Σ weight × parent + offset</em>. A dependent is never
          sampled or varied on its own — Monte Carlo, sensitivity, and tornado all propagate its parents'
          values through the graph. Optional noise (Monte Carlo only) adds scatter around the implied value.
        </p>
        {rules.length === 0 && (
          <p style={{ color: "#868e96", fontSize: 13 }}>No dependencies defined.</p>
        )}
        {rules.map((rule) => (
          <div key={rule.target} style={{ border: "1px solid var(--border, #dee2e6)", borderRadius: 6, padding: 12, marginBottom: 10 }}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center", marginBottom: 8 }}>
              <label style={{ fontSize: 13, fontWeight: 600 }}>Dependent</label>
              <select
                value={rule.target}
                onChange={(e) => setConfig((prev) => writeDep(prev, rule.target, { ...rule, target: e.target.value }))}
              >
                {availableTargets(rule.target).map((r) => <option key={r} value={r}>{refLabel(r)}</option>)}
              </select>
              <span style={{ fontSize: 14 }}>=</span>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {rule.parents.map((par, pi) => (
                  <div key={pi} style={{ display: "flex", gap: 6, alignItems: "center" }}>
                    {pi > 0 && <span style={{ fontSize: 13 }}>+</span>}
                    <input
                      type="number" step="any" style={{ width: 90 }} value={par.weight}
                      onChange={(e) => setConfig((prev) => writeDep(prev, rule.target, {
                        ...rule,
                        parents: rule.parents.map((q, qi) => (qi === pi ? { ...q, weight: +e.target.value } : q)),
                      }))}
                    />
                    <span style={{ fontSize: 13 }}>×</span>
                    <select
                      value={par.ref}
                      onChange={(e) => setConfig((prev) => writeDep(prev, rule.target, {
                        ...rule,
                        parents: rule.parents.map((q, qi) => (qi === pi ? { ...q, ref: e.target.value } : q)),
                      }))}
                    >
                      {nodeRefs.filter((r) => r !== rule.target).map((r) => <option key={r} value={r}>{refLabel(r)}</option>)}
                    </select>
                    {rule.parents.length > 1 && (
                      <button
                        className="btn-danger" style={{ padding: "2px 8px", fontSize: 12 }}
                        onClick={() => setConfig((prev) => writeDep(prev, rule.target, {
                          ...rule, parents: rule.parents.filter((_, qi) => qi !== pi),
                        }))}
                      >×</button>
                    )}
                  </div>
                ))}
              </div>
              <span style={{ fontSize: 13 }}>+ offset</span>
              <input
                type="number" step="any" style={{ width: 90 }} value={rule.offset}
                onChange={(e) => setConfig((prev) => writeDep(prev, rule.target, { ...rule, offset: +e.target.value }))}
              />
            </div>
            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <button
                className="btn-secondary" style={{ padding: "3px 10px", fontSize: 12 }}
                onClick={() => setConfig((prev) => writeDep(prev, rule.target, {
                  ...rule,
                  parents: [...rule.parents, { ref: nodeRefs.find((r) => r !== rule.target) ?? nodeRefs[0], weight: 1 }],
                }))}
              >+ Parent</button>
              <label style={{ fontSize: 13 }}>Noise std (MC, optional)</label>
              <input
                type="number" step="any" style={{ width: 100 }} placeholder="none"
                value={rule.noise ?? ""}
                onChange={(e) => setConfig((prev) => writeDep(prev, rule.target, {
                  ...rule, noise: e.target.value === "" ? null : +e.target.value,
                }))}
              />
              <button
                className="btn-danger" style={{ marginLeft: "auto", padding: "3px 10px", fontSize: 12 }}
                onClick={() => setConfig((prev) => clearDep(prev, rule.target))}
              >Remove</button>
            </div>
          </div>
        ))}
        <button
          className="btn-primary" style={{ marginTop: 4 }}
          disabled={availableTargets(null).length === 0 || nodeRefs.length < 2}
          onClick={() => {
            const target = availableTargets(null)[0];
            const parent = nodeRefs.find((r) => r !== target);
            if (!target || !parent) return;
            setConfig((prev) => writeDep(prev, null, { target, parents: [{ ref: parent, weight: 1 }], offset: 0, noise: null }));
          }}
        >+ Add Dependency</button>
      </div>

      {/* Save */}
      <div style={{ textAlign: "right", marginTop: 8 }}>
        {saveError && <span style={{ color: "#e63946", marginRight: 12, fontSize: 13 }}>{saveError}</span>}
        <button className="btn-primary" style={{ padding: "10px 32px", fontSize: 15 }} onClick={save}>
          {saved ? "Saved!" : "Save Configuration"}
        </button>
      </div>
    </div>
  );
}
