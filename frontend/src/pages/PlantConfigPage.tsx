import { useEffect, useState } from "react";
import { getPlantConfig, setPlantConfig, getLocations } from "../api/client";
import type { PlantConfig } from "../types";

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

export default function PlantConfigPage() {
  const [config, setConfig] = useState<PlantConfig>({ ...defaultConfig });
  const [locations, setLocations] = useState<Record<string, unknown>>({});
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    getPlantConfig().then((c) => {
      if (c && Object.keys(c).length > 0) setConfig(c as PlantConfig);
    }).catch(() => {});
    getLocations().then(setLocations).catch(() => {});
  }, []);

  const regions = (() => {
    const loc = locations[config.country];
    if (!loc) return [];
    if (typeof loc === "number") return [];
    return Object.keys(loc as Record<string, number>);
  })();

  const save = async () => {
    await setPlantConfig(config);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
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

  const updateProduct = (key: string, field: string, value: number) => {
    setConfig((prev) => ({
      ...prev,
      plant_products: {
        ...prev.plant_products,
        [key]: { ...prev.plant_products[key], [field]: value },
      },
    }));
  };

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
            <input type="number" step="0.1" value={config.operator_hourly_rate.std}
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
                  <td><strong>{key}</strong></td>
                  <td><input type="number" value={val.production ?? 0} onChange={(e) => updateProduct(key, "production", +e.target.value)} style={{ width: 100 }} /></td>
                  <td><input type="number" value={val.price ?? 0} onChange={(e) => updateProduct(key, "price", +e.target.value)} style={{ width: 80 }} /></td>
                  <td><input type="number" value={val.std ?? 0} onChange={(e) => updateProduct(key, "std", +e.target.value)} style={{ width: 60 }} /></td>
                  <td><input type="number" value={val.min ?? 0} onChange={(e) => updateProduct(key, "min", +e.target.value)} style={{ width: 60 }} /></td>
                  <td><input type="number" value={val.max ?? 99999} onChange={(e) => updateProduct(key, "max", +e.target.value)} style={{ width: 70 }} /></td>
                  <td><button className="btn-danger" onClick={() => removeProduct(key)}>Remove</button></td>
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
                  <td><strong>{key}</strong></td>
                  <td><input type="number" value={val.consumption ?? 0} onChange={(e) => updateVarOpex(key, "consumption", +e.target.value)} style={{ width: 100 }} /></td>
                  <td><input type="number" value={val.price ?? 0} onChange={(e) => updateVarOpex(key, "price", +e.target.value)} style={{ width: 80 }} /></td>
                  <td><input type="number" value={val.std ?? 0} onChange={(e) => updateVarOpex(key, "std", +e.target.value)} style={{ width: 60 }} /></td>
                  <td><input type="number" value={val.min ?? 0} onChange={(e) => updateVarOpex(key, "min", +e.target.value)} style={{ width: 60 }} /></td>
                  <td><input type="number" value={val.max ?? 99999} onChange={(e) => updateVarOpex(key, "max", +e.target.value)} style={{ width: 70 }} /></td>
                  <td><button className="btn-danger" onClick={() => removeVarOpex(key)}>Remove</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <button className="btn-primary" style={{ marginTop: 12 }} onClick={addVarOpex}>+ Add Item</button>
      </div>

      {/* Save */}
      <div style={{ textAlign: "right", marginTop: 8 }}>
        <button className="btn-primary" style={{ padding: "10px 32px", fontSize: 15 }} onClick={save}>
          {saved ? "Saved!" : "Save Configuration"}
        </button>
      </div>
    </div>
  );
}
