import { useEffect, useState } from "react";
import {
  getEquipment, addEquipment, updateEquipment, deleteEquipment,
  getCostDBCategories, getProcessTypes, getMaterials,
} from "../api/client";
import type { EquipmentItem, EquipmentInput, CostDBEntry } from "../types";

const defaultInput: EquipmentInput = {
  name: "", param: null, process_type: "Fluids", category: "",
  type: null, material: null, target_year: 2024,
  purchased_cost: null, cost_year: null, num_units: null, cost_func: null,
};

// "belt_conveyor_seider_2013" → "Seider 2013", used to tell apart correlations
// that share a category + type but come from different sources.
const sourceLabel = (key: string): string | null => {
  const m = key.match(/_([a-z]+)_(\d{4}[a-z]?)$/i);
  return m ? `${m[1][0].toUpperCase()}${m[1].slice(1)} ${m[2]}` : null;
};

interface Props {
  setError: (e: string | null) => void;
  markDirty: () => void;
}

export default function EquipmentPage({ setError, markDirty }: Props) {
  const [items, setItems] = useState<EquipmentItem[]>([]);
  const [categories, setCategories] = useState<Record<string, CostDBEntry[]>>({});
  const [processTypes, setProcessTypes] = useState<string[]>([]);
  const [materials, setMaterials] = useState<string[]>([]);
  const [showModal, setShowModal] = useState(false);
  const [editIndex, setEditIndex] = useState<number | null>(null);
  const [form, setForm] = useState<EquipmentInput>({ ...defaultInput });
  const [param1, setParam1] = useState<number | null>(null);
  const [param2, setParam2] = useState<number | null>(null);
  const [useDirectCost, setUseDirectCost] = useState(false);
  const [modalError, setModalError] = useState<string | null>(null);

  const refresh = () => getEquipment().then(setItems).catch((e: unknown) => {
    setError(e instanceof Error ? e.message : "Failed to load equipment");
  });

  useEffect(() => {
    refresh();
    getCostDBCategories().then(setCategories);
    getProcessTypes().then(setProcessTypes);
    getMaterials().then(setMaterials);
  }, []);

  const selectedTypes = form.category ? (categories[form.category] || []) : [];
  // Prefer the pinned correlation key; fall back to the type name for
  // items created before cost_func was tracked (legacy saves, presets).
  const selectedEntry = form.cost_func
    ? selectedTypes.find((t) => t.key === form.cost_func)
    : form.type
      ? selectedTypes.find((t) => t.type === form.type)
      : undefined;
  const isTwoVar = selectedEntry != null && (selectedEntry.s2_lower != null || selectedEntry.s2_upper != null);
  // 2-var correlations describe both size parameters in one units string, e.g. "Width, in & length, ft"
  const [units1, units2] = (selectedEntry?.units || "").split(" & ");

  const openAdd = () => {
    setForm({ ...defaultInput });
    setParam1(null);
    setParam2(null);
    setEditIndex(null);
    setUseDirectCost(false);
    setModalError(null);
    setShowModal(true);
  };

  const openEdit = (item: EquipmentItem) => {
    setForm({
      name: item.name, param: item.param, process_type: item.process_type,
      category: item.category, type: item.type, material: item.material,
      num_units: item.num_units_input, cost_func: item.cost_func,
      target_year: item.target_year, purchased_cost: item.param === null ? item.purchased_cost : null,
      cost_year: item.cost_year,
    });
    setParam1(Array.isArray(item.param) ? item.param[0] : item.param);
    setParam2(Array.isArray(item.param) ? item.param[1] : null);
    setEditIndex(item.index);
    setUseDirectCost(item.param === null);
    setModalError(null);
    setShowModal(true);
  };

  const handleSubmit = async () => {
    try {
      const payload = { ...form };
      if (useDirectCost) {
        payload.param = null;
      } else {
        payload.param = isTwoVar ? [param1 ?? 0, param2 ?? 0] : param1;
        payload.purchased_cost = null;
        payload.cost_year = null;
      }
      if (editIndex !== null) {
        await updateEquipment(editIndex, payload);
      } else {
        await addEquipment(payload);
      }
      markDirty();
      setShowModal(false);
      refresh();
    } catch (e: unknown) {
      setModalError(e instanceof Error ? e.message : "Failed");
    }
  };

  const handleDelete = async (idx: number) => {
    await deleteEquipment(idx);
    markDirty();
    refresh();
  };

  const fmt = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 0 });

  return (
    <div>
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <h2>Equipment List</h2>
          <button className="btn-primary" onClick={openAdd}>+ Add Equipment</button>
        </div>
        {items.length === 0 ? (
          <p style={{ color: "#868e96" }}>No equipment added yet. Click "Add Equipment" to begin.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>#</th><th>Name</th><th>Category</th><th>Type</th>
                <th>Material</th><th>Process</th><th>Param</th><th>Units</th>
                <th>Purchased ($)</th><th>Direct ($)</th><th></th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.index}>
                  <td>{item.index + 1}</td>
                  <td>{item.name}</td>
                  <td>{item.category}</td>
                  <td>{item.type || "-"}</td>
                  <td>{item.material}</td>
                  <td>{item.process_type}</td>
                  <td className="number">{item.param == null ? "-" : Array.isArray(item.param) ? item.param.map(fmt).join(" × ") : fmt(item.param)}</td>
                  <td className="number">{item.num_units ?? 1}</td>
                  <td className="number">{fmt(item.purchased_cost)}</td>
                  <td className="number">{fmt(item.direct_cost)}</td>
                  <td>
                    <button className="btn-primary" style={{ marginRight: 4, padding: "4px 10px", fontSize: 12 }} onClick={() => openEdit(item)}>Edit</button>
                    <button className="btn-danger" onClick={() => handleDelete(item.index)}>Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr style={{ fontWeight: 600 }}>
                <td colSpan={8}>Total</td>
                <td className="number">{fmt(items.reduce((s, i) => s + i.purchased_cost, 0))}</td>
                <td className="number">{fmt(items.reduce((s, i) => s + i.direct_cost, 0))}</td>
                <td></td>
              </tr>
            </tfoot>
          </table>
        )}
      </div>

      {showModal && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>{editIndex !== null ? "Edit Equipment" : "Add Equipment"}</h2>
            {modalError && <div style={{ color: "#e63946", marginBottom: 12, fontSize: 13 }}>{modalError}</div>}
            <div className="form-grid">
              <div className="form-group">
                <label>Name</label>
                <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
              </div>
              <div className="form-group">
                <label>Category</label>
                <select value={form.category} onChange={(e) => setForm({ ...form, category: e.target.value, type: null, cost_func: null })}>
                  <option value="">-- Select --</option>
                  {Object.keys(categories).sort().map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label>Type</label>
                <select
                  value={selectedEntry?.key || ""}
                  onChange={(e) => {
                    const entry = selectedTypes.find((t) => t.key === e.target.value);
                    setForm({ ...form, type: entry?.type ?? null, cost_func: entry?.key ?? null });
                  }}
                >
                  <option value="">-- None --</option>
                  {selectedTypes.map((t) => {
                    const dup = selectedTypes.filter((o) => o.type === t.type).length > 1;
                    const src = dup ? sourceLabel(t.key) : null;
                    return (
                      <option key={t.key} value={t.key}>
                        {(t.type || t.key) + (src ? ` (${src})` : "")}
                      </option>
                    );
                  })}
                </select>
              </div>
              <div className="form-group">
                <label>Process Type</label>
                <select value={form.process_type} onChange={(e) => setForm({ ...form, process_type: e.target.value })}>
                  {processTypes.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label>Material</label>
                <select value={form.material ?? ""} onChange={(e) => setForm({ ...form, material: e.target.value || null })}>
                  <option value="">
                    Auto{selectedEntry?.default_material ? ` — ${selectedEntry.default_material}` : " (correlation default)"}
                  </option>
                  {materials.map((m) => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label>Target Year</label>
                <input type="number" value={form.target_year} onChange={(e) => setForm({ ...form, target_year: +e.target.value })} />
              </div>
              <div className="form-group">
                <label>Number of Units</label>
                <input
                  type="number" min={1} placeholder="auto"
                  value={form.num_units ?? ""}
                  onChange={(e) => setForm({ ...form, num_units: e.target.value ? +e.target.value : null })}
                />
              </div>
            </div>

            <div style={{ margin: "16px 0 8px" }}>
              <label style={{ fontSize: 13, cursor: "pointer" }}>
                <input type="checkbox" checked={useDirectCost} onChange={(e) => setUseDirectCost(e.target.checked)} style={{ marginRight: 6 }} />
                Use direct cost input (instead of size parameter)
              </label>
            </div>

            {useDirectCost ? (
              <div className="form-grid">
                <div className="form-group">
                  <label>Purchased Cost ($)</label>
                  <input type="number" value={form.purchased_cost ?? ""} onChange={(e) => setForm({ ...form, purchased_cost: e.target.value ? +e.target.value : null })} />
                </div>
                <div className="form-group">
                  <label>Cost Year</label>
                  <input type="number" value={form.cost_year ?? ""} onChange={(e) => setForm({ ...form, cost_year: e.target.value ? +e.target.value : null })} />
                </div>
              </div>
            ) : (
              <div className="form-grid">
                <div className="form-group">
                  <label>
                    Size Parameter
                    {selectedEntry && (
                      <span style={{ fontWeight: 400, textTransform: "none" }}>
                        {" "}({units1 || selectedEntry.units}, range: {selectedEntry.s_lower ?? "?"} - {selectedEntry.s_upper ?? "?"})
                      </span>
                    )}
                  </label>
                  <input type="number" value={param1 ?? ""} onChange={(e) => setParam1(e.target.value ? +e.target.value : null)} />
                </div>
                {isTwoVar && (
                  <div className="form-group">
                    <label>
                      2nd Size Parameter
                      {selectedEntry && (
                        <span style={{ fontWeight: 400, textTransform: "none" }}>
                          {" "}({units2 || ""}, range: {selectedEntry.s2_lower ?? "?"} - {selectedEntry.s2_upper ?? "?"})
                        </span>
                      )}
                    </label>
                    <input type="number" value={param2 ?? ""} onChange={(e) => setParam2(e.target.value ? +e.target.value : null)} />
                  </div>
                )}
              </div>
            )}

            <div className="modal-actions">
              <button className="btn-secondary" style={{ color: "#495057", borderColor: "#dee2e6" }} onClick={() => setShowModal(false)}>Cancel</button>
              <button className="btn-primary" onClick={handleSubmit}>{editIndex !== null ? "Update" : "Add"}</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
