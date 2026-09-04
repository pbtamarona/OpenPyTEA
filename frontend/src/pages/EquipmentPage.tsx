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

// Editable state for one (non-composite) equipment: the input spec plus the
// UI-only split of param into two fields and the cost-mode toggle.
interface Draft {
  form: EquipmentInput;
  param1: number | null;
  param2: number | null;
  useDirectCost: boolean;
}

const draftFromSpec = (spec: EquipmentInput): Draft => ({
  form: { ...defaultInput, ...spec },
  param1: Array.isArray(spec.param) ? spec.param[0] : spec.param ?? null,
  param2: Array.isArray(spec.param) ? spec.param[1] : null,
  useDirectCost: spec.param == null && spec.purchased_cost != null,
});

const emptyDraft = (): Draft => ({
  form: { ...defaultInput }, param1: null, param2: null, useDirectCost: false,
});

const findEntry = (
  categories: Record<string, CostDBEntry[]>, form: EquipmentInput,
): CostDBEntry | undefined => {
  const types = form.category ? (categories[form.category] || []) : [];
  // Prefer the pinned correlation key; fall back to the type name for
  // items created before cost_func was tracked (legacy saves, presets).
  if (form.cost_func) return types.find((t) => t.key === form.cost_func);
  if (form.type) return types.find((t) => t.type === form.type);
  return undefined;
};

const specFromDraft = (
  draft: Draft, categories: Record<string, CostDBEntry[]>,
): EquipmentInput => {
  const spec = { ...draft.form };
  if (draft.useDirectCost) {
    spec.param = null;
  } else {
    const entry = findEntry(categories, draft.form);
    const twoVar = entry != null && (entry.s2_lower != null || entry.s2_upper != null);
    spec.param = twoVar ? [draft.param1 ?? 0, draft.param2 ?? 0] : draft.param1;
    spec.purchased_cost = null;
    spec.cost_year = null;
  }
  return spec;
};

// Settings of the composite itself (components are kept separately).
interface CompositeForm {
  name: string;
  category: string;
  type: string;
  process_type: string;
  installation: string;
  num_units: number | null;
  target_year: number;
  useQuote: boolean;
  purchased_cost: number | null;
  cost_year: number | null;
}

const emptyComposite = (): CompositeForm => ({
  name: "", category: "Composite", type: "", process_type: "Fluids",
  installation: "component", num_units: null, target_year: 2024,
  useQuote: false, purchased_cost: null, cost_year: null,
});

const fmt = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 0 });

const paramLabel = (p: number | number[] | null | undefined): string =>
  p == null ? "-" : Array.isArray(p) ? p.map(fmt).join(" × ") : fmt(p);

// Shared field editor for one equipment (used by the main modal and by the
// component sub-editor inside a composite).
function EquipmentFields({ draft, setDraft, categories, processTypes, materials, showTargetYear }: {
  draft: Draft;
  setDraft: (d: Draft) => void;
  categories: Record<string, CostDBEntry[]>;
  processTypes: string[];
  materials: string[];
  showTargetYear: boolean;
}) {
  const { form } = draft;
  const setForm = (f: EquipmentInput) => setDraft({ ...draft, form: f });
  const selectedTypes = form.category ? (categories[form.category] || []) : [];
  const selectedEntry = findEntry(categories, form);
  const isTwoVar = selectedEntry != null && (selectedEntry.s2_lower != null || selectedEntry.s2_upper != null);
  // 2-var correlations describe both size parameters in one units string, e.g. "Width, in & length, ft"
  const [units1, units2] = (selectedEntry?.units || "").split(" & ");

  return (
    <>
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
        {showTargetYear && (
          <div className="form-group">
            <label>Target Year</label>
            <input type="number" value={form.target_year} onChange={(e) => setForm({ ...form, target_year: +e.target.value })} />
          </div>
        )}
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
          <input type="checkbox" checked={draft.useDirectCost} onChange={(e) => setDraft({ ...draft, useDirectCost: e.target.checked })} style={{ marginRight: 6 }} />
          Use direct cost input (instead of size parameter)
        </label>
      </div>

      {draft.useDirectCost ? (
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
            <input type="number" value={draft.param1 ?? ""} onChange={(e) => setDraft({ ...draft, param1: e.target.value ? +e.target.value : null })} />
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
              <input type="number" value={draft.param2 ?? ""} onChange={(e) => setDraft({ ...draft, param2: e.target.value ? +e.target.value : null })} />
            </div>
          )}
        </div>
      )}
    </>
  );
}

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
  const [modalError, setModalError] = useState<string | null>(null);

  const [isComposite, setIsComposite] = useState(false);
  const [draft, setDraft] = useState<Draft>(emptyDraft());
  const [composite, setComposite] = useState<CompositeForm>(emptyComposite());
  const [components, setComponents] = useState<EquipmentInput[]>([]);
  // When non-null the modal shows the component sub-editor instead of the
  // composite view; `index` is null while adding a new component.
  const [componentEdit, setComponentEdit] = useState<{ index: number | null } | null>(null);

  const refresh = () => getEquipment().then(setItems).catch((e: unknown) => {
    setError(e instanceof Error ? e.message : "Failed to load equipment");
  });

  useEffect(() => {
    refresh();
    getCostDBCategories().then(setCategories);
    getProcessTypes().then(setProcessTypes);
    getMaterials().then(setMaterials);
  }, []);

  const openAdd = () => {
    setDraft(emptyDraft());
    setComposite(emptyComposite());
    setComponents([]);
    setIsComposite(false);
    setComponentEdit(null);
    setEditIndex(null);
    setModalError(null);
    setShowModal(true);
  };

  const openEdit = (item: EquipmentItem) => {
    if (item.is_composite) {
      setComposite({
        name: item.name,
        category: item.category,
        type: item.type ?? "",
        process_type: item.process_type,
        installation: item.installation ?? "component",
        num_units: item.num_units_input,
        target_year: item.target_year,
        useQuote: item.quoted_purchased_cost != null,
        purchased_cost: item.quoted_purchased_cost,
        cost_year: item.quoted_cost_year,
      });
      setComponents((item.components ?? []).map((c) => c.spec));
      setIsComposite(true);
      setDraft(emptyDraft());
    } else {
      setDraft(draftFromSpec({
        name: item.name, param: item.param, process_type: item.process_type,
        category: item.category, type: item.type, material: item.material,
        num_units: item.num_units_input, cost_func: item.cost_func,
        target_year: item.target_year,
        purchased_cost: item.param === null ? item.purchased_cost : null,
        cost_year: item.cost_year,
      }));
      setComposite(emptyComposite());
      setComponents([]);
      setIsComposite(false);
    }
    setComponentEdit(null);
    setEditIndex(item.index);
    setModalError(null);
    setShowModal(true);
  };

  const handleSubmit = async () => {
    try {
      let payload: EquipmentInput;
      if (isComposite) {
        payload = {
          name: composite.name,
          category: composite.category || "Composite",
          type: composite.type || null,
          process_type: composite.process_type,
          material: null,
          param: null,
          installation: composite.installation,
          num_units: composite.num_units,
          target_year: composite.target_year,
          purchased_cost: composite.useQuote ? composite.purchased_cost : null,
          cost_year: composite.useQuote ? composite.cost_year : null,
          // Components must share the composite's target year (library rule)
          components: components.map((c) => ({ ...c, target_year: composite.target_year })),
        };
      } else {
        payload = specFromDraft(draft, categories);
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

  const openComponentAdd = () => {
    setDraft(emptyDraft());
    setComponentEdit({ index: null });
  };

  const openComponentEdit = (i: number) => {
    setDraft(draftFromSpec(components[i]));
    setComponentEdit({ index: i });
  };

  const saveComponent = () => {
    const spec = specFromDraft(draft, categories);
    if (componentEdit?.index != null) {
      setComponents(components.map((c, i) => (i === componentEdit.index ? spec : c)));
    } else {
      setComponents([...components, spec]);
    }
    setComponentEdit(null);
  };

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
                  <td>{item.material ?? "-"}</td>
                  <td>{item.process_type}</td>
                  <td className="number">
                    {item.is_composite
                      ? `${item.components?.length ?? 0} component${(item.components?.length ?? 0) === 1 ? "" : "s"}`
                      : paramLabel(item.param)}
                  </td>
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
            {componentEdit ? (
              <>
                <h2>{componentEdit.index != null ? "Edit Component" : "Add Component"}</h2>
                <p style={{ fontSize: 13, color: "#868e96", marginBottom: 12 }}>
                  Sub-component of <strong>{composite.name || "the composite"}</strong> — costed like ordinary
                  equipment, with its own correlation or direct cost.
                </p>
                <EquipmentFields
                  draft={draft} setDraft={setDraft}
                  categories={categories} processTypes={processTypes} materials={materials}
                  showTargetYear={false}
                />
                <div className="modal-actions">
                  <button className="btn-secondary" style={{ color: "#495057", borderColor: "#dee2e6" }} onClick={() => setComponentEdit(null)}>Back</button>
                  <button className="btn-primary" onClick={saveComponent}>
                    {componentEdit.index != null ? "Update Component" : "Add Component"}
                  </button>
                </div>
              </>
            ) : (
              <>
                <h2>{editIndex !== null ? "Edit Equipment" : "Add Equipment"}</h2>
                {modalError && <div style={{ color: "#e63946", marginBottom: 12, fontSize: 13 }}>{modalError}</div>}

                <div style={{ margin: "0 0 14px" }}>
                  <label style={{ fontSize: 13, cursor: "pointer" }}>
                    <input type="checkbox" checked={isComposite} onChange={(e) => setIsComposite(e.target.checked)} style={{ marginRight: 6 }} />
                    Composite equipment (package unit assembled from sub-components, e.g. a PSA skid or compressor train)
                  </label>
                </div>

                {!isComposite ? (
                  <EquipmentFields
                    draft={draft} setDraft={setDraft}
                    categories={categories} processTypes={processTypes} materials={materials}
                    showTargetYear={true}
                  />
                ) : (
                  <>
                    <div className="form-grid">
                      <div className="form-group">
                        <label>Name</label>
                        <input value={composite.name} onChange={(e) => setComposite({ ...composite, name: e.target.value })} />
                      </div>
                      <div className="form-group">
                        <label>Category (label)</label>
                        <input value={composite.category} onChange={(e) => setComposite({ ...composite, category: e.target.value })} />
                      </div>
                      <div className="form-group">
                        <label>Type (label)</label>
                        <input value={composite.type} onChange={(e) => setComposite({ ...composite, type: e.target.value })} />
                      </div>
                      <div className="form-group">
                        <label>Process Type</label>
                        <select value={composite.process_type} onChange={(e) => setComposite({ ...composite, process_type: e.target.value })}>
                          {processTypes.map((p) => <option key={p} value={p}>{p}</option>)}
                        </select>
                      </div>
                      <div className="form-group">
                        <label>Installation</label>
                        <select value={composite.installation} onChange={(e) => setComposite({ ...composite, installation: e.target.value })}>
                          <option value="component">Per component (each keeps its own factors)</option>
                          <option value="composite">As one item (composite's factors on the total)</option>
                        </select>
                      </div>
                      <div className="form-group">
                        <label>Number of Units</label>
                        <input
                          type="number" min={1} placeholder="1"
                          value={composite.num_units ?? ""}
                          onChange={(e) => setComposite({ ...composite, num_units: e.target.value ? +e.target.value : null })}
                        />
                      </div>
                      <div className="form-group">
                        <label>Target Year</label>
                        <input type="number" value={composite.target_year} onChange={(e) => setComposite({ ...composite, target_year: +e.target.value })} />
                      </div>
                    </div>

                    <div style={{ margin: "14px 0 8px" }}>
                      <label style={{ fontSize: 13, cursor: "pointer" }}>
                        <input type="checkbox" checked={composite.useQuote} onChange={(e) => setComposite({ ...composite, useQuote: e.target.checked })} style={{ marginRight: 6 }} />
                        Use a vendor quote for the whole package (components kept as breakdown only)
                      </label>
                    </div>
                    {composite.useQuote && (
                      <div className="form-grid">
                        <div className="form-group">
                          <label>Quoted Purchased Cost ($)</label>
                          <input type="number" value={composite.purchased_cost ?? ""} onChange={(e) => setComposite({ ...composite, purchased_cost: e.target.value ? +e.target.value : null })} />
                        </div>
                        <div className="form-group">
                          <label>Quote Year</label>
                          <input type="number" value={composite.cost_year ?? ""} onChange={(e) => setComposite({ ...composite, cost_year: e.target.value ? +e.target.value : null })} />
                        </div>
                      </div>
                    )}

                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", margin: "16px 0 8px" }}>
                      <label style={{ fontSize: 13, fontWeight: 600 }}>Components ({components.length})</label>
                      <button className="btn-secondary" style={{ padding: "4px 10px", fontSize: 12 }} onClick={openComponentAdd}>+ Add Component</button>
                    </div>
                    {components.length === 0 ? (
                      <p style={{ fontSize: 13, color: "#868e96" }}>No components yet — a composite needs at least one.</p>
                    ) : (
                      <table style={{ fontSize: 13 }}>
                        <thead>
                          <tr><th>Name</th><th>Category / Type</th><th>Sizing</th><th>Units</th><th></th></tr>
                        </thead>
                        <tbody>
                          {components.map((cItem, i) => (
                            <tr key={i}>
                              <td>{cItem.name}</td>
                              <td>{cItem.category}{cItem.type ? ` / ${cItem.type}` : ""}</td>
                              <td className="number">
                                {cItem.param != null
                                  ? paramLabel(cItem.param)
                                  : cItem.purchased_cost != null
                                    ? `$${fmt(cItem.purchased_cost)}${cItem.cost_year ? ` (${cItem.cost_year})` : ""}`
                                    : "-"}
                              </td>
                              <td className="number">{cItem.num_units ?? "auto"}</td>
                              <td>
                                <button className="btn-primary" style={{ marginRight: 4, padding: "2px 8px", fontSize: 12 }} onClick={() => openComponentEdit(i)}>Edit</button>
                                <button className="btn-danger" style={{ padding: "2px 8px", fontSize: 12 }} onClick={() => setComponents(components.filter((_, j) => j !== i))}>Remove</button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                  </>
                )}

                <div className="modal-actions">
                  <button className="btn-secondary" style={{ color: "#495057", borderColor: "#dee2e6" }} onClick={() => setShowModal(false)}>Cancel</button>
                  <button className="btn-primary" onClick={handleSubmit} disabled={isComposite && components.length === 0}>
                    {editIndex !== null ? "Update" : "Add"}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
