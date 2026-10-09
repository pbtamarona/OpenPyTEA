/** Hover text for the (?) buttons. Entries named after config keys
    ("plant.interest_rate", "equipment.param", "metric.NPV", ...) come from
    the user guide's reference tables via scripts/build_docs.py
    (docHelp.json — rerun it after editing those tables); GUI-only inputs
    are written here. `doc` is the docs page + section a click opens. */
import docHelp from "./docHelp.json";

export interface HelpEntry {
  text: string;
  doc?: string;
}

const EQUIP = "user_guide/equipment.html";
const ANALYSIS = "user_guide/analysis.html";
const MC_KEYS = `${ANALYSIS}#configuring-input-uncertainties`;

const GUI: Record<string, HelpEntry> = {
  "project.name": { text: "A name for the whole project (all its plants). Used as the default file name when saving." },
  "project.user": { text: "Who made the project. Defaults to your computer login; saved in the project file." },
  "project.created": { text: "Set when the project was started (New Project). For a project file saved before projects had details, the date it was saved." },
  "project.description": { text: "Free text: scope, assumptions, data sources. Saved in the project file." },
  "project.plants": {
    text: "Every plant of the project. Plant Config, Equipment, Results, Analysis and Monte Carlo work on the open plant (●); pick another with Open or the Plant dropdown.",
  },
  "plant.rate_uncertainty": {
    text: "Monte Carlo distribution of the operator hourly rate (nested rate_uncertainty block). Pick a family and its parameters; None samples nothing.",
    doc: MC_KEYS,
  },
  "plant.price_uncertainty": {
    text: "Monte Carlo distribution of the item's price (price_uncertainty). Default family is Normal with std and optional min/max bounds.",
    doc: MC_KEYS,
  },
  "plant.quantity_uncertainty": {
    text: "Monte Carlo distribution of the daily production / consumption (production_uncertainty / consumption_uncertainty).",
    doc: MC_KEYS,
  },
  "plant.dependencies": {
    text: "Tie a parameter to others: dependent = Σ weight × parent + offset. A dependent is never sampled or varied on its own; Monte Carlo, sensitivity and tornado propagate its parents through the graph.",
    doc: `${ANALYSIS}#dependencies`,
  },
  "plant.dependency_noise": {
    text: "Optional Monte Carlo noise: standard deviation of the scatter added around the dependent's implied value.",
    doc: `${ANALYSIS}#dependency-noise`,
  },
  "equipment.use_direct_cost": {
    text: "Enter a purchased cost (e.g. a vendor quote) instead of sizing the item with a cost correlation. The cost is inflated from its cost year to the target year with CEPCI.",
    doc: `${EQUIP}#the-equipment-class`,
  },
  "analysis.parameter": {
    text: "Input varied in the one-way sweep. Bare item names vary the price; '.consumption' / '.production' vary the daily quantity. Parameters set by a dependency are not offered.",
    doc: `${ANALYSIS}#one-way-sensitivity-analysis`,
  },
  "analysis.plus_minus": {
    text: "Range of the sweep as a fraction of the base value: 0.2 varies the parameter from −20 % to +20 %.",
    doc: `${ANALYSIS}#one-way-sensitivity-analysis`,
  },
  "analysis.points": {
    text: "Number of evaluation points across the sweep range (n_points).",
    doc: `${ANALYSIS}#one-way-sensitivity-analysis`,
  },
  "analysis.tornado_plus_minus": {
    text: "Each factor is moved up and down by this fraction of its base value (0.2 = ±20 %); bars show the resulting change in the metric.",
    doc: `${ANALYSIS}#tornado-diagram`,
  },
  "mc.num_samples": {
    text: "Number of Monte Carlo draws. Increase for accuracy, decrease for speed.",
    doc: `${ANALYSIS}#running-the-simulation`,
  },
  "mc.batch_size": {
    text: "Samples evaluated per batch (default 1000). Larger batches are faster but use more memory.",
    doc: `${ANALYSIS}#running-the-simulation`,
  },
};

const DOC = docHelp as Record<string, HelpEntry>;

/** "metric" help: every metric's description in one tooltip. */
function metricHelp(): HelpEntry | undefined {
  const lines = Object.entries(DOC)
    .filter(([k]) => k.startsWith("metric."))
    .map(([k, v]) => `${k.slice(7)}: ${v.text}`);
  if (lines.length === 0) return undefined;
  return { text: lines.join("\n"), doc: DOC["metric.LCOP"]?.doc };
}

export function getHelp(id: string): HelpEntry | undefined {
  if (id === "metric") return metricHelp();
  return GUI[id] ?? DOC[id];
}

