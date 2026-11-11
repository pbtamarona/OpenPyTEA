import difflib
import warnings
from typing import Literal

import numpy as np
import pandas as pd

# --- Fixed CSV data sources ---
from importlib.resources import files, as_file

data_dir = files("openpytea.data")

with as_file(
    data_dir / "cepci_values.csv"
) as CEPCI_CSV_PATH:
    CEPCI_DF = pd.read_csv(CEPCI_CSV_PATH).set_index("year")

with as_file(
    data_dir / "cost_correlations.csv"
) as COST_DB_PATH:
    COST_DB_DF = pd.read_csv(COST_DB_PATH)

# Valid names as type hints, so IDEs complete them and flag typos. Kept in
# sync with Equipment.process_factors, the CSV categories and types, and
# Equipment.material_factors (+ CSV default materials) by
# tests/test_equipment.py::test_literal_hints_match_the_data
ProcessType = Literal["Solids", "Fluids", "Mixed", "Electrical"]

Category = Literal[
    "Adsorption & membrane separation",
    "Agitators, blenders, & mixers",
    "Biomass combustion",
    "Boilers, heaters, & furnaces",
    "CO2 capture",
    "Centrifuges",
    "Compressors, fans, & blowers",
    "Conveyors",
    "Crushers",
    "Crystallizers",
    "Dryers",
    "Dust collectors",
    "Ejectors",
    "Electrolysis",
    "Evaporators",
    "Expression",
    "Filters",
    "Gas separation",
    "Heat exchangers",
    "Liquid-liquid extractors",
    "Motors & generators",
    "Packings & adsorbents",
    "Piping",
    "Pressure vessels",
    "Pumps",
    "Reactors",
    "Screens",
    "Size enlargement",
    "Solid-liquid separators",
    "Solids-handling systems",
    "Tanks",
    "Towers",
    "Trays",
    "Turbines",
    "Utilities",
    "Vaporizers",
]

# Type names across all categories, each listed once; which ones are
# valid depends on the category (see equipment_options(category))
EquipmentType = Literal[
    "Activated alumina",
    "Activated carbon",
    "Agitated falling film",
    "Agitated film (scraped wall)",
    "Agitator, propeller, closed vessel",
    "Agitator, propeller, open tank",
    "Agitator, turbine, closed vessel",
    "Agitator, turbine, open tank",
    "Agitator, turbine, top entry, closed",
    "Agitator, turbine, top entry, open",
    "Air blower",
    "Air cooler",
    "Air-cooled fin-fan",
    "API-fixed roof",
    "API-floating roof",
    "Apron",
    "Atmospheric suspended basket",
    "Atmospheric tray batch dryer",
    "Atmospheric, horizontal cylinder",
    "Auto batch separator",
    "Autoclave",
    "Autoclave, glass lined",
    "Autoclave, stainless steel",
    "Autoclave, steel",
    "Automatic batch, horizontal",
    "Axial gas turbine",
    "Axial gas turbines",
    "Axial tube fan",
    "Axial vane fan",
    "Backward curve fan",
    "Bag filters",
    "Baghouse",
    "Ball mill",
    "Batch",
    "Batch evaporative",
    "Batch tray dryer",
    "Batch, bottom-drive, vertical basket",
    "Batch, top-drive, vertical basket",
    "Bayonet",
    "Belt",
    "Belt, 0.5 m wide",
    "Belt, 1.0 m wide",
    "Belt, excluding motor",
    "Bent",
    "Berl saddles, ceramic, 1 in. ",
    "Berl saddles, ceramic, 1.5 in. ",
    "Berl saddles, ceramic, 2 in. ",
    "Bin",
    "Biogas upgrading, amine scrubber",
    "Biogas upgrading, membrane separation",
    "Biogas upgrading, organic physical scrubber",
    "Biogas upgrading, PSA",
    "Biogas upgrading, water scrubber",
    "Blower",
    "Blower, centrifugal, excluding motor",
    "Boiler, field erected  (10-70 bar)",
    "Boiler, packaged (15-40 bar)",
    "Bubble-cap",
    "Bucket elevator",
    "Bucket elevator, 0.5 m bucket",
    "Burner",
    "Cartridge",
    "Cascade minirings, ceramic, 1 in. ",
    "Cascade minirings, ceramic, 2 in. ",
    "Cascade minirings, ceramic, 3 in. ",
    "Cascade minirings, polypropylene, 1 in. ",
    "Cascade minirings, polypropylene, 2 in. ",
    "Cascade minirings, polypropylene, 3 in. ",
    "Cascade minirings, stainless steel, 1 in. ",
    "Cascade minirings, stainless steel, 1.5 in. ",
    "Cascade minirings, stainless steel, 2 in. ",
    "Cascade minirings, stainless steel, 3 in. ",
    "Cascade minirings, stainless steel, 4 in. ",
    "Centrifugal",
    "Centrifugal (turbo) blower",
    "Centrifugal compressor, electric motor drive",
    "Centrifugal compressor, gas turbine drive",
    "Centrifugal compressor, steam turbine drive",
    "Centrifugal radial fan",
    "Centrifugal separator",
    "Centrifugal, 1-stage, 1800 rpm, HSC",
    "Centrifugal, 1-stage, 1800 rpm, VSC",
    "Centrifugal, 1-stage, 3600 rpm, HSC",
    "Centrifugal, 1-stage, 3600 rpm, VSC",
    "Centrifugal, 2+-stage, 3600 rpm, HSC",
    "Centrifugal, 2-stage, 3600 rpm, HSC",
    "Centrifugal, backward curved fan, head 16-30 in. H2O",
    "Centrifugal, backward curved fan, head 31-40 in. H2O",
    "Centrifugal, backward curved fan, head 5-8 in. H2O",
    "Centrifugal, backward curved fan, head 9-15 in. H2O",
    "Centrifugal, backward curved fan, head less than 4 in. H2O",
    "Centrifugal, straight radial fan, head 16-30 in. H2O",
    "Centrifugal, straight radial fan, head 5-8 in. H2O",
    "Centrifugal, straight radial fan, head 9-15 in. H2O",
    "Centrifugal, straight radial fan, head less than 4 in. H2O",
    "CFB CHP",
    "Clarifier, concrete",
    "Clarifier, steel",
    "Cloth, shaker type, including motors",
    "CO2 compressor",
    "Complex network, FOB",
    "Complex network, INST",
    "Compressor, centrifugal",
    "Compressor, centrifugal / axial / reciprocating",
    "Compressor, reciprocating",
    "Compressor, reciprocating, including motor",
    "Compressor, rotary",
    "Condensing steam turbine",
    "Cone crusher",
    "Cone roof",
    "Cone, crusher only",
    "Cone-roof tank",
    "Continuous reciprocating pusher",
    "Continuous scroll solid bowl",
    "Cooling tower & pumps",
    "Cyclone scrubber",
    "Cyclones",
    "Demisters",
    "Diphenyl heater",
    "Direct contact rotary dryer",
    "Direct-heat rotary",
    "Disc and drum",
    "Disk agglomerator",
    "Distillation including internals (trays)",
    "Double pipe",
    "Double-pipe",
    "Dowtherm A heater",
    "Draft-tube baffled, continuous evaporative",
    "Drum",
    "Drum agglomerator",
    "Drum, excluding motor",
    "DSM",
    "Electric arc furnace",
    "Electric motor, explosion-proof enclosure, 1,800 rpm ",
    "Electric motor, explosion-proof enclosure, 3,600 rpm",
    "Electric motor, open drip-proof enclosure, 1,800 rpm",
    "Electric motor, open drip-proof enclosure, 3,600 rpm",
    "Electric motor, totally enclosed, fan-cooled, 1,800 rpm",
    "Electric motor, totally enclosed, fan-cooled, 3,600 rpm",
    "Electrolyzer (PEM / alkaline)",
    "Electrostatic precipitator",
    "Explosion-proof",
    "Explosion-proof motor",
    "External gear",
    "Falling film",
    "Feeder, belt",
    "Feeder, screw",
    "Feeder, vibratory",
    "Fermenter",
    "Fixed tube",
    "Flat plate",
    "Floating head",
    "Floating head shell & tube",
    "Floating roof",
    "Floating-roof tank",
    "Forced circulation",
    "Forced circulation (pumped)",
    "Forced circulation, continuous evaporative",
    "Furnace, box",
    "Furnace, cylindrical",
    "Gas expander, pressure discharge",
    "Gas expander, vacuum discharge",
    "Gas holder",
    "Gas multi-cyclone",
    "Gas permeation",
    "Gas turbine",
    "Gas-fired heater",
    "Glass-lined agitated",
    "Gravity",
    "Gyratory crusher",
    "H2 compressor",
    "Hammer mill",
    "Heat exchanger",
    "High-speed disk",
    "Horizontal",
    "Horizontal 304SS",
    "Horizontal auto-batch",
    "Horizontal CS",
    "Horizontal drum (150 psig)",
    "Horizontal tube",
    "Horizontal vessel",
    "Horizontal vessel platforms & ladders",
    "Hot water heater",
    "Hydroclone",
    "Hydrogen storage vessel, high pressure (860 bar), Type 1 steel",
    "Hydrogen storage vessel, high pressure (875 bar), Type 2",
    "Hydrogen storage vessel, high pressure (875 bar), Type 4",
    "Hydrogen storage vessel, low pressure (160 bar)",
    "Impeller mixer",
    "Indirect fluidized-bed",
    "Indirect-heat steam-tube rotary",
    "Inoculum tank",
    "Intalox saddles, ceramic, 1-1.5 in. ",
    "Intalox saddles, ceramic, 2 in. ",
    "Intalox saddles, ceramic, 3 in. ",
    "Intalox saddles, polypropylene, 1 in. ",
    "Intalox saddles, polypropylene, 2 in. ",
    "Intalox saddles, polypropylene, 3 in. ",
    "Internal coils/jackets",
    "Internal combustion engine",
    "Jacketed agitated",
    "Jacketed nonagitated",
    "Jacketed scraped wall, continuous cooling",
    "Jacketed vessels",
    "Jacketed, including mixer",
    "Jaw crusher",
    "Jaw, excluding motor",
    "Jaw, including motor",
    "Jet mill",
    "Kettle reboiler",
    "Kettle, jacketed, glass-lined",
    "Kneader blender",
    "Kneader, sigma double arm",
    "Kneader, tilting double arm",
    "Large centrifugal, including motor",
    "Large motors, ac induction, wound rotor, TEFC",
    "Leaf",
    "Liquid distributor",
    "Liquid expander",
    "Liquid-ring vacuum pump",
    "Long-tube",
    "Long-tube vertical (rising film)",
    "Loose, for towers",
    "MEA CCS technology",
    "Mixers/settler",
    "Molecular sieves",
    "Molten salt heater",
    "Molten salt, mineral & silicon oil heater",
    "Muller mixer",
    "Multicyclones",
    "Multicyclones at 40 celcius",
    "Multiple pipe",
    "Multistage, including condenser and piping",
    "Nonreactive fired heater",
    "One-stage jet",
    "Open",
    "Open/dried-proof",
    "Oscillating screen",
    "Packaged mechanical refrigerator",
    "Pall rings, carbon steel, 1 in. ",
    "Pall rings, carbon steel, 1.5 in. ",
    "Pall rings, carbon steel, 2 in. ",
    "Pall rings, polypropylene, 1 in. ",
    "Pall rings, polypropylene, 1.5 in. ",
    "Pall rings, polypropylene, 2 in. ",
    "Pall rings, polypropylene, 3 in. ",
    "Pall rings, stainless steel, 1 in. ",
    "Pall rings, stainless steel, 1.5 in. ",
    "Pall rings, stainless steel, 2 in. ",
    "Pan",
    "Pellet mill",
    "PEM electrolyzer, current (2019)",
    "PEM electrolyzer, future (2035)",
    "Pervaporation",
    "Plate & fin 3-streams",
    "Plate & fin 4-to-6-streams",
    "Plate & frame",
    "Plate and frame",
    "Platform & ladders for towers",
    "Pneumatic",
    "Positive displacement",
    "Pressure leaf",
    "Propeller fan, head less than 4 in. H2O",
    "Propeller mixer",
    "Pug mill extruder",
    "Pulverizer",
    "Pyrolysis furnace",
    "Radial expander",
    "Radial gas/liquid expanders",
    "Raschig rings, carbon steel, 1-1.5 in. ",
    "Raschig rings, carbon steel, 2-3 in. ",
    "Raschig rings, ceramic, 1-3 in. ",
    "Raschig rings, stainless steel, 1 in. ",
    "Raschig rings, stainless steel, 1.5 in. ",
    "Raschig rings, stainless steel, 2-3 in. ",
    "Reciprocating",
    "Reciprocating compressor, electric motor drive",
    "Reciprocating compressor, gas turbine drive",
    "Reciprocating compressor, steam turbine drive",
    "Reciprocating plunger",
    "Reformer furnace",
    "Refrigeration, package mechanical",
    "Reverse osmosis, brackish water",
    "Reverse osmosis, seawater",
    "Reversible hammer mill",
    "Ribbon blender",
    "Ribbon mixer",
    "Roll presses",
    "Roll-type press",
    "Rotary",
    "Rotary blender",
    "Rotary pan",
    "Rotary straight-lobe blower",
    "Rotary, gas dryer",
    "Rotary-drum vacuum",
    "Rotating-disk contactor (RDC)",
    "Scraped surface",
    "Scraped wall",
    "Screw",
    "Screw compressor (vacuum service)",
    "Screw compressor, electric motor drive",
    "Screw compressor, gas turbine drive",
    "Screw compressor, steam turbine drive",
    "Screw extruder",
    "Screw presses",
    "Screw, excluding motor",
    "Shell & tube, fixed head",
    "Shell & tube, floating head",
    "Shell & tube, kettle-vaporizer",
    "Shell & tube, U-tube",
    "Shell-tube, fixed tube",
    "Shell-tube, floating head",
    "Shell-tube, kettle",
    "Shell-tube, U-tube",
    "Short-tube",
    "Sieve",
    "Silica gel",
    "Single stage, 100 psig, steam",
    "Single-stage centrifugal",
    "Small centrifugal, excluding motor",
    "Small motors, ac induction, wound rotor, TEFC",
    "Solid bowl, without motor",
    "Spherical, 0-30 psig",
    "Spherical, 30-200 psig",
    "Spiral plate",
    "Spiral ribbon mixer",
    "Spiral tube",
    "Spray",
    "Spray dryer",
    "Static mixer",
    "Stationary",
    "Steam boiler",
    "Steam turbine",
    "Steam turbine, condensing",
    "Steam turbine, noncondensing",
    "Structured packing, corrugated-sheet, stainless steel",
    "Table",
    "Tableting press",
    "Teflon tube",
    "Tellerettes, polyethylene, 1 in. ",
    "Thermal screw, including motor",
    "Thermosiphon reboiler",
    "Thickener, concrete",
    "Thickener, steel",
    "Three-stage claw vacuum pump",
    "Three-stage lobe vacuum pump",
    "Totally enclosed",
    "Totally enclosed motor",
    "Tray",
    "Tray and packed",
    "Tube",
    "Tubular fixed bed",
    "Tumbler, double cone",
    "Tumbler, twin shell",
    "Turbine mixer",
    "Two-stage, including condenser and piping",
    "Typical straight run, FOB",
    "Typical straight run, INST",
    "U-tube",
    "U-tube kettle reboiler",
    "U-tube shell & tube",
    "Ultrafiltration (high estimate)",
    "Ultrafiltration (low estimate)",
    "Vacuum drum",
    "Vacuum rotary drum, including motor",
    "Vacuum, shelf, excluding trays and vacuum equipment",
    "Valve",
    "Vane-axial fan, head 5-8 in. H2O",
    "Vane-axial fan, head 9-15 in. H2O",
    "Vane-axial fan, head less than 4 in. H2O",
    "Venturi scrubber",
    "Vertical",
    "Vertical 304SS",
    "Vertical agitated, excluding motor",
    "Vertical auto-batch",
    "Vertical CS",
    "Vertical cylinder",
    "Vertical jacketed",
    "Vertical pressure vessels for separation",
    "Vertical tube",
    "Vertical vessel",
    "Vertical vessel platforms & ladders",
    "Vertical-pressure leaf",
    "Vibrating",
    "Vibrating grizzly",
    "Vibrating screen, 1 deck",
    "Vibrating screen, 2 decks",
    "Vibrating screen, 3 decks",
    "Vibrating, single-deck, including motor",
    "Vibratory",
    "Wastewater treatment, primary",
    "Wastewater treatment, primary + secondary",
    "Wastewater treatment, primary + secondary + tertiary",
    "Water ion exchange plant",
    "Wet classifier (rake & spiral)",
]

Material = Literal[
    # in Equipment.material_factors
    "Carbon steel",
    "Aluminum",
    "Bronze",
    "Cast steel",
    "Stainless steel",
    "304 stainless steel",
    "316 stainless steel",
    "316 stainless steel (machined)",
    "Glass-reinforced plastic",
    "321 stainless steel",
    "Hastelloy C",
    "Monel",
    "Nickel",
    "Inconel",
    "316 stainless steel (welded)",
    "Exotic material (machined)",
    "Exotic material (welded)",
    # only valid as a correlation's own default material
    "ASME Type 1 steel",
    "Carbon-fibre hoop-wrapped steel",
    "Cast iron",
    "Ceramic",
    "Ductile iron",
    "Polyethylene",
    "Polymer liner with composite overwrap",
    "Polypropylene",
    "Stainless steel + Carbon steel",
]

# Renamed categories: old name (lowercase, as lookups ignore case) -> current
_CATEGORY_ALIASES = {
    "compressors & blowers": "Compressors, fans, & blowers",
}


def _current_category(category):
    """Map a renamed category to its current name, with a warning."""
    new = _CATEGORY_ALIASES.get(str(category).lower())
    if new is not None:
        warnings.warn(
            f"Category '{category}' has been renamed to '{new}'; "
            "the old name will be removed in a future release.",
            FutureWarning,
            stacklevel=3,
        )
        return new
    return category


def _unknown(what, value, valid, scope="", see=None):
    """Error text for a bad name, suggesting the closest valid ones."""
    valid = list(valid)
    text = str(value).lower()
    # "gear" -> "External gear": substring first, then fuzzy matching
    hits = [v for v in valid if text in v.lower()] or difflib.get_close_matches(
        str(value), valid, n=3, cutoff=0.5)
    msg = f"Unknown {what} {value!r}{scope}."
    if hits:
        return msg + " Did you mean " + " or ".join(map(repr, hits[:3])) + "?"
    if see:
        return msg + f" See {see}."
    return msg + f" Valid options are: {valid}"


def equipment_options(category=None):
    """
    List the equipment the bundled cost database covers.

    Parameters
    ----------
    category : str | None, optional
        A category name (case-insensitive). Default is None.

    Returns
    -------
    list[str] | pd.DataFrame
        Without ``category``: the sorted category names. With it: one row
        per type in that category, with the size parameter's units and
        valid range, the default material and the source.

    Raises
    ------
    KeyError
        If ``category`` is not in the database.

    Examples
    --------
    >>> equipment_options()[:2]
    ['Adsorption & membrane separation', 'Agitators, blenders, & mixers']
    >>> equipment_options("Pumps")["type"].head(1).tolist()
    ['Centrifugal']
    """
    df = CostCorrelationDB().df
    if category is None:
        return sorted(df["category"].dropna().unique())
    category = _current_category(category)
    rows = df[df["category"].str.lower() == str(category).lower()]
    if rows.empty:
        raise KeyError(_unknown("category", category, df["category"].unique(),
                                see="openpytea.equipment_options()"))
    columns = ["type", "units", "s_lower", "s_upper", "s2_lower", "s2_upper",
               "default material", "cost_year", "source", "key"]
    return rows[columns].reset_index(drop=True)


def inflation_adjustment(equipment_cost, cost_year, target_year=2024):
    """
    Adjust equipment cost from one year to another using the Chemical
    Engineering Plant Cost Index (CEPCI).

    This function uses historical CEPCI values to convert equipment costs
    between different years, accounting for inflation in the chemical
    engineering industry.

    Parameters
    ----------
    equipment_cost : float
        The cost of the equipment in the cost_year (in USD).
    cost_year : int
        The year in which the equipment_cost is valued.
        Must be available in CEPCI_DF index.
    target_year : int, optional
        The year to adjust the cost to. Default is 2024.
        Must be available in CEPCI_DF index.

    Returns
    -------
    float
        The inflation-adjusted equipment cost in target_year (in USD).

    Raises
    ------
    ValueError
        If cost_year is not found in CEPCI_DF.
    ValueError
        If target_year is not found in CEPCI_DF.

    Notes
    -----
    The adjustment factor is calculated as:
    adjusted_cost = equipment_cost * (CEPCI[target_year] / CEPCI[cost_year])

    Examples
    --------
    >>> # Adjust from 2015 to 2023
    >>> new_cost = inflation_adjustment(50000, 2015, 2023)
    """
    if cost_year not in CEPCI_DF.index:
        raise ValueError(
            f"CEPCI not available for year {cost_year}"
        )
    if target_year not in CEPCI_DF.index:
        raise ValueError(
            f"CEPCI not available for target year {target_year}"
        )
    return float(equipment_cost) * (
        CEPCI_DF.loc[target_year, "cepci"]
        / CEPCI_DF.loc[cost_year, "cepci"]
    )


class CostCorrelationDB:
    """
    Database interface for equipment cost correlations.

    Manages cost estimation correlations for equipment based on size/capacity
    parameters. Supports multiple correlation forms (offset power-law,
    log-log quadratic, ln-ln quartic, power-sizing, 2-var power-law) and
    handles equipment parallelization when capacity limits are exceeded.

    Attributes
    ----------
    df : pd.DataFrame
        Cost correlation data with columns: key, category, type, form,
        s_lower, s_upper, s2_lower, s2_upper, upper_parallel, a, b, n, n2,
        k1, k2, k3, k4, k5, s0, c0, f, cost_year.
    """

    def __init__(self, df=COST_DB_DF):
        """
        Initialize database with cost correlation DataFrame.

        Normalizes column names to lowercase and converts numeric columns.

        Parameters
        ----------
        df : pd.DataFrame
            Cost correlation data. Defaults to the bundled CSV database.
            The input is not modified; normalization happens on a copy.
        """
        # Never normalize in place: df defaults to the shared
        # module-level database, and a user-supplied frame must not be
        # mutated as a construction side effect
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        for col in [
            "s_lower",
            "s_upper",
            "s2_lower",
            "s2_upper",
            "upper_parallel",
            "a",
            "b",
            "n",
            "n2",
            "s0",
            "c0",
            "f",
            "cost_year",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col], errors="coerce"
                )
        df["form"] = df["form"].str.lower()
        self.df = df

    def evaluate(self, key: str, s: float, s2: float | None = None):
        """
        Calculate purchased equipment cost based on correlation key and size.

        Parameters
        ----------
        key : str
            Unique identifier for the cost correlation.
        s : float
            Equipment size/capacity parameter.
        s2 : float | None, optional
            Second size/capacity parameter, required by two-parameter
            correlation forms such as ``"2-var power-law"``. Validated
            against ``s2_lower``/``s2_upper`` but never parallelized.
            Default is None.

        Returns
        -------
        tuple[float, int, int]
            (total_cost, number_of_units, cost_year).

        Raises
        ------
        KeyError
            If correlation key not found in database.
        ValueError
            If size (or ``s2``) is outside its valid bounds, the
            correlation form is unsupported, or the form requires ``s2``
            and none was given.
        """
        row = self.df.loc[self.df["key"] == key]
        if row.empty:
            raise KeyError(
                f"Correlation key not found in CSV: {key}"
            )
        r = row.iloc[0].to_dict()

        s_lower = r.get("s_lower")
        s_upper = r.get("s_upper")
        cap = (
            r.get("upper_parallel")
            if pd.notna(r.get("upper_parallel"))
            else s_upper
        )

        if pd.notna(s_lower) and s < s_lower:
            raise ValueError(
                f"s={s} below lower bound {s_lower} for key '{key}'"
            )

        if s2 is not None:
            s2_lower = r.get("s2_lower")
            s2_upper = r.get("s2_upper")
            if pd.notna(s2_lower) and s2 < s2_lower:
                raise ValueError(
                    f"s2={s2} below lower bound {s2_lower} for key '{key}'"
                )
            if pd.notna(s2_upper) and s2 > s2_upper:
                raise ValueError(
                    f"s2={s2} above upper bound {s2_upper} for key '{key}'"
                )

        # Split a size above the unit capacity into equal parallel units
        units, s_adj = 1, s
        if pd.notna(cap) and s > cap:
            units = int(np.ceil(s / cap))
            s_adj = s / units
        form = r.get("form", "linear")
        year = int(r["cost_year"])

        if form == "offset power-law":
            a, b, n = r["a"], r["b"], r["n"]
            ce = a + b * (s_adj**n)

        elif form == "exponential":
            a, b = r["a"], r["b"]
            ce = a * np.exp(b * s_adj)

        elif form in ("log-log quadratic", "ln-ln quadratic"):
            # log(Ce) is a polynomial (up to 4th order) in log(S), base 10
            # or e
            log, unlog = (
                (np.log10, lambda x: 10**x) if form == "log-log quadratic"
                else (np.log, np.exp)
            )
            K1, K2, K3 = r["k1"], r["k2"], r["k3"]
            K4 = r.get("k4") if pd.notna(r.get("k4")) else 0.0
            K5 = r.get("k5") if pd.notna(r.get("k5")) else 0.0

            logS = log(s_adj)
            ce = unlog(
                K1
                + K2 * logS
                + K3 * (logS**2)
                + K4 * (logS**3)
                + K5 * (logS**4)
            )

        elif form == "power-sizing":
            C0, S0, f = r["c0"], r["s0"], r["f"]
            ce = C0 * (s_adj / S0) ** f

        elif form == "2-var power-law":
            if s2 is None:
                raise ValueError(
                    f"Correlation '{key}' has form '2-var "
                    f"power-law' and requires a second size parameter "
                    f"(s2)."
                )
            a, b, n1, n2 = r["a"], r["b"], r["n"], r["n2"]
            ce = a + b * (s_adj**n1) * (s2**n2)

        else:
            raise ValueError(
                f"Unsupported form '{form}' for key '{key}'"
            )

        purchased = ce * units
        return float(purchased), int(units), year

    def key_for_category_type(
        self, eq_category: str, type: str | None
    ):
        """
        Look up correlation key by equipment category and optional type.

        Parameters
        ----------
        eq_category : str
            Equipment category name.
        type : str | None
            Equipment sub-type (optional).

        Returns
        -------
        str | None
            Correlation key if found, None otherwise.
        """
        t = eq_category.lower()
        st = type.lower() if type else ""
        df = self.df

        if "category" not in df.columns:
            return None

        cand = df[df["category"].str.lower() == t]
        if "type" in df.columns:
            cand = cand[
                cand["type"].fillna("").str.lower() == st
            ]

        if cand.empty:
            return None

        return cand.iloc[0]["key"]

    def default_material_for_key(self, key: str) -> str | None:
        """
        Look up the cost basis's default construction material for a key.

        Parameters
        ----------
        key : str
            Correlation key.

        Returns
        -------
        str | None
            The ``default material`` value for the row, or None if the
            key is not found, the column is absent, or the value is
            unset (e.g. ``"n.a."``).
        """
        if "default material" not in self.df.columns:
            return None
        row = self.df.loc[self.df["key"] == key]
        if row.empty:
            return None
        val = row.iloc[0]["default material"]
        if pd.isna(val) or str(val).strip().lower() == "n.a.":
            return None
        return str(val).strip()


# Installation factor attribute -> key in Equipment.process_factors
_FACTOR_KEYS = {
    "erection_factor": "fer",
    "piping_factor": "fp",
    "instrumentation_factor": "fi",
    "electrical_factor": "fel",
    "civil_factor": "fc",
    "structural_factor": "fs",
    "lagging_factor": "fl",
}


def _set_install_factors(obj, process_factors, **given):
    """
    Set the seven installation factors on ``obj``: a given (non-None)
    value wins, else the process type's default from ``process_factors``.
    """
    for attr, key in _FACTOR_KEYS.items():
        value = given[attr]
        setattr(obj, attr, process_factors[key] if value is None else value)


class Equipment:
    """
    Equipment cost estimation class for process equipment.

    Manages cost calculation of process equipment based on process type,
    material, and equipment parameters. Supports both direct cost input and
    calculated costs from a cost correlation database.

    Attributes
    ----------
    process_factors : dict
        Process type factors affecting cost calculation.
        Keys are process types ("Solids", "Fluids", "Mixed", "Electrical").
        Values are dicts with factors: fer, fp, fi, fel, fc, fs, fl.
    material_factors : dict
        Material type multipliers mapping material names to cost factors
        (1.0 to 2.5).

    Parameters
    ----------
    name : str
        Equipment identifier/name.
    param : float | tuple[float, float] | list[float]
        Equipment parameter (size, capacity) for cost correlation lookup,
        per unit when ``num_units`` is given. Pass a 2-element tuple/list
        ``(s1, s2)`` for two-parameter correlation forms such as
        ``"2-var power-law"``.
    process_type : ProcessType
        Type of process ("Solids", "Fluids", "Mixed", or "Electrical").
    category : Category
        Equipment category for database lookup; see
        :func:`equipment_options` for the list.
    type : EquipmentType | None, optional
        Equipment sub-type for database lookup; see
        ``equipment_options(category)``. Default is None.
    material : Material | None, optional
        Material of construction. Default is None, which uses the
        resolved cost correlation's ``default material`` column,
        falling back to "Carbon steel" if there's no correlation match
        or the value is unset (e.g. "n.a."). Since the correlation's
        cost already prices in whatever material it defaults to, a
        material that matches the resolved default (whether auto-filled
        or passed explicitly) always uses a material factor of 1.0,
        even if it's recognized in ``material_factors`` (e.g. "304
        stainless steel"). Passing a different material instead uses
        ``material_factors[material] / material_factors[default]``
        (the default's own factor, or 1.0 if it has none) so the factor
        is relative to the correlation's actual cost basis rather than
        double-counting it, and raises ValueError if the material isn't
        found in ``material_factors``.
    num_units : int | None, optional
        Number of identical units. With a cost correlation, ``param`` is
        the size of one unit and the correlation cost is multiplied by
        ``num_units``. With a direct ``purchased_cost`` it is a label
        only: the given cost is taken as the total for all units.
        Default is None: 1 for a direct ``purchased_cost``, or, for a
        correlation, the number of parallel units the database splits
        ``param`` into when it exceeds the correlation's capacity (that
        cost already covers all of them).
    purchased_cost : float | None, optional
        Direct purchased cost input, the total for all ``num_units``.
        If provided, param is ignored. Default is None.
    cost_year : int | None, optional
        Year of the purchased_cost quote for inflation adjustment.
        Default is None.
    cost_func : str | None, optional
        Explicit cost correlation key from the database.
        Default is None (auto-resolved from category/type).
    target_year : int, optional
        Target year for inflation adjustment. Default is 2024.
    erection_factor : float | None, optional
        Erection factor override. Default is None (use process_type table).
    piping_factor : float | None, optional
        Piping factor override. Default is None (use process_type table).
    instrumentation_factor : float | None, optional
        Instrumentation & controls factor override. Default is None.
    electrical_factor : float | None, optional
        Electrical factor override. Default is None (use process_type table).
    civil_factor : float | None, optional
        Civil factor override. Default is None (use process_type table).
    structural_factor : float | None, optional
        Structural steel factor override. Default is None (use process_type table).
    lagging_factor : float | None, optional
        Lagging & painting factor override. Default is None
        (use process_type table).
    material_factor : float | None, optional
        Material factor override. Default is None (1.0 if ``material``
        matches the resolved default material, else the ratio of the
        two materials' ``material_factors`` values).

    Raises
    ------
    ValueError
        If process_type or material is not found in the factor dictionaries.
    KeyError
        If the category/type combination is not found in the database and
        cost_func is not specified.

    Examples
    --------
    >>> eq = Equipment(
    ...     name="Reactor",
    ...     param=10,  # volume, m^3
    ...     process_type="Fluids",
    ...     category="Reactors",
    ...     type="Jacketed agitated",
    ...     material="304 stainless steel"
    ... )
    >>> print(eq.direct_cost)
    """

    process_factors = {
        "Solids": {
            "fer": 0.6,
            "fp": 0.2,
            "fi": 0.2,
            "fel": 0.15,
            "fc": 0.2,
            "fs": 0.1,
            "fl": 0.05,
        },
        "Fluids": {
            "fer": 0.3,
            "fp": 0.8,
            "fi": 0.3,
            "fel": 0.2,
            "fc": 0.3,
            "fs": 0.2,
            "fl": 0.1,
        },
        "Mixed": {
            "fer": 0.5,
            "fp": 0.6,
            "fi": 0.3,
            "fel": 0.2,
            "fc": 0.3,
            "fs": 0.2,
            "fl": 0.1,
        },
        "Electrical": {
            "fer": 0.4,
            "fp": 0.1,
            "fi": 0.7,
            "fel": 0.7,
            "fc": 0.2,
            "fs": 0.1,
            "fl": 0.1,
        },
    }

    material_factors = {
        "Carbon steel": 1.0,
        "Aluminum": 1.07,
        "Bronze": 1.07,
        "Cast steel": 1.1,
        "Stainless steel": 1.3,
        "304 stainless steel": 1.3,
        "316 stainless steel": 1.3,
        "316 stainless steel (machined)": 1.3,
        "Glass-reinforced plastic": 1.4,
        "321 stainless steel": 1.5,
        "Hastelloy C": 1.55,
        "Monel": 1.65,
        "Nickel": 1.7,
        "Inconel": 1.7,
        "316 stainless steel (welded)": 1.75,
        # EDF (Aromada et al. 2021) doesn't define "exotic
        # material", so titanium or zirconium aren't
        # mapped to it
        "Exotic material (machined)": 1.75,
        "Exotic material (welded)": 2.5,
    }

    def __init__(
        self,
        name: str,
        param: float | tuple[float, float] | list[float],
        process_type: ProcessType,
        category: Category,
        type: EquipmentType | None = None,
        material: Material | None = None,
        num_units: int | None = None,
        purchased_cost: float | None = None,
        cost_year: int | None = None,
        cost_func: str | None = None,
        target_year: int = 2024,
        erection_factor: float | None = None,
        piping_factor: float | None = None,
        instrumentation_factor: float | None = None,
        electrical_factor: float | None = None,
        civil_factor: float | None = None,
        structural_factor: float | None = None,
        lagging_factor: float | None = None,
        material_factor: float | None = None,
    ):
        """Initialize equipment and compute purchased and direct costs."""
        self.name = name
        self.process_type = process_type
        self.param = (
            None if purchased_cost is not None else param
        )
        self.category = _current_category(category)
        self.type = type
        self.num_units = num_units
        self.cost_year = cost_year
        self.target_year = target_year
        self._cost_func = cost_func
        self._db = CostCorrelationDB()

        resolved_default_material = (
            self._default_material() or "Carbon steel"
        )
        material_was_auto = material is None
        if material_was_auto:
            material = resolved_default_material
        self.material = material
        uses_default_material = (
            material_was_auto or material == resolved_default_material
        )

        if process_type not in self.process_factors:
            raise ValueError(
                _unknown("process_type", process_type, self.process_factors)
            )
        if (
            material not in self.material_factors
            and not uses_default_material
        ):
            raise ValueError(
                _unknown("material", material, self.material_factors)
            )

        _set_install_factors(
            self, self.process_factors[process_type],
            erection_factor=erection_factor,
            piping_factor=piping_factor,
            instrumentation_factor=instrumentation_factor,
            electrical_factor=electrical_factor,
            civil_factor=civil_factor,
            structural_factor=structural_factor,
            lagging_factor=lagging_factor,
        )
        if material_factor is not None:
            self.material_factor = material_factor
        elif uses_default_material:
            # The correlation's own default material is already priced
            # into its cost, so no fm adjustment applies here.
            self.material_factor = 1.0
        else:
            # material_factors is calibrated against a carbon steel
            # base, but the correlation's cost is already priced for
            # resolved_default_material. Dividing out that material's
            # own factor (1.0 if it has none, e.g. "Cast iron") rescales
            # the target factor to be relative to the correlation's
            # actual cost basis instead of double-counting it.
            default_fm = self.material_factors.get(
                resolved_default_material, 1.0
            )
            self.material_factor = (
                self.material_factors[material] / default_fm
            )

        if purchased_cost is not None:
            self.purchased_cost = purchased_cost
            if cost_year is not None:
                self.purchased_cost = inflation_adjustment(
                    purchased_cost,
                    cost_year,
                    target_year=self.target_year,
                )
            if self.num_units is None:
                self.num_units = 1
        else:
            self.purchased_cost = (
                self._calc_purchased_cost()
            )
        self.direct_cost = (
            self.calculate_direct_cost()
        # ponytail: stale comment below
        )  # your existing method

    def _resolve_key(self) -> str:
        """
        Resolve the cost correlation key from the database or explicit input.

        Returns
        -------
        str
            Cost correlation key to use for cost evaluation.

        Raises
        ------
        KeyError
            If no database entry matches the equipment's category and type,
            and no explicit cost_func was provided.
        """
        if self._cost_func:
            return self._cost_func

        key = self._db.key_for_category_type(
            self.category, self.type
        )
        if key is None:
            df = self._db.df
            in_cat = df[df["category"].str.lower() == str(self.category).lower()]
            if in_cat.empty:
                msg = _unknown("category", self.category, df["category"].unique(),
                               see="openpytea.equipment_options()")
            else:
                msg = _unknown("type", self.type, in_cat["type"].dropna(),
                               f" for category '{self.category}'",
                               see=f"openpytea.equipment_options({self.category!r})")
            raise KeyError(
                msg + " Or add a row to the CSV, or specify cost_func manually."
            )
        return key

    def _default_material(self) -> str | None:
        """
        Resolve the default construction material from the correlation DB.

        Returns
        -------
        str | None
            The resolved correlation's ``default material`` value, or
            None if there is no correlation match or the value is unset
            (e.g. "n.a."). Names not recognized in ``material_factors``
            (e.g. "Cast iron", "Ceramic") are still returned as-is; the
            caller applies a material factor of 1.0 for those.
        """
        try:
            key = self._resolve_key()
        except KeyError:
            return None
        return self._db.default_material_for_key(key)

    def _calc_purchased_cost(self) -> float:
        """
        Calculate purchased cost using the database correlation.

        Resolves the correlation key, evaluates the cost for the equipment's
        size parameter, and applies inflation adjustment to the target year.
        If ``num_units`` was given, the correlation cost is the cost of one
        unit of size ``param`` and is multiplied by ``num_units``;
        otherwise ``num_units`` is set to the number of parallel units the
        database split ``param`` into (already included in the cost).
        Also sets ``cost_year`` as a side effect.

        Returns
        -------
        float
            Inflation-adjusted purchased equipment cost for all units.
        """
        key = self._resolve_key()
        if isinstance(self.param, (tuple, list)):
            s, s2 = self.param
        else:
            s, s2 = self.param, None
        purchased, units, year = self._db.evaluate(key, s, s2)
        if self.num_units is None:
            self.num_units = units
        else:
            purchased = purchased * self.num_units
        self.cost_year = year
        return inflation_adjustment(
            purchased, year, target_year=self.target_year
        )

    def calculate_direct_cost(self) -> float:
        """
        Calculate total direct cost including process and material factors.

        Applies erection, piping, instrumentation, electrical, civil,
        structural, lagging, and material factors to the purchased cost.

        Returns
        -------
        float
            Total direct installed cost.
        """
        self.direct_cost = self.purchased_cost * (
            (1 + self.piping_factor) * self.material_factor
            + (
                self.erection_factor
                + self.electrical_factor
                + self.instrumentation_factor
                + self.civil_factor
                + self.structural_factor
                + self.lagging_factor
            )
        )
        return self.direct_cost

    def to_dict(self):
        """
        Convert equipment specifications and costs to a dictionary.

        Returns
        -------
        dict
            Keys: name, category, type, material, process_type, param,
            num_units, cost_year, target_year, purchased_cost, direct_cost.
        """
        return {
            "name": self.name,
            "category": self.category,
            "type": self.type,
            "material": self.material,
            "process_type": self.process_type,
            "param": self.param,
            "num_units": self.num_units,
            "cost_year": self.cost_year,
            "target_year": self.target_year,
            "purchased_cost": float(self.purchased_cost),
            "direct_cost": float(self.direct_cost),
        }

    def __str__(self) -> str:
        """
        Return a formatted string summary of the equipment.

        Returns
        -------
        str
            Human-readable representation of equipment specifications
            and computed costs.
        """
        return (
            f"Name={self.name}, "
            f"Category={self.category}, Sub-type={self.type}, "
            f"Material={self.material}, Process Type={self.process_type}, "
            f"Parameter={self.param}, Number of units={self.num_units}, "
            f"Purchased Cost={self.purchased_cost:.3f}, "
            f"Direct Cost={self.direct_cost:.3f}"
        )


class CompositeEquipment:
    """
    Equipment assembled from individually priced sub-components.

    Models a package unit such as a PSA skid, a compressor train or a
    reactor with its catalyst charge: one line item in the plant's
    equipment list whose cost is built up from sub-components. Each
    sub-component is an ordinary :class:`Equipment` (so it brings its own
    cost correlation or user-defined purchased cost, its own material
    factor and its own process-type installation factors) or another
    ``CompositeEquipment`` (nesting is allowed).

    The composite exposes the same attributes :class:`~openpytea.plant.Plant`
    reads from an ``Equipment`` (``name``, ``category``, ``type``,
    ``process_type``, ``param``, ``num_units``, ``cost_year``,
    ``target_year``, ``purchased_cost``, ``direct_cost``, ``to_dict``), so
    it can be placed directly in ``equipment_list``. For operator
    estimation it counts as a single process step.

    Parameters
    ----------
    name : str
        Composite identifier/name.
    process_type : ProcessType
        Type of process ("Solids", "Fluids", "Mixed", or "Electrical").
        Used by the plant's operator estimate and, when
        ``installation="composite"``, as the installation-factor table for
        the whole composite.
    components : list
        Sub-components: ``Equipment`` or ``CompositeEquipment`` objects.
        Several identical units of a component are priced by setting
        ``num_units`` on that component, exactly as for stand-alone
        equipment; its ``purchased_cost`` and ``direct_cost`` then already
        cover all of its units.
    category : str, optional
        Category label for reporting. Default is "Composite".
    type : str | None, optional
        Sub-type label for reporting. Default is None.
    installation : {"component", "composite"}, optional
        How the direct (installed) cost is built. ``"component"`` (default)
        sums each sub-component's own ``direct_cost``, so every part keeps
        its own process-type and material factors. ``"composite"`` applies
        this composite's process-type factors (and any factor overrides)
        once to the total purchased cost, i.e. the composite is installed
        as one item.
    purchased_cost : float | None, optional
        User-defined purchased cost for the composite (e.g. a vendor quote
        for the skid), taken as the total for all ``num_units`` just as
        for ``Equipment``. When given it replaces the component sum; the
        components are kept only as a breakdown, and the direct cost is
        computed with the ``"composite"`` rule. Default is None.
    cost_year : int | None, optional
        Year of the composite ``purchased_cost`` quote, for inflation
        adjustment. Default is None (no adjustment).
    target_year : int, optional
        Target year for cost reporting. Every component must share it.
        Default is 2024.
    num_units : int, optional
        Number of identical composites. Multiplies the component-based
        purchased and direct cost, consistent with ``Equipment``; a
        composite ``purchased_cost`` quote is not multiplied. Default
        is 1.
    erection_factor : float | None, optional
        Override used by the ``"composite"`` rule. Default is None
        (use the ``process_type`` table).
    piping_factor : float | None, optional
        Override used by the ``"composite"`` rule. Default is None
        (use the ``process_type`` table).
    instrumentation_factor : float | None, optional
        Override used by the ``"composite"`` rule. Default is None
        (use the ``process_type`` table).
    electrical_factor : float | None, optional
        Override used by the ``"composite"`` rule. Default is None
        (use the ``process_type`` table).
    civil_factor : float | None, optional
        Override used by the ``"composite"`` rule. Default is None
        (use the ``process_type`` table).
    structural_factor : float | None, optional
        Override used by the ``"composite"`` rule. Default is None
        (use the ``process_type`` table).
    lagging_factor : float | None, optional
        Override used by the ``"composite"`` rule. Default is None
        (use the ``process_type`` table).
    material_factor : float | None, optional
        Material factor used by the ``"composite"`` rule. Default is
        None (1.0).

    Attributes
    ----------
    components : list
        The component objects as given.
    components_purchased_cost : float
        Sum of the components' ``purchased_cost`` times ``num_units``, in
        ``target_year`` money. Equals ``purchased_cost`` unless a composite
        quote was supplied.
    purchased_cost : float
        Composite purchased cost for all units in ``target_year`` money.
    direct_cost : float
        Composite direct (installed) cost for all units.

    Raises
    ------
    ValueError
        If ``process_type`` or ``installation`` is invalid, ``components``
        is empty, ``num_units`` is not positive, or a component's
        ``target_year`` differs from the composite's.
    TypeError
        If a component has no ``purchased_cost``/``direct_cost``.

    Examples
    --------
    >>> vessel = Equipment("Adsorber vessel", 12.0, "Fluids",
    ...                    "Pressure vessels", type="Vertical",
    ...                    material="304 stainless steel", num_units=4)
    >>> zeolite = Equipment("Zeolite 5A", 1000, "Solids",
    ...                     "Packings & adsorbents", type="Molecular sieves")
    >>> psa = CompositeEquipment("PSA", "Fluids",
    ...                          components=[vessel, zeolite],
    ...                          category="Adsorption", type="PSA")
    >>> psa.breakdown()  # DataFrame with one row per sub-component
    """

    def __init__(
        self,
        name: str,
        process_type: ProcessType,
        components: list,
        category: str = "Composite",
        type: str | None = None,
        installation: Literal["component", "composite"] = "component",
        purchased_cost: float | None = None,
        cost_year: int | None = None,
        target_year: int = 2024,
        num_units: int = 1,
        erection_factor: float | None = None,
        piping_factor: float | None = None,
        instrumentation_factor: float | None = None,
        electrical_factor: float | None = None,
        civil_factor: float | None = None,
        structural_factor: float | None = None,
        lagging_factor: float | None = None,
        material_factor: float | None = None,
    ):
        """Initialize the composite and compute purchased and direct costs."""
        if process_type not in Equipment.process_factors:
            raise ValueError(
                _unknown("process_type", process_type, Equipment.process_factors)
            )
        if installation not in ("component", "composite"):
            raise ValueError(
                _unknown("installation", installation, ["component", "composite"])
            )
        if not components:
            raise ValueError(
                f"CompositeEquipment '{name}' needs at least one component."
            )
        if num_units <= 0:
            raise ValueError(
                f"CompositeEquipment '{name}' has num_units={num_units}; "
                f"it must be positive."
            )

        self.name = name
        self.process_type = process_type
        self.category = _current_category(category)
        self.type = type
        self.installation = installation
        self.target_year = target_year
        self.num_units = num_units
        self.param = None
        self.material = None
        # Component costs are already escalated, so the composite's own
        # cost basis is the target year.
        self.cost_year = target_year

        self.components = []
        for obj in components:
            if not hasattr(obj, "purchased_cost") or not hasattr(
                obj, "direct_cost"
            ):
                raise TypeError(
                    f"Component {obj!r} of composite '{name}' must be an "
                    f"Equipment or CompositeEquipment."
                )
            if getattr(obj, "target_year", target_year) != target_year:
                raise ValueError(
                    f"Component '{obj.name}' has target_year="
                    f"{obj.target_year} but composite '{name}' has "
                    f"target_year={target_year}."
                )
            self.components.append(obj)

        _set_install_factors(
            self, Equipment.process_factors[process_type],
            erection_factor=erection_factor,
            piping_factor=piping_factor,
            instrumentation_factor=instrumentation_factor,
            electrical_factor=electrical_factor,
            civil_factor=civil_factor,
            structural_factor=structural_factor,
            lagging_factor=lagging_factor,
        )
        self.material_factor = (
            material_factor if material_factor is not None else 1.0
        )

        self.components_purchased_cost = float(
            sum(obj.purchased_cost for obj in self.components) * num_units
        )
        self._quoted = purchased_cost is not None
        if self._quoted:
            quote = float(purchased_cost)
            if cost_year is not None:
                quote = inflation_adjustment(
                    quote, cost_year, target_year=target_year
                )
            self.purchased_cost = quote
        else:
            self.purchased_cost = self.components_purchased_cost

        self.direct_cost = self.calculate_direct_cost()

    def calculate_direct_cost(self) -> float:
        """
        Calculate the composite's direct (installed) cost.

        With ``installation="component"`` and no composite quote, this is
        the sum of the components' own direct costs times ``num_units``.
        Otherwise the composite's installation factors are applied once to
        ``purchased_cost``, using the same formula as
        :meth:`Equipment.calculate_direct_cost`.

        Returns
        -------
        float
            Total direct installed cost.
        """
        if self.installation == "component" and not self._quoted:
            self.direct_cost = float(
                sum(obj.direct_cost for obj in self.components)
                * self.num_units
            )
        else:
            Equipment.calculate_direct_cost(self)
        return self.direct_cost

    def leaves(self, _prefix: str = "", _multiplier: int | None = None):
        """
        Iterate over the leaf components, flattening nested composites.

        Yields
        ------
        tuple[str, Equipment, int]
            ``(label, equipment, multiplier)`` where ``label`` is the
            component's name prefixed by the names of enclosing composites
            (``"Inner composite / Vessel"``) and ``multiplier`` is the
            product of the ``num_units`` of every enclosing composite,
            this one included. The leaf's own ``purchased_cost`` and
            ``direct_cost`` already cover its own ``num_units``.
        """
        multiplier = self.num_units if _multiplier is None else _multiplier
        for obj in self.components:
            label = f"{_prefix}{obj.name}"
            if isinstance(obj, CompositeEquipment):
                yield from obj.leaves(
                    label + " / ", multiplier * obj.num_units
                )
            else:
                yield label, obj, multiplier

    def breakdown(self) -> pd.DataFrame:
        """
        Tabulate the leaf components and their costs.

        Returns
        -------
        pd.DataFrame
            One row per leaf component with columns ``component``,
            ``category``, ``type``, ``material``, ``param``, ``num_units``
            (total count of that component across all enclosing
            composites), ``purchased_each`` (cost of one unit),
            ``purchased_total`` and ``direct_total`` (all units).
            ``direct_total`` is the component's own installed cost; see
            :meth:`direct_cost_breakdown` for a split that always sums to
            the composite's ``direct_cost``.
        """
        rows = []
        for label, leaf, mult in self.leaves():
            leaf_units = leaf.num_units or 1
            rows.append(
                {
                    "component": label,
                    "category": leaf.category,
                    "type": leaf.type,
                    "material": leaf.material,
                    "param": leaf.param,
                    "num_units": leaf_units * mult,
                    "purchased_each": float(leaf.purchased_cost / leaf_units),
                    "purchased_total": float(leaf.purchased_cost * mult),
                    "direct_total": float(leaf.direct_cost * mult),
                }
            )
        return pd.DataFrame(rows)

    def direct_cost_breakdown(self) -> dict:
        """
        Split the composite's direct cost over its leaf components.

        Under the ``"component"`` rule this is each component's own direct
        cost. When the composite is installed as one item (``"composite"``
        rule or a composite quote) the composite direct cost is pro-rated by
        each component's share of ``components_purchased_cost``. Either way
        the values sum to ``direct_cost``.

        Returns
        -------
        dict
            Mapping of ``"Composite name / component label"`` to direct cost.
        """
        leaves = list(self.leaves(f"{self.name} / "))
        if self.installation == "component" and not self._quoted:
            return {
                label: float(leaf.direct_cost * mult)
                for label, leaf, mult in leaves
            }
        base = self.components_purchased_cost
        return {
            label: float(
                self.direct_cost * (leaf.purchased_cost * mult) / base
            )
            for label, leaf, mult in leaves
        }

    def to_dict(self):
        """
        Convert the composite and its components to a dictionary.

        Returns
        -------
        dict
            The same keys as :meth:`Equipment.to_dict` plus
            ``installation``, ``components_purchased_cost`` and
            ``components`` (a list of the components' own dicts).
        """
        return {
            **Equipment.to_dict(self),
            "installation": self.installation,
            "components_purchased_cost": self.components_purchased_cost,
            "components": [obj.to_dict() for obj in self.components],
        }

    def __str__(self) -> str:
        """
        Return a formatted string summary of the composite and its parts.

        Returns
        -------
        str
            Human-readable representation of the composite costs followed by
            one indented line per direct component.
        """
        head = (
            f"Name={self.name}, "
            f"Category={self.category}, Sub-type={self.type}, "
            f"Process Type={self.process_type}, "
            f"Installation={self.installation}, "
            f"Number of units={self.num_units}, "
            f"Purchased Cost={self.purchased_cost:.3f}, "
            f"Direct Cost={self.direct_cost:.3f}"
        )
        body = "\n".join(
            f"    - {obj.name} (x{obj.num_units}): "
            f"Purchased Cost={obj.purchased_cost:.3f}, "
            f"Direct Cost={obj.direct_cost:.3f}"
            for obj in self.components
        )
        return f"{head}\n{body}"
