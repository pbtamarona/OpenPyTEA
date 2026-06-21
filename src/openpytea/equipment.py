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
    parameters. Supports multiple correlation forms (power-law, quad log-log)
    and handles equipment parallelization when capacity limits are exceeded.

    Attributes
    ----------
    df : pd.DataFrame
        Cost correlation data with columns: key, category, type, form,
        s_lower, s_upper, upper_parallel, a, b, n, k1, k2, k3, cost_year.
    """

    def __init__(self, df=COST_DB_DF):
        """
        Initialize database with cost correlation DataFrame.

        Normalizes column names to lowercase and converts numeric columns.

        Parameters
        ----------
        df : pd.DataFrame
            Cost correlation data. Defaults to the bundled CSV database.
        """
        df.columns = [c.strip().lower() for c in df.columns]
        for col in [
            "s_lower",
            "s_upper",
            "upper_parallel",
            "a",
            "b",
            "n",
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

    def _parallelize(self, s: float, cap: float | None):
        """
        Calculate parallel units and adjusted size when capacity is exceeded.

        Parameters
        ----------
        s : float
            Equipment size/capacity.
        cap : float | None
            Unit capacity limit. If None or NaN, no parallelization occurs.

        Returns
        -------
        tuple[int, float]
            (number_of_units, adjusted_size_per_unit).
        """
        if pd.notna(cap) and s > cap:
            units = int(np.ceil(s / cap))
            return units, s / units
        return 1, s

    def evaluate(self, key: str, s: float):
        """
        Calculate purchased equipment cost based on correlation key and size.

        Parameters
        ----------
        key : str
            Unique identifier for the cost correlation.
        s : float
            Equipment size/capacity parameter.

        Returns
        -------
        tuple[float, int, int]
            (total_cost, number_of_units, cost_year).

        Raises
        ------
        KeyError
            If correlation key not found in database.
        ValueError
            If size is below the lower bound or the correlation form is unsupported.
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

        units, s_adj = self._parallelize(s, cap)
        form = r.get("form", "linear")
        year = int(r["cost_year"])

        if form == "power-law":
            a, b, n = r["a"], r["b"], r["n"]
            ce = a + b * (s_adj**n)
            purchased = ce * units

        elif form == "quad log-log":
            K1, K2, K3 = r["k1"], r["k2"], r["k3"]

            logS = np.log10(s_adj)
            logCe = K1 + K2 * logS + K3 * (logS**2)

            ce = 10**logCe
            purchased = ce * units

        else:
            raise ValueError(
                f"Unsupported form '{form}' for key '{key}'"
            )

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
        (1.0 to 1.7).

    Parameters
    ----------
    name : str
        Equipment identifier/name.
    param : float
        Equipment parameter (size, capacity) for cost correlation lookup.
    process_type : str
        Type of process ("Solids", "Fluids", "Mixed", or "Electrical").
    category : str
        Equipment category for database lookup.
    type : str | None, optional
        Equipment sub-type for database lookup. Default is None.
    material : str, optional
        Material of construction. Default is "Carbon steel".
    num_units : int | None, optional
        Number of identical units. Default is None (set to 1 when
        purchased_cost is provided).
    purchased_cost : float | None, optional
        Direct purchased cost input. If provided, param is ignored.
        Default is None.
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
        Material factor override. Default is None (use material table).

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
    ...     param=100,
    ...     process_type="Fluids",
    ...     category="Reactor",
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
        "304 stainless steel": 1.3,
        "316 stainless steel": 1.3,
        "321 stainless steel": 1.5,
        "Hastelloy C": 1.55,
        "Monel": 1.65,
        "Nickel": 1.7,
        "Inconel": 1.7,
    }

    def __init__(
        self,
        name: str,
        param: float,
        process_type: str,
        category: str,
        type: str | None = None,
        material: str = "Carbon steel",
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
        self.material = material
        self.param = (
            None if purchased_cost is not None else param
        )
        self.category = category
        self.type = type
        self.num_units = num_units
        self.cost_year = (
            cost_year if cost_year is not None else None
        )
        self.target_year = target_year
        self._cost_func = cost_func
        self._db = CostCorrelationDB()

        valid_process_types = list(self.process_factors.keys())
        if process_type not in self.process_factors:
            raise ValueError(
                f"Invalid process_type '{process_type}'. "
                f"Valid options are: {valid_process_types}"
            )
        valid_materials = list(self.material_factors.keys())
        if material not in self.material_factors:
            raise ValueError(
                f"Invalid material '{material}'. "
                f"Valid options are: {valid_materials}"
            )

        _pf = self.process_factors[process_type]
        self.erection_factor = (
            erection_factor if erection_factor is not None else _pf["fer"]
        )
        self.piping_factor          = (
            piping_factor          if piping_factor          is not None else _pf["fp"]
        )
        self.instrumentation_factor = (
            instrumentation_factor if instrumentation_factor is not None else _pf["fi"]
        )
        self.electrical_factor      = (
            electrical_factor      if electrical_factor      is not None else _pf["fel"]
        )
        self.civil_factor           = (
            civil_factor           if civil_factor           is not None else _pf["fc"]
        )
        self.structural_factor      = (
            structural_factor      if structural_factor      is not None else _pf["fs"]
        )
        self.lagging_factor         = (
            lagging_factor         if lagging_factor         is not None else _pf["fl"]
        )
        self.material_factor = (
            material_factor
            if material_factor is not None
            else self.material_factors[material]
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
            raise KeyError(
                f"No CSV correlation matches category='{self.category}', "
                f"type='{self.type}'. "
                f"Add a row to the CSV or specify cost_func manually."
            )
        return key

    def _calc_purchased_cost(self) -> float:
        """
        Calculate purchased cost using the database correlation.

        Resolves the correlation key, evaluates the cost for the equipment's
        size parameter, and applies inflation adjustment to the target year.
        Also sets ``num_units`` and ``cost_year`` as side effects.

        Returns
        -------
        float
            Inflation-adjusted purchased equipment cost.
        """
        key = self._resolve_key()
        s = self.param
        purchased, units, year = self._db.evaluate(key, s)
        self.num_units = self.num_units or units
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
            f"Purchased Cost={self.purchased_cost}, "
            f"Direct Cost={self.direct_cost})"
        )
