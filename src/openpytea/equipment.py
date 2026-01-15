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


def inflation_adjustment(
    equipment_cost, cost_year, target_year=2024
):
    """
    Adjust equipment cost from one year to another using
    the Chemical Engineering Plant Cost Index (CEPCI).

    Parameters
    ----------
    equipment_cost : float
        The cost of the equipment in the cost_year.
    cost_year : int
        The year in which the equipment_cost is valued.
        Must be available in CEPCI_DF.
    target_year : int, optional
        The year to adjust the cost to. Default is 2024.
        Must be available in CEPCI_DF.

    Returns
    -------
    float
        The inflation-adjusted equipment cost in the target_year.

    Raises
    ------
    ValueError
        If cost_year is not available in CEPCI_DF.
    ValueError
        If target_year is not available in CEPCI_DF.

    Examples
    --------
    >>> adjusted_cost = inflation_adjustment(10000, 2020, 2024)
    >>> print(adjusted_cost)
    12345.67
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
    A database interface for equipment cost correlations.
    This class manages cost estimation correlations for equipment based on
    size/capacity parameters. It supports multiple correlation forms
    (power-law, quad log-log) and handles equipment parallelization when
    capacity limits are exceeded.
    Attributes:
        df (pd.DataFrame): DataFrame containing cost correlation data with
        columns including: key, category, type, form, s_lower, s_upper,
        upper_parallel, a, b, n, k1, k2, k3, cost_year, and other parameters.
    Methods:
        __init__(df: pd.DataFrame) -> None:
            Initialize the database with a cost correlation DataFrame.
            Normalizes column names to lowercase and converts numeric columns.
        _parallelize(s: float, cap: float | None) -> tuple[int, float]:
            Calculate number of parallel units and adjusted size when capacity
            is exceeded.
            Args:
                s: Equipment size/capacity.
                cap: Unit capacity limit. If None, no parallelization occurs.
            Returns:
                Tuple of (number_of_units, adjusted_size_per_unit).
        evaluate(key: str, s: float) -> tuple[float, int, int]:
            Calculate purchased equipment cost based on
            correlation key and size.
            Args:
                key: Unique identifier for the cost correlation.
                s: Equipment size/capacity parameter.
            Returns:
                Tuple of (total_cost, number_of_units, cost_year).
            Raises:
                KeyError: If correlation key not found in database.
                ValueError: If size is below lower bound or
                form is unsupported.
        key_for_category_type(eq_category: str, type: str | None)
        -> str | None:
        Look up correlation key by equipment category and optional type.
            Args:
                eq_category: Equipment category name.
                type: Equipment type (optional).
            Returns:
                Correlation key if found, None otherwise.
    """

    def __init__(self, df=COST_DB_DF):
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
        if pd.notna(cap) and s > cap:
            units = int(np.ceil(s / cap))
            return units, s / units
        return 1, s

    def evaluate(self, key: str, s: float):
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
            K1, K2, K3 = r["k1"], r["k3"], r["k3"]

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

        # ✅ Return the first match (take the first listed in the CSV)
        return cand.iloc[0]["key"]


class Equipment:
    """
    Equipment cost estimation class for process equipment.
    This class manages the cost calculation of process equipment
    based on process type, material, and equipment parameters.
    It supports both direct cost input and calculated costs
    based on correlations from a cost database.
    Attributes:
        process_factors (dict):
        Dictionary of process type factors affecting cost calculation.
            Keys are process types ("Solids", "Fluids", "Mixed", "Electrical").
            Values are dicts with factors: fer, fp, fi, fel, fc, fs, fl.
        material_factors (dict): Dictionary of material type multipliers.
            Maps material names to cost multiplication factors (1.0 to 1.7).
    Args:
        name (str): Equipment identifier/name.
        param (float):
        Equipment parameter (size, capacity) for cost correlation lookup.
        process_type (str): Type of process
        ("Solids", "Fluids", "Mixed", or "Electrical").
        category (str): Equipment category for database lookup.
        type (str | None): Equipment sub-type for database lookup.
        Default is None.
        material (str): Material of construction.
        Default is "Carbon steel".
        num_units (int | None): Number of identical units.
        Default is None (set to 1 if purchased_cost provided).
        purchased_cost (float | None): Direct purchased cost input.
        If provided, param is ignored. Default is None.
        cost_year (int | None): Year of the purchased_cost quote.
        Default is None.
        cost_func (str | None): Explicit cost correlation key from database.
        Default is None (auto-resolved).
        target_year (int): Target year for inflation adjustment.
        Default is 2024.
    Methods:
        _resolve_key() -> str: Resolves the cost correlation key
        from database or explicit input.
        _calc_purchased_cost() -> float: Calculates purchased cost
        using database correlation.
        calculate_direct_cost() -> float: Calculates total direct cost
        including process and material factors.
        __str__() -> str: Returns formatted string representation of equipment
        specifications and costs.
    Raises:
        ValueError:
        If process_type or material not found in factor dictionaries.
        KeyError:
        If category/type combination not found in database
        and cost_func not specified.
    ️"""

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
        cost_func: (
            str | None
        ) = None,  # explicit correlation key
        target_year: int = 2024,
    ):

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
        self._db = (
            CostCorrelationDB()
        )  # always loads from the fixed CSV file

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
        key = self._resolve_key()
        s = self.param
        purchased, units, year = self._db.evaluate(key, s)
        self.num_units = self.num_units or units
        self.cost_year = year
        return inflation_adjustment(
            purchased, year, target_year=self.target_year
        )

    def calculate_direct_cost(self) -> float:

        if self.process_type not in self.process_factors:
            raise ValueError(
                f"Process type not found: {self.process_type}"
            )

        if self.material not in self.material_factors:
            raise ValueError(
                f"Material not found: {self.material}"
            )

        factors = self.process_factors[self.process_type]
        fm = self.material_factors[self.material]

        self.direct_cost = self.purchased_cost * (
            (1 + factors["fp"]) * fm
            + (
                factors["fer"]
                + factors["fel"]
                + factors["fi"]
                + factors["fc"]
                + factors["fs"]
                + factors["fl"]
            )
        )
        return self.direct_cost

    def __str__(self) -> str:
        return (
            f"Name={self.name}, "
            f"Category={self.category}, Sub-type={self.type}, "
            f"Material={self.material}, Process Type={self.process_type}, "
            f"Parameter={self.param}, Number of units={self.num_units}, "
            f"Purchased Cost={self.purchased_cost}, "
            f"Direct Cost={self.direct_cost})"
        )
