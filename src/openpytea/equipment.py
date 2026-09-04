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

        units, s_adj = self._parallelize(s, cap)
        form = r.get("form", "linear")
        year = int(r["cost_year"])

        if form == "offset power-law":
            a, b, n = r["a"], r["b"], r["n"]
            ce = a + b * (s_adj**n)
            purchased = ce * units

        elif form == "exponential":
            a, b = r["a"], r["b"]
            ce = a * np.exp(b * s_adj)
            purchased = ce * units

        elif form == "log-log quadratic":
            K1, K2, K3 = r["k1"], r["k2"], r["k3"]
            K4 = r.get("k4") if pd.notna(r.get("k4")) else 0.0
            K5 = r.get("k5") if pd.notna(r.get("k5")) else 0.0

            logS = np.log10(s_adj)
            logCe = (
                K1
                + K2 * logS
                + K3 * (logS**2)
                + K4 * (logS**3)
                + K5 * (logS**4)
            )

            ce = 10**logCe
            purchased = ce * units

        elif form == "ln-ln quadratic":
            K1, K2, K3 = r["k1"], r["k2"], r["k3"]
            K4 = r.get("k4") if pd.notna(r.get("k4")) else 0.0
            K5 = r.get("k5") if pd.notna(r.get("k5")) else 0.0

            lnS = np.log(s_adj)
            lnCe = (
                K1
                + K2 * lnS
                + K3 * (lnS**2)
                + K4 * (lnS**3)
                + K5 * (lnS**4)
            )

            ce = np.exp(lnCe)
            purchased = ce * units

        elif form == "power-sizing":
            C0, S0, f = r["c0"], r["s0"], r["f"]
            ce = C0 * (s_adj / S0) ** f
            purchased = ce * units

        elif form == "2-var power-law":
            if s2 is None:
                raise ValueError(
                    f"Correlation '{key}' has form '2-var "
                    f"power-law' and requires a second size parameter "
                    f"(s2)."
                )
            a, b, n1, n2 = r["a"], r["b"], r["n"], r["n2"]
            ce = a + b * (s_adj**n1) * (s2**n2)
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
    param : float | tuple[float, float] | list[float]
        Equipment parameter (size, capacity) for cost correlation lookup,
        per unit when ``num_units`` is given. Pass a 2-element tuple/list
        ``(s1, s2)`` for two-parameter correlation forms such as
        ``"2-var power-law"``.
    process_type : str
        Type of process ("Solids", "Fluids", "Mixed", or "Electrical").
    category : str
        Equipment category for database lookup.
    type : str | None, optional
        Equipment sub-type for database lookup. Default is None.
    material : str | None, optional
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
        "Stainless steel": 1.3,
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
        param: float | tuple[float, float] | list[float],
        process_type: str,
        category: str,
        type: str | None = None,
        material: str | None = None,
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
        self.category = category
        self.type = type
        self.num_units = num_units
        self.cost_year = (
            cost_year if cost_year is not None else None
        )
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

        valid_process_types = list(self.process_factors.keys())
        if process_type not in self.process_factors:
            raise ValueError(
                f"Invalid process_type '{process_type}'. "
                f"Valid options are: {valid_process_types}"
            )
        valid_materials = list(self.material_factors.keys())
        if (
            material not in self.material_factors
            and not uses_default_material
        ):
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
            f"Purchased Cost={self.purchased_cost}, "
            f"Direct Cost={self.direct_cost})"
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
    process_type : str
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
        process_type: str,
        components: list,
        category: str = "Composite",
        type: str | None = None,
        installation: str = "component",
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
                f"Invalid process_type '{process_type}'. "
                f"Valid options are: {list(Equipment.process_factors)}"
            )
        if installation not in ("component", "composite"):
            raise ValueError(
                f"Invalid installation '{installation}'. "
                f"Valid options are: ['component', 'composite']"
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
        self.category = category
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

        _pf = Equipment.process_factors[process_type]
        self.erection_factor = (
            erection_factor if erection_factor is not None else _pf["fer"]
        )
        self.piping_factor = (
            piping_factor if piping_factor is not None else _pf["fp"]
        )
        self.instrumentation_factor = (
            instrumentation_factor
            if instrumentation_factor is not None
            else _pf["fi"]
        )
        self.electrical_factor = (
            electrical_factor if electrical_factor is not None else _pf["fel"]
        )
        self.civil_factor = (
            civil_factor if civil_factor is not None else _pf["fc"]
        )
        self.structural_factor = (
            structural_factor if structural_factor is not None else _pf["fs"]
        )
        self.lagging_factor = (
            lagging_factor if lagging_factor is not None else _pf["fl"]
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
            f"Purchased Cost={self.purchased_cost}, "
            f"Direct Cost={self.direct_cost})"
        )
        body = "\n".join(
            f"    - {obj.name} (x{obj.num_units}): "
            f"Purchased Cost={obj.purchased_cost}, "
            f"Direct Cost={obj.direct_cost}"
            for obj in self.components
        )
        return f"{head}\n{body}"
