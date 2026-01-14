import numpy as np
import pandas as pd

# --- Fixed CSV data sources ---
from importlib.resources import files, as_file

data_dir = files("openpytea.data")

with as_file(data_dir / "cepci_values.csv") as CEPCI_CSV_PATH:
    CEPCI_DF = pd.read_csv(CEPCI_CSV_PATH).set_index("year")

with as_file(data_dir / "cost_correlations.csv") as COST_DB_PATH:
    COST_DB_DF = pd.read_csv(COST_DB_PATH)

def inflation_adjustment(equipment_cost, cost_year, target_year=2024):
    if cost_year not in CEPCI_DF.index:
        raise ValueError(f"CEPCI not available for year {cost_year}")
    if target_year not in CEPCI_DF.index:
        raise ValueError(f"CEPCI not available for target year {target_year}")
    return float(equipment_cost) * (CEPCI_DF.loc[target_year, "cepci"] / CEPCI_DF.loc[cost_year, "cepci"])


class CostCorrelationDB:

    def __init__(self, df=COST_DB_DF):
        df.columns = [c.strip().lower() for c in df.columns]
        for col in ["s_lower","s_upper","upper_parallel","a","b","n","s0","c0","f","cost_year"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
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
            raise KeyError(f"Correlation key not found in CSV: {key}")
        r = row.iloc[0].to_dict()

        s_lower = r.get("s_lower")
        s_upper = r.get("s_upper")
        cap     = r.get("upper_parallel") if pd.notna(r.get("upper_parallel")) else s_upper

        if pd.notna(s_lower) and s < s_lower:
            raise ValueError(f"s={s} below lower bound {s_lower} for key '{key}'")

        units, s_adj = self._parallelize(s, cap)
        form = r.get("form", "linear")
        year = int(r["cost_year"])

        if form == "power-law":
            a, b, n = r["a"], r["b"], r["n"]
            ce = a + b * (s_adj ** n)
            purchased = ce * units

        elif form == "quad log-log":
            K1, K2, K3 = r["k1"], r["k3"], r["k3"]

            logS = np.log10(s_adj)
            logCe = K1 + K2 * logS + K3 * (logS ** 2)

            ce = 10 ** logCe
            purchased = ce * units 
        
        else:
            raise ValueError(f"Unsupported form '{form}' for key '{key}'")

        return float(purchased), int(units), year
    

    def key_for_category_type(self, eq_category: str, type: str | None):

        t = eq_category.lower()
        st = type.lower() if type else ""
        df = self.df

        if "category" not in df.columns:
            return None

        cand = df[df["category"].str.lower() == t]
        if "type" in df.columns:
            cand = cand[cand["type"].fillna("").str.lower() == st]

        if cand.empty:
            return None

        # ✅ Return the first match (take the first listed in the CSV)
        return cand.iloc[0]["key"]


class Equipment:
    process_factors = {
        'Solids': {'fer': 0.6, 'fp': 0.2, 'fi': 0.2, 'fel': 0.15, 'fc': 0.2, 'fs': 0.1, 'fl': 0.05},
        'Fluids': {'fer': 0.3, 'fp': 0.8, 'fi': 0.3, 'fel': 0.2, 'fc': 0.3, 'fs': 0.2, 'fl': 0.1},
        'Mixed': {'fer': 0.5, 'fp': 0.6, 'fi': 0.3, 'fel': 0.2, 'fc': 0.3, 'fs': 0.2, 'fl': 0.1},
        'Electrical': {'fer': 0.4, 'fp': 0.1, 'fi': 0.7, 'fel': 0.7, 'fc': 0.2, 'fs': 0.1, 'fl': 0.1}
    }

    material_factors = {
        'Carbon steel': 1.0,
        'Aluminum': 1.07,
        'Bronze': 1.07,
        'Cast steel': 1.1,
        '304 stainless steel': 1.3,
        '316 stainless steel': 1.3,
        '321 stainless steel': 1.5,
        'Hastelloy C': 1.55,
        'Monel': 1.65,
        'Nickel': 1.7,
        'Inconel': 1.7,
    }

    def __init__(self,
                 name: str,
                 param: float,
                 process_type: str,
                 category: str,
                 type: str | None = None,
                 material: str = 'Carbon steel',
                 num_units: int | None = None,
                 purchased_cost: float | None = None,
                 cost_year: int | None = None,
                 cost_func: str | None = None,       # explicit correlation key
                 target_year: int = 2024):
        
        self.name = name
        self.process_type = process_type
        self.material = material
        self.param = None if purchased_cost is not None else param
        self.category = category
        self.type = type
        self.num_units = num_units
        self.cost_year = cost_year if cost_year is not None else None
        self.target_year = target_year
        self._cost_func = cost_func
        self._db = CostCorrelationDB()  # always loads from the fixed CSV file

        if purchased_cost is not None:
            self.purchased_cost = purchased_cost
            if cost_year is not None:
                self.purchased_cost = inflation_adjustment(purchased_cost, cost_year, target_year=self.target_year)
            if self.num_units is None:
                self.num_units = 1
        else:
            self.purchased_cost = self._calc_purchased_cost()
        self.direct_cost = self.calculate_direct_cost()  # your existing method


    def _resolve_key(self) -> str:

        if self._cost_func:
            return self._cost_func

        key = self._db.key_for_category_type(self.category, self.type)
        if key is None:
            raise KeyError(
                f"No CSV correlation matches category='{self.category}', type='{self.type}'. "
                f"Add a row to the CSV or specify cost_func manually."
            )
        return key

    def _calc_purchased_cost(self) -> float:
        key = self._resolve_key()
        s = self.param
        purchased, units, year = self._db.evaluate(key, s)
        self.num_units = self.num_units or units
        self.cost_year = year
        return inflation_adjustment(purchased, year, target_year=self.target_year)


    def calculate_direct_cost(self) -> float:

        if self.process_type not in self.process_factors:
            raise ValueError(f'Process type not found: {self.process_type}')

        if self.material not in self.material_factors:
            raise ValueError(f'Material not found: {self.material}')

        factors = self.process_factors[self.process_type]
        fm = self.material_factors[self.material]

        self.direct_cost = self.purchased_cost * ((1 + factors['fp']) * fm + (
                factors['fer'] + factors['fel'] + factors['fi'] + factors['fc'] + factors['fs'] + factors['fl']))
        return self.direct_cost
    

    def __str__(self) -> str:
        """
        Return a string representation of the equipment object, including its name, type, sub-type, material, process type, parameter, purchased cost, and direct cost.
        """
        return (f"Name={self.name}, Category={self.category}, Sub-type={self.type}, "
                f"Material={self.material}, Process Type={self.process_type}, "
                f"Parameter={self.param}, Number of units={self.num_units}, "
                f"Purchased Cost={self.purchased_cost}, Direct Cost={self.direct_cost})")
