import math
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import List, Dict, Literal, Optional
from scipy.optimize import root_scalar


class Plant:
    """
    Plant cost estimation and financial analysis module.
    This module provides the Plant class for comprehensive TEA
    of industrial process plants, including capital cost estimation,
    operating cost calculation, and financial metrics computation.
    Classes:
        Plant: Main class for plant configuration and financial analysis.
    The Plant class supports:
    - Capital cost estimation (ISBL, OSBL, fixed capital)
    - Operating expense calculation (fixed and variable OPEX)
    - Revenue estimation from multiple products
    - Cash flow analysis with production ramps and depreciation
    - Financial metrics (NPV, IRR, ROI, payback period, levelized cost)
    - Support for Monte Carlo uncertainty analysis
    - Location-based cost factors for multiple countries/regions
    - Flexible equipment and process type configurations
    Typical workflow:
        1. Initialize Plant with configuration dictionary
        2. Add equipment to equipment_list
        3. Configure products, OPEX inputs, and operator rates
        4. Call calculate_cash_flow() for financial analysis
        5. Compute metrics like calculate_npv(), calculate_irr(), etc.
    Attributes:
        processTypes (dict):
        Default process multipliers for Solids, Fluids, Mixed
        locFactors (dict):
        Location-based cost adjustment factors by country/region
    """

    processTypes = {
        "Solids": {"OS": 0.4, "DE": 0.2, "X": 0.1},
        "Fluids": {"OS": 0.3, "DE": 0.3, "X": 0.1},
        "Mixed": {"OS": 0.4, "DE": 0.25, "X": 0.1},
    }

    locFactors = {
        "United States": {
            "Gulf Coast": 1.00,
            "East Coast": 1.04,
            "West Coast": 1.07,
            "Midwest": 1.02,
        },
        "Canada": {"Ontario": 1.00, "Fort McMurray": 1.60},
        "Mexico": 1.03,
        "Brazil": 1.14,
        "China": {"imported": 1.12, "indigenous": 0.61},
        "Japan": 1.26,
        "Southeast Asia": 1.12,
        "Australia": 1.21,
        "India": 1.02,
        "Middle East": 1.07,
        "France": 1.13,
        "Germany": 1.11,
        "Italy": 1.14,
        "Netherlands": 1.19,
        "Russia": 1.53,
        "United Kingdom": 1.02,
    }

    def __init__(self, configuration: dict):

        # keep a copy of the original config so code can read from it later
        self.config = deepcopy(configuration)

        self.name = configuration.get("plant_name")
        self.process_type = configuration.get(
            "process_type"
        )
        self.country = configuration.get(
            "country", "United States"
        )
        self.region = configuration.get(
            "region", "Gulf Coast"
        )
        self.working_capital = configuration.get(
            "working_capital", None
        )
        self.interest_rate = configuration.get(
            "interest_rate", 0.09
        )
        self.project_lifetime = configuration.get(
            "project_lifetime", 20
        )
        self.plant_utilization = configuration.get(
            "plant_utilization", 1
        )
        self.tax_rate = configuration.get("tax_rate", 0)
        self.depreciation = configuration.get(
            "depreciation", None
        )
        self.operators_per_shift = configuration.get(
            "operators_per_shift", None
        )
        self.operators_hired = configuration.get(
            "operators_hired", None
        )
        self.working_weeks_per_year = configuration.get(
            "working_weeks_per_year", 49
        )
        self.working_shifts_per_week = configuration.get(
            "working_shifts_per_week", 5
        )
        self.operating_shifts_per_day = configuration.get(
            "operating_shifts_per_day", 3
        )
        self.additional_capex_years = configuration.get(
            "additional_capex_years", None
        )
        self.additional_capex_cost = configuration.get(
            "additional_capex_cost", None
        )

        self.equipment_list = configuration.get(
            "equipment", []
        )
        self.operator_hourly_rate = configuration.get(
            "operator_hourly_rate", {}
        )
        self.variable_opex_inputs = configuration.get(
            "variable_opex_inputs", {}
        )
        self.plant_products = configuration.get(
            "plant_products", {}
        )

        self.fc = None
        self.fp = None

        self.monte_carlo_inputs = None
        self.monte_carlo_metrics = None

    def update_configuration(self, configuration: dict):

        # keep the stored config up to date
        if (
            not hasattr(self, "config")
            or self.config is None
        ):
            self.config = {}
        # shallow-merge top-level keys first
        self.config.update(
            {
                k: v
                for k, v in configuration.items()
                if k
                not in [
                    "variable_opex_inputs",
                    "plant_products",
                    "operator_hourly_rate",
                ]
            }
        )

        self.name = configuration.get(
            "plant_name", self.name
        )
        self.process_type = configuration.get(
            "process_type", self.process_type
        )
        self.country = configuration.get(
            "country", self.country
        )
        self.region = configuration.get(
            "region", self.region
        )
        self.equipment_list = configuration.get(
            "equipment", self.equipment_list
        )
        self.working_capital = configuration.get(
            "working_capital", self.working_capital
        )
        self.interest_rate = configuration.get(
            "interest_rate", self.interest_rate
        )
        self.project_lifetime = configuration.get(
            "project_lifetime", self.project_lifetime
        )
        self.plant_utilization = configuration.get(
            "plant_utilization", self.plant_utilization
        )
        self.tax_rate = configuration.get(
            "tax_rate", self.tax_rate
        )
        self.operators_per_shift = configuration.get(
            "operators_per_shift", self.operators_per_shift
        )
        self.operators_hired = configuration.get(
            "operators_hired", self.operators_hired
        )
        self.working_weeks_per_year = configuration.get(
            "working_weeks_per_year",
            self.working_weeks_per_year,
        )
        self.working_shifts_per_week = configuration.get(
            "working_shifts_per_week",
            self.working_shifts_per_week,
        )
        self.operating_shifts_per_day = configuration.get(
            "operating_shifts_per_day",
            self.operating_shifts_per_day,
        )
        self.additional_capex_years = configuration.get(
            "additional_capex_years",
            self.additional_capex_years,
        )
        self.additional_capex_cost = configuration.get(
            "additional_capex_cost",
            self.additional_capex_cost,
        )

        # NEW: allow updating depreciation block
        if "depreciation" in configuration:
            self.depreciation = configuration[
                "depreciation"
            ]

        # merge nested variable_opex_inputs without clobbering
        def recursive_update(original, updates):
            for key, value in updates.items():
                if isinstance(value, dict) and isinstance(
                    original.get(key), dict
                ):
                    recursive_update(original[key], value)
                else:
                    original[key] = value

        if "variable_opex_inputs" in configuration:
            if (
                not hasattr(self, "variable_opex_inputs")
                or self.variable_opex_inputs is None
            ):
                self.variable_opex_inputs = {}
            recursive_update(
                self.variable_opex_inputs,
                configuration["variable_opex_inputs"],
            )

            # also mirror into stored config
            if "variable_opex_inputs" not in self.config:
                self.config["variable_opex_inputs"] = {}
            recursive_update(
                self.config["variable_opex_inputs"],
                configuration["variable_opex_inputs"],
            )

        if "plant_products" in configuration:
            if (
                not hasattr(self, "plant_products")
                or self.plant_products is None
            ):
                self.plant_products = {}
            recursive_update(
                self.plant_products,
                configuration["plant_products"],
            )

            # also mirror into stored config
            if "plant_products" not in self.config:
                self.config["plant_products"] = {}
            recursive_update(
                self.config["plant_products"],
                configuration["plant_products"],
            )

        if "operator_hourly_rate" in configuration:
            if (
                not hasattr(self, "operator_hourly_rate")
                or self.operator_hourly_rate is None
            ):
                self.operator_hourly_rate = {}
            recursive_update(
                self.operator_hourly_rate,
                configuration["operator_hourly_rate"],
            )

            # also mirror into stored config
            if "operator_hourly_rate" not in self.config:
                self.config["operator_hourly_rate"] = {}
            recursive_update(
                self.config["operator_hourly_rate"],
                configuration["operator_hourly_rate"],
            )

    def calculate_purchased_cost(self, print_results=False):

        self.purchased_cost = sum(
            equipment.purchased_cost
            for equipment in self.equipment_list
        )

        if print_results:
            # Print the results
            print("Purchased cost estimation")
            print("===================================")
            for equipment in self.equipment_list:
                print(
                    f"  - {equipment.name}: ${equipment.purchased_cost:,.2f}"
                )
            print("===================================")
            print(
                f"Total Purchased Cost: ${self.purchased_cost:,.2f}"
            )
        else:
            return self.purchased_cost

    def calculate_isbl(self, fc=1.0, print_results=False):

        def location_factors() -> float:

            if self.country not in self.locFactors:
                raise ValueError(
                    f"Country not found: {self.country}. "
                    f"Available countries: {list(self.locFactors.keys())}"
                )

            loc_factor = self.locFactors[self.country]
            if isinstance(loc_factor, dict):
                if self.region in loc_factor:
                    return loc_factor[self.region]
                else:
                    raise ValueError(
                        f"Region not found: {self.region}. "
                        f"Available regions: {list(loc_factor.keys())}"
                    )
            return loc_factor

        self.isbl = (
            sum(
                equipment.direct_cost
                for equipment in self.equipment_list
            )
            * location_factors()
            * fc
        )

        if print_results:
            # Print the resultS
            print("ISBL cost estimation")
            print("===================================")
            for equipment in self.equipment_list:
                print(
                    f"  - {equipment.name}: ${equipment.direct_cost:,.2f}"
                )
            print("===================================")
            print(f"Total ISBL: ${self.isbl:,.2f}")
        else:
            return self.isbl

    def calculate_fixed_capital(
        self,
        fc=None,
        additional_capex: bool = False,
        print_results=False,
    ):

        if fc is None:
            self.fc = 1.0
        else:
            self.fc = fc
        self.calculate_isbl(self.fc)

        if self.process_type not in self.processTypes:
            raise ValueError(
                f"Unsupported process_type '{self.process_type}'. "
                f"Valid types: {list(self.processTypes)}"
            )

        params = self.processTypes[self.process_type]
        self.osbl = params["OS"] * self.isbl
        self.dne = params["DE"] * (self.isbl + self.osbl)
        self.contigency = params["X"] * (
            self.isbl + self.osbl
        )
        self.fixed_capital = (
            self.isbl
            + self.osbl
            + self.dne
            + self.contigency
        )

        if print_results:
            if (
                additional_capex
                and self.additional_capex_cost is not None
            ):
                # Print the results
                print("Capital cost estimation")
                print("===================================")
                print(f"ISBL: ${self.isbl:,.2f}")
                print(f"OSBL: ${self.osbl:,.2f}")
                print(
                    f"Design and engineering: ${self.dne:,.2f}"
                )
                print(
                    f"Contingency: ${self.contigency:,.2f}"
                )
                print(
                    f"Additional CAPEX: "
                    f"${sum(self.additional_capex_cost):,.2f}"
                )
                print("===================================")
                print(
                    f"Fixed capital investment: "
                    f"${self.fixed_capital
                        + sum(self.additional_capex_cost):,.2f}"
                )
            else:
                # Print the results
                print("Capital cost estimation")
                print("===================================")
                print(f"ISBL: ${self.isbl:,.2f}")
                print(f"OSBL: ${self.osbl:,.2f}")
                print(
                    f"Design and engineering: ${self.dne:,.2f}"
                )
                print(
                    f"Contingency: ${self.contigency:,.2f}"
                )
                print("===================================")
                print(
                    f"Fixed capital investment: ${self.fixed_capital:,.2f}"
                )
        else:
            return self.fixed_capital

    def calculate_variable_opex(self, print_results=False):
        self.variable_production_costs = 0
        self.variable_opex_breakdown = {}

        for (
            item,
            details,
        ) in self.variable_opex_inputs.items():
            consumption = details.get("consumption", 0)
            price = details.get("price", 0)

            cost = (
                consumption
                * price
                * 365
                * self.plant_utilization
            )
            self.variable_opex_breakdown[item] = cost
            self.variable_production_costs += cost

        if print_results:
            print("Variable production costs estimation")
            print("===================================")
            for (
                item,
                cost,
            ) in self.variable_opex_breakdown.items():
                item_name = item.replace(
                    "_", " "
                ).capitalize()
                print(
                    f"  - {item_name}: ${cost:,.2f} per year"
                )
            print("===================================")
            print(
                f"Total Variable OPEX: "
                f"${self.variable_production_costs:,.2f} per year"
            )
        else:
            return self.variable_production_costs

    def calculate_revenue(self, print_results=False):
        self.revenue = 0
        self.revenue_breakdown = {}

        self.main_product = (
            next(iter(self.plant_products))
            if self.plant_products
            else None
        )

        for product, details in self.plant_products.items():
            production = details.get("production", 0)
            price = details.get("price", 0)

            revenue = (
                production
                * price
                * 365
                * self.plant_utilization
            )
            self.revenue_breakdown[product] = revenue
            self.revenue += revenue

        if print_results:
            print("Revenue estimation")
            print("===================================")
            for (
                product,
                revenue,
            ) in self.revenue_breakdown.items():
                product_name = product.replace(
                    "_", " "
                ).capitalize()
                print(
                    f"  - {product_name}: ${revenue:,.2f} per year"
                )
            print("===================================")
            print(
                f"Total Revenue: ${self.revenue:,.2f} per year"
            )
        else:
            return self.revenue

    def count_process_steps(
        self,
        equipments,
        target_process_types,
        excluded_cats=None,
    ):
        if excluded_cats is None:
            excluded_cats = {}
        count = 0
        for equipment in equipments:
            if (
                equipment.process_type
                in target_process_types
                and equipment.category not in excluded_cats
            ):
                count += 1
        return count

    def calculate_operators_per_shift(
        self, no_fluid_process=None, no_solid_process=None
    ):
        if self.operators_per_shift is not None:
            return self.operators_per_shift
        else:
            if no_fluid_process is None:
                no_fluid_process = self.count_process_steps(
                    self.equipment_list,
                    {"Fluids", "Mixed"},
                    {"Pumps", "Pressure vessels"},
                )
            if no_solid_process is None:
                no_solid_process = self.count_process_steps(
                    self.equipment_list,
                    {"Solids", "Mixed"},
                    {"Pumps", "Pressure vessels"},
                )

            if no_solid_process > 2:
                raise ValueError(
                    "Number of solid processes needs "
                    "to be less than or equal to 2."
                )

            operators_per_shifts = (
                6.29
                + 31.7 * (no_solid_process**2)
                + 0.23 * no_fluid_process
            ) ** 0.5
            return operators_per_shifts

    def calculate_operators_hired(
        self, no_fluid_process=None, no_solid_process=None
    ):
        if self.operators_hired is not None:
            return self.operators_hired

        else:
            operators_per_shifts = (
                self.calculate_operators_per_shift(
                    no_fluid_process, no_solid_process
                )
            )

            operating_shifts_per_year = (
                365 * self.operating_shifts_per_day
            )

            working_shifts_per_year = (
                self.working_weeks_per_year
                * self.working_shifts_per_week
            )

            operators_hired = math.ceil(
                operators_per_shifts
                * operating_shifts_per_year
                / working_shifts_per_year
            )
            return operators_hired

    def calculate_operating_labor(
        self, no_fluid_process=None, no_solid_process=None
    ):
        operators_hired = self.calculate_operators_hired(
            no_fluid_process, no_solid_process
        )

        working_shifts_per_year = (
            self.working_weeks_per_year
            * self.working_shifts_per_week
        )
        working_hours_per_year = working_shifts_per_year * (
            24 / self.operating_shifts_per_day
        )

        rate_cfg = self.operator_hourly_rate
        if isinstance(rate_cfg, dict):
            rate = rate_cfg.get("rate", 38.11)
        else:
            rate = (
                38.11
                if rate_cfg is None
                else float(rate_cfg)
            )

        self.operating_labor_costs = (
            operators_hired * working_hours_per_year * rate
        )
        return self.operating_labor_costs

    def calculate_fixed_opex(
        self, fp=None, print_results=False
    ):
        if fp is None:
            self.fp = 1.0
        else:
            self.fp = fp

        self.calculate_fixed_capital(fc=self.fc)
        self.calculate_variable_opex()
        self.calculate_operating_labor()
        self.supervision_costs = (
            0.25 * self.operating_labor_costs
        )
        self.direct_salary_overhead = 0.5 * (
            self.operating_labor_costs
            + self.supervision_costs
        )
        self.laboratory_charges = (
            0.10 * self.operating_labor_costs
        )
        self.maintenance_costs = 0.05 * self.isbl
        self.taxes_insurance_costs = 0.015 * self.isbl
        self.rent_of_land_costs = 0.015 * (
            self.isbl + self.osbl
        )
        self.environmental_charges = 0.01 * (
            self.isbl + self.osbl
        )
        self.operating_supplies = 0.009 * self.isbl
        self.general_plant_overhead = 0.65 * (
            self.operating_labor_costs
            + self.supervision_costs
            + self.direct_salary_overhead
        )

        if self.working_capital is not None:
            self.interest_working_capital = (
                self.working_capital * self.interest_rate
            )
        else:
            self.working_capital = 0.15 * self.fixed_capital
            self.interest_working_capital = (
                self.working_capital * self.interest_rate
            )

        self.fixed_production_costs = (
            self.operating_labor_costs
            + self.supervision_costs
            + self.direct_salary_overhead
            + self.laboratory_charges
            + self.maintenance_costs
            + self.taxes_insurance_costs
            + self.rent_of_land_costs
            + self.environmental_charges
            + self.operating_supplies
            + self.general_plant_overhead
            + self.interest_working_capital
        )

        cash_cost_of_production = (
            self.variable_production_costs
            + self.fixed_production_costs
        ) / (1 - 0.07)

        self.patents_royalties = (
            0.02 * cash_cost_of_production
        )
        self.distribution_selling_costs = (
            0.02 * cash_cost_of_production
        )
        self.RnD_costs = 0.03 * cash_cost_of_production

        self.fixed_production_costs += (
            self.patents_royalties
            + self.distribution_selling_costs
            + self.RnD_costs
        )
        self.fixed_production_costs *= self.fp

        if print_results:
            # Print the results
            print("Fixed production costs estimation")
            print("===================================")
            print(
                f"Operating labor costs: "
                f"${self.operating_labor_costs:,.2f} per year"
            )
            print(
                f"Supervision costs: "
                f"${self.supervision_costs:,.2f} per year"
            )
            print(
                f"Direct salary overhead: "
                f"${self.direct_salary_overhead:,.2f} per year"
            )
            print(
                f"Laboratory charges: "
                f"${self.laboratory_charges:,.2f} per year"
            )
            print(
                f"Maintenance costs: "
                f"${self.maintenance_costs:,.2f} per year"
            )
            print(
                f"Taxes and insurance costs: "
                f"${self.taxes_insurance_costs:,.2f} per year"
            )
            print(
                f"Rent of land costs: "
                f"${self.rent_of_land_costs:,.2f} per year"
            )
            print(
                f"Environmental charges: "
                f"${self.environmental_charges:,.2f} per year"
            )
            print(
                f"Operating supplies: "
                f"${self.operating_supplies:,.2f} per year"
            )
            print(
                f"General plant overhead: "
                f"${self.general_plant_overhead:,.2f} per year"
            )
            print(
                f"Interest on working capital: "
                f"${self.interest_working_capital:,.2f} per year"
            )
            print(
                f"Patents and royalties: "
                f"${self.patents_royalties:,.2f} per year"
            )
            print(
                f"Distribution and selling costs: "
                f"${self.distribution_selling_costs:,.2f} per year"
            )
            print(
                f"R&D costs: ${self.RnD_costs:,.2f} per year"
            )
            print("===================================")
            print(
                f"Fixed OPEX: ${self.fixed_production_costs:,.2f} per year"
            )

        else:
            return self.fixed_production_costs

    def calculate_cash_flow(
        self, print_results: bool = False
    ):

        # 0) Upstream calcs (capital, opex breakdowns)
        self.calculate_fixed_capital(fc=self.fc)
        self.calculate_variable_opex()
        self.calculate_fixed_opex(fp=self.fp)
        self.calculate_revenue()

        # --- Normalize shapes ---
        lifetime = np.atleast_1d(
            self.project_lifetime
        ).astype(int)
        if np.any(lifetime < 3):
            raise ValueError(
                "All project_lifetime values must be ≥3."
            )
        n_samples = lifetime.shape[0]
        n_years = np.max(lifetime)

        fixed_capital = np.atleast_1d(
            self.fixed_capital
        ).astype(float)
        fixed_opex = np.atleast_1d(
            self.fixed_production_costs
        ).astype(float)
        var_opex = np.atleast_1d(
            self.variable_production_costs
        ).astype(float)
        interest = np.atleast_1d(self.interest_rate).astype(
            float
        )

        # Broadcast all scalars to same length
        def broadcast(x):
            return np.broadcast_to(x, n_samples)

        fixed_capital, fixed_opex, var_opex, interest = map(
            broadcast,
            (fixed_capital, fixed_opex, var_opex, interest),
        )

        # --- Initialize result arrays ---
        shape = (n_samples, n_years)
        capex = np.zeros(shape)
        main_revenue = np.zeros(shape)
        side_revenue = np.zeros(shape)
        revenue = np.zeros(shape)
        cash_cost = np.zeros(shape)
        gross_profit = np.zeros(shape)
        depreciation = np.zeros(shape)
        taxable_income = np.zeros(shape)
        tax_paid = np.zeros(shape)
        cash_flow = np.zeros(shape)
        prod_array = np.zeros(shape)

        # --- CAPEX profile (30/60/10) + WC draw/release ---
        for yr, frac in zip([0, 1, 2], [0.3, 0.6, 0.1]):
            if yr < n_years:
                capex[:, yr] += fixed_capital * frac
        if 2 < n_years:
            capex[:, 2] += self.working_capital
        capex[:, -1] -= self.working_capital

        # --- Add additional CAPEX at specified years ---
        if (
            self.additional_capex_years is not None
            and self.additional_capex_cost is not None
        ):
            additional_capex_years = np.atleast_1d(
                self.additional_capex_years
            ).astype(int)
            additional_capex_cost = np.atleast_1d(
                self.additional_capex_cost
            ).astype(float)

            # Check if the number of years matches the number of costs
            if (
                additional_capex_years.shape[0]
                != additional_capex_cost.shape[0]
            ):
                raise ValueError(
                    "The number of additional_capex_years must "
                    "match the number of additional_capex_costs."
                )

            for i, year in enumerate(
                additional_capex_years
            ):
                # Ignore invalid years
                if year < 1 or year > n_years:
                    continue

                # Apply only to samples whose lifetime includes this year
                alive_mask = lifetime >= year

                # Arrays are 0-indexed; NumPy will broadcast the scalar cost
                capex[
                    alive_mask, year - 1
                ] += additional_capex_cost[i]

        # --- Production ramp ---
        if (
            not self.plant_products
            or self.main_product is None
        ):
            raise ValueError(
                "No plant_products defined; "
                "cannot build cash flow / production profile."
            )

        self.daily_prod = self.plant_products[
            self.main_product
        ]["production"]
        nameplate = (
            self.daily_prod * 365.0 * self.plant_utilization
        )
        ramp = np.concatenate(
            ([0, 0, 0.4, 0.8], np.ones(max(0, n_years - 4)))
        )
        ramp = ramp[:n_years]

        # --- Revenue & cost arrays ---
        for yr in range(n_years):
            prod = nameplate * ramp[yr]
            prod_array[:, yr] = prod
            main_prod_price = self.plant_products[
                self.main_product
            ].get("price")
            if main_prod_price is None:
                main_revenue[:, yr] = 0
            else:
                main_revenue[:, yr] = prod * main_prod_price
            side_revenue[:, yr] = sum(
                self.plant_products[p]["production"]
                * 365.0
                * self.plant_utilization
                * ramp[yr]
                * self.plant_products[p].get("price", 0)
                for p in self.plant_products
                if p != self.main_product
            )
            revenue[:, yr] = (
                main_revenue[:, yr] + side_revenue[:, yr]
            )
            cash_cost[:, yr] = (
                fixed_opex + var_opex * ramp[yr]
            )
            gross_profit[:, yr] = (
                revenue[:, yr] - cash_cost[:, yr]
            )

        # --- Depreciation (each sample has its own config) ---
        dep_cfg = getattr(self, "depreciation", None)
        for i in range(n_samples):
            capex_dict = {
                0: 0.3 * fixed_capital[i],
                1: 0.6 * fixed_capital[i],
                2: 0.1 * fixed_capital[i],
            }
            depreciation[i, : lifetime[i]] = (
                build_depreciation_array(
                    project_life=lifetime[i],
                    capex_by_year=capex_dict,
                    dep_cfg=dep_cfg,
                )
            )

        # --- Tax and cash flow (with 1-year lag) ---
        for yr in range(n_years):
            taxable_income[:, yr] = (
                gross_profit[:, yr] - depreciation[:, yr]
            )
            if yr == 0:
                tax_paid[:, yr] = 0
            else:
                prev = taxable_income[:, yr - 1]
                tax_paid[:, yr] = np.where(
                    prev > 0, self.tax_rate * prev, 0
                )
            cash_flow[:, yr] = (
                gross_profit[:, yr]
                - tax_paid[:, yr]
                - capex[:, yr]
            )

        # --- Save arrays to instance ---
        self.capital_cost_array = capex
        self.side_revenue_array = side_revenue
        self.main_revenue_array = main_revenue
        self.revenue_array = revenue
        self.cash_cost_array = cash_cost
        self.gross_profit_array = gross_profit
        self.depreciation_array = depreciation
        self.taxable_income_array = taxable_income
        self.tax_paid_array = tax_paid
        self.cash_flow = cash_flow
        self.prod_array = prod_array

        # --- Optional: return formatted summary if scalar case ---
        if print_results and n_samples == 1:
            years = np.arange(1, n_years + 1)
            data = {
                "Year": years,
                "Capital cost": capex[0],
                "Revenue": revenue[0],
                "Cash cost": cash_cost[0],
                "Gross profit": gross_profit[0],
                "Depreciation": depreciation[0],
                "Taxable income": taxable_income[0],
                "Tax paid": tax_paid[0],
                "Cash flow": cash_flow[0],
            }
            df = pd.DataFrame(data)
            fmt = {
                c: "${:,.2f}"
                for c in df.columns
                if c not in ["Year"]
            }
            return df.style.format(fmt)

    def calculate_npv(self, print_results: bool = False):
        # Ensure 2D cash_flow: [n_scenarios, n_years]
        cf = np.asarray(self.cash_flow, dtype=float)
        if cf.ndim == 1:
            cf = cf[None, :]  # [1, n_years]
        n_scenarios, n_years = cf.shape

        years = np.arange(1, n_years + 1, dtype=float)

        # Interest rate: scalar or per-scenario
        r = np.atleast_1d(self.interest_rate).astype(float)
        if r.size == 1:
            # Same rate for all scenarios
            discount_factors = (
                1.0 + r[0]
            ) ** years  # [n_years]
        else:
            if r.size != n_scenarios:
                raise ValueError(
                    "interest_rate must be scalar or have length equal to "
                    "the number of scenarios in cash_flow."
                )
            # Per-scenario rates
            discount_factors = (1.0 + r)[:, None] ** years[
                None, :
            ]  # [n_scenarios, n_years]

        # Broadcast division: cf / discount_factors
        pv_array = cf / discount_factors
        npv_array = np.cumsum(pv_array, axis=-1)

        self.pv_array = (
            pv_array  # shape [n_scenarios, n_years]
        )
        self.npv_array = (
            npv_array  # shape [n_scenarios, n_years]
        )

        if print_results:
            print(
                "Year | Present Value (PV) | Cumulative NPV"
            )
            print(
                "-------------------------------------------"
            )
            pv_to_print = pv_array[0]
            npv_to_print = npv_array[0]
            for year, pv, npv in zip(
                range(1, n_years + 1),
                pv_to_print,
                npv_to_print,
            ):
                print(
                    f"{year:4d} | ${float(pv):15,.2f} | ${float(npv):15,.2f}"
                )
            return

        # Final-year NPV per scenario
        final_npv = npv_array[:, -1]
        if final_npv.size == 1:
            return float(final_npv[0])
        return final_npv

    def calculate_levelized_cost(self, print_results=False):
        self.calculate_fixed_capital(
            fc=1.0 if self.fc is None else self.fc
        )
        self.calculate_variable_opex()
        self.calculate_fixed_opex(
            fp=1.0 if self.fp is None else self.fp
        )
        self.calculate_revenue()
        self.calculate_cash_flow()

        n_components = (
            len(self.project_lifetime)
            if isinstance(
                self.project_lifetime, (list, np.ndarray)
            )
            else int(self.project_lifetime)
        )

        capital_cost, prod, cash_cost, side_rev = (
            self.capital_cost_array,
            self.prod_array,
            self.cash_cost_array,
            self.side_revenue_array,
        )
        (
            disc_capex,
            disc_opex,
            disc_prod,
            disc_side_rev,
        ) = (
            np.zeros(n_components),
            np.zeros(n_components),
            np.zeros(n_components),
            np.zeros(n_components),
        )

        if isinstance(
            self.project_lifetime, (list, np.ndarray)
        ):
            for i in range(n_components):
                for year in range(len(cash_cost[i])):
                    discount_factor = (
                        1 + self.interest_rate[i]
                    ) ** (year + 1)
                    disc_capex[year] += (
                        capital_cost[i][year]
                    ) / discount_factor
                    disc_opex[year] += (
                        cash_cost[i][year]
                    ) / discount_factor
                    disc_side_rev[year] += (
                        side_rev[i][year]
                    ) / discount_factor
                    disc_prod[year] += (
                        prod[i][year] / discount_factor
                    )
        else:
            for year in range(n_components):
                disc_capex[year] = (
                    capital_cost[0][year]
                ) / ((1 + self.interest_rate) ** (year + 1))
                disc_opex[year] = (cash_cost[0][year]) / (
                    (1 + self.interest_rate) ** (year + 1)
                )
                disc_side_rev[year] = (
                    side_rev[0][year]
                ) / ((1 + self.interest_rate) ** (year + 1))
                disc_prod[year] = prod[0][year] / (
                    (1 + self.interest_rate) ** (year + 1)
                )

        self.levelized_cost = max(
            np.sum(disc_capex + disc_opex - disc_side_rev)
            / np.sum(disc_prod),
            0,
        )

        if print_results:
            print(
                f"Levelized cost: ${self.levelized_cost:,.3f}/unit"
            )
        else:
            return self.levelized_cost

    def calculate_payback_time(
        self, additional_capex=False, print_results=False
    ):
        revenue, cash_flow = (
            self.revenue_array,
            self.cash_flow,
        )

        revenue_generating_years = cash_flow[revenue > 0]

        if len(revenue_generating_years) == 0:
            self.payback_time = float("nan")
        else:
            if (
                additional_capex
                and self.additional_capex_cost is not None
            ):
                total_fixed_capital = (
                    self.fixed_capital
                    + sum(self.additional_capex_cost)
                )
            else:
                total_fixed_capital = self.fixed_capital
            average_annual_cash_flow = np.mean(
                revenue_generating_years
            )
            self.payback_time = (
                total_fixed_capital
                / average_annual_cash_flow
                if average_annual_cash_flow > 0
                else float("nan")
            )

        if print_results:
            print(
                f"Payback time: {self.payback_time:.2f} years"
            )
        else:
            return self.payback_time

    def calculate_roi(
        self, additional_capex=False, print_results=False
    ):
        net_profit = (
            self.gross_profit_array - self.tax_paid_array
        )
        if (
            additional_capex
            and self.additional_capex_cost is not None
        ):
            total_investment = (
                self.fixed_capital
                + sum(self.additional_capex_cost)
                + self.working_capital
            )
        else:
            total_investment = (
                self.fixed_capital + self.working_capital
            )

        self.roi = (
            np.sum(net_profit)
            * 100
            / (
                self.project_lifetime
                * np.sum(total_investment)
            )
        )

        if print_results:
            print(f"Return of investment: {self.roi:.2f}%")
        else:
            return self.roi

    def calculate_irr(self, print_results: bool = False):
        cf = np.asarray(self.cash_flow, dtype=float)
        n = cf.size
        if n == 0:
            self.irr = float("nan")
            if print_results:
                print(
                    "Internal Rate of Return: undefined (empty cash flow)."
                )
            return self.irr

        # Must have at least one negative and one positive cash flow
        if not (np.any(cf < 0) and np.any(cf > 0)):
            self.irr = float("nan")
            if print_results:
                print(
                    "Internal Rate of Return: undefined "
                    "(no sign change in cash flows)."
                )
            return self.irr

        years = np.arange(n, dtype=float) + 1

        def npv_at(r: float) -> float:
            # r <= -1 is out of domain
            if r <= -1.0:
                return np.inf
            return float(np.sum(cf / (1.0 + r) ** years))

        # 1) Scan for a bracket with a sign change in NPV
        #    (dense near -1, then spread out to high positives)
        grid = np.concatenate(
            [
                np.linspace(
                    -0.95, -0.01, 120, endpoint=True
                ),
                np.array(
                    [0.0]
                ),  # allow exact 0 as a candidate
                np.linspace(0.01, 10.0, 240, endpoint=True),
            ]
        )

        npv_vals = np.array([npv_at(r) for r in grid])

        # Find adjacent points where NPV changes sign (ignore infinities)
        bracket = None
        for i in range(len(grid) - 1):
            a, b = grid[i], grid[i + 1]
            fa, fb = npv_vals[i], npv_vals[i + 1]
            if not np.isfinite(fa) or not np.isfinite(fb):
                continue
            if fa == 0.0:
                bracket = (
                    a - 1e-6,
                    a + 1e-6,
                )  # degenerate bracket around exact root
                break
            if np.sign(fa) != np.sign(fb):
                bracket = (a, b)
                break

        if bracket is None:
            # Fallback: try widening upper bound up to, say, 1000%
            a = 0.01
            b = 10.0
            fa = npv_at(a)
            fb = npv_at(b)
            while (
                np.isfinite(fb)
                and np.sign(fa) == np.sign(fb)
                and b < 10.0
            ):
                b *= 1.5
                fb = npv_at(b)
            bracket = (
                (a, b)
                if np.isfinite(fb)
                and np.sign(fa) != np.sign(fb)
                else None
            )

        if bracket is None:
            self.irr = float("nan")
            if print_results:
                print(
                    "Internal Rate of Return: "
                    "undefined (could not bracket a root)."
                )
            return self.irr

        # 2) Root finding with Brent's method on the bracket
        try:
            sol = root_scalar(
                npv_at,
                bracket=bracket,
                method="brentq",
                xtol=1e-10,
                rtol=1e-10,
                maxiter=200,
            )
            self.irr = (
                sol.root
                if sol.converged and math.isfinite(sol.root)
                else float("nan")
            )
        except Exception:
            self.irr = float("nan")

        if print_results:
            if math.isfinite(self.irr):
                print(
                    f"Internal Rate of Return: {self.irr * 100:.2f}%"
                )
            else:
                print("Internal Rate of Return: undefined")
        else:
            return self.irr

    def __str__(self):
        """Pretty string representation of all plant configuration inputs."""

        # Helper for formatting dicts cleanly
        import json

        def fmt(obj):
            if obj is None:
                return "None"
            if isinstance(obj, dict):
                return json.dumps(obj, indent=4)
            return str(obj)

        # Equipment formatting
        if self.equipment_list:
            eq_strings = []
            for i, eq in enumerate(self.equipment_list):
                label = getattr(
                    eq,
                    "name",
                    f"{eq.__class__.__name__}({i})",
                )
                cost = getattr(eq, "direct_cost", "N/A")
                eq_strings.append(
                    f"    - {label}: direct_cost={cost}"
                )
            eq_block = "\n".join(eq_strings)
        else:
            eq_block = "    None"

        return (
            f"ProcessPlant Configuration\n"
            f"{'-'*40}\n"
            f"Plant Name:                 {self.name}\n"
            f"Process Type:               {self.process_type}\n"
            f"Country / Region:           {self.country} / {self.region}\n"
            f"Interest Rate:              {self.interest_rate}\n"
            f"Project Lifetime (years):   {self.project_lifetime}\n"
            f"Plant Utilization:          {self.plant_utilization}\n"
            f"Tax Rate:                   {self.tax_rate}\n"
            f"Working Capital:            {self.working_capital}\n"
            f"Depreciation Settings:      {fmt(self.depreciation)}\n"
            f"\n"
            f"Operator Labor Inputs\n"
            f"  Hourly Rate:              {fmt(self.operator_hourly_rate)}\n"
            f"  Operators per Shift:      {self.operators_per_shift}\n"
            f"  Operators Hired:          {self.operators_hired}\n"
            f"  Working Weeks / Year:     {self.working_weeks_per_year}\n"
            f"  Working Shifts / Week:    {self.working_shifts_per_week}\n"
            f"  Operating Shifts / Day:   {self.operating_shifts_per_day}\n"
            f"\n"
            f"Products\n"
            f"{fmt(self.plant_products)}\n"
            f"\n"
            f"Variable OPEX Inputs:\n{fmt(self.variable_opex_inputs)}\n"
            f"\n"
            f"Additional CAPEX:\n"
            f"  Years:                    {self.additional_capex_years}\n"
            f"  Costs:                    {self.additional_capex_cost}\n"
            f"\n"
            f"Equipment List:\n{eq_block}\n"
            f"\n"
            f"Cost Multipliers:\n"
            f"  fc (installed cost factor): {self.fc}\n"
            f"  fp (fixed OPEX factor):     {self.fp}\n"
        )


# Depreciation models
DepMethod = Literal[
    "straight_line", "declining_balance", "macrs"
]

# MACRS half-year convention percentage tables (IRS Pub 946).
# https://www.irs.gov/pub/irs-pdf/p946.pdf
# Values are FRACTIONS (not %). Sum to 1.0 within rounding.
_MACRS_HALF_YEAR: Dict[int, List[float]] = {
    3: [0.3333, 0.4445, 0.1481, 0.0741],
    5: [0.2000, 0.3200, 0.1920, 0.1152, 0.1152, 0.0576],
    7: [
        0.1429,
        0.2449,
        0.1749,
        0.1249,
        0.0893,
        0.0892,
        0.0893,
        0.0446,
    ],
    10: [
        0.1000,
        0.1800,
        0.1440,
        0.1152,
        0.0922,
        0.0737,
        0.0655,
        0.0655,
        0.0656,
        0.0328,
    ],
    15: [
        0.0500,
        0.0950,
        0.0855,
        0.0770,
        0.0693,
        0.0623,
        0.0590,
        0.0590,
        0.0591,
        0.0590,
        0.0591,
        0.0590,
        0.0591,
        0.0590,
        0.0591,
        0.0295,
    ],
    20: [
        0.0375,
        0.07219,
        0.06677,
        0.06177,
        0.05713,
        0.05285,
        0.04888,
        0.04522,
        0.04462,
        0.04461,
        0.04462,
        0.04461,
        0.04462,
        0.04461,
        0.04462,
        0.04461,
        0.04462,
        0.04461,
        0.04462,
        0.04461,
        0.02231,
    ],
}


class DepreciationConfig:
    """
    Configuration for asset depreciation calculations.

    This class defines the parameters needed to compute depreciation
    using various methods.

    Attributes:
        method (DepMethod): The depreciation method to use.
        Defaults to "straight_line".
        Options: "straight_line", "declining_balance", "macrs".
        life (Optional[int]): The useful life of the asset in years.
            Used by straight_line and declining_balance methods.
            Defaults to None.
        db_factor (float): The declining balance factor (multiplier).
            Only used by the declining_balance method. Defaults to 2.0.
        salvage_fraction (float): The salvage value as a fraction
            of the initial cost.
            Used by straight_line and declining_balance methods.
            Defaults to 0.0.
        macrs_class (int): The MACRS property class (1-20).
            Only used by the macrs method. Defaults to 7.
        convention (str): The depreciation convention for MACRS.
            Only used by the macrs method. Defaults to "half_year".
        service_start_year (int): The year index (starting from 0)
            when the asset is placed in service. Defaults to 2.
    """

    method: DepMethod = "straight_line"
    life: Optional[int] = (
        None  # straight_line / declining_balance
    )
    db_factor: float = 2.0  # declining_balance only
    salvage_fraction: float = (
        0.0  # straight_line / declining_balance only
    )
    macrs_class: int = 7  # macrs only
    convention: str = "half_year"  # macrs only
    service_start_year: int = (
        2  # year index when asset is placed in service
    )


def _normalize_dep_config(
    project_life: int, dep_cfg: Optional[dict]
) -> DepreciationConfig:
    """
    Normalize and validate a depreciation configuration.
    This function creates a DepreciationConfig object from a dictionary of
    configuration parameters, applies sensible defaults, and validates
    the configuration based on the depreciation method and project life.
    Args:
        project_life (int): The expected life of the project in years.
            Used to set a sensible default for the depreciation life
            if not specified.
        dep_cfg (Optional[dict]): A dictionary containing depreciation
            configuration parameters. Keys should correspond to
            DepreciationConfig attributes. If None, defaults are applied.
    Returns:
        DepreciationConfig: A validated depreciation configuration object
            with all required parameters set.
    Raises:
        ValueError: If the MACRS convention is not "half_year" when
            using the "macrs" depreciation method.
        ValueError: If the specified MACRS class is not supported.
            Only classes defined in _MACRS_HALF_YEAR are accepted.
    Notes:
        - If cfg.life is not specified, it defaults to the minimum of
            project_life and 15 years.
        - Only the "half_year" MACRS convention is currently supported.
        - MACRS class validation only occurs when the depreciation
            method is "macrs".
    """
    cfg = DepreciationConfig()
    if dep_cfg:
        for k, v in dep_cfg.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    # Sensible defaults
    if cfg.life is None:
        cfg.life = min(project_life, 15)
    if cfg.method == "macrs":
        if cfg.convention != "half_year":
            raise ValueError(
                "Only half_year MACRS convention is supported currently."
            )
        if cfg.macrs_class not in _MACRS_HALF_YEAR:
            raise ValueError(
                f"Unsupported MACRS class {cfg.macrs_class}. "
                f"Choose one of {sorted(_MACRS_HALF_YEAR.keys())}."
            )
    return cfg


def _straight_line_schedule(
    basis: float,
    life: int,
    salvage_frac: float,
    horizon: int,
) -> np.ndarray:
    """
    Calculate a straight-line depreciation schedule over a given horizon.

    This function computes an annual depreciation amount based
    on the asset basis, useful life, and salvage value, then
    creates a schedule array that distributes this depreciation a
    cross the analysis horizon.

    Args:
        basis: The initial cost or basis of the asset.
        life: The useful life of the asset in years.
        salvage_frac: The salvage value as a fraction of the basis (0 to 1).
        horizon: The analysis horizon in years.

    Returns:
        A numpy array of shape (horizon,) containing the annual depreciation
        amounts. The array is zero-filled for years beyond the asset's useful
        life.

    Notes:
        - Depreciation is distributed equally across the asset's useful life.
        - A rounding correction is applied to the final depreciation year to
        ensure the sum of the schedule equals the total depreciable amount.
    """
    salvage = basis * salvage_frac
    dep_total = basis - salvage
    annual = dep_total / life
    sched = np.zeros(horizon, dtype=float)
    years = min(life, horizon)
    sched[:years] = annual
    # Small rounding fix to ensure sum equals dep_total
    diff = dep_total - sched.sum()
    if abs(diff) > 1e-6 and years > 0:
        sched[years - 1] += diff
    return sched


def _declining_balance_schedule(
    basis: float,
    life: int,
    factor: float,
    salvage_frac: float,
    horizon: int,
) -> np.ndarray:
    """
    Calculate a declining balance depreciation schedule with salvage
    value protection.

    This function computes a depreciation schedule using a declining
    balance method that switches to straight-line depreciation
    when beneficial, ensuring the asset depreciates from its basis
    to its salvage value over the specified life.

    Parameters
    ----------
    basis : float
        The initial cost or book value of the asset.
    basis : float
        The initial cost or book value of the asset.
    life : int
        The useful life of the asset in years.
    factor : float
        The declining balance factor
        (typically 1 or 2 for standard or double declining balance).
    salvage_frac : float
        The salvage value as a fraction of the basis
        (e.g., 0.1 for 10% salvage).
    horizon : int
        The time horizon in years for which to generate the schedule.

    Returns
    -------
    np.ndarray
        A 1D array of shape (horizon,) containing the depreciation
        amount for each year. Values are zero for years beyond
        the asset's life.

    Notes
    -----
    - The depreciation method automatically switches between declining balance
        and straight-line when straight-line yields a higher depreciation
        amount.
    - The schedule respects the salvage value, preventing
        depreciation below it.
    - A rounding correction is applied to ensure the total depreciation equals
        (basis - salvage) within numerical precision.
    """
    salvage = basis * salvage_frac
    remaining = basis
    sched = np.zeros(horizon, dtype=float)
    for y in range(min(life, horizon)):
        # Candidate DB amount
        db = remaining * (factor / life)
        # Candidate SL amount on remaining (including salvage protection)
        years_left = life - y
        sl_total_left = max(0.0, remaining - salvage)
        sl = (
            sl_total_left / years_left
            if years_left > 0
            else 0.0
        )
        dep = max(
            0.0, min(max(db, sl), remaining - salvage)
        )  # cannot dip below salvage
        sched[y] = dep
        remaining -= dep
    # Tiny rounding correction
    diff = (basis - salvage) - sched.sum()
    if abs(diff) > 1e-6:
        last = (
            np.nonzero(sched)[0][-1] if sched.any() else 0
        )
        sched[last] += diff
    return sched


def _macrs_schedule(
    basis: float, macrs_class: int, horizon: int
) -> np.ndarray:
    """
    Generate a MACRS depreciation schedule for an asset.

    This function calculates the annual depreciation amounts using the Modified
    Accelerated Cost Recovery System (MACRS) half-year convention over a
    specified time horizon.

    Args:
        basis (float): The initial cost basis of the asset to be depreciated.
        macrs_class (int): The MACRS asset class that determines the
                        depreciation percentages and recovery period.
        horizon (int): The number of years over which to generate the
                        depreciation schedule.

    Returns:
        np.ndarray: An array of depreciation amounts for each year
                    in the horizon. The sum of all depreciation amounts
                    equals the basis (within floating-point tolerance).

    Notes:
        - If the standard MACRS schedule is shorter than the horizon,
            the schedule is padded with zeros for remaining years.
        - If the standard MACRS schedule is longer than the horizon,
            it is truncated.
        - A rounding adjustment is applied to the final year to ensure
            the total depreciation does not exceed the basis.
    """
    pct = _MACRS_HALF_YEAR[macrs_class]
    sched = np.array(pct, dtype=float) * basis
    if len(sched) < horizon:
        sched = np.pad(sched, (0, horizon - len(sched)))
    else:
        sched = sched[:horizon]
    # Rounding fix to make sure we don't exceed basis:
    if sched.sum() - basis > 1e-6:
        sched[-1] -= sched.sum() - basis
    return sched


def build_depreciation_array(
    project_life: int,
    capex_by_year: Dict[int, float],
    dep_cfg: Optional[dict] = None,
) -> np.ndarray:
    """
    Build a depreciation schedule array over the project lifecycle.
    Calculates annual depreciation amounts for capital expenditures using the
    specified depreciation method. Supports multiple depreciation methods
    including straight-line,
    declining balance, and MACRS.
    Parameters
    ----------
    project_life : int
        The total duration of the project in years.
    capex_by_year : Dict[int, float]
        Dictionary mapping year to capital expenditure amount for that year.
    dep_cfg : Optional[dict], optional
        Depreciation configuration dictionary containing method, life,
        salvage fraction, and method-specific parameters.
        If None, uses normalized default configuration.
        Default is None.
    Returns
    -------
    np.ndarray
        1D array of shape (project_life,) containing annual depreciation
        amounts. Values are floats representing depreciation in each year.
    Raises
    ------
    ValueError
        If the depreciation method specified in dep_cfg is not one of the
        supported methods: 'straight_line', 'declining_balance', or 'macrs'.
    Notes
    -----
    - Capital expenditures are placed in service starting at the configured
        service start year.
    - Depreciation schedules respect the project horizon after placement
        in service.
    - Zero amounts and expired horizons are skipped without error.
    """
    cfg = _normalize_dep_config(project_life, dep_cfg)
    dep = np.zeros(project_life, dtype=float)

    for capex_year, amount in capex_by_year.items():
        # place-in-service timing
        start = max(cfg.service_start_year, capex_year)
        horizon = max(0, project_life - start)
        if horizon <= 0 or amount == 0:
            continue

        if cfg.method == "straight_line":
            sched = _straight_line_schedule(
                amount,
                cfg.life,
                cfg.salvage_fraction,
                horizon,
            )
        elif cfg.method == "declining_balance":
            sched = _declining_balance_schedule(
                amount,
                cfg.life,
                cfg.db_factor,
                cfg.salvage_fraction,
                horizon,
            )
        elif cfg.method == "macrs":
            sched = _macrs_schedule(
                amount, cfg.macrs_class, horizon
            )
        else:
            raise ValueError(
                f"Unknown depreciation method: {cfg.method}"
            )

        dep[start: start + len(sched)] += sched

    return dep
