import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','ieee'])
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
from copy import deepcopy
from scipy.optimize import root_scalar
from scipy.stats import truncnorm, norm
from IPython.display import clear_output
from itertools import cycle

from equipment import *


class ProcessPlant:
    """
    A class to represent a process plant and perform economic calculations.

    Attributes:
    processTypes (dict): A dictionary containing process types and their parameters.
    locFactors (dict): A dictionary containing location factors for different regions.
    """

    processTypes = {
        'Solids': {'OS': 0.4, 'DE': 0.2, 'X': 0.1},
        'Fluids': {'OS': 0.3, 'DE': 0.3, 'X': 0.1},
        'Mixed': {'OS': 0.4, 'DE': 0.25, 'X': 0.1},
    }

    locFactors = {
        'United States': {'Gulf Coast': 1.00, 'East Coast': 1.04, 'West Coast': 1.07, 'Midwest': 1.02},
        'Canada': {'Ontario': 1.00, 'Fort McMurray': 1.60},
        'Mexico': 1.03, 'Brazil': 1.14, 'China': {'imported': 1.12, 'indigenous': 0.61},
        'Japan': 1.26, 'Southeast Asia': 1.12, 'Australia': 1.21, 'India': 1.02, 'Middle East': 1.07,
        'France': 1.13, 'Germany': 1.11, 'Italy': 1.14, 'Netherlands': 1.19, 'Russia': 1.53, 'United Kingdom': 1.02,
    }

    def __init__(self, configuration: dict):
        """
        Initialize the ProcessPlant with configuration, parameters, and equipment list.

        Parameters:
        configuration (dict): A dictionary containing plant and economic specifications.
        params (dict): A dictionary containing various parameters for the plant.
        equipment_list (List[Equipment]): A list of equipment objects used in the plant.
        """

        self.name = configuration['plant_name']
        self.process_type = configuration['process_type'] if 'process_type' in configuration else None
        self.country = configuration['country'] if 'country' in configuration else None
        self.region = configuration['region'] if 'region' in configuration else None
        self.equipment_list = configuration['equipment'] if 'equipment' in configuration else []
        self.variable_opex_inputs = configuration['variable_opex_inputs'] if 'variable_opex_inputs' in configuration else {}
        self.working_capital = configuration['working_capital'] if 'working_capital' in configuration else None
        self.interest_rate = configuration['interest_rate'] if 'interest_rate' in configuration else None
        self.operator_hourly_rate = configuration['operator_hourly_rate'] if 'operator_hourly_rate' in configuration else None
        self.project_lifetime = configuration['project_lifetime'] if 'project_lifetime' in configuration else None
        self.daily_prod = configuration['daily_prod'] if 'daily_prod' in configuration else 0
        self.plant_utilization = configuration['plant_utilization'] if 'plant_utilization' in configuration else 1
        self.product_price = configuration['product_price'] if 'product_price' in configuration else 0
        self.tax_rate = configuration['tax_rate'] if 'tax_rate' in configuration else 0
        self.fc = None
        self.fp = None

        self.monte_carlo_lcops = None

    def update_configuration(self, configuration: dict):
        """
        Update the configuration of the ProcessPlant instance.

        Parameters:
        configuration (dict): A dictionary containing updated plant and economic specifications.
        """
        self.name = configuration['plant_name'] if 'plant_name' in configuration else self.name
        self.process_type = configuration['process_type'] if 'process_type' in configuration else self.process_type
        self.country = configuration['country'] if 'country' in configuration else self.country
        self.region = configuration['region'] if 'region' in configuration else self.region
        self.equipment_list = configuration['equipment'] if 'equipment' in configuration else self.equipment_list
        self.working_capital = configuration['working_capital'] if 'working_capital' in configuration else self.working_capital
        self.interest_rate = configuration['interest_rate'] if 'interest_rate' in configuration else self.interest_rate
        self.operator_hourly_rate = configuration['operator_hourly_rate'] if 'operator_hourly_rate' in configuration else self.operator_hourly_rate
        self.project_lifetime = configuration['project_lifetime'] if 'project_lifetime' in configuration else self.project_lifetime
        self.daily_prod = configuration['daily_prod'] if 'daily_prod' in configuration else self.daily_prod
        self.plant_utilization = configuration['plant_utilization'] if 'plant_utilization' in configuration else self.plant_utilization
        self.product_price = configuration['product_price'] if 'product_price' in configuration else self.product_price
        self.tax_rate = configuration['tax_rate'] if 'tax_rate' in configuration else self.tax_rate

        def recursive_update(original, updates):
        # Recursively update dicts without overwriting whole subdicts.
            for key, value in updates.items():
                if isinstance(value, dict) and isinstance(original.get(key), dict):
                    recursive_update(original[key], value)
                else:
                    original[key] = value

        # Instead of replacing, update variable_opex_inputs
        if 'variable_opex_inputs' in configuration:
            if not hasattr(self, 'variable_opex_inputs') or self.variable_opex_inputs is None:
                self.variable_opex_inputs = {}
            recursive_update(self.variable_opex_inputs, configuration['variable_opex_inputs'])     


    def calculate_isbl(self, fc=1.0, print_results=False):
        """
        Calculate the Inside Battery Limits (ISBL) cost for a given plant configuration, adjusted by location factors.
        Args:
            config (dict): A dictionary containing the plant configuration with the following keys:
                - 'equipment' (list): List of equipment objects, each having a 'direct_cost' attribute.
                - 'country' (str): The country where the plant is located.
                - 'region' (str): The region within the country (if applicable).
        Returns:
            float: The total ISBL cost adjusted for the specified location.
        Raises:
            ValueError: If the specified country or region is not found in the location factors.
        """
    
        def location_factors() -> float:
            """
            Returns the location factor for a given country and optionally a region.
            Args:
                country (str): The name of the country for which to retrieve the location factor.
                region (str): The name of the region within the country (if applicable).
            Returns:
                float: The location factor corresponding to the specified country and region.
            Raises:
                ValueError: If the specified country is not found in the location factors.
                ValueError: If the specified region is not found within the country's location factors.
            Notes:
                - If the location factor for a country is a dictionary, the region must be specified and found within that dictionary.
                - If the location factor for a country is a float, the region argument is ignored.
            """

            if self.country not in self.locFactors:
                raise ValueError(f'Country not found: {self.country}. Available countries: {list(self.locFactors.keys())}')

            loc_factor = self.locFactors[self.country]
            if isinstance(loc_factor, dict):
                if self.region in loc_factor:
                    return loc_factor[self.region]
                else:
                    raise ValueError(f'Region not found: {self.region}. Available regions: {list(loc_factor.keys())}')
            return loc_factor

        self.isbl = sum(
            equipment.direct_cost for equipment in self.equipment_list
        ) * location_factors() * fc

        if print_results:
            # Print the resultS
            print(f"ISBL cost estimation: ${self.isbl:,.2f}")



    def calculate_fixed_capital(self, fc=None, print_results=False):
        """
        Calculate the fixed capital investment for a process based on its configuration.
        This function computes the inside battery limits (ISBL), outside battery limits (OSBL),
        design and engineering (DnE), contingency, and fixed capital investment for a given process
        type. The calculation uses process-specific parameters for solids, fluids, or mixed processes.
        Parameters
        ----------
        config : dict
            Configuration dictionary containing at least the key 'process_type', which must be one of
            'Solids', 'Fluids', or 'Mixed'. Additional keys may be used by the `calculate_isbl` function.
        Returns
        -------
        tuple
            A tuple containing:
                - isbl (float): Inside battery limits cost
                - osbl (float): Outside battery limits cost
                - dne (float): Design and engineering cost
                - contigency (float): Contingency cost
                - fixed_capital (float): Fixed capital investment
        Raises
        ------
        ValueError
            If the provided 'process_type' in config is not supported.
        Notes
        -----
        The function updates the input `config` dictionary in-place with the calculated values.
        """
        if fc is None:
            self.fc = 1.0
        else:
            self.fc = fc
        self.calculate_isbl(self.fc)

        if self.process_type not in self.processTypes:
            raise ValueError(f"Unsupported process_type '{self.process_type}'. Valid types: {list(self.processTypes)}")

        params = self.processTypes[self.process_type]
        self.osbl = params['OS'] * self.isbl
        self.dne = params['DE'] * (self.isbl + self.osbl)
        self.contigency = params['X'] * (self.isbl + self.osbl)
        self.fixed_capital = (self.isbl + self.osbl + self.dne + self.contigency)
        
        if print_results:
            # Print the resultS
            print("Capital cost estimation")
            print("===================================")
            print(f"ISBL: ${self.isbl:,.2f}")
            print(f"OSBL: ${self.osbl:,.2f}")
            print(f"Design and engineering: ${self.dne:,.2f}")
            print(f"Contingency: ${self.contigency:,.2f}")
            print("===================================")
            print(f"Fixed capital investment: ${self.fixed_capital:,.2f}")

    def calculate_variable_opex(self, print_results=False):
        """
        Calculate the total variable operating expenses (OPEX) based on the provided configuration.
        Args:
            config (dict): A configuration dictionary containing a 'variable_opex' key. 
                The value should be a dictionary where each key is an item name and each value is a dictionary 
                with 'consumption' (float or int) and 'price' (float or int) keys.
        Returns:
            float: The total variable production costs calculated as the sum of (consumption * price) for each item.
        Example:
            config = {
                'variable_opex': {
                    'electricity': {'consumption': 1000, 'price': 0.1},
                    'water': {'consumption': 500, 'price': 0.05}
                }
            }
            total_opex = calculate_variable_opex(config)
        """
        
        self.variable_production_costs = 0
        self.variable_opex_breakdown = {}

        for item, details in self.variable_opex_inputs.items():
            consumption = details.get('consumption', 0)
            price = details.get('price', 0)

            cost = consumption * price * 365 * self.plant_utilization
            self.variable_opex_breakdown[item] = cost
            self.variable_production_costs += cost

        if print_results:
            print("Variable production costs estimation")
            print("===================================")
            for item, cost in self.variable_opex_breakdown.items():
                item_name = item.replace("_", " ").capitalize()
                print(f"  - {item_name}: ${cost:,.2f} per year")
            print("===================================")
            print(f"Total Variable OPEX: ${self.variable_production_costs:,.2f} per year")


    def calculate_operating_labor(self):
        """
        Calculate the annual operating labor costs based on the equipment configuration.
        This function estimates the number of operators required per shift and the total annual labor cost,
        taking into account the types and numbers of process units, operator hourly rate, and working schedule.
        Args:
            config (dict): Configuration dictionary containing:
                - 'equipment' (list): List of equipment objects, each with attributes:
                    - process_type (str): Type of process ('Fluids', 'Solids', or 'Mixed').
                    - type (str): Equipment type (e.g., 'Pump', 'Vessel', 'Cyclone', etc.).
                    - num_units (int): Number of units for this equipment.
                - 'operator_hourly_rate' (float): Hourly wage rate for operators.
        Returns:
            float: Total annual operating labor costs.
        Raises:
            ValueError: If the number of solid process units exceeds 2.
        """

        def count_units_by_process_type(equipments, target_process_types):
            excluded_types = {'Pump', 'Vessel', 'Cyclone'}
            count = 0
            for equipment in equipments:
                if equipment.process_type in target_process_types and equipment.type not in excluded_types:
                    count += equipment.num_units
            return count
            
        no_fluid_process = count_units_by_process_type(self.equipment_list, {'Fluids', 'Mixed'})
        no_solid_process = count_units_by_process_type(self.equipment_list, {'Solids', 'Mixed'})

        operators_per_shifts = (6.29 + 31.7 * (no_solid_process ** 2) + 0.23 * no_fluid_process) ** 0.5

        if no_solid_process > 2:
            raise ValueError("Number of solid processes needs to be less than or equal to 2.")

        working_weeks_per_year = 49
        working_shifts_per_week = 5  # 8-hour shifts
        operating_shifts_per_year = 365 * 3

        working_shifts_per_year = working_weeks_per_year * working_shifts_per_week
        working_hours_per_year = working_shifts_per_year * 8

        self.operators_hired = math.ceil(operators_per_shifts * operating_shifts_per_year / working_shifts_per_year)
        self.operating_labor_costs = self.operators_hired * working_hours_per_year * self.operator_hourly_rate


    def calculate_fixed_opex(self, fp=None, print_results=False):
        """
        Calculate the fixed operating expenses (OPEX) for a chemical process based on the provided configuration.
        This function computes various components of fixed OPEX, including labor, supervision, overheads, maintenance, taxes, insurance, rent, environmental charges, supplies, and general plant overhead. It also accounts for interest on working capital, patents and royalties, distribution and selling costs, and R&D costs. The result is stored in the 'fixed_opex' key of the input config dictionary.
        Args:
            config (dict): A dictionary containing the following keys:
                - isbl (float): Inside battery limits capital cost.
                - osbl (float): Outside battery limits capital cost.
                - working_capital (float): Working capital amount.
                - interest_rate (float): Interest rate for working capital.
                - variable_opex (float): Variable operating expenses.
                - Any additional keys required by `calculate_operating_labor`.
        Returns:
            float: The total fixed operating expenses.
        """
        if fp is None:
            self.fp = 1.0
        else:
            self.fp = fp

        self.calculate_fixed_capital(fc=self.fc)
        self.calculate_variable_opex()
        self.calculate_operating_labor()
        self.supervision_costs = 0.25 * self.operating_labor_costs
        self.direct_salary_overhead = 0.5 * (self.operating_labor_costs + self.supervision_costs)
        self.laboratory_charges = 0.10 * self.operating_labor_costs
        self.maintenance_costs = 0.05 * self.isbl
        self.taxes_insurance_costs = 0.015 * self.isbl
        self.rent_of_land_costs = 0.015 * (self.isbl + self.osbl)
        self.environmental_charges = 0.01 * (self.isbl + self.osbl)
        self.operating_supplies = 0.009 * self.isbl
        self.general_plant_overhead = 0.65 * (self.operating_labor_costs + self.supervision_costs + self.direct_salary_overhead)

        if self.working_capital != None:
            self.interest_working_capital = self.working_capital*self.interest_rate
        else:
            self.working_capital = 0.15 * self.fixed_capital
            self.interest_working_capital = self.working_capital*self.interest_rate

        self.fixed_production_costs = (self.operating_labor_costs + self.supervision_costs + self.direct_salary_overhead
                                        + self.laboratory_charges + self.maintenance_costs + self.taxes_insurance_costs
                                        + self.rent_of_land_costs + self.environmental_charges + self.operating_supplies
                                        + self.general_plant_overhead + self.interest_working_capital)

        cash_cost_of_production = (self.variable_production_costs + self.fixed_production_costs) / (1 - 0.07)

        self.patents_royalties = 0.02 * cash_cost_of_production
        self.distribution_selling_costs = 0.02 * cash_cost_of_production
        self.RnD_costs = 0.03 * cash_cost_of_production

        self.fixed_production_costs += self.patents_royalties + self.distribution_selling_costs + self.RnD_costs
        self.fixed_production_costs *= self.fp

        if print_results:
            # Print the results
            print("Fixed production costs estimation")
            print("===================================")
            print(f"Operating labor costs: ${self.operating_labor_costs:,.2f} per year")
            print(f"Supervision costs: ${self.supervision_costs:,.2f} per year")
            print(f"Direct salary overhead: ${self.direct_salary_overhead:,.2f} per year")
            print(f"Laboratory charges: ${self.laboratory_charges:,.2f} per year")
            print(f"Maintenance costs: ${self.maintenance_costs:,.2f} per year")
            print(f"Taxes and insurance costs: ${self.taxes_insurance_costs:,.2f} per year")
            print(f"Rent of land costs: ${self.rent_of_land_costs:,.2f} per year")
            print(f"Environmental charges: ${self.environmental_charges:,.2f} per year")
            print(f"Operating supplies: ${self.operating_supplies:,.2f} per year")
            print(f"General plant overhead: ${self.general_plant_overhead:,.2f} per year")
            print(f"Interest on working capital: ${self.interest_working_capital:,.2f} per year")
            print(f"Patents and royalties: ${self.patents_royalties:,.2f} per year")
            print(f"Distribution and selling costs: ${self.distribution_selling_costs:,.2f} per year")
            print(f"R&D costs: ${self.RnD_costs:,.2f} per year")
            print("===================================")
            print(f"Fixed OPEX: ${self.fixed_production_costs:,.2f} per year")


    def calculate_cash_flow(self, print_results=False):
        
        self.calculate_fixed_capital(fc=self.fc)
        self.calculate_variable_opex()
        self.calculate_fixed_opex(fp=self.fp)
        
        # Assume self.project_lifetime is a NumPy array with shape (n_components)
        n_components = len(self.project_lifetime) if isinstance(self.project_lifetime, (list, np.ndarray)) else 1

        if isinstance(self.project_lifetime, (list, np.ndarray)):
            self.project_lifetime = np.array(self.project_lifetime, dtype=float)  # ensure numeric

            self.project_lifetime = self.project_lifetime.astype(int)

            # Check if all elements are integer-like (e.g., 3.0 is OK, 3.5 is not)
            if not np.all(np.equal(np.mod(self.project_lifetime, 1), 0)):
                raise TypeError("All values in project_lifetime must be integers.")

            if np.any(np.array(self.project_lifetime < 3)):
                raise ValueError("All project_lifetime values must be greater than 3.")
            
        else:
            self.project_lifetime = int(self.project_lifetime)
            if self.project_lifetime < 3:
                raise ValueError("Project lifetime must be greater than 3.")
            
        # Initialize arrays
        self.capital_cost_array = [np.zeros(lifetime) for lifetime in self.project_lifetime] if isinstance(self.project_lifetime, (list, np.ndarray)) else np.zeros(self.project_lifetime)
        self.prod_array = [np.zeros(lifetime) for lifetime in self.project_lifetime] if isinstance(self.project_lifetime, (list, np.ndarray)) else np.zeros(self.project_lifetime)
        self.revenue_array = [np.zeros(lifetime) for lifetime in self.project_lifetime] if isinstance(self.project_lifetime, (list, np.ndarray)) else np.zeros(self.project_lifetime)
        self.cash_cost_array = [np.zeros(lifetime) for lifetime in self.project_lifetime] if isinstance(self.project_lifetime, (list, np.ndarray)) else np.zeros(self.project_lifetime)
        self.gross_profit_array = [np.zeros(lifetime) for lifetime in self.project_lifetime] if isinstance(self.project_lifetime, (list, np.ndarray)) else np.zeros(self.project_lifetime)
        self.depreciation_array = [np.zeros(lifetime) for lifetime in self.project_lifetime] if isinstance(self.project_lifetime, (list, np.ndarray)) else np.zeros(self.project_lifetime)
        self.taxable_income_array = [np.zeros(lifetime) for lifetime in self.project_lifetime] if isinstance(self.project_lifetime, (list, np.ndarray)) else np.zeros(self.project_lifetime)
        self.tax_paid_array = [np.zeros(lifetime) for lifetime in self.project_lifetime] if isinstance(self.project_lifetime, (list, np.ndarray)) else np.zeros(self.project_lifetime)

        self.cash_flow = [np.zeros(lifetime) for lifetime in self.project_lifetime] if isinstance(self.project_lifetime, (list, np.ndarray)) else np.zeros(self.project_lifetime)

        previous_taxable_income = 0
        depreciation_counter = 0
        depreciation_duration = self.project_lifetime // 2  # array of durations
        depreciation_amount = self.fixed_capital / depreciation_duration  # array of amounts

        if isinstance(self.project_lifetime, (list, np.ndarray)):
            for i in range(n_components):
                for year in range(self.project_lifetime[i]):
                    if year == 0:
                        prod = 0
                        cash_cost = 0
                        capital_cost = self.fixed_capital[i] * 0.3
                        revenue = 0
                    elif year == 1:
                        prod = 0
                        cash_cost = 0
                        capital_cost = self.fixed_capital[i] * 0.6
                        revenue = 0
                    elif year == 2:
                        prod = 0.4 * self.daily_prod * 365 * self.plant_utilization
                        cash_cost = self.fixed_production_costs[i] + 0.4 * self.variable_production_costs[i]
                        capital_cost = self.fixed_capital[i] * 0.1 + self.working_capital
                        revenue = self.product_price * prod
                    elif year == 3:
                        prod = 0.8 * self.daily_prod * 365 * self.plant_utilization
                        cash_cost = self.fixed_production_costs[i] + 0.8 * self.variable_production_costs[i]
                        capital_cost = 0
                        revenue = self.product_price * prod
                    else:
                        prod = self.daily_prod * 365 * self.plant_utilization
                        cash_cost = self.fixed_production_costs[i] + self.variable_production_costs[i]
                        capital_cost = 0
                        revenue = self.product_price * prod

                    gross_profit = revenue - cash_cost

                    if gross_profit > 0 and depreciation_counter < depreciation_duration[i]:
                        depreciation = depreciation_amount[i]
                        depreciation_counter += 1
                    else:
                        depreciation = 0

                    taxable_income = gross_profit - depreciation
                    tax_paid = self.tax_rate * previous_taxable_income if previous_taxable_income > 0 else 0
    
                    self.capital_cost_array[i][year] = capital_cost
                    self.prod_array[i][year] = prod
                    self.cash_cost_array[i][year] = cash_cost
                    self.revenue_array[i][year] = revenue
                    self.gross_profit_array[i][year] = gross_profit
                    self.depreciation_array[i][year] = depreciation
                    self.taxable_income_array[i][year] = taxable_income
                    self.tax_paid_array[i][year] = tax_paid
                    self.cash_flow[i][year] = gross_profit - tax_paid - capital_cost

                    previous_taxable_income = taxable_income

                self.capital_cost_array[i][-1] -= self.working_capital
                self.cash_flow[i][-1] += self.working_capital
        else:
            for year in range(self.project_lifetime):
                if year == 0:
                    prod = 0
                    cash_cost = 0
                    capital_cost = self.fixed_capital * 0.3
                    revenue = 0
                elif year == 1:
                    prod = 0
                    cash_cost = 0
                    capital_cost = self.fixed_capital * 0.6
                    revenue = 0
                elif year == 2:
                    prod = 0.4 * self.daily_prod * 365 * self.plant_utilization
                    cash_cost = self.fixed_production_costs + 0.4 * self.variable_production_costs
                    capital_cost = self.fixed_capital * 0.1 + self.working_capital
                    revenue = self.product_price * prod
                elif year == 3:
                    prod = 0.8 * self.daily_prod * 365 * self.plant_utilization
                    cash_cost = self.fixed_production_costs + 0.8 * self.variable_production_costs
                    capital_cost = 0
                    revenue = self.product_price * prod
                else:
                    prod = self.daily_prod * 365 * self.plant_utilization
                    cash_cost = self.fixed_production_costs + self.variable_production_costs
                    capital_cost = 0
                    revenue = self.product_price * prod

                gross_profit = revenue - cash_cost

                if gross_profit > 0 and depreciation_counter < (self.project_lifetime/2):
                    depreciation = depreciation_amount
                    depreciation_counter += 1
                else:
                    depreciation = 0

                taxable_income = gross_profit - depreciation
                tax_paid = self.tax_rate * previous_taxable_income if previous_taxable_income > 0 else 0

                self.capital_cost_array[year] = capital_cost
                self.prod_array[year] = prod
                self.cash_cost_array[year] = cash_cost
                self.revenue_array[year] = revenue
                self.gross_profit_array[year] = gross_profit
                self.depreciation_array[year] = depreciation
                self.taxable_income_array[year] = taxable_income
                self.tax_paid_array[year] = tax_paid
                self.cash_flow[year] = gross_profit - tax_paid - capital_cost

                previous_taxable_income = taxable_income

            self.capital_cost_array[-1] -= self.working_capital
            self.cash_flow[-1] += self.working_capital

        if print_results:

            self.calculate_npv()

            data = {
            "Year": np.arange(len(self.cash_flow)) + 1,
            "Capital cost": self.capital_cost_array,
            "Production [unit]": self.prod_array,
            "Revenue": self.revenue_array,
            "Cash cost of prod.": self.cash_cost_array,
            "Gross profit": self.gross_profit_array,
            "Depreciation": self.depreciation_array,
            "Taxable income": self.taxable_income_array,
            "Tax paid": self.tax_paid_array,
            "Cash flow": self.cash_flow,
            "PV of cash flow": self.pv_array,
            "NPV": self.npv_array,
        }

            df = pd.DataFrame(data)


            formatted_df = df.style.format(
                {col: "${:,.2f}" for col in df.columns if col != "Year"}
            )

            return formatted_df


    def calculate_npv(self):
        """
        Calculates the present value (PV) and cumulative net present value (NPV) of a series of cash flows.
        Parameters:
            config (dict): A dictionary containing the following keys:
                - 'cash_flow' (array-like): Sequence of cash flows for each period (e.g., yearly).
                - 'interest_rate' (float): Discount rate to be applied to the cash flows.
        Returns:
            tuple:
                - pv_array (numpy.ndarray): Array of present values for each cash flow.
                - npv_array (numpy.ndarray): Cumulative sum of present values (NPV) over time.
        """

        years = np.arange(1, len(self.cash_flow) + 1)
        pv_array = self.cash_flow / ((1 + self.interest_rate) ** years)
        npv_array = np.cumsum(pv_array)

        self.pv_array = pv_array
        self.npv_array = npv_array        

    def calculate_levelized_cost(self, print_results=False):
        self.calculate_fixed_capital(fc=1.0 if self.fc is None else self.fc)
        self.calculate_variable_opex()
        self.calculate_fixed_opex(fp=1.0 if self.fp is None else self.fp)
        self.calculate_cash_flow()

        n_components = len(self.project_lifetime) if isinstance(self.project_lifetime, (list, np.ndarray)) else int(self.project_lifetime)

        capital_cost, prod, cash_cost = self.capital_cost_array, self.prod_array, self.cash_cost_array
        (disc_capex, disc_opex, disc_prod) = (np.zeros(n_components), np.zeros(n_components), np.zeros(n_components))

        if isinstance(self.project_lifetime, (list, np.ndarray)):
            for i in range(n_components):
                for year in range(len(cash_cost[i])):
                    discount_factor = (1 + self.interest_rate[i]) ** (year+1)
                    disc_capex[year] += (capital_cost[i][year]) / discount_factor
                    disc_opex[year] += (cash_cost[i][year]) / discount_factor
                    disc_prod[year] += prod[i][year] / discount_factor
        else:
            for year in range(n_components):
                disc_capex[year] = (capital_cost[year]) / ((1 + self.interest_rate) ** (year+1))
                disc_opex[year] = (cash_cost[year]) / ((1 + self.interest_rate) ** (year+1))
                disc_prod[year] = prod[year] / ((1 + self.interest_rate) ** (year+1))

        self.levelized_cost = max(np.sum(disc_capex + disc_opex) / np.sum(disc_prod), 0)

        if print_results:
            print(f"Levelized cost: ${self.levelized_cost:,.3f}/kg")

    def calculate_payback_time(self, print_results=False):
        """
        Calculate the payback time for a given project configuration.
        The payback time is computed as the ratio of the fixed capital investment to the average annual cash flow during years when revenue is positive. If there are no years with positive revenue or the average annual cash flow is not positive, the function returns NaN.
        Parameters:
            config (dict): A dictionary containing the following keys:
                - "revenue_array" (np.ndarray): Array of revenue values for each year.
                - "cash_flow" (np.ndarray): Array of cash flow values for each year.
                - "fixed_capital" (float): The fixed capital investment amount.
        Returns:
            float: The payback time in years, or NaN if it cannot be calculated.
        """
            
        revenue, cash_flow = self.revenue_array, self.cash_flow
        
        revenue_generating_years = cash_flow[revenue > 0]

        if len(revenue_generating_years) == 0:
            self.payback_time = float('nan')
        else:
            average_annual_cash_flow = np.mean(revenue_generating_years)
            self.payback_time = self.fixed_capital / average_annual_cash_flow if average_annual_cash_flow > 0 else float(
                'nan')

        if print_results:
            print(f"Payback time: {self.payback_time:.2f} years")

    def calculate_roi(self, print_results=False):
        """
        Calculate the return on investment (ROI) for a given project configuration.
        Args:
            config (dict): A dictionary containing the following keys:
                - "gross_profit_array" (array-like): Array of gross profits for each period.
                - "tax_paid_array" (array-like): Array of taxes paid for each period.
                - "fixed_capital" (float): The fixed capital investment.
                - "working_capital" (float): The working capital investment.
                - "project_lifetime" (int or float): The total lifetime of the project.
        Returns:
            float: The calculated ROI as a fraction.
        Notes:
            ROI is calculated as the sum of net profits over the project lifetime divided by
            the product of project lifetime and total investment (fixed + working capital).
        """

        net_profit = self.gross_profit_array - self.tax_paid_array
        total_investment = self.fixed_capital + self.working_capital

        self.roi = np.sum(net_profit) / (self.project_lifetime * np.sum(total_investment))

        if print_results:
            print(f"Return of investment: {self.roi*100:.2f}%")

    def calculate_irr(self, print_results=False):
        """
        Calculates the Internal Rate of Return (IRR) for a given cash flow series.
        Args:
            config (dict): A configuration dictionary containing the key "cash_flow",
                which should be a list or array of cash flow values for each period.
        Returns:
            float: The IRR value as a decimal (e.g., 0.1 for 10%) if a solution is found,
                otherwise NaN if the IRR cannot be computed.
        Notes:
            - Uses the Brent's method to find the root of the NPV function.
            - The search for IRR is performed in the bracket [-10, 10].
            - Returns NaN if the root finding fails or if an exception occurs.
        """
        
        cash_flow = self.cash_flow

        def npv(irr):
            return sum(cash_flow[year] / ((1 + irr)**(year+1)) for year in range(len(cash_flow)))
        try:
            sol = root_scalar(npv, bracket=[-10, 10], method='brentq')
            self.irr = sol.root if sol.converged else float('nan')

        except (ValueError, RuntimeError):
            self.irr = float('nan')

        if print_results:
            print(f"Internal Rate of Return: {self.irr*100:.2f}%")

def tornado_plot(process_plant, plus_minus_value, label=r'Levelized cost of product / [US\$$\cdot$kg$^{-1}$]'):
    """
    Generate a tornado plot to visualize the sensitivity of the levelized cost of product to key input parameters.
    This function performs a one-at-a-time sensitivity analysis on selected parameters in the `config` dictionary,
    varying each parameter by ±`plus_minus_value` (as a fraction, e.g., 0.2 for ±20%) and recalculating the
    levelized cost of product (lcop). The results are displayed as a horizontal bar chart (tornado plot), showing
    the impact of each parameter on the lcop.
    Parameters
    ----------
    config : dict
        Configuration dictionary containing all techno-economic parameters, including top-level keys and nested
        variable OPEX input prices.
    plus_minus_value : float
        Fractional change to apply to each parameter for sensitivity analysis (e.g., 0.2 for ±20%).
    Returns
    -------
    None
        Displays a matplotlib tornado plot. Does not return any value.
    Notes
    -----
    - The function assumes the existence of several calculation functions:
      `calculate_levelized_cost`, `calculate_variable_opex`, `calculate_fixed_opex`,
      `calculate_fixed_capital`, and `calculate_cash_flow`.
    - The tornado plot highlights which parameters have the largest effect on the lcop.
    - The plot uses blue bars for -X% changes and red bars for +X% changes.
    """

    # Top-level keys (assuming these are variable names already defined)
    top_level_keys = [
        'fixed_capital',
        'fixed_opex',
        'project_lifetime',
        'interest_rate',
        'operator_hourly_rate'
    ]

    # Get all nested 'price' keys from variable_opex_inputs
    nested_price_keys = [
        f"variable_opex_inputs.{k}" for k in process_plant.variable_opex_inputs.keys()
    ]

    all_keys = top_level_keys + nested_price_keys

    lcop_base = process_plant.levelized_cost

    # Function to traverse nested attributes/dictionaries
    def get_original_value(process_plant, full_key):
        keys = full_key.split('.')
        ref = process_plant
        for k in keys:
            if isinstance(ref, dict):
                ref = ref[k]["price"]
            else:
                ref = getattr(ref, k)
        return ref

    def update_and_evaluate(process_plant, factor, value):
        process_plant_copy = deepcopy(process_plant)
        if factor == 'fixed_capital':
            process_plant_copy.calculate_fixed_capital(fc=value)

        elif factor == 'fixed_opex':
            process_plant_copy.calculate_fixed_opex(fp=value)

        elif factor in nested_price_keys:
            parts = factor.split('.')
            config = {
                'variable_opex_inputs': {
                    parts[1] : {
                        'price': value,
                    }
                }
            }
            process_plant_copy.update_configuration(config)

        else:
            config = {
                factor: value
            }
            process_plant_copy.update_configuration(config)

        process_plant_copy.calculate_levelized_cost()
        
        return process_plant_copy.levelized_cost

    # Perform sensitivity analysis
    sensitivity_results = {}
    for key in all_keys:
        if key == 'fixed_capital' or key == 'fixed_opex':
            low = (1 - plus_minus_value)
            high = (1 + plus_minus_value)
        else:
            original = get_original_value(process_plant, key)
            low = original * (1 - plus_minus_value)
            high = original * (1 + plus_minus_value)
        lcop_high = update_and_evaluate(process_plant, key, high)    
        lcop_low = update_and_evaluate(process_plant, key, low)
        
        sensitivity_results[key] = [lcop_low, lcop_high]

    # Extract lcop values
    factors = list(sensitivity_results.keys())
    lcop_lows = np.array([sensitivity_results[f][0] for f in factors])
    lcop_highs = np.array([sensitivity_results[f][1] for f in factors])
    total_effects = np.abs(lcop_highs - lcop_lows)

    # Sort by total effect
    sorted_indices = np.argsort(total_effects)
    factors_sorted = [factors[i] for i in sorted_indices]
    lcop_lows_sorted = lcop_lows[sorted_indices]
    lcop_highs_sorted = lcop_highs[sorted_indices]

    # # Prepare bar components
    # bar_left = np.minimum(lcop_lows_sorted, lcop_highs_sorted)
    # bar_width = np.abs(lcop_highs_sorted - lcop_lows_sorted)

    # Assign colors: blue for -X%, red for +X%
    colors_low = ['#87CEEB'] * len(factors_sorted)   # blue for -X%
    colors_high = ['#FF9999'] * len(factors_sorted)  # red for +X%

    # Label mapping
    label_map = {
        "fixed_capital": "Fixed CAPEX",
        "fixed_opex": "Fixed OPEX",
        "project_lifetime": "Project lifetime",
        "interest_rate": "Interest rate",
        "operator_hourly_rate": "Operator hourly rate",
    }
    for var in process_plant.variable_opex_inputs:
        label_map[f"variable_opex_inputs.{var}.price"] = f"{var.capitalize()} price"

    labels_sorted = [label_map.get(f, f.replace("variable_opex_inputs.", "").replace(".price", "").replace("_", " ").capitalize() + " price" if "variable_opex_inputs" in f else f) for f in factors_sorted]
    y_pos = np.arange(len(labels_sorted))

    # Plot
    plt.figure(figsize=(3.4, 2.4))
    for i in range(len(y_pos)):
        # Bar for -X% (blue)
        plt.barh(y_pos[i], abs(lcop_lows_sorted[i] - lcop_base), left=min(lcop_base,lcop_lows_sorted[i]),
                 color=colors_low[i], edgecolor='black', label=r'-{}\%'.format(int(plus_minus_value * 100)) if i == 0 else "")
        # Bar for +X% (red)
        plt.barh(y_pos[i], abs(lcop_highs_sorted[i] - lcop_base), left=min(lcop_base,lcop_highs_sorted[i]),
                 color=colors_high[i], edgecolor='black', label=r'+{}\%'.format(int(plus_minus_value * 100)) if i == 0 else "")

    plt.axvline(x=lcop_base, color='black', linestyle='--', linewidth=0.75)
    plt.yticks(y_pos, labels_sorted)
    plt.xlim(min(lcop_lows_sorted) * 0.95, max(lcop_highs_sorted) * 1.05)
    plt.xlabel(label)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def monte_carlo(process_plant, num_samples: int = 1_000_000, batch_size: int = 1000, 
                show_input_distributions: bool = False, 
                show_plot_updates: bool = False,
                show_final_plot: bool = True,
                label=r'Levelized cost of product / [US\$$\cdot$kg$^{-1}$]'):

    process_plant_copy = deepcopy(process_plant)
    num_samples = int(num_samples)
    num_batches = num_samples // batch_size

    def truncated_normal_samples(mean, std, low, high, size):
        a, b = (low - mean) / std, (high - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

    # Preallocate arrays
    fixed_capitals = np.zeros(num_samples)
    fixed_opexs = np.zeros(num_samples)
    operator_hourlys = np.zeros(num_samples)
    project_lifetimes = np.zeros(num_samples)
    interests = np.zeros(num_samples)
    lcops = np.zeros(num_samples)

    # Preallocate variable opex price samples
    variable_opex_price_samples = {
        item: np.zeros(num_samples)
        for item in process_plant_copy.variable_opex_inputs.keys()
    }

    def plot_monte_carlo(lcops, label=label):
        mu_lcop, std_lcop = norm.fit(lcops)
        plt.figure()
        plt.hist(lcops, bins=30, density=True, 
                color='skyblue', edgecolor='black', zorder=2)
        # Plot fitted normal curve
        x = np.linspace(min(lcops), max(lcops), 1000)
        p = norm.pdf(x, mu_lcop, std_lcop)
        plt.plot(x, p, '-', color='indianred', 
                label=fr'$\mu$={mu_lcop:.3g}, $\sigma$={std_lcop:.2e}', zorder=2)
        plt.xlabel(label)
        plt.ylabel("Probability density")
        plt.legend(loc='best', fontsize='x-small')
        plt.show()

    update_interval = num_batches // 10  # Every 1/10 of simulation

    for i in tqdm(range(num_batches), desc="Running Monte Carlo"):
        start = i * batch_size
        end = start + batch_size

        fixed_capitals[start:end] = truncated_normal_samples(
            1, 0.6/2, 0.25, 2, batch_size
        )
        fixed_opexs[start:end] = truncated_normal_samples(
            1, 0.6/2, 0.25, 2, batch_size
        )
        operator_hourlys[start:end] = truncated_normal_samples(
            process_plant.operator_hourly_rate, 20/2, 10, 100, batch_size
        )
        project_lifetimes[start:end] = truncated_normal_samples(
            process_plant.project_lifetime, 10/2, 5, 40, batch_size
        )
        interests[start:end] = truncated_normal_samples(
            process_plant.interest_rate, 0.03/2, 0.01, 2, batch_size
        )

        # Sample variable opex prices
        for item, props in process_plant.variable_opex_inputs.items():
            price_mean = props['price']
            price_std = props['price_std']
            price_min = props['price_min']
            price_max = props['price_max']
            variable_opex_price_samples[item][start:end] = truncated_normal_samples(
                price_mean, price_std, price_min, price_max, batch_size
            )

        # Update batch config
        process_plant_copy.update_configuration({
            'operator_hourly_rate': operator_hourlys[start:end],
            'project_lifetime': project_lifetimes[start:end],
            'interest_rate': interests[start:end],
        })

        # Update variable opex prices in config for this batch
        for item in process_plant_copy.variable_opex_inputs.keys():
            process_plant_copy.variable_opex_inputs[item]['price'] = variable_opex_price_samples[item][start:end]

        # Run calculations
        process_plant_copy.calculate_fixed_capital(fc=fixed_capitals[start:end])
        process_plant_copy.calculate_variable_opex()
        process_plant_copy.calculate_fixed_opex(fp=fixed_opexs[start:end])
        process_plant_copy.calculate_cash_flow()
        process_plant_copy.calculate_levelized_cost()
        lcops[start:end] = process_plant_copy.levelized_cost

        # Show live plot every 1/10 of the simulation
        if show_plot_updates:
            if (i + 1) % update_interval == 0 or (i + 1) == num_batches:
                clear_output(wait=True)  # Clear previous output
                plot_monte_carlo(lcops[:end])

    # Final plot after all batches
    if show_final_plot:
        clear_output(wait=True)
        plot_monte_carlo(lcops)
    process_plant.monte_carlo_lcops = lcops

    # Plotting
    if show_input_distributions:
        mu, std = norm.fit(operator_hourlys)
        # Collect all input parameter arrays for plotting
        input_distributions = {
            'Fixed capital investment': fixed_capitals,
            'Fixed production costs': fixed_opexs,
            'Operator hourly rate': operator_hourlys,
            'Project lifetime': project_lifetimes,
            'Interest rate': interests,
        }

        # Include variable opex price inputs
        for item, samples in variable_opex_price_samples.items():
            input_distributions[f'{item.capitalize()} price'] = samples

        # Set up subplots: adjust layout depending on number of variables
        n_params = len(input_distributions)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
        axes = axes.flatten()  # Flatten for easy indexing

        # Plot each distribution
        for idx, (label, data) in enumerate(input_distributions.items()):
            ax = axes[idx]
            mu, std = norm.fit(data)
            ax.hist(data, bins=50, density=True, color='#FFFFE0', edgecolor='black', zorder=2)
            x = np.linspace(min(data), max(data), 1000)
            ax.plot(x, norm.pdf(x, mu, std), 'r-', label=fr'$\mu$={mu:.2e}, $\sigma$={std:.2e}')
            ax.set_xlabel(label)
            ax.legend(loc='best', fontsize='medium')

        # Turn off any unused subplots
        for i in range(n_params, len(axes)):
            axes[i].axis('off')

        fig.tight_layout()
        plt.show()


def plot_multiple_monte_carlo(process_plants, bins=30, label=r'Levelized cost of product / [US\$$\cdot$kg$^{-1}$]'):
    plt.figure()
    
    # Separate color cycles for histograms and lines
    hist_colors = cycle(plt.cm.tab10.colors)   # e.g., from tab10 colormap
    line_colors = cycle(plt.cm.Set2.colors)    # different palette for lines
    
    for plant in process_plants:
        if plant.monte_carlo_lcops is not None:
            hist_color = next(hist_colors)
            line_color = next(line_colors)
            
            # Fit normal distribution
            mu_lcop, std_lcop = norm.fit(plant.monte_carlo_lcops)
            
            # Histogram
            plt.hist(
                plant.monte_carlo_lcops,
                bins=bins,
                alpha=0.5,
                density=True,
                edgecolor='black',
                color=hist_color,
                zorder=1,
                label=plant.name
            )
            
            # Normal curve
            x = np.linspace(min(plant.monte_carlo_lcops), max(plant.monte_carlo_lcops), 1000)
            p = norm.pdf(x, mu_lcop, std_lcop)
            plt.plot(
                x, p, '-',
                color=line_color,
                linewidth=1.2,
                label=fr'$\mu$={mu_lcop:.3g}, $\sigma$={std_lcop:.2e}',
                zorder=2
            )
    
    plt.xlabel(label)
    plt.ylabel("Probability density")
    
      # --- Adaptive legend settings ---
    handles, labels = plt.gca().get_legend_handles_labels()
    n_items = len(labels)
    
    # Dynamic number of columns
    if n_items <= 4:
        ncol = 1
    elif n_items <= 6:
        ncol = 3
    else:
        ncol = 4
    
    # For many items, force legend outside plot
    if n_items > 4:
        loc = 'upper center'
        bbox_to_anchor = (0.5, 1.20)  # place below plot
    else:
        loc = 'best'
        bbox_to_anchor = None
    
    plt.legend(
        loc=loc, ncol=ncol, fontsize=4,
        frameon=True, facecolor='white', framealpha=0.6,
        fancybox=True, bbox_to_anchor=bbox_to_anchor
    )

     # Adjust layout if legend outside
    if bbox_to_anchor:
        plt.tight_layout(rect=[0, 0, 1, 0.92])  # leave space on top
    else:
        plt.tight_layout()
    plt.show()