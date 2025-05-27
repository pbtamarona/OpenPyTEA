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

from equipment import *

def calculate_isbl(config):
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

    locFactors = {
        'United States': {'Gulf Coast': 1.00, 'East Coast': 1.04, 'West Coast': 1.07, 'Midwest': 1.02},
        'Canada': {'Ontario': 1.00, 'Fort McMurray': 1.60},
        'Mexico': 1.03, 'Brazil': 1.14, 'China': {'imported': 1.12, 'indigenous': 0.61},
        'Japan': 1.26, 'Southeast Asia': 1.12, 'Australia': 1.21, 'India': 1.02, 'Middle East': 1.07,
        'France': 1.13, 'Germany': 1.11, 'Italy': 1.14, 'Netherlands': 1.19, 'Russia': 1.53, 'United Kingdom': 1.02,
    }
 
    def location_factors(country, region) -> float:
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

        if country not in locFactors:
            raise ValueError(f'Country not found: {country}. Available countries: {list(locFactors.keys())}')

        loc_factor = locFactors[country]
        if isinstance(loc_factor, dict):
            if region in loc_factor:
                return loc_factor[region]
            else:
                raise ValueError(f'Region not found: {region}. Available regions: {list(loc_factor.keys())}')
        return loc_factor

    isbl = sum(
        equipment.direct_cost for equipment in config['equipment']
    ) * location_factors(config['country'], config['region'])

    return isbl

def calculate_fixed_capital(config, fc=1.0):
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

    isbl = calculate_isbl(config) * fc
    
    processTypes = {
    'Solids': {'OS': 0.4, 'DE': 0.2, 'X': 0.1},
    'Fluids': {'OS': 0.3, 'DE': 0.3, 'X': 0.1},
    'Mixed': {'OS': 0.4, 'DE': 0.25, 'X': 0.1},
    }

    if config['process_type'] not in processTypes:
        raise ValueError(f"Unsupported process_type '{config['process_type']}'. Valid types: {list(processTypes)}")

    params = processTypes[config['process_type']]
    osbl = params['OS'] * isbl
    dne = params['DE'] * (isbl + osbl)
    contigency = params['X'] * (isbl + osbl)
    fixed_capital = (isbl + osbl + dne + contigency)

    config.update({
    'isbl': isbl,
    'osbl': osbl,
    'dne': dne,
    'contigency': contigency,
    'fixed_capital': fixed_capital
    })

    return isbl, osbl, dne, contigency, fixed_capital


def calculate_operating_labor(config):
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
        
    no_fluid_process = count_units_by_process_type(config['equipment'], {'Fluids', 'Mixed'})
    no_solid_process = count_units_by_process_type(config['equipment'], {'Solids', 'Mixed'})

    operators_per_shifts = (6.29 + 31.7 * (no_solid_process ** 2) + 0.23 * no_fluid_process) ** 0.5

    if no_solid_process > 2:
        raise ValueError("Number of solid processes needs to be less than or equal to 2.")

    working_weeks_per_year = 49
    working_shifts_per_week = 5  # 8-hour shifts
    operating_shifts_per_year = 365 * 3

    working_shifts_per_year = working_weeks_per_year * working_shifts_per_week
    working_hours_per_year = working_shifts_per_year * 8

    operators_hired = math.ceil(operators_per_shifts * operating_shifts_per_year / working_shifts_per_year)
    operating_labor_costs = operators_hired * working_hours_per_year * config['operator_hourly_rate']

    return operating_labor_costs


def calculate_variable_opex(config):
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
    
    variable_opex_items = config.get('variable_opex_inputs', {})
    variable_production_costs = 0

    for item, details in variable_opex_items.items():
        consumption = details.get('consumption', 0)
        price = details.get('price', 0)

        cost = consumption * price
        variable_production_costs += cost

    config['variable_opex'] = variable_production_costs

    return variable_production_costs


def calculate_fixed_opex(config, fp=1.0):
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

    isbl = config["isbl"]
    osbl = config["osbl"]
    working_capital = config["working_capital"]
    interest_rate = config["interest_rate"]
    variable_opex = config["variable_opex"]

    operating_labor_costs = calculate_operating_labor(config) 
    supervision_costs = 0.25 * operating_labor_costs
    direct_salary_overhead = 0.5 * (operating_labor_costs + supervision_costs)
    laboratory_charges = 0.10 * operating_labor_costs
    maintenance_costs = 0.05 * isbl
    taxes_insurance_costs = 0.015 * isbl
    rent_of_land_costs = 0.015 * (isbl + osbl)
    environmental_charges = 0.01 * (isbl + osbl)
    operating_supplies = 0.009 * isbl
    general_plant_overhead = 0.65 * (operating_labor_costs + supervision_costs + direct_salary_overhead)

    interest_working_capital = working_capital*interest_rate

    fixed_production_costs = (operating_labor_costs + supervision_costs + direct_salary_overhead
                                    + laboratory_charges + maintenance_costs + taxes_insurance_costs
                                    + rent_of_land_costs + environmental_charges + operating_supplies
                                    + general_plant_overhead + interest_working_capital)

    cash_cost_of_production = (variable_opex + fixed_production_costs) / (1 - 0.07)

    patents_royalties = 0.02 * cash_cost_of_production
    distribution_selling_costs = 0.02 * cash_cost_of_production
    RnD_costs = 0.03 * cash_cost_of_production

    fixed_production_costs += patents_royalties + distribution_selling_costs + RnD_costs

    config['fixed_opex'] = fixed_production_costs * fp

    return fixed_production_costs

def calculate_cash_flow(config):
    
    # Assume self.project_lifetime is a NumPy array with shape (n_components)
    n_components = len(config["project_lifetime"]) if isinstance(config["project_lifetime"], (list, np.ndarray)) else 1
    
    project_lifetime = config["project_lifetime"]

    if isinstance(project_lifetime, (list, np.ndarray)):
        project_lifetime = np.array(project_lifetime, dtype=float)  # ensure numeric

        project_lifetime = project_lifetime.astype(int)

        # Check if all elements are integer-like (e.g., 3.0 is OK, 3.5 is not)
        if not np.all(np.equal(np.mod(project_lifetime, 1), 0)):
            raise TypeError("All values in project_lifetime must be integers.")

        if np.any(np.array(project_lifetime < 3)):
            raise ValueError("All project_lifetime values must be greater than 3.")
        
    else:
        project_lifetime = int(project_lifetime)
        if project_lifetime < 3:
            raise ValueError("Project lifetime must be greater than 3.")
           
    fixed_capital = config["fixed_capital"]
    working_capital = config["working_capital"]
    fixed_opex = config["fixed_opex"]
    variable_opex = config["variable_opex"]
    annual_prod = config["annual_prod"]
    product_price = config["product_price"]
    tax_rate = config["tax_rate"]
        
    # Initialize arrays
    capital_cost_array = [np.zeros(lifetime) for lifetime in project_lifetime] if isinstance(project_lifetime, (list, np.ndarray)) else np.zeros(project_lifetime)
    prod_array = [np.zeros(lifetime) for lifetime in project_lifetime] if isinstance(project_lifetime, (list, np.ndarray)) else np.zeros(project_lifetime)
    revenue_array = [np.zeros(lifetime) for lifetime in project_lifetime] if isinstance(project_lifetime, (list, np.ndarray)) else np.zeros(project_lifetime)
    cash_cost_array = [np.zeros(lifetime) for lifetime in project_lifetime] if isinstance(project_lifetime, (list, np.ndarray)) else np.zeros(project_lifetime)
    gross_profit_array = [np.zeros(lifetime) for lifetime in project_lifetime] if isinstance(project_lifetime, (list, np.ndarray)) else np.zeros(project_lifetime)
    depreciation_array = [np.zeros(lifetime) for lifetime in project_lifetime] if isinstance(project_lifetime, (list, np.ndarray)) else np.zeros(project_lifetime)
    taxable_income_array = [np.zeros(lifetime) for lifetime in project_lifetime] if isinstance(project_lifetime, (list, np.ndarray)) else np.zeros(project_lifetime)
    tax_paid_array = [np.zeros(lifetime) for lifetime in project_lifetime] if isinstance(project_lifetime, (list, np.ndarray)) else np.zeros(project_lifetime)

    cash_flow = [np.zeros(lifetime) for lifetime in project_lifetime] if isinstance(project_lifetime, (list, np.ndarray)) else np.zeros(project_lifetime)

    previous_taxable_income = 0
    depreciation_counter = 0
    depreciation_duration = project_lifetime // 2  # array of durations
    depreciation_amount = fixed_capital / depreciation_duration  # array of amounts
    
    if isinstance(project_lifetime, (list, np.ndarray)):
        for i in range(n_components): 
            for year in range(project_lifetime[i]):
                if year == 0:
                    prod = 0
                    cash_cost = 0
                    capital_cost = fixed_capital[i] * 0.3
                    revenue = 0
                elif year == 1:
                    prod = 0
                    cash_cost = 0
                    capital_cost = fixed_capital[i] * 0.6
                    revenue = 0
                elif year == 2:
                    prod = 0.4 * annual_prod
                    cash_cost = fixed_opex[i] + 0.4 * variable_opex[i]
                    capital_cost = fixed_capital[i] * 0.1 + working_capital
                    revenue = product_price * prod
                elif year == 3:
                    prod = 0.8 * annual_prod
                    cash_cost = fixed_opex[i] + 0.8 * variable_opex[i]
                    capital_cost = 0
                    revenue = product_price * prod
                else:
                    prod = annual_prod
                    cash_cost = fixed_opex[i] + variable_opex[i]
                    capital_cost = 0
                    revenue = product_price * prod

                gross_profit = revenue - cash_cost

                if gross_profit > 0 and depreciation_counter < depreciation_duration[i]:
                    depreciation = depreciation_amount[i]
                    depreciation_counter += 1
                else:
                    depreciation = 0

                taxable_income = gross_profit - depreciation
                tax_paid = tax_rate * previous_taxable_income if previous_taxable_income > 0 else 0

                capital_cost_array[i][year] = capital_cost
                prod_array[i][year] = prod
                cash_cost_array[i][year] = cash_cost
                revenue_array[i][year] = revenue
                gross_profit_array[i][year] = gross_profit
                depreciation_array[i][year] = depreciation
                taxable_income_array[i][year] = taxable_income
                tax_paid_array[i][year] = tax_paid
                cash_flow[i][year] = gross_profit - tax_paid - capital_cost

                previous_taxable_income = taxable_income

            capital_cost_array[i][-1] -= working_capital
            cash_flow[i][-1] += working_capital
    else:
        for year in range(project_lifetime):
            if year == 0:
                prod = 0
                cash_cost = 0
                capital_cost = fixed_capital * 0.3
                revenue = 0
            elif year == 1:
                prod = 0
                cash_cost = 0
                capital_cost = fixed_capital * 0.6
                revenue = 0
            elif year == 2:
                prod = 0.4 * annual_prod
                cash_cost = fixed_opex + 0.4 * variable_opex
                capital_cost = fixed_capital * 0.1 + working_capital
                revenue = product_price * prod
            elif year == 3:
                prod = 0.8 * annual_prod
                cash_cost = fixed_opex + 0.8 * variable_opex
                capital_cost = 0
                revenue = product_price * prod
            else:
                prod = annual_prod
                cash_cost = fixed_opex + variable_opex
                capital_cost = 0
                revenue = product_price * prod

            gross_profit = revenue - cash_cost

            if gross_profit > 0 and depreciation_counter < (project_lifetime/2):
                depreciation = depreciation_amount
                depreciation_counter += 1
            else:
                depreciation = 0

            taxable_income = gross_profit - depreciation
            tax_paid = tax_rate * previous_taxable_income if previous_taxable_income > 0 else 0

            capital_cost_array[year] = capital_cost
            prod_array[year] = prod
            cash_cost_array[year] = cash_cost
            revenue_array[year] = revenue
            gross_profit_array[year] = gross_profit
            depreciation_array[year] = depreciation
            taxable_income_array[year] = taxable_income
            tax_paid_array[year] = tax_paid
            cash_flow[year] = gross_profit - tax_paid - capital_cost

            previous_taxable_income = taxable_income

        capital_cost_array[-1] -= working_capital  
        cash_flow[-1] += working_capital

    config.update({
        "capital_cost_array": capital_cost_array,
        "prod_array": prod_array,
        "cash_cost_array": cash_cost_array,
        "revenue_array": revenue_array,
        "gross_profit_array": gross_profit_array,
        "depreciation_array": depreciation_array,
        "taxable_income_array": taxable_income_array,
        "tax_paid_array": tax_paid_array,
        "cash_flow": cash_flow
    })

    return (capital_cost_array, prod_array, cash_cost_array, revenue_array, gross_profit_array,
            depreciation_array, taxable_income_array, tax_paid_array, cash_flow)


def calculate_npv(config):
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

    years = np.arange(1, len(config['cash_flow']) + 1)
    pv_array = config['cash_flow'] / ((1 + config['interest_rate']) ** years)
    npv_array = np.cumsum(pv_array)

    config['pv_array'] = pv_array
    config['npv_array'] = npv_array

    return pv_array, npv_array


def create_cash_flow_table(config):
    """
    Generates a formatted cash flow table for a given project configuration.
    This function calculates various financial metrics such as capital cost, production,
    revenue, cash costs, gross profit, depreciation, taxable income, tax paid, cash flow,
    present value (PV), and net present value (NPV) over the project years. It returns a
    pandas Styler object with the data formatted for display.
    Args:
        config (dict): Configuration dictionary containing project and financial parameters.
    Returns:
        pandas.io.formats.style.Styler: A styled DataFrame with yearly cash flow and related financial metrics,
        formatted with currency style for monetary columns.
    """

    (capital_cost_array, prod_array, cash_cost_array, revenue_array, gross_profit_array, 
     depreciation_array, taxable_income_array, tax_paid_array, cash_flow) = calculate_cash_flow(config)
    pv_array, npv_array = calculate_npv(config)

    data = {
        "Year": np.arange(len(cash_flow)) + 1,
        "Capital cost": capital_cost_array,
        "Production [unit]": prod_array,
        "Revenue": revenue_array,
        "Cash cost of prod.": cash_cost_array,
        "Gross profit": gross_profit_array,
        "Depreciation": depreciation_array,
        "Taxable income": taxable_income_array,
        "Tax paid": tax_paid_array,
        "Cash flow": cash_flow,
        "PV of cash flow": pv_array,
        "NPV": npv_array,
    }

    df = pd.DataFrame(data)


    formatted_df = df.style.format(
        {col: "${:,.2f}" for col in df.columns if col != "Year"}
    )

    return formatted_df

def calculate_levelized_cost(config):

    n_components = len(config["project_lifetime"]) if isinstance(config["project_lifetime"], (list, np.ndarray)) else int(config["project_lifetime"])

    capital_cost, prod, cash_cost = config["capital_cost_array"], config["prod_array"], config["cash_cost_array"]
    (disc_capex, disc_opex, disc_prod) = (np.zeros(n_components), np.zeros(n_components), np.zeros(n_components))

    if isinstance(config["project_lifetime"], (list, np.ndarray)):
        for i in range(n_components):
            for year in range(len(cash_cost[i])):
                discount_factor = (1 + config["interest_rate"][i]) ** (year+1)
                disc_capex[year] += (capital_cost[i][year]) / discount_factor
                disc_opex[year] += (cash_cost[i][year]) / discount_factor
                disc_prod[year] += prod[i][year] / discount_factor
    else:
        for year in range(n_components):
            disc_capex[year] = (capital_cost[year]) / ((1 + config["interest_rate"]) ** (year+1))
            disc_opex[year] = (cash_cost[year]) / ((1 + config["interest_rate"]) ** (year+1))
            disc_prod[year] = prod[year] / ((1 + config["interest_rate"]) ** (year+1))

    return max(np.sum(disc_capex + disc_opex) / np.sum(disc_prod), 0)

def calculate_payback_time(config):
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
        
    revenue, cash_flow = config["revenue_array"], config["cash_flow"]
    
    revenue_generating_years = cash_flow[revenue > 0]

    if len(revenue_generating_years) == 0:
        payback_time = float('nan')
    else:
        average_annual_cash_flow = np.mean(revenue_generating_years)
        payback_time = config["fixed_capital"] / average_annual_cash_flow if average_annual_cash_flow > 0 else float(
            'nan')
        
    return payback_time

def calculate_roi(config):
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

    net_profit = config["gross_profit_array"] - config["tax_paid_array"]
    total_investment = config["fixed_capital"] + config["working_capital"]

    roi = np.sum(net_profit) / (config['project_lifetime'] * np.sum(total_investment))
        
    return roi

def calculate_irr(config):
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
    
    cash_flow = config["cash_flow"]

    def npv(irr):
        return sum(cash_flow[year] / ((1 + irr)**(year+1)) for year in range(len(cash_flow)))
    try:
        sol = root_scalar(npv, bracket=[-10, 10], method='brentq')
        return sol.root if sol.converged else float('nan')
    
    except (ValueError, RuntimeError):
        return float('nan')

def tornado_plot(config, plus_minus_value):
    """
    Generate a tornado plot to visualize the sensitivity of the levelized cost of product to key input parameters.
    This function performs a one-at-a-time sensitivity analysis on selected parameters in the `config` dictionary,
    varying each parameter by ±`plus_minus_value` (as a fraction, e.g., 0.2 for ±20%) and recalculating the
    levelized cost of product (LCOH). The results are displayed as a horizontal bar chart (tornado plot), showing
    the impact of each parameter on the LCOH.
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
    - The tornado plot highlights which parameters have the largest effect on the LCOH.
    - The plot uses blue bars for -X% changes and red bars for +X% changes.
    """

    top_level_keys = [
        'fixed_capital',
        'fixed_opex',
        'project_lifetime',
        'interest_rate',
        'operator_hourly_rate',
    ]

    nested_price_keys = [
        f"variable_opex_inputs.{k}.price" 
        for k in config['variable_opex_inputs'].keys()
    ]

    all_keys = top_level_keys + nested_price_keys

    lcoh_base = calculate_levelized_cost(config)

    def get_original_value(config, full_key):
        keys = full_key.split('.')
        ref = config
        for k in keys:
            ref = ref[k]
        return ref

    def update_and_evaluate(config, factor, value):
        config_copy = deepcopy(config)

        if factor == 'fixed_capital':
            calculate_fixed_capital(config_copy, fc=value)
            calculate_variable_opex(config_copy)
            calculate_fixed_opex(config_copy)
        elif factor == 'fixed_opex':
            config_copy[factor] = value
            calculate_fixed_capital(config_copy)
            calculate_variable_opex(config_copy)
            calculate_fixed_opex(config_copy, fp=value)
        elif factor in nested_price_keys:
            config_copy[factor] = value
            # Update the price of the specific variable OPEX item
            item_name = factor.split('.')[1]
            config_copy['variable_opex_inputs'][item_name]['price'] = value
            calculate_fixed_capital(config_copy)
            calculate_variable_opex(config_copy)
            calculate_fixed_opex(config_copy)
        else:
            config_copy[factor] = value
            calculate_fixed_capital(config_copy)
            calculate_variable_opex(config_copy)
            calculate_fixed_opex(config_copy)

        calculate_cash_flow(config_copy)
        return calculate_levelized_cost(config_copy)

    # Perform sensitivity analysis
    sensitivity_results = {}
    for key in all_keys:
        original = get_original_value(config, key)
        if key == 'fixed_capital' or key == 'fixed_opex':
            low = (1 - plus_minus_value)
            high = (1 + plus_minus_value)
        else:
            low = original * (1 - plus_minus_value)
            high = original * (1 + plus_minus_value)
        lcoh_low = update_and_evaluate(config, key, low)
        lcoh_high = update_and_evaluate(config, key, high)
        sensitivity_results[key] = [lcoh_low, lcoh_high]

    # Extract LCOH values
    factors = list(sensitivity_results.keys())
    lcoh_lows = np.array([sensitivity_results[f][0] for f in factors])
    lcoh_highs = np.array([sensitivity_results[f][1] for f in factors])
    total_effects = np.abs(lcoh_highs - lcoh_lows)

    # Sort by total effect
    sorted_indices = np.argsort(total_effects)
    factors_sorted = [factors[i] for i in sorted_indices]
    lcoh_lows_sorted = lcoh_lows[sorted_indices]
    lcoh_highs_sorted = lcoh_highs[sorted_indices]

    # Prepare bar components
    bar_left = np.minimum(lcoh_lows_sorted, lcoh_highs_sorted)
    bar_width = np.abs(lcoh_highs_sorted - lcoh_lows_sorted)

    # Assign colors: blue for -X%, red for +X%
    colors_low = ['#87CEEB'] * len(factors_sorted)   # blue for -X%
    colors_high = ['#FF9999'] * len(factors_sorted)  # red for +X%

    # Label mapping
    label_map = {
        "fixed_capital": "Fixed capital",
        "fixed_opex": "Fixed OPEX",
        "project_lifetime": "Project lifetime",
        "interest_rate": "Interest rate",
        "operator_hourly_rate": "Operator hourly rate",
    }
    for var in config['variable_opex_inputs']:
        label_map[f"variable_opex_inputs.{var}.price"] = f"{var.capitalize()} price"

    labels_sorted = [label_map[f] for f in factors_sorted]
    y_pos = np.arange(len(labels_sorted))

    # Plot
    plt.figure(figsize=(3, 2))
    for i in range(len(y_pos)):
        # Bar for -X% (blue)
        plt.barh(y_pos[i], abs(lcoh_base - lcoh_lows_sorted[i]), left=min(lcoh_base, lcoh_lows_sorted[i]),
                 color=colors_low[i], edgecolor='black', label=r'-{}\%'.format(int(plus_minus_value * 100)) if i == 0 else "")
        # Bar for +X% (red)
        plt.barh(y_pos[i], abs(lcoh_highs_sorted[i] - lcoh_base), left=min(lcoh_base, lcoh_highs_sorted[i]),
                 color=colors_high[i], edgecolor='black', label=r'+{}\%'.format(int(plus_minus_value * 100)) if i == 0 else "")

    plt.axvline(x=lcoh_base, color='black', linestyle='--', linewidth=1)
    plt.yticks(y_pos, labels_sorted)
    plt.xlim(left=min(lcoh_lows_sorted) * 0.95, right=max(lcoh_highs_sorted) * 1.05)
    plt.xlabel(r'Levelized cost of product / [US\$$\cdot$unit$^{-1}$]')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def monte_carlo(config, num_samples: int = 1_000_000, batch_size: int = 1000, show_input_distributions: bool = True):

    config_copy = deepcopy(config)
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
    lcohs = np.zeros(num_samples)

    # Preallocate variable opex price samples
    variable_opex_price_samples = {
        item: np.zeros(num_samples)
        for item in config_copy['variable_opex_inputs']
    }

    for i in tqdm(range(num_batches), desc="Running Monte Carlo"):
        start = i * batch_size
        end = start + batch_size


        fixed_capitals[start:end] = truncated_normal_samples(
            1, 0.3/2, 0.25, 2, batch_size
        )
        fixed_opexs[start:end] = truncated_normal_samples(
            1, 0.3/2, 0.25, 2, batch_size
        )
        operator_hourlys[start:end] = truncated_normal_samples(
            config['operator_hourly_rate'], 20/2, 10, 100, batch_size
        )
        project_lifetimes[start:end] = truncated_normal_samples(
            config['project_lifetime'], 10/2, 5, 40, batch_size
        )
        interests[start:end] = truncated_normal_samples(
            config['interest_rate'], 0.03/2, 0.01, 2, batch_size
        )

        # Sample variable opex prices
        for item, props in config['variable_opex_inputs'].items():
            price_mean = props['price']
            price_std = props['price_std']
            price_min = props['price_min']
            price_max = props['price_max']
            variable_opex_price_samples[item][start:end] = truncated_normal_samples(
                price_mean, price_std, price_min, price_max, batch_size
            )

        # Update batch config
        config_copy.update({
            'operator_hourly_rate': operator_hourlys[start:end],
            'project_lifetime': project_lifetimes[start:end],
            'interest_rate': interests[start:end],
        })

        # Update variable opex prices in config for this batch
        for item in config_copy['variable_opex_inputs']:
            config_copy['variable_opex_inputs'][item]['price'] = variable_opex_price_samples[item][start:end]

        # Run calculations
        calculate_fixed_capital(config_copy, fixed_capitals[start:end])
        calculate_variable_opex(config_copy)
        calculate_fixed_opex(config_copy, fixed_opexs[start:end])
        calculate_cash_flow(config_copy)
        lcohs[start:end] = calculate_levelized_cost(config_copy)

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

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2))
        axes = axes.flatten()  # Flatten for easy indexing

        # Plot each distribution
        for idx, (label, data) in enumerate(input_distributions.items()):
            ax = axes[idx]
            mu, std = norm.fit(data)
            ax.hist(data, bins=50, density=True, color='#FFFFE0', edgecolor='black', zorder=2)
            x = np.linspace(min(data), max(data), 1000)
            ax.plot(x, norm.pdf(x, mu, std), 'r-', label=fr'$\mu$={mu:.2f}, $\sigma$={std:.2f}')
            ax.set_xlabel(label)
            ax.legend(loc='best')

        # Turn off any unused subplots
        for i in range(n_params, len(axes)):
            axes[i].axis('off')

        fig.tight_layout()
        plt.show()

    # Histogram of LCOP values
    plt.figure(figsize=(3, 2))
    mu_lcoh, std_lcoh = norm.fit(lcohs)
    count, bins, _ = plt.hist(lcohs, bins=30, density=True, color='skyblue', edgecolor='black', zorder=2)
    # Plot fitted normal curve
    x = np.linspace(min(bins), max(bins), 10000)
    p = norm.pdf(x, mu_lcoh, std_lcoh)
    plt.plot(x, p, '-', color='indianred', label=fr'$\mu$={mu_lcoh:.2f}, $\sigma$={std_lcoh:.2f}', zorder=2)
    plt.xlabel(r'Levelized cost of product / [US\$$\cdot$kg$^{-1}$]')
    plt.ylabel("Probability density")
    plt.legend(loc='upper right', fontsize=8)
    plt.show()

    return lcohs
