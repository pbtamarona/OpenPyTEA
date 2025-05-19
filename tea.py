import pandas as pd
import numpy as np
import math
from scipy.optimize import root_scalar

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

def calculate_fixed_capital(config):
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

    isbl = calculate_isbl(config)
    
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

def calculate_fixed_opex(config):
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

    config['fixed_opex'] = fixed_production_costs

    return fixed_production_costs

def calculate_cash_flow(config):
    """
    Calculates the annual cash flow and related financial arrays for a project based on the provided configuration.
    Parameters
    ----------
    config : dict
        Dictionary containing the following keys:
            - "project_lifetime" (int): Total number of years for the project (must be > 3).
            - "fixed_capital" (float): Total fixed capital investment.
            - "working_capital" (float): Working capital required.
            - "fixed_opex" (float): Annual fixed operating expenses.
            - "variable_opex" (float): Annual variable operating expenses.
            - "annual_prod" (float): Annual production quantity.
            - "product_price" (float): Price per unit of product.
            - "tax_rate" (float): Corporate tax rate (as a fraction, e.g., 0.3 for 30%).
    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing the following arrays, each of length `project_lifetime`:
            - capital_cost_array: Annual capital expenditures.
            - prod_array: Annual production quantities.
            - cash_cost_array: Annual cash operating costs.
            - revenue_array: Annual revenues.
            - gross_profit_array: Annual gross profits (revenue - cash costs).
            - depreciation_array: Annual depreciation amounts.
            - taxable_income_array: Annual taxable incomes.
            - tax_paid_array: Annual taxes paid.
            - cash_flow: Annual cash flows.
    Notes
    -----
    - The function updates the input `config` dictionary with the computed arrays.
    - Depreciation is calculated over half the project lifetime.
    - Working capital is recovered in the final year.
    - The function raises a ValueError if `project_lifetime` is less than or equal to 3.
    """
    
    project_lifetime = config["project_lifetime"]
    fixed_capital = config["fixed_capital"]
    working_capital = config["working_capital"]
    fixed_opex = config["fixed_opex"]
    variable_opex = config["variable_opex"]
    annual_prod = config["annual_prod"]
    product_price = config["product_price"]
    tax_rate = config["tax_rate"]
    
    if project_lifetime <= 3:
        raise ValueError("Project lifetime must be greater than 3.")
        
    # Initialize arrays
    capital_cost_array = np.zeros(project_lifetime)
    prod_array = np.zeros(project_lifetime)
    revenue_array = np.zeros(project_lifetime)
    cash_cost_array = np.zeros(project_lifetime)
    gross_profit_array = np.zeros(project_lifetime)
    depreciation_array = np.zeros(project_lifetime)
    taxable_income_array = np.zeros(project_lifetime)
    tax_paid_array = np.zeros(project_lifetime)

    previous_taxable_income = 0
    depreciation_counter = 0
    depreciation_amount = fixed_capital / (project_lifetime / 2)

    cash_flow = np.zeros(project_lifetime)

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
    """
    Calculates the levelized cost of production for a given configuration.
    This function discounts capital and operating expenditures, as well as production output,
    over the project lifetime using the provided interest rate, and computes the levelized cost
    (i.e., the discounted total cost per unit of discounted production).
    Args:
        config (dict): A dictionary containing the following keys:
            - "capital_cost_array" (array-like): Annual capital expenditures for each year.
            - "prod_array" (array-like): Annual production output for each year.
            - "cash_cost_array" (array-like): Annual operating (cash) costs for each year.
            - "interest_rate" (float): Discount rate to apply to future cash flows.
    Returns:
        float: The levelized cost of production (non-negative), calculated as the ratio of
                the sum of discounted costs (capital + operating) to the sum of discounted production.
    """

    capital_cost, prod, cash_cost = config["capital_cost_array"], config["prod_array"], config["cash_cost_array"]
    (disc_capex, disc_opex, disc_prod) = (np.zeros(len(cash_cost)), np.zeros(len(cash_cost)), np.zeros(len(cash_cost)))

    for year in range(len(cash_cost)):
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
