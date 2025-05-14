import numpy as np


def calculate_parallel_units(s, upper_bound):
    num_units = np.ones_like(s, dtype=int)

    if np.any(s > upper_bound):
        num_units = np.ceil(s / upper_bound).astype(int)
        adjusted_s = s / num_units
    else:
        adjusted_s = s

    return num_units, adjusted_s


# cost correlations from various literature
def calculate_cost(s, s0, c0, f, cost_year, num_units):
    ci = (c0 * (s / s0) ** f) * 1e6
    return ci*num_units, num_units, cost_year

def calculate_cost_psa(s, s0, c0, f, cost_year, num_units):
    ci = ((c0 * (s / s0) ** f) * 1e6) / 3.2 # Divided by 3.2 to convert from overnight capital cost to purchased cost
    return ci*num_units, num_units, cost_year

def calculate_cost_linear(s, a, b, n, cost_year, num_units):
    ce = a + b * (s ** n)  # Us$
    return ce*num_units, num_units, cost_year


def validate_input_range(s, lower_bound, upper_bound):
    if s < lower_bound:
        raise ValueError(
            f'One or more parameter values fall below the minimum limit of the cost correlation range:'
            f' [{lower_bound}, {upper_bound}]'
        )
    return s


# Burner / Furnace / Boilers
def burner_parkinson_2016(s):  # doi: https://doi.org/10.1002/ceat.201600414  /  https://doi.org/10.1016/j.ijhydene.2020.11.079
    # s in m3 of vessel volume
    return calculate_cost(s, 81.3, 0.16*1.11, 0.60, 2016, 1)


def electric_arc_furnace_parkinson_2016(s):  # doi: https://doi.org/10.1002/ceat.201600414  /  https://doi.org/10.1016/j.ijhydene.2020.11.079
    # s in MWe of net electric power
    return calculate_cost(s, 175, 44*1.11, 0.60, 2016, 1)


def boilers_packaged_towler_2010(s):  # 15-40 bar, doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in kg/h of steam (5000-200,000)
    s = validate_input_range(s, 5000, 200000)
    num_units, s = calculate_parallel_units(s, 200000)
    return calculate_cost_linear(s, 124000, 10, 1, 2010, num_units)


def boilers_erected_towler_2010(s):  # 10-70 bar, doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in kg/h of steam (20,000-800,000)
    s = validate_input_range(s, 20000, 800000)
    num_units, s = calculate_parallel_units(s, 800000)
    return calculate_cost_linear(s, 130000, 53, 0.9, 2010, num_units)


def cylindrical_furnace_towler_2010(s):  # doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in duty, MW (0.2-60)
    s = validate_input_range(s, 0.2, 60)
    num_units, s = calculate_parallel_units(s, 60)
    return calculate_cost_linear(s, 80000, 109000, 0.8, 2010, num_units)


def box_furnace_towler_2010(s):  # doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in duty, MW (30-120)
    s = validate_input_range(s, 30, 120)
    num_units, s = calculate_parallel_units(s, 120)
    return calculate_cost_linear(s, 43000, 111000, 0.8, 2010, num_units)


# compressors / AirBlower
def air_blower_manzolini_2006(s):  # doi: https://doi.org/10.1115/GT2006-90353
    # s in MWe of electric power
    return calculate_cost(s, 1, 0.23*1.26, 0.67, 2006, 1)


def co2_compressor_manzolini_2011(s):  # doi: https://doi.org/10.1016/j.ijggc.2012.06.021
    # s in MWe of electric power
    return calculate_cost(s, 13, 9.95*1.39, 0.67, 2011, 1)


def h2_compressor_pandolfo_1987(s):  # doi: https://doi.org/10.1001/jama.2014.2634
    # s in MWe of electric power
    return calculate_cost(s, 1, 0.0012, 0.82, 1987, 1)


def blower_towler_2010(s):  # doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in m3/h (200-5000)
    s = validate_input_range(s, 200, 5000)
    num_units, s = calculate_parallel_units(s, 5000)
    return calculate_cost_linear(s, 4450, 57, 0.8, 2010, num_units)


def centrifugal_compressor_towler_2010(s):  # doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in kW of driver power (75-30,000)
    s = validate_input_range(s, 75, 30000)
    num_units, s = calculate_parallel_units(s, 30000)
    return calculate_cost_linear(s, 580000, 20000, 0.6, 2010, num_units)


def reciprocating_compressor_towler_2010(s):  # doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in kW of driver power (93-16,800)
    s = validate_input_range(s, 93, 16800)
    num_units, s = calculate_parallel_units(s, 16800)
    return calculate_cost_linear(s, 260000, 2700, 0.75, 2010, num_units)


# cyclones
def gas_multi_cyclone_ulrich_2003(s):  # doi: https://www.xanedu.com/catalog-product-details/chemical-engineering
    # s in m3/s of volumetric gas rate (0.1-45)
    s = validate_input_range(s, 0.1, 50)
    num_units, s = calculate_parallel_units(s, 50)
    return calculate_cost_linear(s, 2379.542, 585.534, 1.218, 2003, num_units)

# CCS Technology
def mea_ccs_technology_2011(s):  # doi: https://www.nrel.gov/docs/fy06osti/39947.pdf
    # s in kg/s
    return calculate_cost(s, 38.4, 28.95*1.39, 0.8, 2011, 1)

# Heat exchangers
def hx_nexant_2006(s):  # doi: https://www.nrel.gov/docs/fy06osti/39947.pdf
    # s in MMBTU/h of duty
    return calculate_cost(s, 659.9, 10.84*1.26, 0.6, 2006, 1)

def u_shell_tube_hx_towler_2010(s):  # doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in m2 of heat exchange area (10-1000)
    s = validate_input_range(s, 10, 1000)
    num_units, s = calculate_parallel_units(s, 1000)
    return calculate_cost_linear(s, 28000, 54, 1.2, 2010, num_units)


def floating_head_shell_tube_hx_towler_2010(s):  # doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in m2 of heat exchange area (10-1000)
    s = validate_input_range(s, 10, 1000)
    num_units, s = calculate_parallel_units(s, 1000)
    return calculate_cost_linear(s, 32000, 70, 1.2, 2010, num_units)


def double_pipe_hx_towler_2010(s):  # doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in m2 of heat exchange area (1-80)
    s = validate_input_range(s, 1, 80)
    num_units, s = calculate_parallel_units(s, 80)
    return calculate_cost_linear(s, 1900, 2500, 1, 2010, num_units)


def thermosiphon_reboiler_towler_2010(s):  # doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in m2 of heat exchange area (10-500)
    s = validate_input_range(s, 10, 500)
    num_units, s = calculate_parallel_units(s, 500)
    return calculate_cost_linear(s, 30400, 122, 1.1, 2010, num_units)


def u_tube_kettle_reboiler_towler_2010(s):  # doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in m2 of heat exchange area (10-500)
    s = validate_input_range(s, 10, 500)
    num_units, s = calculate_parallel_units(s, 500)
    return calculate_cost_linear(s, 29000, 400, 0.9, 2010, num_units)


def plate_frame_hx_towler_2010(s):  # doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in m2 of heat exchange area (1-500)
    s = validate_input_range(s, 1, 500)
    num_units, s = calculate_parallel_units(s, 500)
    return calculate_cost_linear(s, 1600, 210, 0.95, 2010, num_units)


# Motors
def explosion_proof_motor_towler_2010(s):  # doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in kW of power (1.0-2500)
    s = validate_input_range(s, 1, 2500)
    num_units, s = calculate_parallel_units(s, 2500)
    return calculate_cost_linear(s, -1100, 2100, 0.6, 2010, num_units)


def explosion_proof_motor_ulrich_2003(s):  # doi: https://www.xanedu.com/catalog-product-details/chemical-engineering
    # s in kW of shaft power (2.8-5000)
    s = validate_input_range(s, 2.5, 5000)
    num_units, s = calculate_parallel_units(s, 5000)
    return calculate_cost_linear(s, -206.374, 177.886, 0.923, 2003, num_units)


def totally_enclosed_motor_ulrich_2003(s):  # doi: https://www.xanedu.com/catalog-product-details/chemical-engineering
    # s in kW of shaft power (0.2-9000)
    s = validate_input_range(s, 0.2, 9000)
    num_units, s = calculate_parallel_units(s, 9000)
    return calculate_cost_linear(s, 6.567, 116.02, 0.921, 2003, num_units)

# Pumps
def single_stage_centrifugal_pump_towler_2010(s):  # doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in liters/s of flow (0.2-126)
    s = validate_input_range(s, 0.2, 126)
    num_units, s = calculate_parallel_units(s, 126)
    return calculate_cost_linear(s, 8000, 240, 0.9, 2010, num_units)

def centrifugal_pump_almeland_2009(s):  # doi: https://doi.org/10.1115/GT2006-90353
    # s in kW
    return calculate_cost(s, 197, 0.12, 0.67, 2009, 1)

# Pressure-swing adsorber
def psa_kreutz_2002(s):  # doi: https://doi.org/10.1016/j.ijhydene.2004.08.001
    # s in kmol/s of purge gas flow
    return calculate_cost_psa(s/3600, 0.294, 7.1, 0.74, 2002, 1) # Overnight capital cost


def psa_towler_1994(s):  # doi: https://doi.org/10.1021/ie950359+
    # s in kmol/h
    return calculate_cost_linear(s, 19753.8, 258.858, 1, 1994, 1)


# Pressure vessels
def vertical_cs_press_vessel_towler_2010(s):  # doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in kg of shell mass (160-250,000)
    s = validate_input_range(s, 160, 250000)
    num_units, s = calculate_parallel_units(s, 250000)
    return calculate_cost_linear(s, 11600, 34, 0.85, 2010, num_units)


def horizontal_cs_press_vessel_towler_2010(s):  # doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in kg of shell mass (160-50,000)
    s = validate_input_range(s, 160, 50000)
    num_units, s = calculate_parallel_units(s, 50000)
    return calculate_cost_linear(s, 10200, 31, 0.85, 2010, num_units)


def vertical_304ss_press_vessel_towler_2010(s):  # doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in kg of shell mass (120-250,000)
    s = validate_input_range(s, 120, 250000)
    num_units, s = calculate_parallel_units(s, 250000)
    return calculate_cost_linear(s, 17400, 79, 0.85, 2010, num_units)


def horizontal_304ss_press_vessel_towler_2010(s):  # doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in kg of shell mass (120-50,000)
    s = validate_input_range(s, 120, 50000)
    num_units, s = calculate_parallel_units(s, 50000)
    return calculate_cost_linear(s, 12800, 73, 0.85, 2010, num_units)


# Reactors
def indirect_fluidbed_ulrich_2003(s):  # doi: https://www.xanedu.com/catalog-product-details/chemical-engineering
    # s in m3 of reactor volume (0.3-2400)
    s = validate_input_range(s, 0.3, 2400)
    num_units, s = calculate_parallel_units(s, 2400)
    return calculate_cost_linear(s, 12272.260, 8973.191, 0.647, 2003, num_units)


def jacketed_agitated_reactor_towler_2010(s):  # doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in m3 of volume (0.5-100)
    s = validate_input_range(s, 0.5, 100)
    num_units, s = calculate_parallel_units(s, 100)
    return calculate_cost_linear(s, 61500, 32500, 0.8, 2010, num_units)


def jacketed_agitated_glass_lined_reactor_towler_2010(s):  # doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in m3 of volume (0.5-25)
    s = validate_input_range(s, 0.5, 25)
    num_units, s = calculate_parallel_units(s, 25)
    return calculate_cost_linear(s, 12800, 88200, 0.4, 2010, num_units)


def pyrolysis_furnace_ulrich_2003(s):  # doi: https://www.xanedu.com/catalog-product-details/chemical-engineering
    # s in kW of heating duty (3000-60000)
    s = validate_input_range(s, 2900, 60000)
    num_units, s = calculate_parallel_units(s, 60000)
    return calculate_cost_linear(s, 12158.694, 277.701, 0.837, 2003, num_units)


# Turbines
def steam_turbine_kreutz_2002(s):  # with condenser, doi: https://doi.org/10.1016/j.jasrep.2018.09.011
    # s in MWe of turbine gross power
    return calculate_cost(s, 136, 52.1*0.95, 0.67, 2002, 1)


def condensing_steam_turbine_towler_2010(s):  # doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    # s in kW of turbine power (100-20,000)
    s = validate_input_range(s, 100, 20000)
    num_units, s = calculate_parallel_units(s, 20000)
    return calculate_cost_linear(s, -14000, 1900, 0.75, 2010, num_units)


def axial_gas_turbine_ulrich_2003(s):  # doi: https://www.xanedu.com/catalog-product-details/chemical-engineering
    # s in kW of turbine shaft power (10-10,000)
    s = validate_input_range(s, 10, 10000)
    num_units, s = calculate_parallel_units(s, 10000)
    return calculate_cost_linear(s, -462.417, 3191.121, 0.597, 2003, num_units)


def radial_expander_ulrich_2003(s):  # doi: https://www.xanedu.com/catalog-product-details/chemical-engineering
    # s in kW of turbine shaft power (2.5-1500)
    s = validate_input_range(s, 2.5, 1500)
    num_units, s = calculate_parallel_units(s, 1500)
    return calculate_cost_linear(s, -295.849, 1066.906, 0.698, 2003, num_units)


def inflation_adjustment(equipment_cost, cost_year):
    cepci_2023 = 800.8
    cepci_values = {
        1990: 357.6, 1991: 361.3, 1992: 358.2, 1993: 359.2, 1994: 368.1,
        1995: 381.1, 1996: 381.7, 1997: 386.5, 1998: 389.5, 1999: 390.6,
        2000: 394.1, 2001: 394.3, 2002: 395.6, 2003: 402.0, 2004: 444.2,
        2005: 468.2, 2006: 499.6, 2007: 525.4, 2008: 575.4, 2009: 521.9,
        2010: 550.8, 2011: 585.7, 2012: 584.6, 2013: 567.3, 2014: 576.1,
        2015: 556.8, 2016: 541.7, 2017: 567.5, 2018: 603.1, 2019: 607.5,
        2020: 596.2, 2021: 708.8, 2022: 816.0
    }

    cepci_original = cepci_values.get(cost_year)
    if cepci_original is None:
        raise ValueError(f'CEPCI is not available for the year {cost_year}')

    adjusted_cost = equipment_cost * (cepci_2023 / cepci_original)
    return adjusted_cost
