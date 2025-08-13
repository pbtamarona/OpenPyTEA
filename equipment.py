from cost_correlations import *

class Equipment:
    """
    Represents a process equipment item with cost estimation capabilities.
    Attributes:
        process_factors (dict): Process type factors for cost estimation.
        material_factors (dict): Material factors for cost estimation.
        cost_funcs (dict): Mapping of equipment types and subtypes to cost correlation functions.
    Args:
        name (str): Name of the equipment.
        process_type (str): Process type (e.g., 'Solids', 'Fluids', etc.).
        material (str): Construction material of the equipment.
        param (float): Main sizing parameter for the equipment (e.g., flow rate, area).
        type (str): Equipment type (e.g., 'Pump', 'Reactor').
        subtype (str, optional): Equipment subtype (e.g., 'Centrifugal' for compressors).
    Properties:
        purchased_cost (float): Estimated purchased cost of the equipment (inflation-adjusted).
        direct_cost (float): Estimated direct cost including installation and material factors.
        cost_year (int): Reference year for the cost correlation.
        num_units (int): Number of units required.
    Methods:
        calculate_equipment_cost() -> float:
            Calculates the purchased cost of the equipment using the appropriate cost correlation and adjusts for inflation.
        calculate_direct_cost() -> float:
            Calculates the direct cost of the equipment, including process and material factors.
        __str__() -> str:
            Returns a string representation of the equipment and its estimated costs.
    Raises:
        ValueError: If the equipment type, subtype, process type, or material is not found in the corresponding dictionaries.
    """

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

    cost_funcs = {
        'Blower': blower_towler_2010,
        'Compressor': {
            'Centrifugal': centrifugal_compressor_towler_2010,
            'Reciprocating': reciprocating_compressor_towler_2010
        },
        'Cyclone': gas_multi_cyclone_ulrich_2003,
        'Heat exchanger': {
            'U-tube shell & tube': u_shell_tube_hx_towler_2010,
            'Floating head shell & tube': floating_head_shell_tube_hx_towler_2010,
            'Double pipe': double_pipe_hx_towler_2010,
            'Thermosiphon reboiler': thermosiphon_reboiler_towler_2010,
            'U-tube kettle reboiler': u_tube_kettle_reboiler_towler_2010,
            'Plate & frame': plate_frame_hx_towler_2010,
            'Plate fin three-streams': three_stream_plate_fin_hx_esdu_1997,
            'Plate fin four-streams': four_stream_plate_fin_hx_esdu_1997
        },
        'Furnace/heater': {
            'Cylindrical furnace': cylindrical_furnace_towler_2010,
            'Box furnace': box_furnace_towler_2010,
            'Pyrolysis furnace': pyrolysis_furnace_ulrich_2003,
            'Electric furnace': electric_arc_furnace_parkinson_2016
        },
        'Motor/generator': {
            'Explosion proof': explosion_proof_motor_ulrich_2003,
            'Totally enclosed': totally_enclosed_motor_ulrich_2003
        },
        'PSA': psa_towler_1994,
        'Pump': single_stage_centrifugal_pump_towler_2010,
        'Reactor': {
            'Fluidized Bed': indirect_fluidbed_ulrich_2003,
            'Vertical CS Vessel': vertical_cs_press_vessel_towler_2010,
            'Horizontal CS Vessel': horizontal_cs_press_vessel_towler_2010,
            'Vertical 304SS Vessel': vertical_304ss_press_vessel_towler_2010,
            'Horizontal 304SS Vessel': horizontal_304ss_press_vessel_towler_2010,
        },
        'Turbine': {
            'Condensing steam': condensing_steam_turbine_towler_2010,
            'Axial gas': axial_gas_turbine_ulrich_2003,
            'Radial expander': radial_expander_ulrich_2003
        },
        'Vessel': {
            'Vertical CS': vertical_cs_press_vessel_towler_2010,
            'Horizontal CS': horizontal_cs_press_vessel_towler_2010,
            'Vertical 304SS': vertical_304ss_press_vessel_towler_2010,
            'Horizontal 304SS': horizontal_304ss_press_vessel_towler_2010
        }
    }

    def __init__(self, name: str, process_type: str, material: str, param: float, type: str, subtype: str = None):
        """
        Initialize an equipment instance with specified attributes.

        Parameters:
            name (str): The name of the equipment.
            process_type (str): The type of process the equipment is used for.
            material (str): The material of construction for the equipment.
            param (float): A parameter relevant to the equipment (e.g., size, capacity).
            type (str): The general type/category of the equipment.
            subtype (str, optional): The specific subtype of the equipment. Defaults to None.

        Attributes:
            name (str): Equipment name.
            type (str): Equipment type.
            subtype (str or None): Equipment subtype.
            param (float): Equipment parameter.
            material (str): Equipment material.
            process_type (str): Equipment process type.
            cost_year (None): Year of cost estimation (initialized as None).
            num_units (None): Number of units (initialized as None).
            purchased_cost (float): Calculated purchased cost of the equipment.
            direct_cost (float): Calculated direct cost of the equipment.
        """
        self.name = name
        self.type = type
        self.subtype = subtype
        self.param = param
        self.material = material
        self.process_type = process_type
        self.cost_year, self.num_units = None, None
        self.purchased_cost = self.calculate_equipment_cost()
        self.direct_cost = self.calculate_direct_cost()

    def calculate_equipment_cost(self) -> float:
        """
        Calculates the total equipment cost based on the equipment type, subtype, and parameters.
        This method selects the appropriate cost correlation function for the equipment type and subtype,
        computes the purchased cost, and applies an inflation adjustment to return the final cost.
        Returns:
            float: The inflation-adjusted equipment cost.
        Raises:
            ValueError: If no cost correlation is available for the specified equipment type or subtype.
        """
        if self.type not in self.cost_funcs:
            raise ValueError(f'No available cost correlations for equipment type: {self.type}')

        cost_funcs = self.cost_funcs[self.type]
        if isinstance(cost_funcs, dict):
            if self.subtype not in cost_funcs:
                raise ValueError(f'No available cost correlations for subtype: {self.subtype}')
            cost_funcs = cost_funcs[self.subtype]

        purchased_cost, self.num_units, self.cost_year = cost_funcs(self.param)
        return inflation_adjustment(purchased_cost, self.cost_year)

    def calculate_direct_cost(self) -> float:
        """
        Calculates the direct cost of equipment based on process and material factors.
        Returns:
            float: The calculated direct cost.
        Raises:
            ValueError: If the process type is not found in the process factors.
            ValueError: If the material is not found in the material factors.
        Notes:
            The direct cost is computed using the purchased cost, process factors, and material factors.
            The formula applied is:
                direct_cost = purchased_cost * ((1 + fp) * fm + (fer + fel + fi + fc + fs + fl))
            where:
                - fp: process factor
                - fm: material factor
                - fer, fel, fi, fc, fs, fl: additional process-related factors
        """
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
        return (f"Name={self.name}, Type={self.type}, Sub-type={self.subtype}, "
                f"Material={self.material}, Process Type={self.process_type}, "
                f"Parameter={self.param}, Purchased Cost={self.purchased_cost}, "
                f"Direct Cost={self.direct_cost})")
