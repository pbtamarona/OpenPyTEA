import pytest
from openpytea import CompositeEquipment, Equipment, equipment_options


def test_equipment_fixture_objects(test_equipment):
    assert len(test_equipment) == 2

    reactor = test_equipment[0]
    pump = test_equipment[1]

    assert isinstance(reactor, Equipment)
    assert reactor.name == "Reactor"
    assert reactor.process_type == "Fluids"
    assert reactor.purchased_cost > 0
    assert reactor.direct_cost > 0

    assert isinstance(pump, Equipment)
    assert pump.name == "Pump"
    assert pump.category == "Pumps"
    assert pump.purchased_cost > 0
    assert pump.direct_cost > 0


def test_equipment_to_dict(test_equipment):
    equipment_dict = test_equipment[0].to_dict()

    assert isinstance(equipment_dict, dict)
    assert equipment_dict["name"] == "Reactor"
    assert "purchased_cost" in equipment_dict
    assert "direct_cost" in equipment_dict


def test_cost_db_does_not_mutate_input_dataframe():
    from openpytea.equipment import COST_DB_DF, CostCorrelationDB

    custom = COST_DB_DF.copy()
    custom.columns = [c.upper() for c in custom.columns]
    before = custom.columns.tolist()

    db = CostCorrelationDB(custom)

    assert custom.columns.tolist() == before  # caller's frame untouched
    assert all(c == c.lower() for c in db.df.columns)  # copy normalized


def test_num_units_multiplies_correlation_cost():
    one = Equipment(
        name="Fridge",
        param=180,
        process_type="Fluids",
        category="Utilities",
        type="Packaged mechanical refrigerator",
    )
    three = Equipment(
        name="Fridge",
        param=180,
        process_type="Fluids",
        category="Utilities",
        type="Packaged mechanical refrigerator",
        num_units=3,
    )
    assert one.num_units == 1
    assert three.num_units == 3
    assert three.purchased_cost == pytest.approx(3 * one.purchased_cost)
    assert three.direct_cost == pytest.approx(3 * one.direct_cost)


def test_num_units_does_not_multiply_direct_purchased_cost():
    eq = Equipment(
        name="Pump",
        param=1.0,
        process_type="Fluids",
        category="Pumps",
        purchased_cost=10_000,
        cost_year=2024,
        num_units=3,
    )
    # A direct purchased_cost is the total for all units; num_units is a label
    assert eq.num_units == 3
    assert eq.purchased_cost == pytest.approx(10_000)


def test_auto_parallel_units_unchanged_without_num_units():
    comp = Equipment(
        name="Air Compressor",
        param=50_000,
        process_type="Fluids",
        category="Compressors, fans, & blowers",
        type="Compressor, centrifugal",
    )
    # The database splits the duty into parallel units and the returned
    # cost already covers them; num_units reports that count.
    assert comp.num_units == 2
    half = Equipment(
        name="Air Compressor",
        param=25_000,
        process_type="Fluids",
        category="Compressors, fans, & blowers",
        type="Compressor, centrifugal",
    )
    assert comp.purchased_cost == pytest.approx(2 * half.purchased_cost)


def test_renamed_category_still_accepted():
    kwargs = dict(name="C", param=500, process_type="Fluids",
                  type="Centrifugal compressor, electric motor drive")
    new = Equipment(category="Compressors, fans, & blowers", **kwargs)
    with pytest.warns(FutureWarning, match="renamed"):
        old = Equipment(category="compressors & blowers", **kwargs)
    assert old.category == new.category
    assert old.purchased_cost == new.purchased_cost


@pytest.mark.parametrize("kwargs, error, match", [
    # fuzzy match on a category
    ({"category": "Reactor"}, KeyError, "Did you mean 'Reactors'"),
    ({"category": "Compressors, fans and blowers"}, KeyError,
     "Did you mean 'Compressors, fans, & blowers'"),
    # types are suggested only from within the given category
    ({"category": "Pumps", "type": "centrifugal pump"}, KeyError,
     "type 'centrifugal pump' for category 'Pumps'. Did you mean 'Centrifugal'"),
    # a partial word matches by substring
    ({"category": "Pumps", "type": "gear"}, KeyError,
     "Did you mean 'External gear'"),
    # nothing close: point to the lookup
    ({"category": "Pumps", "type": "zzz"}, KeyError,
     r"See openpytea.equipment_options\('Pumps'\)"),
    ({"category": "zzz"}, KeyError, r"See openpytea.equipment_options\(\)"),
    ({"category": "Pumps", "type": "Centrifugal", "process_type": "Fluid"},
     ValueError, "process_type 'Fluid'. Did you mean 'Fluids'"),
    ({"category": "Pumps", "type": "Centrifugal", "material": "304 stainless"},
     ValueError, "Did you mean '304 stainless steel'"),
])
def test_unknown_names_suggest_valid_ones(kwargs, error, match):
    kwargs = {"name": "X", "param": 10, "process_type": "Fluids", **kwargs}
    with pytest.raises(error, match=match):
        Equipment(**kwargs)


def test_composite_unknown_names_suggest_valid_ones():
    with pytest.raises(ValueError, match="Did you mean 'Fluids'"):
        CompositeEquipment(name="X", process_type="Fluid", components=[])
    with pytest.raises(ValueError, match="Did you mean 'composite'"):
        CompositeEquipment(name="X", process_type="Fluids",
                           installation="composit", components=[])


def test_equipment_options():
    categories = equipment_options()
    assert "Pumps" in categories and categories == sorted(categories)

    pumps = equipment_options("pumps")  # case-insensitive
    assert "Centrifugal" in pumps["type"].tolist()
    assert {"units", "s_lower", "s_upper", "default material"} <= set(pumps)

    # a listed type and its size range can be used as-is
    row = pumps.iloc[0]
    eq = Equipment(name="P", param=row["s_lower"], process_type="Fluids",
                   category="Pumps", type=row["type"])
    assert eq.purchased_cost > 0

    with pytest.raises(KeyError, match="Did you mean 'Reactors'"):
        equipment_options("Reactor")


def test_literal_hints_match_the_data():
    from typing import get_args
    from openpytea import Category, EquipmentType, Material, ProcessType
    from openpytea.equipment import CostCorrelationDB

    # A failure here means the CSV or a factor table changed: update the
    # Literal in equipment.py to match
    db = CostCorrelationDB().df
    assert set(get_args(ProcessType)) == set(Equipment.process_factors)
    assert set(get_args(Category)) == set(equipment_options())
    types = get_args(EquipmentType)
    assert len(types) == len(set(types))  # each name listed once
    assert set(types) == set(db["type"].dropna())
    defaults = db["default material"].dropna().str.strip()
    expected = (set(Equipment.material_factors) | set(defaults)) - {"n.a."}
    assert set(get_args(Material)) == expected


def test_old_cost_func_key_spelling_still_resolves():
    kw = dict(name="P", param=1000, process_type="Fluids", category="Pumps")
    new = Equipment(cost_func="centrifugal_1-stage_3600_vsc_seider_2013", **kw)
    with pytest.warns(DeprecationWarning, match="renamed"):
        old = Equipment(cost_func="centrifugal_1-stage_3600_VSC_seider_2013", **kw)
    assert old.purchased_cost == new.purchased_cost


def test_old_vsc_key_maps_to_hsc():
    kw = dict(name="P", param=1000, process_type="Fluids", category="Pumps")
    new = Equipment(cost_func="centrifugal_2-stage_3600_hsc_seider_2013", **kw)
    with pytest.warns(DeprecationWarning, match="hsc"):
        old = Equipment(cost_func="centrifugal_2-stage_3600_VSC_seider_2013", **kw)
    assert old.purchased_cost == new.purchased_cost


def test_old_type_spelling_with_extra_whitespace_still_matches():
    kw = dict(name="T", param=10, process_type="Fluids",
              category="Packings & adsorbents")
    new = Equipment(type="Pall rings, carbon steel, 1 in.", **kw)
    old = Equipment(type="Pall rings, carbon steel, 1 in. ", **kw)
    assert old.purchased_cost == new.purchased_cost


def test_old_key_year_maps_to_cost_year():
    with pytest.warns(DeprecationWarning, match="propeller_fan_seider_2013"):
        Equipment(name="F", param=5000, process_type="Fluids",
                  category="Compressors, fans, & blowers",
                  cost_func="propeller_fan_seider_2023")


@pytest.mark.parametrize(
    "old", ["Adsorption & membrane separation", "Gas separation", "CO2 capture"]
)
def test_merged_categories_still_work(old):
    new = "Absorption, adsorption, & membrane separation"
    with pytest.warns(FutureWarning, match="renamed"):
        options = equipment_options(old)
    assert options.equals(equipment_options(new))
    with pytest.warns(FutureWarning):
        eq = Equipment("RO", 1, "Fluids", old, "Reverse osmosis, brackish water")
    assert eq.category == new


def test_old_cfb_chp_type_still_works():
    new = "Circulating fluidized bed (CFB), combined heat & power (CHP)"
    ref = Equipment("CHP", 2, "Fluids", "Boilers, heaters, & furnaces", new)
    with pytest.warns(FutureWarning, match="renamed"):
        old = Equipment("CHP", 2, "Fluids", "Boilers, heaters, & furnaces", "CFB CHP")
    assert old.type == new
    assert old.purchased_cost == ref.purchased_cost


def test_biomass_combustion_merged_into_boilers():
    new = "Circulating fluidized bed (CFB), combined heat & power (CHP)"
    ref = Equipment("CHP", 2, "Fluids", "Boilers, heaters, & furnaces", new)
    with pytest.warns(FutureWarning, match="renamed"):
        old = Equipment("CHP", 2, "Fluids", "Biomass combustion", new)
    assert old.category == "Boilers, heaters, & furnaces"
    assert old.purchased_cost == ref.purchased_cost


def test_electrolysis_renamed_to_electrolyzers_and_fuel_cells():
    new = "Electrolyzers & fuel cells"
    ref = Equipment("EL", 10, "Fluids", new, "Electrolyzer (PEM / alkaline)")
    with pytest.warns(FutureWarning, match="renamed"):
        old = Equipment("EL", 10, "Fluids", "Electrolysis",
                        "Electrolyzer (PEM / alkaline)")
    assert old.category == new
    assert old.purchased_cost == ref.purchased_cost


# The first CSV row matching a category and type is its default correlation,
# so re-sorting the CSV silently switches these (often to other units)
@pytest.mark.parametrize("category, type, key", [
    ("Boilers, heaters, & furnaces", "Hot water heater",
     "hot_water_heater_turton_2001"),
    ("Boilers, heaters, & furnaces", "Pyrolysis furnace",
     "pyrolysis_furnace_turton_2001"),
    ("Boilers, heaters, & furnaces", "Reformer furnace",
     "reformer_furnace_turton_2001"),
    ("Boilers, heaters, & furnaces", "Steam boiler", "steam_boiler_turton_2001"),
    ("Conveyors", "Belt", "belt_conveyor_turton_2001"),
    ("Conveyors", "Pneumatic", "pneumatic_conveyor_turton_2001"),
    ("Conveyors", "Screw", "screw_conveyor_turton_2001"),
    ("Dryers", "Drum", "drum_dryer_turton_2001"),
    ("Dust collectors", "Electrostatic precipitator",
     "electrostatic_precipitator_turton_2001"),
    ("Evaporators", "Falling film", "falling_film_evaporator_turton_2001"),
    ("Filters", "Plate & frame", "plate_n_frame_filter_towler_2010"),
    ("Heat exchangers", "Plate & frame", "plate_frame_hx_towler_2010"),
    ("Heat exchangers", "Spiral plate", "spiral_plate_hx_turton_2001"),
    ("Heat exchangers", "Spiral tube", "spiral_tube_hx_turton_2001"),
    ("Motors & generators", "Internal combustion engine",
     "internal_combustion_engine_turton_2001"),
    ("Trays", "Bubble-cap", "bubble_cap_tray_towler_2010"),
    ("Trays", "Sieve", "sieve_tray_towler_2010"),
    ("Trays", "Valve", "valve_tray_towler_2010"),
    ("Turbines", "Gas turbine", "gas_turbine_turton_2001"),
])
def test_default_correlation_per_category_type(category, type, key):
    from openpytea.equipment import CostCorrelationDB
    assert CostCorrelationDB().key_for_category_type(category, type) == key
