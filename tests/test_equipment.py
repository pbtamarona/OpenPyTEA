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
