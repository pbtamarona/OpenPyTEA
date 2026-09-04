import pytest
from openpytea import Equipment


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
