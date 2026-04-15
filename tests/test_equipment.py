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
