from openpytea import Plant


def test_plant_fixture_object(test_plant):
    assert isinstance(test_plant, Plant)
    assert test_plant.name == "Test Plant"
    assert test_plant.process_type == "Fluids"
    assert len(test_plant.equipment_list) == 2
    assert "electricity" in test_plant.variable_opex_inputs
    assert "hydrogen" in test_plant.plant_products


def test_plant_core_calculations(test_plant):
    assert test_plant.calculate_purchased_cost() > 0
    assert test_plant.calculate_fixed_capital() > 0
    assert test_plant.calculate_variable_opex() > 0
    assert test_plant.calculate_revenue() > 0
    assert test_plant.calculate_fixed_opex() > 0


def test_plant_financial_metrics(test_plant):
    npv = test_plant.calculate_npv()
    lcop = test_plant.calculate_levelized_cost()
    roi = test_plant.calculate_roi()

    assert isinstance(npv, (int, float))
    assert isinstance(lcop, (int, float))
    assert isinstance(roi, (int, float))


def test_plant_calculate_all(test_plant):
    test_plant.calculate_all()

    assert hasattr(test_plant, "fixed_capital")
    assert hasattr(test_plant, "revenue")
    assert hasattr(test_plant, "variable_production_costs")
