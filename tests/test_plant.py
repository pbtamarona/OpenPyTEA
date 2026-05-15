import pytest
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


def test_project_uncertainties_valid(test_plant):
    test_plant.update_configuration({
        "project_uncertainties": {
            "interest_rate": {"std": 0.01, "min": 0.04, "max": 0.15},
            "plant_utilization": {"std": 0.05, "min": 0.7, "max": 1.0},
        }
    })
    assert test_plant.project_uncertainties["interest_rate"]["std"] == 0.01
    assert test_plant.project_uncertainties["plant_utilization"]["std"] == 0.05


def test_project_uncertainties_invalid_key(test_plant):
    with pytest.raises(ValueError, match="Unknown key"):
        test_plant.update_configuration({
            "project_uncertainties": {"nonexistent_param": {"std": 0.1}}
        })


def test_project_uncertainties_invalid_std(test_plant):
    with pytest.raises(ValueError, match="std.*≥ 0"):
        Plant({
            **test_plant.config,
            "project_uncertainties": {"interest_rate": {"std": -0.01}},
        })


def test_capex_ramp_custom(test_plant):
    # Custom 2-year build schedule should produce a valid fixed_capital
    test_plant.capex_ramp = [0.5, 0.5]
    npv_custom = test_plant.calculate_npv()
    assert isinstance(npv_custom, (int, float))


def test_capex_ramp_invalid_sum(test_plant):
    test_plant.capex_ramp = [0.5, 0.3]  # sums to 0.8, not 1.0
    with pytest.raises(ValueError, match="sum to 1.0"):
        test_plant.calculate_npv()


def test_production_ramp_custom(test_plant):
    test_plant.production_ramp = [0.0, 0.5, 1.0]
    npv = test_plant.calculate_npv()
    assert isinstance(npv, (int, float))


def test_production_ramp_out_of_bounds(test_plant):
    test_plant.production_ramp = [0.0, 1.5]  # 1.5 > 1.0
    with pytest.raises(ValueError, match="between 0 and 1"):
        test_plant.calculate_npv()


def test_capital_cost_factor_overrides(test_plant):
    baseline = test_plant.calculate_fixed_capital()
    test_plant.loc_factor = 1.5
    overridden = test_plant.calculate_fixed_capital()
    assert overridden != baseline


def test_fixed_capital_factors_override(test_plant):
    baseline = test_plant.calculate_fixed_capital()
    test_plant.fixed_capital_factors = {
        "osbl": 0.1,
        "de": 0.1,
        "contingency": 0.05,
    }
    overridden = test_plant.calculate_fixed_capital()
    assert overridden != baseline


def test_fixed_capital_components_override(test_plant):
    test_plant.fixed_capital_components = {"osbl": 999_999}
    test_plant.calculate_fixed_capital()
    assert test_plant.osbl == 999_999


def test_fixed_opex_factors_override(test_plant):
    test_plant.calculate_fixed_capital()
    baseline = test_plant.calculate_fixed_opex()
    test_plant.fixed_opex_factors = {"maintenance": 0.10}  # double the default 0.05
    overridden = test_plant.calculate_fixed_opex()
    assert overridden > baseline


def test_fixed_opex_components_override(test_plant):
    test_plant.calculate_fixed_capital()
    test_plant.fixed_opex_components = {"maintenance_costs": 999_999}
    test_plant.calculate_fixed_opex()
    assert test_plant.maintenance_costs == 999_999
