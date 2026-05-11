from openpytea import (
    direct_costs_data,
    fixed_capital_data,
    fixed_opex_data,
    variable_opex_data,
    sensitivity_data,
    tornado_data,
    monte_carlo
)


def test_cost_breakdown_data(test_plant):
    direct = direct_costs_data(test_plant)
    capex = fixed_capital_data(test_plant)
    fixed_opex = fixed_opex_data(test_plant)
    variable_opex = variable_opex_data(test_plant)

    for result in [direct, capex, fixed_opex, variable_opex]:
        assert isinstance(result, dict)
        assert "values" in result
        assert "labels" in result
        assert "xlabels" in result


def test_sensitivity_data(test_plant):
    result = sensitivity_data(
        test_plant,
        parameter="interest_rate",
        plus_minus_value=0.2,
        n_points=5,
        metric="NPV",
    )

    assert isinstance(result, dict)
    assert "curves" in result
    assert "xlabel" in result
    assert "ylabel" in result
    assert len(result["curves"]) == 1


def test_tornado_data(test_plant):
    result = tornado_data(
        test_plant,
        plus_minus_value=0.1,
        metric="NPV",
    )

    assert isinstance(result, dict)
    assert "lows" in result
    assert "highs" in result
    assert "labels" in result


def test_monte_carlo_data(test_plant):
    # Configure price uncertainty directly on the plant attributes (new API)
    test_plant.variable_opex_inputs["electricity"].update(
        {"std": 0.01, "min": 0.05, "max": 0.12}
    )
    test_plant.plant_products["hydrogen"].update(
        {"std": 0.5, "min": 4.0, "max": 6.0}
    )
    # Scalar parameter uncertainty via project_uncertainties
    test_plant.project_uncertainties = {
        "interest_rate": {"std": 0.01, "min": 0.05, "max": 0.10},
    }

    result = monte_carlo(
        test_plant,
        num_samples=1000,
        batch_size=1000,
        additional_capex=False,
    )

    assert isinstance(result, dict)
    assert "metrics" in result
    assert "inputs" in result
    assert "NPV" in result["metrics"]
    assert "ROI" in result["metrics"]
    assert "PBT" in result["metrics"]
    assert "LCOP" in result["metrics"]
    assert len(result["metrics"]["NPV"]) == 1000
    # Inputs dict should contain the sampled parameters
    assert "Interest rate" in result["inputs"]
    assert "Electricity price" in result["inputs"]
    assert "Hydrogen product price" in result["inputs"]


def test_monte_carlo_utilization_tax_uncertainty(test_plant):
    # plant_utilization and tax_rate only appear in inputs when std > 0
    test_plant.project_uncertainties = {
        "plant_utilization": {"std": 0.05, "min": 0.7, "max": 1.0},
        "tax_rate": {"std": 0.02, "min": 0.15, "max": 0.35},
    }

    result = monte_carlo(test_plant, num_samples=200, batch_size=200)

    assert "Plant utilization" in result["inputs"]
    assert "Tax rate" in result["inputs"]
    assert len(result["inputs"]["Plant utilization"]) == 200
    assert len(result["inputs"]["Tax rate"]) == 200


def test_monte_carlo_no_price_variation(test_plant):
    # When no std is set on variable_opex or products, MC still runs via
    # default distributions for scalar params (project_lifetime, etc.)
    result = monte_carlo(test_plant, num_samples=100, batch_size=100)

    assert "LCOP" in result["metrics"]
    assert "NPV" in result["metrics"]
    assert len(result["metrics"]["LCOP"]) == 100
    # plant_utilization and tax_rate should NOT be in inputs by default
    assert "Plant utilization" not in result["inputs"]
    assert "Tax rate" not in result["inputs"]
