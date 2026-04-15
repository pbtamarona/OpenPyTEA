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
    test_plant.monte_carlo_inputs = {
        "interest_rate": {
            "mean": 0.08,
            "std": 0.01,
            "min": 0.05,
            "max": 0.10,
        },
        "variable_opex_inputs": {
            "electricity": {
                "price": {
                    "mean": 0.08,
                    "std": 0.01,
                    "min": 0.05,
                    "max": 0.12,
                }
            }
        },
        "plant_products": {
            "hydrogen": {
                "price": {
                    "mean": 5.0,
                    "std": 0.5,
                    "min": 4.0,
                    "max": 6.0,
                }
            }
        },
    }

    result = monte_carlo(
        test_plant,
        n_samples=1000,
        metrics=["NPV", "LCOP"],
    )

    assert isinstance(result, dict)
    assert "NPV" in result
    assert "LCOP" in result
    assert len(result["NPV"]) == 1000
    assert len(result["LCOP"]) == 1000
