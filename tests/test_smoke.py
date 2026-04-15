from openpytea import (
    Equipment,
    Plant,
    direct_costs_data,
    fixed_capital_data,
    fixed_opex_data,
    variable_opex_data,
    sensitivity_data,
    tornado_data,
    plot_stacked_bar,
    plot_sensitivity,
    plot_tornado,
)


def test_public_api_import():
    assert Equipment is not None
    assert Plant is not None


def test_end_to_end_smoke(test_plant):
    purchased_cost = test_plant.calculate_purchased_cost()
    fixed_capital = test_plant.calculate_fixed_capital()
    variable_opex = test_plant.calculate_variable_opex()
    revenue = test_plant.calculate_revenue()
    fixed_opex = test_plant.calculate_fixed_opex()
    npv = test_plant.calculate_npv()
    lcop = test_plant.calculate_levelized_cost()

    assert purchased_cost > 0
    assert fixed_capital > 0
    assert variable_opex > 0
    assert revenue > 0
    assert fixed_opex > 0
    assert isinstance(npv, (int, float))
    assert isinstance(lcop, (int, float))

    direct = direct_costs_data(test_plant)
    capex = fixed_capital_data(test_plant)
    fixed_opex_data_result = fixed_opex_data(test_plant)
    variable_opex_data_result = variable_opex_data(test_plant)
    sensitivity = sensitivity_data(
        test_plant,
        parameter="interest_rate",
        plus_minus_value=0.2,
        n_points=5,
        metric="NPV",
    )
    tornado = tornado_data(
        test_plant,
        plus_minus_value=0.1,
        metric="NPV",
    )

    assert "values" in direct
    assert "values" in capex
    assert "values" in fixed_opex_data_result
    assert "values" in variable_opex_data_result
    assert "curves" in sensitivity
    assert "lows" in tornado
    assert "highs" in tornado

    ax1 = plot_stacked_bar(direct, show=False)
    ax2 = plot_sensitivity(sensitivity, show=False)
    ax3 = plot_tornado(tornado, show=False)

    assert ax1 is not None
    assert ax2 is not None
    assert ax3 is not None
