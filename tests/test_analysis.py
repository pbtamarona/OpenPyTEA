from copy import deepcopy

import numpy as np
import pytest

from openpytea import (
    direct_costs_data,
    fixed_capital_data,
    fixed_opex_data,
    variable_opex_data,
    cash_flow_data,
    sensitivity_data,
    tornado_data,
    monte_carlo
)
from openpytea.helpers import _apply_dependencies


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


def test_cash_flow_data(test_plant):
    result = cash_flow_data(test_plant)

    assert isinstance(result, dict)
    assert "curves" in result
    assert len(result["curves"]) == 1

    curve = result["curves"][0]
    assert curve["years"][0] == 0
    assert curve["years"][-1] == curve["project_life"]
    assert len(curve["years"]) == len(curve["cumulative"])
    assert curve["max_investment"] >= 0

    if curve["breakeven_year"] is not None:
        assert 0 <= curve["breakeven_year"] <= curve["project_life"]


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
    # Legacy (pre-price_uncertainty) layout: price dist fields sit directly
    # on the item, alongside "price" -- kept working for backward
    # compatibility (see test_monte_carlo_price_uncertainty_backward_compatible
    # below for a direct comparison against the new nested layout).
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


def test_monte_carlo_price_uncertainty_nested(test_plant):
    # Preferred layout: price dist fields nested under "price_uncertainty",
    # mirroring consumption_uncertainty/production_uncertainty
    test_plant.variable_opex_inputs["electricity"]["price_uncertainty"] = {
        "std": 0.01, "min": 0.05, "max": 0.12,
    }
    test_plant.plant_products["hydrogen"]["price_uncertainty"] = {
        "std": 0.5, "min": 4.0, "max": 6.0,
    }

    result = monte_carlo(test_plant, num_samples=1000, batch_size=1000)

    assert "Electricity price" in result["inputs"]
    assert "Hydrogen product price" in result["inputs"]
    elec_price = result["inputs"]["Electricity price"]
    assert elec_price.min() >= 0.05 and elec_price.max() <= 0.12


def test_monte_carlo_price_uncertainty_backward_compatible(test_plant):
    # The legacy flat layout and the new nested "price_uncertainty" layout
    # must draw identically for the same seed
    flat_plant = test_plant
    flat_plant.variable_opex_inputs["electricity"].update(
        {"dist_id": 5, "min": 0.05, "max": 0.12}
    )

    nested_plant = deepcopy(test_plant)
    nested_plant.variable_opex_inputs["electricity"].pop("dist_id")
    nested_plant.variable_opex_inputs["electricity"].pop("min")
    nested_plant.variable_opex_inputs["electricity"].pop("max")
    nested_plant.variable_opex_inputs["electricity"]["price_uncertainty"] = {
        "dist_id": 5, "min": 0.05, "max": 0.12,
    }

    r_flat = monte_carlo(flat_plant, num_samples=500, batch_size=500, random_seed=42)
    r_nested = monte_carlo(
        nested_plant, num_samples=500, batch_size=500, random_seed=42
    )

    assert np.allclose(
        r_flat["inputs"]["Electricity price"],
        r_nested["inputs"]["Electricity price"],
    )


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


def test_monte_carlo_dependency_multi_parent(test_plant):
    # water's consumption depends on BOTH hydrogen production and
    # electricity consumption -- a linear combination of two parents
    test_plant.plant_products["hydrogen"]["production_uncertainty"] = {"std": 5}
    test_plant.variable_opex_inputs["electricity"][
        "consumption_uncertainty"
    ] = {"std": 10}
    test_plant.variable_opex_inputs["water"]["consumption_dependency"] = {
        "depends_on": {
            "production:hydrogen": 2.0,
            "consumption:electricity": 0.1,
        },
    }

    result = monte_carlo(test_plant, num_samples=2000, batch_size=500, random_seed=1)
    inputs = result["inputs"]
    hydrogen = np.array(inputs["Hydrogen production"])
    electricity = np.array(inputs["Electricity consumption"])
    water = np.array(inputs["Water consumption"])

    assert np.allclose(water, 2.0 * hydrogen + 0.1 * electricity)


def test_monte_carlo_dependency_own_noise(test_plant):
    # A dependent can now also carry its own additive noise on top of its
    # DAG-implied mean instead of being perfectly deterministic
    test_plant.plant_products["hydrogen"]["production_uncertainty"] = {"std": 5}
    test_plant.variable_opex_inputs["water"]["consumption_dependency"] = {
        "depends_on": {"production:hydrogen": 2.0},
    }
    test_plant.variable_opex_inputs["water"]["consumption_uncertainty"] = {"noise": 3.0}

    result = monte_carlo(test_plant, num_samples=20000, batch_size=5000, random_seed=2)
    inputs = result["inputs"]
    hydrogen = np.array(inputs["Hydrogen production"])
    water = np.array(inputs["Water consumption"])

    residual = water - 2.0 * hydrogen
    assert abs(residual.mean()) < 1.0
    assert residual.std() > 0


def test_monte_carlo_dependency_std_scale_rejected(test_plant):
    # "std"/"scale" describe an item's own absolute value; once it's a
    # dependent, its own uncertainty must use "noise" instead -- "std"/
    # "scale" now raise ValueError rather than being silently accepted as
    # aliases (this feature is unreleased, so no backward compatibility
    # with the old std/scale/noise-interchangeable behavior is kept here)
    def configure(plant, field):
        plant.plant_products["hydrogen"]["production_uncertainty"] = {"std": 5}
        plant.variable_opex_inputs["water"]["consumption_dependency"] = {
            "depends_on": {"production:hydrogen": 2.0},
        }
        plant.variable_opex_inputs["water"]["consumption_uncertainty"] = {field: 3.0}
        return plant

    for field in ("std", "scale"):
        plant = deepcopy(test_plant)
        with pytest.raises(ValueError, match='"noise" instead of "std"/"scale"'):
            monte_carlo(configure(plant, field), num_samples=100,
                        batch_size=100, random_seed=9)

    # "noise" itself must still work and actually vary
    result = monte_carlo(configure(deepcopy(test_plant), "noise"),
                          num_samples=2000, batch_size=1000, random_seed=9)
    assert np.std(result["inputs"]["Water consumption"]) > 0


def test_monte_carlo_dependency_noise_propagates_downstream(test_plant):
    # electricity depends on hydrogen and carries its own noise; water
    # chains onto electricity, so it must track electricity's *actual*
    # (noisy) value exactly, not just the noiseless hydrogen-implied mean
    test_plant.plant_products["hydrogen"]["production_uncertainty"] = {"std": 5}
    test_plant.variable_opex_inputs["electricity"]["consumption_dependency"] = {
        "depends_on": {"production:hydrogen": 2.0},
    }
    test_plant.variable_opex_inputs["electricity"][
        "consumption_uncertainty"
    ] = {"noise": 3.0}
    test_plant.variable_opex_inputs["water"]["consumption_dependency"] = {
        "depends_on": {"consumption:electricity": 0.5},
    }

    result = monte_carlo(test_plant, num_samples=500, batch_size=500, random_seed=3)
    inputs = result["inputs"]
    electricity = np.array(inputs["Electricity consumption"])
    water = np.array(inputs["Water consumption"])

    assert np.allclose(water, 0.5 * electricity)


def test_monte_carlo_dependency_project_scalar(test_plant):
    # An economic scalar (fixed_capital_factor) can depend on a process
    # parameter -- e.g. higher production capacity needs more fixed capital
    test_plant.plant_products["hydrogen"]["production_uncertainty"] = {"std": 5}
    test_plant.project_uncertainties["fixed_capital_factor"] = {
        "dependency": {"depends_on": {"production:hydrogen": 0.01}, "offset": 0.5},
    }

    result = monte_carlo(test_plant, num_samples=2000, batch_size=500, random_seed=1)
    inputs = result["inputs"]
    hydrogen = np.array(inputs["Hydrogen production"])
    fc = np.array(inputs["Fixed capital factor"])

    assert np.allclose(fc, 0.01 * hydrogen + 0.5)


def test_monte_carlo_dependency_project_own_noise(test_plant):
    test_plant.plant_products["hydrogen"]["production_uncertainty"] = {"std": 5}
    test_plant.project_uncertainties["fixed_capital_factor"] = {
        "dependency": {"depends_on": {"production:hydrogen": 0.01}, "offset": 0.5},
        "noise": 0.05,
    }

    result = monte_carlo(test_plant, num_samples=20000, batch_size=5000, random_seed=2)
    inputs = result["inputs"]
    hydrogen = np.array(inputs["Hydrogen production"])
    fc = np.array(inputs["Fixed capital factor"])

    residual = fc - (0.01 * hydrogen + 0.5)
    assert abs(residual.mean()) < 0.01
    assert residual.std() > 0


def test_monte_carlo_dependency_reverse_direction(test_plant):
    # A process parameter can equally depend on an economic scalar
    test_plant.variable_opex_inputs["electricity"]["consumption_dependency"] = {
        "depends_on": {"project:interest_rate": 1000.0}, "offset": 50.0,
    }

    result = monte_carlo(test_plant, num_samples=2000, batch_size=500, random_seed=3)
    inputs = result["inputs"]
    interest_rate = np.array(inputs["Interest rate"])
    electricity = np.array(inputs["Electricity consumption"])

    assert np.allclose(electricity, 1000.0 * interest_rate + 50.0)


def test_monte_carlo_dependency_operator_hourly_rate(test_plant):
    # operator_hourly_rate lives in its own config dict (not
    # project_uncertainties) but is dependency-capable the same way
    test_plant.plant_products["hydrogen"]["production_uncertainty"] = {"std": 5}
    test_plant.operator_hourly_rate["dependency"] = {
        "depends_on": {"production:hydrogen": 0.1}, "offset": 20.0,
    }

    result = monte_carlo(test_plant, num_samples=2000, batch_size=500, random_seed=4)
    inputs = result["inputs"]
    hydrogen = np.array(inputs["Hydrogen production"])
    op_rate = np.array(inputs["Operator hourly rate"])

    assert np.allclose(op_rate, 0.1 * hydrogen + 20.0)


def test_tornado_baseline_not_stale_after_direct_edit(test_plant):
    # The LCOP branch of _evaluate_metric must recompute like every
    # other metric -- a cached levelized_cost is never invalidated, so
    # trusting it de-centers the tornado baseline after a direct edit
    test_plant.calculate_levelized_cost()  # populate the cache

    test_plant.variable_opex_inputs["electricity"]["price"] *= 10
    fresh_lcop = deepcopy(test_plant).calculate_levelized_cost()

    data = tornado_data(test_plant, plus_minus_value=0.1, metric="LCOP")
    assert np.isclose(data["base_value"], fresh_lcop)


def test_sample_distribution_impossible_bounds_raises():
    from openpytea.analysis import sample_distribution

    # A window excluding essentially all probability mass must raise a
    # clear error instead of spinning in the rejection loop forever
    with pytest.raises(ValueError, match="probability mass"):
        sample_distribution(3, 100, loc=0.0, scale=1.0, minimum=50, maximum=60)


def test_monte_carlo_out_of_range_baselines(test_plant):
    # Default truncation bounds are centered on each baseline, so
    # negative prices (disposal credits), very large prices (JPY/IDR
    # scale), and operator rates outside the old [10, 100] window all
    # sample correctly instead of hanging or piling up at a bound
    test_plant.variable_opex_inputs["waste"] = {
        "consumption": 5, "price": -20.0,
    }
    test_plant.variable_opex_inputs["catalyst"] = {
        "consumption": 1, "price": 150_000.0,
    }
    test_plant.operator_hourly_rate = {"rate": 150}

    result = monte_carlo(test_plant, num_samples=2000, batch_size=500, random_seed=11)
    inputs = result["inputs"]

    assert np.allclose(inputs["Waste price"], -20.0)
    assert np.allclose(inputs["Catalyst price"], 150_000.0)
    op_rate = np.array(inputs["Operator hourly rate"])
    assert abs(op_rate.mean() - 150.0) < 2.0
    assert (op_rate >= 130.0).all() and (op_rate <= 170.0).all()


def test_monte_carlo_explicit_price_bounds_still_win(test_plant):
    test_plant.plant_products["hydrogen"]["price_uncertainty"] = {
        "std": 2.0, "min": 4.0, "max": 5.5,
    }

    result = monte_carlo(test_plant, num_samples=2000, batch_size=500, random_seed=12)
    price = np.array(result["inputs"]["Hydrogen product price"])

    assert (price >= 4.0).all() and (price <= 5.5).all()


def test_monte_carlo_dependent_noise_ignores_rate_alias(test_plant):
    # Legacy flat layout: "noise" sits right next to "rate" in the same
    # dict. The noise must be centered at 0, not at the "rate" value --
    # reading rate=25 as the noise center used to put the truncation
    # window [-4, 4] ~10 sigma away and hang the rejection sampling
    # forever.
    test_plant.plant_products["hydrogen"]["production_uncertainty"] = {"std": 5}
    test_plant.operator_hourly_rate["dependency"] = {
        "depends_on": {"production:hydrogen": 0.1}, "offset": 20.0,
    }
    test_plant.operator_hourly_rate["noise"] = 2.0

    result = monte_carlo(test_plant, num_samples=2000, batch_size=500, random_seed=6)
    inputs = result["inputs"]
    hydrogen = np.array(inputs["Hydrogen production"])
    op_rate = np.array(inputs["Operator hourly rate"])

    residual = op_rate - (0.1 * hydrogen + 20.0)
    assert abs(residual.mean()) < 0.2
    assert residual.std() > 0


def test_monte_carlo_operator_rate_uncertainty_nested_block(test_plant):
    # Preferred layout: uncertainty in a nested "rate_uncertainty" dict,
    # uniform with price/consumption/production_uncertainty; loc defaults
    # to the item's own baseline rate
    test_plant.operator_hourly_rate["rate_uncertainty"] = {"std": 2.0}

    result = monte_carlo(test_plant, num_samples=2000, batch_size=500, random_seed=7)
    op_rate = np.array(result["inputs"]["Operator hourly rate"])

    assert abs(op_rate.mean() - 25.0) < 0.5
    assert op_rate.std() > 0


def test_monte_carlo_operator_rate_nested_noise_on_dependent(test_plant):
    test_plant.plant_products["hydrogen"]["production_uncertainty"] = {"std": 5}
    test_plant.operator_hourly_rate["dependency"] = {
        "depends_on": {"production:hydrogen": 0.1}, "offset": 20.0,
    }
    test_plant.operator_hourly_rate["rate_uncertainty"] = {"noise": 2.0}

    result = monte_carlo(test_plant, num_samples=2000, batch_size=500, random_seed=8)
    inputs = result["inputs"]
    hydrogen = np.array(inputs["Hydrogen production"])
    op_rate = np.array(inputs["Operator hourly rate"])

    residual = op_rate - (0.1 * hydrogen + 20.0)
    assert abs(residual.mean()) < 0.2
    assert residual.std() > 0


def test_monte_carlo_dependent_parent_not_seeded_as_baseline(test_plant):
    # A dependent used as another dependent's parent must feed its
    # DAG-resolved samples to the child, not get lazily seeded as a
    # baseline constant. plant_utilization is opt-in (lazily seedable),
    # and consumption nodes are walked before project nodes, so the
    # child is reached while its parent is still pending -- the exact
    # ordering that used to trigger the eager seeding.
    test_plant.plant_products["hydrogen"]["production_uncertainty"] = {"std": 5}
    test_plant.project_uncertainties["plant_utilization"] = {
        "dependency": {"depends_on": {"production:hydrogen": 0.004}, "offset": 0.70},
    }
    test_plant.variable_opex_inputs["electricity"]["consumption_dependency"] = {
        "depends_on": {"project:plant_utilization": 100.0},
    }

    result = monte_carlo(test_plant, num_samples=2000, batch_size=500, random_seed=5)
    inputs = result["inputs"]
    hydrogen = np.array(inputs["Hydrogen production"])
    utilization = np.array(inputs["Plant utilization"])
    electricity = np.array(inputs["Electricity consumption"])

    assert np.allclose(utilization, 0.004 * hydrogen + 0.70)
    assert np.allclose(electricity, 100.0 * utilization)
    assert electricity.std() > 0


def test_monte_carlo_dependency_makes_opt_in_scalar_visible(test_plant):
    # plant_utilization/tax_rate normally stay absent from "inputs" unless
    # independently configured; becoming a dependent opts them in too
    test_plant.plant_products["hydrogen"]["production_uncertainty"] = {"std": 5}
    test_plant.project_uncertainties["tax_rate"] = {
        "dependency": {"depends_on": {"production:hydrogen": 0.001}, "offset": 0.2},
    }

    result = monte_carlo(test_plant, num_samples=200, batch_size=200, random_seed=5)

    assert "Tax rate" in result["inputs"]
    assert "Plant utilization" not in result["inputs"]


def test_monte_carlo_dependency_unknown_project_param(test_plant):
    test_plant.variable_opex_inputs["electricity"]["consumption_dependency"] = {
        "depends_on": {"project:nonexistent": 1.0},
    }

    with pytest.raises(ValueError, match="must name one of"):
        monte_carlo(test_plant, num_samples=100, batch_size=100, random_seed=1)


def test_monte_carlo_dependency_cycle(test_plant):
    test_plant.variable_opex_inputs["electricity"]["consumption_dependency"] = {
        "depends_on": {"consumption:water": 1.0},
    }
    test_plant.variable_opex_inputs["water"]["consumption_dependency"] = {
        "depends_on": {"consumption:electricity": 1.0},
    }

    with pytest.raises(ValueError, match="cycle"):
        monte_carlo(test_plant, num_samples=100, batch_size=100, random_seed=1)


# ---------------------------------------------------------------------
# Dependencies in the deterministic analyses (sensitivity / tornado).
# The same "depends_on" configuration Monte Carlo honours above must also
# be honoured here -- see openpytea.helpers._apply_dependencies.
# ---------------------------------------------------------------------


def _tie_water_to_hydrogen(plant, factor=0.2):
    # water consumption = factor * hydrogen production. At the fixture's
    # baseline (hydrogen 50, water 10) factor=0.2 reproduces it exactly,
    # so only the *response* to a perturbation changes, not the baseline.
    plant.update_configuration({
        "variable_opex_inputs": {
            "water": {
                "consumption_dependency": {
                    "depends_on": {"production:hydrogen": factor},
                },
            },
        },
    })
    return plant


def test_apply_dependencies_propagates_through_a_chain(test_plant):
    # electricity follows hydrogen, water follows electricity
    test_plant.update_configuration({
        "variable_opex_inputs": {
            "electricity": {"consumption_dependency": {
                "depends_on": {"production:hydrogen": 2.0}}},
            "water": {"consumption_dependency": {
                "depends_on": {"consumption:electricity": 0.5}}},
        },
        "project_uncertainties": {
            "fixed_capital_factor": {"dependency": {
                "depends_on": {"production:hydrogen": 0.01}, "offset": 0.5}},
        },
    })
    test_plant.plant_products["hydrogen"]["production"] = 80
    _apply_dependencies(test_plant)

    assert test_plant.variable_opex_inputs["electricity"]["consumption"] == 160
    assert test_plant.variable_opex_inputs["water"]["consumption"] == 80
    assert test_plant.fc == pytest.approx(1.3)


def test_sensitivity_reflects_dependency(test_plant):
    # Varying a driver must move its dependents too, so the curve differs
    # from the same sweep on an otherwise identical independent plant
    independent = deepcopy(test_plant)
    dependent = _tie_water_to_hydrogen(test_plant)

    kwargs = dict(parameter="hydrogen.production", plus_minus_value=0.5,
                  n_points=5, metric="LCOP")
    dep_curve = sensitivity_data(dependent, **kwargs)["curves"][0]
    ind_curve = sensitivity_data(independent, **kwargs)["curves"][0]

    # Same baseline (the dependency reproduces it), different response
    assert dep_curve["baseline"] == pytest.approx(ind_curve["baseline"])
    assert dep_curve["y"][2] == pytest.approx(dep_curve["baseline"])
    assert not np.allclose(dep_curve["y"], ind_curve["y"])


def test_sensitivity_quantity_parameter_labels(test_plant):
    result = sensitivity_data(test_plant, parameter="electricity.consumption",
                              plus_minus_value=0.2, n_points=3)

    assert result["parameter"] == "variable_opex_inputs.electricity.consumption"
    assert "Electricity consumption" in result["xlabel"]
    assert "price" not in result["xlabel"]


def test_sensitivity_rejects_a_dependent_parameter(test_plant):
    _tie_water_to_hydrogen(test_plant)

    with pytest.raises(ValueError, match="set by a dependency"):
        sensitivity_data(test_plant, parameter="water.consumption",
                         plus_minus_value=0.5, n_points=3)


def test_sensitivity_baseline_resolves_dependencies(test_plant):
    # A dependency that does NOT reproduce the configured value must move
    # the baseline, not just the perturbed points
    test_plant.calculate_levelized_cost()
    unresolved = test_plant.levelized_cost

    # water consumption becomes 50, not the configured 10
    _tie_water_to_hydrogen(test_plant, factor=1.0)
    curve = sensitivity_data(test_plant, parameter="hydrogen.production",
                             plus_minus_value=0.5, n_points=3)["curves"][0]

    assert curve["baseline"] != pytest.approx(unresolved)
    assert curve["baseline"] == pytest.approx(curve["y"][1])


def test_tornado_factors_default_to_prices_and_scalars(test_plant):
    baseline = {
        "fixed_capital", "fixed_opex", "project_lifetime", "interest_rate",
        "operator_hourly_rate", "variable_opex_inputs.electricity",
        "variable_opex_inputs.water",
    }
    assert set(tornado_data(
        test_plant, plus_minus_value=0.1, metric="LCOP")["factors"]) == baseline

    # Configuring a dependency must NOT switch process parameters on: the
    # two are independent knobs
    _tie_water_to_hydrogen(test_plant)
    factors = set(tornado_data(
        test_plant, plus_minus_value=0.1, metric="LCOP")["factors"])

    assert "plant_products.hydrogen.production" not in factors
    assert "variable_opex_inputs.electricity.consumption" not in factors
    # ...except that the dependent itself can no longer be varied, so a
    # dependent economic scalar still drops out
    assert factors == baseline


def test_tornado_include_process_params_without_dependencies(test_plant):
    # No dependency anywhere -- process parameters are still available,
    # because they are ordinary economic drivers in their own right
    factors = set(tornado_data(test_plant, plus_minus_value=0.1,
                               metric="LCOP",
                               include_process_params=True)["factors"])

    assert "variable_opex_inputs.electricity.consumption" in factors
    assert "variable_opex_inputs.water.consumption" in factors
    assert "plant_products.hydrogen.production" in factors
    # Prices and scalars are still there alongside them
    assert "variable_opex_inputs.electricity" in factors
    assert "fixed_capital" in factors


def test_tornado_include_process_params_excludes_dependents(test_plant):
    # water's consumption is set by hydrogen production, and
    # fixed_capital_factor by it too -- neither can be varied on its own
    test_plant.update_configuration({
        "variable_opex_inputs": {
            "water": {"consumption_dependency": {
                "depends_on": {"production:hydrogen": 0.2}}},
        },
        "project_uncertainties": {
            "fixed_capital_factor": {"dependency": {
                "depends_on": {"production:hydrogen": 0.004}}},
        },
    })

    result = tornado_data(test_plant, plus_minus_value=0.1, metric="LCOP",
                          include_process_params=True)
    factors = set(result["factors"])

    # Independent process parameters are ranked
    assert "plant_products.hydrogen.production" in factors
    assert "variable_opex_inputs.electricity.consumption" in factors
    # Dependents are not, whether process or economic
    assert "variable_opex_inputs.water.consumption" not in factors
    assert "fixed_capital" not in factors
    # Prices are untouched by the graph
    assert "variable_opex_inputs.water" in factors

    # Quantity labels are abbreviated to keep the y-axis readable
    labels = dict(zip(result["factors"], result["labels"]))
    assert labels["plant_products.hydrogen.production"] == "Hydrogen prod."
    assert (labels["variable_opex_inputs.electricity.consumption"]
            == "Electricity cons.")


def test_tornado_dependency_widens_the_driver_bar(test_plant):
    # With water tied to hydrogen, perturbing hydrogen production also
    # moves water consumption, so the driver's bar spans more than it
    # would if the same parameter moved alone
    independent = deepcopy(test_plant)
    dependent = _tie_water_to_hydrogen(test_plant)

    dep = tornado_data(dependent, plus_minus_value=0.5, metric="LCOP",
                       include_process_params=True)
    dep_i = dep["factors"].index("plant_products.hydrogen.production")
    dep_span = abs(dep["highs"][dep_i] - dep["lows"][dep_i])

    ind = tornado_data(independent, plus_minus_value=0.5, metric="LCOP",
                       include_process_params=True)
    ind_i = ind["factors"].index("plant_products.hydrogen.production")
    ind_span = abs(ind["highs"][ind_i] - ind["lows"][ind_i])

    assert dep_span != pytest.approx(ind_span)
    assert dep["base_value"] == pytest.approx(ind["base_value"])


def test_tornado_never_ranks_opt_in_economic_scalars(test_plant):
    # plant_utilization/tax_rate are not tornado factors, and a dependency
    # referencing one does not make them into factors either
    assert "plant_utilization" not in tornado_data(
        test_plant, plus_minus_value=0.1)["factors"]

    test_plant.update_configuration({
        "variable_opex_inputs": {
            "water": {"consumption_dependency": {
                "depends_on": {"project:plant_utilization": 11.0},
                "offset": 0.1}},
        },
    })

    factors = tornado_data(test_plant, plus_minus_value=0.1,
                           include_process_params=True)["factors"]
    assert "plant_utilization" not in factors
    assert "tax_rate" not in factors
    # Vary them by name through sensitivity_data instead
    curve = sensitivity_data(test_plant, parameter="plant_utilization",
                             plus_minus_value=0.1, n_points=3)["curves"][0]
    assert not np.allclose(curve["y"], curve["y"][0])


def test_explicit_none_dependency_is_sampled_normally(test_plant):
    # "consumption_dependency": None (programmatically disabled) must
    # mean "no dependency": the item is sampled like any other instead
    # of falling through both the sampler and the DAG into a KeyError
    test_plant.variable_opex_inputs["electricity"][
        "consumption_dependency"
    ] = None

    result = monte_carlo(test_plant, num_samples=200, batch_size=100, random_seed=13)
    electricity = np.array(result["inputs"]["Electricity consumption"])
    assert np.allclose(electricity, 100.0)  # baseline constant


def test_tornado_and_sensitivity_center_on_configured_fc(test_plant):
    # With fc != 1 the perturbation must straddle the actual baseline,
    # not the assumed factor of 1.0
    test_plant.fc = 1.3
    data = tornado_data(test_plant, plus_minus_value=0.1, metric="LCOP")
    idx = data["factors"].index("fixed_capital")
    low, high = data["lows"][idx], data["highs"][idx]
    assert low < data["base_value"] < high

    sens = sensitivity_data(
        test_plant, "fixed_capital", plus_minus_value=0.1,
        n_points=3, metric="LCOP",
    )
    curve = sens["curves"][0]
    mid = curve["y"][len(curve["y"]) // 2]
    assert np.isclose(mid, curve["baseline"])
