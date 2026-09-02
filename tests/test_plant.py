from copy import deepcopy

import numpy as np
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


def test_project_uncertainties_dependency_valid(test_plant):
    # "dependency" is a new allowed key (a dict, not a number) in a
    # project_uncertainties entry, for the process<->economic DAG
    test_plant.update_configuration({
        "project_uncertainties": {
            "fixed_capital_factor": {
                "dependency": {"depends_on": {"production:hydrogen": 0.01}},
            },
        },
    })
    assert test_plant.project_uncertainties["fixed_capital_factor"]["dependency"] == {
        "depends_on": {"production:hydrogen": 0.01},
    }


def test_project_uncertainties_noise_key_allowed(test_plant):
    # "noise" is the required spelling of the scale parameter for a
    # dependent's own additive noise, so update_configuration's whitelist
    # must accept it alongside "std"/"scale" (it previously rejected it,
    # forcing users to bypass validation with a direct dict assignment)
    test_plant.update_configuration({
        "project_uncertainties": {
            "fixed_capital_factor": {
                "dependency": {"depends_on": {"production:hydrogen": 0.01}},
                "noise": 0.05,
            },
        },
    })
    cfg = test_plant.project_uncertainties["fixed_capital_factor"]
    assert cfg["noise"] == 0.05


def test_dependency_supersedes_stale_absolute_uncertainty(test_plant):
    # Turning an already-uncertain parameter into a dependent must drop its
    # old absolute-value spec instead of merging it. Previously the stale
    # "std" raised a confusing error naming a key the caller never wrote,
    # and stale "min"/"max" were silently re-read as noise-band bounds,
    # shifting the dependent off its DAG line with no warning.
    test_plant.update_configuration({
        "project_uncertainties": {
            "fixed_capital_factor": {
                "dist_id": 3, "std": 0.3, "min": 0.25, "max": 1.75,
            },
        },
    })
    test_plant.update_configuration({
        "project_uncertainties": {
            "fixed_capital_factor": {
                "dependency": {"depends_on": {"production:hydrogen": 0.01}},
                "noise": 0.05,
            },
        },
    })
    cfg = test_plant.project_uncertainties["fixed_capital_factor"]
    assert set(cfg) == {"dependency", "noise"}


def test_dependency_supersedes_stale_process_uncertainty(test_plant):
    # Same rule for a process item, whose spec lives in a nested sub-dict
    test_plant.update_configuration({
        "variable_opex_inputs": {
            "water": {"consumption_uncertainty": {"std": 2.0, "min": 8, "max": 12}},
        },
    })
    test_plant.update_configuration({
        "variable_opex_inputs": {
            "water": {
                "consumption_dependency": {"depends_on": {"production:hydrogen": 0.2}},
                "consumption_uncertainty": {"noise": 1.0},
            },
        },
    })
    assert test_plant.variable_opex_inputs["water"]["consumption_uncertainty"] == {
        "noise": 1.0,
    }


def test_dependency_allows_negative_noise_bounds(test_plant):
    # A dependent's min/max bound its additive noise around zero, so they
    # legitimately go negative -- the absolute-range rules (which require
    # e.g. fixed_capital_factor bounds > 0) must not apply to dependents
    test_plant.update_configuration({
        "project_uncertainties": {
            "fixed_capital_factor": {
                "dependency": {"depends_on": {"production:hydrogen": 0.01}},
                "noise": 0.05, "min": -0.1, "max": 0.1,
            },
        },
    })
    assert test_plant.project_uncertainties["fixed_capital_factor"]["min"] == -0.1


def test_absolute_bounds_still_validated_without_dependency(test_plant):
    # ...but a non-dependent keeps the absolute-range validation
    with pytest.raises(ValueError, match="must be > 0"):
        test_plant.update_configuration({
            "project_uncertainties": {"fixed_capital_factor": {"min": -0.1}},
        })


def test_project_uncertainties_dependency_must_be_dict(test_plant):
    with pytest.raises(TypeError, match="dependency.*must be a dict"):
        Plant({
            **test_plant.config,
            "project_uncertainties": {
                "fixed_capital_factor": {"dependency": "not-a-dict"},
            },
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


def test_vectorized_lifetimes_match_scalar_runs(test_plant):
    # Each sample in a vectorized (Monte Carlo) run must reproduce a
    # scalar run of its own lifetime: no revenue, costs, or taxes may
    # accrue past a sample's lifetime just because a longer-lived
    # sample shares the arrays.
    lifetimes = np.array([10, 20, 30])
    n = len(lifetimes)

    mc = deepcopy(test_plant)
    mc.plant_products["hydrogen"]["price"] = 300.0
    mc.update_configuration({
        "project_lifetime": lifetimes,
        "interest_rate": np.full(n, 0.08),
    })
    mc.calculate_fixed_capital(fc=np.ones(n))
    mc_npv = mc.calculate_npv()
    mc_lcop = mc.calculate_levelized_cost()
    mc_pbt = mc.calculate_payback_time()
    mc_roi = mc.calculate_roi()

    assert mc.revenue_array[0, lifetimes[0]:].sum() == 0
    assert mc.cash_flow[0, lifetimes[0]:].sum() == 0

    for i, lt in enumerate(lifetimes):
        scalar = deepcopy(test_plant)
        scalar.plant_products["hydrogen"]["price"] = 300.0
        scalar.update_configuration({"project_lifetime": int(lt)})
        assert np.isclose(mc_npv[i], scalar.calculate_npv())
        assert np.isclose(mc_lcop[i], scalar.calculate_levelized_cost())
        assert np.isclose(mc_pbt[i], scalar.calculate_payback_time())
        assert np.isclose(mc_roi[i], scalar.calculate_roi())
