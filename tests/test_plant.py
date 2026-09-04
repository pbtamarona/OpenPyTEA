import warnings
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


def test_bare_calculate_calls_keep_configured_fc_fp(test_plant):
    # None means "keep the configured factor", not "reset to 1.0" -- a
    # bare call (e.g. from the fixed_capital_data plotting helper) must
    # not silently change the plant's economics
    test_plant.fc = 1.3
    test_plant.fp = 1.2

    with_factors = test_plant.calculate_fixed_capital()
    assert test_plant.fc == 1.3

    test_plant.calculate_variable_opex()
    test_plant.calculate_fixed_opex()
    assert test_plant.fp == 1.2

    plain = deepcopy(test_plant)
    plain.fc = None
    assert with_factors > plain.calculate_fixed_capital()
    assert plain.fc == 1.0  # unset still defaults to 1.0

    # An explicit argument still overrides
    test_plant.calculate_fixed_capital(fc=2.0)
    assert test_plant.fc == 2.0


def test_plotting_helpers_do_not_change_economics(test_plant):
    from openpytea import fixed_capital_data, fixed_opex_data

    test_plant.fc = 1.3
    test_plant.fp = 1.2
    lcop_before = test_plant.calculate_levelized_cost()

    fixed_capital_data(test_plant)
    fixed_opex_data(test_plant)

    assert test_plant.fc == 1.3
    assert test_plant.fp == 1.2
    assert np.isclose(test_plant.calculate_levelized_cost(), lcop_before)


def test_update_configuration_currency_and_exchange_rate(test_plant):
    base_cost = test_plant.calculate_purchased_cost()

    test_plant.update_configuration(
        {"currency": "EUR", "exchange_rate": 0.5}
    )
    assert test_plant.currency == "EUR"
    assert test_plant.exchange_rate == 0.5
    assert np.isclose(
        test_plant.calculate_purchased_cost(), 0.5 * base_cost
    )


def test_straight_line_truncation_keeps_statutory_amounts():
    from openpytea.plant import _straight_line_schedule

    # Life longer than horizon: every year deducts the statutory annual
    # amount and the tail is NOT dumped into the final year
    sched = _straight_line_schedule(1200.0, 12, 0.0, 10)
    assert np.allclose(sched, 100.0)
    assert np.isclose(sched.sum(), 1000.0)

    # Life within horizon: full write-off, exact to the last bit
    sched = _straight_line_schedule(1000.0, 3, 0.0, 20)
    assert np.isclose(sched.sum(), 1000.0)
    assert np.allclose(sched[:3], 1000.0 / 3)
    assert np.all(sched[3:] == 0)


def test_declining_balance_truncation_not_topped_up():
    from openpytea.plant import _declining_balance_schedule

    full = _declining_balance_schedule(1000.0, 12, 2.0, 0.0, 12)
    truncated = _declining_balance_schedule(1000.0, 12, 2.0, 0.0, 8)
    # A truncated schedule is exactly the first years of the full one
    assert np.allclose(truncated, full[:8])
    assert truncated.sum() < 1000.0


def test_default_depreciation_life_fits_usable_horizon():
    from openpytea.plant import build_depreciation_array

    # Default config on a short project: full write-off, uniform
    # statutory amounts, no stranded value, no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dep = build_depreciation_array(12, {0: 1200.0})
    assert np.isclose(dep.sum(), 1200.0)
    # service starts year 2 -> default life = min(15, 12-2) = 10 years
    assert np.allclose(dep[2:12], 120.0)
    assert np.all(dep[:2] == 0)


def test_user_depreciation_life_beyond_horizon_warns():
    from openpytea.plant import build_depreciation_array

    with pytest.warns(UserWarning, match="usable horizon"):
        dep = build_depreciation_array(12, {0: 1200.0}, {"life": 15})
    # 10 usable years at the statutory 1200/15 = 80/year, rest stranded
    assert np.allclose(dep[2:12], 80.0)
    assert np.isclose(dep.sum(), 800.0)


def test_to_dict_serializes_equipment_purchased_cost(test_plant):
    test_plant.calculate_fixed_capital()
    items = test_plant.to_dict()["equipment_summary"]["items"]
    by_name = {item["name"]: item for item in items}
    # conftest fixture: Reactor purchased at 100k, Pump at 20k (2024),
    # inflation-adjusted to the target year, so > 0 is the contract here
    assert by_name["Reactor"]["purchased_cost"] > 0
    assert by_name["Pump"]["purchased_cost"] > 0
    assert by_name["Reactor"]["direct_cost"] > 0


def test_working_capital_tracks_fixed_capital(test_plant):
    # Auto-computed working capital must follow fixed_capital on every
    # recalculation, not freeze at its first computed value -- including
    # monte_carlo's per-sample fc arrays applied to a plant that already
    # ran a deterministic analysis.
    test_plant.calculate_fixed_capital()
    test_plant.calculate_variable_opex()
    test_plant.calculate_fixed_opex()
    assert np.isclose(
        test_plant.working_capital, 0.15 * test_plant.fixed_capital
    )

    test_plant.calculate_fixed_capital(fc=2.0)
    test_plant.calculate_fixed_opex()
    assert np.isclose(
        test_plant.working_capital, 0.15 * test_plant.fixed_capital
    )

    batch = deepcopy(test_plant)
    fc_samples = np.array([0.5, 1.0, 1.5])
    batch.calculate_fixed_capital(fc=fc_samples)
    batch.calculate_fixed_opex()
    assert np.asarray(batch.working_capital).shape == (3,)
    assert np.allclose(
        batch.working_capital, 0.15 * batch.fixed_capital
    )


def test_user_working_capital_is_preserved(test_plant):
    test_plant.update_configuration({"working_capital": 123_456.0})
    test_plant.calculate_fixed_capital()
    test_plant.calculate_variable_opex()
    test_plant.calculate_fixed_opex()
    assert test_plant.working_capital == 123_456.0

    test_plant.calculate_fixed_capital(fc=2.0)
    test_plant.calculate_fixed_opex()
    assert test_plant.working_capital == 123_456.0


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


def test_to_dict_with_multi_entry_additional_capex(test_plant):
    # After calculate_cash_flow, additional_capex_cost is a numpy array;
    # to_dict's truthiness check must not raise for 2+ entries
    test_plant.additional_capex_cost = [100_000, 100_000]
    test_plant.additional_capex_years = [5, 10]
    test_plant.calculate_npv()

    d = test_plant.to_dict()
    assert "roi_with_additional_capex" in d["metrics"]
