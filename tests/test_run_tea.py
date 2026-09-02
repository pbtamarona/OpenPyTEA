import json
from openpytea import run_tea, run_openpytea


def _equipment_data():
    return {
        "equipment": [
            {
                "name": "Reactor",
                "param": 1.0,
                "process_type": "Fluids",
                "category": "Reactors",
                "purchased_cost": 100000,
                "cost_year": 2024,
            },
            {
                "name": "Pump",
                "param": 1.0,
                "process_type": "Fluids",
                "category": "Pumps",
                "purchased_cost": 20000,
                "cost_year": 2024,
            },
        ]
    }


def _plant_data():
    return {
        "plant": {
            "plant_name": "Test Plant",
            "process_type": "Fluids",
            "country": "United States",
            "region": "Gulf Coast",
            "currency": "USD",
            "exchange_rate": 1.0,
            "interest_rate": 0.08,
            "project_lifetime": 20,
            "plant_utilization": 0.9,
            "tax_rate": 0.25,
            "operator_hourly_rate": {
                "rate": 25,
                "std": 2,
                "min": 20,
                "max": 30,
            },
            "plant_products": {
                "hydrogen": {
                    "production": 50,
                    "price": 5.0,
                    "std": 0.5,
                    "min": 4.0,
                    "max": 6.0,
                }
            },
            "variable_opex_inputs": {
                "electricity": {
                    "consumption": 100,
                    "price": 0.08,
                    "std": 0.01,
                    "min": 0.05,
                    "max": 0.12,
                },
                "water": {
                    "consumption": 10,
                    "price": 0.5,
                    "std": 0.05,
                    "min": 0.4,
                    "max": 0.6,
                },
            },
        }
    }


def _analysis_data(output_dir=None):
    output = {
        "save_json": True,
        "save_plots": True,
    }
    if output_dir is not None:
        output["directory"] = str(output_dir)

    return {
        "analysis": {
            "direct_costs": {"run": True},
            "fixed_capital": {"run": True},
            "fixed_opex": {"run": True},
            "variable_opex": {"run": True},
            "levelized_cost": {"run": True},
            "cash_flow": {"run": True},
            "tornado": {
                "run": True,
                "args": {
                    "plus_minus_value": 0.1,
                    "metric": "NPV",
                },
            },
            "sensitivity": {
                "run": True,
                "cases": [
                    {
                        "name": "interest_rate_case",
                        "args": {
                            "parameter": "interest_rate",
                            "plus_minus_value": 0.2,
                            "n_points": 5,
                            "metric": "NPV",
                        },
                    }
                ],
            },
            "monte_carlo": {
                "run": True,
                "args": {
                    "num_samples": 100,
                    "batch_size": 20,
                    "additional_capex": False,
                },
                # lowercase on purpose: the metric check must be
                # case-insensitive (metrics keys are uppercase)
                "metric": ["lcop"],
                "plot_inputs": True,
            },
        },
        "output": output,
    }


def _assert_results(results, output_dir):
    assert isinstance(results, dict)

    assert "direct_costs" in results
    assert "fixed_capital" in results
    assert "fixed_opex" in results
    assert "variable_opex" in results
    assert "levelized_cost" in results
    assert "cash_flow" in results
    assert "tornado" in results
    assert "sensitivity" in results
    assert "monte_carlo" in results

    assert "interest_rate_case" in results["sensitivity"]
    assert "metrics" in results["monte_carlo"]
    assert "LCOP" in results["monte_carlo"]["metrics"]
    assert "inputs" in results["monte_carlo"]

    assert (output_dir / "Test Plant_equipment_results.json").exists()
    assert (output_dir / "Test Plant_plant_results.json").exists()
    assert (output_dir / "Test Plant_analysis_results.json").exists()

    plot_format = "png"
    for stem in (
        "direct_costs",
        "fixed_capital",
        "fixed_opex",
        "variable_opex",
        "levelized_cost",
        "cash_flow",
        "tornado",
        "sensitivity_interest_rate_case",
        "monte_carlo_lcop",
        # Every item's consumption/production is always reported (as a
        # constant when no *_uncertainty is configured), so "process" is
        # produced here too even though the fixture only sets price/rate
        # uncertainty.
        "monte_carlo_inputs_process",
        "monte_carlo_inputs_economic",
    ):
        assert (
            output_dir / f"Test Plant_{stem}.{plot_format}"
        ).exists()


def test_run_tea_minimal_workflow(tmp_path):
    output_dir = tmp_path / "results"

    equipment_path = tmp_path / "equipment.json"
    plant_path = tmp_path / "plant.json"
    analysis_path = tmp_path / "analysis.json"

    equipment_path.write_text(
        json.dumps(_equipment_data()), encoding="utf-8"
    )
    plant_path.write_text(json.dumps(_plant_data()), encoding="utf-8")
    analysis_path.write_text(
        json.dumps(_analysis_data()), encoding="utf-8"
    )

    results = run_tea(
        equipment_input_path=equipment_path,
        plant_input_path=plant_path,
        analysis_input_path=analysis_path,
        output_dir=output_dir,
    )

    _assert_results(results, output_dir)


def test_run_openpytea_minimal_workflow(tmp_path):
    output_dir = tmp_path / "results"

    config = {
        **_equipment_data(),
        **_plant_data(),
        **_analysis_data(),
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    results = run_openpytea(
        config_path=config_path,
        output_dir=output_dir,
    )

    _assert_results(results, output_dir)


def test_run_openpytea_uses_output_directory_from_config(tmp_path):
    output_dir = tmp_path / "config_results"

    config = {
        **_equipment_data(),
        **_plant_data(),
        **_analysis_data(output_dir=output_dir),
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    results = run_openpytea(config_path=config_path, output_dir=None)

    _assert_results(results, output_dir)


def test_plant_name_with_reserved_characters(tmp_path):
    """A path-hostile plant name must not lose the run at save time."""
    output_dir = tmp_path / "results"

    config = {**_equipment_data(), **_plant_data()}
    config["plant"]["plant_name"] = 'CO2/MeOH: "Case A"'
    config["analysis"] = {"levelized_cost": {"run": True}}
    config["output"] = {"save_json": True, "save_plots": False}

    config_path = tmp_path / "openpytea.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    results = run_openpytea(config_path, output_dir=output_dir)
    assert "levelized_cost" in results

    expected = output_dir / "CO2_MeOH_ _Case A__analysis_results.json"
    assert expected.exists()
    # the display name is untouched inside the results
    saved = json.loads(expected.read_text(encoding="utf-8"))
    assert saved["results"]["levelized_cost"]["xlabels"] == [
        'CO2/MeOH: "Case A"'
    ]
