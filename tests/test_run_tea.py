import json
from openpytea import run_tea


def test_run_tea_minimal_workflow(tmp_path):
    equipment_data = {
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

    plant_data = {
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

    analysis_data = {
        "analysis": {
            "direct_costs": {"run": True},
            "fixed_capital": {"run": True},
            "fixed_opex": {"run": True},
            "variable_opex": {"run": True},
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
                "metric": ["LCOP"],
            },
        },
        "output": {
            "save_json": True,
            "save_plots": False,
        },
    }

    equipment_path = tmp_path / "equipment.json"
    plant_path = tmp_path / "plant.json"
    analysis_path = tmp_path / "analysis.json"
    output_dir = tmp_path / "results"

    equipment_path.write_text(json.dumps(equipment_data), encoding="utf-8")
    plant_path.write_text(json.dumps(plant_data), encoding="utf-8")
    analysis_path.write_text(json.dumps(analysis_data), encoding="utf-8")

    results = run_tea(
        equipment_input_path=equipment_path,
        plant_input_path=plant_path,
        analysis_input_path=analysis_path,
        output_dir=output_dir,
    )

    assert isinstance(results, dict)

    assert "direct_costs" in results
    assert "fixed_capital" in results
    assert "fixed_opex" in results
    assert "variable_opex" in results
    assert "tornado" in results
    assert "sensitivity" in results
    assert "monte_carlo" in results

    assert "interest_rate_case" in results["sensitivity"]
    assert "metrics" in results["monte_carlo"]
    assert "LCOP" in results["monte_carlo"]["metrics"]

    assert (output_dir / "Test Plant_equipment_results.json").exists()
    assert (output_dir / "Test Plant_plant_results.json").exists()
    assert (output_dir / "Test Plant_analysis_results.json").exists()
