import matplotlib
import pytest
from openpytea import Equipment, Plant

matplotlib.use("Agg")


@pytest.fixture
def test_equipment():
    return [
        Equipment(
            name="Reactor",
            param=1.0,
            process_type="Fluids",
            category="Reactors",
            purchased_cost=100000,
            cost_year=2024,
        ),
        Equipment(
            name="Pump",
            param=1.0,
            process_type="Fluids",
            category="Pumps",
            purchased_cost=20000,
            cost_year=2024,
        ),
    ]


@pytest.fixture
def test_plant(test_equipment):
    config = {
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
        "operator_hourly_rate": {"rate": 25},
        "equipment": test_equipment,
        "variable_opex_inputs": {
            "electricity": {"consumption": 100, "price": 0.08},
            "water": {"consumption": 10, "price": 0.5},
        },
        "plant_products": {
            "hydrogen": {"production": 50, "price": 5.0}
        },
    }
    return Plant(config)
