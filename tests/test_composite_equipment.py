import pytest

from openpytea import CompositeEquipment, Equipment, Plant, direct_costs_data
from openpytea.equipment import inflation_adjustment


def _vessel(num_units=4):
    return Equipment(
        name="Vessel",
        param=1.0,
        process_type="Fluids",
        category="Pressure vessels",
        purchased_cost=10_000,
        cost_year=2024,
        num_units=num_units,
    )


def _adsorbent():
    return Equipment(
        name="Adsorbent",
        param=1.0,
        process_type="Solids",
        category="Packings & adsorbents",
        purchased_cost=5_000,
        cost_year=2024,
    )


@pytest.fixture
def psa():
    return CompositeEquipment(
        name="PSA",
        process_type="Fluids",
        components=[_vessel(), _adsorbent()],
        category="Adsorption",
        type="PSA",
    )


def test_composite_sums_component_costs(psa):
    vessel, adsorbent = _vessel(), _adsorbent()
    # a direct purchased_cost is the total for the vessel's 4 units
    assert vessel.purchased_cost == pytest.approx(10_000)
    assert psa.purchased_cost == pytest.approx(
        vessel.purchased_cost + adsorbent.purchased_cost
    )
    assert psa.components_purchased_cost == pytest.approx(psa.purchased_cost)
    # "component" rule: each part keeps its own installation factors
    assert psa.direct_cost == pytest.approx(
        vessel.direct_cost + adsorbent.direct_cost
    )
    assert psa.installation == "component"
    assert psa.num_units == 1
    assert psa.param is None


def test_composite_num_units_multiplies_cost():
    one = CompositeEquipment(
        name="PSA", process_type="Fluids",
        components=[_vessel(), _adsorbent()],
    )
    two = CompositeEquipment(
        name="PSA", process_type="Fluids",
        components=[_vessel(), _adsorbent()], num_units=2,
    )
    assert two.purchased_cost == pytest.approx(2 * one.purchased_cost)
    assert two.direct_cost == pytest.approx(2 * one.direct_cost)
    assert two.breakdown()["num_units"].tolist() == [8, 2]


def test_composite_installation_rule_applies_own_factors():
    vessel, adsorbent = _vessel(), _adsorbent()
    comp = CompositeEquipment(
        name="PSA",
        process_type="Fluids",
        components=[vessel, adsorbent],
        installation="composite",
    )
    purchased = vessel.purchased_cost + adsorbent.purchased_cost
    f = Equipment.process_factors["Fluids"]
    lang = (1 + f["fp"]) * 1.0 + (
        f["fer"] + f["fel"] + f["fi"] + f["fc"] + f["fs"] + f["fl"]
    )
    assert comp.purchased_cost == pytest.approx(purchased)
    assert comp.direct_cost == pytest.approx(purchased * lang)


def test_composite_quote_overrides_component_sum():
    comp = CompositeEquipment(
        name="PSA",
        process_type="Fluids",
        components=[_vessel(), _adsorbent()],
        purchased_cost=100_000,
        cost_year=2019,
        num_units=2,
    )
    # The composite quote is the total; num_units does not multiply it
    expected = inflation_adjustment(100_000, 2019, 2024)
    assert comp.purchased_cost == pytest.approx(expected)
    assert comp.components_purchased_cost == pytest.approx(2 * 15_000)
    # A quoted composite is installed as one item
    assert comp.direct_cost > comp.purchased_cost
    assert sum(comp.direct_cost_breakdown().values()) == pytest.approx(
        comp.direct_cost
    )


def test_composite_breakdown_and_direct_cost_breakdown(psa):
    df = psa.breakdown()
    assert list(df["component"]) == ["Vessel", "Adsorbent"]
    assert list(df["num_units"]) == [4, 1]
    assert list(df["purchased_each"]) == pytest.approx([2_500, 5_000])
    assert df["purchased_total"].sum() == pytest.approx(psa.purchased_cost)
    assert df["direct_total"].sum() == pytest.approx(psa.direct_cost)

    split = psa.direct_cost_breakdown()
    assert set(split) == {"PSA / Vessel", "PSA / Adsorbent"}
    assert sum(split.values()) == pytest.approx(psa.direct_cost)


def test_composite_nesting_flattens_num_units(psa):
    train = CompositeEquipment(
        name="Train",
        process_type="Fluids",
        components=[psa, _vessel(num_units=1)],
        num_units=2,
    )
    leaves = [(label, mult) for label, _, mult in train.leaves()]
    assert leaves == [
        ("PSA / Vessel", 2),
        ("PSA / Adsorbent", 2),
        ("Vessel", 2),
    ]
    assert train.breakdown()["num_units"].tolist() == [8, 2, 2]
    assert train.purchased_cost == pytest.approx(
        2 * (psa.purchased_cost + 10_000)
    )
    assert train.direct_cost == pytest.approx(
        2 * (psa.direct_cost + _vessel(num_units=1).direct_cost)
    )
    assert train.to_dict()["components"][0]["components"][0]["name"] == "Vessel"


def test_composite_to_dict_and_str(psa):
    d = psa.to_dict()
    assert d["name"] == "PSA"
    assert d["category"] == "Adsorption"
    assert d["installation"] == "component"
    assert len(d["components"]) == 2
    assert d["components"][0]["num_units"] == 4
    text = str(psa)
    assert "Vessel (x4)" in text
    assert "Direct Cost" in text


@pytest.mark.parametrize(
    "kwargs, err",
    [
        ({"process_type": "Gas"}, ValueError),
        ({"installation": "lumped"}, ValueError),
        ({"components": []}, ValueError),
        ({"num_units": 0}, ValueError),
        ({"components": ["not equipment"]}, TypeError),
        ({"target_year": 2020}, ValueError),
    ],
)
def test_composite_validation(kwargs, err):
    base = dict(
        name="PSA",
        process_type="Fluids",
        components=[_vessel()],
    )
    base.update(kwargs)
    with pytest.raises(err):
        CompositeEquipment(**base)


def test_composite_in_plant_and_direct_costs_data(psa):
    pump = Equipment(
        name="Pump",
        param=1.0,
        process_type="Fluids",
        category="Pumps",
        purchased_cost=20_000,
        cost_year=2024,
    )
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
        "equipment": [psa, pump],
        "variable_opex_inputs": {
            "electricity": {"consumption": 100, "price": 0.08},
        },
        "plant_products": {"hydrogen": {"production": 50, "price": 5.0}},
    }
    plant = Plant(config)
    plant.calculate_purchased_cost()

    assert plant.purchased_cost == pytest.approx(
        psa.purchased_cost + pump.purchased_cost
    )
    items = plant.to_dict()["equipment_summary"]["items"]
    assert [i["name"] for i in items] == ["PSA", "Pump"]

    lumped = direct_costs_data(plant)["components"][0]
    assert set(lumped) == {"PSA", "Pump"}

    expanded = direct_costs_data(plant, expand_composites=True)["components"][0]
    assert set(expanded) == {"PSA / Vessel", "PSA / Adsorbent", "Pump"}
    assert sum(expanded.values()) == pytest.approx(sum(lumped.values()))
