import asyncio
from copy import deepcopy

import pytest

from openpytea import mcp_server as m
from openpytea.io import _build_equipment_list
from openpytea.plant import Plant


@pytest.fixture(autouse=True)
def empty_session():
    m._SESSION.clear()
    yield
    m._SESSION.clear()


def test_examples_list_and_load():
    ids = [e["id"] for e in m.list_examples()]
    assert "h2_electrolysis" in ids and "h2_smr" in ids
    for example_id in ids:
        example = m.get_example(example_id)
        assert example["equipment"] and example["plant"]["plant_name"]


def test_evaluate_matches_library():
    example = m.get_example("h2_smr")
    out = m.evaluate_plant(example["equipment"], example["plant"])

    config = deepcopy(example["plant"])
    config["equipment"] = _build_equipment_list(
        {"equipment": deepcopy(example["equipment"])}
    )
    plant = Plant(config)
    plant.calculate_all()

    assert out["metrics"]["levelized_cost"] == pytest.approx(
        plant.levelized_cost
    )
    assert out["capital_costs"]["fixed_capital"] == pytest.approx(
        plant.fixed_capital
    )
    assert out["plant_name"] in m._SESSION


def test_evaluate_accepts_file_layout():
    example = m.get_example("h2_electrolysis")
    bare = m.evaluate_plant(example["equipment"], example["plant"])
    wrapped = m.evaluate_plant(
        {"equipment": example["equipment"]}, {"plant": example["plant"]}
    )
    assert wrapped["metrics"] == bare["metrics"]
    assert "Replaced" in wrapped["notes"][-1]


def test_get_plant_config_round_trip():
    first = m.load_example("h2_electrolysis")
    config = m.get_plant_config(first["plant_name"])
    config["plant"]["variable_opex_inputs"]["electricity"]["price"] = 0.10
    second = m.evaluate_plant(config["equipment"], config["plant"])
    assert (second["metrics"]["levelized_cost"]
            > first["metrics"]["levelized_cost"])


def test_bad_inputs_raise_value_error():
    with pytest.raises(ValueError, match="non-empty list"):
        m.cost_equipment([])
    with pytest.raises(ValueError, match="missing required keys"):
        m.cost_equipment([{"name": "P1"}])
    with pytest.raises(ValueError, match="No evaluated plant"):
        m.run_tornado("Nope")
    with pytest.raises(ValueError, match="Unknown example"):
        m.get_example("nope")


def test_cost_equipment_totals():
    out = m.cost_equipment([
        {"name": "P1", "process_type": "Fluids", "category": "Pumps",
         "type": "Single-stage centrifugal", "param": 15},
        {"name": "Known", "process_type": "Fluids", "category": "Reactors",
         "purchased_cost": 100000, "cost_year": 2024},
    ])
    items = out["equipment"]
    assert items[0]["purchased_cost"] > 0
    assert out["totals"]["total_direct_cost"] == pytest.approx(
        sum(i["direct_cost"] for i in items)
    )


def test_search_cost_correlations():
    out = m.search_cost_correlations("centrifugal", category="Pumps")
    assert out["total_matches"] > 0
    assert all(c["category"] == "Pumps" for c in out["correlations"])
    assert "Carbon steel" in out["materials"]
    with pytest.raises(ValueError, match="Unknown category"):
        m.search_cost_correlations(category="Spaceships")


def test_analyses():
    name = m.load_example("h2_electrolysis")["plant_name"]

    params = m.list_sensitivity_parameters(name)
    assert "fixed_capital" in params["economic"]

    sens = m.run_sensitivity(name, "electricity", 0.5, n_points=5)
    assert len(sens["points"]) == 5
    assert sens["points"][2]["metric_value"] == pytest.approx(
        sens["baseline"]
    )
    assert sens["points"][0]["parameter_value"] == pytest.approx(0.025)
    assert "$" not in sens["metric_label"]

    tornado = m.run_tornado(name, 0.2, category="both")
    swings = [b["swing"] for b in tornado["bars"]]
    assert swings == sorted(swings, reverse=True)

    mc = m.run_monte_carlo(name, num_samples=500, random_seed=1)
    assert mc["metrics"]["LCOP"]["p5"] < mc["metrics"]["LCOP"]["p95"]
    again = m.run_monte_carlo(name, num_samples=500, random_seed=1)
    assert again["metrics"] == mc["metrics"]


@pytest.mark.parametrize("label, plain", [
    (r"$\mathrm{CO_{2}}$ price", "CO2 price"),
    (r"LCOP / [USD$\cdot$kg$^{-1}$]", "LCOP / [USD·kg⁻¹]"),
    (r"NPV / [\$]", "NPV / [$]"),
    ("Fixed CAPEX", "Fixed CAPEX"),
])
def test_plain_labels(label, plain):
    assert m._plain(label) == plain


def test_server_tools():
    pytest.importorskip("mcp.server.mcpserver")
    from mcp import Client

    async def run():
        async with Client(m.build_server()) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert {"evaluate_plant", "run_tornado",
                    "run_monte_carlo"} <= names

            res = await client.call_tool(
                "load_example", {"example_id": "h2_smr"}
            )
            assert not res.is_error
            assert res.structured_content["metrics"]["levelized_cost"] > 0

            # library errors reach the model, not just "Error executing"
            res = await client.call_tool("run_tornado", {"plant_name": "x"})
            assert res.is_error
            assert "No evaluated plant" in res.content[0].text

    asyncio.run(run())


PNG = b"\x89PNG\r\n\x1a\n"


def test_plots(tmp_path):
    wn = m.load_example("h2_electrolysis")["plant_name"]
    smr = m.load_example("h2_smr")["plant_name"]

    for breakdown in m._BREAKDOWNS:
        out = m.plot_costs([wn, smr], breakdown, dpi=60)
        assert len(out["images"]) == 1 and out["images"][0].startswith(PNG)

    assert m.plot_cash_flow([wn, smr], dpi=60)["images"][0].startswith(PNG)

    out = m.plot_sensitivity([wn, smr], "fixed_capital", x_axis="value",
                             n_points=5, dpi=60)
    assert set(out["baselines"]) == {wn, smr}

    out = m.plot_tornado(smr, category="both", dpi=60,
                         save_to=str(tmp_path / "sub" / "tornado.pdf"))
    assert (tmp_path / "sub" / "tornado.pdf").exists()
    assert out["saved_to"] == [str((tmp_path / "sub" / "tornado.pdf"))]

    with pytest.raises(ValueError, match="save_to must end"):
        m.plot_tornado(smr, save_to=str(tmp_path / "tornado.txt"))


def test_monte_carlo_plots_reuse_last_run(tmp_path):
    name = m.load_example("h2_smr")["plant_name"]
    run = m.run_monte_carlo(name, num_samples=500, random_seed=2)

    out = m.plot_monte_carlo([name], dpi=60)
    assert out["num_samples"][name] == 500
    assert out["stats"][name] == run["metrics"]["LCOP"]

    out = m.plot_monte_carlo([name], "NPV", num_samples=300, dpi=60)
    assert out["num_samples"][name] == 300

    out = m.plot_monte_carlo_inputs(
        name, save_to=str(tmp_path / "inputs.png"), dpi=60
    )
    assert len(out["images"]) == 2
    assert {p.split("_")[-1] for p in out["saved_to"]} == {
        "process.png", "economic.png"
    }

    with pytest.raises(ValueError, match="not IRR"):
        m.plot_monte_carlo([name], "IRR")


def test_monte_carlo_plot_needs_prices():
    example = m.get_example("h2_electrolysis")
    for product in example["plant"]["plant_products"].values():
        product.pop("price", None)
    name = m.evaluate_plant(example["equipment"],
                            example["plant"])["plant_name"]
    with pytest.raises(ValueError, match="needs a price"):
        m.plot_monte_carlo([name], "NPV")


def test_server_plot_tool_returns_image():
    pytest.importorskip("mcp.server.mcpserver")
    from mcp import Client

    async def run():
        async with Client(m.build_server()) as client:
            await client.call_tool("load_example", {"example_id": "h2_smr"})
            res = await client.call_tool(
                "plot_costs",
                {"plant_names": ["Steam Reforming"], "dpi": 60},
            )
            assert not res.is_error
            kinds = [block.type for block in res.content]
            assert kinds == ["image", "text"]
            assert res.content[0].mime_type == "image/png"

    asyncio.run(run())
