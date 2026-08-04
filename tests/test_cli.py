import json

import pytest

from openpytea.cli import main
from test_run_tea import _equipment_data, _plant_data, _analysis_data


def test_cli_run_minimal_workflow(tmp_path, capsys):
    output_dir = tmp_path / "results"

    config = {
        **_equipment_data(),
        **_plant_data(),
        **_analysis_data(),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    exit_code = main(["run", str(config_path), "-o", str(output_dir)])

    assert exit_code == 0
    assert (output_dir / "Test Plant_analysis_results.json").exists()

    captured = capsys.readouterr()
    assert "Ran:" in captured.out
    assert "monte_carlo" in captured.out


def test_cli_tea_minimal_workflow(tmp_path, capsys):
    output_dir = tmp_path / "results"

    equipment_path = tmp_path / "equipment.json"
    plant_path = tmp_path / "plant.json"
    analysis_path = tmp_path / "analysis.json"

    equipment_path.write_text(json.dumps(_equipment_data()), encoding="utf-8")
    plant_path.write_text(json.dumps(_plant_data()), encoding="utf-8")
    analysis_path.write_text(json.dumps(_analysis_data()), encoding="utf-8")

    exit_code = main([
        "tea",
        "--equipment", str(equipment_path),
        "--plant", str(plant_path),
        "--analysis", str(analysis_path),
        "-o", str(output_dir),
    ])

    assert exit_code == 0
    assert (output_dir / "Test Plant_analysis_results.json").exists()

    captured = capsys.readouterr()
    assert "Ran:" in captured.out


def test_cli_equipment(tmp_path, capsys):
    equipment_path = tmp_path / "equipment.json"
    output_path = tmp_path / "equipment_results.json"
    equipment_path.write_text(json.dumps(_equipment_data()), encoding="utf-8")

    exit_code = main(["equipment", str(equipment_path), str(output_path)])

    assert exit_code == 0
    assert output_path.exists()

    captured = capsys.readouterr()
    assert "Wrote 2 equipment item(s)" in captured.out


def test_cli_plant(tmp_path, capsys):
    equipment_path = tmp_path / "equipment.json"
    plant_path = tmp_path / "plant.json"
    output_path = tmp_path / "plant_results.json"

    equipment_path.write_text(json.dumps(_equipment_data()), encoding="utf-8")
    plant_path.write_text(json.dumps(_plant_data()), encoding="utf-8")

    exit_code = main([
        "plant", str(plant_path), str(output_path),
        "--equipment", str(equipment_path),
    ])

    assert exit_code == 0
    assert output_path.exists()

    captured = capsys.readouterr()
    assert "Wrote plant 'Test Plant' results" in captured.out


def test_cli_missing_file_returns_error(tmp_path, capsys):
    exit_code = main(["run", str(tmp_path / "nonexistent.json")])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "OpenPyTEA" in captured.out


def test_cli_requires_a_command(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code != 0
