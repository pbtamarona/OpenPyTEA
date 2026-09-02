import json
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version

import matplotlib.pyplot as plt

from openpytea.equipment import Equipment
from openpytea.plant import Plant
from openpytea.analysis import (
    direct_costs_data,
    fixed_capital_data,
    fixed_opex_data,
    variable_opex_data,
    levelized_cost_data,
    cash_flow_data,
    sensitivity_data,
    tornado_data,
    monte_carlo
    )
from openpytea.plotting import (
    plot_stacked_bar,
    plot_cash_flow,
    plot_sensitivity,
    plot_tornado,
    plot_monte_carlo,
    plot_monte_carlo_inputs,
)
from openpytea.helpers import (
    _to_jsonable,
    _read_json
)


__version__ = version("openpytea")


def _safe_filename(name):
    """
    Return ``name`` with filesystem-reserved characters replaced by
    ``_``, for use as an output-file stem.

    Covers ``/ \\ : * ? " < > |`` (reserved on Windows; ``/`` everywhere)
    and control characters. Only filenames are sanitized -- the plant's
    display name stays untouched in titles, reports, and JSON contents.
    """
    cleaned = "".join(
        "_" if (c in '/\\:*?"<>|' or ord(c) < 32) else c
        for c in str(name)
    ).strip()
    return cleaned or "plant"


def load_equipment_config(filepath):
    """
    Load equipment configuration from a JSON file.
    Parses a JSON file containing equipment specifications and returns a list
    of Equipment objects. The JSON file must have a top-level 'equipment' key
    containing a list of equipment entries.
    Args:
        filepath (str): Path to the JSON file containing equipment data.
    Returns:
        list[Equipment]: A list of Equipment objects constructed from the JSON
        data.
    Raises:
        ValueError: If the JSON file is missing the 'equipment' key.
        ValueError: If 'equipment' is not a list.
        ValueError: If any equipment entry is not a dictionary.
        ValueError: If any equipment entry is missing required keys
            ('name', 'process_type', 'category').
        ValueError: If any equipment entry defines neither 'param' nor
        'purchased_cost'.
    Notes:
        - Required keys per equipment entry: 'name', 'process_type', 'category'
        - Each entry must specify either 'param' or 'purchased_cost'
        - Default values: material=resolved cost correlation's default
          material (falls back to 'Carbon steel'), target_year=2024
        - Optional keys: 'type', 'material', 'num_units', 'purchased_cost',
          'cost_year', 'cost_func', 'target_year'
    """
    data = _read_json(filepath)
    return _build_equipment_list(data)


def _build_equipment_list(data):
    """
    Build a list of Equipment objects from an already-parsed config dict.

    Shared by :func:`load_equipment_config` (single-file workflow) and
    :func:`load_openpytea_config` (combined-file workflow) so both paths
    apply identical validation.
    """
    if "equipment" not in data:
        raise ValueError(
            "JSON file must contain a top-level 'equipment' key."
        )

    if not isinstance(data["equipment"], list):
        raise ValueError(
            "'equipment' must be a list of equipment entries."
        )

    equipment_list = []

    for i, entry in enumerate(data["equipment"], start=1):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Equipment entry #{i} must be a dictionary."
            )

        required = ["name", "process_type", "category"]
        missing = [k for k in required if k not in entry]
        if missing:
            raise ValueError(
                f"Equipment entry #{i} is missing required keys: {missing}"
            )

        if "purchased_cost" not in entry and "param" not in entry:
            raise ValueError(
                f"Equipment entry #{i} must define either 'param' "
                f"or 'purchased_cost'."
            )

        eq = Equipment(
            name=entry["name"],
            param=entry.get("param", 0.0),
            process_type=entry["process_type"],
            category=entry["category"],
            type=entry.get("type"),
            material=entry.get("material"),
            num_units=entry.get("num_units"),
            purchased_cost=entry.get("purchased_cost"),
            cost_year=entry.get("cost_year"),
            cost_func=entry.get("cost_func"),
            target_year=entry.get("target_year", 2024),
        )

        equipment_list.append(eq)

    return equipment_list


def load_plant_config(filepath, equipment_list):
    """
    Load a plant configuration from a JSON file and create a Plant instance.
    This function reads a JSON configuration file, validates its structure,
    and combines it with a provided equipment list to instantiate a Plant
    object.
    Args:
        filepath (str): Path to the JSON configuration file containing plant
        data.
            The file must contain a top-level 'plant' key with the plant
            configuration.
        equipment_list (list): A list of equipment objects to be associated
        with the plant.
    Returns:
        Plant: A Plant instance initialized with the configuration data and
        equipment list.
    Raises:
        ValueError: If the JSON file does not contain a top-level 'plant' key.
        FileNotFoundError: If the specified filepath does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    Example:
        >>> equipment = [Equipment(...), Equipment(...)]
        >>> plant = load_plant_config('plant_config.json', equipment)
    """
    data = _read_json(filepath)
    return _build_plant(data, equipment_list)


def _build_plant(data, equipment_list):
    """
    Build a Plant object from an already-parsed config dict.

    Shared by :func:`load_plant_config` (single-file workflow) and
    :func:`load_openpytea_config` (combined-file workflow) so both paths
    apply identical validation.
    """
    if "plant" not in data:
        raise ValueError("JSON file must contain a top-level 'plant' key.")

    config = data["plant"]
    config["equipment"] = equipment_list

    return Plant(config)


def load_analysis_config(filepath):
    """
    Load analysis configuration from a JSON file.
    This function reads a JSON file from the specified filepath and validates
    that it contains the required 'analysis' key.
    Args:
        filepath (str): The path to the JSON configuration file to load.
    Returns:
        dict: The parsed JSON data containing the analysis configuration.
    Raises:
        ValueError: If the JSON file does not contain an 'analysis' key.
        FileNotFoundError: If the specified filepath does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    Example:
        >>> config = load_analysis_config('analysis.json')
        >>> print(config['analysis'])
    """
    data = _read_json(filepath)

    if "analysis" not in data:
        raise ValueError("analysis.json must contain 'analysis' key")

    return data


def load_openpytea_config(filepath):
    """
    Load a single, combined OpenPyTEA configuration file.

    This is the CLI-facing counterpart to :func:`load_equipment_config`,
    :func:`load_plant_config`, and :func:`load_analysis_config`: instead of
    three separate files, one JSON file carries the top-level 'equipment',
    'plant', and 'analysis' keys (plus the optional 'output' key normally
    nested under 'analysis.json'). See :func:`run_openpytea`.

    Parameters
    ----------
    filepath : str or Path
        Path to the combined JSON configuration file.

    Returns
    -------
    dict
        The parsed JSON data, containing at least 'equipment', 'plant',
        and 'analysis' keys.

    Raises
    ------
    ValueError
        If the file is missing any of the 'equipment', 'plant', or
        'analysis' top-level keys.
    FileNotFoundError
        If the specified filepath does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.

    Examples
    --------
    >>> config = load_openpytea_config('config.json')
    >>> print(config['plant']['plant_name'])
    """
    data = _read_json(filepath)

    required = ["equipment", "plant", "analysis"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            "Combined config file must contain top-level "
            f"key(s): {missing}."
        )

    return data


def load_results(filepath):
    """
    Load results from a JSON file.
    Args:
        filepath (str): The path to the JSON file containing results.
    Returns:
        list or dict: The results data extracted from the JSON file's
        'results' key.
    Raises:
        ValueError: If the JSON file does not contain a 'results' key.
    Examples:
        >>> results = load_results('results.json')
        >>> print(results)
    """
    data = _read_json(filepath)

    if "results" not in data:
        raise ValueError("Results JSON must contain 'results' key")

    return data["results"]


def export_equipment_strings(equipment_list, filepath):
    """
    Export a list of equipment objects to a text file.
    Each equipment object is converted to a string representation and written
    to a separate line in the output file.
    Args:
        equipment_list (list): A list of equipment objects to export.
        filepath (str or Path): The file path where the equipment strings will
                                be written. Can be a string or a Path object.
    Returns:
        None
    Raises:
        IOError: If the file cannot be opened or written to.
        TypeError: If equipment_list is not iterable.
    Example:
        >>> equipment_list = [Equipment("pump"), Equipment("motor")]
        >>> export_equipment_strings(equipment_list, "equipment.txt")
    """
    filepath = Path(filepath)

    # Ensure directory exists, like the sibling exporters
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open("w", encoding="utf-8") as f:
        for eq in equipment_list:
            f.write(str(eq) + "\n")


def export_equipment_results(equipment_list, filepath):
    """
    Export equipment results to a JSON file.
    Converts a list of equipment objects to a dictionary format and writes them
    to a JSON file along with metadata and cost totals.
    Parameters
    ----------
    equipment_list : list
        A list of equipment objects that have a `to_dict()` method for
        serialization.
    filepath : str or Path
        The file path where the JSON output will be written. Can be a string
        or a `pathlib.Path` object.
    Returns
    -------
    None
    Notes
    -----
    The output JSON file contains:
    - metadata: Information about the export including OpenPyTEA version,
      generation timestamp (UTC), and number of equipment items.
    - equipment: List of equipment dictionaries.
    - totals: Aggregated cost totals including purchased cost and direct cost.
    The JSON file is written with 4-space indentation for readability.
    Examples
    --------
    >>> equipment_list = [eq1, eq2, eq3]
    >>> export_equipment_results(equipment_list, "equipment_export.json")
    """
    filepath = Path(filepath)

    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    equipment_data = [eq.to_dict() for eq in equipment_list]

    total_purchased = sum((eq.get("purchased_cost") or 0.0)
                          for eq in equipment_data)
    total_direct = sum((eq.get("direct_cost") or 0.0)
                       for eq in equipment_data)

    output = {
        "metadata": {
            "generated_by": f"OpenPyTEA Version {__version__}",
            "date_generated": datetime.now(timezone.utc).isoformat(),
            "n_equipment": len(equipment_data),
        },
        "equipment": equipment_data,
        "totals": {
            "total_purchased_cost": total_purchased,
            "total_direct_cost": total_direct,
        },
    }

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)


def export_plant_results(plant, filepath):
    """
    Export plant results to a JSON file.
    Serializes the plant object and metadata to a JSON file at the specified
    filepath. The output includes version information and the timestamp of
    when the export was generated.
    Parameters
    ----------
    plant : Plant
        The plant object containing results to be exported.
    filepath : str or Path
        The destination file path where the JSON file will be written.
        Can be a string or pathlib.Path object.
    Returns
    -------
    None
    Notes
    -----
    The exported JSON file contains:
    - metadata: Generated by information and UTC timestamp
    - plant data: All plant object attributes via to_dict() method
    The file is created with UTF-8 encoding and indented JSON formatting
    (4 spaces).
    Examples
    --------
    >>> from openpytea.io import export_plant_results
    >>> plant = Plant(...)
    >>> export_plant_results(plant, "output/plant_results.json")
    """
    filepath = Path(filepath)

    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "generated_by": f"OpenPyTEA Version {__version__}",
            "date_generated": datetime.now(timezone.utc).isoformat(),
        },
        **plant.to_dict(),
    }

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)


def run_equipment(input_path, output_path):
    """
    Load equipment configuration from file and export results to a specified
    output path.

    This function reads equipment configuration data from an input file,
    processes it, and writes the results to an output file.

    Parameters
    ----------
    input_path : str
        Path to the input file containing equipment configuration data.
    output_path : str
        Path where the equipment results will be exported.

    Returns
    -------
    list
        A list of equipment objects loaded from the input configuration file.

    Examples
    --------
    >>> equipment_list = run_equipment('config/equipment.json',
    'output/results.json')
    >>> print(len(equipment_list))
    """
    equipment_list = load_equipment_config(input_path)
    export_equipment_results(equipment_list, output_path)
    return equipment_list


def run_plant(plant_input_path, plant_output_path,
              equipment_input_path=None, equipment_list=None):
    """
    Load a plant configuration, calculate all plant metrics, and export
    results. This function orchestrates the complete workflow for plant
    analysis by loading the plant configuration, computing all relevant
    calculations, and saving the results to a specified output path.
    Parameters
    ----------
    plant_input_path : str
        File path to the plant configuration input file.
    plant_output_path : str
        File path where the plant results will be exported.
    equipment_input_path : str, optional
        File path to the equipment configuration input file. Required if
        equipment_list is not provided. Default is None.
    equipment_list : list, optional
        List of equipment configurations. If not provided, will be loaded from
        equipment_input_path. Default is None.
    Returns
    -------
    Plant
        The calculated plant object containing all computed metrics
        and results.
    Raises
    ------
    ValueError
        If neither equipment_input_path nor equipment_list is provided.
    Examples
    --------
    >>> plant = run_plant('plant_config.yaml', 'results.csv',
    ...                   equipment_input_path='equipment_config.yaml')
    """
    if equipment_list is None:
        if equipment_input_path is None:
            raise ValueError(
                "Either equipment_input_path or "
                "equipment_list must be provided."
                )
        equipment_list = load_equipment_config(equipment_input_path)

    plant = load_plant_config(plant_input_path, equipment_list)
    plant.calculate_all()
    export_plant_results(plant, plant_output_path)
    return plant


def run_tea(equipment_input_path, plant_input_path, analysis_input_path,
            output_dir="results"):
    """
    Execute a complete Techno-Economic Analysis (TEA) workflow.
    This function orchestrates the entire TEA pipeline by loading
    configuration files, performing calculations, running specified analyses,
    and exporting results as JSON files and/or plots.
    Parameters
    ----------
    equipment_input_path : str or Path
        Path to the equipment configuration file.
    plant_input_path : str or Path
        Path to the plant configuration file.
    analysis_input_path : str or Path
        Path to the analysis configuration file specifying which analyses to
        run and their parameters.
    output_dir : str or Path, optional
        Directory where results will be saved. If None, uses the directory
        specified in the analysis configuration file. Defaults to "results".
    Returns
    -------
    dict
        Dictionary containing analysis results with keys corresponding to the
        analyses that were run:
        - "direct_costs": Direct cost breakdown by equipment
        - "fixed_capital": Fixed capital cost breakdown
        - "fixed_opex": Fixed operating expenditure breakdown
        - "variable_opex": Variable operating expenditure breakdown
        - "levelized_cost": Levelized cost of production (LCOP) breakdown
        - "cash_flow": Cumulative cash flow diagram data
        - "tornado": Tornado/sensitivity analysis results
        - "monte_carlo": Monte Carlo simulation results with metrics
          and sampled input distributions
        - "sensitivity": Dictionary of sensitivity analysis cases
    Raises
    ------
    ValueError
        If a requested Monte Carlo metric is not found in the results.
    Notes
    -----
    - Creates output directory if it does not exist when saving results
    - Automatically clears figures after saving to free memory
    - JSON exports include equipment results, plant results, and
        analysis results with metadata
    - Plot format and resolution (DPI) are configurable via the
        analysis configuration file
    """
    # --- Load inputs ---
    equipment_list = load_equipment_config(equipment_input_path)
    analysis_cfg = load_analysis_config(analysis_input_path)
    plant = load_plant_config(plant_input_path, equipment_list)

    return _run_analyses(equipment_list, plant, analysis_cfg, output_dir)


def run_openpytea(config_path, output_dir="results"):
    """
    Execute a complete Techno-Economic Analysis (TEA) workflow from a
    single, combined JSON configuration file.

    This is the single-file counterpart to :func:`run_tea`, intended for
    CLI use: instead of three separate input files, one JSON file carries
    the 'equipment', 'plant', 'analysis', and (optional) 'output' blocks
    that would otherwise live in ``equipment.json``, ``plant.json``, and
    ``analysis.json``. See :func:`load_openpytea_config` for the expected
    file structure. Aside from the input loading step, the pipeline
    (calculations, JSON exports, and optional plots) is identical to
    :func:`run_tea`.

    Parameters
    ----------
    config_path : str or Path
        Path to the combined configuration file, with top-level
        'equipment', 'plant', and 'analysis' keys (and optionally
        'output').
    output_dir : str or Path, optional
        Directory where results will be saved. If None, uses the
        directory specified under the config's 'output' key. Defaults to
        "results".

    Returns
    -------
    dict
        Same structure as returned by :func:`run_tea`.

    Examples
    --------
    >>> results = run_openpytea('config.json', output_dir='outputs')
    >>> results["monte_carlo"]["metrics"]["LCOP"]
    """
    config = load_openpytea_config(config_path)

    equipment_list = _build_equipment_list(config)
    plant = _build_plant(config, equipment_list)

    return _run_analyses(equipment_list, plant, config, output_dir)


def _run_analyses(equipment_list, plant, analysis_cfg, output_dir):
    """
    Run the configured analyses for a loaded plant/equipment pair and
    export results (and optionally plots) to ``output_dir``.

    This holds the shared pipeline behind :func:`run_tea` and
    :func:`run_openpytea`, which differ only in how they load their
    inputs.

    Parameters
    ----------
    equipment_list : list[Equipment]
        Equipment objects associated with the plant.
    plant : Plant
        A loaded (but not yet calculated) Plant object.
    analysis_cfg : dict
        Parsed analysis configuration, containing 'analysis' and
        (optionally) 'output' keys.
    output_dir : str, Path, or None
        Directory where results will be saved. If None, uses the
        directory specified in ``analysis_cfg['output']``.

    Returns
    -------
    dict
        Same structure as returned by :func:`run_tea`.
    """
    analysis_block = analysis_cfg.get("analysis", {})
    output_cfg = analysis_cfg.get("output", {})

    if output_dir is None:
        output_dir = output_cfg.get("directory", "results")
    output_dir = Path(output_dir)

    # Filename-safe form of the display name: characters like "/" or ":"
    # in a plant name (e.g. "CO2/MeOH Plant") would otherwise make every
    # save below fail -- after the full computation already ran
    fname = _safe_filename(plant.name)

    save_json = output_cfg.get("save_json", True)
    save_plots = output_cfg.get("save_plots", False)
    plot_format = output_cfg.get("plot_format", "png")
    dpi = output_cfg.get("dpi", 300)

    if save_json or save_plots:
        output_dir.mkdir(parents=True, exist_ok=True)

    plant.calculate_all()

    results = {}

    analysis_map = {
        "direct_costs": direct_costs_data,
        "fixed_capital": fixed_capital_data,
        "fixed_opex": fixed_opex_data,
        "variable_opex": variable_opex_data,
        "levelized_cost": levelized_cost_data,
        "cash_flow": cash_flow_data,
        "tornado": tornado_data,
        "monte_carlo": monte_carlo,
    }

    for key, func in analysis_map.items():
        cfg = analysis_block.get(key, {})
        if cfg.get("run"):
            args = cfg.get("args", {})
            results[key] = func(plant, **args)

    sens_cfg = analysis_block.get("sensitivity", {})
    if sens_cfg.get("run", False):
        results["sensitivity"] = {}
        for i, case in enumerate(sens_cfg.get("cases", []), start=1):
            name = case.get("name", f"case_{i}")
            args = case.get("args", {})
            results["sensitivity"][name] = sensitivity_data(plant, **args)

    if save_json:
        export_equipment_results(
            equipment_list,
            output_dir / f"{fname}_equipment_results.json",
        )
        export_plant_results(
            plant,
            output_dir / f"{fname}_plant_results.json",
        )

        analysis_output = {
            "metadata": {
                "generated_by": f"OpenPyTEA Version {__version__}",
                "date_generated": datetime.now(timezone.utc).isoformat(),
            },
            "results": _to_jsonable(results),
        }

        results_file = output_dir / f"{fname}_analysis_results.json"
        with results_file.open("w", encoding="utf-8") as f:
            json.dump(analysis_output, f, indent=4)

    # ======================================================
    # EXPORT PLOTS
    # ======================================================
    if save_plots:
        if "direct_costs" in results:
            fig, ax = plot_stacked_bar(
                results["direct_costs"], show=False
            )
            fig.savefig(
                output_dir / f"{fname}_direct_costs.{plot_format}",
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(fig)  # Free the figure

        if "fixed_capital" in results:
            fig, ax = plot_stacked_bar(
                results["fixed_capital"], show=False
            )
            fig.savefig(
                output_dir / f"{fname}_fixed_capital.{plot_format}",
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(fig)  # Free the figure

        if "fixed_opex" in results:
            fig, ax = plot_stacked_bar(
                results["fixed_opex"], show=False
            )
            fig.savefig(
                output_dir / f"{fname}_fixed_opex.{plot_format}",
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(fig)  # Free the figure

        if "variable_opex" in results:
            fig, ax = plot_stacked_bar(
                results["variable_opex"], show=False
            )
            fig.savefig(
                output_dir / f"{fname}_variable_opex.{plot_format}",
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(fig)  # Free the figure

        if "levelized_cost" in results:
            fig, ax = plot_stacked_bar(
                results["levelized_cost"], show=False
            )
            fig.savefig(
                output_dir / f"{fname}_levelized_cost.{plot_format}",
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(fig)  # Free the figure

        if "cash_flow" in results:
            fig, ax = plot_cash_flow(
                results["cash_flow"], show=False
            )
            fig.savefig(
                output_dir / f"{fname}_cash_flow.{plot_format}",
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(fig)  # Free the figure

        if "sensitivity" in results:
            for name, data in results["sensitivity"].items():
                fig, ax = plot_sensitivity(data, show=False)
                fig.savefig(
                    output_dir /
                    f"{fname}_sensitivity_{name}.{plot_format}",
                    dpi=dpi,
                    bbox_inches="tight",
                )
                plt.close(fig)  # Free the figure

        if "tornado" in results:
            fig, ax = plot_tornado(
                results["tornado"], show=False
            )
            fig.savefig(
                output_dir / f"{fname}_tornado.{plot_format}",
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(fig)  # Free the figure

        if "monte_carlo" in results:
            mc_results = results["monte_carlo"]
            mc_metrics = mc_results["metrics"]
            mc_cfg = analysis_block.get("monte_carlo", {})
            requested_metrics = mc_cfg.get("metric")

            if requested_metrics is None:
                metrics_to_plot = []
            elif isinstance(requested_metrics, str):
                metrics_to_plot = [requested_metrics]
            else:
                metrics_to_plot = list(requested_metrics)

            for metric_name in metrics_to_plot:
                # metrics keys are always uppercase (LCOP/NPV/ROI/PBT)
                # and plot_monte_carlo uppercases its argument, so a
                # lowercase config entry must not abort the run
                metric_name = metric_name.upper()
                if metric_name not in mc_metrics:
                    available = ", ".join(mc_metrics.keys())
                    raise ValueError(
                        f"Monte Carlo metric '{metric_name}' not found. "
                        f"Available metrics: {available}"
                    )

                fig, ax = plot_monte_carlo(
                    mc_results,
                    metric=metric_name,
                    show=False,
                )

                filename = (
                    f"{fname}_monte_carlo_"
                    f"{metric_name.lower()}.{plot_format}"
                )
                fig.savefig(
                    output_dir / filename,
                    dpi=dpi,
                    bbox_inches="tight",
                )

                plt.close(fig)  # Free the figure

            if mc_cfg.get("plot_inputs", False):
                fig_process, _, fig_economic, _ = plot_monte_carlo_inputs(
                    mc_results, show=False
                )
                for group_name, fig in (
                    ("process", fig_process),
                    ("economic", fig_economic),
                ):
                    if fig is None:
                        continue  # no inputs sampled for this group
                    fig.savefig(
                        output_dir /
                        f"{fname}_monte_carlo_inputs_"
                        f"{group_name}.{plot_format}",
                        dpi=dpi,
                        bbox_inches="tight",
                    )
                    plt.close(fig)  # Free the figure

    return results
