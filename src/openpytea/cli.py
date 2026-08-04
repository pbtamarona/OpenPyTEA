"""Command-line interface for OpenPyTEA.

Thin argparse wrapper around the functions in :mod:`openpytea.io`. Installed
as the ``openpytea`` console command (see the ``[project.scripts]`` entry in
``pyproject.toml``).
"""

import argparse
import sys
from importlib.metadata import version

from openpytea.io import run_equipment, run_plant, run_tea, run_openpytea


def _print_run_summary(results, output_dir):
    ran = ", ".join(sorted(results)) or "(none)"
    dest = (
        output_dir
        if output_dir is not None
        else "the directory configured in the analysis file (default 'results')"
    )
    print(f"Ran: {ran}")
    print(f"Results written to: {dest}")


def _cmd_run(args):
    results = run_openpytea(
        config_path=args.config,
        output_dir=args.output_dir,
    )
    _print_run_summary(results, args.output_dir)


def _cmd_tea(args):
    results = run_tea(
        equipment_input_path=args.equipment,
        plant_input_path=args.plant,
        analysis_input_path=args.analysis,
        output_dir=args.output_dir,
    )
    _print_run_summary(results, args.output_dir)


def _cmd_equipment(args):
    equipment_list = run_equipment(
        input_path=args.input,
        output_path=args.output,
    )
    print(f"Wrote {len(equipment_list)} equipment item(s) to {args.output}")


def _cmd_plant(args):
    plant = run_plant(
        plant_input_path=args.plant,
        plant_output_path=args.output,
        equipment_input_path=args.equipment,
    )
    print(f"Wrote plant '{plant.name}' results to {args.output}")


def build_parser():
    """Build the top-level ``openpytea`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="openpytea",
        description=(
            "Run OpenPyTEA techno-economic analyses from JSON "
            "configuration files."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"OpenPyTEA {version('openpytea')}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help=(
            "Run the full TEA pipeline from a single combined config file "
            "(equipment + plant + analysis)."
        ),
    )
    run_parser.add_argument(
        "config",
        help=(
            "Path to the combined JSON config file, with top-level "
            "'equipment', 'plant', and 'analysis' keys."
        ),
    )
    run_parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help=(
            "Output directory. Overrides the config's 'output.directory'. "
            "Defaults to 'results' if neither is set."
        ),
    )
    run_parser.set_defaults(func=_cmd_run)

    tea_parser = subparsers.add_parser(
        "tea",
        help="Run the full TEA pipeline from three separate config files.",
    )
    tea_parser.add_argument(
        "--equipment", required=True, help="Path to the equipment config file."
    )
    tea_parser.add_argument(
        "--plant", required=True, help="Path to the plant config file."
    )
    tea_parser.add_argument(
        "--analysis", required=True, help="Path to the analysis config file."
    )
    tea_parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help=(
            "Output directory. Overrides the analysis file's "
            "'output.directory'. Defaults to 'results' if neither is set."
        ),
    )
    tea_parser.set_defaults(func=_cmd_tea)

    equipment_parser = subparsers.add_parser(
        "equipment",
        help="Evaluate equipment costs from a config file.",
    )
    equipment_parser.add_argument(
        "input", help="Path to the equipment config file."
    )
    equipment_parser.add_argument(
        "output", help="Path to write the equipment results JSON file."
    )
    equipment_parser.set_defaults(func=_cmd_equipment)

    plant_parser = subparsers.add_parser(
        "plant",
        help="Construct and evaluate a plant configuration.",
    )
    plant_parser.add_argument(
        "plant", help="Path to the plant config file."
    )
    plant_parser.add_argument(
        "output", help="Path to write the plant results JSON file."
    )
    plant_parser.add_argument(
        "--equipment", required=True, help="Path to the equipment config file."
    )
    plant_parser.set_defaults(func=_cmd_plant)

    return parser


def main(argv=None):
    """Entry point for the ``openpytea`` console command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
