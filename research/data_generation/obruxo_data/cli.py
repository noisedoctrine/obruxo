from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from .errors import ObruxoDataError, OutputExistsError, ValidationError


DEFAULT_PROFILES = Path(__file__).resolve().parents[1] / "configs" / "component_profiles.yaml"
DEFAULT_SOURCE_ATLAS = Path(__file__).resolve().parents[2] / "vital" / "vital_source_parameter_atlas.json"


def _output_path(value: str, force: bool) -> Path:
    path = Path(value)
    if path.exists() and not force:
        raise OutputExistsError(f"refusing to overwrite {path}; pass --force")
    return path


def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Author and render validated OBRUXO training data")
    parser.add_argument("--version", action="version", version="obruxo-data 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    vital = commands.add_parser("vital", help="author and validate Vital presets")
    vital_commands = vital.add_subparsers(dest="vital_command", required=True)

    schema_probe = vital_commands.add_parser("schema-probe", help="probe Vita and create a reviewed schema bundle")
    _add_output(schema_probe)
    schema_probe.add_argument("--source-atlas", default=str(DEFAULT_SOURCE_ATLAS))

    init = vital_commands.add_parser("init", help="write the canonical init preset")
    _add_output(init)

    validate = vital_commands.add_parser("validate", help="validate a Vital preset")
    validate.add_argument("preset")
    validate.add_argument("--runtime", action="store_true")
    validate.add_argument("--json", action="store_true")

    set_command = vital_commands.add_parser("set", help="set validated raw or normalized controls")
    set_command.add_argument("preset")
    set_command.add_argument("--raw", action="append", default=[], metavar="NAME=VALUE")
    set_command.add_argument("--normalized", action="append", default=[], metavar="NAME=VALUE")
    _add_output(set_command)

    apply_profile = vital_commands.add_parser("apply-profile", help="reset a preset into a component profile")
    apply_profile.add_argument("preset")
    apply_profile.add_argument("--profile", required=True)
    apply_profile.add_argument("--profiles", default=str(DEFAULT_PROFILES))
    _add_output(apply_profile)
    return parser


def _assignments(values: list[str]) -> list[tuple[str, float]]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=VALUE, got {value!r}")
        name, raw = value.split("=", 1)
        result.append((name, float(raw)))
    return result


def _run_vital(args: argparse.Namespace) -> int:
    from .vital import ComponentProfile, VitalPreset

    if args.vital_command == "schema-probe":
        from .vital.probe import probe_schema

        result = probe_schema(Path(args.output), source_atlas_path=Path(args.source_atlas), force=args.force)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.vital_command == "init":
        VitalPreset.init().save(_output_path(args.output, args.force))
        return 0
    if args.vital_command == "validate":
        report = VitalPreset.load(args.preset).validate(runtime=args.runtime)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            print("valid" if report.valid else "invalid")
            for diagnostic in report.diagnostics:
                location = diagnostic.parameter or diagnostic.pointer or "preset"
                print(f"{diagnostic.severity.value}: {location}: {diagnostic.message}")
        return 0 if report.valid else 2
    if args.vital_command == "set":
        preset = VitalPreset.load(args.preset)
        for name, value in _assignments(args.raw):
            preset.set_raw(name, value)
        for name, value in _assignments(args.normalized):
            preset.set_normalized(name, value)
        preset.save(_output_path(args.output, args.force))
        return 0
    if args.vital_command == "apply-profile":
        preset = VitalPreset.load(args.preset)
        preset.apply_profile(ComponentProfile.load(args.profiles, args.profile))
        preset.save(_output_path(args.output, args.force))
        return 0
    raise AssertionError(f"unhandled Vital command {args.vital_command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "vital":
            return _run_vital(args)
        raise AssertionError(f"unhandled command {args.command}")
    except (ObruxoDataError, OSError, ValueError, KeyError) as error:
        if isinstance(error, ValidationError):
            for diagnostic in error.report.diagnostics:
                print(f"{diagnostic.severity.value}: {diagnostic.message}", file=sys.stderr)
        else:
            print(f"error: {error}", file=sys.stderr)
        return 2
