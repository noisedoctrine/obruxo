from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import Any

from .errors import ObruxoDataError, OutputExistsError, ValidationError


DEFAULT_PROFILES = Path(__file__).resolve().parents[1] / "configs" / "component_profiles.yaml"
DEFAULT_RENDERER = Path(__file__).resolve().parents[1] / "configs" / "renderer.yaml"
DEFAULT_SOURCE_ATLAS = (
    Path(__file__).resolve().parent / "vital" / "schema" / "vital-1.0.8-vita-0.1.0" / "parameter_inventory.json"
)


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

    midi = commands.add_parser("midi", help="author and validate Standard MIDI Files")
    midi_commands = midi.add_subparsers(dest="midi_command", required=True)
    midi_validate = midi_commands.add_parser("validate", help="validate a MIDI performance")
    midi_validate.add_argument("performance")
    midi_validate.add_argument("--json", action="store_true")
    create_note = midi_commands.add_parser("create-note", help="create one fixed-tempo note")
    create_note.add_argument("--pitch", type=int, required=True)
    create_note.add_argument("--velocity", type=int, required=True)
    create_note.add_argument("--beats", type=float, required=True)
    create_note.add_argument("--bpm", type=float, default=120)
    create_note.add_argument("--ticks-per-beat", type=int, default=480)
    create_note.add_argument("--end-beats", type=float)
    _add_output(create_note)
    midi_profile = midi_commands.add_parser("apply-profile", help="validate or explicitly simplify a MIDI profile")
    midi_profile.add_argument("performance")
    midi_profile.add_argument("--profile", required=True)
    midi_profile.add_argument("--profiles", default=str(DEFAULT_PROFILES))
    midi_profile.add_argument("--simplify", action="store_true")
    _add_output(midi_profile)

    render = commands.add_parser("render", help="render one validated Vital/MIDI pair")
    render.add_argument("preset")
    render.add_argument("performance")
    render.add_argument("--sample-rate", type=int, default=44_100)
    render.add_argument("--tail", type=float, default=2.0)
    render.add_argument("--renderer-config", default=str(DEFAULT_RENDERER))
    render.add_argument("--plugin-path")
    render.add_argument("--output", required=True)
    render.add_argument("--result", required=True)
    render.add_argument("--force", action="store_true")

    batch = commands.add_parser("batch", help="run a bounded local JSONL render batch")
    batch.add_argument("requests")
    batch.add_argument("--output", required=True)
    batch.add_argument("--workers", type=int, default=1)
    batch.add_argument("--renderer-config", default=str(DEFAULT_RENDERER))
    batch.add_argument("--plugin-path")
    batch.add_argument("--no-resume", action="store_true")
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


def _print_report(report: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return
    print("valid" if report.valid else "invalid")
    for diagnostic in report.diagnostics:
        location = diagnostic.parameter or diagnostic.pointer or "performance"
        print(f"{diagnostic.severity.value}: {location}: {diagnostic.message}")


def _run_midi(args: argparse.Namespace) -> int:
    from .midi import Performance, PerformanceProfile

    if args.midi_command == "validate":
        report = Performance.from_midi(args.performance).validate()
        _print_report(report, as_json=args.json)
        return 0 if report.valid else 2
    if args.midi_command == "create-note":
        performance = Performance(ticks_per_beat=args.ticks_per_beat, bpm=args.bpm)
        duration = round(args.beats * args.ticks_per_beat)
        performance.add_note(pitch=args.pitch, velocity=args.velocity, start_tick=0, duration_ticks=duration)
        if args.end_beats is not None:
            performance.end_tick = round(args.end_beats * args.ticks_per_beat)
        performance.save_midi(_output_path(args.output, args.force))
        return 0
    if args.midi_command == "apply-profile":
        performance = Performance.from_midi(args.performance)
        profile = PerformanceProfile.load(args.profiles, args.profile)
        performance.apply_profile(profile, violations="remove" if args.simplify else "error")
        performance.save_midi(_output_path(args.output, args.force))
        return 0
    raise AssertionError(f"unhandled MIDI command {args.midi_command}")


def _renderer(args: argparse.Namespace):
    from .render import VitalRenderer

    return VitalRenderer.from_config(args.renderer_config, plugin_path=args.plugin_path)


def _run_render(args: argparse.Namespace) -> int:
    from .midi import Performance
    from .render import RenderRequest
    from .vital import VitalPreset

    output = _output_path(args.output, args.force)
    result_path = _output_path(args.result, args.force)
    preset = VitalPreset.load(args.preset)
    performance = Performance.from_midi(args.performance)
    renderer = _renderer(args)
    request = RenderRequest(
        preset=preset, performance=performance, sample_rate=args.sample_rate, end_tick=performance.end_tick,
        tail_seconds=args.tail, renderer_id=renderer.renderer_id,
    )
    result = renderer.render(request)
    errors = [item for item in result.diagnostics if item.severity.value == "error"]
    if errors:
        for diagnostic in errors:
            print(f"error: {diagnostic.message}", file=sys.stderr)
        return 2
    result.write_wav(output)
    result.write_json(result_path)
    print(json.dumps({"request_id": request.request_id, "qa": result.qa}, indent=2, sort_keys=True))
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    from .render import load_requests, run_batch

    summary = run_batch(
        _renderer(args), load_requests(args.requests), args.output,
        workers=args.workers, resume=not args.no_resume,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "vital":
            return _run_vital(args)
        if args.command == "midi":
            return _run_midi(args)
        if args.command == "render":
            return _run_render(args)
        if args.command == "batch":
            return _run_batch(args)
        raise AssertionError(f"unhandled command {args.command}")
    except (ObruxoDataError, OSError, ValueError, KeyError) as error:
        if isinstance(error, ValidationError):
            for diagnostic in error.report.diagnostics:
                print(f"{diagnostic.severity.value}: {diagnostic.message}", file=sys.stderr)
        else:
            print(f"error: {error}", file=sys.stderr)
        return 2
