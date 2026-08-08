from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from obruxo_data.midi import Performance  # noqa: E402
from obruxo_data.vital import ComponentProfile, VitalPreset  # noqa: E402


ROOT = Path(__file__).resolve().parent


def _write_preset(name: str, preset: VitalPreset, *, force: bool) -> None:
    path = ROOT / "vital" / name
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}")
    preset.save(path)


def _write_performance(name: str, performance: Performance, *, force: bool) -> None:
    path = ROOT / "midi" / name
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}")
    performance.save_midi(path)


def generate(*, force: bool = False) -> None:
    _write_preset("canonical_init.vital", VitalPreset.init(), force=force)

    oscillator = VitalPreset.init()
    oscillator.apply_profile(ComponentProfile.only(oscillators=[1]))
    _write_preset("oscillator_1.vital", oscillator, force=force)

    filtered = VitalPreset.init()
    filtered.apply_profile(ComponentProfile.only(oscillators=[1], filters=[1]))
    filtered.set_raw("osc_1_on", 1.0)
    filtered.set_raw("osc_1_destination", 0.0)
    filtered.set_raw("filter_1_on", 1.0)
    _write_preset("oscillator_1_filter_1.vital", filtered, force=force)

    routed = VitalPreset.init()
    routed.apply_profile(ComponentProfile.only(oscillators=[1], lfos=[1], max_active_routes=1))
    routed.connect_modulation(1, "lfo_1", "osc_1_level", amount=0.25)
    _write_preset("lfo_1_to_oscillator_1.vital", routed, force=force)

    effected = VitalPreset.init()
    effected.apply_profile(ComponentProfile.only(oscillators=[1], effects=["reverb"]))
    effected.set_raw("reverb_on", 1.0)
    _write_preset("reverb_enabled.vital", effected, force=force)

    canonicalization = VitalPreset.init().to_dict()
    canonicalization["settings"].pop("osc_2_level")
    canonicalization_path = ROOT / "vital" / "missing_scalar_canonicalization.vital"
    if canonicalization_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {canonicalization_path}")
    canonicalization_path.parent.mkdir(parents=True, exist_ok=True)
    canonicalization_path.write_text(
        json.dumps(canonicalization, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )

    held = Performance(bpm=120)
    held.add_note(pitch=60, velocity=100, start_tick=0, duration_ticks=480)
    held.end_tick = 960
    _write_performance("held_note_with_tail.mid", held, force=force)

    sequential = Performance(bpm=120)
    sequential.add_note(pitch=60, velocity=100, start_tick=0, duration_ticks=480)
    sequential.add_note(pitch=64, velocity=100, start_tick=480, duration_ticks=480)
    _write_performance("sequential_boundary.mid", sequential, force=force)

    chord = Performance(bpm=120)
    for pitch in (60, 64, 67):
        chord.add_note(pitch=pitch, velocity=90, start_tick=0, duration_ticks=960)
    chord.end_tick = 1200
    _write_performance("overlapping_chord.mid", chord, force=force)

    tempo_change = Performance(bpm=120)
    tempo_change.add_note(pitch=60, velocity=100, start_tick=0, duration_ticks=960)
    tempo_change.add_tempo(90, tick=480)
    _write_performance("tempo_change_unsupported.mid", tempo_change, force=force)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    generate(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
