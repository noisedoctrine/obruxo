from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from obruxo_data.midi import Performance  # noqa: E402
from obruxo_data.render import RenderRequest, VitalRenderer  # noqa: E402
from obruxo_data.render.qa import RepeatabilityTolerance, compare_audio  # noqa: E402
from obruxo_data.vital import VitalPreset, VitalSchema  # noqa: E402


ROOT = Path(__file__).resolve().parent
PRESETS = (
    "canonical_init.vital", "oscillator_1.vital", "oscillator_1_filter_1.vital",
    "lfo_1_to_oscillator_1.vital", "reverb_enabled.vital",
)
PERFORMANCES = ("held_note_with_tail.mid", "sequential_boundary.mid", "overlapping_chord.mid")


def generate(plugin_path: Path, output: Path, *, force: bool = False) -> None:
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {output}")
    renderer = VitalRenderer(plugin_path)
    tolerance = RepeatabilityTolerance()
    cases = []
    failures = []
    for preset_name in PRESETS:
        preset = VitalPreset.load(ROOT / "vital" / preset_name)
        for performance_name in PERFORMANCES:
            performance = Performance.from_midi(ROOT / "midi" / performance_name)
            request = RenderRequest(preset=preset, performance=performance, tail_seconds=0.25)
            first = renderer.render(request)
            second = renderer.render(request)
            comparison = compare_audio(first.audio, second.audio, tolerance=tolerance)
            if not comparison["within_tolerance"]:
                failures.append({"preset": preset_name, "performance": performance_name, "comparison": comparison})
            cases.append({
                "preset": preset_name,
                "performance": performance_name,
                "request_id": request.request_id,
                "audio_float32_sha256": first.qa["audio_float32_sha256"],
                "repeat_audio_float32_sha256": second.qa["audio_float32_sha256"],
                "qa": first.qa,
                "repeatability": comparison,
            })
    tempo_change = Performance.from_midi(ROOT / "midi" / "tempo_change_unsupported.mid")
    rejection = tempo_change.validate(renderer.capabilities)
    if rejection.valid:
        raise RuntimeError("tempo-change fixture was not rejected by renderer capabilities")
    result = {
        "artifact_schema": "obruxo_vital_reference_results_v1",
        "schema_id": VitalSchema.load().schema_id,
        "renderer_id": renderer.renderer_id,
        "engine_fingerprint": first.provenance.engine_fingerprint,
        "plugin_sha256": first.provenance.engine_fingerprint,
        "tolerance": {
            "rms_relative": tolerance.rms_relative,
            "peak_absolute": tolerance.peak_absolute,
            "log_spectral_rmse": tolerance.log_spectral_rmse,
        },
        "cases": cases,
        "repeatability_failures": failures,
        "unsupported_tempo_change": rejection.to_dict(),
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    if failures:
        raise RuntimeError(f"repeatability tolerance failed for {len(failures)} case(s); inspect {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-path", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reference_results.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    generate(args.plugin_path, args.output, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
