"""Audit the completed controlled probes from saved waveforms without new rendering."""

import argparse
from itertools import combinations, product
import json
from pathlib import Path
import sys
from time import perf_counter

REPOSITORY = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPOSITORY / "research/training"), str(REPOSITORY / "research/data_generation")]


def main():
    import numpy as np
    import torch
    from obruxo_data.hashing import canonical_sha256, sha256_bytes
    from obruxo_data.midi import Performance
    from obruxo_data.render.base import RenderRequest
    from obruxo_training.audit import write_json
    from obruxo_training.basic_direct import restore_direct
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.category_scale_refresh import cpu_settings
    from obruxo_training.conditioned_bridge import read
    from obruxo_training.controlled_probes import FACTORS, validate_grid
    from obruxo_training.native_fit import digest
    from obruxo_training.pilot import PilotPool
    from obruxo_training.reproducibility import fixed_generator_phases
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "initial", "pool", "output", "meter-deps"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.meter_deps))
    import pyloudnorm
    plan, summary, loudness = (read(args.run / name) for name in ("plan.json", "summary.json", "loudness.json"))
    evidence = {str(path.relative_to(args.run)): digest(path) for path in args.run.rglob("*") if path.is_file()}
    if (not summary["complete"] or summary["plan_id"] != canonical_sha256(plan)
            or digest(args.run / "features.pt") != summary["features_sha256"] or any(digest(Path(path)) != value for path, value in plan["inputs_sha256"].items())
            or any(sha256_bytes((REPOSITORY / "research/training" / name).read_bytes().replace(b"\r\n", b"\n")) != value for name, value in plan["source_sha256_lf"].items())):
        raise ValueError("completed controlled evidence, source or inputs changed")
    rows, pool = plan["observations"], PilotPool.load(args.pool)
    states = validate_grid(rows)
    if len(rows) != 144 or len(states) != 12 or len(list((args.run / "attempts").glob("*.json"))) != 144 or len(list((args.run / "repeats").glob("*.json"))) != 12:
        raise ValueError("controlled native denominators changed")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", {"run_plan_id": summary["plan_id"], "evidence_sha256": evidence,
               "verifier_sha256_lf": sha256_bytes(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
               "limits": {"wall_seconds": 600, "process_tree_bytes": 4 * 1024**3}, "new_native_requests": 0})
    started, monitor, measured, waveforms = perf_counter(), MemoryMonitor(), [], []
    monitor.start()
    try:
        with cpu_settings():
            model, frontend, codec = restore_direct(torch.load(args.initial, map_location="cpu", weights_only=True))
            del model
            entries = {entry.state_id: entry for entry in pool.entries}
            features = torch.load(args.run / "features.pt", map_location="cpu", weights_only=True)
            meter = pyloudnorm.Meter(44100, filter_class="DeMan")
            for index, row in enumerate(rows):
                performance = Performance.from_dict(row["performance"])
                spans = performance.note_spans()
                if (len(spans) != 1 or spans[0].pitch != row["pitch"] or spans[0].velocity != row["velocity"]
                        or spans[0].duration_ticks != row["duration_ticks"] or spans[0].start_tick != 0 or performance.end_tick != 3840
                        or pool.manifest.by_state[row["state_id"]].split != "train"):
                    raise ValueError("factor labels do not describe their actual training MIDI")
                target = entries[row["state_id"]].target
                request = RenderRequest(preset_json=fixed_generator_phases(codec.decode(target)).decode(), performance=performance,
                                        tail_seconds=.25, renderer_id=pool.codec.contract.renderer_id, preroll_frames=32768)
                record = read(args.run / "recordings" / f"{index:03d}.json")
                saved = torch.load(args.run / "recordings" / f"{index:03d}.pt", map_location="cpu", weights_only=True)
                audio, metadata = saved["audio"].numpy(), record["metadata"]
                if (record["request_id"] != request.request_id or saved["request_id"] != request.request_id
                        or metadata["status"] != "accepted" or metadata["provenance"]["request_id"] != request.request_id
                        or {key: metadata["provenance"][key] for key in plan["expected_renderer_identity"]} != plan["expected_renderer_identity"]
                        or digest(args.run / "recordings" / f"{index:03d}.pt") != record["file_sha256"]
                        or audio.shape != (187425, 2) or audio.dtype != np.float32 or not np.isfinite(audio).all()
                        or sha256_bytes(audio.tobytes()) != record["audio_sha256"]):
                    raise ValueError("saved native waveform or request identity changed")
                if record["repeat"] is not None and (not record["repeat"]["exact"] or record["repeat"]["metadata"]["status"] != "accepted"):
                    raise ValueError("source repeat did not pass its live exact-sample comparison")
                processing = loudness["clips"][index]
                matched = (audio.astype(np.float64) * 10**(processing["gain_db"] / 20)).astype(np.float32)
                level = float(meter.integrated_loudness(matched.astype(np.float64)))
                if abs(level - loudness["target_lufs"]) > .05 or level != processing["independent_lufs"]:
                    raise ValueError("constant-gain matched waveform misses independent loudness check")
                for condition, value in (("original", audio), ("matched", matched)):
                    rebuilt = frontend(value)
                    if any(not torch.equal(rebuilt[name], features[condition][index]["features"][name]) for name in rebuilt):
                        raise ValueError("saved probe features do not reproduce from complete waveforms")
                measured.append(level)
                waveforms.append(record["audio_sha256"])
                if (index + 1) % 48 == 0:
                    print(json.dumps({"verified_waveforms_and_two_feature_versions": index + 1}), flush=True)
            frontend.require_frozen()
    finally:
        monitor.close()
    lookup = {(row["state_id"], *(row[name] for name in FACTORS)): index for index, row in enumerate(rows)}
    contrasts = {}
    for name, values in FACTORS.items():
        total, identical = 0, 0
        other = [key for key in FACTORS if key != name]
        for state in states:
            for setting in product(*(FACTORS[key] for key in other)):
                for lower, upper in combinations(values, 2):
                    a = {**dict(zip(other, setting, strict=True)), name: lower}
                    b = {**a, name: upper}
                    left = lookup[(state, *(a[key] for key in FACTORS))]
                    right = lookup[(state, *(b[key] for key in FACTORS))]
                    total += 1
                    identical += waveforms[left] == waveforms[right]
        contrasts[name] = {"pairs": total, "identical_waveform_pairs": identical, "rendered_waveform_changes": identical < total}
    resources = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
    if (resources["elapsed_seconds"] > 600 or not monitor.peak_bytes or monitor.peak_bytes > 4 * 1024**3 or monitor.errors
            or any(digest(args.run / name) != value for name, value in evidence.items())):
        raise ValueError("controlled verification exceeded resources or changed input evidence")
    verification = {"verified": True, "plan_id": summary["plan_id"], "resources": resources, "native_waveforms_verified": 144,
                    "feature_extractions_reproduced_exactly": 288, "source_factor_contrasts": contrasts,
                    "velocity_track_template_value": json.loads(codec.decode(entries[states[0]].target))["settings"]["velocity_track"],
                    "velocity_isolation_test_informative": contrasts["velocity"]["rendered_waveform_changes"],
                    "maximum_independent_spread_lu": max(measured) - min(measured), "all_evidence_unchanged": True,
                    "new_native_requests": 0, "inverse_model_optimizer_updates": 0, "qualified_for_release": False,
                    "repeat_limit": "Twelve repeat waveforms were compared sample-for-sample live; duplicate audio was not retained for a second offline sample comparison."}
    write_json(args.output / "verification.json", verification)
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
