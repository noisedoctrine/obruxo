"""Score all saved fit recordings with level error and matched-tone error separated."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

REPOSITORY = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPOSITORY / "research/training"), str(REPOSITORY / "research/data_generation")]


def main():
    import pyloudnorm
    import torch
    from obruxo_data.hashing import canonical_sha256
    from obruxo_data.render.qa import audio_float32_sha256
    from obruxo_training.audit import write_json
    from obruxo_training.basic_level_diagnosis import ARMS, match_recordings, matched_scores, summarize
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.category_scale_refresh import cpu_settings
    from obruxo_training.conditioned_bridge import read
    from obruxo_training.listening_loudness import POLICY, LoudnessMeter
    from obruxo_training.local_execution import active_computation
    from obruxo_training.native_fit import append_durable, digest
    from obruxo_training.unison_sound_fit import lines
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fit", "parent", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    fit, proof = read(args.fit / "plan.json"), read(args.fit / "verification.json")
    cases, finals = read(args.parent / "fit_cases.json"), lines(args.fit / "finals.jsonl")
    if (not proof["verified"] or not proof["complete"] or proof["plan_id"] != canonical_sha256(fit)
            or proof["test_split_used"] or digest(args.parent / "fit_cases.json") != fit["input_files"]["parent/fit_cases.json"]
            or any(digest(args.fit / name) != proof["evidence_sha256"].get(name) for name in ("plan.json", "finals.jsonl"))
            or len(fit["fit_audit_cells"]) != 104 or len(cases) != 104 or len(finals) != 208):
        raise ValueError("level diagnosis requires the unchanged verified 104-case fit")
    lookup = {(row["index"], row["arm"]): row for row in finals}
    if len(lookup) != 208:
        raise ValueError("duplicate saved fit comparison")
    meter = LoudnessMeter(args.ffmpeg)
    independent = pyloudnorm.Meter(44100, filter_class="DeMan")
    cache, files = {}, {}

    class CachedMeter:
        def measure(self, audio):
            key = audio_float32_sha256(audio)
            if key not in cache:
                cache[key] = meter.measure(audio)
            return cache[key]

    def load(name, sha):
        path = args.fit / name
        file_sha = digest(path)
        if proof["evidence_sha256"].get(name) != file_sha:
            raise ValueError("saved fit waveform file changed")
        audio = torch.load(path, map_location="cpu", weights_only=True)["audio"].numpy()
        if audio_float32_sha256(audio) != sha:
            raise ValueError("saved fit waveform samples changed")
        files[name] = file_sha
        return audio

    args.output.mkdir(parents=True)
    names = ("obruxo_training/basic_level_diagnosis.py", "obruxo_training/listening_loudness.py", "obruxo_training/spectral.py",
             "obruxo_training/evaluate.py", "reports/scripts/diagnose_basic_levels.py")
    source = {name: digest(REPOSITORY / "research/training" / name) for name in names}
    write_json(args.output / "plan.json", {"implementation_commit": subprocess.check_output(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip(),
                                          "fit_plan_id": proof["plan_id"], "fit_verification_sha256": digest(args.fit / "verification.json"),
                                          "parent_cases_sha256": digest(args.parent / "fit_cases.json"), "source": source, "cases": 104,
                                          "prediction_waveforms": 312, "matching": POLICY, "meter": meter.identity,
                                          "independent_meter": "pyloudnorm_" + importlib.metadata.version("pyloudnorm") + "_DeMan",
                                          "grouping": "same_six_training_groups_plus_earlier_32; equal_weight_per_independent_patch",
                                          "original_scores": "reused_from_independently_rescored_verified_fit; not_modified",
                                          "limits": {"wall_seconds": 600, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 64 * 1024**2},
                                          "native_requests": 0, "inverse_inference_calls": 0, "optimizer_applications": 0,
                                          "test_split_used": False, "model_admission_allowed": False})
    started, monitor, rows, failure = perf_counter(), MemoryMonitor(), [], None
    monitor.start()
    try:
        with active_computation(), cpu_settings():
            for index, (cell, parent) in enumerate(zip(fit["fit_audit_cells"], cases, strict=True)):
                if parent["label"] != cell["label"] or parent["state_id"] != cell["recipe"]["state_id"]:
                    raise ValueError("saved fit case order changed")
                recipe, view = cell["recipe"], cell["playback_view"]
                sha = recipe["source_audio_sha256"][view]
                audio = {"reference": load(f"source_audio/{sha}.pt", sha)}
                arms = {"initial": parent["rows"]["initial"], **{arm: lookup[index, arm] for arm in ("parameter", "spectral")}}
                for arm, record in arms.items():
                    receipt = record["receipt"]
                    audio[arm] = load(receipt["retained_audio"]["file"], receipt["qa"]["audio_float32_sha256"])
                matched, matching = match_recordings(audio, CachedMeter(), independent.integrated_loudness)
                group = "earlier_32" if cell["training_index"] is None else fit["training_view_roles"][cell["input_view"]] + ("_own" if cell["input_view"] == view else "_transfer")
                row = {"index": index, "group": group, "state_id": recipe["state_id"], "training_index": cell["training_index"],
                       "input_view": cell["input_view"], "playback_view": view, "raw": {arm: arms[arm]["scores"] for arm in ARMS},
                       "matched": matched_scores(matched), "matching": matching}
                rows.append(row)
                append_durable(args.output / "cases.jsonl", row)
                elapsed = perf_counter() - started
                if elapsed > 600 or (monitor.peak_bytes or 0) > 4 * 1024**3:
                    raise RuntimeError("level diagnosis exceeded its fixed time or memory allocation")
                write_json(args.output / "progress.json", {"completed_cases": len(rows), "planned_cases": 104, "elapsed_seconds": elapsed})
            if any(digest(args.fit / name) != sha for name, sha in files.items()) or any(digest(REPOSITORY / "research/training" / name) != sha for name, sha in source.items()):
                raise ValueError("level diagnosis source or evidence changed during analysis")
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        monitor.close()
        resources = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
        report = {"complete": failure is None and len(rows) == 104, "failure": failure, "groups": summarize(rows) if rows else {},
                  "cases": len(rows), "source_and_prediction_files_sha256": files, "unique_metered_recordings": len(cache), "resources": resources,
                  "maximum_independent_spread_lu": max((row["matching"]["independent_spread_lu"] for row in rows), default=None),
                  "maximum_meter_disagreement_lu": max((row["matching"]["meter_disagreement_lu"] for row in rows), default=None),
                  "native_requests": 0, "inverse_inference_calls": 0, "optimizer_applications": 0, "qualified_for_release": False, "test_split_used": False}
        archive = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
        report["within_resource_limits"] = bool(resources["elapsed_seconds"] <= 600 and monitor.peak_bytes and monitor.peak_bytes <= 4 * 1024**3 and not monitor.errors and archive + 1024**2 < 64 * 1024**2)
        write_json(args.output / "summary.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("source_and_prediction_files_sha256", "groups")}), flush=True)


if __name__ == "__main__":
    main()
