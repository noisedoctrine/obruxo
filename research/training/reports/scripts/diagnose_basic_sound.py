"""Bounded oracle-control diagnosis of five saved basic-model predictions."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]

GROUPS = {"envelope": ("env_1_attack", "env_1_decay", "env_1_sustain", "env_1_release"),
          "level": ("osc_1_level",), "filter": ("filter_1_on", "filter_1_cutoff", "filter_1_resonance"),
          "unison_stereo": ("osc_1_unison_voices", "osc_1_unison_detune", "osc_1_pan")}


def corrections(codec, source, predicted):
    """Copy one control family at a time, retaining legal inactive defaults."""
    from obruxo_data.hashing import canonical_json
    codec.encode(source)
    codec.encode(predicted)
    settings = json.loads(source)["settings"]
    result = {"source": source, "source_repeat": source, "prediction": predicted}
    for group, names in GROUPS.items():
        document = json.loads(predicted)
        document["settings"].update({name: settings[name] for name in names})
        result[group] = canonical_json(document).encode()
        codec.encode(result[group])
    return result


def sound_metrics(source, candidate):
    import numpy as np
    from obruxo_training.evaluate import audio_distance
    if source is None or candidate is None:
        return {"failure": "source_or_candidate_not_accepted"}
    result = audio_distance(source, candidate)
    # 10 ms stereo RMS envelope, including a zero-padded final window.
    def envelope(audio):
        padded = np.pad(audio.astype(np.float64), ((0, (-len(audio)) % 441), (0, 0)))
        return np.sqrt(np.mean(padded.reshape(-1, 441, 2) ** 2, axis=1))
    left, right = envelope(source), envelope(candidate)
    result["envelope_relative_rmse"] = float(np.sqrt(np.mean((left - right) ** 2)) / max(np.sqrt(np.mean(left ** 2)), 1e-8))
    def stereo(audio):
        energy = np.mean(audio.astype(np.float64) ** 2, axis=0)
        total = max(float(energy.sum()), 1e-16)
        return np.array([energy[0] / total, np.mean((audio[:, 0].astype(np.float64) - audio[:, 1]) ** 2) / (2 * total)])
    result["stereo_energy_error"] = float(np.abs(stereo(source) - stereo(candidate)).mean())
    return result


def prepare(args):
    from obruxo_data.hashing import sha256_bytes
    from obruxo_data.render.base import RenderRequest
    from obruxo_training.pilot import PilotPool
    from obruxo_training.reproducibility import FULL_PHASE_CONTROL, fixed_generator_phases
    from obruxo_training.trainer import _require_prediction_scope
    reports = Path(__file__).resolve().parents[1] / "data"
    audit_path = reports / "basic_reconstruction_audit.json"
    audit = json.loads(audit_path.read_text())
    historical = json.loads((reports / "fresh_validation_summary.json").read_text())
    def checked(path, expected):
        raw = path.read_bytes()
        if sha256_bytes(raw) != expected:
            raise ValueError(f"historical artifact changed: {path.name}")
        return raw
    old_plan = json.loads(checked(args.evaluation / "plan.json", historical["evaluation_plan_sha256"]))
    logs = [json.loads(line) for line in checked(args.evaluation / "renders.jsonl", historical["render_log_sha256"]).splitlines()]
    receipts = {(row["case"], row["method"], row["role"]): row for row in logs}
    pool = PilotPool.load(args.pool)
    _require_prediction_scope(pool.codec, "basic")
    if pool.pool_id != old_plan["pool_id"]:
        raise ValueError("historical evaluation pool differs")
    candidate = audit["diagnostic_selection"]["candidate"]
    checkpoint_hash = old_plan["models"][candidate]["file_sha256"]
    checked(args.checkpoint, checkpoint_hash)
    entries = {entry.state_id: entry for entry in pool.entries}
    performances = {entry.performance_id: entry for entry in pool.codec.contract.performances()}
    cases, jobs, identities = [], [], []
    for case in audit["diagnostic_selection"]["cases"]:
        index = case["case"]
        old_case = json.loads(checked(args.evaluation / f"case-{index:03d}.json", historical["case_file_sha256"][f"case-{index:03d}.json"]))
        if any(old_case[key] != case[key] for key in ("state_id", "reference_performance_id", "transfer_performance_id")):
            raise ValueError("case identities differ")
        if pool.manifest.by_state[case["state_id"]].split != "validation":
            raise ValueError("diagnostic case is not validation")
        source = pool.codec.decode(entries[case["state_id"]].target)
        prediction = (args.evaluation / f"prediction-{candidate}-{index:03d}.vital").read_bytes()
        variants = corrections(pool.codec, source, prediction)
        def request(raw, performance):
            return RenderRequest(preset_json=raw.decode(), performance=performance, sample_rate=44100, channels=2,
                                 tail_seconds=pool.codec.contract.tail_seconds, renderer_id=pool.codec.contract.renderer_id)
        for role in ("reference", "transfer"):
            performance_entry = performances[case[f"{role}_performance_id"]]
            if performance_entry.role != role:
                raise ValueError("performance role differs")
            performance = performance_entry.performance()
            for name, raw in (("source", source), (candidate, prediction)):
                receipt = receipts[index, name, role]
                if request(raw, performance).request_id != receipt["request_id"] or receipt["status"] != "accepted":
                    raise ValueError("saved prediction/source does not match its accepted native receipt")
                identities.append({k: v for k, v in receipt["provenance"].items() if k != "request_id"})
            for method, raw in variants.items():
                render_request = request(fixed_generator_phases(raw), performance)
                jobs.append(({"case": index, "role": role, "method": method, "request_id": render_request.request_id}, render_request))
        cases.append({**case, "source_sha256": sha256_bytes(source), "prediction_sha256": sha256_bytes(prediction),
                      "variant_sha256": {name: sha256_bytes(raw) for name, raw in variants.items()}})
    if len(jobs) != 70 or not identities or any(value != identities[0] for value in identities):
        raise ValueError("incomplete plan or inconsistent native identity")
    plan = {"diagnostic": "basic_oracle_group_interventions_v1", "audit_sha256": sha256_bytes(audit_path.read_bytes()),
            "script_sha256_lf": sha256_bytes(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
            "checkpoint_sha256": checkpoint_hash, "pool_id": pool.pool_id, "codec_id": pool.codec.codec_id,
            "candidate": candidate, "cases": cases, "groups": GROUPS, "jobs": [item for item, _ in jobs],
            "renderer_identity": identities[0], "phase_control": FULL_PHASE_CONTROL, "worker_jobs": 1,
            "planned_requests": 70, "wall_limit_seconds": 900, "memory_limit_bytes": 8 * 1024**3,
            "training_updates": 0, "test_split_evaluated": False, "quality_claim": "oracle_diagnostic_only",
            "metrics": {"primary": "existing_mrstft", "envelope": "10ms_stereo_rms_relative_rmse_zero_pad_last",
                        "stereo": "mean_absolute_left_energy_fraction_and_side_energy_fraction_error"}}
    return pool, plan, jobs


def main():
    import numpy as np
    from scipy.io.wavfile import write as write_wav
    import torch
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.stream import NativeRendererFactory, RenderWorker
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "evaluation", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    pool, plan, jobs = prepare(args)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    factory = NativeRendererFactory(renderer_id=pool.codec.contract.renderer_id)
    monitor, rows, source, failure = MemoryMonitor(), [], None, None
    monitor.start()
    started = perf_counter()
    try:
        for identity, request in jobs:
            remaining = plan["wall_limit_seconds"] - (perf_counter() - started)
            if remaining <= 0 or (monitor.peak_bytes or 0) > plan["memory_limit_bytes"]:
                failure = "resource_budget_exhausted"
                break
            with RenderWorker(renderer_factory=factory, max_jobs=1) as worker:
                audio, metadata = worker.render(request, timeout=min(60., remaining))
            row = {**identity, **metadata}
            rows.append(row)
            if audio is not None and {k: v for k, v in metadata.get("provenance", {}).items() if k != "request_id"} != plan["renderer_identity"]:
                raise ValueError("native renderer identity changed")
            if identity["method"] == "source":
                source = audio
            if audio is not None:
                filename = f"case-{identity['case']:03d}-{identity['role']}-{identity['method']}.wav"
                write_wav(args.output / filename, 44100, audio)
                row["audio_file"] = filename
            if identity["method"] == "source_repeat":
                row["repeat_exact"] = source is not None and audio is not None and np.array_equal(source, audio)
                if not row["repeat_exact"]:
                    failure = "controlled_source_repeat_failed"
            if identity["method"] != "source":
                row["scores"] = sound_metrics(source, audio)
            write_json(args.output / "events.json", rows)
            print(json.dumps({"completed": len(rows), **identity, "status": metadata["status"]}), flush=True)
            if failure:
                break
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        monitor.close()
        elapsed = perf_counter() - started
        summary = {"plan": plan, "events": rows, "failure": failure, "completed_requests": len(rows),
                   "accepted_requests": sum(row["status"] == "accepted" for row in rows),
                   "complete": len(rows) == len(jobs) and failure is None,
                   "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                   "within_resource_limits": monitor.errors == 0 and monitor.peak_bytes is not None
                       and monitor.peak_bytes <= plan["memory_limit_bytes"] and elapsed <= plan["wall_limit_seconds"],
                   "rss_sampling_errors": monitor.errors, "quality_claim": "oracle_diagnostic_not_a_trained_model_improvement"}
        write_json(args.output / "summary.json", summary)
    return 0 if summary["complete"] and summary["accepted_requests"] == 70 and summary["within_resource_limits"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
