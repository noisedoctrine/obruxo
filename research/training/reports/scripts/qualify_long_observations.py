"""Observe every previously too-slow training source, then train/recover a declared six-source panel."""
import argparse
from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
from pathlib import Path
import shutil
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


@dataclass(frozen=True)
class SourcePanel:
    parent: object
    indices: tuple

    @property
    def codec(self):
        return self.parent.codec

    @property
    def manifest(self):
        return self.parent.manifest

    @property
    def entries(self):
        return tuple(self.parent.entries[index] for index in self.indices)

    @cached_property
    def pool_id(self):
        from obruxo_data.hashing import canonical_sha256
        return canonical_sha256({"panel": "all_training_states_without_previous_attack_compatible_reference_v1",
                                 "parent_pool_id": self.parent.pool_id, "indices": list(self.indices)})

    def source_preset(self, target, index):
        return self.parent.source_preset(target, self.indices[index])


def equal(a, b):
    import torch
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def main():
    import numpy as np
    import torch
    from obruxo_data.hashing import canonical_json, canonical_sha256
    from obruxo_data.render.base import RenderRequest
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.extended_features import features_for
    from obruxo_training.features import tensor_digest
    from obruxo_training.observation_bank import PROFILE
    from obruxo_training.performance_eligibility import attack_seconds, eligible, longest_held_seconds
    from obruxo_training.performances import bank_for
    from obruxo_training.pilot import PilotPool, PilotSampler
    from obruxo_training.reproducibility import FULL_PHASE_CONTROL
    from obruxo_training.stream import NativeRendererFactory, RenderWorker, StreamConfig, request_frames
    from obruxo_training.trainer import TrainConfig, load_model, train_pilot, validate_replay
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    monitor = MemoryMonitor()
    monitor.start()
    try:
        pool = PilotPool.load(args.pool)
        if pool.codec.contract.performance_profile != PROFILE:
            raise ValueError("qualification requires the explicit long-observation profile")
        references = [entry for entry in pool.codec.contract.performances() if entry.role == "reference"]
        old_max = max(longest_held_seconds(entry.performance()) for entry in bank_for(pool.codec.contract.renderer_id) if entry.role == "reference")
        selected = tuple(index for index, entry in enumerate(pool.entries) if attack_seconds(pool.codec, entry.target) > old_max)
        panel = SourcePanel(pool, selected)
        sampler = PilotSampler(pool, policy="attack_compatible")
        frontend = features_for(pool.codec)
        files = [Path(__file__), *(RESEARCH / "training/obruxo_training").glob("*.py"),
                 *(RESEARCH / "data_generation/obruxo_data/render").glob("*.py"), RESEARCH / "data_generation/obruxo_data/vital/full_contract.py"]

        def sources():
            return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in files}

        plan = {"qualification": "long_observation_sources_features_and_training_v1", "source_sha256_lf": sources(),
                "pool_id": pool.pool_id, "codec_id": pool.codec.codec_id, "split_manifest_id": pool.manifest.manifest_id,
                "training_states": len(pool.entries), "previous_maximum_reference_seconds": old_max,
                "slow_training_states": [entry.state_id for entry in panel.entries], "panel_id": panel.pool_id,
                "frontend": frontend.describe(), "bank": [entry.to_dict() for entry in pool.codec.contract.performances()],
                "native_diagnostic_phase_control": FULL_PHASE_CONTROL, "training_phase_control": "original_requested_values_unchanged",
                "native_selection": "shortest_adequate_reference_then_lowest_velocity_then_performance_id",
                "diagnostic_attempt_budget": 2 * len(selected), "training_attempt_budget": 24,
                "rss_limit_bytes": 8 * 1024**3, "final_test_audio": False, "model_promotion": False}
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "plan.json", plan)
        factory = NativeRendererFactory(renderer_id=pool.codec.contract.renderer_id)
        checks, events = [], []
        for index in selected:
            entry = pool.entries[index]
            raw = pool.source_preset(entry.target, index)
            document = json.loads(raw)
            document["settings"].update(FULL_PHASE_CONTROL)
            choices = [value for value in references if eligible(pool.codec, entry.target, value.performance())]
            chosen = min(choices, key=lambda value: (longest_held_seconds(value.performance()), max(note.velocity for note in value.performance().note_spans()), value.performance_id))
            performance = chosen.performance()
            buffers, pair = [], []
            for role in ("long", "repeat"):
                request = RenderRequest(preset_json=canonical_json(document), performance=performance, tail_seconds=pool.codec.contract.tail_seconds,
                                        renderer_id=pool.codec.contract.renderer_id)
                with RenderWorker(renderer_factory=factory, max_jobs=1) as worker:
                    audio, metadata = worker.render(request, timeout=60.)
                row = {"state_id": entry.state_id, "role": role, "request_id": request.request_id, **metadata}
                events.append(row)
                pair.append(row)
                buffers.append(audio)
                write_json(args.output / "events.json", events)
                print(json.dumps({"state_id": entry.state_id, "role": role, "status": metadata["status"]}), flush=True)
            positive = all(row["status"] == "accepted" for row in pair)
            features = frontend(buffers[0]) if positive else None
            late = buffers[0][16 * 44100:] if positive else np.zeros((0, 2), dtype=np.float32)
            checks.append({"state_id": entry.state_id, "attack_seconds": attack_seconds(pool.codec, entry.target),
                           "held_seconds": longest_held_seconds(performance), "audio_seconds": request_frames(request) / 44100,
                           "accepted": positive, "repeat_exact": positive and np.array_equal(buffers[0], buffers[1]),
                           "feature_sha256": tensor_digest(features) if features else None,
                           "late_rms_after_16s": float(np.sqrt(np.mean(late.astype(np.float64) ** 2))) if late.size else None,
                           "full_readback": positive and all(row["renderer_qa"]["native_state"]["verified_loaded_scalars"] == 775
                               and len(row["renderer_qa"]["native_state"]["nested_families_verified"]) == 6 for row in pair)})
            write_json(args.output / "checks.json", checks)
        frontend.require_frozen()
        del frontend, buffers
        config = TrainConfig(steps=3, replay_capacity=len(selected), batch_size=len(selected), render_budget=24,
                             checkpoint_every=1, cpu_threads=2, sampling_policy="attack_compatible")

        def snapshot(step):
            if step == 1:
                shutil.copyfile(args.output / "training/checkpoint.pt", args.output / "step1.pt")

        complete = train_pilot(panel, args.output / "training", config=config, stream_config=StreamConfig(prefetch_clips=1),
                               renderer_factory=factory, after_step=snapshot, progress=lambda row: print(json.dumps(row), flush=True))
        recovered = train_pilot(panel, args.output / "recovered", config=config, stream_config=StreamConfig(prefetch_clips=1),
                                renderer_factory=factory, resume=args.output / "step1.pt")
        left, right = [torch.load(args.output / name / "checkpoint.pt", weights_only=True) for name in ("training", "recovered")]
        validate_replay(left["replay"], PilotSampler(panel, policy=config.sampling_policy), left["cursor"], config)
        _, restored, exported = load_model(args.output / "training/model.pt")
        gates = {"every_training_state_has_adequate_reference": len(sampler._eligible) == len(pool.entries) and all(sampler._eligible.values()),
                 "six_prior_gaps_covered": len(selected) == 6 and all(row["accepted"] and row["repeat_exact"] and row["full_readback"] for row in checks),
                 "long_audio_features_used": any(row["accepted"] and row["audio_seconds"] > 16 and row["late_rms_after_16s"] > 0 and row["feature_sha256"] for row in checks),
                 "native_buffers_fit_worker": all(row["audio_seconds"] * 44100 * 8 < 32 * 1024**2 for row in checks),
                 "training_complete": complete["complete"] and recovered["complete"],
                 "all_six_sources_consumed": {row["state_id"] for row in left["replay"]} == {entry.state_id for entry in panel.entries},
                 "exact_recovery": all(equal(left[key], right[key]) for key in ("model", "optimizer", "rng", "batch_rng", "replay", "cursor")),
                 "no_recovery_renders": recovered["fresh_accepted_this_segment"] == 0,
                 "extended_frontend_export": restored.describe()["maximum_audio_seconds"] == 80 and exported.codec_id == pool.codec.codec_id,
                 "frozen_frontend": complete["frozen_weights_and_buffers_unchanged"] and recovered["frozen_weights_and_buffers_unchanged"],
                 "source_unchanged": sources() == plan["source_sha256_lf"]}
    finally:
        monitor.close()
    gates["memory_bounded"] = monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["rss_limit_bytes"]
    summary = {"plan_sha256": canonical_sha256(plan), "passed": all(gates.values()), "gates": gates, "source_checks": checks,
               "training": complete, "recovery": recovered, "peak_process_tree_rss_bytes": monitor.peak_bytes,
               "claim": "bounded_observation_and_training_qualification_not_full_corpus_admission_or_model_quality",
               "final_test_audio": False, "model_promotion": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"passed": summary["passed"], "gates": gates}), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
