"""Replay the small bootstrap from its original weights without rendering or changing it."""

from collections import Counter
import json
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter

import numpy as np

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def verify(root, pool_path, frozen_commit, output):
    import torch
    from obruxo_data.hashing import canonical_sha256, sha256_bytes
    from obruxo_data.render.base import RenderRequest
    from obruxo_training.basic_bootstrap import rows
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.checkpoints import load_training_checkpoint
    from obruxo_training.codec import PilotTarget
    from obruxo_training.evaluate import nearest_training_target
    from obruxo_training.extended_features import features_for
    from obruxo_training.features import tensor_digest
    from obruxo_training.model import PilotInverse, batch_features, loss_terms, predicted_targets
    from obruxo_training.pilot import PilotPool
    from obruxo_training.profiled_sampler import profile_metadata, training_sampler
    from obruxo_training.reproducibility import fixed_generator_phases
    from obruxo_training.trainer import TrainConfig, replay_metrics, validate_replay
    started = perf_counter()
    monitor = MemoryMonitor()
    monitor.start()
    try:
        plan = json.loads((root / "plan.json").read_text())
        summary = json.loads((root / "summary.json").read_text())
        if canonical_sha256(plan) != summary["plan_id"]:
            raise ValueError("saved preflight plan identity changed")
        for name, expected in plan["source"].items():
            raw = subprocess.check_output(["git", "show", f"{frozen_commit}:research/training/obruxo_training/{name}"], cwd=RESEARCH.parent)
            if sha256_bytes(raw.replace(b"\r\n", b"\n")) != expected:
                raise ValueError("frozen source does not match the executed plan")
        segments = sorted(root.glob("fit-*"))
        if len(segments) != 1:
            raise ValueError("this small audit requires the uninterrupted preflight")
        fit_root = segments[0]
        saved = load_training_checkpoint(fit_root / "checkpoint.pt")
        initial = torch.load(fit_root / "checkpoint.initial.pt", weights_only=True)["state"]
        config = TrainConfig(**plan["config"])
        pool = PilotPool.load(pool_path)
        if pool.pool_id != plan["pool_id"] or saved["step"] != config.steps:
            raise ValueError("pool or completed update count changed")
        validate_replay(saved["replay"], training_sampler(pool, config), saved["cursor"], config)
        torch.set_num_threads(config.cpu_threads)
        torch.use_deterministic_algorithms(True)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        frontend = features_for(pool.codec, initial["basic_pitch"])
        model = PilotInverse(pool.codec, config.conditioning, config.cyclic_head)
        if tensor_digest(model.state_dict()) != tensor_digest(initial["model"]):
            raise ValueError("original seeded initialization differs")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0)
        generator = torch.Generator().manual_seed(config.seed)
        replay = saved["replay"]
        assert replay_metrics(model, replay, pool.codec, "cpu", config) == summary["training"]["initial_metrics"]
        for _step in range(config.steps):
            items = [replay[index] for index in torch.randperm(len(replay), generator=generator)[:config.batch_size].tolist()]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            terms = loss_terms(model(batch_features(items)), [PilotTarget.from_dict(item["target"]) for item in items], pool.codec)
            terms["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
        assert tensor_digest(model.state_dict()) == tensor_digest(saved["model"]) == summary["model_state_sha256"]
        for index, state in optimizer.state_dict()["state"].items():
            assert all(torch.equal(value, saved["optimizer"]["state"][index][name]) for name, value in state.items())
        assert torch.equal(generator.get_state(), saved["batch_rng"])
        assert torch.equal(torch.get_rng_state(), saved["rng"]["torch"])
        assert replay_metrics(model, replay, pool.codec, "cpu", config) == summary["training"]["final_metrics"]
        assert tensor_digest(frontend.basic_pitch.state_dict()) == tensor_digest(saved["basic_pitch"])
        exported = torch.load(fit_root / "model.pt", weights_only=True)
        assert tensor_digest(exported["model"]) == tensor_digest(saved["model"])
        assert all(exported[key] == value for key, value in profile_metadata(32768).items())
        source_rows = rows(fit_root / "renders.jsonl")
        audit_rows = rows(root / "training-audit/renders.jsonl")
        attempts = rows(root / "attempts.jsonl")
        outcomes = [*(dict(phase="training", **row) for row in source_rows), *(dict(phase="evaluation", **row) for row in audit_rows)]
        assert Counter((row["phase"], row["request_id"]) for row in attempts) == Counter(
            (row["phase"], row.get("request_id", row["provenance"]["request_id"])) for row in outcomes)
        assert len(outcomes) == 88 and all(row["status"] == "accepted" for row in outcomes)
        assert all({key: value for key, value in row["provenance"].items() if key != "request_id"} == plan["renderer_identity"] for row in outcomes)
        lookup = {(row["case"], row["method"], row["role"]): row for row in audit_rows}
        performances = {item.performance_id: item.performance() for item in pool.codec.contract.performances()}
        cases = [json.loads((root / f"training-audit/case-{index:03d}.json").read_text()) for index in range(8)]
        sampler = training_sampler(pool, config)
        expected_requests = []
        export_differences = []
        for index, item in enumerate(replay):
            request = sampler.recipe_at(item["sequence_id"]).render_request()
            assert item["audio_sha256"] == lookup[index, "source", "reference"]["qa"]["audio_float32_sha256"]
            assert source_rows[index]["provenance"]["request_id"] == request.request_id
            with torch.no_grad():
                target = predicted_targets(model.eval()(batch_features([item])), pool.codec)[0]
            exported_patch = (root / f"training-audit/prediction-{index:03d}.vital").read_bytes()
            stored_target = pool.codec.encode(exported_patch)
            assert target.categorical == stored_target.categorical
            delta = max(abs(left - right) for left, right in zip(pool.codec.canonical_target(target).continuous, stored_target.continuous, strict=True))
            assert delta <= 1e-6
            export_differences.append(delta)
            patches = {"source": request.preset_json.encode(), "source_repeat": request.preset_json.encode(),
                       "model": exported_patch, "fixed_template": pool.codec.decode(pool.codec.default_target()),
                       "train_retrieval": pool.codec.decode(nearest_training_target(item["features"], replay))}
            for role, performance in (("reference", request.performance), ("transfer", performances[cases[index]["transfer_performance_id"]])):
                assert lookup[index, "source", role]["qa"]["audio_float32_sha256"] == lookup[index, "source_repeat", role]["qa"]["audio_float32_sha256"]
                for method, raw in patches.items():
                    derived = RenderRequest(preset_json=fixed_generator_phases(raw).decode(), performance=performance,
                                            renderer_id=pool.codec.contract.renderer_id, tail_seconds=pool.codec.contract.tail_seconds, preroll_frames=32768)
                    assert derived.request_id == lookup[index, method, role]["request_id"]
                    expected_requests.append(derived.request_id)
        assert Counter(expected_requests) == Counter(row["request_id"] for row in audit_rows)
        ratios = {role: float(np.mean([case["methods"]["model"][role]["mrstft"] for case in cases])) /
                       float(np.mean([case["methods"]["fixed_template"][role]["mrstft"] for case in cases])) for role in ("reference", "transfer")}
        assert ratios == summary["model_vs_template_ratio"]
        result = {"plan_id": summary["plan_id"], "frozen_commit": frozen_commit, "replayed_updates": config.steps,
                  "model_optimizer_batch_and_global_rng_exact": True, "source_audio_hashes_match_training": 8,
                  "native_source_repeat_pairs_exact": 16, "hard_patch_request_exports_reconstructed": 8,
                  "cached_vs_fresh_input_predictions_max_normalized_delta": max(export_differences),
                  "cached_prediction_tolerance": 1e-6, "hard_categories_exact": True,
                  "all_eighty_audit_requests_reconstructed": True, "native_renders_added": 0,
                  "model_vs_template_ratio_reconstructed": ratios,
                  "elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                  "limitations": "candidate waveforms discarded; saved spectral scores not independently rescored; historical evaluator uses four CPU threads vs two for training, so cached/fresh predictions are compared numerically, not bytewise",
                  "evidence_sha256": {path.relative_to(root).as_posix(): sha256_bytes(path.read_bytes())
                                      for path in root.rglob("*") if path.is_file()}, "qualified_for_release": False}
        if result["elapsed_seconds"] > 600 or (monitor.peak_bytes or 0) > 4 * 1024**3 or monitor.peak_bytes is None or monitor.errors:
            raise RuntimeError("zero-render replay audit resource limit failed")
        output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
        return {key: value for key, value in result.items() if key != "evidence_sha256"}
    finally:
        monitor.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--frozen-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.pool, args.frozen_commit, args.output)), flush=True)
