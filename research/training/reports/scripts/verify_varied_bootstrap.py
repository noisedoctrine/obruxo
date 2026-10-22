"""Replay wider-input learning and reconstruct both native comparison panels."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]

from obruxo_data.hashing import canonical_sha256, sha256_bytes  # noqa: E402
from obruxo_data.render.base import RenderRequest  # noqa: E402
from obruxo_training.benchmark import MemoryMonitor  # noqa: E402
from obruxo_training.checkpoints import load_training_checkpoint  # noqa: E402
from obruxo_training.codec import PilotTarget  # noqa: E402
from obruxo_training.evaluate import nearest_training_target  # noqa: E402
from obruxo_training.extended_features import features_for  # noqa: E402
from obruxo_training.features import tensor_digest  # noqa: E402
from obruxo_training.model import PilotInverse, batch_features, loss_terms, predicted_targets  # noqa: E402
from obruxo_training.paired import identity_terms, paired_batch_indices  # noqa: E402
from obruxo_training.pilot import PilotPool  # noqa: E402
from obruxo_training.profiled_sampler import profile_metadata, training_sampler  # noqa: E402
from obruxo_training.reproducibility import fixed_generator_phases  # noqa: E402
from obruxo_training.trainer import TrainConfig, replay_metrics, restore_model, validate_replay  # noqa: E402
from verify_basic_bootstrap import wait_for_native  # noqa: E402


def json_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def replay_fit(saved, initial, initial_model, pool, config, fit_summary):
    seed = config.seed if config.learning_seed_override is None else config.learning_seed_override
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    frontend = features_for(pool.codec, initial["basic_pitch"])
    model = PilotInverse(pool.codec, config.conditioning, config.cyclic_head)
    assert initial["step"] == 0 and initial["optimizer"]["state"] == {}
    assert tensor_digest(initial["model"]) == tensor_digest(initial_model["model"])
    model.load_state_dict(initial["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0)
    generator = torch.Generator().manual_seed(seed)
    replay = saved["replay"]
    assert replay_metrics(model, replay, pool.codec, "cpu", config) == fit_summary["initial_metrics"]
    for _step in range(config.steps):
        items = [replay[index] for index in paired_batch_indices(replay, config.batch_size, generator)]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch_features(items))
        terms = loss_terms(prediction, [PilotTarget.from_dict(item["target"]) for item in items], pool.codec)
        identity = identity_terms(prediction["representation"], config.identity_temperature)
        loss = terms["loss"] + config.identity_weight * identity["identity_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
    assert tensor_digest(model.state_dict()) == tensor_digest(saved["model"]) == fit_summary["model_state_sha256"]
    for index, state in optimizer.state_dict()["state"].items():
        assert all(torch.equal(value, saved["optimizer"]["state"][index][name]) for name, value in state.items())
    assert torch.equal(generator.get_state(), saved["batch_rng"]) and torch.equal(torch.get_rng_state(), saved["rng"]["torch"])
    assert replay_metrics(model, replay, pool.codec, "cpu", config) == fit_summary["final_metrics"]
    assert tensor_digest(frontend.basic_pitch.state_dict()) == tensor_digest(saved["basic_pitch"])
    return model.eval()


def verify(root, pool_path, initial_root, frozen_commit, output, *, wait_native_seconds=0):
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        plan = json.loads((root / "plan.json").read_text())
        assert canonical_sha256(plan) == json.loads((root / "run.json").read_text())["plan_id"]
        for name, expected in plan["source"].items():
            raw = subprocess.check_output(["git", "show", f"{frozen_commit}:research/training/obruxo_training/{name}"], cwd=RESEARCH.parent)
            assert sha256_bytes(raw.replace(b"\r\n", b"\n")) == expected
        pool, config = PilotPool.load(pool_path), TrainConfig(**plan["config"])
        assert pool.pool_id == plan["pool_id"]
        segments = sorted(root.glob("fit-*"))
        fit_root = segments[-1]
        fit_summary = json.loads((fit_root / "summary.json").read_text())
        assert fit_summary["complete"]
        saved = load_training_checkpoint(fit_root / "checkpoint.pt")
        initial = torch.load(segments[0] / "checkpoint.initial.pt", weights_only=True)["state"]
        initial_path = initial_root / "fit-000/model.pt"
        assert sha256_bytes(initial_path.read_bytes()) == plan["initial_model"]["file_sha256"] == initial["run_contract"]["initial_model"]["file_sha256"]
        initial_model = torch.load(initial_path, weights_only=True)
        assert tensor_digest(initial_model["model"]) == plan["initial_model"]["model_state_sha256"]
        validate_replay(saved["replay"], training_sampler(pool, config), saved["cursor"], config)
        model = replay_fit(saved, initial, initial_model, pool, config, fit_summary)
        replay_seconds = perf_counter() - started
        summary = wait_for_native(root, wait_native_seconds)
        assert summary["complete"] and summary["plan_id"] == canonical_sha256(plan) and summary["training"] == fit_summary
        exported = torch.load(fit_root / "model.pt", weights_only=True)
        assert tensor_digest(exported["model"]) == tensor_digest(saved["model"])
        assert all(exported[key] == value for key, value in profile_metadata(32768).items())
        baseline, _, _ = restore_model(initial_model)
        old_saved = load_training_checkpoint(initial_root / "fit-000/checkpoint.pt")
        models = {"baseline": baseline, "varied": model}
        samplers = {"original": training_sampler(pool, TrainConfig(**plan["original_sampler_config"])), "wide": training_sampler(pool, config)}
        memories = {"original": old_saved["replay"], "wide": saved["replay"]}
        source_rows = [row for segment in segments for row in json_rows(segment / "renders.jsonl")]
        sources = {row["provenance"]["request_id"]: row for row in source_rows}
        assert len(sources) == len(saved["replay"]) == len(plan["sources"]) == 808
        for item, spec in zip(saved["replay"], plan["sources"], strict=True):
            assert item["recipe_id"] == spec["recipe_id"] and item["audio_sha256"] == sources[spec["request_id"]]["qa"]["audio_float32_sha256"]
        all_outcomes = [{"phase": "training", **row} for row in source_rows]
        differences, pairs, reconstructed, observed, ratios, template_ratios = [], 0, 0, 0, {}, {}
        performances = {item.performance_id: item.performance() for item in pool.codec.contract.performances()}
        for panel, sampler in samplers.items():
            audit_root = root / f"audit-{panel}"
            receipts = json_rows(audit_root / "renders.jsonl")
            all_outcomes += [{"phase": "evaluation", **row} for row in receipts]
            lookup = {(row["case"], row["method"], row["role"]): row for row in receipts}
            cases, expected = summary["audits"][panel]["cases"], {}
            for index, spec in enumerate(plan["panels"][panel]):
                recipe = sampler.recipe_at(spec["sequence_id"])
                request = recipe.render_request()
                assert recipe.recipe_id == spec["recipe_id"] and recipe.state_id == cases[index]["state_id"] == spec["state_id"]
                item = next(item for item in memories[panel] if item["sequence_id"] == spec["sequence_id"])
                assert item["audio_sha256"] == lookup[index, "source", "reference"]["qa"]["audio_float32_sha256"]
                patches = {"source": request.preset_json.encode(), "source_repeat": request.preset_json.encode(),
                           "fixed_template": pool.codec.decode(pool.codec.default_target()),
                           "train_retrieval": pool.codec.decode(nearest_training_target(item["features"], saved["replay"]))}
                for name, candidate in models.items():
                    with torch.no_grad():
                        predicted = predicted_targets(candidate(batch_features([item])), pool.codec)[0]
                    raw = (audit_root / f"prediction-{name}-{index:03d}.vital").read_bytes()
                    stored = pool.codec.encode(raw)
                    assert predicted.categorical == stored.categorical
                    delta = max(abs(left - right) for left, right in zip(pool.codec.canonical_target(predicted).continuous, stored.continuous, strict=True))
                    assert delta <= 1e-6
                    differences.append(delta)
                    patches[name] = raw
                for role, performance in (("reference", request.performance), ("transfer", performances[cases[index]["transfer_performance_id"]])):
                    assert lookup[index, "source", role]["qa"]["audio_float32_sha256"] == lookup[index, "source_repeat", role]["qa"]["audio_float32_sha256"]
                    pairs += 1
                    for method, raw in patches.items():
                        derived = RenderRequest(preset_json=fixed_generator_phases(raw).decode(), performance=performance,
                                                renderer_id=pool.codec.contract.renderer_id, tail_seconds=pool.codec.contract.tail_seconds, preroll_frames=32768)
                        assert derived.request_id == lookup[index, method, role]["request_id"]
                        expected[index, method, role] = derived.request_id
            assert all(row["request_id"] == expected[row["case"], row["method"], row["role"]] for row in receipts)
            actual_counts, expected_counts = Counter(row["request_id"] for row in receipts), Counter(expected.values())
            assert actual_counts.keys() == expected_counts.keys() and all(actual_counts[key] >= value for key, value in expected_counts.items())
            reconstructed += len(expected)
            observed += len(receipts)
            ratios[panel], template_ratios[panel] = {}, {}
            for role in ("reference", "transfer"):
                scores = {name: float(np.mean([case["methods"][name][role]["mrstft"] for case in cases])) for name in (*models, "fixed_template")}
                assert all(scores[name] == summary["audits"][panel]["aggregates"][name][role]["mean_mrstft"] for name in scores)
                ratios[panel][role] = (scores["varied"] + 1e-12) / (scores["baseline"] + 1e-12)
                template_ratios[panel][role] = (scores["varied"] + 1e-12) / (scores["fixed_template"] + 1e-12)
        assert ratios == summary["candidate_baseline_ratios"] and template_ratios == summary["candidate_template_ratios"]
        attempts = json_rows(root / "attempts.jsonl")
        assert Counter((row["phase"], row["request_id"]) for row in attempts) == Counter(
            (row["phase"], row.get("request_id", row["provenance"]["request_id"])) for row in all_outcomes)
        assert all(row["status"] == "accepted" for row in all_outcomes)
        assert len(attempts) == summary["attempts"] and 1000 <= len(attempts) <= 1012 and reconstructed == 192 and pairs == 32
        result = {"plan_id": summary["plan_id"], "frozen_commit": frozen_commit, "replayed_updates": config.steps,
                  "model_optimizer_batch_and_global_rng_exact": True, "training_source_receipts_verified": 808,
                  "all_native_audit_requests_reconstructed": observed, "planned_native_audit_requests_reconstructed": reconstructed,
                  "hard_patch_request_exports_reconstructed": len(differences),
                  "native_source_repeat_pairs_exact": pairs, "cached_prediction_tolerance": 1e-6,
                  "cached_vs_fresh_input_predictions_max_normalized_delta": max(differences), "hard_categories_exact": True,
                  "candidate_baseline_ratios_reconstructed": ratios, "charged_native_attempts": len(attempts),
                  "native_renders_added": 0, "training_replay_seconds": replay_seconds, "elapsed_seconds": perf_counter() - started,
                  "waited_for_native_audit": bool(wait_native_seconds), "verifier_sha256_lf": sha256_bytes(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
                  "peak_process_tree_rss_bytes": monitor.peak_bytes,
                  "limitations": "waveforms discarded; spectral scores reconstructed from saved cases, not independently rescored",
                  "evidence_sha256": {path.relative_to(root).as_posix(): sha256_bytes(path.read_bytes()) for path in root.rglob("*") if path.is_file()},
                  "qualified_for_release": False}
        assert result["elapsed_seconds"] < 600 and monitor.peak_bytes is not None and monitor.peak_bytes < 4 * 1024**3 and monitor.errors == 0
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return {key: value for key, value in result.items() if key != "evidence_sha256"}
    finally:
        monitor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "pool", "initial-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--frozen-commit", required=True)
    parser.add_argument("--wait-native-seconds", type=int, choices=range(481), default=0, metavar="0..480")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.pool, args.initial_root, args.frozen_commit, args.output, wait_native_seconds=args.wait_native_seconds)), flush=True)
