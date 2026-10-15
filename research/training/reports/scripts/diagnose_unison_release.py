import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "tmp/training-pipeline-pr"
for directory in (CODE / "research/data_generation", CODE / "research/training"):
    sys.path.insert(0, str(directory))

import numpy as np
import torch
from obruxo_training.codec import CONTINUOUS, PilotTarget
from obruxo_training.evaluate import load_evaluation_checkpoint, retrieval_digest
from obruxo_training.model import batch_features, predicted_targets
from obruxo_training.pilot import PilotPool


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correlation(left, right):
    return float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 1e-10 and np.std(right) > 1e-10 else None


def metrics(actual, predicted, prior):
    categories = np.asarray([item.categorical[0] for item in actual])
    guesses = np.asarray([item.categorical[0] for item in predicted])
    confusion = np.zeros((3, 3), dtype=int)
    np.add.at(confusion, (categories, guesses), 1)
    release = np.asarray([item.continuous[8] for item in actual])
    estimate = np.asarray([item.continuous[8] for item in predicted])
    error = estimate - release
    return {"count": len(actual), "unison_counts_1_2_4": np.bincount(categories, minlength=3).tolist(),
            "unison_confusion_actual_rows_predicted_columns": confusion.tolist(), "unison_accuracy": float((categories == guesses).mean()),
            "release_mae": float(np.abs(error).mean()), "release_mse": float(np.square(error).mean()),
            "release_bias": float(error.mean()), "release_correlation": correlation(release, estimate),
            "release_target_std": float(release.std()), "release_prediction_std": float(estimate.std()),
            "release_prediction_extreme_fraction": float(((estimate < .05) | (estimate > .95)).mean()),
            "training_median_release_mae": float(np.abs(release - prior).mean()),
            "release_mae_by_target_quartile": [float(np.abs(error[(release >= lo) & (release < hi)]).mean())
                                               if ((release >= lo) & (release < hi)).any() else None
                                               for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.000001))]}


def main():
    started = perf_counter()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    reports = CODE / "research/training/reports"
    frozen_path = reports / "data/fresh_validation_plan.json"
    frozen = json.loads(frozen_path.read_text())
    pools = {name: PilotPool.load(ROOT / item["path"]) for name, item in frozen["pools"].items()}
    targets = {entry.state_id: entry.target for entry in pools["expanded"].entries}
    panels = {}
    input_artifacts = {str(frozen_path.relative_to(CODE)): sha(frozen_path)}
    for name, directory, evidence in (
        ("original21", "state-coverage-replication-validation-20261014", "state_coverage_replication_summary.json"),
        ("fresh23", "fresh-validation-20261014", "fresh_validation_summary.json"),
    ):
        folder = ROOT / "research/training/outputs" / directory
        result = json.loads((folder / "summary.json").read_text())
        proof = json.loads((reports / "data" / evidence).read_text())
        assert sha(folder / "summary.json") == proof["evaluation_summary_sha256"]
        input_artifacts[str((folder / "summary.json").relative_to(ROOT))] = sha(folder / "summary.json")
        for row in json.loads((folder / "progress.json").read_text())["completed_cases"]:
            assert sha(folder / f"case-{row['case']:03}.json") == row["case_sha256"]
            for filename, digest in row["predictions"].items():
                assert sha(folder / filename) == digest
        assert all(pools["expanded"].manifest.by_state[row["state_id"]].split == "validation" for row in result["cases"])
        panels[name] = (folder, result["cases"])
    assert not {row["state_id"] for row in panels["original21"][1]} & {row["state_id"] for row in panels["fresh23"][1]}
    results, emitted = {}, {}
    for name in frozen["candidate_order"]:
        artifact = frozen["candidates"][name]
        path = ROOT / artifact["path"]
        assert sha(path) == artifact["sha256"]
        pool = pools["expanded" if name.startswith("expanded") else "one"]
        saved, model, _, _ = load_evaluation_checkpoint(path, pool)
        assert retrieval_digest(saved["replay"]) == artifact["replay_digest"]
        actual = [PilotTarget.from_dict(item["target"]) for item in saved["replay"]]
        with torch.no_grad():
            raw = model(batch_features(saved["replay"]))
            predicted = predicted_targets(raw, pool.codec)
        prior = float(np.median([item.continuous[8] for item in actual]))
        head_losses = {"unison_ce": float(torch.nn.functional.cross_entropy(raw["categorical"][0], torch.tensor([item.categorical[0] for item in actual]))),
                       "release_mse_contribution_to_total_loss": float(sum((guess.continuous[8] - target.continuous[8]) ** 2 for target, guess in zip(actual, predicted, strict=True)) / sum(sum(pool.codec.continuous_mask(item)) for item in actual))}
        results[name] = {"checkpoint_sha256": artifact["sha256"], "training_median_release": prior,
                         "training": metrics(actual, predicted, prior), "training_head_losses": head_losses}
        emitted[name] = {}
        for panel, (folder, rows) in panels.items():
            actual = [targets[row["state_id"]] for row in rows]
            predicted = [pool.codec.encode((folder / f"prediction-{name}-{row['case']:03}.vital").read_bytes()) for row in rows]
            results[name][panel] = metrics(actual, predicted, prior)
            emitted[name][panel] = predicted
        print(json.dumps({"candidate": name, "train": results[name]["training"], "fresh23": results[name]["fresh23"]}), flush=True)
    relationships = {}
    for seed in (1, 2):
        one, expanded = f"one_seed{seed}_16000", f"expanded_seed{seed}_16000"
        relationships[str(seed)] = {}
        for panel, (_, rows) in panels.items():
            entries = []
            for index, row in enumerate(rows):
                target = targets[row["state_id"]]
                before, after = emitted[one][panel][index], emitted[expanded][panel][index]
                audio = {name: float(np.mean([row["methods"][name][role]["mrstft"] for role in ("reference", "transfer")])) for name in (one, expanded)}
                entries.append({"state_id": row["state_id"], "unison_became_wrong": before.categorical[0] == target.categorical[0] and after.categorical[0] != target.categorical[0],
                                "unison_became_correct": before.categorical[0] != target.categorical[0] and after.categorical[0] == target.categorical[0],
                                "release_error_delta": abs(after.continuous[8] - target.continuous[8]) - abs(before.continuous[8] - target.continuous[8]),
                                "audio_delta": audio[expanded] - audio[one]})
            worsened = [row for row in entries if row["unison_became_wrong"]]
            release_worse = [row for row in entries if row["release_error_delta"] > 0]
            relationships[str(seed)][panel] = {"cases": entries, "new_unison_errors": len(worsened),
                                               "new_unison_errors_with_audio_improvement": sum(row["audio_delta"] < 0 for row in worsened),
                                               "release_worsened_cases": len(release_worse),
                                               "release_worsened_with_audio_improvement": sum(row["audio_delta"] < 0 for row in release_worse),
                                               "release_error_delta_vs_audio_delta_correlation": correlation([row["release_error_delta"] for row in entries], [row["audio_delta"] for row in entries])}
    result = {"analysis": "unison_release_existing_artifacts_v1", "exploratory": True, "input_artifacts": input_artifacts,
              "candidates": results, "paired_error_audio_relationships": relationships, "training_updates": 0, "new_renders": 0,
              "test_split_evaluated": False, "promotion": "none", "elapsed_seconds": perf_counter() - started,
              "release_units": "normalized stored Vital control units; not seconds", "control_names": [row[0] for row in CONTINUOUS]}
    (reports / "data/unison_release_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"elapsed_seconds": result["elapsed_seconds"], "new_renders": 0, "training_updates": 0}))


if __name__ == "__main__":
    main()
