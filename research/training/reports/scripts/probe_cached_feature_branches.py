"""Fixed, train-only leave-one-state-out retrieval probe; no optimizer or rendering."""
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
from obruxo_training.evaluate import retrieval_digest
from obruxo_training.pilot import PilotPool


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = perf_counter()
    plan_path = CODE / "research/training/reports/data/cached_head_probe_plan.json"
    plan = json.loads(plan_path.read_text())
    assert sha(Path(__file__)) == plan["probe_script_sha256"]
    torch.set_num_threads(4)
    results = {}
    for population, entry in plan["populations"].items():
        path = ROOT / entry["checkpoint"]["path"]
        assert sha(path) == entry["checkpoint"]["sha256"]
        saved = torch.load(path, map_location="cpu", weights_only=True)
        replay = saved["replay"]
        assert retrieval_digest(replay) == entry["checkpoint"]["replay_digest"]
        pool_path = ROOT / entry["pool"]["path"]
        assert sha(pool_path) == entry["pool"]["sha256"]
        pool = PilotPool.load(pool_path)
        ids = [item["state_id"] for item in replay]
        assert len(ids) == len(set(ids)) == entry["states"]
        assert all(pool.manifest.by_state[state_id].split == "train" for state_id in ids)
        categories = np.asarray([item["target"]["categorical"][0] for item in replay])
        release = np.asarray([item["target"]["continuous"][8] for item in replay])
        priors = {"unison": np.asarray([np.bincount(np.delete(categories, index), minlength=3).argmax() for index in range(len(ids))]),
                  "release": np.asarray([np.median(np.delete(release, index)) for index in range(len(ids))])}
        branches = {}
        for name in ("timbre", "pitch"):
            values = torch.stack([item["features"][name].flatten() for item in replay]).double()
            norm = values.square().mean(1)
            distance = (norm[:, None] + norm[None, :] - 2 * (values @ values.T) / values.shape[1]).clamp_min(0)
            distance.fill_diagonal_(float("inf"))
            # Check the efficient distance calculation against the direct definition.
            for left, right in ((0, 1), (2, 3), (len(ids) - 2, len(ids) - 1)):
                assert torch.isclose(distance[left, right], (values[left] - values[right]).square().mean(), rtol=1e-10, atol=1e-12)
            branches[name] = distance
        branches["combined"] = branches["timbre"] + branches["pitch"]
        arms = {}
        for name in plan["arms"]:
            nearest = branches[name].argmin(1).numpy()
            assert all(ids[left] != ids[right] for left, right in enumerate(nearest))
            predicted = categories[nearest]
            confusion = np.zeros((3, 3), dtype=int)
            np.add.at(confusion, (categories, predicted), 1)
            errors = np.abs(release - release[nearest])
            prior_errors = np.abs(release - priors["release"])
            arms[name] = {"unison_accuracy": float((categories == predicted).mean()),
                          "unison_confusion_actual_rows_predicted_columns": confusion.tolist(),
                          "release_mae": float(errors.mean()), "release_mae_minus_leave_one_out_median": float((errors - prior_errors).mean()),
                          "release_wins_against_median": int((errors < prior_errors).sum()),
                          "neighbor_state_ids": [ids[index] for index in nearest]}
        results[population] = {"states": len(ids), "state_ids": ids,
                               "leave_one_out_prior": {"unison_accuracy": float((categories == priors["unison"]).mean()),
                                                       "release_mae": float(np.abs(release - priors["release"]).mean())},
                               "arms": arms}
    result = {"plan_sha256": sha(plan_path), "populations": results, "optimizer_updates": 0, "native_renders": 0,
              "validation_or_test_used": False, "elapsed_seconds": perf_counter() - started,
              "interpretation": "Exploratory train-domain retrieval probe; not inverse-network retraining or native audio evaluation."}
    path = CODE / "research/training/reports/data/cached_head_probe_summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({name: {"prior": value["leave_one_out_prior"], "arms": {arm: {key: score[key] for key in ("unison_accuracy", "release_mae")} for arm, score in value["arms"].items()}}
                      for name, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
