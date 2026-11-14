from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time

from .core import (MODEL_REGISTRY, _rank_metrics, cluster_bootstrap, mixed_gallery_midi_id,
                   paired_cluster_bootstrap, random_baseline)
from .encoders import list_clip_cache
from .dataset import validate_dataset_manifests


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _cosine_matrix(left, right, np):
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True).T
    if (left_norm == 0).any() or (right_norm == 0).any():
        raise ValueError("zero-norm cached embedding cannot be compared")
    return (left @ right.T) / (left_norm * right_norm)


def _all_scores(episode, states, pair_to_request, features, variant, candidates, condition):
    target, query_midi = episode["target_state_id"], episode["query_midi_id"]
    query_request = pair_to_request[(target, query_midi)]
    query = features[query_request][variant]
    gallery_ids = []
    for state in states:
        if condition == "cross_midi_common_gallery":
            midi_id = episode["gallery_midi_id"]
        else:
            midi_id = mixed_gallery_midi_id(episode, state, candidates)
        gallery_ids.append(pair_to_request[(state, midi_id)])
    return query, [features[request][variant] for request in gallery_ids]


def _family_metrics(scores, state_ids, candidate_set, target_family):
    grouped = defaultdict(list)
    for state_id, score in zip(state_ids, scores, strict=True):
        family = candidate_set.get("family_by_state", {}).get(state_id, state_id)
        grouped[family].append(float(score))
    return _rank_metrics([(family, max(values)) for family, values in grouped.items()], target_family)


def _evaluate_condition(episodes, candidates, pair_to_request, features, variant, condition):
    import numpy as np

    rows = []
    by_query = defaultdict(list)
    for episode in episodes:
        if episode["condition"] == condition:
            by_query[(episode["candidate_set_id"], episode["query_midi_id"])].append(episode)
    for (candidate_set_id, query_midi), grouped_episodes in sorted(by_query.items()):
        candidate_set = candidates[candidate_set_id]
        states = candidate_set["state_ids"]
        same_request_ids = [pair_to_request[(state, query_midi)] for state in states]
        same_matrix = np.stack([features[request][variant] for request in same_request_ids]).astype(np.float32)
        same_scores = _cosine_matrix(same_matrix, same_matrix, np)
        if condition == "cross_midi_common_gallery":
            gallery_id = grouped_episodes[0]["gallery_midi_id"]
            gallery_matrix = np.stack([features[pair_to_request[(state, gallery_id)]][variant] for state in states]).astype(np.float32)
            score_matrix = _cosine_matrix(same_matrix, gallery_matrix, np)
            episode_by_target = {row["target_state_id"]: row for row in grouped_episodes}
            for target_index, state_id in enumerate(states):
                episode = episode_by_target[state_id]
                row = _row_metrics(episode, score_matrix[target_index], same_scores[target_index], states, candidate_set)
                rows.append(row)
        else:
            for episode in grouped_episodes:
                query, gallery = _all_scores(episode, states, pair_to_request, features, variant, candidates, condition)
                gallery_matrix = np.stack(gallery).astype(np.float32)
                scores = _cosine_matrix(query.reshape(1, -1), gallery_matrix, np)[0]
                target_index = states.index(episode["target_state_id"])
                rows.append(_row_metrics(episode, scores, same_scores[target_index], states, candidate_set))
    return rows


def _row_metrics(episode, scores, same_midi_scores, state_ids, candidate_set):
    target_index = state_ids.index(episode["target_state_id"])
    rank = _rank_metrics(list(zip(state_ids, scores.tolist(), strict=True)), episode["target_state_id"])
    families = _family_metrics(scores.tolist(), state_ids, candidate_set, episode["target_family_id"])
    negatives = [float(score) for index, score in enumerate(same_midi_scores) if index != target_index]
    row = {**episode, **rank, "target_cosine": float(scores[target_index]),
           "family_recall_at_1": families["recall_at_1"], "family_recall_at_5": families["recall_at_5"],
           "family_mrr": families["mrr"], "same_midi_negative_max_cosine": max(negatives, default=None),
           "same_midi_negative_win": float(bool(negatives) and scores[target_index] > max(negatives)),
           "gallery_size": len(state_ids)}
    return row


def _summary(rows):
    metrics = ("recall_at_1", "recall_at_5", "mrr", "family_recall_at_1", "same_midi_negative_win")
    result = {key: (sum(row[key] for row in rows) / len(rows) if rows else None) for key in metrics}
    result["episodes"] = len(rows)
    result["gallery_size"] = sorted({row["gallery_size"] for row in rows})
    result["confidence_intervals"] = {key: cluster_bootstrap([(row["target_family_id"], row[key]) for row in rows])
                                      for key in metrics if rows}
    return result


def _strata(rows):
    axes = {key for row in rows for key in row.get("strata", {})
            if key in {"query_polyphony", "query_articulation", "query_expression", "query_duration_band",
                       "pitch_register_gap_band", "duration_gap_band", "query_note_density_band"}}
    report = {}
    for axis in sorted(axes):
        groups = defaultdict(list)
        for row in rows:
            groups[row.get("strata", {}).get(axis, "unknown")].append(row)
        report[axis] = {label: {"episodes": len(items), "recall_at_1": sum(row["recall_at_1"] for row in items) / len(items),
                                "recall_at_5": sum(row["recall_at_5"] for row in items) / len(items),
                                "mrr": sum(row["mrr"] for row in items) / len(items)}
                        for label, items in sorted(groups.items())}
    return report


def midi_shortcut_baseline(episodes, candidates, performances):
    import numpy as np

    feature_names = ("pitch_min", "pitch_max", "pitch_median", "note_count", "max_polyphony", "duration_seconds",
                     "velocity_mean", "notes_per_second", "note_duration_median_seconds", "tempo_bpm")
    feature_values = {}
    for row in performances:
        stats = row["stats"]
        values = [stats.get(name, 0.0) for name in feature_names]
        values[-1] = sum(stats.get("tempo_bpm", [0.0])) / max(len(stats.get("tempo_bpm", [])), 1)
        feature_values[row["midi_id"]] = np.asarray(values, dtype=np.float64)
    matrix = np.stack(list(feature_values.values()))
    means, scales = matrix.mean(axis=0), matrix.std(axis=0)
    scales[scales == 0] = 1.0
    normalized = {key: (value - means) / scales for key, value in feature_values.items()}
    rows_by_condition = defaultdict(list)
    for episode in episodes:
        candidate_set = candidates[episode["candidate_set_id"]]
        states = candidate_set["state_ids"]
        query = normalized[episode["query_midi_id"]]
        if episode["condition"] == "cross_midi_common_gallery":
            scores = [0.0] * len(states)
        else:
            scores = []
            for state in states:
                midi_id = mixed_gallery_midi_id(episode, state, candidates)
                gallery = normalized[midi_id]
                denom = float(np.linalg.norm(query) * np.linalg.norm(gallery))
                scores.append(float(query @ gallery / denom) if denom else 0.0)
        rows_by_condition[episode["condition"]].append({**episode,
            **_rank_metrics(list(zip(states, scores, strict=True)), episode["target_state_id"]),
            "target_family_id": episode["target_family_id"], "gallery_size": len(states)})
    output = {}
    for condition, rows in rows_by_condition.items():
        output[condition] = {"episodes": len(rows), "recall_at_1": sum(row["recall_at_1"] for row in rows) / len(rows),
                             "recall_at_5": sum(row["recall_at_5"] for row in rows) / len(rows),
                             "mrr": sum(row["mrr"] for row in rows) / len(rows),
                             "confidence_intervals": {key: cluster_bootstrap([(row["target_family_id"], row[key]) for row in rows])
                                                       for key in ("recall_at_1", "recall_at_5", "mrr")}}
        output[condition]["random"] = random_baseline([len(candidates[row["candidate_set_id"]]["state_ids"]) for row in rows])
    output["random"] = {condition: row["random"] for condition, row in output.items() if condition != "random"}
    for condition in [key for key in output if key != "random"]:
        del output[condition]["random"]
    return output


def evaluate_dataset(dataset_dir: Path) -> dict:
    dataset_dir = dataset_dir.resolve(strict=True)
    dataset_summary = json.loads((dataset_dir / "dataset_summary.private.json").read_text(encoding="utf-8"))
    validate_dataset_manifests(dataset_dir, dataset_summary)
    candidates = json.loads((dataset_dir / "candidate_sets.private.json").read_text(encoding="utf-8"))
    episodes = _jsonl(dataset_dir / "episodes.private.jsonl")
    plan = _jsonl(dataset_dir / "render_plan.private.jsonl")
    performances = _jsonl(dataset_dir / "performances.private.jsonl")
    pair_to_request = {(row["state_id"], row["midi_id"]): row["request_id"] for row in plan}
    model_features, model_summaries = {}, {}
    for model_dir in sorted((dataset_dir / "features").glob("*")) if (dataset_dir / "features").is_dir() else []:
        if not model_dir.is_dir() or not (model_dir / "feature_manifest.private.jsonl").is_file():
            continue
        model_name = model_dir.name
        feature_manifest = _jsonl(model_dir / "feature_manifest.private.jsonl")
        feature_ids = {row["feature_id"] for row in feature_manifest}
        if len(feature_ids) != len(feature_manifest):
            raise ValueError("feature manifest contains duplicate cache identities")
        summary = json.loads((model_dir / "extraction.latest.json").read_text(encoding="utf-8"))
        loaded = list_clip_cache(model_dir, model_name, feature_ids=feature_ids,
                                 expected_checkpoint_sha256=summary["checkpoint_sha256"])
        if set(loaded) != {row["request_id"] for row in feature_manifest}:
            raise ValueError("feature cache entries and extraction manifest disagree")
        variants = sorted({variant for row in loaded.values() for variant in row})
        for variant in variants:
            model_features[f"{model_name}:{variant}"] = (model_name, variant, loaded)
            model_summaries[f"{model_name}:{variant}"] = summary
    common_episode_ids = set(row["episode_id"] for row in episodes)
    for arm, (_, _, features) in model_features.items():
        valid = set()
        for episode in episodes:
            candidate_set = candidates[episode["candidate_set_id"]]
            required = [pair_to_request.get((episode["target_state_id"], episode["query_midi_id"]))]
            for state in candidate_set["state_ids"]:
                if episode["condition"] == "cross_midi_common_gallery":
                    gallery_midi = episode["gallery_midi_id"]
                else:
                    gallery_midi = mixed_gallery_midi_id(episode, state, candidates)
                required.append(pair_to_request.get((state, gallery_midi)))
                required.append(pair_to_request.get((state, episode["query_midi_id"])))
            if all(request and request in features for request in required):
                valid.add(episode["episode_id"])
        common_episode_ids &= valid
    paired_episodes = [row for row in episodes if row["episode_id"] in common_episode_ids]
    results = {}
    all_result_rows = {}
    for arm, (model_name, variant, features) in model_features.items():
        results[arm], all_result_rows[arm] = {}, {}
        for condition in ("cross_midi_common_gallery", "cross_midi_mixed_gallery"):
            rows = _evaluate_condition(paired_episodes, candidates, pair_to_request, features, variant, condition)
            results[arm][condition] = _summary(rows)
            results[arm][condition]["strata"] = _strata(rows)
            all_result_rows[arm][condition] = rows
    arm_names = sorted(results)
    differences = {}
    for left_index, left_name in enumerate(arm_names):
        for right_name in arm_names[left_index + 1:]:
            differences[f"{left_name} - {right_name}"] = {}
            for condition in ("cross_midi_common_gallery", "cross_midi_mixed_gallery"):
                left_rows = all_result_rows[left_name][condition]
                right_rows = all_result_rows[right_name][condition]
                if left_rows and right_rows:
                    differences[f"{left_name} - {right_name}"][condition] = {
                        metric: paired_cluster_bootstrap(left_rows, right_rows, metric=metric)
                        for metric in ("recall_at_1", "recall_at_5", "mrr")}
    baseline = midi_shortcut_baseline(paired_episodes, candidates, performances) if paired_episodes else {}
    result = {"version": 1, "policy": "frozen_encoder_real_vital_cross_performance_v1",
              "common_episode_count_across_available_arms": len(paired_episodes),
              "available_arms": arm_names, "unavailable_encoders": [name for name in MODEL_REGISTRY if not any(arm.startswith(name + ":") for arm in arm_names)],
              "model_results": results, "paired_family_bootstrap_differences": differences,
              "midi_only_and_random_baselines": baseline,
              "no_source_audio_used": True, "held_out_reserved_sources_used": False,
              "primary_metric": "Recall@1", "gallery_candidate_set_is_identical_across_arms": True}
    (dataset_dir / "evaluation.private.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with (dataset_dir / "episode_metrics.private.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for arm in arm_names:
            for condition in all_result_rows[arm]:
                for row in all_result_rows[arm][condition]:
                    stream.write(json.dumps({"arm": arm, **row}, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
    return result
