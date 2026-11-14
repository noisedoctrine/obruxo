from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from .core import (MODEL_REGISTRY, _rank_metrics, canonical_json, cluster_bootstrap, digest, mixed_gallery_midi_id,
                   paired_cluster_bootstrap, random_baseline)
from .encoders import list_clip_cache
from .dataset import validate_dataset_manifests
from .rendering import RENDER_POLICY_ID, render_plan_identity


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _primary_request_ids(episode, candidates, pair_to_request):
    candidate_set = candidates[episode["candidate_set_id"]]
    requests = [pair_to_request.get((episode["target_state_id"], episode["query_midi_id"]))]
    for state in candidate_set["state_ids"]:
        midi_id = (episode["gallery_midi_id"] if episode["condition"] == "cross_midi_common_gallery"
                   else mixed_gallery_midi_id(episode, state, candidates))
        requests.append(pair_to_request.get((state, midi_id)))
    if any(request is None for request in requests):
        raise ValueError("episode is missing a frozen target-query or gallery render request")
    return sorted(set(requests))


def _render_receipts(dataset_dir: Path, plan: list[dict]) -> dict[str, dict]:
    plan_by_id = {row["request_id"]: row for row in plan}
    latest = {}
    for receipt in _jsonl(dataset_dir / "render_receipts.private.jsonl"):
        row = plan_by_id.get(receipt["request_id"])
        if row is None or receipt.get("render_policy_id") != RENDER_POLICY_ID:
            raise ValueError("render receipt refers to an unknown request or policy")
        if receipt.get("plan_identity") != render_plan_identity(row):
            raise ValueError("render receipt plan identity differs from the frozen request manifest")
        latest[receipt["request_id"]] = receipt
    missing = set(plan_by_id) - set(latest)
    if missing:
        raise ValueError(f"render stage is incomplete: {len(missing):,} planned requests have no terminal receipt")
    return latest


def _freeze_qa_qualified_episodes(dataset_dir: Path, episodes: list[dict], candidates: dict,
                                  pair_to_request: dict, receipts: dict[str, dict]) -> tuple[list[dict], dict]:
    coverage = defaultdict(lambda: {"planned_episodes": 0, "qa_qualified_episodes": 0,
                                    "excluded_episodes": 0, "missing_required_request_status_counts": {}})
    qualified, qualification_receipts = [], {}
    for episode in episodes:
        condition = episode["condition"]
        record = coverage[condition]
        record["planned_episodes"] += 1
        required = _primary_request_ids(episode, candidates, pair_to_request)
        missing = []
        for request_id in required:
            receipt = receipts.get(request_id)
            if not receipt or not receipt.get("accepted"):
                missing.append((request_id, (receipt or {}).get("qa_status", "render_not_completed")))
            qualification_receipts[request_id] = (None if not receipt else {
                key: receipt.get(key) for key in ("plan_identity", "audio_sha256", "accepted", "qa_status")})
        if missing:
            record["excluded_episodes"] += 1
            counts = Counter(status for _, status in missing)
            merged = Counter(record["missing_required_request_status_counts"])
            merged.update(counts)
            record["missing_required_request_status_counts"] = dict(sorted(merged.items()))
            continue
        qualified.append(episode)
        record["qa_qualified_episodes"] += 1
    coverage = {condition: value for condition, value in sorted(coverage.items())}
    qualified.sort(key=lambda row: row["episode_id"])
    qualification_id = digest({"policy": "qa_qualified_primary_episode_set_v1", "render_policy_id": RENDER_POLICY_ID,
                               "episode_ids": [row["episode_id"] for row in qualified], "coverage": coverage,
                               "required_render_receipts": qualification_receipts})
    manifest_path = dataset_dir / "episodes_qa_qualified.private.jsonl"
    summary_path = dataset_dir / "qa_episode_coverage.private.json"
    summary = {"policy": "qa_qualified_primary_episode_set_v1", "qualification_id": qualification_id,
               "same_midi_and_near_neighbor_diagnostics_gate_primary_episodes": False,
               "coverage_by_condition": coverage}
    if manifest_path.exists() or summary_path.exists():
        if not manifest_path.is_file() or not summary_path.is_file():
            raise ValueError("frozen QA qualification artifacts are incomplete")
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        previous_episodes = _jsonl(manifest_path)
        if (previous.get("qualification_id") != qualification_id or
                [row["episode_id"] for row in previous_episodes] != [row["episode_id"] for row in qualified] or
                digest(previous_episodes) != digest(qualified)):
            raise ValueError("QA-qualified episode set changed; use a fresh dataset output to freeze the new render coverage")
    else:
        with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in qualified:
                stream.write(canonical_json(row) + "\n")
        summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return qualified, summary


def _cosine_matrix(left, right, np):
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True).T
    if (left_norm == 0).any() or (right_norm == 0).any():
        raise ValueError("zero-norm cached embedding cannot be compared")
    return (left @ right.T) / (left_norm * right_norm)


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
        same_states = [state for state in states if (request := pair_to_request.get((state, query_midi))) in features]
        same_matrix = np.stack([features[pair_to_request[(state, query_midi)]][variant] for state in same_states]).astype(np.float32)
        if condition == "cross_midi_common_gallery":
            gallery_id = grouped_episodes[0]["gallery_midi_id"]
            gallery_matrix = np.stack([features[pair_to_request[(state, gallery_id)]][variant] for state in states]).astype(np.float32)
            query_episodes = [episode for episode in grouped_episodes
                              if pair_to_request.get((episode["target_state_id"], query_midi)) in features]
            query_matrix = np.stack([features[pair_to_request[(episode["target_state_id"], query_midi)]][variant]
                                     for episode in query_episodes]).astype(np.float32)
            score_matrix = _cosine_matrix(query_matrix, gallery_matrix, np)
            same_score_matrix = _cosine_matrix(query_matrix, same_matrix, np)
            for target_index, episode in enumerate(query_episodes):
                same_by_state = {state: float(same_score_matrix[target_index, index]) for index, state in enumerate(same_states)}
                row = _row_metrics(episode, score_matrix[target_index], states, candidate_set, same_by_state)
                rows.append(row)
        else:
            for episode in grouped_episodes:
                query_request = pair_to_request[(episode["target_state_id"], query_midi)]
                query = features[query_request][variant]
                gallery_requests = [pair_to_request[(state, mixed_gallery_midi_id(episode, state, candidates))] for state in states]
                gallery_matrix = np.stack([features[request][variant] for request in gallery_requests]).astype(np.float32)
                scores = _cosine_matrix(query.reshape(1, -1), gallery_matrix, np)[0]
                same_by_state = {}
                if same_states:
                    same_scores = _cosine_matrix(query.reshape(1, -1), same_matrix, np)[0]
                    same_by_state = {state: float(same_scores[index]) for index, state in enumerate(same_states)}
                rows.append(_row_metrics(episode, scores, states, candidate_set, same_by_state))
    return rows


def _row_metrics(episode, scores, state_ids, candidate_set, same_midi_scores):
    target_index = state_ids.index(episode["target_state_id"])
    rank = _rank_metrics(list(zip(state_ids, scores.tolist(), strict=True)), episode["target_state_id"])
    families = _family_metrics(scores.tolist(), state_ids, candidate_set, episode["target_family_id"])
    target_cosine = float(scores[target_index])
    negatives = [score for state, score in same_midi_scores.items() if state != episode["target_state_id"]]
    near_ids = candidate_set.get("known_near_neighbor_state_ids_by_state", {}).get(episode["target_state_id"], [])
    near_negatives = [same_midi_scores[state] for state in near_ids if state in same_midi_scores]
    row = {**episode, **rank, "target_cosine": float(scores[target_index]),
           "family_recall_at_1": families["recall_at_1"], "family_recall_at_5": families["recall_at_5"],
           "family_mrr": families["mrr"], "same_midi_negative_max_cosine": max(negatives, default=None),
           "same_midi_negative_candidate_count": len(negatives),
           "same_midi_negative_win": (float(target_cosine > max(negatives)) if negatives else None),
           "known_near_neighbor_ids": near_ids,
           "known_near_neighbor_negative_candidate_count": len(near_negatives),
           "known_near_neighbor_negative_max_cosine": max(near_negatives, default=None),
           "known_near_neighbor_negative_win": (float(target_cosine > max(near_negatives)) if near_negatives else None),
           "gallery_size": len(state_ids)}
    return row


def _summary(rows):
    metrics = ("recall_at_1", "recall_at_5", "mrr", "family_recall_at_1")
    result = {key: (sum(row[key] for row in rows) / len(rows) if rows else None) for key in metrics}
    result["episodes"] = len(rows)
    result["gallery_size"] = sorted({row["gallery_size"] for row in rows})
    diagnostic_metrics = ("same_midi_negative_win", "known_near_neighbor_negative_win")
    for key in diagnostic_metrics:
        observed = [row for row in rows if row[key] is not None]
        result[f"{key}_rate"] = sum(row[key] for row in observed) / len(observed) if observed else None
        result[f"{key}_episodes"] = len(observed)
        result[f"{key}_candidates"] = sum(row[f"{key.replace('_win', '_candidate_count')}"] for row in rows)
    result["confidence_intervals"] = {key: cluster_bootstrap([(row["target_family_id"], row[key]) for row in rows])
                                      for key in metrics if rows}
    for key in diagnostic_metrics:
        observed = [row for row in rows if row[key] is not None]
        if observed:
            result["confidence_intervals"][key] = cluster_bootstrap([(row["target_family_id"], row[key]) for row in observed])
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
    all_plan = _jsonl(dataset_dir / "render_plan.private.jsonl")
    plan = [row for row in all_plan if row.get("source_kind") == "newly_rendered_original_preset"]
    performances = [row for row in _jsonl(dataset_dir / "performances.private.jsonl")
                    if row.get("split") == dataset_summary["source_split"]]
    pair_to_request = {(row["state_id"], row["midi_id"]): row["request_id"] for row in plan}
    receipts = _render_receipts(dataset_dir, all_plan)
    qa_episodes, qa_summary = _freeze_qa_qualified_episodes(dataset_dir, episodes, candidates, pair_to_request, receipts)
    accepted_request_ids = {request_id for request_id, receipt in receipts.items() if receipt.get("accepted")}
    model_features = {}
    for model_dir in sorted((dataset_dir / "features").glob("*")) if (dataset_dir / "features").is_dir() else []:
        if not model_dir.is_dir() or not (model_dir / "feature_manifest.private.jsonl").is_file():
            continue
        model_name = model_dir.name
        feature_manifest = _jsonl(model_dir / "feature_manifest.private.jsonl")
        feature_ids = {row["feature_id"] for row in feature_manifest}
        if len(feature_ids) != len(feature_manifest):
            raise ValueError("feature manifest contains duplicate cache identities")
        if any(row["request_id"] not in accepted_request_ids for row in feature_manifest):
            raise ValueError("feature manifest contains a render that is not QA-accepted by the current receipt ledger")
        summary = json.loads((model_dir / "extraction.latest.json").read_text(encoding="utf-8"))
        loaded = list_clip_cache(model_dir, model_name, feature_ids=feature_ids,
                                 expected_checkpoint_sha256=summary["checkpoint_sha256"])
        if set(loaded) != {row["request_id"] for row in feature_manifest}:
            raise ValueError("feature cache entries and extraction manifest disagree")
        variants = sorted({variant for row in loaded.values() for variant in row})
        for variant in variants:
            model_features[f"{model_name}:{variant}"] = (model_name, variant, loaded)
    conditions = ("cross_midi_common_gallery", "cross_midi_mixed_gallery")
    eligible_by_arm = {}
    for arm, (_, _, features) in model_features.items():
        eligible_by_arm[arm] = {}
        for condition in conditions:
            eligible_by_arm[arm][condition] = {
                episode["episode_id"] for episode in qa_episodes if episode["condition"] == condition and
                all(request in features for request in _primary_request_ids(episode, candidates, pair_to_request))}
    common_by_condition = {}
    for condition in conditions:
        common = {row["episode_id"] for row in qa_episodes if row["condition"] == condition}
        for valid_by_condition in eligible_by_arm.values():
            common &= valid_by_condition[condition]
        common_by_condition[condition] = common if model_features else set()
    coverage_by_condition = {}
    for condition in conditions:
        qa_counts = qa_summary["coverage_by_condition"].get(condition, {"planned_episodes": 0, "qa_qualified_episodes": 0,
                                                                          "excluded_episodes": 0, "missing_required_request_status_counts": {}})
        coverage_by_condition[condition] = {**qa_counts,
            "feature_qualified_episodes_by_arm": {arm: len(rows[condition]) for arm, rows in sorted(eligible_by_arm.items())},
            "common_feature_qualified_episodes_across_arms": len(common_by_condition[condition])}
    results = {}
    all_result_rows = {}
    for arm, (model_name, variant, features) in model_features.items():
        results[arm], all_result_rows[arm] = {}, {}
        for condition in conditions:
            selected = [episode for episode in qa_episodes if episode["episode_id"] in common_by_condition[condition]]
            rows = _evaluate_condition(selected, candidates, pair_to_request, features, variant, condition)
            results[arm][condition] = _summary(rows)
            results[arm][condition]["strata"] = _strata(rows)
            all_result_rows[arm][condition] = rows
    arm_names = sorted(results)
    differences = {}
    for left_index, left_name in enumerate(arm_names):
        for right_name in arm_names[left_index + 1:]:
            differences[f"{left_name} - {right_name}"] = {}
            for condition in conditions:
                left_rows = all_result_rows[left_name][condition]
                right_rows = all_result_rows[right_name][condition]
                if left_rows and right_rows:
                    differences[f"{left_name} - {right_name}"][condition] = {
                        metric: paired_cluster_bootstrap(left_rows, right_rows, metric=metric)
                        for metric in ("recall_at_1", "recall_at_5", "mrr")}
    paired_episodes = [episode for condition in conditions for episode in qa_episodes
                       if episode["condition"] == condition and episode["episode_id"] in common_by_condition[condition]]
    baseline = midi_shortcut_baseline(paired_episodes, candidates, performances) if paired_episodes else {}
    result = {"version": 1, "policy": "frozen_encoder_real_vital_cross_performance_v1",
              "qa_qualification_id": qa_summary["qualification_id"],
              "common_episode_count_across_available_arms": len(paired_episodes),
              "coverage_by_condition": coverage_by_condition,
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
