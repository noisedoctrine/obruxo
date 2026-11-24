from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Sequence

from .core import (MODEL_REGISTRY, _rank_metrics, canonical_json, cluster_bootstrap, digest, mixed_gallery_midi_id,
                   paired_cluster_bootstrap, paired_performance_bootstrap, paired_two_way_cluster_bootstrap,
                   performance_cluster_bootstrap, performance_ranking_bootstrap, random_baseline, two_way_cluster_bootstrap)
from .encoders import list_clip_cache
from .dataset import validate_dataset_manifests
from .rendering import QA_POLICY_ID, RENDER_POLICY_ID, render_plan_identity


QUALIFICATION_POLICY_ID = "render_qualified_primary_episode_set_v3"


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
    stale = [row for row in latest.values() if row.get("qa_policy_id") != QA_POLICY_ID]
    if stale:
        raise ValueError("render receipts use a stale QA policy; rerun the resumable render stage to refresh them")
    missing = set(plan_by_id) - set(latest)
    if missing:
        raise ValueError(f"render stage is incomplete: {len(missing):,} planned requests have no terminal receipt")
    return latest


def _render_qualified_candidates(candidates: dict, pair_to_request: dict, receipts: dict[str, dict]):
    qualified, source_to_qualified, failures_by_source = {}, {}, {}
    for source_id, candidate in sorted(candidates.items()):
        gallery_midi_ids = candidate["gallery_midi_ids"]
        complete_states, failures = [], {}
        for state_id in candidate["state_ids"]:
            missing = []
            for midi_id in gallery_midi_ids:
                request_id = pair_to_request.get((state_id, midi_id))
                receipt = receipts.get(request_id) if request_id else None
                if not receipt or not receipt.get("accepted"):
                    missing.append({"midi_id": midi_id, "request_id": request_id,
                                    "qa_status": (receipt or {}).get("qa_status", "render_not_planned")})
            if missing:
                failures[state_id] = missing
            else:
                complete_states.append(state_id)
        failures_by_source[source_id] = failures
        if len(complete_states) < 2:
            source_to_qualified[source_id] = None
            continue
        state_set = set(complete_states)
        reduced = {"split": candidate["split"], "state_ids": complete_states,
                   "family_by_state": {state: candidate.get("family_by_state", {}).get(state, state)
                                       for state in complete_states},
                   "known_near_neighbor_state_ids_by_state": {
                       state: [neighbor for neighbor in candidate.get("known_near_neighbor_state_ids_by_state", {}).get(state, [])
                               if neighbor in state_set]
                       for state in complete_states},
                   "gallery_midi_ids": gallery_midi_ids}
        receipt_identities = {state: {midi: receipts[pair_to_request[(state, midi)]].get("audio_sha256")
                                      for midi in gallery_midi_ids} for state in complete_states}
        qualified_id = digest({"policy": "render_qualified_gallery_candidate_set_v1",
                               "qa_policy_id": QA_POLICY_ID, "source_candidate_set_id": source_id,
                               "candidate_set": reduced, "gallery_audio_sha256": receipt_identities})
        qualified[qualified_id] = reduced
        source_to_qualified[source_id] = qualified_id
    return qualified, source_to_qualified, failures_by_source


def _freeze_qa_qualified_episodes(dataset_dir: Path, episodes: list[dict], candidates: dict,
                                  pair_to_request: dict, receipts: dict[str, dict]) -> tuple[list[dict], dict, dict]:
    qualified_candidates, candidate_ids, candidate_failures = _render_qualified_candidates(
        candidates, pair_to_request, receipts)
    candidate_manifest = {"qa_policy_id": QA_POLICY_ID, "source_to_qualified_candidate_set_id": candidate_ids,
                          "candidate_sets": qualified_candidates}
    candidate_path = dataset_dir / "candidate_sets.render_qualified.private.json"
    if candidate_path.is_file() and json.loads(candidate_path.read_text(encoding="utf-8")) != candidate_manifest:
        raise ValueError("render-qualified candidate set changed; use a fresh dataset output after new renders complete")
    if not candidate_path.is_file():
        candidate_path.write_text(json.dumps(candidate_manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    coverage = defaultdict(lambda: {"planned_episodes": 0, "qa_qualified_episodes": 0,
                                    "excluded_episodes": 0, "excluded_gallery_target_episodes": 0,
                                    "excluded_query_render_episodes": 0, "missing_required_request_status_counts": {},
                                    "missing_gallery_candidate_status_counts": {},
                                    "original_gallery_candidate_counts": [], "render_qualified_gallery_candidate_counts": []})
    seen_candidate_sets = defaultdict(set)
    qualified, qualification_receipts = [], {}
    for episode in episodes:
        condition = episode["condition"]
        record = coverage[condition]
        record["planned_episodes"] += 1
        source_candidate_id = episode["candidate_set_id"]
        candidate_id = candidate_ids.get(source_candidate_id)
        candidate_set = qualified_candidates.get(candidate_id) if candidate_id else None
        if source_candidate_id not in seen_candidate_sets[condition]:
            original_count = len(candidates[source_candidate_id]["state_ids"])
            qualified_count = len(candidate_set["state_ids"]) if candidate_set else 0
            record["original_gallery_candidate_counts"].append(original_count)
            record["render_qualified_gallery_candidate_counts"].append(qualified_count)
            seen_candidate_sets[condition].add(source_candidate_id)
        if not candidate_set or episode["target_state_id"] not in candidate_set["state_ids"]:
            record["excluded_episodes"] += 1
            record["excluded_gallery_target_episodes"] += 1
            status_counts = Counter(item["qa_status"] for item in
                candidate_failures.get(source_candidate_id, {}).get(episode["target_state_id"], []))
            current = Counter(record["missing_gallery_candidate_status_counts"])
            current.update(status_counts)
            record["missing_gallery_candidate_status_counts"] = dict(sorted(current.items()))
            continue

        qualified_episode = {**episode, "candidate_set_id": candidate_id}
        query_request = pair_to_request.get((episode["target_state_id"], episode["query_midi_id"]))
        query_receipt = receipts.get(query_request) if query_request else None
        if not query_receipt or not query_receipt.get("accepted"):
            record["excluded_episodes"] += 1
            record["excluded_query_render_episodes"] += 1
            status = (query_receipt or {}).get("qa_status", "render_not_planned")
            current = Counter(record["missing_required_request_status_counts"])
            current.update([status])
            record["missing_required_request_status_counts"] = dict(sorted(current.items()))
            if query_request:
                qualification_receipts[query_request] = {key: query_receipt.get(key)
                    for key in ("plan_identity", "audio_sha256", "accepted", "qa_status")}
            continue

        required = _primary_request_ids(qualified_episode, qualified_candidates, pair_to_request)
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
        qualified.append(qualified_episode)
        record["qa_qualified_episodes"] += 1
    coverage = {condition: {**value,
        "original_gallery_candidate_count": max(value.pop("original_gallery_candidate_counts"), default=0),
        "render_qualified_gallery_candidate_count": max(value.pop("render_qualified_gallery_candidate_counts"), default=0)}
        for condition, value in sorted(coverage.items())}
    qualified.sort(key=lambda row: row["episode_id"])
    qualification_id = digest({"policy": QUALIFICATION_POLICY_ID, "render_policy_id": RENDER_POLICY_ID,
                               "episode_ids": [row["episode_id"] for row in qualified], "coverage": coverage,
                               "required_render_receipts": qualification_receipts})
    manifest_path = dataset_dir / "episodes_qa_qualified.private.jsonl"
    summary_path = dataset_dir / "qa_episode_coverage.private.json"
    summary = {"policy": QUALIFICATION_POLICY_ID, "qa_policy_id": QA_POLICY_ID,
               "qualification_id": qualification_id,
               "same_midi_and_near_neighbor_diagnostics_gate_primary_episodes": False,
               "coverage_by_condition": coverage}
    if manifest_path.exists() or summary_path.exists():
        if not manifest_path.is_file() or not summary_path.is_file():
            raise ValueError("frozen QA qualification artifacts are incomplete")
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        previous_episodes = _jsonl(manifest_path)
        if (previous.get("qa_policy_id") == QA_POLICY_ID and previous.get("policy") == QUALIFICATION_POLICY_ID and
                (previous.get("qualification_id") != qualification_id or
                [row["episode_id"] for row in previous_episodes] != [row["episode_id"] for row in qualified] or
                digest(previous_episodes) != digest(qualified))):
            raise ValueError("QA-qualified episode set changed; use a fresh dataset output to freeze the new render coverage")
        if previous.get("qa_policy_id") != QA_POLICY_ID or previous.get("policy") != QUALIFICATION_POLICY_ID:
            with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
                for row in qualified:
                    stream.write(canonical_json(row) + "\n")
            summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    else:
        with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in qualified:
                stream.write(canonical_json(row) + "\n")
        summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return qualified, summary, qualified_candidates


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


def _acoustic_hard_negative_candidates(dataset_dir, episodes, candidates, pair_to_request, receipts, *, top_k=5):
    import numpy as np
    from scipy.io import wavfile

    descriptor_cache = {}
    fft_size, hop_size = 4096, 2048
    frequency_edges = np.geomspace(45.0, 16_000.0, 65)

    def descriptor(request_id):
        if request_id in descriptor_cache:
            return descriptor_cache[request_id]
        receipt = receipts.get(request_id)
        if not receipt or not receipt.get("accepted"):
            return None
        sample_rate, audio = wavfile.read(dataset_dir / receipt["wav_relative_path"])
        audio = np.asarray(audio)
        if np.issubdtype(audio.dtype, np.integer):
            limits = np.iinfo(audio.dtype)
            audio = audio.astype(np.float32) / max(abs(limits.min), limits.max)
        else:
            audio = audio.astype(np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        elif audio.ndim != 1:
            raise ValueError(f"unsupported audio shape for acoustic hard-negative request {request_id}")
        if sample_rate < 32_000:
            raise ValueError(f"acoustic hard-negative audio is below 32 kHz for request {request_id}")
        if audio.size < fft_size:
            audio = np.pad(audio, (0, fft_size - audio.size))
        frames = np.lib.stride_tricks.sliding_window_view(audio, fft_size)[::hop_size]
        window = np.hanning(fft_size).astype(np.float32)
        frames = frames * window
        frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
        active = frames[frame_rms > max(float(frame_rms.max()) * 0.01, 1e-8)]
        if not len(active):
            descriptor_cache[request_id] = None
            return None
        power = np.square(np.abs(np.fft.rfft(active, axis=1)))
        frame_totals = power.sum(axis=1, keepdims=True)
        power = power / np.maximum(frame_totals, 1e-12)
        frequencies = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        bands = []
        for low, high in zip(frequency_edges[:-1], frequency_edges[1:], strict=True):
            selected = (frequencies >= low) & (frequencies < high)
            bands.append(float(power[:, selected].sum(axis=1).mean()) if selected.any() else 0.0)
        vector = np.log(np.asarray(bands, dtype=np.float32) + 1e-12)
        vector -= vector.mean()
        norm = np.linalg.norm(vector)
        descriptor_cache[request_id] = vector / norm if norm else None
        return descriptor_cache[request_id]

    selected_by_episode = {}
    for episode in episodes:
        if episode["condition"] not in {"cross_midi_common_gallery", "cross_midi_mixed_gallery"}:
            continue
        query_request = pair_to_request.get((episode["target_state_id"], episode["query_midi_id"]))
        query_descriptor = descriptor(query_request) if query_request else None
        if query_descriptor is None:
            selected_by_episode[episode["episode_id"]] = []
            continue
        ranked = []
        candidate_set = candidates[episode["candidate_set_id"]]
        for state_id in candidate_set["state_ids"]:
            if state_id == episode["target_state_id"]:
                continue
            request_id = pair_to_request.get((state_id, episode["query_midi_id"]))
            candidate_descriptor = descriptor(request_id) if request_id else None
            if candidate_descriptor is not None:
                ranked.append((state_id, float(query_descriptor @ candidate_descriptor)))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        selected_by_episode[episode["episode_id"]] = ranked[:top_k]
    return selected_by_episode


def _evaluate_condition(episodes, candidates, pair_to_request, features, variant, condition, acoustic_hard_negatives):
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
                row = _row_metrics(episode, score_matrix[target_index], states, candidate_set, same_by_state,
                                   acoustic_hard_negatives.get(episode["episode_id"], []))
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
                rows.append(_row_metrics(episode, scores, states, candidate_set, same_by_state,
                                         acoustic_hard_negatives.get(episode["episode_id"], [])))
    return rows


def _render_quality_bucket(status: str | None) -> str:
    if status == "accepted":
        return "clean"
    if status and status.startswith("accepted_with_"):
        return "warning"
    return "excluded_or_unknown"


def _attach_render_quality(rows, episodes, candidates, pair_to_request, receipts):
    episode_by_id = {episode["episode_id"]: episode for episode in episodes}
    for row in rows:
        episode = episode_by_id[row["episode_id"]]
        query_request = pair_to_request.get((episode["target_state_id"], episode["query_midi_id"]))
        query_status = receipts.get(query_request, {}).get("qa_status") if query_request else None
        candidate_set = candidates[episode["candidate_set_id"]]
        if episode["condition"] == "cross_midi_common_gallery":
            gallery_requests = [pair_to_request.get((state_id, episode["gallery_midi_id"]))
                                for state_id in candidate_set["state_ids"]]
        else:
            gallery_requests = [pair_to_request.get((state_id, mixed_gallery_midi_id(episode, state_id, candidates)))
                                for state_id in candidate_set["state_ids"]]
        gallery_statuses = [receipts.get(request_id, {}).get("qa_status") for request_id in gallery_requests]
        target_request = pair_to_request.get((episode["target_state_id"], episode["gallery_midi_id"]))
        if episode["condition"] != "cross_midi_common_gallery":
            target_request = pair_to_request.get((episode["target_state_id"],
                                                  mixed_gallery_midi_id(episode, episode["target_state_id"], candidates)))
        target_status = receipts.get(target_request, {}).get("qa_status") if target_request else None
        query_bucket = _render_quality_bucket(query_status)
        row["query_render_quality"] = query_bucket
        row["correct_match_gallery_render_quality"] = _render_quality_bucket(target_status)
        warning_count = sum(_render_quality_bucket(status) == "warning" for status in gallery_statuses)
        row["gallery_warning_candidate_count"] = warning_count
        row["gallery_candidate_count"] = len(gallery_statuses)
        row["gallery_warning_candidate_fraction"] = warning_count / len(gallery_statuses) if gallery_statuses else None
        row["gallery_unknown_candidate_count"] = sum(_render_quality_bucket(status) == "excluded_or_unknown"
                                                     for status in gallery_statuses)
        row["strata"] = {**row.get("strata", {}), "query_render_quality": query_bucket,
                          "correct_match_gallery_render_quality": row["correct_match_gallery_render_quality"]}
    return rows


def _row_metrics(episode, scores, state_ids, candidate_set, same_midi_scores, acoustic_hard_negative_candidates=()):
    target_index = state_ids.index(episode["target_state_id"])
    rank = _rank_metrics(list(zip(state_ids, scores.tolist(), strict=True)), episode["target_state_id"])
    families = _family_metrics(scores.tolist(), state_ids, candidate_set, episode["target_family_id"])
    target_cosine = float(scores[target_index])
    negatives = [score for state, score in same_midi_scores.items() if state != episode["target_state_id"]]
    near_ids = candidate_set.get("known_near_neighbor_state_ids_by_state", {}).get(episode["target_state_id"], [])
    near_negatives = [same_midi_scores[state] for state in near_ids if state in same_midi_scores]
    acoustic_ids = [state for state, _ in acoustic_hard_negative_candidates]
    acoustic_negatives = [same_midi_scores[state] for state in acoustic_ids if state in same_midi_scores]
    target_family_size = sum(family == episode["target_family_id"]
                             for family in candidate_set.get("family_by_state", {}).values())
    strata = {**episode.get("strata", {}), "source_preset_lineage_group_size":
              "single-state" if target_family_size <= 1 else "multi-state"}
    row = {**episode, **rank, "target_cosine": float(scores[target_index]),
           "family_recall_at_1": families["recall_at_1"], "family_recall_at_5": families["recall_at_5"],
           "family_mrr": families["mrr"], "same_midi_negative_max_cosine": max(negatives, default=None),
           "same_midi_negative_candidate_count": len(negatives),
           "same_midi_negative_win": (float(target_cosine > max(negatives)) if negatives else None),
           "known_near_neighbor_ids": near_ids,
           "known_near_neighbor_negative_candidate_count": len(near_negatives),
           "known_near_neighbor_negative_max_cosine": max(near_negatives, default=None),
           "known_near_neighbor_negative_win": (float(target_cosine > max(near_negatives)) if near_negatives else None),
           "acoustic_hard_negative_state_ids": [state for state in acoustic_ids if state in same_midi_scores],
           "acoustic_hard_negative_selector_similarities": [similarity for state, similarity in acoustic_hard_negative_candidates
                                                             if state in same_midi_scores],
           "acoustic_hard_negative_candidate_count": len(acoustic_negatives),
           "acoustic_hard_negative_max_cosine": max(acoustic_negatives, default=None),
           "acoustic_hard_negative_win": (float(target_cosine > max(acoustic_negatives)) if acoustic_negatives else None),
           "strata": strata,
           "gallery_size": len(state_ids)}
    return row


def _summary(rows):
    metrics = ("recall_at_1", "recall_at_5", "mrr", "family_recall_at_1")
    result = {key: (sum(row[key] for row in rows) / len(rows) if rows else None) for key in metrics}
    result["episodes"] = len(rows)
    result["gallery_size"] = sorted({row["gallery_size"] for row in rows})
    quality_rows = [row for row in rows if row.get("gallery_candidate_count", 0)]
    if quality_rows:
        result["gallery_render_quality"] = {
            "episodes": len(quality_rows),
            "mean_warning_candidates": sum(row["gallery_warning_candidate_count"] for row in quality_rows) / len(quality_rows),
            "mean_warning_candidate_fraction": sum(row["gallery_warning_candidate_fraction"] for row in quality_rows) / len(quality_rows),
            "episodes_with_any_warning": sum(row["gallery_warning_candidate_count"] > 0 for row in quality_rows),
            "episodes_with_all_candidates_warned": sum(row["gallery_warning_candidate_count"] == row["gallery_candidate_count"]
                                                       for row in quality_rows),
            "episodes_with_unknown_candidates": sum(row["gallery_unknown_candidate_count"] > 0 for row in quality_rows),
        }
    diagnostic_metrics = ("same_midi_negative_win", "known_near_neighbor_negative_win", "acoustic_hard_negative_win")
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
    result["diagnostic_performance_cluster_bootstrap"] = {
        key: performance_cluster_bootstrap([row for row in rows if row[key] is not None], metric=key)
        for key in diagnostic_metrics if any(row[key] is not None for row in rows)}
    return result


def _strata(rows):
    axes = {key for row in rows for key in row.get("strata", {})
            if key in {"query_polyphony", "query_articulation", "query_expression", "query_duration_band",
                       "pitch_register_gap_band", "duration_gap_band", "query_note_density_band",
                       "source_preset_lineage_group_size", "query_render_quality",
                       "correct_match_gallery_render_quality"}}
    report = {}
    for axis in sorted(axes):
        groups = defaultdict(list)
        for row in rows:
            groups[row.get("strata", {}).get(axis, "unknown")].append(row)
        report[axis] = {label: {"episodes": len(items), "recall_at_1": sum(row["recall_at_1"] for row in items) / len(items),
                                "recall_at_5": sum(row["recall_at_5"] for row in items) / len(items),
                                "mrr": sum(row["mrr"] for row in items) / len(items),
                                "performance_cluster_bootstrap": performance_cluster_bootstrap(
                                    items, metric="recall_at_1") if items else {},
                                "two_way_cluster_bootstrap": two_way_cluster_bootstrap(
                                    items, metric="recall_at_1") if items else {}}
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


def evaluate_dataset(dataset_dir: Path, *, models: Sequence[str] | None = None,
                     feature_root: Path | None = None, result_root: Path | None = None,
                     hard_negative_source: Path | None = None) -> dict:
    dataset_dir = dataset_dir.resolve(strict=True)
    feature_root = (feature_root or dataset_dir / "features").resolve(strict=True)
    result_root = (result_root or dataset_dir).resolve()
    selected_models = set(models) if models else None
    if selected_models is not None and not selected_models.issubset(MODEL_REGISTRY):
        raise ValueError(f"unknown encoder models: {sorted(selected_models - set(MODEL_REGISTRY))}")
    previous_result, previous_rows, previous_common_by_condition, previous_hard_negatives = {}, [], {}, {}
    if selected_models is not None:
        result_path, rows_path = result_root / "evaluation.private.json", result_root / "episode_metrics.private.jsonl"
        if not result_path.is_file() or not rows_path.is_file():
            raise ValueError("selected-model evaluation requires an existing complete evaluation to extend")
        previous_result = json.loads(result_path.read_text(encoding="utf-8"))
        previous_rows = _jsonl(rows_path)
        for row in previous_rows:
            state_ids = row.get("acoustic_hard_negative_state_ids")
            similarities = row.get("acoustic_hard_negative_selector_similarities")
            if state_ids is None or similarities is None:
                continue
            if len(state_ids) != len(similarities):
                raise ValueError("cached acoustic hard-negative states and selector scores disagree")
            selected = list(zip(state_ids, similarities, strict=True))
            episode_id = row["episode_id"]
            if episode_id in previous_hard_negatives and previous_hard_negatives[episode_id] != selected:
                raise ValueError(f"saved acoustic hard-negative selection differs across arms for {episode_id}")
            previous_hard_negatives[episode_id] = selected
        previous_arms = previous_result.get("available_arms", [])
        for condition in ("cross_midi_common_gallery", "cross_midi_mixed_gallery"):
            episode_ids_by_arm = [{row["episode_id"] for row in previous_rows
                                   if row.get("arm") == arm and row.get("condition") == condition}
                                  for arm in previous_arms]
            if episode_ids_by_arm and any(episode_ids_by_arm) and not all(episode_ids_by_arm):
                raise ValueError(f"existing evaluation is missing paired episode rows for {condition}")
            previous_common_by_condition[condition] = (set.intersection(*episode_ids_by_arm)
                if episode_ids_by_arm and all(episode_ids_by_arm) else set())
        if selected_models & {arm.split(":", 1)[0] for arm in previous_result.get("available_arms", [])}:
            previous_rows = [row for row in previous_rows if row.get("arm", "").split(":", 1)[0] not in selected_models]
            previous_result["model_results"] = {arm: data for arm, data in previous_result.get("model_results", {}).items()
                if arm.split(":", 1)[0] not in selected_models}
            previous_result["available_arms"] = sorted(previous_result["model_results"])
    if hard_negative_source is not None:
        source_rows = _jsonl(hard_negative_source / "episode_metrics.private.jsonl")
        for row in source_rows:
            state_ids = row.get("acoustic_hard_negative_state_ids")
            similarities = row.get("acoustic_hard_negative_selector_similarities")
            if state_ids is None or similarities is None:
                continue
            if len(state_ids) != len(similarities):
                raise ValueError("cached acoustic hard-negative states and selector scores disagree")
            selected = list(zip(state_ids, similarities, strict=True))
            episode_id = row["episode_id"]
            if episode_id in previous_hard_negatives and previous_hard_negatives[episode_id] != selected:
                raise ValueError(f"saved acoustic hard-negative selection differs across arms for {episode_id}")
            previous_hard_negatives[episode_id] = selected
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
    qa_episodes, qa_summary, candidates = _freeze_qa_qualified_episodes(
        dataset_dir, episodes, candidates, pair_to_request, receipts)
    accepted_request_ids = {request_id for request_id, receipt in receipts.items() if receipt.get("accepted")}
    model_features = {}
    for model_dir in sorted(feature_root.glob("*")):
        if not model_dir.is_dir() or not (model_dir / "feature_manifest.private.jsonl").is_file():
            continue
        model_name = model_dir.name
        if selected_models is not None and model_name not in selected_models:
            continue
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
    if selected_models is not None and not model_features:
        raise ValueError(f"no extracted feature caches found for selected encoder models: {sorted(selected_models)}")
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
        if selected_models is not None:
            prior_common = previous_common_by_condition[condition]
            common &= prior_common
        common_by_condition[condition] = common if model_features else set()
    selected_episode_ids = {episode_id for episode_ids in common_by_condition.values() for episode_id in episode_ids}
    acoustic_hard_negatives = {episode_id: previous_hard_negatives[episode_id] for episode_id in selected_episode_ids
                               if episode_id in previous_hard_negatives}
    missing_hard_negative_ids = selected_episode_ids - set(acoustic_hard_negatives)
    if missing_hard_negative_ids:
        missing_episodes = [episode for episode in qa_episodes if episode["episode_id"] in missing_hard_negative_ids]
        acoustic_hard_negatives.update(_acoustic_hard_negative_candidates(
            dataset_dir, missing_episodes, candidates, pair_to_request, receipts))
    coverage_by_condition = {}
    for condition in conditions:
        qa_counts = qa_summary["coverage_by_condition"].get(condition, {"planned_episodes": 0, "qa_qualified_episodes": 0,
                                                                          "excluded_episodes": 0, "missing_required_request_status_counts": {}})
        prior_coverage = previous_result.get("coverage_by_condition", {}).get(condition, {})
        coverage_by_condition[condition] = {**qa_counts,
            "feature_qualified_episodes_by_arm": {**prior_coverage.get("feature_qualified_episodes_by_arm", {}),
                **{arm: len(rows[condition]) for arm, rows in sorted(eligible_by_arm.items())}},
            "common_feature_qualified_episodes_across_arms": len(common_by_condition[condition])}
    results = {}
    all_result_rows = {}
    for arm, (model_name, variant, features) in model_features.items():
        results[arm], all_result_rows[arm] = {}, {}
        for condition in conditions:
            selected = [episode for episode in qa_episodes if episode["episode_id"] in common_by_condition[condition]]
            rows = _evaluate_condition(selected, candidates, pair_to_request, features, variant, condition,
                                       acoustic_hard_negatives)
            rows = _attach_render_quality(rows, selected, candidates, pair_to_request, receipts)
            results[arm][condition] = _summary(rows)
            results[arm][condition]["strata"] = _strata(rows)
            results[arm][condition]["performance_cluster_bootstrap"] = ({
                metric: performance_cluster_bootstrap(rows, metric=metric) for metric in
                ("recall_at_1", "recall_at_5", "mrr")} if rows else {})
            results[arm][condition]["two_way_cluster_bootstrap"] = ({
                metric: two_way_cluster_bootstrap(rows, metric=metric) for metric in
                ("recall_at_1", "recall_at_5", "mrr")} if rows else {})
            all_result_rows[arm][condition] = rows
    new_result_rows = all_result_rows
    new_arm_names = sorted(results)
    arm_names = sorted(set(previous_result.get("available_arms", [])) | set(new_arm_names))
    all_result_rows = {arm: {condition: [] for condition in conditions} for arm in arm_names}
    for row in previous_rows:
        if row.get("arm") in all_result_rows:
            all_result_rows[row["arm"]][row["condition"]].append({key: value for key, value in row.items() if key != "arm"})
    for arm in new_arm_names:
        all_result_rows[arm] = new_result_rows[arm]
    differences = dict(previous_result.get("paired_cluster_bootstrap_differences", {}))
    family_differences = dict(previous_result.get("paired_family_bootstrap_differences", {}))
    for left_index, left_name in enumerate(arm_names):
        for right_name in arm_names[left_index + 1:]:
            if left_name not in new_arm_names and right_name not in new_arm_names:
                continue
            pair = f"{left_name} - {right_name}"
            reverse_pair = f"{right_name} - {left_name}"
            if pair in differences or reverse_pair in differences:
                continue
            differences[pair], family_differences[pair] = {}, {}
            for condition in conditions:
                left_rows = all_result_rows[left_name][condition]
                right_rows = all_result_rows[right_name][condition]
                if left_rows and right_rows:
                    family = {metric: paired_cluster_bootstrap(left_rows, right_rows, metric=metric)
                              for metric in ("recall_at_1", "recall_at_5", "mrr")}
                    differences[pair][condition] = {
                        "family": family,
                        "performance": {metric: paired_performance_bootstrap(left_rows, right_rows, metric=metric)
                                        for metric in ("recall_at_1", "recall_at_5", "mrr")},
                        "performance_and_family": {
                            metric: paired_two_way_cluster_bootstrap(left_rows, right_rows, metric=metric)
                            for metric in ("recall_at_1", "recall_at_5", "mrr")}}
                    family_differences[pair][condition] = family
    ranking_stability = {condition: performance_ranking_bootstrap({arm: all_result_rows[arm][condition]
        for arm in arm_names if all_result_rows[arm][condition]}) for condition in conditions
        if len([arm for arm in arm_names if all_result_rows[arm][condition]]) > 1}
    paired_episodes = [episode for condition in conditions for episode in qa_episodes
                       if episode["condition"] == condition and episode["episode_id"] in common_by_condition[condition]]
    baseline = (midi_shortcut_baseline(paired_episodes, candidates, performances)
                if paired_episodes and dataset_summary.get("performance_source_split", dataset_summary["source_split"])
                == dataset_summary["source_split"] else
                {"status": "not_applicable", "reason": "query MIDI source states are outside the validation preset gallery",
                 "random": {condition: random_baseline([len(candidates[episode["candidate_set_id"]]["state_ids"])
                     for episode in paired_episodes if episode["condition"] == condition])
                     for condition in conditions if any(episode["condition"] == condition for episode in paired_episodes)}})
    result = {**previous_result, "version": 1, "policy": "frozen_encoder_real_vital_cross_performance_v1",
              "qa_qualification_id": qa_summary["qualification_id"],
              "common_episode_count_across_available_arms": len(paired_episodes),
              "coverage_by_condition": coverage_by_condition,
              "gallery_candidate_states_original": sum(len(row["state_ids"]) for row in
                    json.loads((dataset_dir / "candidate_sets.private.json").read_text(encoding="utf-8")).values()),
              "gallery_candidate_states_render_qualified": sum(len(row["state_ids"]) for row in candidates.values()),
              "gallery_candidate_states_excluded": sum(len(row["state_ids"]) for row in
                    json.loads((dataset_dir / "candidate_sets.private.json").read_text(encoding="utf-8")).values()) -
                    sum(len(row["state_ids"]) for row in candidates.values()),
              "available_arms": arm_names, "unavailable_encoders": [name for name in MODEL_REGISTRY if not any(arm.startswith(name + ":") for arm in arm_names)],
              "model_results": {**previous_result.get("model_results", {}), **results}, "paired_family_bootstrap_differences": family_differences,
              "paired_cluster_bootstrap_differences": differences, "ranking_stability_by_performance": ranking_stability,
              "acoustic_hard_negative_protocol": "For each query, rank same-MIDI candidate renders by cosine over normalized 64-bin log-frequency spectral power; score each model's cross-MIDI positive against the top five descriptor-selected negatives rendered on the query MIDI.",
              "midi_only_and_random_baselines": baseline,
              "no_source_audio_used": True, "held_out_reserved_sources_used": False,
              "performance_source_split": dataset_summary.get("performance_source_split", dataset_summary["source_split"]),
              "primary_metric": "Recall@1", "gallery_candidate_set_is_identical_across_arms": True}
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "evaluation.private.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with (result_root / "episode_metrics.private.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in previous_rows:
            stream.write(canonical_json(row) + "\n")
        for arm in arm_names:
            if arm not in results:
                continue
            for condition in new_result_rows[arm]:
                for row in new_result_rows[arm][condition]:
                    stream.write(json.dumps({"arm": arm, **row}, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
    return result
