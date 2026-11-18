from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
for directory in (REPO / "research/data_generation", REPO / "research/training",
                  REPO / "research/modelling/basic_pitch", PACKAGE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from frozen_retrieval import core, evaluation
from frozen_retrieval.dataset import validate_dataset_manifests
from frozen_retrieval.encoders import (cache_feature, file_sha256, list_clip_cache, load_adapter,
                                       load_checkpoint_lock, resolve_checkpoint, resolve_source_checkout)


ASSIGNMENT_SCORE_CONTRACT_VERSION = "issue155-gallery-assignment-v1"
_CACHED_ROWS_UNSET = object()


def _jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _mean(rows: list[dict], metric: str) -> float | None:
    return sum(float(row[metric]) for row in rows) / len(rows) if rows else None


def _metric_result(rows: list[dict], metrics=("recall_at_1", "recall_at_5", "mrr")) -> dict:
    result = {"queries": len(rows), "metrics": {}}
    for metric in metrics:
        ci = evaluation.two_way_cluster_bootstrap(rows, metric=metric) if rows else {}
        result["metrics"][metric] = {"mean": _mean(rows, metric), "crossed_95_ci": ci}
    result["query_midi_ids"] = len({row["query_midi_id"] for row in rows})
    result["preset_families"] = len({row["target_family_id"] for row in rows})
    return result


def _paired_rows(left: list[dict], right: list[dict], *, key_fields=("target_state_id", "query_midi_id")) -> tuple[list[dict], list[dict]]:
    left_by_key = {tuple(row[key] for key in key_fields): row for row in left}
    right_by_key = {tuple(row[key] for key in key_fields): row for row in right}
    if left_by_key.keys() != right_by_key.keys():
        raise ValueError(f"paired comparison has different query cohorts ({len(left_by_key)} vs {len(right_by_key)})")
    paired_left, paired_right = [], []
    for key in sorted(left_by_key):
        row_left, row_right = left_by_key[key], right_by_key[key]
        if (row_left["target_family_id"], row_left["query_midi_id"]) != (
                row_right["target_family_id"], row_right["query_midi_id"]):
            raise ValueError("paired query differs in performance or preset family")
        identity = "|".join(key)
        paired_left.append({**row_left, "episode_id": identity})
        paired_right.append({**row_right, "episode_id": identity})
    return paired_left, paired_right


def _paired_result(left: list[dict], right: list[dict], metrics=("recall_at_1",)) -> dict:
    a, b = _paired_rows(left, right)
    return {metric: evaluation.paired_two_way_cluster_bootstrap(a, b, metric=metric)
            for metric in metrics}


def _feature_cache_identity(feature_root: Path, expected_normalization: dict | None = None) -> tuple[dict, dict[str, set[str]]]:
    models, request_ids_by_model = {}, {}
    for model_name in sorted(core.MODEL_REGISTRY):
        cache_dir = feature_root / model_name
        manifest_path = cache_dir / "feature_manifest.private.jsonl"
        summary = json.loads((cache_dir / "extraction.latest.json").read_text(encoding="utf-8"))
        normalization = summary.get("normalization")
        if expected_normalization is not None and any(
                normalization is None or normalization.get(key) != value
                for key, value in expected_normalization.items()):
            raise ValueError(f"{model_name} normalized cache has a stale or missing normalization contract")
        manifest = _jsonl(manifest_path)
        if len(manifest) != summary.get("feature_count") or not manifest:
            raise ValueError(f"{model_name} feature manifest does not match its extraction summary")
        request_ids = {row["request_id"] for row in manifest}
        feature_ids = {row["feature_id"] for row in manifest}
        if len(request_ids) != len(manifest) or len(feature_ids) != len(manifest):
            raise ValueError(f"{model_name} feature manifest has duplicate request or feature identities")
        request_ids_by_model[model_name] = request_ids
        models[model_name] = {"checkpoint_sha256": summary.get("checkpoint_sha256"),
            "source_revision": summary.get("source_revision"), "loader_revision": summary.get("loader_revision"),
            "feature_count": summary["feature_count"], "normalization": normalization,
            "manifest_sha256": file_sha256(manifest_path),
            "request_ids_sha256": core.digest(sorted(request_ids))}
    return {"models": models, "identity_sha256": core.digest(models)}, request_ids_by_model


def _assignment_cache_identity(context: dict, offsets: tuple[int, ...],
                               expected_episode_ids_by_offset: dict[int, set[str]]) -> dict:
    episodes = {str(offset): sorted(episode_ids)
                for offset, episode_ids in sorted(expected_episode_ids_by_offset.items())}
    return {"score_contract_version": ASSIGNMENT_SCORE_CONTRACT_VERSION,
        "feature_cache_identity": context["feature_cache_identity"],
        "candidate_set_id": context["source_candidate_id"], "arms": sorted(context["arms"]),
        "offsets": list(offsets), "episode_ids_sha256": core.digest(episodes)}


def _assignment_cache_contract_path(path: Path) -> Path:
    return path.with_name(path.name + ".contract.json")


def _write_assignment_cache_contract(path: Path, identity: dict) -> None:
    contract = {"identity": identity, "scores_sha256": file_sha256(path)}
    _assignment_cache_contract_path(path).write_text(
        json.dumps(contract, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _load_context(dataset: Path, *, feature_root: Path | None = None, load_features: bool = True,
                  expected_normalization: dict | None = None) -> dict:
    dataset = dataset.resolve(strict=True)
    feature_root = (feature_root or dataset / "features").resolve(strict=True)
    feature_identity, feature_request_ids = _feature_cache_identity(feature_root, expected_normalization)
    request_set_digests = {core.digest(sorted(request_ids)) for request_ids in feature_request_ids.values()}
    if len(request_set_digests) != 1:
        raise ValueError("encoder feature manifests do not share the same source request cohort")
    summary = json.loads((dataset / "dataset_summary.private.json").read_text(encoding="utf-8"))
    validate_dataset_manifests(dataset, summary)
    plans = _jsonl(dataset / "render_plan.private.jsonl")
    receipts = evaluation._render_receipts(dataset, plans)
    render_plans = [row for row in plans if row.get("source_kind") == "newly_rendered_original_preset"]
    pair_to_request = {(row["state_id"], row["midi_id"]): row["request_id"] for row in render_plans}
    all_candidates = json.loads((dataset / "candidate_sets.render_qualified.private.json").read_text(encoding="utf-8"))["candidate_sets"]
    episodes = [row for row in _jsonl(dataset / "episodes_qa_qualified.private.jsonl")
                if row.get("condition") == "cross_midi_common_gallery"]
    if not episodes:
        raise ValueError("the frozen QA-qualified common-gallery cohort is empty")
    performances = [row for row in _jsonl(dataset / "performances.private.jsonl")
                    if row.get("split") == summary["source_split"]]
    if len({row["midi_id"] for row in performances}) != 50:
        raise ValueError("issue #155 expects the existing 50-MIDI performance cohort")

    model_caches, model_summaries = {}, {}
    if load_features:
        for model_name in sorted(core.MODEL_REGISTRY):
            cache_dir = feature_root / model_name
            manifest = _jsonl(cache_dir / "feature_manifest.private.jsonl")
            feature_ids = {row["feature_id"] for row in manifest}
            if len(feature_ids) != len(manifest):
                raise ValueError(f"{model_name} has duplicate feature identities")
            model_summary = json.loads((cache_dir / "extraction.latest.json").read_text(encoding="utf-8"))
            features = list_clip_cache(cache_dir, model_name, feature_ids=feature_ids,
                                       expected_checkpoint_sha256=model_summary["checkpoint_sha256"])
            if set(features) != {row["request_id"] for row in manifest}:
                raise ValueError(f"{model_name} feature manifest and cache differ")
            for row in manifest:
                identity = json.loads((cache_dir / f"{row['feature_id']}.json").read_text(encoding="utf-8"))["cache_identity"]
                receipt = receipts.get(row["request_id"])
                if (not receipt or identity.get("renderer_request_id") != row["request_id"] or
                        identity.get("audio_file_sha256") != receipt.get("audio_sha256") or
                        identity.get("checkpoint_sha256") != model_summary.get("checkpoint_sha256")):
                    raise ValueError(f"{model_name} cache does not match its accepted WAV/checkpoint identity")
            if any(request_id not in receipts or not receipts[request_id].get("accepted") for request_id in features):
                raise ValueError(f"{model_name} features include a render outside the accepted receipt set")
            model_caches[model_name], model_summaries[model_name] = features, model_summary
        arms = {f"{model}:{variant}": (model, variant, features)
                for model, features in model_caches.items()
                for variant in sorted({variant for value in features.values() for variant in value})}
    else:
        from frozen_retrieval.sensitivity import EXPECTED_VARIANTS

        model_caches = {model: feature_request_ids[model] for model in core.MODEL_REGISTRY}
        arms = {f"{model}:{variant}": (model, variant, {})
                for model, variants in EXPECTED_VARIANTS.items() for variant in variants}
    expected_arms = sum(1 if spec["kind"] == "clip" else 2 for spec in core.MODEL_REGISTRY.values())
    if len(arms) != expected_arms or set(model_caches) != set(core.MODEL_REGISTRY):
        raise ValueError(f"the assignment comparison requires all {len(core.MODEL_REGISTRY)} frozen encoders and {expected_arms} pooling arms")
    candidate_ids = {episode["candidate_set_id"] for episode in episodes}
    if len(candidate_ids) != 1:
        raise ValueError("the common-gallery cohort must use one frozen candidate set")
    source_candidate_id = next(iter(candidate_ids))
    candidate = all_candidates[source_candidate_id]
    if len(candidate["state_ids"]) != 129:
        raise ValueError("the frozen expanded cohort must retain 129 render-qualified gallery states")
    return {"dataset": dataset, "summary": summary, "plans": render_plans,
            "receipts": receipts, "pair_to_request": pair_to_request, "candidates": all_candidates,
            "source_candidate_id": source_candidate_id, "source_candidate": candidate,
            "episodes": episodes, "performances": performances, "model_caches": model_caches,
            "model_summaries": model_summaries, "arms": arms,
            "feature_cache_identity": feature_identity}


def _updated_episode_strata(episode: dict, query: dict, gallery: dict) -> dict:
    from frozen_retrieval.core import _density_band, _duration_band, _duration_gap_band, _pitch_gap_band

    query_stats, gallery_stats = query.get("stats", {}), gallery.get("stats", {})
    pitch_gap = abs(query_stats.get("pitch_median", 0) - gallery_stats.get("pitch_median", 0))
    duration_gap = abs(query_stats.get("duration_seconds", 0) - gallery_stats.get("duration_seconds", 0))
    return {**episode.get("strata", {}),
            "pitch_median_gap": pitch_gap, "pitch_register_gap_band": _pitch_gap_band(pitch_gap),
            "duration_gap_seconds": round(duration_gap, 3), "duration_gap_band": _duration_gap_band(duration_gap),
            "query_duration_band": _duration_band(query_stats.get("duration_seconds")),
            "query_note_density_band": _density_band(query_stats.get("notes_per_second"))}


def _ordered_midi_pool(performances: list[dict], split: str, offsets: tuple[int, ...]) -> list[dict]:
    ordered = core.stable_order(performances, "midi_id", salt=f"{core.POLICY_ID}:{split}:gallery-query-v1")
    if len(ordered) < 2 or any(offset <= 0 or offset >= len(ordered) for offset in offsets) or len(set(offsets)) != len(offsets):
        raise ValueError("gallery assignment offsets must be distinct, non-zero positions within the MIDI pool")
    return ordered


def _build_assignments(context: dict, offsets: tuple[int, ...]) -> tuple[list[dict], dict, dict]:
    split = context["summary"]["source_split"]
    ordered = _ordered_midi_pool(context["performances"], split, offsets)
    index_by_midi = {row["midi_id"]: index for index, row in enumerate(ordered)}
    performance_by_midi = {row["midi_id"]: row for row in ordered}
    candidate = context["source_candidate"]
    source_id = context["source_candidate_id"]
    candidate_state_ids = candidate["state_ids"]
    feature_requests = {model: set(features) for model, features in context["model_caches"].items()}
    candidate_cache, all_candidates = {}, dict(context["candidates"])
    output, excluded = [], Counter()
    source_episodes = sorted(context["episodes"], key=lambda row: (row["query_midi_id"], row["target_state_id"]))
    for offset in offsets:
        for episode in source_episodes:
            query_index = index_by_midi[episode["query_midi_id"]]
            gallery_midi = ordered[(query_index + offset) % len(ordered)]["midi_id"]
            if gallery_midi == episode["query_midi_id"]:
                raise AssertionError("query and gallery MIDI IDs must differ")
            candidate_key = (source_id, gallery_midi)
            if candidate_key not in candidate_cache:
                states = []
                for state_id in candidate_state_ids:
                    request_id = context["pair_to_request"].get((state_id, gallery_midi))
                    receipt = context["receipts"].get(request_id) if request_id else None
                    if (receipt and receipt.get("accepted") and
                            all(request_id in feature_requests[model] for model in feature_requests)):
                        states.append(state_id)
                if len(states) < 2:
                    candidate_cache[candidate_key] = None
                else:
                    family_map = candidate["family_by_state"]
                    candidate_set = {"split": candidate["split"], "state_ids": states,
                                     "family_by_state": {state: family_map[state] for state in states},
                                     "known_near_neighbor_state_ids_by_state": {
                                         state: [other for other in candidate.get("known_near_neighbor_state_ids_by_state", {}).get(state, [])
                                                 if other in states] for state in states},
                                     "gallery_midi_ids": [gallery_midi]}
                    candidate_id = core.digest({"policy": "issue155_gallery_assignment_candidate_v1",
                                                "source_candidate_set_id": source_id,
                                                "gallery_midi_id": gallery_midi,
                                                "state_ids": states,
                                                "gallery_receipt_hashes": {
                                                    state: context["receipts"][context["pair_to_request"][(state, gallery_midi)]]
                                                    ["audio_sha256"] for state in states}})
                    candidate_cache[candidate_key] = candidate_id
                    all_candidates[candidate_id] = candidate_set
            candidate_id = candidate_cache[candidate_key]
            if candidate_id is None:
                excluded["gallery_has_fewer_than_two_accepted_candidates"] += 1
                continue
            query_request = context["pair_to_request"].get((episode["target_state_id"], episode["query_midi_id"]))
            query_receipt = context["receipts"].get(query_request) if query_request else None
            if not query_receipt or not query_receipt.get("accepted"):
                excluded["query_render_not_accepted"] += 1
                continue
            if episode["target_state_id"] not in all_candidates[candidate_id]["state_ids"]:
                excluded["target_not_in_cleanly_available_gallery"] += 1
                continue
            rows = {model: features for model, features in context["model_caches"].items()}
            if any(query_request not in features for features in rows.values()):
                excluded["query_feature_missing"] += 1
                continue
            query_performance = performance_by_midi[episode["query_midi_id"]]
            gallery_performance = performance_by_midi[gallery_midi]
            assignment = {**episode, "candidate_set_id": candidate_id, "gallery_midi_id": gallery_midi,
                          "gallery_assignment_offset": offset,
                          "strata": _updated_episode_strata(episode, query_performance, gallery_performance)}
            assignment["episode_id"] = core.digest({"policy": "issue155_gallery_assignment_v1",
                "source_query_episode_id": episode["episode_id"], "offset": offset,
                "gallery_midi_id": gallery_midi, "candidate_set_id": candidate_id})
            output.append(assignment)
    by_offset = {offset: [row for row in output if row["gallery_assignment_offset"] == offset] for offset in offsets}
    gallery_ids = {offset: {row["gallery_midi_id"] for row in rows} for offset, rows in by_offset.items()}
    return output, all_candidates, {"by_offset": by_offset, "excluded": dict(excluded),
                                    "gallery_midi_count_by_offset": {offset: len(ids) for offset, ids in gallery_ids.items()},
                                    "midi_order": [row["midi_id"] for row in ordered]}


def _baseline_rows(dataset: Path, arms: set[str]) -> dict[str, list[dict]]:
    rows = {arm: [] for arm in arms}
    path = dataset / "episode_metrics.private.jsonl"
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            arm = row.get("arm")
            if arm in rows and row.get("condition") == "cross_midi_common_gallery":
                rows[arm].append(row)
    if any(len(arm_rows) != 6450 for arm_rows in rows.values()):
        raise ValueError("the frozen four-MIDI baseline is missing one or more of the 6,450 paired queries")
    return rows


def _reduce_row(row: dict) -> dict:
    keys = ("episode_id", "target_state_id", "target_family_id", "query_midi_id", "gallery_midi_id",
            "gallery_assignment_offset", "recall_at_1", "recall_at_5", "mrr", "rank_expected",
            "gallery_size", "query_render_quality", "correct_match_gallery_render_quality",
            "gallery_warning_candidate_count", "gallery_candidate_count", "gallery_warning_candidate_fraction",
            "gallery_unknown_candidate_count")
    return {key: row[key] for key in keys if key in row}


def _collapse_assignments(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["target_state_id"], row["query_midi_id"])].append(row)
    collapsed = []
    for (state_id, midi_id), items in sorted(grouped.items()):
        exemplar = items[0]
        collapsed.append({"episode_id": f"{state_id}|{midi_id}", "target_state_id": state_id,
                          "target_family_id": exemplar["target_family_id"], "query_midi_id": midi_id,
                          **{metric: sum(float(row[metric]) for row in items) / len(items)
                             for metric in ("recall_at_1", "recall_at_5", "mrr")}})
    return collapsed


def _load_cached_assignment_rows(path: Path, arms: set[str],
                                expected_episode_ids_by_offset: dict[int, set[str]],
                                expected_identity: dict) -> dict[str, list[dict]] | None:
    if not path.is_file():
        return None
    try:
        contract = json.loads(_assignment_cache_contract_path(path).read_text(encoding="utf-8"))
        if contract != {"identity": expected_identity, "scores_sha256": file_sha256(path)}:
            return None
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    rows_by_arm = {arm: [] for arm in arms}
    seen = {arm: {offset: set() for offset in expected_episode_ids_by_offset} for arm in arms}
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                arm = record.pop("arm")
                offset, episode_id = record["gallery_assignment_offset"], record["episode_id"]
                if (arm not in rows_by_arm or offset not in expected_episode_ids_by_offset or
                        episode_id not in expected_episode_ids_by_offset[offset] or episode_id in seen[arm][offset]):
                    return None
                seen[arm][offset].add(episode_id)
                rows_by_arm[arm].append(record)
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    if any(seen[arm][offset] != expected for arm in arms
           for offset, expected in expected_episode_ids_by_offset.items()):
        return None
    return rows_by_arm


def _family_best(arms: dict[str, list[dict]]) -> dict[str, str]:
    best = {}
    for arm, rows in arms.items():
        family = arm.split(":", 1)[0]
        current = best.get(family)
        if current is None or _mean(rows, "recall_at_1") > _mean(arms[current], "recall_at_1"):
            best[family] = arm
    return best


def _score_assignments(context: dict, offsets: tuple[int, ...], output: Path, *,
                       reuse_cached_only: bool = False,
                       cached_rows_override: object = _CACHED_ROWS_UNSET) -> tuple[dict, dict]:
    assignments, candidates, coverage = _build_assignments(context, offsets)
    if any(len(coverage["by_offset"][offset]) != 6450 for offset in offsets):
        raise ValueError(f"gallery assignment coverage differs from 6,450 queries: {coverage}")
    output.mkdir(parents=True, exist_ok=True)
    baseline = _baseline_rows(context["dataset"], set(context["arms"]))
    result_rows: dict[str, list[dict]] = {arm: [] for arm in context["arms"]}
    per_assignment = {offset: {} for offset in offsets}
    output_path = output / "gallery_assignment_episode_metrics.private.jsonl"
    expected_episode_ids = {offset: {row["episode_id"] for row in episodes}
                             for offset, episodes in coverage["by_offset"].items()}
    cache_identity = _assignment_cache_identity(context, offsets, expected_episode_ids)
    cached_rows = (_load_cached_assignment_rows(output_path, set(result_rows), expected_episode_ids, cache_identity)
                   if cached_rows_override is _CACHED_ROWS_UNSET else cached_rows_override)
    if cached_rows is not None:
        result_rows = cached_rows
        for offset in offsets:
            for arm, rows in result_rows.items():
                per_assignment[offset][arm] = _metric_result(
                    [row for row in rows if row["gallery_assignment_offset"] == offset])
        print(f"Reusing verified issue #155 assignment scores from {output_path}.", flush=True)
    else:
        if reuse_cached_only:
            raise ValueError(f"cached issue #155 assignment scores are missing or incomplete: {output_path}")
        with output_path.open("w", encoding="utf-8", newline="\n") as stream:
            for arm, (model, variant, features) in sorted(context["arms"].items()):
                all_rows = []
                for offset in offsets:
                    episodes = coverage["by_offset"][offset]
                    rows = evaluation._evaluate_condition(episodes, candidates, context["pair_to_request"],
                        features, variant, "cross_midi_common_gallery", {})
                    rows = evaluation._attach_render_quality(rows, episodes, candidates,
                        context["pair_to_request"], context["receipts"])
                    reduced = [_reduce_row(row) for row in rows]
                    all_rows.extend(reduced)
                    per_assignment[offset][arm] = _metric_result(reduced)
                    for row in reduced:
                        stream.write(json.dumps({"arm": arm, **row}, sort_keys=True, separators=(",", ":")) + "\n")
                result_rows[arm] = all_rows
        _write_assignment_cache_contract(output_path, cache_identity)

    aggregate_rows = {arm: _collapse_assignments(rows) for arm, rows in result_rows.items()}
    aggregate = {arm: _metric_result(rows) for arm, rows in aggregate_rows.items()}
    comparisons = {"cir_minus_beats_temporal": {}}
    cir_arm, beats_arm = "cir_ast:official_clip", "beats_iter3_plus:temporal_pyramid_8"
    for offset in offsets:
        comparisons["cir_minus_beats_temporal"][str(offset)] = _paired_result(
            [r for r in result_rows[cir_arm] if r["gallery_assignment_offset"] == offset],
            [r for r in result_rows[beats_arm] if r["gallery_assignment_offset"] == offset])
    comparisons["cir_minus_beats_temporal"]["aggregate"] = _paired_result(
        aggregate_rows[cir_arm], aggregate_rows[beats_arm])
    comparisons["assignment_average_minus_four_midi_baseline"] = {}
    for arm, rows in aggregate_rows.items():
        comparisons["assignment_average_minus_four_midi_baseline"][arm] = _paired_result(
            rows, baseline[arm], metrics=("recall_at_1", "recall_at_5", "mrr"))
    best_arms = _family_best(aggregate_rows)
    ranking_by_assignment, per_assignment_winners = {}, {}
    for offset in offsets:
        rows_for_offset = {arm: [row for row in result_rows[arm]
                                 if row["gallery_assignment_offset"] == offset]
                           for arm in best_arms.values()}
        ranking_by_assignment[str(offset)] = core.performance_ranking_bootstrap(rows_for_offset)
        per_assignment_winners[str(offset)] = max(
            per_assignment[offset].items(), key=lambda item: item[1]["metrics"]["recall_at_1"]["mean"])[0]
    ranking_by_assignment["aggregate"] = core.performance_ranking_bootstrap(
        {family: aggregate_rows[arm] for family, arm in best_arms.items()})
    sizes = [row["gallery_size"] for rows in result_rows.values() for row in rows]
    result = {"policy": "issue155_gallery_assignment_offsets_v1", "offsets": list(offsets),
              "midi_order_sha256": hashlib.sha256("\n".join(coverage["midi_order"]).encode()).hexdigest(),
              "query_count_per_assignment": 6450, "assignment_rounds_are_correlated_repeats": True,
              "aggregate_collapses_rounds_within_each_query_before_bootstrap": True,
              "gallery_midi_count_by_offset": coverage["gallery_midi_count_by_offset"],
              "gallery_candidate_count": {"min": min(sizes), "median": sorted(sizes)[len(sizes) // 2], "max": max(sizes)},
              "excluded": coverage["excluded"], "per_assignment_winner": per_assignment_winners,
              "per_assignment": {str(offset): per_assignment[offset] for offset in offsets},
              "aggregate": aggregate, "best_pooling_arm_by_family": best_arms,
              "ranking_stability_by_performance": ranking_by_assignment, "paired_differences": comparisons}
    return result, {"assignments": assignments, "candidates": candidates,
                    "rows_by_arm": result_rows, "aggregate_rows": aggregate_rows, "baseline": baseline}


def _clean_query_analysis(context: dict, baseline_rows: dict[str, list[dict]]) -> dict:
    by_arm = {}
    for arm, rows in baseline_rows.items():
        clean = [row for row in rows if row.get("query_render_quality") == "clean"]
        by_arm[arm] = clean
    query_sets = {arm: {row["episode_id"] for row in rows} for arm, rows in by_arm.items()}
    if len({frozenset(values) for values in query_sets.values()}) != 1:
        raise ValueError("clean-query results are not paired across all arms")
    return {"source": "filtered existing per-episode scores from cached embeddings; no inference",
            "arms": {arm: _metric_result(rows) for arm, rows in by_arm.items()},
            "paired_cir_minus_beats_temporal": _paired_result(by_arm["cir_ast:official_clip"],
                by_arm["beats_iter3_plus:temporal_pyramid_8"], metrics=("recall_at_1", "recall_at_5", "mrr"))}


def _strict_clean_gallery(context: dict, assignment_data: dict, output: Path) -> dict:
    np = __import__("numpy")
    all_rows = assignment_data["rows_by_arm"]
    candidate_source = context["source_candidate"]
    state_ids = candidate_source["state_ids"]
    family_by_state = candidate_source["family_by_state"]
    clean_candidates, episodes, candidate_counts = {}, [], []
    rejected = Counter()
    feature_models = {model: set(features) for model, features in context["model_caches"].items()}
    for episode in assignment_data["assignments"]:
        query_request = context["pair_to_request"].get((episode["target_state_id"], episode["query_midi_id"]))
        query_receipt = context["receipts"].get(query_request) if query_request else None
        if not query_receipt or query_receipt.get("qa_status") != "accepted":
            rejected["query_not_clean"] += 1
            continue
        gallery_midi = episode["gallery_midi_id"]
        clean_states = []
        for state_id in state_ids:
            request_id = context["pair_to_request"].get((state_id, gallery_midi))
            receipt = context["receipts"].get(request_id) if request_id else None
            if (receipt and receipt.get("qa_status") == "accepted" and
                    all(request_id in feature_models[model] for model in feature_models)):
                clean_states.append(state_id)
        if len(clean_states) < 2:
            rejected["fewer_than_two_clean_gallery_candidates"] += 1
            continue
        if episode["target_state_id"] not in clean_states:
            rejected["correct_match_not_clean"] += 1
            continue
        candidate_id = core.digest({"policy": "issue155_warning_free_gallery_v1",
                                    "gallery_midi_id": gallery_midi, "state_ids": clean_states,
                                    "source_candidate_set_id": context["source_candidate_id"]})
        if candidate_id not in clean_candidates:
            clean_candidates[candidate_id] = {"split": candidate_source["split"], "state_ids": clean_states,
                "family_by_state": {state: family_by_state[state] for state in clean_states},
                "known_near_neighbor_state_ids_by_state": {state: [] for state in clean_states},
                "gallery_midi_ids": [gallery_midi]}
        clean_episode = {**episode, "candidate_set_id": candidate_id}
        episodes.append(clean_episode)
        candidate_counts.append(len(clean_states))
    if not episodes:
        return {"status": "not estimable", "reason": "no query has a clean query WAV and a clean correct-match gallery WAV"}

    results, clean_rows_by_arm, paired_to_full = {}, {}, {}
    for arm, (model, variant, features) in sorted(context["arms"].items()):
        rows = evaluation._evaluate_condition(episodes, clean_candidates, context["pair_to_request"],
            features, variant, "cross_midi_common_gallery", {})
        rows = evaluation._attach_render_quality(rows, episodes, clean_candidates,
            context["pair_to_request"], context["receipts"])
        reduced = [_reduce_row(row) for row in rows]
        clean_collapsed = _collapse_assignments(reduced)
        clean_rows_by_arm[arm] = clean_collapsed
        full_by_id = {row["episode_id"]: row for row in all_rows[arm]}
        full = [full_by_id[row["episode_id"]] for row in reduced]
        paired_to_full[arm] = _paired_result(clean_collapsed, _collapse_assignments(full),
            metrics=("recall_at_1", "recall_at_5", "mrr"))
        results[arm] = _metric_result(clean_collapsed)
    return {"status": "completed", "selection": "query WAV, correct-match gallery WAV and every retained candidate WAV have qa_status=accepted",
            "episodes": len(clean_collapsed), "assignment_observations": len(episodes),
            "assignment_rounds_are_correlated_repeats": True,
            "query_midi_ids": len({row["query_midi_id"] for row in episodes}),
            "gallery_midi_ids": len({row["gallery_midi_id"] for row in episodes}),
            "target_preset_states": len({row["target_state_id"] for row in episodes}),
            "clean_gallery_candidate_count": {"min": min(candidate_counts),
                "median": sorted(candidate_counts)[len(candidate_counts) // 2], "max": max(candidate_counts)},
            "candidate_count_by_encoder_is_identical": True, "excluded": dict(rejected), "arms": results,
            "paired_clean_gallery_minus_full_gallery_same_queries": paired_to_full,
            "paired_cir_minus_beats_temporal": _paired_result(clean_rows_by_arm["cir_ast:official_clip"],
                clean_rows_by_arm["beats_iter3_plus:temporal_pyramid_8"])}


def _warning_sample(context: dict, count: int) -> tuple[list[dict], dict]:
    if not 50 <= count <= 100:
        raise ValueError("the issue #155 warning sample must contain 50 to 100 clips")
    plan_by_request = {row["request_id"]: row for row in context["plans"]}
    performance = {row["midi_id"]: row for row in context["performances"]}
    records = []
    for request_id, receipt in context["receipts"].items():
        if not receipt.get("accepted") or "amplitude_above_unit_range" not in receipt.get("qa_warnings", []):
            continue
        plan = plan_by_request.get(request_id)
        if not plan:
            continue
        midi = performance.get(plan["midi_id"], {})
        stats = midi.get("stats", {})
        peak = float(receipt.get("qa_metrics", {}).get("peak", 0.0))
        records.append({"request_id": request_id, "state_id": plan["state_id"], "midi_id": plan["midi_id"],
                        "family_id": plan.get("family_id", plan["state_id"]), "peak": peak,
                        "polyphony": "polyphonic" if stats.get("max_polyphony", 1) > 1 else "monophonic",
                        "articulation": stats.get("articulation", "unknown"),
                        "variable_velocity": bool(stats.get("velocity_min") != stats.get("velocity_max")),
                        "wav_relative_path": receipt["wav_relative_path"],
                        "source_audio_sha256": receipt["audio_sha256"],
                        "warnings": receipt.get("qa_warnings", [])})
    if len(records) < count:
        raise ValueError(f"only {len(records)} accepted amplitude-warning WAVs are available for a {count}-WAV sample")
    peaks = sorted(row["peak"] for row in records)
    median_peak = peaks[len(peaks) // 2]
    groups = defaultdict(list)
    for row in records:
        severity = "higher_peak" if row["peak"] >= median_peak else "lower_peak"
        key = (row["polyphony"], row["articulation"], severity)
        row["severity_stratum"] = severity
        groups[key].append(row)
    ordered_groups = sorted(groups)
    quota = {key: count // len(ordered_groups) + int(index < count % len(ordered_groups))
             for index, key in enumerate(ordered_groups)}
    for key in ordered_groups:
        quota[key] = min(quota[key], len(groups[key]))
    spare = count - sum(quota.values())
    while spare:
        options = [key for key in ordered_groups if quota[key] < len(groups[key])]
        if not options:
            raise ValueError("stratified warning sample cannot meet the requested sample size")
        key = min(options, key=lambda item: (quota[item], item))
        quota[key] += 1
        spare -= 1

    selected, state_counts, midi_counts = [], Counter(), Counter()
    for key in ordered_groups:
        pool = list(groups[key])
        for _ in range(quota[key]):
            candidate = min(pool, key=lambda row: (state_counts[row["state_id"]], midi_counts[row["midi_id"]],
                hashlib.sha256(("issue155-gain-sample-v1:" + row["request_id"]).encode()).hexdigest()))
            selected.append(candidate)
            state_counts[candidate["state_id"]] += 1
            midi_counts[candidate["midi_id"]] += 1
            pool.remove(candidate)
    selected.sort(key=lambda row: row["request_id"])
    return selected, {"accepted_amplitude_warning_wavs": len(records), "sample_size": len(selected),
        "peak_stratum_median": median_peak,
        "sample_presets": len({row["state_id"] for row in selected}),
        "sample_midi_ids": len({row["midi_id"] for row in selected}),
        "sample_strata": {" / ".join(key): quota[key] for key in ordered_groups},
        "sample_variable_velocity": sum(row["variable_velocity"] for row in selected),
        "selection": "deterministic SHA tie-break with balanced mono/polyphony, articulation, and lower/higher peak strata; greedy preset/MIDI coverage"}


def _normalize_wavs(context: dict, sample: list[dict], output: Path) -> dict[str, dict]:
    import numpy as np
    from scipy.io import wavfile

    directory = output / "normalized_audio"
    directory.mkdir(parents=True, exist_ok=True)
    normalized = {}
    for row in sample:
        source = context["dataset"] / row["wav_relative_path"]
        if file_sha256(source) != row["source_audio_sha256"]:
            raise ValueError(f"source warning WAV changed after its accepted render receipt: {row['request_id']}")
        sample_rate, audio = wavfile.read(source)
        if not np.issubdtype(audio.dtype, np.floating):
            raise ValueError("the amplitude-warning sample unexpectedly contains non-floating WAV data")
        audio = np.asarray(audio)
        peak = float(np.max(np.abs(audio)))
        if not math.isfinite(peak) or peak <= 0:
            raise ValueError("warning WAV has no finite positive peak")
        gain = min(1.0, 0.95 / peak) if peak > 0.95 else 1.0
        adjusted = (audio.astype(np.float64) * gain).astype(audio.dtype)
        if adjusted.shape != audio.shape or adjusted.dtype != audio.dtype or float(np.max(np.abs(adjusted))) > 0.95001:
            raise ValueError("gain-adjusted WAV changed format or still exceeds the peak ceiling")
        name = f"{row['request_id']}.wav"
        destination = directory / name
        wavfile.write(destination, sample_rate, adjusted)
        check_rate, check_audio = wavfile.read(destination)
        if check_rate != sample_rate or check_audio.shape != audio.shape or check_audio.dtype != audio.dtype:
            raise ValueError("gain-adjusted WAV did not preserve sample rate, shape, or precision")
        normalized[row["request_id"]] = {**row, "normalized_audio_path": destination,
            "normalized_audio_sha256": file_sha256(destination), "sample_rate": sample_rate,
            "samples": int(len(audio)), "channels": 1 if audio.ndim == 1 else int(audio.shape[1]),
            "dtype": str(audio.dtype), "gain_linear": gain, "gain_db": 20 * math.log10(gain),
            "adjusted_peak": float(np.max(np.abs(check_audio)))}
    return normalized


def _extract_adjusted_features(context: dict, normalized: dict, checkpoint_root: Path,
                               lock_path: Path, device: str, output: Path) -> tuple[dict, dict]:
    import numpy as np

    lock = load_checkpoint_lock(lock_path)
    adjusted_by_model, extraction_status = {}, {}
    for model_name in core.MODEL_REGISTRY:
        started = time.perf_counter()
        try:
            checkpoint, entry, digest = resolve_checkpoint(model_name, checkpoint_root, lock)
            source_root = resolve_source_checkout(checkpoint_root, entry)
            adapter = load_adapter(model_name, checkpoint, entry, digest, checkpoint_root=checkpoint_root,
                                   device=device, source_root=source_root)
        except Exception as error:
            extraction_status[model_name] = {"status": "failed_to_load", "error": f"{type(error).__name__}: {error}"[:400]}
            continue
        feature_dir = output / "adjusted_features" / model_name
        adjusted_by_model[model_name] = {}
        failures, records = [], []
        for request_id, row in sorted(normalized.items()):
            adjusted_request_id = f"issue155-gain-0.95:{request_id}"
            try:
                record = cache_feature(adapter, row["normalized_audio_path"], request_id=adjusted_request_id,
                    render_metadata={"provenance": {"request_id": adjusted_request_id}}, output=feature_dir,
                    source_precision=row["dtype"], preprocessing_policy="linear_peak_gain_to_0.95_v1")
                records.append({"source_request_id": request_id, "adjusted_request_id": adjusted_request_id,
                    "audio_sha256": row["normalized_audio_sha256"], "feature_id": record["feature_id"],
                    "cache_path": record["cache_path"], "reused": record["reused"]})
                with np.load(feature_dir / record["cache_path"], allow_pickle=False) as arrays:
                    adjusted_by_model[model_name][request_id] = {
                        name: arrays[name].astype(np.float32, copy=False) for name in arrays.files}
            except Exception as error:
                failures.append({"request_id": request_id, "error": f"{type(error).__name__}: {error}"[:240]})
        with (feature_dir / "issue155_gain_feature_manifest.private.jsonl").open(
                "w", encoding="utf-8", newline="\n") as stream:
            for row in records:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        extraction_status[model_name] = {"status": "completed" if not failures else "partial",
            "features": len(adjusted_by_model[model_name]), "expected_features": len(normalized),
            "failures": failures, "elapsed_seconds": round(time.perf_counter() - started, 3),
            "checkpoint_sha256": digest}
    return adjusted_by_model, extraction_status


def _cosine_np(left, right, np) -> float:
    a, b = np.asarray(left, dtype=np.float64).reshape(-1), np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0 or not np.isfinite(denominator):
        raise ValueError("cannot compare a zero or non-finite feature")
    return float(a @ b / denominator)


def _prediction(query, gallery, states: list[str], target: str, np) -> dict:
    scores = evaluation._cosine_matrix(np.asarray(query, dtype=np.float32).reshape(1, -1),
        np.stack(gallery).astype(np.float32), np)[0]
    score_rows = list(zip(states, scores.tolist(), strict=True))
    rank = core._rank_metrics(score_rows, target)
    target_score = float(scores[states.index(target)])
    impostor = max(float(score) for state, score in score_rows if state != target)
    predicted = min((state for state, score in score_rows if float(score) >= max(scores) - 1e-12))
    return {**rank, "prediction": predicted, "target_cosine": target_score,
            "best_impostor_cosine": impostor, "target_margin": target_score - impostor}


def _gain_sensitivity(context: dict, sample: list[dict], normalized: dict, adjusted: dict,
                      extraction_status: dict, output: Path) -> dict:
    import numpy as np

    ordered = _ordered_midi_pool(context["performances"], context["summary"]["source_split"], (7,))
    midi_index = {row["midi_id"]: index for index, row in enumerate(ordered)}
    base_states = context["source_candidate"]["state_ids"]
    families = context["source_candidate"]["family_by_state"]
    rows_by_arm, summaries = {}, {}
    for arm, (model, variant, feature_cache) in sorted(context["arms"].items()):
        if extraction_status.get(model, {}).get("features") != len(sample):
            continue
        vector_rows, query_before, query_after, gallery_before, gallery_after = [], [], [], [], []
        role_changes = {}
        for row in sample:
            request_id, state_id, midi_id = row["request_id"], row["state_id"], row["midi_id"]
            original = feature_cache[request_id][variant]
            changed = adjusted[model][request_id][variant]
            vector_rows.append({"episode_id": request_id, "target_family_id": row["family_id"],
                                "query_midi_id": midi_id, "embedding_distance": 1 - _cosine_np(original, changed, np)})

            query_idx = midi_index[midi_id]
            gallery_midi = ordered[(query_idx + 7) % len(ordered)]["midi_id"]
            gallery_states = [candidate for candidate in base_states
                if (candidate, gallery_midi) in context["pair_to_request"] and
                context["receipts"][context["pair_to_request"][(candidate, gallery_midi)]].get("accepted") and
                all(context["pair_to_request"][(candidate, gallery_midi)] in cache
                    for cache in context["model_caches"].values())]
            if state_id not in gallery_states:
                raise ValueError("gain-sensitivity query lost its target from the accepted gallery")
            gallery_vectors = [feature_cache[context["pair_to_request"][(candidate, gallery_midi)]][variant]
                               for candidate in gallery_states]
            query_before.append({"episode_id": request_id, "target_state_id": state_id,
                "target_family_id": row["family_id"], "query_midi_id": midi_id,
                "gallery_size": len(gallery_states),
                **_prediction(original, gallery_vectors, gallery_states, state_id, np)})
            query_after.append({"episode_id": request_id, "target_state_id": state_id,
                "target_family_id": row["family_id"], "query_midi_id": midi_id,
                "gallery_size": len(gallery_states),
                **_prediction(changed, gallery_vectors, gallery_states, state_id, np)})

            target_gallery_midi = midi_id
            query_midi = ordered[(midi_index[midi_id] - 7) % len(ordered)]["midi_id"]
            target_query_request = context["pair_to_request"][(state_id, query_midi)]
            target_query = feature_cache[target_query_request][variant]
            gallery_states = [candidate for candidate in base_states
                if (candidate, target_gallery_midi) in context["pair_to_request"] and
                context["receipts"][context["pair_to_request"][(candidate, target_gallery_midi)]].get("accepted") and
                all(context["pair_to_request"][(candidate, target_gallery_midi)] in cache
                    for cache in context["model_caches"].values())]
            if state_id not in gallery_states:
                raise ValueError("gain-sensitivity gallery sample lost its target candidate")
            gallery_vectors = [feature_cache[context["pair_to_request"][(candidate, target_gallery_midi)]][variant]
                               for candidate in gallery_states]
            baseline_gallery = _prediction(target_query, gallery_vectors, gallery_states, state_id, np)
            changed_vectors = [adjusted[model][request_id][variant] if candidate == state_id else vector
                               for candidate, vector in zip(gallery_states, gallery_vectors, strict=True)]
            changed_gallery = _prediction(target_query, changed_vectors, gallery_states, state_id, np)
            gallery_before.append({"episode_id": request_id, "target_state_id": state_id,
                "target_family_id": row["family_id"], "query_midi_id": query_midi,
                "gallery_size": len(gallery_states), **baseline_gallery})
            gallery_after.append({"episode_id": request_id, "target_state_id": state_id,
                "target_family_id": row["family_id"], "query_midi_id": query_midi,
                "gallery_size": len(gallery_states), **changed_gallery})

        rows_by_arm[arm] = {"query_before": query_before, "query_after": query_after,
                            "gallery_before": gallery_before, "gallery_after": gallery_after}
        roles = {}
        for role, before, after in (("query", query_before, query_after), ("correct_gallery_candidate", gallery_before, gallery_after)):
            changed_predictions = [(left, right) for left, right in zip(before, after, strict=True)
                                   if left["prediction"] != right["prediction"]]
            roles[role] = {"samples": len(before),
                "changed_predictions": len(changed_predictions),
                "corrected_predictions": sum(left["recall_at_1"] == 0 and right["recall_at_1"] == 1
                                             for left, right in zip(before, after, strict=True)),
                "newly_wrong_predictions": sum(left["recall_at_1"] == 1 and right["recall_at_1"] == 0
                                                for left, right in zip(before, after, strict=True)),
                "before": _metric_result(before, ("recall_at_1", "rank_expected", "target_margin")),
                "after": _metric_result(after, ("recall_at_1", "rank_expected", "target_margin")),
                "gallery_candidate_count": {"min": min(item["gallery_size"] for item in before),
                    "median": sorted(item["gallery_size"] for item in before)[len(before) // 2],
                    "max": max(item["gallery_size"] for item in before)},
                "paired_after_minus_before": _paired_result(after, before,
                    metrics=("recall_at_1", "rank_expected", "target_margin", "target_cosine", "best_impostor_cosine"))}
        shift_ci = evaluation.two_way_cluster_bootstrap(vector_rows, metric="embedding_distance")
        summaries[arm] = {"adjusted_clip_embedding_cosine_distance": {"mean": _mean(vector_rows, "embedding_distance"),
                "crossed_95_ci": shift_ci, "samples": len(vector_rows)}, "roles": roles,
            "gallery_candidates": "per-role min/median/max, QA-driven and identical across encoders"}
    return {"source": "100 accepted amplitude-warning WAVs; only those WAVs received new inference",
            "sample": {"size": len(sample), "normalized_copies": len(normalized),
                       "sampled_preset_states": len({row['state_id'] for row in sample}),
                       "sampled_midi_ids": len({row['midi_id'] for row in sample}),
                       "gain_db_mean": sum(row["gain_db"] for row in normalized.values()) / len(normalized),
                       "peak_before_max": max(row["peak"] for row in sample),
                       "peak_after_max": max(row["adjusted_peak"] for row in normalized.values()),
                       "preprocessing": "linear gain min(1, 0.95/peak); no limiter, compression, clipping, or other DSP"},
            "encoder_extraction": extraction_status, "arms": summaries,
            "reserved_final_test_or_confirmation_accessed": False}


def run(args: argparse.Namespace) -> dict:
    dataset = args.dataset.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    context = _load_context(dataset)
    assignment_result, assignment_data = _score_assignments(context, tuple(args.gallery_offsets), output)
    clean_query = _clean_query_analysis(context, assignment_data["baseline"])
    clean_gallery = _strict_clean_gallery(context, assignment_data, output)
    sample, sample_info = _warning_sample(context, args.sample_count)
    normalized = _normalize_wavs(context, sample, output)
    with (output / "gain_warning_sample.private.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in normalized.values():
            stream.write(json.dumps({key: value for key, value in row.items()
                if key != "normalized_audio_path"}, sort_keys=True, separators=(",", ":")) + "\n")
    adjusted, extraction = _extract_adjusted_features(context, normalized, args.checkpoint_root.resolve(strict=True),
        args.lock.resolve(strict=True), args.device, output)
    gain = _gain_sensitivity(context, sample, normalized, adjusted, extraction, output)
    result = {"version": 1, "issue": 155,
        "input_dataset": {"preset_states": len(context["source_candidate"]["state_ids"]),
            "performance_ids": len(context["performances"]), "accepted_cached_clips": sum(
                len(features) for features in context["model_caches"].values()) // len(context["model_caches"]),
            "encoder_families": len(context["model_caches"]), "arms": len(context["arms"]),
            "reserved_test_accessed": False},
        "original_four_midi_results_preserved": True,
        "gallery_assignment_experiment": assignment_result,
        "clean_query_only": clean_query, "strict_clean_gallery": clean_gallery,
        "gain_warning_sample_selection": sample_info, "gain_sensitivity": gain}
    path = output / "issue155_validation_summary.private.json"
    path.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"summary": str(path), "gallery_assignment_offsets": list(args.gallery_offsets),
            "warning_sample_count": len(sample), "models_with_adjusted_features": len(adjusted),
            "arms_with_gain_results": len(gain["arms"]), "strict_clean_gallery_status": clean_gallery["status"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bounded issue #155 analyses on cached renders and embeddings")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "xpu"), default="xpu")
    parser.add_argument("--gallery-offsets", type=int, nargs="+", default=[1, 7, 19])
    parser.add_argument("--sample-count", type=int, default=100)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), sort_keys=True, indent=2))
