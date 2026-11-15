from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import random
from typing import Iterable, Sequence


POLICY_ID = "cross_performance_retrieval_v2"
MODEL_REGISTRY = {
    "clap_music": {"name": "LAION-CLAP Music HTSAT-base", "sample_rate": 48_000, "kind": "clip", "dimension": 512,
                   "checkpoint": "lukewys/laion_clap/music_audioset_epoch_15_esc_90.14.pt", "source": "https://huggingface.co/lukewys/laion_clap",
                   "repository_id": "lukewys/laion_clap",
                   "tap": "official audio clip embedding", "license": "checkpoint license must be checked at acquisition"},
    "muq_mulan": {"name": "MuQ-MuLan", "sample_rate": 24_000, "kind": "clip", "dimension": 512,
                   "checkpoint": "OpenMuQ/MuQ-MuLan-large/pytorch_model.bin", "source": "https://huggingface.co/OpenMuQ/MuQ-MuLan-large",
                   "repository_id": "OpenMuQ/MuQ-MuLan-large",
                   "tap": "official audio embedding",
                   "license": "CC-BY-NC-4.0; non-commercial use only"},
    "cir_ast": {"name": "CIR single-timbre AST", "sample_rate": 16_000, "kind": "clip", "dimension": 768,
                "checkpoint": "Google Drive: ICASSP26_inst-direct-with-data-aug_final.pt", "source": "https://github.com/gwendal-lv/CIR",
                "repository_id": "gwendal-lv/CIR",
                "tap": "mean of final CLS and distillation tokens",
                "license": "checkpoint license not specified in the upstream repository"},
    "beats_iter3_plus": {"name": "BEATs iter3+ AS2M", "sample_rate": 16_000, "kind": "sequence", "dimension": 768,
                         "checkpoint": "lpepino/beats_ckpts/BEATs_iter3_plus_AS2M.pt", "source": "https://huggingface.co/lpepino/beats_ckpts",
                         "repository_id": "lpepino/beats_ckpts",
                         "expected_sha256": "d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34",
                         "tap": "pretrained iter3+ final pre-predictor features",
                         "license": "mirror labels MIT; upstream checkpoint license remains unconfirmed",
                         "license_source": "https://github.com/microsoft/unilm/issues/1760"},
    "encodec_24khz": {"name": "Meta EnCodec 24 kHz", "sample_rate": 24_000, "kind": "sequence", "dimension": 128,
                       "checkpoint": "facebook/encodec_24khz", "source": "https://huggingface.co/facebook/encodec_24khz",
                       "repository_id": "facebook/encodec_24khz",
                       "tap": "continuous encoder output before RVQ",
                       "license": "checkpoint-specific license must be verified"},
    "dac_44khz": {"name": "Descript Audio Codec 44.1 kHz", "sample_rate": 44_100, "kind": "sequence", "dimension": 1024,
                   "checkpoint": "descript/dac_44khz", "source": "https://huggingface.co/descript/dac_44khz",
                   "repository_id": "descript/dac_44khz",
                   "tap": "continuous encoder output before RVQ",
                   "license": "MIT (upstream release)"},
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def clip_id(preset_id: str, midi_id: str, excerpt_id: str) -> str:
    return digest({"preset_id": preset_id, "midi_id": midi_id, "excerpt_id": excerpt_id, "policy": POLICY_ID})


def stable_order(values: Iterable[dict], *keys: str, salt: str) -> list[dict]:
    def score(value: dict) -> str:
        return hashlib.sha256((salt + ":" + ":".join(str(value[key]) for key in keys)).encode()).hexdigest()

    return sorted(values, key=score)


def build_episodes(presets: Sequence[dict], performances: Sequence[dict], *, gallery_cap: int = 4) -> tuple[dict, list[dict]]:
    """Cross every candidate preset with the shared split MIDI pool for queries and galleries."""
    if gallery_cap < 2:
        raise ValueError("gallery MIDI pool must hold at least two performances so query and gallery can differ")
    splits = sorted({row["split"] for row in presets})
    candidate_sets, episodes = {}, []
    for split in splits:
        members = sorted({row["state_id"]: row for row in presets if row["split"] == split}.values(), key=lambda row: row["state_id"])
        midi_pool = sorted({row["midi_id"]: row for row in performances if row["split"] == split}.values(), key=lambda row: row["midi_id"])
        if len(members) < 2 or len(midi_pool) < 2:
            continue
        ordered_midi = stable_order(midi_pool, "midi_id", salt=f"{POLICY_ID}:{split}:gallery-query-v1")
        gallery_pool = ordered_midi[:min(len(ordered_midi), gallery_cap)]
        gallery_midi_ids = [row["midi_id"] for row in gallery_pool]
        states_by_family = defaultdict(list)
        for row in members:
            states_by_family[row.get("family_id", row["state_id"])].append(row["state_id"])
        candidate_set_id = digest({"split": split, "candidate_state_ids": [row["state_id"] for row in members],
                                   "family_by_state": {row["state_id"]: row.get("family_id", row["state_id"]) for row in members},
                                   "gallery_midi_ids": gallery_midi_ids, "policy": POLICY_ID})
        candidate_sets[candidate_set_id] = {"split": split, "state_ids": [row["state_id"] for row in members],
                                            "family_by_state": {row["state_id"]: row.get("family_id", row["state_id"]) for row in members},
                                            "known_near_neighbor_state_ids_by_state": {
                                                row["state_id"]: sorted(state for state in states_by_family[row.get("family_id", row["state_id"])]
                                                                        if state != row["state_id"])
                                                for row in members},
                                            "gallery_midi_ids": gallery_midi_ids}
        for preset in members:
            queries = ordered_midi
            for query_index, query in enumerate(queries):
                choices = [row for row in gallery_pool if row["midi_id"] != query["midi_id"]]
                if not choices:
                    continue
                rotation = query_index % len(choices)
                galleries = [choices[rotation]]
                for gallery_index, gallery in enumerate(galleries):
                    common = {"condition": "cross_midi_common_gallery", "split": split,
                              "candidate_set_id": candidate_set_id, "target_state_id": preset["state_id"],
                              "target_family_id": preset.get("family_id", preset["state_id"]),
                              "query_midi_id": query["midi_id"], "gallery_midi_id": gallery["midi_id"],
                              "performance_source_split": query.get("performance_source_split", split),
                              "query_index": query_index, "gallery_index": gallery_index}
                    query_stats, gallery_stats = query.get("stats", {}), gallery.get("stats", {})
                    common["strata"] = {
                        "query_polyphony": "polyphonic" if query_stats.get("max_polyphony", 1) > 1 else "monophonic",
                        "query_articulation": query_stats.get("articulation", "unknown"),
                        "query_expression": "expressive" if any(query_stats.get(key, 0) for key in
                                                                     ("sustain_events", "expression_events", "pitch_bend_events", "channel_pressure_events")) else "plain",
                        "query_duration_band": _duration_band(query_stats.get("duration_seconds")),
                        "pitch_median_gap": abs(query_stats.get("pitch_median", 0) - gallery_stats.get("pitch_median", 0)),
                        "pitch_register_gap_band": _pitch_gap_band(abs(query_stats.get("pitch_median", 0) - gallery_stats.get("pitch_median", 0))),
                        "duration_gap_seconds": round(abs(query_stats.get("duration_seconds", 0) - gallery_stats.get("duration_seconds", 0)), 3),
                        "duration_gap_band": _duration_gap_band(abs(query_stats.get("duration_seconds", 0) - gallery_stats.get("duration_seconds", 0))),
                        "query_note_density_band": _density_band(query_stats.get("notes_per_second")),
                        "target_source_family": preset.get("family_id", preset["state_id"]),
                    }
                    common["episode_id"] = digest(common)
                    episodes.append(common)
                    assignment = {}
                    for candidate in members:
                        mixed_choices = [row for row in gallery_pool if row["midi_id"] != query["midi_id"]]
                        position = int(hashlib.sha256((common["episode_id"] + candidate["state_id"]).encode()).hexdigest()[:16], 16) % len(mixed_choices)
                        assignment[candidate["state_id"]] = mixed_choices[position]["midi_id"]
                    mixed = {**common, "condition": "cross_midi_mixed_gallery", "assignment_seed": common["episode_id"],
                             "gallery_assignments_id": digest(assignment)}
                    mixed["episode_id"] = digest(mixed)
                    episodes.append(mixed)
    episodes.sort(key=lambda row: row["episode_id"])
    return candidate_sets, episodes


def check_render_budget(preset_count: int, performance_count: int, maximum: int) -> int:
    if min(preset_count, performance_count) < 0 or maximum < 1:
        raise ValueError("render counts must be non-negative and maximum must be positive")
    requests = preset_count * performance_count
    if requests > maximum:
        raise ValueError(f"full preset × MIDI render matrix requires {requests:,} requests, above the configured "
                         f"budget of {maximum:,}; raise --max-render-requests explicitly or choose a smaller source split")
    return requests


def _duration_band(value):
    if value is None:
        return "unknown"
    return "2-4s" if value < 4 else "4-6s" if value < 6 else "6-8s"


def _density_band(value):
    if value is None:
        return "unknown"
    return "sparse" if value < 2 else "moderate" if value < 5 else "dense"


def _pitch_gap_band(value):
    return "0-5 semitones" if value < 6 else "6-12 semitones" if value < 13 else "13+ semitones"


def _duration_gap_band(value):
    return "0-1s" if value <= 1 else "1-3s" if value <= 3 else "3+s"


def mixed_gallery_midi_id(episode: dict, state_id: str, candidates: dict[str, dict]) -> str:
    """Recreate a mixed-gallery assignment without storing one MIDI ID per candidate in every episode."""
    pool = candidates[episode["candidate_set_id"]]["gallery_midi_ids"]
    choices = [midi_id for midi_id in pool if midi_id != episode["query_midi_id"]]
    if not choices:
        raise ValueError("mixed gallery has no MIDI distinct from its query")
    position = int(hashlib.sha256((episode.get("assignment_seed", episode["episode_id"]) + state_id).encode()).hexdigest()[:16], 16) % len(choices)
    return choices[position]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("cosine vectors must be non-empty and have equal dimensions")
    if any(not math.isfinite(float(value)) for value in (*left, *right)):
        raise ValueError("cosine vectors must contain only finite values")
    dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("zero-norm embedding cannot be compared")
    return dot / (left_norm * right_norm)


def _rank_metrics(scores: Sequence[tuple[str, float]], target: str, *, ks: tuple[int, ...] = (1, 5)) -> dict:
    if target not in {candidate for candidate, _ in scores}:
        raise ValueError("target is not in the frozen candidate set")
    target_score = next(score for candidate, score in scores if candidate == target)
    higher = sum(score > target_score + 1e-12 for _, score in scores)
    tied = sum(abs(score - target_score) <= 1e-12 for _, score in scores)
    first, last = higher + 1, higher + tied
    result = {f"recall_at_{k}": min(max(k - higher, 0), tied) / tied for k in ks}
    result["mrr"] = sum(1.0 / rank for rank in range(first, last + 1)) / tied
    result["rank_expected"] = (first + last) / 2
    result["tie_count"] = tied
    return result


def retrieval_metrics(episodes: Sequence[dict], candidates: dict[str, dict], embeddings: dict[str, Sequence[float]],
                      *, gallery_clip: callable, query_clip: callable, mixed_gallery_clip: callable | None = None,
                      same_midi_gallery_clip: callable | None = None) -> tuple[dict, list[dict]]:
    rows = []
    for episode in episodes:
        candidate_set = candidates[episode["candidate_set_id"]]
        query = embeddings[query_clip(episode)]
        scores = []
        for state_id in candidate_set["state_ids"]:
            if episode["condition"] == "cross_midi_mixed_gallery":
                if mixed_gallery_clip is None:
                    raise ValueError("mixed-gallery evaluation requires a deterministic mixed-gallery assignment")
                gallery_key = mixed_gallery_clip(episode, state_id)
            else:
                gallery_key = gallery_clip(state_id, episode["gallery_midi_id"])
            score = cosine(query, embeddings[gallery_key])
            scores.append((state_id, score))
        metrics = _rank_metrics(scores, episode["target_state_id"])
        metrics["target_cosine"] = next(score for state_id, score in scores if state_id == episode["target_state_id"])
        family_scores = defaultdict(list)
        family_by_state = candidate_set.get("family_by_state", {})
        for state_id, score in scores:
            family_scores[family_by_state.get(state_id, state_id)].append(score)
        family_rank = _rank_metrics([(family, max(values)) for family, values in family_scores.items()],
                                    episode["target_family_id"])
        if same_midi_gallery_clip is not None:
            same_midi_scores = [cosine(query, embeddings[same_midi_gallery_clip(state_id, episode["query_midi_id"])])
                                for state_id in candidate_set["state_ids"] if state_id != episode["target_state_id"]]
            metrics["same_midi_negative_win"] = float(bool(same_midi_scores) and metrics["target_cosine"] > max(same_midi_scores))
            metrics["same_midi_negative_max_cosine"] = max(same_midi_scores, default=None)
        metrics.update({f"family_{key}": value for key, value in family_rank.items()
                        if key in {"recall_at_1", "recall_at_5", "mrr", "rank_expected"}})
        rows.append({**episode, **metrics,
                     "gallery_size": len(scores)})
    aggregate = {key: sum(row[key] for row in rows) / len(rows) if rows else None
                 for key in ("recall_at_1", "recall_at_5", "mrr", "rank_expected")}
    aggregate["episodes"] = len(rows)
    aggregate["gallery_size"] = sorted({row["gallery_size"] for row in rows})
    if rows and "same_midi_negative_win" in rows[0]:
        aggregate["same_midi_negative_win_rate"] = sum(row["same_midi_negative_win"] for row in rows) / len(rows)
    if rows and "family_recall_at_1" in rows[0]:
        aggregate["family_recall_at_1"] = sum(row["family_recall_at_1"] for row in rows) / len(rows)
    return aggregate, rows


def paired_cluster_bootstrap(left: Sequence[dict], right: Sequence[dict], *, metric: str = "recall_at_1", seed: int = 0,
                             replicates: int = 2000) -> dict:
    """Paired family-cluster bootstrap for identical episode IDs across encoders."""
    left_by_id = {row["episode_id"]: row for row in left}
    right_by_id = {row["episode_id"]: row for row in right}
    if replicates < 100 or left_by_id.keys() != right_by_id.keys() or not left_by_id:
        raise ValueError("paired bootstrap requires identical non-empty episode sets")
    family_values = defaultdict(list)
    for episode_id, row in left_by_id.items():
        other = right_by_id[episode_id]
        family = row["target_family_id"]
        if family != other["target_family_id"]:
            raise ValueError("paired episode family differs")
        family_values[family].append(float(row[metric]) - float(other[metric]))
    means = [sum(values) / len(values) for _, values in sorted(family_values.items())]
    observed = sum(means) / len(means)
    rng, draws = random.Random(seed), []
    for _ in range(replicates):
        draws.append(sum(means[rng.randrange(len(means))] for _ in means) / len(means))
    draws.sort()
    return {"metric": metric, "difference": observed, "lower_95": draws[int(0.025 * (replicates - 1))],
            "upper_95": draws[int(0.975 * (replicates - 1))], "family_clusters": len(means),
            "replicates": replicates, "seed": seed}


def _bootstrap_interval(cluster_values: dict[str, list[float]], *, metric: str, seed: int,
                        replicates: int, cluster_name: str) -> dict:
    if replicates < 100 or not cluster_values:
        raise ValueError("bootstrap needs clusters and at least 100 replicates")
    means = [sum(values) / len(values) for _, values in sorted(cluster_values.items())]
    rng, draws = random.Random(seed), []
    for _ in range(replicates):
        draws.append(sum(means[rng.randrange(len(means))] for _ in means) / len(means))
    draws.sort()
    return {"metric": metric, "estimate": sum(means) / len(means),
            "lower_95": draws[int(0.025 * (replicates - 1))],
            "upper_95": draws[int(0.975 * (replicates - 1))],
            f"{cluster_name}_clusters": len(means), "replicates": replicates, "seed": seed}


def performance_cluster_bootstrap(rows: Sequence[dict], *, metric: str = "recall_at_1", seed: int = 0,
                                  replicates: int = 2000) -> dict:
    """Resample independent normalized MIDI identities, giving each performance equal weight."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("query_midi_id", row["episode_id"])].append(float(row[metric]))
    return _bootstrap_interval(grouped, metric=metric, seed=seed, replicates=replicates,
                               cluster_name="performance")


def two_way_cluster_bootstrap(rows: Sequence[dict], *, metric: str = "recall_at_1", seed: int = 0,
                              replicates: int = 2000) -> dict:
    """Resample independent performances and preset families as crossed clusters."""
    if replicates < 100 or not rows:
        raise ValueError("bootstrap needs rows and at least 100 replicates")
    cells = defaultdict(list)
    for row in rows:
        cells[(row.get("query_midi_id", row["episode_id"]), row.get("target_family_id", row["episode_id"]))].append(float(row[metric]))
    cell_means = {key: sum(values) / len(values) for key, values in cells.items()}
    performances = sorted({key[0] for key in cell_means})
    families = sorted({key[1] for key in cell_means})
    observed = sum(cell_means.values()) / len(cell_means)
    rng = random.Random(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None and hasattr(np, "full") and len(performances) * len(families) >= 500:
        performance_index = {value: index for index, value in enumerate(performances)}
        family_index = {value: index for index, value in enumerate(families)}
        matrix = np.zeros((len(performances), len(families)), dtype=np.float64)
        valid = np.zeros(matrix.shape, dtype=np.bool_)
        for (performance, family), value in cell_means.items():
            row, column = performance_index[performance], family_index[family]
            matrix[row, column], valid[row, column] = value, True
        sampled_performances = np.empty((replicates, len(performances)), dtype=np.intp)
        sampled_families = np.empty((replicates, len(families)), dtype=np.intp)
        for replicate in range(replicates):
            sampled_performances[replicate] = [rng.randrange(len(performances)) for _ in performances]
            sampled_families[replicate] = [rng.randrange(len(families)) for _ in families]
        row_indexes, column_indexes = sampled_performances[:, :, None], sampled_families[:, None, :]
        selected_valid = valid[row_indexes, column_indexes]
        counts = selected_valid.sum(axis=(1, 2))
        totals = np.where(selected_valid, matrix[row_indexes, column_indexes], 0.0).sum(axis=(1, 2))
        draws = np.divide(totals, counts, out=np.full(replicates, np.nan), where=counts > 0).tolist()
        for replicate, count in enumerate(counts):
            while count == 0:
                rows = [rng.randrange(len(performances)) for _ in performances]
                columns = [rng.randrange(len(families)) for _ in families]
                sampled = valid[np.ix_(rows, columns)]
                count = int(sampled.sum())
                if count:
                    draws[replicate] = float(matrix[np.ix_(rows, columns)][sampled].mean())
    else:
        draws = []
        for _ in range(replicates):
            sampled_performances = [performances[rng.randrange(len(performances))] for _ in performances]
            sampled_families = [families[rng.randrange(len(families))] for _ in families]
            values = [cell_means[(performance, family)] for performance in sampled_performances
                      for family in sampled_families if (performance, family) in cell_means]
            draws.append(sum(values) / len(values))
    draws.sort()
    return {"metric": metric, "estimate": observed,
            "lower_95": draws[int(0.025 * (replicates - 1))],
            "upper_95": draws[int(0.975 * (replicates - 1))],
            "performance_clusters": len(performances), "family_clusters": len(families),
            "replicates": replicates, "seed": seed}


def paired_performance_bootstrap(left: Sequence[dict], right: Sequence[dict], *, metric: str = "recall_at_1",
                                 seed: int = 0, replicates: int = 2000) -> dict:
    left_by_id, right_by_id = ({row["episode_id"]: row for row in rows} for rows in (left, right))
    if replicates < 100 or left_by_id.keys() != right_by_id.keys() or not left_by_id:
        raise ValueError("paired bootstrap requires identical non-empty episode sets")
    grouped = defaultdict(list)
    for episode_id, row in left_by_id.items():
        other = right_by_id[episode_id]
        if row.get("query_midi_id", episode_id) != other.get("query_midi_id", episode_id):
            raise ValueError("paired episode performance differs")
        grouped[row["query_midi_id"]].append(float(row[metric]) - float(other[metric]))
    result = _bootstrap_interval(grouped, metric=metric, seed=seed, replicates=replicates,
                                 cluster_name="performance")
    result["difference"] = result.pop("estimate")
    return result


def paired_two_way_cluster_bootstrap(left: Sequence[dict], right: Sequence[dict], *,
                                     metric: str = "recall_at_1", seed: int = 0,
                                     replicates: int = 2000) -> dict:
    left_by_id, right_by_id = ({row["episode_id"]: row for row in rows} for rows in (left, right))
    if replicates < 100 or left_by_id.keys() != right_by_id.keys() or not left_by_id:
        raise ValueError("paired bootstrap requires identical non-empty episode sets")
    differences = []
    for episode_id, row in left_by_id.items():
        other = right_by_id[episode_id]
        left_clusters = (row.get("query_midi_id", episode_id), row.get("target_family_id", episode_id))
        right_clusters = (other.get("query_midi_id", episode_id), other.get("target_family_id", episode_id))
        if left_clusters != right_clusters:
            raise ValueError("paired episode performance or family differs")
        differences.append({**row, metric: float(row[metric]) - float(other[metric])})
    result = two_way_cluster_bootstrap(differences, metric=metric, seed=seed, replicates=replicates)
    result["difference"] = result.pop("estimate")
    return result


def performance_ranking_bootstrap(rows_by_arm: dict[str, Sequence[dict]], *, metric: str = "recall_at_1",
                                  seed: int = 1729, replicates: int = 2000) -> dict:
    """Estimate per-performance rank leads and bootstrap top-rank frequency."""
    if replicates < 100 or len(rows_by_arm) < 2:
        raise ValueError("performance ranking needs at least two arms and 100 replicates")
    scores = {}
    for arm, rows in rows_by_arm.items():
        grouped = defaultdict(list)
        for row in rows:
            grouped[row.get("query_midi_id", row["episode_id"])].append(float(row[metric]))
        scores[arm] = {performance: sum(values) / len(values) for performance, values in grouped.items()}
    performances = sorted(set.intersection(*(set(values) for values in scores.values())))
    if not performances:
        raise ValueError("performance ranking requires shared performance IDs")

    def winners(sampled: Sequence[str]) -> list[str]:
        means = {arm: sum(values[performance] for performance in sampled) / len(sampled)
                 for arm, values in scores.items()}
        best = max(means.values())
        return [arm for arm, value in means.items() if abs(value - best) <= 1e-12]

    first_places = Counter(arm for performance in performances for arm in winners([performance]))
    rng, lead_counts = random.Random(seed), Counter()
    for _ in range(replicates):
        sampled = [performances[rng.randrange(len(performances))] for _ in performances]
        leaders = winners(sampled)
        for arm in leaders:
            lead_counts[arm] += 1 / len(leaders)
    return {arm: {"performance_clusters": len(performances),
                  "mean_performance_metric": sum(scores[arm][performance] for performance in performances) / len(performances),
                  "first_place_performances": first_places[arm],
                  "first_place_fraction": first_places[arm] / len(performances),
                  "bootstrap_top_rank_probability": lead_counts[arm] / replicates,
                  "replicates": replicates, "seed": seed}
            for arm in sorted(rows_by_arm)}


def cluster_bootstrap(values: Sequence[tuple[str, float]], *, seed: int = 0, replicates: int = 2000) -> dict:
    """Percentile interval over preset families, resampling family means rather than clips."""
    if replicates < 100 or not values:
        raise ValueError("bootstrap needs values and at least 100 replicates")
    clusters = defaultdict(list)
    for family_id, value in values:
        clusters[family_id].append(float(value))
    means = [sum(items) / len(items) for _, items in sorted(clusters.items())]
    observed = sum(means) / len(means)
    rng, draws = random.Random(seed), []
    for _ in range(replicates):
        draws.append(sum(means[rng.randrange(len(means))] for _ in means) / len(means))
    draws.sort()
    return {"estimate": observed, "lower_95": draws[int(0.025 * (replicates - 1))],
            "upper_95": draws[int(0.975 * (replicates - 1))], "family_clusters": len(means),
            "replicates": replicates, "seed": seed}


def masked_temporal_pools(frames: Sequence[Sequence[float]], *, valid_frames: int, bins: int = 8) -> dict[str, list[float]]:
    """Training-free mean and order-aware pyramid; invalid/padded frames never enter a pool."""
    if not frames or not bins <= valid_frames <= len(frames):
        raise ValueError("sequence must contain at least one valid frame per temporal bin")
    dimension = len(frames[0])
    if dimension == 0 or any(len(frame) != dimension or any(not math.isfinite(float(value)) for value in frame) for frame in frames):
        raise ValueError("sequence feature dimensions must be non-empty and consistent")
    valid = frames[:valid_frames]

    def moments(part: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
        count = len(part)
        means = [sum(float(row[index]) for row in part) / count for index in range(dimension)]
        stds = [math.sqrt(sum((float(row[index]) - means[index]) ** 2 for row in part) / count)
                for index in range(dimension)]
        return means, stds

    mean, _ = moments(valid)
    pyramid = []
    for bin_index in range(bins):
        start = bin_index * valid_frames // bins
        end = (bin_index + 1) * valid_frames // bins
        part = valid[start:end]
        bin_mean, bin_std = moments(part)
        pyramid.extend(bin_mean + bin_std)
    return {"masked_mean": mean, f"temporal_pyramid_{bins}": pyramid}


def random_baseline(gallery_sizes: Sequence[int]) -> dict:
    if not gallery_sizes or any(size < 1 for size in gallery_sizes):
        raise ValueError("gallery sizes must be positive")
    recall1 = sum(1 / size for size in gallery_sizes) / len(gallery_sizes)
    recall5 = sum(min(5, size) / size for size in gallery_sizes) / len(gallery_sizes)
    mrr = sum(sum(1 / rank for rank in range(1, size + 1)) / size for size in gallery_sizes) / len(gallery_sizes)
    return {"recall_at_1": recall1, "recall_at_5": recall5, "mrr": mrr, "episodes": len(gallery_sizes)}
