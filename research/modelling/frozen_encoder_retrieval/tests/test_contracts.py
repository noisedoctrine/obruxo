from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import importlib.util
import math
import types
import unittest
from unittest.mock import patch


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[2]
for directory in (PACKAGE, REPO / "research/data_generation", REPO / "research/training"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from frozen_retrieval import core
from frozen_retrieval.dataset import (_file_sha256, _paired_source_midi, _performance_tags,
                                     _remove_nonperformance_meta_events, _select_diverse_performances,
                                     bounded_excerpt, midi_features,
                                     validate_dataset_manifests)
from frozen_retrieval.encoders import (_beats_sequence, _clap_roberta_config, _decode_audio, _freeze_model,
                                       _import_source_module_without_bytecode, _make_muq_audio_initializer, cache_feature,
                                       checkpoint_sha256, resolve_checkpoint, resolve_source_checkout)
from frozen_retrieval import evaluation
from frozen_retrieval.encoders import _load_cir_ast_offline, _load_muq_mulan_offline, _resolve_local_configs
from frozen_retrieval.rendering import (QA_POLICY_ID, RENDER_POLICY_ID, _accepted, _qa_warnings,
                                        _receipt_matches_cached_audio, render_plan_identity)
from frozen_retrieval.reporting import (_retrieval_explainer_lines, _control_group, _inventory_summary, _performance_profile, build_control_inventory, build_controlled_inventory,
                                        generate_report, _issue155_render_quality, _execution_status_lines,
                                        _collapse_issue155_rank_rows, _issue155_rank_statistics,
                                        _issue155_rank_failure_analysis)
import run_controlled_perturbation as controlled
import run_controlled_candidate_shortlist as candidate_screen
import run_issue155_validation as issue155
import run_issue164_normalization as issue164


def _presets():
    return [{"state_id": f"state-{index}", "family_id": f"family-{index // 2}", "split": "validation"}
            for index in range(3)]


def _performances():
    return [{"midi_id": f"midi-{index}", "source_state_id": f"state-{index}", "split": "validation",
             "stats": {"max_polyphony": 1 + index % 2, "articulation": "held", "duration_seconds": 3 + index,
                       "pitch_median": 48 + index * 12, "notes_per_second": 2 + index}}
            for index in range(3)]


class _Vector(list):
    def reshape(self, *_shape):
        return _Matrix([self])

    def tolist(self):
        return list(self)


class _Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    def astype(self, _dtype):
        return self

    def reshape(self, *_shape):
        return _Matrix([self.rows[0]])

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return self.rows[index[0]][index[1]]
        return _Vector(self.rows[index])

    def tolist(self):
        return self.rows


class _FakeNumpy:
    float32 = object()

    @staticmethod
    def stack(values):
        return _Matrix(values)


def _fake_cosine_matrix(left, right, _np):
    scores = []
    for query in left.rows:
        row = []
        for candidate in right.rows:
            denominator = math.sqrt(sum(value * value for value in query) * sum(value * value for value in candidate))
            row.append(sum(a * b for a, b in zip(query, candidate, strict=True)) / denominator)
        scores.append(row)
    return _Matrix(scores)


class CheckpointLockTests(unittest.TestCase):
    def test_basic_pitch_trunk_contract_uses_frozen_semitone_activations(self):
        spec = core.MODEL_REGISTRY["basic_pitch_trunk"]
        self.assertEqual("sequence", spec["kind"])
        self.assertEqual(704, spec["dimension"])
        self.assertIn("contour_conv1", spec["tap"])
        self.assertIn("3 bins per semitone", spec["tap"])
        self.assertEqual("obruxo_basic_pitch@frozen_contour_trunk_v1", spec["loader_revision"])

    def _lock(self, root: Path, license_status: str) -> dict:
        checkpoint = root / "clap.bin"
        checkpoint.write_bytes(b"fixture checkpoint")
        runtime_versions = {"numpy": "test", "scipy": "test", "torch": "test", "laion_clap": "test",
                            "transformers": "test"}
        return {"models": {"clap_music": {"repository_id": "lukewys/laion_clap", "path": checkpoint.name,
                                            "sha256": checkpoint_sha256(checkpoint), "revision": "a" * 40,
                                            "loader_revision": "laion_clap==test", "runtime_versions": runtime_versions,
                                            "license_status": license_status}}}

    def test_research_use_with_unverified_terms_is_not_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lock = self._lock(root, "research-use-unverified")
            path, entry, digest = resolve_checkpoint("clap_music", root, lock)
            self.assertEqual(path.name, "clap.bin")
            self.assertEqual(entry["license_status"], "research-use-unverified")
            self.assertEqual(digest, entry["sha256"])

    def test_explicit_research_prohibition_stops_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lock = self._lock(root, "explicitly-prohibited")
            with self.assertRaisesRegex(PermissionError, "explicitly prohibit"):
                resolve_checkpoint("clap_music", root, lock)

    def test_locked_source_checkout_resolves_under_checkpoint_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkout = root / "sources" / "beats"
            checkout.mkdir(parents=True)
            self.assertEqual(checkout.resolve(), resolve_source_checkout(root, {"source_root": "sources/beats"}))
            with self.assertRaisesRegex(ValueError, "stay under"):
                resolve_source_checkout(root, {"source_root": "../beats"})


class SourceLoaderTests(unittest.TestCase):
    def test_source_import_does_not_dirty_pinned_checkout_with_bytecode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            name = f"_issue152_source_loader_{id(self)}"
            (root / f"{name}.py").write_text("VALUE = 1\n", encoding="utf-8")
            sys.path.insert(0, str(root))
            try:
                module = _import_source_module_without_bytecode(name)
                self.assertEqual(module.VALUE, 1)
                self.assertFalse((root / "__pycache__").exists())
            finally:
                sys.modules.pop(name, None)
                sys.path.remove(str(root))


class EpisodeContractTests(unittest.TestCase):
    def test_full_matrix_budget_fails_before_silent_preset_or_midi_truncation(self):
        with self.assertRaisesRegex(ValueError, "817,216 requests"):
            core.check_render_budget(904, 904, 50_000)
        self.assertEqual(17_161, core.check_render_budget(131, 131, 50_000))

    def test_every_state_queries_every_unique_midi_and_gallery_is_cross_performance(self):
        candidate_sets, episodes = core.build_episodes(_presets(), _performances(), gallery_cap=3)
        self.assertEqual(18, len(episodes))
        self.assertEqual(1, len(candidate_sets))
        candidate = next(iter(candidate_sets.values()))
        self.assertEqual({row["state_id"] for row in _presets()}, set(candidate["state_ids"]))
        self.assertEqual(["state-1"], candidate["known_near_neighbor_state_ids_by_state"]["state-0"])
        common = [row for row in episodes if row["condition"] == "cross_midi_common_gallery"]
        self.assertEqual(9, len(common))
        self.assertEqual({"midi-0", "midi-1", "midi-2"}, {row["query_midi_id"] for row in common})
        self.assertTrue(all(row["query_midi_id"] != row["gallery_midi_id"] for row in common))
        self.assertEqual((candidate_sets, episodes), core.build_episodes(_presets(), _performances(), gallery_cap=3))

    def test_mixed_gallery_assignments_are_deterministic_and_query_distinct(self):
        candidates, episodes = core.build_episodes(_presets(), _performances(), gallery_cap=3)
        episode = next(row for row in episodes if row["condition"] == "cross_midi_mixed_gallery")
        candidate = candidates[episode["candidate_set_id"]]
        first = {state: core.mixed_gallery_midi_id(episode, state, candidates) for state in candidate["state_ids"]}
        second = {state: core.mixed_gallery_midi_id(episode, state, candidates) for state in candidate["state_ids"]}
        self.assertEqual(first, second)
        self.assertTrue(set(first.values()).issubset(candidate["gallery_midi_ids"]))
        self.assertTrue(all(midi_id != episode["query_midi_id"] for midi_id in first.values()))

    def test_retrieval_uses_other_patch_same_midi_as_hard_negative(self):
        candidate = {"state_ids": ["target", "near", "far"],
                     "family_by_state": {"target": "target-family", "near": "near-family", "far": "far-family"}}
        candidates = {"candidates": candidate}
        episode = {"episode_id": "episode", "candidate_set_id": "candidates", "condition": "cross_midi_common_gallery",
                   "target_state_id": "target", "target_family_id": "target-family", "query_midi_id": "query",
                   "gallery_midi_id": "gallery"}
        embeddings = {("target", "query"): [1.0, 0.0], ("target", "gallery"): [1.0, 0.0],
                      ("near", "gallery"): [0.0, 1.0], ("far", "gallery"): [-1.0, 0.0],
                      ("near", "query"): [0.8, 0.6], ("far", "query"): [0.0, 1.0]}
        result, rows = core.retrieval_metrics([episode], candidates, embeddings,
                                               gallery_clip=lambda state, midi: (state, midi),
                                               query_clip=lambda item: (item["target_state_id"], item["query_midi_id"]),
                                               same_midi_gallery_clip=lambda state, midi: (state, midi))
        self.assertEqual(1.0, result["recall_at_1"])
        self.assertEqual(1.0, rows[0]["same_midi_negative_win"])
        self.assertEqual("target-family", rows[0]["target_family_id"])

    def test_tied_ranks_are_fractional_and_random_baseline_matches_gallery_sizes(self):
        tied = core._rank_metrics([("a", 0.5), ("b", 0.5), ("c", 0.5)], "b")
        self.assertAlmostEqual(1 / 3, tied["recall_at_1"])
        self.assertEqual(1.0, tied["recall_at_5"])
        self.assertAlmostEqual((1 + 0.5 + 1 / 3) / 3, tied["mrr"])
        self.assertEqual(2, core.random_baseline([2, 5])["episodes"])

    def test_family_bootstraps_are_reproducible_and_require_paired_episode_ids(self):
        left = [{"episode_id": "a", "target_family_id": "f1", "recall_at_1": 1.0},
                {"episode_id": "b", "target_family_id": "f2", "recall_at_1": 0.0}]
        right = [{"episode_id": "a", "target_family_id": "f1", "recall_at_1": 0.0},
                 {"episode_id": "b", "target_family_id": "f2", "recall_at_1": 0.0}]
        first = core.paired_cluster_bootstrap(left, right, replicates=100)
        second = core.paired_cluster_bootstrap(left, right, replicates=100)
        self.assertEqual(first, second)
        self.assertAlmostEqual(0.5, first["difference"])
        with self.assertRaises(ValueError):
            core.paired_cluster_bootstrap(left, right[:1], replicates=100)

    def test_performance_and_crossed_cluster_bootstraps_cover_both_sampling_axes(self):
        left = [{"episode_id": f"{midi}-{family}", "query_midi_id": midi, "target_family_id": family,
                 "recall_at_1": value}
                for midi, values in (("m1", (1.0, 0.0)), ("m2", (0.0, 1.0)))
                for family, value in zip(("f1", "f2"), values, strict=True)]
        right = [{**row, "recall_at_1": 0.0} for row in left]
        performance = core.performance_cluster_bootstrap(left, replicates=100)
        crossed = core.two_way_cluster_bootstrap(left, replicates=100)
        paired = core.paired_two_way_cluster_bootstrap(left, right, replicates=100)
        self.assertEqual(2, performance["performance_clusters"])
        self.assertEqual((2, 2), (crossed["performance_clusters"], crossed["family_clusters"]))
        self.assertAlmostEqual(0.5, paired["difference"])

    def test_crossed_bootstrap_vectorizes_large_performance_family_grids(self):
        rows = [{"episode_id": f"p{performance}-f{family}", "query_midi_id": f"p{performance}",
                 "target_family_id": f"f{family}", "recall_at_1": float((performance + family) % 2)}
                for performance in range(25) for family in range(20)]
        result = core.two_way_cluster_bootstrap(rows, replicates=100)
        self.assertAlmostEqual(0.5, result["estimate"])
        self.assertEqual((25, 20), (result["performance_clusters"], result["family_clusters"]))
        self.assertEqual(result, core.two_way_cluster_bootstrap(rows, replicates=100))

    def test_performance_ranking_bootstrap_counts_performance_leads(self):
        rows = {
            "a": [{"episode_id": "a1", "query_midi_id": "p1", "recall_at_1": 1.0},
                  {"episode_id": "a2", "query_midi_id": "p2", "recall_at_1": 0.0}],
            "b": [{"episode_id": "b1", "query_midi_id": "p1", "recall_at_1": 0.0},
                  {"episode_id": "b2", "query_midi_id": "p2", "recall_at_1": 1.0}],
        }
        result = core.performance_ranking_bootstrap(rows, replicates=100)
        self.assertEqual(result, core.performance_ranking_bootstrap(rows, replicates=100))
        self.assertEqual(2, result["a"]["performance_clusters"])
        self.assertEqual(1, result["a"]["first_place_performances"])
        self.assertEqual(1, result["b"]["first_place_performances"])
        self.assertAlmostEqual(1.0, sum(row["bootstrap_top_rank_probability"] for row in result.values()))

    def test_midi_diversity_selector_is_deterministic_and_tag_balanced(self):
        rows = [{"midi_id": f"m{index}", "stats": {
            "max_polyphony": 1 + index % 2, "articulation": "held" if index < 4 else "phrased",
            "pitch_median": 36 + index * 8, "notes_per_second": float(index),
            "velocity_min": 40, "velocity_max": 40 + index,
            "pitch_bend_events": index % 2, "sustain_events": 0,
            "expression_events": 0, "channel_pressure_events": 0}}
            for index in range(8)]
        selected = _select_diverse_performances(rows, 4)
        self.assertEqual(selected, _select_diverse_performances(list(reversed(rows)), 4))
        self.assertEqual(4, len(selected))
        tags = set().union(*(_performance_tags(row) for row in selected))
        self.assertTrue({"polyphony:mono", "polyphony:poly", "velocity:varied", "expression:present"} <= tags)

    def test_controlled_query_and_gallery_are_distinct_randomized_renders(self):
        query, candidates = controlled._cross_seed_keys("original", "midi-1",
                                                         ["original", "filter", "lfo"], "source", "independent")
        self.assertNotIn(query, candidates)
        self.assertTrue(all(key[1] == "midi-1" and key[2] == "independent" for key in candidates))
        with self.assertRaisesRegex(ValueError, "distinct independent gallery render seed"):
            controlled._cross_seed_keys("original", "midi-1", ["original", "filter"], "source", "source")

    def test_controlled_diagnostic_keeps_all_distinct_performance_clusters(self):
        performances = [{"midi_id": f"m{index:02}"} for index in range(50)]
        chosen = controlled._select_diagnostic_performances(list(reversed(performances)))
        self.assertEqual(50, len(chosen))
        self.assertEqual(sorted(row["midi_id"] for row in performances), [row["midi_id"] for row in chosen])
        with self.assertRaisesRegex(ValueError, "three distinct real MIDI excerpts"):
            controlled._select_diagnostic_performances(performances[:-1] + [performances[0]])

    def test_render_quality_strata_include_performance_uncertainty(self):
        rows = [{"episode_id": f"{midi}-{family}", "query_midi_id": midi,
                 "target_family_id": family, "recall_at_1": float(midi == "m1"),
                 "recall_at_5": 1.0, "mrr": float(midi == "m1"), "family_recall_at_1": 1.0,
                 "strata": {"query_render_quality": quality,
                            "correct_match_gallery_render_quality": "clean"}}
                for midi in ("m1", "m2") for family, quality in (("f1", "clean"), ("f2", "warning"))]
        strata = evaluation._strata(rows)
        clean = strata["query_render_quality"]["clean"]
        warning = strata["query_render_quality"]["warning"]
        self.assertEqual(2, clean["performance_cluster_bootstrap"]["performance_clusters"])
        self.assertEqual(2, warning["two_way_cluster_bootstrap"]["performance_clusters"])

    def test_one_warned_candidate_in_129_is_not_a_warning_gallery(self):
        states = [f"state-{index}" for index in range(129)]
        episode = {"episode_id": "episode", "condition": "cross_midi_common_gallery",
                   "candidate_set_id": "candidate", "target_state_id": states[0],
                   "target_family_id": "family-0", "query_midi_id": "query-midi",
                   "gallery_midi_id": "gallery-midi"}
        pair_to_request = {(state, "gallery-midi"): f"gallery-{index}"
                           for index, state in enumerate(states)}
        pair_to_request[(states[0], "query-midi")] = "query"
        receipts = {request: {"qa_status": "accepted"} for request in pair_to_request.values()}
        receipts["gallery-128"] = {"qa_status": "accepted_with_amplitude_above_unit_range"}
        candidates = {"candidate": {"state_ids": states}}
        row = {"episode_id": "episode", "query_midi_id": "query-midi", "target_family_id": "family-0",
               "recall_at_1": 1.0, "recall_at_5": 1.0, "mrr": 1.0, "family_recall_at_1": 1.0,
               "strata": {}}

        evaluation._attach_render_quality([row], [episode], candidates, pair_to_request, receipts)

        self.assertEqual("clean", row["query_render_quality"])
        self.assertEqual("clean", row["correct_match_gallery_render_quality"])
        self.assertEqual(1, row["gallery_warning_candidate_count"])
        self.assertEqual(129, row["gallery_candidate_count"])
        self.assertAlmostEqual(1 / 129, row["gallery_warning_candidate_fraction"])
        self.assertNotIn("gallery_render_quality", row)
        self.assertNotIn("gallery_render_quality", row["strata"])

    def test_issue155_gallery_offsets_use_all_50_midis_and_never_self_match(self):
        performances = [{"midi_id": f"midi-{index:02d}"} for index in range(50)]
        ordered = issue155._ordered_midi_pool(performances, "validation", (1, 7, 19))
        self.assertEqual(ordered, issue155._ordered_midi_pool(list(reversed(performances)), "validation", (1, 7, 19)))
        for offset in (1, 7, 19):
            galleries = [ordered[(index + offset) % len(ordered)]["midi_id"] for index in range(len(ordered))]
            queries = [row["midi_id"] for row in ordered]
            self.assertEqual(set(queries), set(galleries))
            self.assertTrue(all(query != gallery for query, gallery in zip(queries, galleries, strict=True)))
        with self.assertRaisesRegex(ValueError, "distinct, non-zero"):
            issue155._ordered_midi_pool(performances, "validation", (1, 1))


class FeatureAndManifestTests(unittest.TestCase):
    def test_issue164_gain_probe_rows_include_bootstrap_episode_identity(self):
        rows = []
        for index, (family_id, midi_id) in enumerate((("family-a", "midi-a"), ("family-a", "midi-b"),
                                                       ("family-b", "midi-a"), ("family-b", "midi-b"))):
            case = {"triplet_index": index, "target_state_id": f"state-{index}",
                    "candidate_state_id": f"candidate-{index}", "target_family_id": family_id,
                    "query_midi_id": midi_id}
            rows.append(issue164._gain_probe_result_row(case, "query", -6, embedding_distance=(index + 1) / 10))
        summary = issue164.evaluation_two_way_ci(rows, "embedding_distance")
        self.assertEqual(0.25, summary["estimate"])
        self.assertTrue(all(row["episode_id"].startswith("gain164:") for row in rows))

    def test_issue164_wav_mapping_uses_accepted_common_requests_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = {"request_by_pair": {("state-a", "midi-a"): "accepted"},
                     "receipts": {"accepted": {"accepted": True, "wav_relative_path": "audio/a.wav"},
                                  "rejected": {"accepted": False}}}
            self.assertEqual({("state-a", "midi-a"): root / "audio/a.wav"},
                             issue164._accepted_wav_paths(root, audit))

    def test_issue164_active_matrix_cache_rejects_old_contract_version(self):
        import numpy as np

        matrices = {index: np.zeros((129, 129), dtype=np.float32) for index in range(12)}
        identity = {"cache_contract_version": issue164.ACTIVE_MATRIX_CACHE_VERSION,
                    "normalization_version": "active-rms-v1", "normalization_target_active_rms": 0.025}
        contract = {**identity, "matrix_archive_sha256": "matrix-sha256"}
        self.assertTrue(issue164._active_matrix_cache_matches(matrices, contract, identity, "matrix-sha256"))
        stale = {**contract, "cache_contract_version": "old-active-rms-matrix-v0"}
        self.assertFalse(issue164._active_matrix_cache_matches(matrices, stale, identity, "matrix-sha256"))
        self.assertFalse(issue164._active_matrix_cache_matches(matrices, contract, identity, "changed-matrix-sha256"))

    def test_issue155_runner_exposes_basic_pitch_adapter_checkout(self):
        self.assertIn(str(REPO / "research/modelling/basic_pitch"), sys.path)

    def test_issue155_cached_assignment_rows_require_full_arm_offset_coverage(self):
        expected = {1: {"e1", "e2"}, 7: {"e3"}}
        identity = {"score_contract_version": issue155.ASSIGNMENT_SCORE_CONTRACT_VERSION,
                    "feature_cache_identity": {"identity_sha256": "cache-v1", "normalization": {
                        "version": "active-rms-v1", "target_active_rms": 0.025}}}
        rows = [{"arm": arm, "gallery_assignment_offset": offset, "episode_id": episode}
                for arm in ("arm-a", "arm-b") for offset, episode_ids in expected.items()
                for episode in episode_ids]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            issue155._write_assignment_cache_contract(path, identity)
            cached = issue155._load_cached_assignment_rows(path, {"arm-a", "arm-b"}, expected, identity)
            self.assertEqual(3, len(cached["arm-a"]))

            stale_identity = json.loads(json.dumps(identity))
            stale_identity["feature_cache_identity"]["normalization"]["target_active_rms"] = 0.03
            self.assertIsNone(issue155._load_cached_assignment_rows(
                path, {"arm-a", "arm-b"}, expected, stale_identity))

            path.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]), encoding="utf-8")
            self.assertIsNone(issue155._load_cached_assignment_rows(path, {"arm-a", "arm-b"}, expected, identity))

    def test_issue155_report_quality_collapses_assignment_repeats_by_query(self):
        rows = []
        for state_id, family_id, midi_id, query_quality in (
                ("state-a", "family-a", "midi-a", "clean"), ("state-b", "family-b", "midi-b", "warning")):
            for offset, target_quality, warning_count, recall in (
                    (1, "warning", 1, 1.0), (7, "clean", 2, 0.0), (19, "warning", 3, 1.0)):
                rows.append({"arm": "clap_music:official_clip", "episode_id": f"{state_id}-{offset}",
                    "gallery_assignment_offset": offset, "target_state_id": state_id,
                    "target_family_id": family_id, "query_midi_id": midi_id,
                    "query_render_quality": query_quality, "correct_match_gallery_render_quality": target_quality,
                    "gallery_candidate_count": 129, "gallery_warning_candidate_count": warning_count,
                    "gallery_warning_candidate_fraction": warning_count / 129,
                    "gallery_unknown_candidate_count": 0, "recall_at_1": recall, "recall_at_5": recall,
                    "mrr": recall})
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            metrics_path = Path(directory) / "gallery_assignment_episode_metrics.private.jsonl"
            metrics_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            summary = _issue155_render_quality(metrics_path)["clap_music:official_clip"]

        self.assertEqual((6, 2, 2), (summary["assignment_observations"], summary["queries"], summary["query_midi_ids"]))
        self.assertEqual(1, summary["query_render_quality"]["clean"]["queries"])
        self.assertEqual(1, summary["query_render_quality"]["warning"]["queries"])
        self.assertEqual(2, summary["correct_match_gallery_render_quality"]["clean"]["queries"])
        self.assertEqual(2, summary["mean_warning_candidates"])
        self.assertAlmostEqual(2 / 129, summary["mean_warning_fraction"], places=8)
        self.assertEqual(2, summary["correct_match_gallery_quality_overlap"]["clean_warning_overlap_queries"])
        self.assertEqual(2, summary["correct_match_gallery_quality_overlap"]["queries_in_either_group"])
        self.assertEqual(2, summary["correct_match_gallery_render_quality"]["clean"]["assignment_observations"])
        self.assertEqual(4, summary["correct_match_gallery_render_quality"]["warning"]["assignment_observations"])

    def test_issue155_rank_failures_use_individual_attempts_with_expected_ties(self):
        rows_by_arm = {"clap_music:official_clip": []}
        values = {
            ("state-a", "family-a", "midi-a"): ((1.5, 0.5), (1.5, 0.5), (1.5, 0.5)),
            ("state-a", "family-a", "midi-b"): ((4.0, 0.0), (3.0, 0.0), (5.0, 0.0)),
            ("state-b", "family-b", "midi-a"): ((6.0, 0.0), (7.0, 0.0), (5.0, 0.0)),
            ("state-b", "family-b", "midi-b"): ((9.0, 0.0), (10.0, 0.0), (8.0, 0.0)),
        }
        for (state_id, family_id, midi_id), rounds in values.items():
            for offset, (rank, recall) in zip((1, 7, 19), rounds, strict=True):
                rows_by_arm["clap_music:official_clip"].append({"target_state_id": state_id,
                    "target_family_id": family_id, "query_midi_id": midi_id,
                    "gallery_assignment_offset": offset, "rank_expected": rank,
                    "recall_at_1": recall, "recall_at_5": float(rank <= 5)})
        collapsed, offsets = _collapse_issue155_rank_rows(rows_by_arm)
        self.assertEqual([1, 7, 19], offsets)
        self.assertEqual(4, len(collapsed["clap_music:official_clip"]))
        tied = next(row for row in collapsed["clap_music:official_clip"]
                    if row["target_state_id"] == "state-a" and row["query_midi_id"] == "midi-a")
        self.assertEqual(1.5, tied["rank_expected"])
        self.assertEqual(0.5, tied["recall_at_1"])
        summary = _issue155_rank_statistics(collapsed["clap_music:official_clip"],
            attempts=rows_by_arm["clap_music:official_clip"], replicates=100)
        self.assertEqual(2, summary["performance_clusters"])
        self.assertEqual(2, summary["preset_family_clusters"])
        self.assertEqual(12, summary["individual_retrieval_attempts"])
        self.assertEqual(3, summary["assignment_rounds_per_query_pair"])
        self.assertEqual(12, len(summary["rank_expected_values"]))
        self.assertEqual(5.0, summary["rank_distribution"]["median_expected_rank"])
        self.assertEqual(8.9, summary["rank_distribution"]["p90_expected_rank"])
        self.assertAlmostEqual(7 / 12, summary["rank_distribution"]["near_miss_fraction"])
        self.assertAlmostEqual(5 / 12, summary["rank_distribution"]["far_failure_fraction"])

    def test_issue155_rank_failure_reports_all_seven_families_with_shared_resamples(self):
        families = ("clap_music", "muq_mulan", "cir_ast", "beats_iter3_plus", "dac_44khz", "encodec_24khz", "basic_pitch_trunk")
        arms = []
        for family in families:
            arms.append(f"{family}:official_clip" if family in {"clap_music", "muq_mulan", "cir_ast"}
                else f"{family}:masked_mean")
        rows = []
        for arm_index, arm in enumerate(arms):
            for state_index in range(2):
                for midi_index in range(2):
                    for offset in (1, 7, 19):
                        rank = 1.0 + arm_index + (5 * state_index) + midi_index
                        rows.append({"arm": arm, "target_state_id": f"state-{state_index}",
                            "target_family_id": f"family-{state_index}", "query_midi_id": f"midi-{midi_index}",
                            "gallery_assignment_offset": offset, "rank_expected": rank,
                            "recall_at_1": float(rank == 1), "recall_at_5": float(rank <= 5)})
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            path = Path(directory) / "episodes.private.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = _issue155_rank_failure_analysis(path, replicates=100)

        self.assertEqual(set(arms), set(result["arms"]))
        self.assertTrue(result["shared_bootstrap_resamples_across_arms"])
        self.assertTrue(all(summary["query_pairs"] == 4 and summary["individual_retrieval_attempts"] == 12
            and summary["assignment_rounds_per_query_pair"] == 3 for summary in result["arms"].values()))
        self.assertEqual({155}, {summary["bootstrap_seed"] for summary in result["arms"].values()})

    def test_controlled_candidate_screen_labels_consistency_not_audibility(self):
        def candidate(changed, repeated):
            return {"median_change_to_repeat_ratio": 5.0, "midi_measurements": [
                {"change_spectral_distance": c, "repeat_spectral_distance": r}
                for c, r in zip(changed, repeated, strict=True)]}

        strong = candidate_screen._candidate_screen_strength(candidate([.01] * 4, [.001] * 4))
        borderline = candidate_screen._candidate_screen_strength(candidate([.01, .01, .01, .001], [.001] * 4))
        weak = candidate_screen._candidate_screen_strength(candidate([.01, .001, .001, .001], [.001] * 4))
        zero_repeat = candidate_screen._candidate_screen_strength({"median_change_to_repeat_ratio": None,
            "midi_measurements": [{"change_spectral_distance": .01, "repeat_spectral_distance": .001},
                {"change_spectral_distance": 0.0, "repeat_spectral_distance": 0.0}]})
        self.assertEqual("strong", strong[0])
        self.assertEqual("borderline", borderline[0])
        self.assertEqual("weak or inconclusive", weak[0])
        self.assertIn("not an audibility claim", weak[1])
        self.assertEqual("weak or inconclusive", zero_repeat[0])
        self.assertIn("ratio is undefined", zero_repeat[1])
        measured = [{"midi_id": f"midi-{index}", "articulation": "held", "note_duration_median_seconds": 1.0,
            "change_spectral_distance": ratio * .01, "repeat_spectral_distance": .01,
            "change_to_repeat_ratio": ratio} for index, ratio in enumerate((2.0, 4.0, 6.0, 8.0))]
        summary = {"scanned_render_qualified_presets": 1, "midi_context_count": 4, "new_vital_renders": 8,
            "presets_with_valid_control_candidates": {}, "candidates": [{"candidate_id": "candidate-test",
                "state_id": "state-test", "family_id": "family-test", "display_name": None,
                "control": {"family": "filter", "field": "filter_1_cutoff", "before_raw": 0.1,
                    "after_raw": 0.3, "before_normalized": 0.1, "after_normalized": 0.3},
                "median_change_to_repeat_ratio": 8.0, "midi_measurements": measured,
                "mean_change_spectral_distance": .05, "mean_repeat_spectral_distance": .01,
                "performances_exceeding_repeat_variation": 4, "listening_examples": []}]}
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            written = candidate_screen._write_shortlist_files(Path(directory), summary)
        self.assertEqual(5.0, written["candidates"][0]["median_change_to_repeat_ratio"])

    def test_controlled_shortlist_midi_contexts_cover_short_and_long_held_and_phrased_notes(self):
        rows = []
        for articulation, durations in (("held", (0.05, 2.5)), ("phrased", (0.08, 1.2))):
            for index, duration in enumerate(durations):
                rows.append({"midi_id": f"{articulation}-{index}", "stats": {
                    "articulation": articulation, "note_duration_median_seconds": duration}})

        selected = candidate_screen._select_contexts(rows)

        self.assertEqual({row["midi_id"] for row in selected}, {"held-0", "held-1", "phrased-0", "phrased-1"})
        self.assertEqual({row["stats"]["articulation"] for row in selected}, {"held", "phrased"})
        self.assertEqual({row["stats"]["note_duration_median_seconds"] for row in selected}, {0.05, 2.5, 0.08, 1.2})

    def test_gain_adjusted_wav_is_linear_and_preserves_source_shape_precision_and_file(self):
        import numpy as np
        from scipy.io import wavfile

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            source = root / "source.wav"
            audio = np.column_stack((np.linspace(-1.8, 1.8, 2000), np.linspace(0.7, -0.7, 2000))).astype(np.float32)
            wavfile.write(source, 44_100, audio)
            original_bytes = source.read_bytes()
            normalized = issue155._normalize_wavs({"dataset": root}, [{"request_id": "clip-1",
                "state_id": "state-1", "midi_id": "midi-1", "family_id": "family-1",
                "wav_relative_path": "source.wav", "source_audio_sha256": _file_sha256(source), "peak": 1.8}], root / "output")["clip-1"]
            sample_rate, adjusted = wavfile.read(normalized["normalized_audio_path"])

            self.assertEqual((44_100, audio.shape, audio.dtype), (sample_rate, adjusted.shape, adjusted.dtype))
            self.assertAlmostEqual(0.95, float(np.max(np.abs(adjusted))), places=5)
            self.assertAlmostEqual(0.95 / 1.8, normalized["gain_linear"], places=7)
            self.assertEqual(original_bytes, source.read_bytes())

    def test_gain_preprocessing_policy_has_a_separate_feature_cache_identity(self):
        import numpy as np
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            wav = root / "same.wav"
            wav.write_bytes(b"source audio bytes")
            adapter = SimpleNamespace(name="clap_music", checkpoint_digest="checkpoint", device="cpu", sample_rate=48_000,
                lock_entry={"revision": "source", "loader_revision": "loader", "local_configs": {}},
                model=object(), numpy=np, extract=lambda _path: {"official_clip": [1.0, 0.0]})
            metadata = {"provenance": {"request_id": "same-request"}}
            original = cache_feature(adapter, wav, request_id="same-request", render_metadata=metadata,
                                     output=root / "features")
            adjusted = cache_feature(adapter, wav, request_id="same-request", render_metadata=metadata,
                output=root / "features", preprocessing_policy="linear_peak_gain_to_0.95_v1")

            self.assertNotEqual(original["feature_id"], adjusted["feature_id"])
            original_meta = json.loads((root / "features" / f"{original['feature_id']}.json").read_text())
            adjusted_meta = json.loads((root / "features" / f"{adjusted['feature_id']}.json").read_text())
            self.assertEqual("pcm_scale_stereo_mean_mono_resample_poly_no_gain_norm_v2",
                             original_meta["cache_identity"]["preprocessing"])
            self.assertNotIn("input_normalization", original_meta["cache_identity"])
            self.assertEqual("linear_peak_gain_to_0.95_v1", adjusted_meta["cache_identity"]["preprocessing"])

    def test_active_rms_cache_has_distinct_fixed_target_provenance_without_changing_original_id(self):
        import numpy as np
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            wav = root / "same.wav"
            wav.write_bytes(b"source audio bytes")
            adapter = SimpleNamespace(name="clap_music", checkpoint_digest="checkpoint", device="cpu", sample_rate=48_000,
                lock_entry={"revision": "source", "loader_revision": "loader", "local_configs": {}},
                model=object(), numpy=np, extract=lambda _path: {"official_clip": [1.0, 0.0]},
                extract_normalized=lambda _path, target: ({"official_clip": [0.0, 1.0]}, {
                    "version": "active_rms_sliding20ms_relative_minus40db_v1", "target_active_rms": target,
                    "applied_gain": 2.0, "active_rms": 0.1, "peak_to_active_rms": 1.2}))
            metadata = {"provenance": {"request_id": "same-request"}}
            original = cache_feature(adapter, wav, request_id="same-request", render_metadata=metadata,
                                     output=root / "features")
            normalized = cache_feature(adapter, wav, request_id="same-request", render_metadata=metadata,
                output=root / "features_active_rms", active_rms_target=0.08)
            original_meta = json.loads((root / "features" / f"{original['feature_id']}.json").read_text())
            normalized_meta = json.loads((root / "features_active_rms" / f"{normalized['feature_id']}.json").read_text())

            self.assertNotEqual(original["feature_id"], normalized["feature_id"])
            self.assertEqual("pcm_scale_stereo_mean_mono_resample_poly_no_gain_norm_v2",
                             original_meta["cache_identity"]["preprocessing"])
            self.assertNotIn("input_normalization", original_meta["cache_identity"])
            self.assertEqual("pcm_scale_active_rms_stereo_mean_mono_resample_poly_v1",
                             normalized_meta["cache_identity"]["preprocessing"])
            self.assertEqual(0.08, normalized_meta["cache_identity"]["input_normalization"]["target_active_rms"])
            self.assertEqual(0.08, normalized_meta["normalization"]["target_active_rms"])

    def test_clap_offline_roberta_config_matches_checkpoint_embedding_shapes(self):
        class Config:
            def __init__(self, **kwargs):
                self.values = kwargs

        config = _clap_roberta_config(Config)
        self.assertEqual({"max_position_embeddings": 514, "type_vocab_size": 1}, config.values)

    def test_pairing_ignores_damaged_source_wav_but_keeps_structural_ambiguity(self):
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            pair_dir = root / "state"
            pair_dir.mkdir()
            preset = pair_dir / "patch.vital"
            midi = pair_dir / "phrase.mid"
            wav = pair_dir / "damaged.wav"
            preset.write_text("not parsed by the structural pairing helper", encoding="utf-8")
            midi.write_text("synthetic placeholder", encoding="utf-8")
            wav.write_bytes(b"not a valid WAV")
            paired, reason = _paired_source_midi(root, pair_dir, preset.resolve())
            self.assertEqual(midi.resolve(), paired)
            self.assertIsNone(reason)
            (pair_dir / "second.wav").write_bytes(b"also ignored")
            paired, reason = _paired_source_midi(root, pair_dir, preset.resolve())
            self.assertEqual(midi.resolve(), paired)
            self.assertIsNone(reason)

    def test_only_audited_nonperformance_metadata_is_removed_from_midi(self):
        import mido
        from obruxo_data.midi import EventKind, Performance

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            path = Path(directory) / "phrase.mid"
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            midi.tracks.append(track)
            track.extend([
                mido.MetaMessage("track_name", name="private source name"),
                mido.MetaMessage("key_signature", key="C"),
                mido.MetaMessage("set_tempo", tempo=500000),
                mido.Message("control_change", channel=0, control=64, value=127),
                mido.Message("note_on", channel=0, note=60, velocity=92),
                mido.Message("note_off", channel=0, note=60, velocity=0, time=960),
                mido.Message("program_change", channel=0, program=3),
                mido.MetaMessage("end_of_track"),
            ])
            midi.save(path)
            performance = Performance.from_midi(path)
            removed = _remove_nonperformance_meta_events(performance)

        self.assertEqual({"key_signature": 1, "track_name": 1}, removed)
        kinds = [event.kind for event in performance.events]
        self.assertIn(EventKind.TEMPO, kinds)
        self.assertIn(EventKind.CONTROL_CHANGE, kinds)
        self.assertEqual(2, sum(kind in (EventKind.NOTE_ON, EventKind.NOTE_OFF) for kind in kinds))
        self.assertTrue(any(event.kind == EventKind.OPAQUE and
                            json.loads(dict(event.opaque)["message"])["type"] == "program_change"
                            for event in performance.events))

    def test_render_receipts_and_feature_cache_qualify_primary_episode_without_same_midi_diagnostic(self):
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            rows = [
                {"request_id": "target-query", "state_id": "target", "midi_id": "query", "source_kind": "newly_rendered_original_preset"},
                {"request_id": "target-gallery", "state_id": "target", "midi_id": "gallery", "source_kind": "newly_rendered_original_preset"},
                {"request_id": "near-gallery", "state_id": "near", "midi_id": "gallery", "source_kind": "newly_rendered_original_preset"},
                {"request_id": "far-gallery", "state_id": "far", "midi_id": "gallery", "source_kind": "newly_rendered_original_preset"},
                {"request_id": "near-same-midi", "state_id": "near", "midi_id": "query", "source_kind": "newly_rendered_original_preset"},
                {"request_id": "far-same-midi", "state_id": "far", "midi_id": "query", "source_kind": "newly_rendered_original_preset"},
            ]
            candidate = {"split": "validation", "state_ids": ["target", "near", "far"],
                         "family_by_state": {"target": "family", "near": "family", "far": "far-family"},
                         "known_near_neighbor_state_ids_by_state": {"target": ["near"], "near": ["target"], "far": []},
                         "gallery_midi_ids": ["gallery"]}
            episode = {"episode_id": "episode", "condition": "cross_midi_common_gallery", "split": "validation",
                       "candidate_set_id": "candidates", "target_state_id": "target", "target_family_id": "family",
                       "query_midi_id": "query", "gallery_midi_id": "gallery", "strata": {}}

            def write_jsonl(path, items):
                path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in items), encoding="utf-8")

            write_jsonl(root / "render_plan.private.jsonl", rows)
            write_jsonl(root / "episodes.private.jsonl", [episode])
            (root / "candidate_sets.private.json").write_text(json.dumps({"candidates": candidate}), encoding="utf-8")
            write_jsonl(root / "performances.private.jsonl", [
                {"midi_id": "query", "split": "validation", "stats": {}},
                {"midi_id": "gallery", "split": "validation", "stats": {}},
            ])
            manifest_names = ("render_plan.private.jsonl", "episodes.private.jsonl", "candidate_sets.private.json",
                              "performances.private.jsonl")
            summary = {"source_split": "validation", "performance_source_split": "train", "manifest_sha256": {
                name: _file_sha256(root / name) for name in manifest_names}}
            (root / "dataset_summary.private.json").write_text(json.dumps(summary), encoding="utf-8")

            receipts = []
            for row in rows:
                accepted = row["request_id"] not in {"far-same-midi", "far-gallery"}
                receipts.append({"request_id": row["request_id"], "plan_identity": render_plan_identity(row),
                                 "render_policy_id": RENDER_POLICY_ID, "qa_policy_id": QA_POLICY_ID, "accepted": accepted,
                                 "qa_status": "accepted" if accepted else "render_qa_silence",
                                 "audio_sha256": "a" * 64})
            write_jsonl(root / "render_receipts.private.jsonl", receipts)
            cache_dir = root / "features" / "clap_music"
            cache_dir.mkdir(parents=True)
            feature_requests = ("target-query", "target-gallery", "near-gallery", "near-same-midi")
            write_jsonl(cache_dir / "feature_manifest.private.jsonl", [
                {"request_id": request_id, "feature_id": f"feature-{request_id}"} for request_id in feature_requests])
            (cache_dir / "extraction.latest.json").write_text(json.dumps({"checkpoint_sha256": "b" * 64}), encoding="utf-8")
            features = {"target-query": {"official_clip": [1.0, 0.0]},
                        "target-gallery": {"official_clip": [1.0, 0.0]},
                        "near-gallery": {"official_clip": [0.0, 1.0]},
                        "near-same-midi": {"official_clip": [0.0, 1.0]}}
            fake_numpy = types.ModuleType("numpy")
            fake_numpy.stack, fake_numpy.float32 = _FakeNumpy.stack, _FakeNumpy.float32
            with patch.object(evaluation, "list_clip_cache", return_value=features), \
                    patch.object(evaluation, "_acoustic_hard_negative_candidates",
                                 return_value={"episode": [("near", 0.95)]}), \
                    patch.dict(sys.modules, {"numpy": fake_numpy}), \
                    patch.object(evaluation, "_cosine_matrix", side_effect=_fake_cosine_matrix), \
                    patch.object(evaluation, "midi_shortcut_baseline", return_value={}):
                result = evaluation.evaluate_dataset(root)
                previous_row_count = len((root / "episode_metrics.private.jsonl").read_text(encoding="utf-8").splitlines())
                extended = evaluation.evaluate_dataset(root, models=("clap_music",))

            coverage = result["coverage_by_condition"]["cross_midi_common_gallery"]
            self.assertEqual(1, coverage["planned_episodes"])
            self.assertEqual(1, coverage["qa_qualified_episodes"])
            self.assertEqual(1, coverage["common_feature_qualified_episodes_across_arms"])
            self.assertEqual(3, result["gallery_candidate_states_original"])
            self.assertEqual(2, result["gallery_candidate_states_render_qualified"])
            self.assertEqual(1, result["gallery_candidate_states_excluded"])
            scores = result["model_results"]["clap_music:official_clip"]["cross_midi_common_gallery"]
            self.assertEqual(1.0, scores["recall_at_1"])
            self.assertEqual(1.0, scores["same_midi_negative_win_rate"])
            self.assertEqual(1.0, scores["known_near_neighbor_negative_win_rate"])
            self.assertEqual(1.0, scores["acoustic_hard_negative_win_rate"])
            self.assertEqual(1, scores["same_midi_negative_win_candidates"])
            self.assertEqual(1, scores["known_near_neighbor_negative_win_candidates"])
            self.assertEqual(1, scores["acoustic_hard_negative_win_candidates"])
            self.assertEqual(1, scores["diagnostic_performance_cluster_bootstrap"][
                "same_midi_negative_win"]["performance_clusters"])
            self.assertEqual(1, scores["diagnostic_performance_cluster_bootstrap"][
                "acoustic_hard_negative_win"]["performance_clusters"])
            self.assertEqual(1, scores["strata"]["source_preset_lineage_group_size"]["multi-state"]["episodes"])
            self.assertEqual("not_applicable", result["midi_only_and_random_baselines"]["status"])
            self.assertIn("cross_midi_common_gallery", result["midi_only_and_random_baselines"]["random"])
            inventory = root / "inventory.jsonl"
            report = root / "report.md"
            build_control_inventory(REPO / "research/modelling/control_scope/stage_a_controls.jsonl", inventory)
            generate_report(report, inventory, dataset_dir=root)
            report_text = report.read_text(encoding="utf-8")
            self.assertIn("Render and feature coverage", report_text)
            self.assertIn("Known source-lineage near-neighbor win rate", report_text)
            self.assertIn("Acoustic hard-negative win", report_text)
            self.assertIn("not applicable: query MIDI source states are outside the validation preset gallery", report_text)
            self.assertIn("| Random chance |", report_text)
            self.assertEqual(result["available_arms"], extended["available_arms"])
            self.assertEqual(result["model_results"], extended["model_results"])
            self.assertEqual(previous_row_count,
                             len((root / "episode_metrics.private.jsonl").read_text(encoding="utf-8").splitlines()))

    def test_local_config_preflight_requires_cache_assets_without_network_fallback(self):
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            with self.assertRaisesRegex(ValueError, "local architecture/config assets"):
                _resolve_local_configs(Path(directory), "muq_mulan", {})

    def test_evaluation_waits_for_terminal_receipts_before_freezing_coverage(self):
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            plan = [{"request_id": "pending", "source_kind": "newly_rendered_original_preset"}]
            with self.assertRaisesRegex(ValueError, "1 planned requests have no terminal receipt"):
                evaluation._render_receipts(Path(directory), plan)

    def test_cli_defaults_to_corpus_qualified_curve_and_readback_renderer(self):
        from frozen_retrieval.cli import _defaults

        config = _defaults()["renderer"].read_text(encoding="utf-8")
        renderer_id = "vital-1.6.4-vst3-sha256-a622a2c99b4066cd7945a4ab9bbdd698e7632a30702f6f0a7ccbf26a56b576e1-dawdreamer-0.8.3-full-1.5.5-state-verified-seed-bits-v2-expression-midi-v1-curve-input-v1-corpus-readback-v1"
        self.assertIn(f"renderer_id: {renderer_id}", config)
        self.assertIn("max_workers: 1", config)

    def test_muq_loader_uses_only_pinned_local_audio_and_text_configs(self):
        calls = []

        class MuQConfig:
            def __init__(self, **values):
                self.values = values

        class MuQ:
            def __init__(self, config):
                self.config = config

        class AudioPretrained:
            def __init__(self, model_name):
                self.model = None
                self._init_pretrained_model(model_name)

            def _init_pretrained_model(self, _model_name):
                raise AssertionError("the network loader must be replaced")

        class TextPretrained:
            def __init__(self):
                self.model = self._init_pretrained_model()

            def _init_pretrained_model(self):
                raise AssertionError("the network loader must be replaced")

        class MuQMuLan:
            def __init__(self, config):
                self.audio = AudioPretrained(config["audio_model"]["name"])
                self.text = TextPretrained()
                self.sr = 24_000

            def load_state_dict(self, state, strict):
                calls.append(("state", state, strict))

            def to(self, device):
                calls.append(("device", device))
                return self

        class LocalConfig:
            @staticmethod
            def from_pretrained(path, local_files_only):
                calls.append(("text_config", path, local_files_only))
                return path

        class TextModel:
            def __init__(self, config):
                self.config = config

        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            audio_path, text_path = root / "audio", root / "text"
            audio_path.mkdir()
            text_path.mkdir()
            (audio_path / "config.json").write_text('{"width": 3}', encoding="utf-8")
            (text_path / "config.json").write_text("{}", encoding="utf-8")
            checkpoint = root / "mulan"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text(json.dumps({
                "audio_model": {"name": "OpenMuQ/MuQ-large-msd-iter"},
                "text_model": {"name": "xlm-roberta-base"}}), encoding="utf-8")

            def make_module(name, **members):
                module = types.ModuleType(name)
                for key, value in members.items():
                    setattr(module, key, value)
                return module

            modules = {
                "muq": make_module("muq"), "muq.muq": make_module("muq.muq"),
                "muq.muq.muq": make_module("muq.muq.muq", MuQ=MuQ, MuQConfig=MuQConfig),
                "muq.muq_mulan": make_module("muq.muq_mulan"),
                "muq.muq_mulan.models": make_module("muq.muq_mulan.models"),
                "muq.muq_mulan.models.audio": make_module("muq.muq_mulan.models.audio", AudioSpectrogramTransformerPretrained=AudioPretrained),
                "muq.muq_mulan.models.text": make_module("muq.muq_mulan.models.text", TextTransformerPretrained=TextPretrained),
                "muq.muq_mulan.muq_mulan": make_module("muq.muq_mulan.muq_mulan", MuQMuLan=MuQMuLan),
                "transformers": make_module("transformers", XLMRobertaConfig=LocalConfig, XLMRobertaModel=TextModel),
            }

            class FakeTorch:
                @staticmethod
                def load(path, map_location, weights_only):
                    calls.append(("checkpoint", path, map_location, weights_only))
                    return {"weights": 1}

            with patch.dict(sys.modules, modules):
                model = _load_muq_mulan_offline(checkpoint, {"audio": audio_path, "text": text_path}, FakeTorch(), "cpu")

            self.assertEqual(24_000, model.sr)
            self.assertIsInstance(model.audio.model, MuQ)
            self.assertEqual({"width": 3}, model.audio.model.config.values)
            self.assertIsInstance(model.text.model, TextModel)
            self.assertTrue(any(row[0] == "text_config" and row[2] is True for row in calls))
            self.assertTrue(any(row[0] == "checkpoint" and row[-1] is True for row in calls))
            self.assertEqual(("device", "cpu"), calls[-1])

    def test_muq_offline_initializer_attaches_model_on_upstream_audio_wrapper(self):
        try:
            from muq.muq_mulan.models.audio import AudioSpectrogramTransformerPretrained
        except ImportError as error:
            self.skipTest(f"official MuQ source package unavailable: {error}")

        class Config:
            def __init__(self, **values):
                self.values = values

        class AudioModel:
            def __init__(self, config):
                self.config = config

            def parameters(self):
                return iter(())

        original_initializer = AudioSpectrogramTransformerPretrained._init_pretrained_model
        AudioSpectrogramTransformerPretrained._init_pretrained_model = _make_muq_audio_initializer(
            AudioModel, Config, {"width": 3})
        try:
            audio = AudioSpectrogramTransformerPretrained("OpenMuQ/MuQ-large-msd-iter")
        finally:
            AudioSpectrogramTransformerPretrained._init_pretrained_model = original_initializer

        self.assertIsInstance(audio.model, AudioModel)
        self.assertEqual({"width": 3}, audio.model.config.values)

    def test_cir_loader_builds_from_local_ast_assets_and_safe_checkpoint_weights(self):
        calls = []

        class Config:
            @staticmethod
            def from_pretrained(path, local_files_only):
                calls.append(("config", path, local_files_only))
                return object()

        class Extractor:
            sampling_rate = 16_000

            @staticmethod
            def from_pretrained(path, local_files_only):
                calls.append(("extractor", path, local_files_only))
                return Extractor()

        class Model:
            def __init__(self, _config):
                self.classifier = object()

            def load_state_dict(self, weights, strict):
                calls.append(("weights", sorted(weights), strict))

        class FakeTorch:
            nn = types.SimpleNamespace(Identity=lambda: "identity")

            @staticmethod
            def load(path, map_location, weights_only):
                calls.append(("checkpoint", path, map_location, weights_only))
                return {"custom_config": {"model": {"pretrained": "facebook/ast"}},
                        "model_state_dict": {"backbone.weight": 1, "classifier.weight": 2}}

        ast_module = types.ModuleType("AST")
        ast_module.ASTFineTuned = Model
        transformers = types.ModuleType("transformers")
        transformers.ASTConfig, transformers.ASTFeatureExtractor = Config, Extractor
        entry = {"local_configs": {"ast": {"model_name": "facebook/ast"}}}
        with patch.dict(sys.modules, {"AST": ast_module, "transformers": transformers}):
            model = _load_cir_ast_offline(Path("checkpoint.pt"), Path("local-ast"), entry, FakeTorch())

        self.assertEqual("identity", model.classifier)
        self.assertTrue(all(row[2] is True for row in calls if row[0] in {"config", "extractor"}))
        self.assertTrue(any(row[0] == "checkpoint" and row[-1] is True for row in calls))
        self.assertEqual(("weights", ["backbone.weight"], True), calls[-1])

    def test_integer_stereo_pcm_is_scaled_before_channel_mean(self):
        class FakeArray:
            def __init__(self, values, dtype="int16"):
                self.values, self.dtype = values, dtype
                self.ndim = 2 if values and isinstance(values[0], list) else 1

            def astype(self, dtype, copy=False):
                return FakeArray(self.values, "float32")

            def __truediv__(self, divisor):
                return FakeArray([[value / divisor for value in row] for row in self.values], "float32")

            def mean(self, axis):
                return FakeArray([sum(row) / len(row) for row in self.values], "float32")

        class FakeNumpy:
            integer = object()
            float32 = object()

            @staticmethod
            def asarray(values):
                return values if isinstance(values, FakeArray) else FakeArray(values)

            @staticmethod
            def issubdtype(dtype, kind):
                return dtype == "int16"

            @staticmethod
            def iinfo(dtype):
                return type("IntegerInfo", (), {"min": -32768, "max": 32767})()

        decoded = _decode_audio(FakeArray([[32767, 32767], [-32768, -32768]]), FakeNumpy)
        self.assertAlmostEqual(32767 / 32768, decoded.values[0])
        self.assertEqual(-1.0, decoded.values[1])

    def test_temporal_pool_masks_padding_and_emits_mean_plus_eight_bin_pyramid(self):
        frames = [[float(index)] for index in range(8)] + [[1000.0], [2000.0]]
        pooled = core.masked_temporal_pools(frames, valid_frames=8)
        self.assertEqual([3.5], pooled["masked_mean"])
        self.assertEqual(16, len(pooled["temporal_pyramid_8"]))
        self.assertEqual([0.0, 0.0], pooled["temporal_pyramid_8"][:2])
        with self.assertRaises(ValueError):
            core.masked_temporal_pools(frames, valid_frames=7)

    def test_beats_frequency_reduction_preserves_time_patch_order(self):
        class FakeFrames:
            def __init__(self, rows):
                self.rows = rows
                self.shape = (len(rows), 768)

        class FakeTemporal:
            def __init__(self, rows):
                self.rows = rows

            def __getitem__(self, index):
                batch_index, frame_slice = index
                return FakeFrames(self.rows[batch_index][frame_slice])

        class FakeGrid:
            def __init__(self, rows, time_patches, frequency_patches):
                self.rows, self.time_patches, self.frequency_patches = rows, time_patches, frequency_patches

            def mean(self, dim):
                if dim != 2:
                    raise AssertionError("frequency axis must be reduced")
                values = []
                for time_index in range(self.time_patches):
                    patches = self.rows[time_index * self.frequency_patches:(time_index + 1) * self.frequency_patches]
                    values.append([sum(row[feature] for row in patches) / self.frequency_patches for feature in range(768)])
                return FakeTemporal([values])

        class PatternTensor:
            def __init__(self):
                self.rows = [[time_index * 10.0 + frequency_index] * 768
                             for time_index in range(4) for frequency_index in range(2)]
                self.shape = (1, len(self.rows), 768)
                self.ndim = 3

            def reshape(self, batch, time_patches, frequency_patches, dimensions):
                self.asserted_shape = (batch, time_patches, frequency_patches, dimensions)
                return FakeGrid(self.rows, time_patches, frequency_patches)

        output = _beats_sequence(PatternTensor(), patch_size=64, valid_fbank_frames=192)
        self.assertEqual((3, 768), output.shape)
        self.assertEqual([0.5, 10.5, 20.5], [row[0] for row in output.rows])

    def test_freeze_marks_loader_wrapped_backbone_eval_and_nontrainable(self):
        class Parameter:
            requires_grad = True

            def requires_grad_(self, value):
                self.requires_grad = value

        class Backbone:
            training = True

            def __init__(self):
                self.parameter = Parameter()

            def eval(self):
                self.training = False

            def parameters(self):
                return iter((self.parameter,))

        class Loader:
            def __init__(self):
                self.model = Backbone()

        loader = Loader()
        _freeze_model(loader)
        self.assertFalse(loader.model.training)
        self.assertFalse(loader.model.parameter.requires_grad)

    def test_bounded_midi_excerpt_keeps_complete_notes_and_reports_source_features(self):
        if importlib.util.find_spec("yaml") is None:
            sys.modules.setdefault("yaml", types.ModuleType("yaml"))
        from obruxo_data.midi import Performance

        capabilities = type("Capabilities", (), {"notes": True, "polyphony": True, "tempo_changes": False,
                                                  "pitch_bend": False, "channel_pressure": False,
                                                  "control_changes": frozenset(), "max_channels": 16,
                                                  "note_off_velocity": False})()
        performance = Performance(ticks_per_beat=480, bpm=120)
        for index in range(8):
            performance.add_note(pitch=48 + index, velocity=80, start_tick=index * 480, duration_ticks=240)
        excerpt, metadata = bounded_excerpt(performance, capabilities)
        self.assertEqual(8, len(excerpt.note_spans()))
        self.assertGreaterEqual(metadata["excerpt_duration_seconds"], 2.0)
        self.assertLessEqual(metadata["excerpt_duration_seconds"], 8.0)
        self.assertEqual(8, midi_features(excerpt)["note_count"])

    def test_render_qa_rejects_silence_but_records_clipping_and_tail_warnings(self):
        good = {"qa": {"finite": True, "actual_frames": 10, "expected_frames": 10, "actual_channels": 2,
                        "expected_channels": 2, "rms": 0.1, "silence_threshold": 1e-6, "clipping_count": 0,
                        "tail_rms": 0.0, "tail_rms_warning": 0.01}}
        self.assertEqual((True, "accepted"), _accepted(good))
        silence = json.loads(json.dumps(good))
        silence["qa"]["rms"] = 0.0
        self.assertEqual((False, "render_qa_silence"), _accepted(silence))
        clipping = json.loads(json.dumps(good))
        clipping["qa"]["clipping_count"] = 1
        self.assertEqual((True, "accepted_with_amplitude_above_unit_range"), _accepted(clipping))
        self.assertEqual(["amplitude_above_unit_range"], _qa_warnings(clipping))
        tail = json.loads(json.dumps(good))
        tail["qa"]["tail_rms"] = 0.1
        self.assertEqual((True, "accepted_with_tail_rms_above_warning"), _accepted(tail))
        self.assertEqual(["tail_rms_above_warning"], _qa_warnings(tail))

    def test_render_resume_skips_only_hash_and_provenance_matched_receipts(self):
        metadata = {"provenance": {"request_id": "request-1", "renderer_id": "vital-test"}}
        receipt = {"request_id": "request-1", "plan_identity": "plan-1", "render_policy_id": RENDER_POLICY_ID,
                   "qa_policy_id": QA_POLICY_ID, "renderer_id": "vital-test", "audio_sha256": "audio-1"}
        matches = {"request_id": "request-1", "plan_identity": "plan-1", "renderer_id": "vital-test",
                   "metadata": metadata, "audio_sha256": "audio-1"}
        self.assertTrue(_receipt_matches_cached_audio(receipt, **matches))
        self.assertFalse(_receipt_matches_cached_audio(receipt, **{**matches, "audio_sha256": "changed"}))
        self.assertFalse(_receipt_matches_cached_audio(receipt, **{**matches, "plan_identity": "changed"}))
        stale_metadata = {"provenance": {"request_id": "request-1", "renderer_id": "changed"}}
        self.assertFalse(_receipt_matches_cached_audio(receipt, **{**matches, "metadata": stale_metadata}))
        self.assertFalse(_receipt_matches_cached_audio({**receipt, "qa_policy_id": "stale"}, **matches))

    def test_frozen_manifest_digest_rejects_drift(self):
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            manifest = root / "frozen.jsonl"
            manifest.write_text('{"state":"original"}\n', encoding="utf-8")
            summary = {"manifest_sha256": {manifest.name: _file_sha256(manifest)}}
            validate_dataset_manifests(root, summary)
            manifest.write_text('{"state":"changed"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest drift"):
                validate_dataset_manifests(root, summary)


class ReportingAndControlTests(unittest.TestCase):
    def test_execution_summary_reports_benchmark_not_reporter_runtime(self):
        lines = _execution_status_lines(
            {"source_split": "validation", "performance_source_split": "train",
             "presets_verified": 131, "unique_midi_excerpts": 50},
            {"coverage_by_condition": {"cross_midi_common_gallery": {
                "common_feature_qualified_episodes_across_arms": 6450}},
             "gallery_candidate_states_render_qualified": 129,
             "model_results": {"clap_music:official_clip": {"cross_midi_common_gallery": {"episodes": 6450}}}},
            Path("dataset"), {"clap_music:official_clip": {"checkpoint_sha256": "abc123"}},
            {"clap_music": {"license_status": "approved", "license": "CC0-1.0"}})
        text = "\n".join(lines)
        self.assertIn("50 performances; reserved test sources were not used", text)
        self.assertIn("6,450 paired queries", text)
        self.assertIn("512-D", text)
        self.assertNotIn("Runtime preflight", text)
        self.assertNotIn("Extraction pass time", text)

    def test_envelope_attack_controls_are_not_grouped_as_global_settings(self):
        self.assertEqual("Envelopes", _control_group({"data_path": "settings.env_1_attack",
            "control_key": "env_1_attack", "family": "env.attack"}))

    def test_controlled_cli_accepts_all_run_encoder_families(self):
        args = controlled._parser().parse_args([
            "--dataset", "dataset", "--renderer-config", "renderer.yaml", "--checkpoint-root", "checkpoints",
            "--lock", "checkpoint-lock.json", "--output", "output", "--models",
            "clap_music", "muq_mulan", "dac_44khz", "encodec_24khz", "beats_iter3_plus", "cir_ast",
        ])
        self.assertEqual(["clap_music", "muq_mulan", "dac_44khz", "encodec_24khz", "beats_iter3_plus", "cir_ast"], args.models)

    def test_performance_profile_reports_selected_split_and_midi_diversity_only(self):
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            dataset = Path(directory)
            rows = [
                {"midi_id": "private-a", "performance_source_split": "train", "stats": {
                    "max_polyphony": 1, "articulation": "held", "pitch_median": 40,
                    "velocity_min": 30, "velocity_max": 90, "pitch_bend_events": 1,
                    "sustain_events": 0, "expression_events": 0, "channel_pressure_events": 0,
                    "control_change_counts": {}}},
                {"midi_id": "private-b", "performance_source_split": "validation", "stats": {
                    "max_polyphony": 4, "articulation": "phrased", "pitch_median": 80,
                    "velocity_min": 60, "velocity_max": 60, "pitch_bend_events": 0,
                    "sustain_events": 1, "expression_events": 0, "channel_pressure_events": 0,
                    "control_change_counts": {"64": 1}}},
            ]
            (dataset / "performances.private.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            profile = _performance_profile(dataset, "train")
            self.assertEqual(1, profile["performances"])
            self.assertEqual(1, profile["mono"])
            self.assertEqual(40, profile["pitch_median_min"])
            self.assertEqual(1, profile["variable_velocity"])
            self.assertEqual(1, profile["pitch_bend"])
            self.assertNotIn("private-a", str(profile))

    def test_inventory_reconciles_every_vital_control_as_fixed_zero_output(self):
        source = REPO / "research/modelling/control_scope/stage_a_controls.jsonl"
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            destination = Path(directory) / "inventory.jsonl"
            summary = build_control_inventory(source, destination)
            count, groups = _inventory_summary(destination)
            self.assertEqual(979, count)
            self.assertEqual(summary["total_controls"], sum(groups.values()))
            self.assertEqual(0, summary["enabled_controls"])
            rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(row["output_count"] == 0 and row["changeable"] is False for row in rows))

    def test_controlled_inventory_repeats_exact_fields_and_marks_only_the_perturbed_field(self):
        source = REPO / "research/modelling/control_scope/stage_a_controls.jsonl"
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            baseline = root / "main_inventory.jsonl"
            controlled_inventory = root / "controlled_inventory.jsonl"
            build_control_inventory(source, baseline)
            result = build_controlled_inventory(baseline, controlled_inventory, {
                "filter_cutoff_shift": {"field": "filter_1_cutoff", "before_raw": 20.0, "after_raw": 40.0,
                                         "before_normalized": 0.1, "after_normalized": 0.3}})
            rows = [json.loads(line) for line in controlled_inventory.read_text(encoding="utf-8").splitlines()]
            changed = [row for row in rows if row["experiment_version"] == "filter_cutoff_shift" and row["changeable"]]
            self.assertEqual(979 * 2, len(rows))
            self.assertEqual(1, len(changed))
            self.assertEqual("filter_1_cutoff", changed[0]["vital_key"])
            self.assertEqual(0, changed[0]["output_count"])
            self.assertNotIn("preset_state_id", changed[0])
            self.assertNotIn("preset_source_sha256", changed[0])
            self.assertTrue(all("data_sha256" not in row for row in rows))
            self.assertTrue(all("hash recorded" not in str(row.get("fixed_value_source")) for row in rows))
            self.assertEqual(1, result["versions"]["filter_cutoff_shift"]["enabled_controls"])
            self.assertEqual(1, result["versions"]["filter_cutoff_shift"]["controls_by_family"]["Filters"]["enabled"])

    def test_retrieval_report_explains_each_negative_with_same_example(self):
        text = "\n".join(_retrieval_explainer_lines(129))
        self.assertIn("A1", text)
        self.assertIn("A2", text)
        self.assertIn("B1", text)
        self.assertIn("every one of the 128 other presets", text)
        self.assertIn("five", text)
        self.assertIn("64 frequency bands", text)
        self.assertIn("No eligible related-preset pairs", text)
        self.assertIn("Do not compare the two hard-negative percentages directly", text)
        self.assertIn("Common versus mixed comparison MIDI", text)
        self.assertIn("6,450 queries reuse the same 129 presets", text)

    def test_report_lists_all_arms_without_inventing_scores(self):
        source = REPO / "research/modelling/control_scope/stage_a_controls.jsonl"
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            inventory = root / "inventory.jsonl"
            report = root / "report.md"
            build_control_inventory(source, inventory)
            generate_report(report, inventory)
            text = report.read_text(encoding="utf-8")
            self.assertIn("No prepared real dataset was supplied", text)
            self.assertIn("Empirical execution remains unfinished", text)
            self.assertIn("Controlled patch-perturbation diagnostic: not run", text)
            self.assertIn("Status: **unavailable / not run**", text)
            self.assertIn("Recall@1 | not run", text)
            self.assertIn("0 outputs; all source controls stay fixed", text)
            self.assertIn("inventory.jsonl", text)
            self.assertNotIn("| Vital control family | Enabled / total |", text)
            self.assertNotIn("Vital controls for this representation version", text)

    def test_report_presents_issue155_results_and_keeps_candidate_gate_explicit(self):
        source = REPO / "research/modelling/control_scope/stage_a_controls.jsonl"
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            inventory, report = root / "inventory.jsonl", root / "report.md"
            validation, shortlist = root / "issue155.json", root / "shortlist.json"
            assignment_metrics = root / "gallery_assignment_episode_metrics.private.jsonl"
            build_control_inventory(source, inventory)
            assignment_rows = []
            for state_id, family_id in (("state-a", "family-a"), ("state-b", "family-b")):
                for midi_id, query_quality in (("midi-a", "clean"), ("midi-b", "warning")):
                    for offset, target_quality, warning_count, recall in (
                            (1, "warning", 1, 1.0), (7, "clean", 2, 0.0), (19, "warning", 3, 1.0)):
                        assignment_rows.append({"arm": "clap_music:official_clip", "episode_id": f"{state_id}-{midi_id}-{offset}",
                            "gallery_assignment_offset": offset, "target_state_id": state_id,
                            "target_family_id": family_id, "query_midi_id": midi_id,
                            "query_render_quality": query_quality, "correct_match_gallery_render_quality": target_quality,
                            "gallery_candidate_count": 129, "gallery_warning_candidate_count": warning_count,
                            "gallery_warning_candidate_fraction": warning_count / 129,
                            "gallery_unknown_candidate_count": 0, "rank_expected": 1.0 if recall else 9.0,
                            "recall_at_1": recall, "recall_at_5": recall, "mrr": recall})
            assignment_metrics.write_text("".join(json.dumps(row) + "\n" for row in assignment_rows), encoding="utf-8")
            validation.write_text(json.dumps({
                "input_dataset": {"accepted_cached_clips": 6480},
                "gallery_assignment_experiment": {"offsets": [1, 7, 19],
                    "per_assignment": {}, "aggregate": {
                        "clap_music:official_clip": {"metrics": {"recall_at_1": {"mean": 0.6}}},
                        "beats_iter3_plus:temporal_pyramid_8": {"metrics": {"recall_at_1": {"mean": 0.5}}}},
                    "best_pooling_arm_by_family": {"beats_iter3_plus": "beats_iter3_plus:temporal_pyramid_8"},
                    "gallery_midi_count_by_offset": {"1": 50},
                    "gallery_candidate_count": {"min": 129, "max": 129},
                    "paired_differences": {}, "per_assignment_winner": {}},
                "clean_query_only": {"arms": {"clap_music:official_clip": {"queries": 12, "metrics": {
                    "recall_at_1": {"mean": 0.6, "crossed_95_ci": {"lower_95": 0.5, "upper_95": 0.7}},
                    "recall_at_5": {"mean": 0.8}, "mrr": {"mean": 0.65}}}}},
                "strict_clean_gallery": {},
                "gain_sensitivity": {"sample": {"size": 50}, "arms": {"clap_music:official_clip": {
                    "adjusted_clip_embedding_cosine_distance": {"mean": 0.02, "crossed_95_ci": {"lower_95": 0.01, "upper_95": 0.03}},
                    "roles": {role: {"samples": 1, "changed_predictions": 1, "corrected_predictions": 1,
                        "newly_wrong_predictions": 0, "before": {"metrics": {"recall_at_1": {"mean": 0.5}}},
                        "after": {"metrics": {"recall_at_1": {"mean": 1.0}}},
                        "paired_after_minus_before": {"target_margin": {"difference": 0.05,
                            "lower_95": 0.01, "upper_95": 0.09}, "recall_at_1": {"difference": 0.5,
                            "lower_95": 0.2, "upper_95": 0.8}}} for role in ("query", "correct_gallery_candidate")}}}},
                "gain_warning_sample_selection": {},
            }), encoding="utf-8")
            shortlist.write_text(json.dumps({"selected_count": 8,
                "screen_strength_counts": {"strong": 4, "borderline": 3, "weak or inconclusive": 1}}), encoding="utf-8")
            generate_report(report, inventory, issue155_results_path=validation,
                            candidate_shortlist_path=shortlist)
            text = report.read_text(encoding="utf-8")
            self.assertIn("Issue #155 diagnostic results", text)
            self.assertIn("Plain-English readout", text)
            self.assertIn("all 50 existing excerpts", text)
            self.assertIn("Keep `A1` as the input", text)
            self.assertIn("reduce the volume of `A1` alone", text)
            self.assertIn("one fewer wrong preset to beat", text)
            self.assertIn("60.0% [50.0%, 70.0%]", text)
            self.assertIn("target margin ×100 (cosine points", text)
            self.assertNotIn("+5.00 pp", text)
            self.assertIn("Render-quality results for the expanded comparison-MIDI assignments", text)
            self.assertIn("How retrieval fails", text)
            self.assertIn("Median expected rank", text)
            self.assertIn("Per-preset R@1", text)
            self.assertIn("12 attempts per encoder", text)
            self.assertIn("keeping a query's three assignment attempts together", text)
            self.assertIn("Mean warned candidates / fraction of 129", text)
            self.assertIn("groups overlap for 4 pairs, and their union contains 4 pairs", text)
            self.assertIn("selected 8 presets", text)
            self.assertIn("different random seed, `A1*`", text)
            self.assertIn("4 strong, 3 borderline, 1 weak or inconclusive", text)
            self.assertIn("Each assignment round used 50 gallery MIDI excerpts", text)
            self.assertNotIn("All {'1': 50}", text)
            self.assertIn("definitive multi-encoder controlled diagnostic has not been run", text)
            self.assertLess(text.index("Issue #155 diagnostic results"), text.index("## Execution status"))
            self.assertNotIn("| Vital control family | Enabled / total |", text)
            self.assertLess(text.index("## Results and recommendation"), text.index("## Execution status"))

    def test_controlled_report_renders_each_version_family_inventory(self):
        source = REPO / "research/modelling/control_scope/stage_a_controls.jsonl"
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            root = Path(directory)
            inventory = root / "inventory.jsonl"
            report = root / "report.md"
            controlled = root / "controlled.json"
            build_control_inventory(source, inventory)
            controlled.write_text(json.dumps({
                "variants": ["original", "filter_cutoff_shift"],
                "controlled_fields": {"filter_cutoff_shift": {"field": "filter_1_cutoff",
                    "before_raw": 20.0, "after_raw": 40.0, "before_normalized": 0.1,
                    "after_normalized": 0.3}},
                "models": {}, "midi_strata": [
                    {"max_polyphony": 1, "pitch_median": 30, "duration_seconds": 3.0,
                     "note_duration_median_seconds": 1.0, "velocity_min": 50, "velocity_max": 50,
                     "articulation": "held"},
                    {"max_polyphony": 3, "pitch_median": 60, "duration_seconds": 5.0,
                     "note_duration_median_seconds": 2.0, "velocity_min": 40, "velocity_max": 90,
                     "articulation": "phrased"},
                ],
            }), encoding="utf-8")
            generate_report(report, inventory, controlled_summary_path=controlled)
            text = report.read_text(encoding="utf-8")
            self.assertIn("### Original source preset", text)
            self.assertIn("### Filter cutoff perturbation", text)
            self.assertIn("| Filters ✅ 1 /", text)
            self.assertIn("| Envelopes ❌ 0 /", text)
            self.assertIn("2 MIDI IDs; polyphony 1–3, median pitch 30–60", text)
            self.assertIn("MIDI-cluster intervals summarize variation for this one preset", text)
            self.assertIn("one Vital control while keeping MIDI 1 fixed", text)
            self.assertLess(text.index("## Controlled preset perturbations"), text.index("## Execution status"))


if __name__ == "__main__":
    unittest.main()
