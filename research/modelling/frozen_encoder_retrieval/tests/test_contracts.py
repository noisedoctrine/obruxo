from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import importlib.util
import types
import unittest


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[2]
for directory in (PACKAGE, REPO / "research/data_generation", REPO / "research/training"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from frozen_retrieval import core
from frozen_retrieval.dataset import _file_sha256, bounded_excerpt, midi_features, validate_dataset_manifests
from frozen_retrieval.rendering import _accepted
from frozen_retrieval.encoders import _decode_audio
from frozen_retrieval.reporting import _inventory_summary, build_control_inventory, generate_report


def _presets():
    return [{"state_id": f"state-{index}", "family_id": f"family-{index // 2}", "split": "validation"}
            for index in range(3)]


def _performances():
    return [{"midi_id": f"midi-{index}", "source_state_id": f"state-{index}", "split": "validation",
             "stats": {"max_polyphony": 1 + index % 2, "articulation": "held", "duration_seconds": 3 + index,
                       "pitch_median": 48 + index * 12, "notes_per_second": 2 + index}}
            for index in range(3)]


class EpisodeContractTests(unittest.TestCase):
    def test_every_state_queries_every_unique_midi_and_gallery_is_cross_performance(self):
        candidate_sets, episodes = core.build_episodes(_presets(), _performances(), gallery_cap=3)
        self.assertEqual(18, len(episodes))
        self.assertEqual(1, len(candidate_sets))
        candidate = next(iter(candidate_sets.values()))
        self.assertEqual({row["state_id"] for row in _presets()}, set(candidate["state_ids"]))
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


class FeatureAndManifestTests(unittest.TestCase):
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

    def test_render_qa_rejects_silence_clipping_and_truncated_release(self):
        good = {"qa": {"finite": True, "actual_frames": 10, "expected_frames": 10, "actual_channels": 2,
                        "expected_channels": 2, "rms": 0.1, "silence_threshold": 1e-6, "clipping_count": 0,
                        "tail_rms": 0.0, "tail_rms_warning": 0.01}}
        self.assertEqual((True, "accepted"), _accepted(good))
        for field, value, expected in (("rms", 0.0, "render_qa_silence"), ("clipping_count", 1, "render_qa_clipping"),
                                       ("tail_rms", 0.1, "render_qa_tail_truncation")):
            metadata = json.loads(json.dumps(good))
            metadata["qa"][field] = value
            self.assertEqual((False, expected), _accepted(metadata))

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
            self.assertIn("Status: **unavailable / not run**", text)
            self.assertIn("Recall@1 | not run", text)
            self.assertIn("control family", text)
            self.assertEqual(9, text.count("| Global settings ❌ 0 / 610 |"))


if __name__ == "__main__":
    unittest.main()
