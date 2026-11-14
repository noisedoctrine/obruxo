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
from frozen_retrieval.dataset import (_file_sha256, _paired_source_midi, bounded_excerpt, midi_features,
                                     validate_dataset_manifests)
from frozen_retrieval.encoders import _beats_sequence, _decode_audio
from frozen_retrieval import evaluation
from frozen_retrieval.encoders import _load_cir_ast_offline, _load_muq_mulan_offline, _resolve_local_configs
from frozen_retrieval.rendering import RENDER_POLICY_ID, _accepted, render_plan_identity
from frozen_retrieval.reporting import _inventory_summary, build_control_inventory, generate_report


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


class FeatureAndManifestTests(unittest.TestCase):
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
            self.assertIsNone(paired)
            self.assertEqual("pair.ambiguous", reason)

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
            summary = {"source_split": "validation", "manifest_sha256": {
                name: _file_sha256(root / name) for name in manifest_names}}
            (root / "dataset_summary.private.json").write_text(json.dumps(summary), encoding="utf-8")

            receipts = []
            for row in rows:
                accepted = row["request_id"] != "far-same-midi"
                receipts.append({"request_id": row["request_id"], "plan_identity": render_plan_identity(row),
                                 "render_policy_id": RENDER_POLICY_ID, "accepted": accepted,
                                 "qa_status": "accepted" if accepted else "render_qa_silence",
                                 "audio_sha256": "a" * 64})
            write_jsonl(root / "render_receipts.private.jsonl", receipts)
            cache_dir = root / "features" / "clap_music"
            cache_dir.mkdir(parents=True)
            feature_requests = ("target-query", "target-gallery", "near-gallery", "far-gallery", "near-same-midi")
            write_jsonl(cache_dir / "feature_manifest.private.jsonl", [
                {"request_id": request_id, "feature_id": f"feature-{request_id}"} for request_id in feature_requests])
            (cache_dir / "extraction.latest.json").write_text(json.dumps({"checkpoint_sha256": "b" * 64}), encoding="utf-8")
            features = {"target-query": {"official_clip": [1.0, 0.0]},
                        "target-gallery": {"official_clip": [1.0, 0.0]},
                        "near-gallery": {"official_clip": [0.0, 1.0]},
                        "far-gallery": {"official_clip": [-1.0, 0.0]},
                        "near-same-midi": {"official_clip": [0.0, 1.0]}}
            fake_numpy = types.ModuleType("numpy")
            fake_numpy.stack, fake_numpy.float32 = _FakeNumpy.stack, _FakeNumpy.float32
            with patch.object(evaluation, "list_clip_cache", return_value=features), \
                    patch.dict(sys.modules, {"numpy": fake_numpy}), \
                    patch.object(evaluation, "_cosine_matrix", side_effect=_fake_cosine_matrix), \
                    patch.object(evaluation, "midi_shortcut_baseline", return_value={}):
                result = evaluation.evaluate_dataset(root)

            coverage = result["coverage_by_condition"]["cross_midi_common_gallery"]
            self.assertEqual(1, coverage["planned_episodes"])
            self.assertEqual(1, coverage["qa_qualified_episodes"])
            self.assertEqual(1, coverage["common_feature_qualified_episodes_across_arms"])
            scores = result["model_results"]["clap_music:official_clip"]["cross_midi_common_gallery"]
            self.assertEqual(1.0, scores["recall_at_1"])
            self.assertEqual(1.0, scores["same_midi_negative_win_rate"])
            self.assertEqual(1.0, scores["known_near_neighbor_negative_win_rate"])
            self.assertEqual(1, scores["same_midi_negative_win_candidates"])
            self.assertEqual(1, scores["known_near_neighbor_negative_win_candidates"])
            inventory = root / "inventory.jsonl"
            report = root / "report.md"
            build_control_inventory(REPO / "research/modelling/control_scope/stage_a_controls.jsonl", inventory)
            generate_report(report, inventory, dataset_dir=root)
            report_text = report.read_text(encoding="utf-8")
            self.assertIn("Render and feature coverage", report_text)
            self.assertIn("Known source-lineage near-neighbor win rate", report_text)

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
            def _init_pretrained_model(self, _model_name):
                raise AssertionError("the network loader must be replaced")

        class TextPretrained:
            def _init_pretrained_model(self):
                raise AssertionError("the network loader must be replaced")

        class MuQMuLan:
            def __init__(self, config):
                self.audio = AudioPretrained()._init_pretrained_model(config["audio_model"]["name"])
                self.text = TextPretrained()._init_pretrained_model()
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
            self.assertTrue(any(row[0] == "text_config" and row[2] is True for row in calls))
            self.assertTrue(any(row[0] == "checkpoint" and row[-1] is True for row in calls))
            self.assertEqual(("device", "cpu"), calls[-1])

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
            self.assertIn("Empirical execution remains unfinished", text)
            self.assertIn("Controlled patch-perturbation diagnostic: not run", text)
            self.assertIn("Status: **unavailable / not run**", text)
            self.assertIn("Recall@1 | not run", text)
            self.assertIn("control family", text)
            self.assertEqual(9, text.count("| Global settings ❌ 0 / 610 |"))


if __name__ == "__main__":
    unittest.main()
