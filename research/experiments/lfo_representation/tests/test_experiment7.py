from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

import experiment8
import lfo_experiment.experiment7 as experiment7_module
from lfo_experiment.alignment5 import circular_shift_torch, exact_align_cpu
from lfo_experiment.experiment6 import summarize_results
from lfo_experiment.experiment7 import (
    _apply_training_stage,
    _apply_training_stage_torch,
    _align_stage_grouped,
    _circular_shift_xpu_torch,
    _encode_final_clip_beam_torch,
    _estimate_peak_memory_mb,
    _experiment8_size_pairs,
    _experiment9_phase_head_outputs,
    _load_trained_cache,
    _roll_bank_torch,
    _make_experiment8_screen_jobs,
    _make_experiment9_screen_jobs,
    _make_7a_jobs,
    _make_7b_jobs,
    _training_cache_key,
    _write_training_cache,
    encode_final_clip_beam,
    Experiment7Policy,
)
from lfo_experiment.phase4 import PhaseChain, circular_shift
from lfo_experiment.stacked import TOPOLOGY_NAMES


class Experiment7OptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_device = os.environ.get("LFO_ALIGN_DEVICE")
        os.environ["LFO_ALIGN_DEVICE"] = "cpu"

    def tearDown(self) -> None:
        if self.old_device is None:
            os.environ.pop("LFO_ALIGN_DEVICE", None)
        else:
            os.environ["LFO_ALIGN_DEVICE"] = self.old_device

    def test_roll_bank_shift_matches_gather_shift_with_tolerance(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(202608)
        codes = torch.randn((4, 32), generator=generator)
        phases = torch.rand((6, 4), generator=generator)
        expanded = codes[None, :, :].expand(6, -1, -1)
        bank = _roll_bank_torch(codes)[None, :, :, :].expand(6, -1, -1, -1)
        expected = circular_shift_torch(expanded, phases)
        actual = _circular_shift_xpu_torch(expanded, phases, roll_bank=bank)
        self.assertLess(float(torch.max(torch.abs(actual - expected))), 1e-4)

    def test_xpu_shift_feature_flag_can_select_gather_path(self) -> None:
        old_impl = os.environ.get("LFO_XPU_SHIFT_IMPL")
        try:
            os.environ["LFO_XPU_SHIFT_IMPL"] = "gather"
            values = torch.arange(16, dtype=torch.float32).reshape(2, 8)
            phases = torch.tensor([0.0, 0.25], dtype=torch.float32)
            expected = circular_shift_torch(values, phases)
            actual = _circular_shift_xpu_torch(values, phases, roll_bank=_roll_bank_torch(values))
            torch.testing.assert_close(actual, expected)
        finally:
            if old_impl is None:
                os.environ.pop("LFO_XPU_SHIFT_IMPL", None)
            else:
                os.environ["LFO_XPU_SHIFT_IMPL"] = old_impl

    def test_grouped_alignment_matches_per_condition_reference(self) -> None:
        phase = np.linspace(0, 1, 64, endpoint=False)
        stage = np.zeros((len(TOPOLOGY_NAMES), 2, 64), dtype=np.float32)
        for condition in range(len(TOPOLOGY_NAMES)):
            stage[condition, 0] = 0.0
            stage[condition, 1] = np.sin(2 * np.pi * phase * (condition + 1))
        conditions = np.asarray([0, 1, 2], dtype=np.int32)
        beam_width = 2
        residual = np.vstack(
            [
                circular_shift(stage[int(condition), 1], 0.1 + 0.05 * beam)
                for condition in conditions
                for beam in range(beam_width)
            ]
        ).astype(np.float32)
        grouped = _align_stage_grouped(residual, stage, conditions, beam_width)
        expected_error = np.empty_like(grouped.error)
        expected_phase = np.empty_like(grouped.phase)
        expected_gain = np.empty_like(grouped.gain)
        for row, condition in enumerate(np.repeat(conditions, beam_width)):
            aligned = exact_align_cpu(residual[row : row + 1], stage[int(condition)])
            expected_error[row] = aligned.error[0]
            expected_phase[row] = aligned.phase[0]
            expected_gain[row] = aligned.gain[0]
        np.testing.assert_allclose(grouped.error, expected_error, atol=1e-10)
        np.testing.assert_allclose(grouped.phase, expected_phase, atol=1e-10)
        np.testing.assert_allclose(grouped.gain, expected_gain, atol=1e-10)

    def test_encode_final_clip_beam_cpu_fallback_returns_compact_encoding(self) -> None:
        targets, chain, conditions = self._tiny_encoding_fixture()
        reconstructed, encoding, noop = encode_final_clip_beam(
            targets,
            chain,
            conditions,
            beam_width=2,
            batch_size=2,
        )
        self.assertEqual(reconstructed.shape, targets.shape)
        self.assertEqual(encoding.base_indices.shape, (2,))
        self.assertEqual(encoding.stage_indices[0].shape, (2,))
        self.assertIn("stage_1_noop_rate", noop)

    def test_experiment8_cli_default_aligns_xpu(self) -> None:
        args = experiment8.parser().parse_args(["run"])
        self.assertEqual(args.align_device, "xpu")

    def test_experiment8_launcher_default_aligns_xpu(self) -> None:
        launcher = Path(__file__).resolve().parents[1] / "start_experiment8_with_monitor.cmd"
        self.assertIn('set "ALIGN=xpu"', launcher.read_text(encoding="utf-8"))

    def test_encode_final_clip_beam_dispatches_to_torch_when_xpu_available(self) -> None:
        targets, chain, conditions = self._tiny_encoding_fixture()
        sentinel = (np.zeros_like(targets), mock.Mock(), {})
        with (
            mock.patch.object(experiment7_module, "_torch_align_device", return_value="xpu:0"),
            mock.patch.object(experiment7_module, "_encode_final_clip_beam_torch", return_value=sentinel) as encode,
        ):
            result = encode_final_clip_beam(targets, chain, conditions, beam_width=2, batch_size=2)
        self.assertIs(result, sentinel)
        self.assertEqual(encode.call_args.kwargs["device"], "xpu:0")

    def test_encode_final_clip_beam_cpu_fallback_bypasses_torch_encoder(self) -> None:
        targets, chain, conditions = self._tiny_encoding_fixture()
        with (
            mock.patch.object(experiment7_module, "_torch_align_device", return_value=None),
            mock.patch.object(
                experiment7_module,
                "_encode_final_clip_beam_torch",
                side_effect=AssertionError("torch path should not run"),
            ),
        ):
            reconstructed, encoding, _ = encode_final_clip_beam(targets, chain, conditions, beam_width=2, batch_size=2)
        self.assertEqual(reconstructed.shape, targets.shape)
        self.assertEqual(encoding.base_indices.shape, (2,))

    def test_torch_resident_encoder_matches_cpu_reference(self) -> None:
        targets, chain, conditions = self._tiny_encoding_fixture()
        cpu_reconstructed, cpu_encoding, _ = encode_final_clip_beam(targets, chain, conditions, beam_width=2, batch_size=2)
        torch_reconstructed, torch_encoding, _ = _encode_final_clip_beam_torch(
            targets,
            chain,
            conditions,
            beam_width=2,
            batch_size=2,
            device="cpu",
        )
        np.testing.assert_allclose(torch_reconstructed, cpu_reconstructed, atol=1e-4)
        self.assertEqual(torch_encoding.base_indices.shape, cpu_encoding.base_indices.shape)
        self.assertEqual(torch_encoding.stage_indices[0].shape, cpu_encoding.stage_indices[0].shape)

    def test_torch_resident_encoder_gather_flag_matches_cpu_reference_paths(self) -> None:
        targets, chain, conditions = self._tiny_encoding_fixture()
        _, cpu_encoding, _ = encode_final_clip_beam(targets, chain, conditions, beam_width=2, batch_size=2)
        old_impl = os.environ.get("LFO_XPU_SHIFT_IMPL")
        try:
            os.environ["LFO_XPU_SHIFT_IMPL"] = "gather"
            _, torch_encoding, _ = _encode_final_clip_beam_torch(
                targets,
                chain,
                conditions,
                beam_width=2,
                batch_size=2,
                device="cpu",
            )
        finally:
            if old_impl is None:
                os.environ.pop("LFO_XPU_SHIFT_IMPL", None)
            else:
                os.environ["LFO_XPU_SHIFT_IMPL"] = old_impl
        np.testing.assert_array_equal(torch_encoding.base_indices, cpu_encoding.base_indices)
        np.testing.assert_allclose(torch_encoding.base_phases, cpu_encoding.base_phases, atol=1e-5)
        np.testing.assert_array_equal(torch_encoding.stage_indices[0], cpu_encoding.stage_indices[0])
        np.testing.assert_allclose(torch_encoding.stage_phases[0], cpu_encoding.stage_phases[0], atol=1e-5)
        np.testing.assert_allclose(torch_encoding.stage_gains[0], cpu_encoding.stage_gains[0], atol=1e-5)

    def test_chunked_torch_training_stage_matches_cpu_reference(self) -> None:
        phase = np.linspace(0, 1, 64, endpoint=False)
        codes = np.stack(
            [
                np.zeros_like(phase),
                np.sin(2 * np.pi * phase),
                np.cos(2 * np.pi * phase),
            ]
        ).astype(np.float32)
        prefix = np.zeros((5, 64), dtype=np.float32)
        targets = np.stack(
            [
                circular_shift(codes[1], 0.10),
                0.5 * circular_shift(codes[2], 0.25),
                0.2 * circular_shift(codes[1], 0.40) + 0.1,
                np.zeros(64, dtype=np.float32),
                circular_shift(codes[2], 0.75),
            ]
        ).astype(np.float32)
        old_batch = os.environ.get("LFO_TRAIN_STAGE_BATCH_SIZE")
        try:
            os.environ["LFO_ALIGN_DEVICE"] = "cpu"
            cpu = _apply_training_stage(prefix, targets, codes)
            os.environ["LFO_TRAIN_STAGE_BATCH_SIZE"] = "2"
            torch_result = _apply_training_stage_torch(prefix, targets, codes, device="cpu")
        finally:
            if old_batch is None:
                os.environ.pop("LFO_TRAIN_STAGE_BATCH_SIZE", None)
            else:
                os.environ["LFO_TRAIN_STAGE_BATCH_SIZE"] = old_batch
        for actual, expected in zip(torch_result, cpu, strict=True):
            np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_7a_bulk_resolution_is_lower_than_final_7b_resolution(self) -> None:
        jobs_7a = _make_7a_jobs(quick=False, beam_width=4, seed=20260707)
        policy = Experiment7Policy("frequency_first", "none")
        jobs_7b = _make_7b_jobs(policy=policy, quick=False, beam_width=4, seed=7267)
        self.assertEqual({job.eval_resolution for job in jobs_7a}, {960})
        self.assertEqual({job.eval_resolution for job in jobs_7b}, {1920})

    def test_experiment8_size_pairs_match_screen_band(self) -> None:
        pairs = _experiment8_size_pairs()
        self.assertEqual(len(pairs), 21)
        self.assertTrue(all(128 <= width * depth <= 576 for width, depth in pairs))
        self.assertIn((32, 4), pairs)
        self.assertIn((8, 32), pairs)
        self.assertNotIn((32, 20), pairs)

    def test_experiment8_scheduler_emits_documented_jobs(self) -> None:
        jobs = _make_experiment8_screen_jobs(beam_width=4, seed=7267)
        self.assertEqual(len(jobs), 26)
        self.assertEqual({job.eval_resolution for job in jobs}, {120})
        self.assertEqual({job.beam_width for job in jobs}, {4})
        self.assertTrue(all(job.residual_depth == job.d * 2 for job in jobs))
        self.assertTrue(any(job.label.endswith("W12D16 final_only") for job in jobs))
        self.assertTrue(any(job.policy.residual_clip_policy == "intermediate_m11_final_01" for job in jobs))
        anchor = [job for job in jobs if job.k == 12 and job.residual_depth == 16 and job.policy.residual_clip_policy == "final_only"]
        self.assertEqual({job.modifier_label for job in anchor}, {"phase_only", "phase_gain", "phase_offset", "phase_gain_offset"})
        self.assertEqual({job.policy.modifier_policy for job in anchor}, {"none", "base_gain", "global_offset", "base_gain_global_offset"})

    def test_experiment9_scheduler_includes_budget_equivalence_jobs(self) -> None:
        jobs = _make_experiment9_screen_jobs(beam_width=4, seed=7267)
        self.assertEqual(len(jobs), 39)
        self.assertEqual({job.beam_width for job in jobs}, {4})
        self.assertEqual(sum(job.experiment9_section == "9A" for job in jobs), 18)
        self.assertEqual(sum(job.experiment9_section == "9B" for job in jobs), 8)
        self.assertEqual(sum(job.experiment9_section == "9C" for job in jobs), 5)
        budget_jobs = [job for job in jobs if job.experiment9_section == "9D"]
        self.assertEqual(len(budget_jobs), 8)
        self.assertEqual({job.k for job in budget_jobs}, {4, 6})
        self.assertEqual({job.policy.residual_clip_policy for job in budget_jobs}, {"final_only"})
        for job in budget_jobs:
            self.assertIn(job.budget_anchor_depth, {24, 32, 48, 64})
            self.assertEqual(job.budget_anchor_width, 8)
            self.assertEqual(
                job.budget_anchor_head_outputs,
                _experiment9_phase_head_outputs(8, job.budget_anchor_depth),
            )
            self.assertEqual(
                job.budget_actual_head_outputs,
                _experiment9_phase_head_outputs(job.k, job.residual_depth),
            )

    def test_summary_reports_strict_sampled_curve_perfection_rate(self) -> None:
        base = {
            "configuration": "cfg",
            "family": "test",
            "candidate": "fixture",
            "depth": 1,
            "eval_resolution": 120,
            "training_feature_grid": 120,
            "dense_outputs": 1,
            "categorical_logits": 1,
            "continuous_scalars": 0,
            "effective_index_bits": 0.0,
            "stored_codes": 1,
            "stored_floats": 0,
            "stored_bytes_float32": 0,
            "decoder_branches": 1,
            "topology_dependency": "none",
            "stage_widths": "1",
            "topology": "none",
            "rmse": 0.01,
            "derivative_rmse": 0.0,
            "node_max_error": 0.0,
            "duplicate_x_probe_count": 0,
            "elapsed_seconds_total": 0.0,
        }
        results = pd.DataFrame(
            [
                {**base, "dataset_index": 0, "max_abs_error": 0.019},
                {**base, "dataset_index": 1, "max_abs_error": 0.021},
            ]
        )
        subsets = pd.DataFrame(
            {
                "dataset_index": [0, 1],
                "configuration": ["cfg", "cfg"],
                "subset_all": [True, True],
                "subset_custom_ish": [True, True],
                "subset_gate_pulse_heavy": [False, False],
            }
        )
        summary, thresholds, _ = summarize_results(results, subsets)
        self.assertEqual(float(summary.loc[0, "all_eval_points_under_0.02"]), 0.5)
        row = thresholds[
            (thresholds["configuration"] == "cfg")
            & (thresholds["subset"] == "all")
            & (thresholds["metric"] == "all_eval_points")
            & (thresholds["threshold"] == 0.02)
        ].iloc[0]
        self.assertEqual(float(row.coverage), 0.5)

    def test_training_cache_key_includes_sample_and_clipping_when_needed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.csv"
            codebook = root / "codebook.json"
            catalog.write_text("x\n", encoding="utf-8")
            codebook.write_text("{}\n", encoding="utf-8")
            base_policy = Experiment7Policy("topology_balanced_common_then_tail", "none")
            clipped_policy = Experiment7Policy(
                "topology_balanced_common_then_tail",
                "none",
                residual_clip_policy="intermediate_m11_final_01",
            )
            key_a = _training_cache_key(
                experiment="8",
                policy=base_policy,
                k=12,
                d=8,
                eval_resolution=120,
                seed=1,
                quick=False,
                catalog_path=catalog,
                codebook_path=codebook,
                sample_hash="aaa",
            )
            key_b = _training_cache_key(
                experiment="8",
                policy=base_policy,
                k=12,
                d=8,
                eval_resolution=120,
                seed=1,
                quick=False,
                catalog_path=catalog,
                codebook_path=codebook,
                sample_hash="bbb",
            )
            key_c = _training_cache_key(
                experiment="8",
                policy=clipped_policy,
                k=12,
                d=8,
                eval_resolution=120,
                seed=1,
                quick=False,
                catalog_path=catalog,
                codebook_path=codebook,
                sample_hash="aaa",
            )
            self.assertNotEqual(key_a, key_b)
            self.assertNotEqual(key_a, key_c)

    def test_experiment8_memory_estimate_scales_with_width_depth(self) -> None:
        small = _estimate_peak_memory_mb(
            train_count=100,
            validation_count=50,
            resolution=120,
            residual_width=8,
            residual_depth=8,
            beam_width=4,
            batch_size=2,
        )
        large = _estimate_peak_memory_mb(
            train_count=100,
            validation_count=50,
            resolution=120,
            residual_width=32,
            residual_depth=32,
            beam_width=4,
            batch_size=2,
        )
        self.assertGreater(large, small)

    def _tiny_encoding_fixture(self) -> tuple[np.ndarray, PhaseChain, np.ndarray]:
        phase = np.linspace(0, 1, 64, endpoint=False)
        base = np.sin(2 * np.pi * phase).astype(np.float32)
        bases = np.stack([base, np.zeros_like(base)]).astype(np.float32)
        stage = np.zeros((len(TOPOLOGY_NAMES), 2, 64), dtype=np.float32)
        stage[:, 1] = np.cos(2 * np.pi * phase).astype(np.float32)
        chain = PhaseChain(
            "tiny",
            bases,
            (stage,),
            np.asarray([0, -1], dtype=np.int32),
            (np.tile(np.asarray([-1, 0], dtype=np.int32), (len(TOPOLOGY_NAMES), 1)),),
            ("layer_1_shared",),
            True,
            (1,),
            ("shared",),
            (np.zeros((len(TOPOLOGY_NAMES), 2), dtype=np.float32),),
        )
        targets = np.stack([base, base + 0.25 * stage[1, 1]]).astype(np.float32)
        return targets, chain, np.asarray([0, 1], dtype=np.int32)

    def test_training_cache_round_trip(self) -> None:
        phase = np.linspace(0, 1, 16, endpoint=False)
        bases = np.stack([np.sin(2 * np.pi * phase), np.zeros(16)]).astype(np.float32)
        stage = np.zeros((len(TOPOLOGY_NAMES), 1, 16), dtype=np.float32)
        chain = PhaseChain(
            "cached",
            bases,
            (stage,),
            np.asarray([0, -1], dtype=np.int32),
            (np.full((len(TOPOLOGY_NAMES), 1), -1, dtype=np.int32),),
            ("layer_1_shared",),
            True,
            (1,),
            ("shared",),
            (np.zeros((len(TOPOLOGY_NAMES), 1), dtype=np.float32),),
        )
        construction = pd.DataFrame({"stage": ["base"], "layer": [0]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_training_cache(
                root,
                "abc",
                chain=chain,
                construction=construction,
                prefix=bases,
                phase="complete",
                elapsed_seconds=12.5,
                complete=True,
            )
            self.assertTrue((root / "training_cache" / "abc" / "prefix.npy").exists())
            loaded = _load_trained_cache(root, "abc")
        self.assertIsNotNone(loaded)
        loaded_chain, loaded_construction, elapsed = loaded
        self.assertEqual(loaded_chain.name, "cached")
        self.assertEqual(len(loaded_chain.stages), 1)
        self.assertEqual(loaded_construction["stage"].iloc[0], "base")
        self.assertEqual(elapsed, 12.5)


if __name__ == "__main__":
    unittest.main()
