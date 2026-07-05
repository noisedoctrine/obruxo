from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from lfo_era2.rmse_gap_audit import (  # noqa: E402
    Era1Encoding,
    decode_era1_arrays,
    load_era1_chain,
    load_era1_encoding,
    run_rmse_gap_audit,
)
from lfo_era2.curve import circular_shift  # noqa: E402


class RmseGapAuditTests(unittest.TestCase):
    def test_circular_shift_tiny_phase_stays_near_identity(self) -> None:
        curve = np.linspace(0.0, 1.0, 120, dtype=np.float32)
        shifted = circular_shift(curve, np.asarray([1.0e-15], dtype=np.float32))[0]
        self.assertLess(float(np.max(np.abs(shifted - curve))), 1e-5)

    def test_artifact_decoder_applies_residual_layer_gains(self) -> None:
        chain = _tiny_chain()
        encoding = Era1Encoding(
            dataset_index=np.asarray([0], dtype=np.int32),
            base_index=np.asarray([0], dtype=np.int16),
            base_phase=np.asarray([0.0], dtype=np.float32),
            stage_indices=[np.asarray([1], dtype=np.int16)],
            stage_phases=[np.asarray([0.0], dtype=np.float32)],
            stage_gains=[np.asarray([0.25], dtype=np.float32)],
        )
        conditions = np.asarray([0], dtype=np.int32)
        with_gain = decode_era1_arrays(chain, encoding, conditions, use_stage_gains=True)
        without_gain = decode_era1_arrays(chain, encoding, conditions, use_stage_gains=False)
        self.assertAlmostEqual(float(with_gain[0, 1]), 0.625)
        self.assertAlmostEqual(float(without_gain[0, 1]), 1.0)

    def test_chain_and_encoding_load_from_era1_artifact_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chain_dir = root / "chain"
            chain_dir.mkdir()
            manifest = {
                "name": "tiny",
                "stage_widths": [2],
                "stage_labels": ["layer_1_shared"],
                "stage_branches": ["shared"],
                "topology_conditioned": True,
            }
            (chain_dir / "manifest.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")
            np.savez_compressed(
                chain_dir / "codebook.npz",
                bases=np.zeros((1, 4), dtype=np.float32),
                base_sources=np.asarray([0], dtype=np.int32),
                stage_0=np.zeros((3, 2, 4), dtype=np.float32),
                stage_source_0=np.zeros((3, 2), dtype=np.int32),
                stage_rotation_0=np.zeros((3, 2), dtype=np.float32),
            )
            paths = root / "paths.csv"
            paths.write_text(
                "dataset_index,eval_resolution,base_index,base_phase,stage_1_index,stage_1_phase,stage_1_gain\n"
                "7,4,0,0.0,1,0.25,0.5\n",
                encoding="utf-8",
            )
            chain = load_era1_chain(chain_dir)
            encoding = load_era1_encoding(paths, stage_count=1)
            self.assertEqual(chain.stages[0].shape, (3, 2, 4))
            self.assertEqual(int(encoding.dataset_index[0]), 7)
            self.assertAlmostEqual(float(encoding.stage_gains[0][0]), 0.5)

    def test_report_can_be_written_when_artifact_replay_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_rmse_gap_audit(
                checkpoint_dir=root / "missing_checkpoint",
                catalog_path=root / "missing_catalog.csv",
                output_dir=root / "out",
                report_path=root / "report.md",
                row_limit=2,
            )
            report = Path(result["report"]).read_text(encoding="utf-8")
            self.assertIn("RMSE Gap Forensic Audit", report)
            self.assertIn("reconstruction-pipeline problem", report)
            self.assertTrue((root / "out" / "probe_summary.csv").exists())


def _tiny_chain():
    from lfo_era2.rmse_gap_audit import Era1Chain

    bases = np.asarray([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    stage = np.zeros((3, 2, 4), dtype=np.float32)
    stage[:, 1] = np.asarray([0.0, 0.5, 0.0, -0.5], dtype=np.float32)
    return Era1Chain(
        name="tiny",
        bases=bases,
        stages=(stage,),
        stage_labels=("layer_1_shared",),
        stage_branches=("shared",),
        topology_conditioned=True,
    )


if __name__ == "__main__":
    unittest.main()
