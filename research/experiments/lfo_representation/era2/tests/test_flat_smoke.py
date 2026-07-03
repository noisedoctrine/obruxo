from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from lfo_era2.assets import DecoderPolicy  # noqa: E402
from lfo_era2.contracts import find_stage_keys  # noqa: E402
from lfo_era2.flat import make_smoke_targets, run_flat_smoke, synthetic_flat_assets  # noqa: E402


class FlatSmokeTests(unittest.TestCase):
    def test_decode_uses_model_facing_encoding_without_topology(self) -> None:
        assets = synthetic_flat_assets(residual_layer_count=2, width=3, resolution=32)
        targets, encoding = make_smoke_targets(assets, row_count=6)
        reconstructed = run_decode(assets, encoding)
        self.assertEqual(reconstructed.shape, targets.shape)
        self.assertLess(float(abs(reconstructed - targets).max()), 1e-7)

    def test_smoke_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "smoke_flat"
            result = run_flat_smoke(output_dir, backend="numpy")
            for name in ("manifest.json", "summary.csv", "targets_schema.json", "topology_contract.json"):
                self.assertTrue((output_dir / name).exists(), name)

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            schema = json.loads((output_dir / "targets_schema.json").read_text(encoding="utf-8"))
            contract = json.loads((output_dir / "topology_contract.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["head_outputs_actual"], 48)
            self.assertEqual(manifest["head_outputs_formula"], "32 + D * W + (D + 1)")
            self.assertTrue(contract["passed"])
            self.assertFalse(manifest["topology_used_at_runtime"])
            self.assertFalse(manifest["topology_used_in_targets"])
            self.assertEqual(find_stage_keys(schema), [])
            field_names = [field["name"] for field in schema["fields"]]
            self.assertIn("residual_layer_1_index", field_names)
            self.assertNotIn("topology", json.dumps(schema))
            self.assertEqual(result["manifest"]["runtime_interface_id"], "flat_categorical_per_residual_layer")


def run_decode(assets, encoding):
    from lfo_era2.flat import decode_flat

    return decode_flat(assets, encoding, decoder_policy=DecoderPolicy())


if __name__ == "__main__":
    unittest.main()

