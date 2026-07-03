from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from lfo_era2.accounting import flat_categorical_budget  # noqa: E402
from lfo_era2.contracts import TopologyFlags, find_stage_keys  # noqa: E402
from lfo_era2.manifest import ExperimentRowManifest  # noqa: E402


class ManifestTests(unittest.TestCase):
    def test_required_fields_present(self) -> None:
        manifest = ExperimentRowManifest(
            experiment_id="unit",
            oracle_construction_id="construction",
            runtime_interface_id="flat_categorical_per_residual_layer",
            decoder_policy_id="final_clip",
            base_dictionary_size=32,
            residual_layer_count=3,
            scalar_families=["phase"],
            dictionary_scope="per_residual_layer",
            codebook_storage_count=44,
            budget=flat_categorical_budget(residual_layer_count=3, width=4),
            topology_flags=TopologyFlags(),
        )
        self.assertEqual(manifest.missing_required_fields(), [])
        payload = manifest.as_dict()
        self.assertEqual(payload["D"], 3)
        self.assertEqual(find_stage_keys(payload), [])


if __name__ == "__main__":
    unittest.main()

