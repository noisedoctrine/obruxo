from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from lfo_era2.contracts import TopologyFlags, find_stage_keys, validate_topology_contract  # noqa: E402


class ContractTests(unittest.TestCase):
    def test_construction_only_topology_passes(self) -> None:
        result = validate_topology_contract(TopologyFlags(topology_used_in_construction=True))
        self.assertTrue(result.passed)
        self.assertEqual(result.violations, [])

    def test_runtime_topology_fails(self) -> None:
        result = validate_topology_contract(TopologyFlags(topology_used_in_targets=True))
        self.assertFalse(result.passed)
        self.assertEqual(result.violations, ["topology_used_in_targets"])

    def test_public_stage_key_detection(self) -> None:
        payload = {
            "residual_layer_1_index": 0,
            "nested": {"stage_1_index": 2},
        }
        self.assertEqual(find_stage_keys(payload), ["nested.stage_1_index"])


if __name__ == "__main__":
    unittest.main()

