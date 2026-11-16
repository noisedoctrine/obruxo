from __future__ import annotations

import unittest
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from run_efficiency_benchmark import (MODEL_REGISTRY, STRATA, _duration_band, _stratified_sample,
                                      _summarize_measurements)


class EfficiencyBenchmarkTests(unittest.TestCase):
    def test_fixed_seed_selects_equal_distinct_duration_strata(self):
        candidates = [{"request_id": f"{band}-{index:03d}", "duration_seconds": duration}
                      for band, duration in zip(STRATA, (5.0, 7.0, 9.0, 10.0)) for index in range(40)]

        first = _stratified_sample(candidates, count=128, seed=157)
        second = _stratified_sample(candidates, count=128, seed=157)

        self.assertEqual([row["request_id"] for row in first], [row["request_id"] for row in second])
        self.assertEqual(len({row["request_id"] for row in first}), 128)
        self.assertEqual({band: sum(row["duration_band"] == band for row in first) for band in STRATA},
                         {band: 32 for band in STRATA})

    def test_duration_bands_cover_inclusive_ten_second_limit(self):
        self.assertEqual([_duration_band(value) for value in (5.0, 6.0, 8.0, 10.0)], list(STRATA))

    def test_summary_uses_per_recording_medians_and_duration_groups(self):
        records = [{"request_id": "a", "duration_seconds": 5.0, "duration_band": STRATA[0],
                    "latency_seconds": [1.0, 2.0, 3.0], "forward_seconds": [.5, 1.0, 1.5]},
                   {"request_id": "b", "duration_seconds": 7.0, "duration_band": STRATA[1],
                    "latency_seconds": [2.0, 2.0, 2.0], "forward_seconds": [1.0, 1.0, 1.0]}]

        result = _summarize_measurements(records)

        self.assertEqual(result["latency_median_ms"], 2000.0)
        self.assertEqual(result["duration_bands"][STRATA[0]]["latency_median_ms"], 2000.0)
        self.assertEqual(result["timed_calls"], 6)
        self.assertAlmostEqual(result["audio_seconds_per_wall_second"], 3.0)

    def test_sanitized_aggregate_contains_all_models_and_no_private_inputs(self):
        aggregate_path = PACKAGE / "reports" / "model_efficiency_benchmark.json"
        aggregate_text = aggregate_path.read_text(encoding="utf-8")
        result = json.loads(aggregate_text)

        self.assertEqual(set(result["models"]), set(MODEL_REGISTRY))
        self.assertNotIn("wav_relative_path", aggregate_text)
        self.assertNotIn("request_id", aggregate_text)
        self.assertNotIn("audio_sha256", aggregate_text)
        for model in result["models"].values():
            self.assertTrue(model["all_inputs_processed"])
            self.assertEqual(model["successful_recordings_all_passes"], 128)
            self.assertEqual(model["timed_passes"], 3)
            self.assertEqual(model["batch_size"], 1)
            for band in STRATA:
                self.assertEqual(model["aggregate"]["duration_bands"][band]["recordings"], 32)

    def test_report_links_efficiency_results_and_tradeoff_figure(self):
        report = (PACKAGE / "reports" / "FROZEN_ENCODER_RETRIEVAL_RESULTS.md").read_text(encoding="utf-8")
        self.assertIn("## Model efficiency and resource requirements", report)
        self.assertIn("figures/model_efficiency_vs_retrieval.png", report)
        self.assertIn("model_efficiency_benchmark.json", report)
        self.assertIn("original common-gallery Recall@1", report)
        for model in MODEL_REGISTRY.values():
            self.assertIn(model["name"], report)


if __name__ == "__main__":
    unittest.main()
