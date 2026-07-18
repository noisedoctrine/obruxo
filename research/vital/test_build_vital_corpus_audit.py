from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from research.vital.build_vital_corpus_audit import main


class VitalCorpusAuditTests(unittest.TestCase):
    def test_recursive_scan_and_payload_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            (root / "preset_a").mkdir(parents=True)
            (root / "preset_b" / "nested").mkdir(parents=True)
            output = Path(directory) / "audit.zip"

            wave_source = {
                "synth_version": "1.5.5",
                "preset_name": "Wave",
                "settings": {
                    "volume": 5473.0,
                    "osc_1_on": 1,
                    "sample": {"samples": "A" * 10000, "sample_rate": 44100},
                    "modulations": [{"source": "lfo_1", "destination": "filter_1_cutoff"}],
                    "lfos": [],
                    "wavetables": [{
                        "groups": [{
                            "components": [{
                                "type": "Wave Source",
                                "keyframes": [{"position": 0, "wave_data": "B" * 10000}],
                            }],
                        }],
                    }],
                },
            }
            audio_source = {
                "synth_version": "1.6.1",
                "preset_name": "Audio",
                "settings": {
                    "volume": 5000.0,
                    "osc_1_on": 0,
                    "sample": {"samples": "C" * 10000, "sample_rate": 44100},
                    "modulations": [],
                    "lfos": [],
                    "wavetables": [{
                        "groups": [{
                            "components": [{"type": "Audio File Source", "audio_file": "D" * 12000}],
                        }],
                    }],
                },
            }
            (root / "preset_a" / "one.vital").write_text(json.dumps(wave_source), encoding="utf-8")
            (root / "preset_b" / "nested" / "two.VITAL").write_text(json.dumps(audio_source), encoding="utf-8")
            (root / "preset_b" / "broken.vital").write_text("{broken", encoding="utf-8")

            self.assertEqual(0, main([str(root), "--output", str(output), "--progress-every", "0"]))
            with zipfile.ZipFile(output) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(3, manifest["discovered_vital_files"])
                self.assertEqual(2, manifest["parsed_files"])
                self.assertEqual(1, manifest["failed_files"])
                self.assertNotIn("A" * 100, archive.read("json_paths.csv").decode("utf-8-sig"))
                components = list(csv.DictReader(io.TextIOWrapper(archive.open("wavetable_components.csv"), encoding="utf-8-sig")))
                self.assertEqual({"Audio File Source", "Wave Source"}, {row["component_type"] for row in components})


if __name__ == "__main__":
    unittest.main()
