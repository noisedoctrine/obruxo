from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[2]
for directory in (PACKAGE, REPO / "research/data_generation", REPO / "research/training"):
    if str(directory) not in os.sys.path:
        os.sys.path.insert(0, str(directory))

from frozen_retrieval.core import digest
from frozen_retrieval.dataset import (_load_snapshot, _paired_source_midi, _remove_nonperformance_meta_events,
                                      bounded_excerpt, prepare_dataset)
from frozen_retrieval.evaluation import evaluate_dataset
from frozen_retrieval import feature_stage
from frozen_retrieval.rendering import render_dataset


REQUIRED_ENV = ("OBRUXO_ISSUE152_SNAPSHOT", "OBRUXO_ISSUE152_DATA_ROOT",
                "OBRUXO_ISSUE152_RENDERER_CONFIG", "OBRUXO_VITAL_PLUGIN")


@unittest.skipUnless(os.environ.get("OBRUXO_RUN_REAL_INTEGRATION") == "1",
                     "set OBRUXO_RUN_REAL_INTEGRATION=1 and the OBRUXO_ISSUE152_* data/plugin paths to run real Vital integration")
class RealPipelineIntegrationTests(unittest.TestCase):
    def test_real_validation_prepare_render_embed_evaluate(self):
        missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
        self.assertFalse(missing, f"missing integration configuration: {', '.join(missing)}")

        import yaml
        from obruxo_data.midi import Performance
        from obruxo_data.render.capabilities import RendererCapabilities
        from obruxo_data.hashing import canonical_sha256

        snapshot = Path(os.environ["OBRUXO_ISSUE152_SNAPSHOT"]).resolve(strict=True)
        data_root = Path(os.environ["OBRUXO_ISSUE152_DATA_ROOT"]).resolve(strict=True)
        renderer_config = Path(os.environ["OBRUXO_ISSUE152_RENDERER_CONFIG"]).resolve(strict=True)
        docs, status, _ = _load_snapshot(snapshot)
        config = yaml.safe_load(renderer_config.read_text(encoding="utf-8"))
        capabilities = RendererCapabilities.from_dict(config["capabilities"])

        eligible, used_midi = [], set()
        for member in docs["members.json"]["records"]:
            if member.get("split") != "validation" or not member.get("static_eligible"):
                continue
            source = data_root / Path(member["relative_path"])
            midi_path, reason = _paired_source_midi(data_root, source.parent, source.resolve(strict=True))
            if reason or midi_path is None or source.parent.parent != data_root / "data":
                continue
            try:
                performance = Performance.from_midi(midi_path)
                _remove_nonperformance_meta_events(performance)
                if not performance.validate(capabilities).valid:
                    continue
                excerpt, _ = bounded_excerpt(performance, capabilities)
            except Exception:
                continue
            midi_id = digest({"normalized_performance": excerpt.to_dict(), "policy": "normalized_midi_events_v1"})
            if midi_id in used_midi:
                continue
            used_midi.add(midi_id)
            eligible.append(member)
            if len(eligible) == 2:
                break
        self.assertEqual(2, len(eligible), "the validation corpus needs two distinct renderer-compatible preset/MIDI pairs")

        selected_states = {member["identity"]["state"] for member in eligible}
        mini_members = docs["members.json"]
        for member in mini_members["records"]:
            member["static_eligible"] = member.get("static_eligible") and member["identity"]["state"] in selected_states
        mini_docs = dict(docs)
        mini_docs["members.json"] = mini_members

        output_parent = PACKAGE / "outputs"
        output_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="obruxo-issue152-integration-", dir=output_parent) as temporary:
            root = Path(temporary)
            mini_snapshot = root / "snapshot"
            mini_snapshot.mkdir()
            status = dict(status)
            status["artifact_ids"] = {}
            for name, document in mini_docs.items():
                (mini_snapshot / name).write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                status["artifact_ids"][name] = canonical_sha256(document)
            (mini_snapshot / "status.json").write_text(json.dumps(status, sort_keys=True, indent=2) + "\n", encoding="utf-8")

            dataset = root / "dataset"
            prepared = prepare_dataset(mini_snapshot, data_root, renderer_config, dataset, split="validation",
                                       gallery_cap=2, max_render_requests=16)
            self.assertEqual(4, prepared["render_requests"])
            rendered = render_dataset(dataset, renderer_config, plugin_path=Path(os.environ["OBRUXO_VITAL_PLUGIN"]),
                                      batch_size=2, queue_depth=1)
            self.assertEqual(4, rendered["requested"])
            self.assertGreater(rendered["accepted"], 0)

            import numpy as np
            class TestAdapter:
                name = "clap_music"
                checkpoint_digest = "b" * 64
                lock_entry = {"revision": "a" * 40, "loader_revision": "integration_test_adapter_v1"}
                device, torch, numpy = "cpu", object(), np
                sample_rate = 48_000

                @staticmethod
                def extract(wav_path):
                    values = np.frombuffer(hashlib.sha256(wav_path.read_bytes()).digest(), dtype=np.uint8)
                    return {"official_clip": values.astype(np.float32) + 1.0}

            entry = TestAdapter.lock_entry
            with patch.object(feature_stage, "load_checkpoint_lock", return_value={"models": {"clap_music": entry}}), \
                    patch.object(feature_stage, "resolve_checkpoint", return_value=(root, entry, "b" * 64)), \
                    patch.object(feature_stage, "load_adapter", return_value=TestAdapter()):
                features = feature_stage.extract_dataset(dataset, "clap_music", root, root / "test-lock.json", device="cpu")
            failures = (dataset / "features/clap_music/feature_failures.private.jsonl").read_text(encoding="utf-8")
            self.assertEqual(rendered["accepted"], features["feature_count"],
                             f"extraction summary={features}; failures={failures}")
            self.assertEqual(0, features["failure_count"])
            evaluated = evaluate_dataset(dataset)
            self.assertIn("cross_midi_common_gallery", evaluated["coverage_by_condition"])
            primary = evaluated["model_results"]["clap_music:official_clip"]["cross_midi_common_gallery"]
            self.assertGreater(primary["acoustic_hard_negative_win_episodes"], 0)
            self.assertIn("acoustic_hard_negative_protocol", evaluated)


if __name__ == "__main__":
    unittest.main()
