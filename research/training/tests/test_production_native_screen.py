from collections import Counter
import copy
import json
from time import time
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from obruxo_data.hashing import sha256_bytes  # noqa: E402
from obruxo_data.midi import Performance  # noqa: E402
from obruxo_data.render.base import RenderRequest  # noqa: E402
from obruxo_data.render.qa import audio_float32_sha256  # noqa: E402
from obruxo_training import production_native_audit as audit  # noqa: E402
from obruxo_training import production_native_screen as screen  # noqa: E402
from obruxo_training.audit import write_json  # noqa: E402
from obruxo_training.codec import PilotCodec  # noqa: E402
from obruxo_training.features import PITCH_SIZE, TIMBRE_SHAPE, tensor_digest  # noqa: E402
from obruxo_training.mixed_input_performances import MixedSampler  # noqa: E402
from obruxo_training.model import PilotInverse, batch_features  # noqa: E402
from obruxo_training.pilot import PilotPool  # noqa: E402
from obruxo_training.production_controls import ProductionControlAdapter  # noqa: E402
from obruxo_training.production_direction import GATES, UPDATE, raw_prediction  # noqa: E402
from obruxo_training.profiled_sampler import profile_metadata  # noqa: E402
from obruxo_training.reproducibility import fixed_generator_phases  # noqa: E402
from obruxo_training.temporal_proxy_data import NativeSession  # noqa: E402


@pytest.fixture(scope="module")
def selection():
    torch.set_num_threads(2)
    pool = PilotPool.create(96, seed=0)
    sampler, codec = MixedSampler(pool), pool.codec
    features = {"timbre": torch.zeros(*TIMBRE_SHAPE), "pitch": torch.zeros(PITCH_SIZE)}
    targets = {entry.state_id: entry.target.to_dict() for entry in pool.entries}
    replay = []
    for sequence in range(sampler.epoch_size):
        recipe = sampler.recipe_at(sequence)
        replay.append({"state_id": recipe.state_id, "sequence_id": sequence, "recipe_id": recipe.recipe_id,
                       "target": targets[recipe.state_id], "audio_sha256": "0" * 64, "features": features})
    proof = {"initial_training_inputs": [{"recipe_id": row["recipe_id"], "audio_sha256": row["audio_sha256"],
                                         "features_sha256": tensor_digest(features)} for row in replay]}
    return pool, {"replay": replay}, PilotInverse(codec).eval(), ProductionControlAdapter(codec, scales=(32., 29.)), proof


def test_selection_is_train_only_reciprocal_and_balanced_within_every_category(selection):
    cases, features = screen.selected_cases(*selection)
    assert len(cases) == len(features) == 24
    pool = selection[0]
    targets = {entry.state_id: entry.target.categorical for entry in pool.entries}
    assert all(pool.manifest.by_state[row["state_id"]].split == "train" for row in cases)
    for index in range(12):
        original, wider = cases[2 * index:2 * index + 2]
        assert original["state_id"] == wider["state_id"]
        assert (original["input_view"], original["target_view"]) == (wider["target_view"], wider["input_view"])
        assert original["target_recipe_id"] == wider["input_recipe_id"]
    assert Counter((targets[row["state_id"]], row["target_view"]) for row in cases if row["role"] == "original_input") == Counter(
        {(category, view): 1 for category in targets.values() for view in (1, 2)})


def test_selection_rejects_unproved_cached_features(selection):
    pool, cache, model, adapter, proof = selection
    cases, _features = screen.selected_cases(*selection)
    changed = copy.deepcopy(proof)
    row = next(row for row in changed["initial_training_inputs"] if row["recipe_id"] == cases[0]["input_recipe_id"])
    row["features_sha256"] = "changed"
    with pytest.raises(ValueError, match="proof"):
        screen.selected_cases(pool, cache, model, adapter, changed)


def signal(candidate):
    values = candidate.continuous
    x = np.arange(8192, dtype=np.float64) / 44100
    frequency = 180 + 90 * values[3] + 20 * candidate.categorical[0]
    wave = (.05 + .1 * values[0]) * np.sin(2 * np.pi * frequency * x)
    return np.stack((wave * (1.1 - values[1]), wave * (.1 + values[1])), -1).astype(np.float32)


class FakeSession(NativeSession):
    def render(self, target, performance, phase, label):
        self.check_limits()
        assert sum(row["event"] == "attempt" for row in self.events) < self.plan["limits"]["qualification_attempts"]
        preset = self.codec.decode(target)
        request = RenderRequest(preset_json=fixed_generator_phases(preset).decode(), performance=performance,
                                renderer_id=self.codec.contract.renderer_id, tail_seconds=self.codec.contract.tail_seconds, preroll_frames=32768)
        identity = {"phase": phase, "label": label, "request_id": request.request_id,
                    "attempt": sum(row["event"] == "attempt" for row in self.events), "source_sha256": sha256_bytes(preset),
                    "controlled_sha256": sha256_bytes(request.preset_json.encode())}
        self._append({"event": "attempt", **identity})
        audio = signal(target)
        self.renderer_identity = self.plan["expected_renderer_identity"]
        receipt = {"event": "outcome", **identity, "status": "accepted", "renderer_identity": self.renderer_identity,
                   "qa": {"audio_float32_sha256": audio_float32_sha256(audio)}}
        self._append(receipt)
        return audio, receipt


@pytest.fixture
def completed(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(9)
    codec, model = PilotCodec(), PilotInverse().eval()
    adapter = ProductionControlAdapter(codec, scales=(32., 29.))
    features = {"timbre": torch.zeros(*TIMBRE_SHAPE), "pitch": torch.zeros(PITCH_SIZE)}
    with torch.no_grad():
        prediction = model(batch_features([{"features": features}]))
    source = codec.default_target()
    cases = [{"index": 0, "role": role, "state_id": "fixture", "source": source.to_dict(), "input_view": index,
              "target_view": 1 - index, "performance": Performance().to_dict(), "source_samples": 8192,
              "target_audio_sha256": audio_float32_sha256(signal(source)), "raw_prediction": raw_prediction(prediction),
              "production_patch": adapter.export(prediction["continuous"], prediction["categorical"])[0].to_dict()}
             for index, role in enumerate(GATES["roles"])]
    plan = {"format": "fixture_not_native_quality_evidence", "cases": cases, "gates": GATES, "update": UPDATE,
            "inference": {"model_sha256": tensor_digest(model.state_dict())}, "limits": screen.LIMITS,
            "expected_renderer_identity": {"fixture": True}, "adapter": adapter.contract(), **profile_metadata(32768)}
    bank = {(0, role): features for role in GATES["roles"]}
    monkeypatch.setattr(screen, "make_plan", lambda *_args: (plan, model, bank))
    monkeypatch.setattr(audit, "make_plan", lambda *_args: (plan, model, bank))
    monkeypatch.setattr(screen, "NativeSession", FakeSession)
    monkeypatch.setattr(screen, "MemoryMonitor", lambda: SimpleNamespace(peak_bytes=1024**2, errors=0, start=lambda: None, close=lambda: None))
    output = tmp_path / "direction"
    assert not screen.run_screen(None, None, None, None, None, None, output)["executed"]
    result = screen.run_screen(SimpleNamespace(codec=codec), None, None, None, None, None, output, execute=True)
    assert not result["qualified_production_direction"]  # Two software fixtures cannot pass a 24-case quality gate.
    return output, SimpleNamespace(codec=codec)


def test_simulated_native_pipeline_and_auditor_preserve_a_failed_quality_decision(completed):
    output, pool = completed
    result = audit.verify_screen(output, pool, None, None, None, None, None)
    assert result["all_model_gradients_and_local_steps_reconstructed"] and result["all_requests_reconstructed"]
    assert not result["qualified_production_direction"] and result["source_audio_hashes_verified"] == 2
    assert result["candidate_audio_rescored"] == 6 and not result["gradient_probe_audio_rescored"]
    assert result["native_requests_added"] == 0 and result["optimizer_updates_added"] == 0
    assert result["native_attempts"] <= 52


@pytest.mark.parametrize("change", ("model_gradient", "probe_loss", "candidate", "source", "sound_score", "waveform", "decision", "receipt"))
def test_auditor_rejects_corruption_beyond_outer_case_summary_equality(completed, change):
    output, pool = completed
    cases = json.loads((output / "cases.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    if change == "model_gradient":
        cases[0]["direction"]["gradients"]["model_gradient_sha256"] = "changed"
    elif change == "probe_loss":
        cases[0]["renders"][3]["scores"]["spectral"] += 1
    elif change == "candidate":
        cases[0]["renders"][-1]["candidate"]["continuous"][0] += .01
    elif change == "source":
        (output / cases[0]["source_tensor"]).write_bytes(b"changed")
    elif change == "sound_score":
        cases[0]["renders"][-1]["scores"]["mrstft"] += .1
        cases[0]["scores"]["joint"]["mrstft"] = cases[0]["renders"][-1]["scores"]["mrstft"]
    elif change == "waveform":
        (output / cases[0]["renders"][-1]["audio_tensor"]).write_bytes(b"changed")
    elif change == "decision":
        summary["qualified_production_direction"] = True
    else:
        path = output / "renders.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["controlled_sha256"] = "changed"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    summary["cases"] = cases
    write_json(output / "cases.json", cases)
    write_json(output / "summary.json", summary)
    with pytest.raises(ValueError, match="changed|reconstructed"):
        audit.verify_screen(output, pool, None, None, None, None, None)


def test_refused_run_does_not_replace_existing_evidence(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    write_json(output / "run.json", {"started_unix": time()})
    write_json(output / "status.json", {"earlier_failure": "preserved"})
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(ValueError, match="preserve evidence"):
        screen.run_screen(None, None, None, None, None, None, output, execute=True)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}
