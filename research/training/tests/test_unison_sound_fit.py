import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from obruxo_data.hashing import sha256_bytes
from obruxo_data.midi import Performance, TempoMap
from obruxo_data.render.base import RenderRequest
from obruxo_data.render.qa import audio_float32_sha256
from obruxo_training import unison_sound_fit as fit
from obruxo_training.audit import write_json
from obruxo_training.codec import PilotCodec
from obruxo_training.conditioned_short_fit import audit_cells
from obruxo_training.features import AudioFeatures, BasicPitchICASSP2022, tensor_digest
from obruxo_training.inference_profile import bind_rendering
from obruxo_training.model import PilotInverse, batch_features
from obruxo_training.native_fit import digest
from obruxo_training.paired_native_evidence import rescore_waveform
from obruxo_training.paired_native_step import reciprocal_updates
from obruxo_training.production_controls import ProductionControlAdapter
from obruxo_training.profiled_sampler import profile_metadata
from obruxo_training.reproducibility import fixed_generator_phases
from obruxo_training.unison_choice import SPEC, UnisonChoice, export_unison
from obruxo_training.unison_fit_audit import verify
from obruxo_training.unison_fit_data import Inputs, LIMITS, storage_bound


def wave(codec, candidate, performance):
    count = TempoMap.from_performance(performance).render_frame_count(performance.end_tick, codec.contract.tail_seconds, 44100)
    offset = int(sha256_bytes(str(performance.to_dict()).encode())[:3], 16) % 20
    t = np.arange(count, dtype=np.float64) / 44100
    audio = .1 * np.sin(2 * np.pi * (180 + 40 * candidate.categorical[0] + offset) * t)
    return np.stack((audio, audio * .8), axis=-1).astype(np.float32)


def receipt(codec, plan, candidate, performance, phase, label, attempt):
    preset = codec.decode(candidate)
    request = RenderRequest(preset_json=fixed_generator_phases(preset).decode(), performance=performance, renderer_id=codec.contract.renderer_id,
                            tail_seconds=codec.contract.tail_seconds, preroll_frames=32768)
    audio = wave(codec, candidate, performance)
    return audio, {"event": "outcome", "phase": phase, "label": label, "attempt": attempt, "status": "accepted",
                   "request_id": request.request_id, "source_sha256": sha256_bytes(preset), "controlled_sha256": sha256_bytes(request.preset_json.encode()),
                   "renderer_identity": plan["expected_renderer_identity"], "qa": {"audio_float32_sha256": audio_float32_sha256(audio)}}


class FakeSession(fit.PairedSession):
    def render(self, candidate, performance, phase, label):
        self.check_limits()
        attempt = sum(row["event"] == "attempt" and row["phase"] == phase for row in self.events)
        audio, result = receipt(self.codec, self.plan, candidate, performance, phase, label, attempt)
        identity = {key: result[key] for key in ("phase", "label", "attempt", "request_id", "source_sha256", "controlled_sha256")}
        self._append({"event": "attempt", **identity})
        self.renderer_identity = self.plan["expected_renderer_identity"]
        self._append(result)
        return audio, result


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    root = tmp_path_factory.mktemp("unison-finite-cost")
    parent, bridge, output = root / "parent", root / "bridge", root / "fit"
    for path in (parent / "audits", parent / "comparison_audio", bridge, output):
        path.mkdir(parents=True, exist_ok=True)
    codec, base = PilotCodec(), PilotInverse().eval()
    with torch.no_grad():
        base.continuous.weight.zero_()
        base.continuous.bias.copy_(torch.tensor(codec.default_target().continuous).logit())
        base.categorical[0].weight.zero_()
        base.categorical[0].bias.copy_(torch.tensor([0., 0., 2.]))
        base.categorical[1].weight.zero_()
        base.categorical[1].bias.copy_(torch.tensor([2., 0.]))
    bind_rendering(base, profile_metadata(32768))
    model = UnisonChoice(base).eval()
    frontend = AudioFeatures(BasicPitchICASSP2022().state_dict())
    torch.save(export_unison(model, frontend, codec, "simulation"), bridge / "model.pt")
    performances = [Performance(bpm=value).to_dict() for value in (100, 120, 140)]
    audios = [wave(codec, codec.default_target(), Performance.from_dict(value)) for value in performances]
    features = [frontend(audio) for audio in audios]
    recipe = {"state_id": "a", "target": codec.default_target().to_dict(), "performances": performances,
              "source_audio_sha256": [audio_float32_sha256(audio) for audio in audios], "source_feature_sha256": [tensor_digest(value) for value in features]}
    other = {**recipe, "state_id": "b"}
    cells = audit_cells([recipe], [recipe, other])
    plan = {"format": "software_simulation_not_native_quality_evidence", "training": [recipe], "fit_audit_cells": cells,
            "regression_state_ids": ["a", "b"], "training_view_roles": ["original", "held", "alternate"], "pool_id": "simulation",
            "expected_renderer_identity": {"simulation": True}, "updates": reciprocal_updates(1) + reciprocal_updates(1, seed=1),
            "optimizer": {"learning_rate": .001, "seed": 0, "gradient_norm_cap": 5.}, "prefix_check_updates": 2,
            "gates": {"fraction_of_available_gain": .5, "original_32_mrstft_p90_ratio": 1.1},
            "reconstruction_objective": "mean_input_and_transfer_v1", "rng_initialization": "python_numpy_and_torch_from_optimizer_seed",
            "limits": {**LIMITS, "optimizer_attempts": 5}, **profile_metadata(32768)}
    ctx = Inputs.__new__(Inputs)
    ctx.parent, ctx.plan, ctx.initial_path, ctx.model, ctx.frontend, ctx.codec = parent, plan, bridge / "model.pt", model, frontend, codec
    ctx.adapter = ProductionControlAdapter(codec, scales=SPEC["adapter_scales"])
    ctx.features = {(state, view): value for state in ("a", "b") for view, value in enumerate(features)}
    with torch.no_grad():
        ctx.baselines = {key: model(batch_features([{"features": value}])) for key, value in ctx.features.items()}
    ctx.lookup = {(cell["recipe"]["state_id"], cell["input_view"], cell["playback_view"]): index for index, cell in enumerate(cells)}
    ctx.cases = []
    for index, cell in enumerate(cells):
        playback = cell["playback_view"]
        performance = Performance.from_dict(performances[playback])
        reference, source = receipt(codec, plan, codec.default_target(), performance, "qualification", cell["label"] + ":source", 2 * index)
        candidate = ctx.target(index)
        audio, observed = receipt(codec, plan, candidate, performance, "qualification", cell["label"] + ":initial", 2 * index + 1)
        path = parent / "comparison_audio" / f"{audio_float32_sha256(audio)}.pt"
        if not path.exists():
            torch.save({"audio": torch.from_numpy(audio)}, path)
        observed["retained_audio"] = {"file": path.relative_to(parent).as_posix(), "sha256": digest(path)}
        initial = {"candidate": candidate.to_dict(), "receipt": observed, "scores": rescore_waveform(parent, observed, reference, codec.contract.tail_seconds)}
        case = {"label": cell["label"], "input_view": cell["input_view"], "playback_view": playback, "source_receipt": source, "rows": {"initial": initial}}
        ctx.cases.append(case)
        torch.save({"audio": torch.from_numpy(reference), "source_receipt": source, "rows": case["rows"],
                    "contract": {"recipe": cell["recipe"], "view": playback}}, parent / "audits" / f"{cell['label']}.pt")
    ctx.validate()
    write_json(output / "plan.json", plan)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fit, "PairedSession", FakeSession)
        result = fit.execute(ctx, output)
    assert result["complete"] and result["committed_updates_per_arm"] == 4
    return ctx, output


def test_complete_two_arm_fit_replays_and_rescores_every_cost_and_final(completed):
    ctx, output = completed
    result = verify(ctx, output)
    assert result["offline_training_replay_exact"] and result["comparison_waveforms_rescored"] == 38
    assert result["render_attempts"] == {"training": 14, "qualification": 16}
    assert result["native_requests_added"] == result["experimental_training_updates_added"] == 0
    assert set(result["final_model_sha256"]) == {"parameter", "spectral"}


@pytest.mark.parametrize("change", ["cost", "waveform", "checkpoint", "prefix"])
def test_independent_auditor_rejects_changed_cost_waveform_and_training(completed, tmp_path, change):
    ctx, source = completed
    output = tmp_path / "changed"
    shutil.copytree(source, output)
    if change == "cost":
        path = output / "costs.jsonl"
        rows = fit.lines(path)
        rows[0]["scores"]["spectral"] += 1.
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif change == "waveform":
        path = next((output / "comparison_audio").glob("*.pt"))
        path.write_bytes(b"changed")
    else:
        path = output / ("prefix-checkpoint.pt" if change == "prefix" else "checkpoint.pt")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        saved["models"]["spectral"]["correction.2.bias"][0] += .1
        torch.save(saved, path)
        if change == "prefix":
            proof = json.loads((output / "prefix-verification.json").read_text())
            proof["checkpoint_file_sha256"] = digest(path)
            write_json(output / "prefix-verification.json", proof)
    with pytest.raises((ValueError, RuntimeError), match="changed|reconstruct|reproduce"):
        verify(ctx, output)


def test_known_native_timeout_recovers_without_repeating_training(completed, tmp_path, monkeypatch):
    ctx, _source = completed
    output = tmp_path / "recovered"
    output.mkdir()
    write_json(output / "plan.json", ctx.plan)
    original, failed = FakeSession.render, False
    def render(session, candidate, performance, phase, label):
        nonlocal failed
        if not failed and label == "final:000:parameter":
            failed = True
            attempt = sum(row["event"] == "attempt" and row["phase"] == phase for row in session.events)
            _audio, normal = receipt(session.codec, session.plan, candidate, performance, phase, label, attempt)
            identity = {key: normal[key] for key in ("phase", "label", "attempt", "request_id", "source_sha256", "controlled_sha256")}
            session._append({"event": "attempt", **identity})
            failure = {"event": "outcome", **identity, "status": "error", "error_type": "RenderTimeout", "renderer_identity": session.renderer_identity}
            session._append(failure)
            return None, failure
        return original(session, candidate, performance, phase, label)
    monkeypatch.setattr(fit, "PairedSession", FakeSession)
    monkeypatch.setattr(FakeSession, "render", render)
    with pytest.raises(RuntimeError, match="not accepted"):
        fit.execute(ctx, output)
    assert torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)["step"] == 4
    assert fit.execute(ctx, output, resume=True)["complete"]
    result = verify(ctx, output)
    assert len(result["recovered_timeouts"]) == 1 and result["committed_updates_per_arm"] == 4
    assert len(fit.lines(output / "updates.jsonl")) == 4


def test_storage_bound_counts_reused_and_fresh_audio_without_exceeding_ceiling():
    parent = Path(__file__).parents[1] / "reports/data/conditioned_short_fit_plan.json"
    plan = json.loads(parent.read_text())
    bound = storage_bound(plan["fit_audit_cells"], PilotCodec().contract.tail_seconds)
    assert bound["candidate_waveforms_maximum"] == 480 and bound["unique_sources"] == 56
    assert bound["estimated_archive_upper_bound_bytes"] < LIMITS["archive_bytes"]


def test_oracle_chooses_one_category_for_both_performances(completed):
    ctx, output = completed
    store = fit.CostStore(output, ctx)
    own, transfer = (ctx.lookup[("a", 1, view)] for view in (1, 0))
    for index, values in ((own, [0., 10., 20.]), (transfer, [10., 0., 20.])):
        for choice, value in enumerate(values):
            row = store.costs[(index, choice)]
            store.costs[(index, choice)] = {**row, "scores": {**row["scores"], "spectral": value}}
    assert store.vectors([{"state": 0, "input_view": 1, "view": 0}]) == [[5., 5., 20.]]
    assert store.signal()["groups"]["1"]["oracle_mean"] == 5.
