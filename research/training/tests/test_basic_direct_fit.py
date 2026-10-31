import copy
import json
import shutil

import numpy as np
import pytest
import torch
from render_fixtures import LifecycleReceipts
from obruxo_training.batching import BatchConfig

from obruxo_data.hashing import sha256_bytes
from obruxo_data.midi import Performance, TempoMap
from obruxo_data.render.base import RenderRequest
from obruxo_data.render.qa import audio_float32_sha256
from obruxo_training import basic_direct_fit as fit
from obruxo_training.audit import write_json
from obruxo_training.basic_direct import SPEC, BasicDirect, export_direct
from obruxo_training.basic_direct_fit_audit import verify
from obruxo_training.basic_direct_fit_data import LIMITS, Inputs, make_plan, storage_bound
from obruxo_training.basic_direct_fit_step import matched_step
from obruxo_training.codec import PilotCodec, PilotTarget
from obruxo_training.conditioned_short_fit import audit_cells
from obruxo_training.features import AudioFeatures, BasicPitchICASSP2022, PITCH_SIZE, TIMBRE_SHAPE, tensor_digest
from obruxo_training.inference_profile import bind_rendering
from obruxo_training.model import PilotInverse, batch_features
from obruxo_training.native_fit import digest
from obruxo_training.paired_native_audit import equal_tree
from obruxo_training.paired_native_evidence import rescore_waveform
from obruxo_training.paired_native_step import reciprocal_updates
from obruxo_training.production_controls import ProductionControlAdapter
from obruxo_training.profiled_sampler import profile_metadata
from obruxo_training.reproducibility import fixed_generator_phases


def wave(codec, candidate, performance):
    count = TempoMap.from_performance(performance).render_frame_count(performance.end_tick, codec.contract.tail_seconds, 44100)
    t = np.arange(count, dtype=np.float64) / 44100
    values = candidate.continuous
    frequency = 180 + 30 * candidate.categorical[0] + 5 * candidate.categorical[1] + 10 * sum(values[2:])
    signal = (.02 + .08 * values[0]) * np.sin(2 * np.pi * frequency * t)
    return np.stack((signal * (1 - .5 * values[1]), signal * (.5 + .5 * values[1])), axis=-1).astype(np.float32)


def receipt(codec, plan, candidate, performance, phase, label, attempt):
    preset = codec.decode(candidate)
    request = RenderRequest(preset_json=fixed_generator_phases(preset).decode(), performance=performance, renderer_id=codec.contract.renderer_id,
                            tail_seconds=codec.contract.tail_seconds, preroll_frames=32768)
    audio = wave(codec, candidate, performance)
    return audio, {"event": "outcome", "phase": phase, "label": label, "attempt": attempt, "status": "accepted",
                   "request_id": request.request_id, "source_sha256": sha256_bytes(preset), "controlled_sha256": sha256_bytes(request.preset_json.encode()),
                   "renderer_identity": plan["expected_renderer_identity"], "qa": {"audio_float32_sha256": audio_float32_sha256(audio)}}


class FakeSession(LifecycleReceipts, fit.PairedSession):
    def render(self, candidate, performance, phase, label):
        self.check_limits()
        attempt = sum(row["event"] == "attempt" and row["phase"] == phase for row in self.events)
        if attempt >= self.plan["limits"][f"{phase}_attempts"]:
            raise RuntimeError("simulation native request limit")
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
    root = tmp_path_factory.mktemp("direct-sound-rehearsal")
    parent, bridge, output = root / "parent", root / "bridge", root / "fit"
    for path in (parent / "audits", parent / "comparison_audio", bridge, output):
        path.mkdir(parents=True, exist_ok=True)
    codec, base = PilotCodec(), PilotInverse().eval()
    with torch.no_grad():
        base.continuous.weight.zero_()
        base.continuous.bias.copy_(torch.full((9,), .4).logit())
        for layer, biases in zip(base.categorical, ([0., 0., 2.], [0., 2.]), strict=True):
            layer.weight.zero_()
            layer.bias.copy_(torch.tensor(biases))
    bind_rendering(base, profile_metadata(32768))
    model = BasicDirect(base).eval()
    frontend = AudioFeatures(BasicPitchICASSP2022().state_dict())
    torch.save(export_direct(model, frontend, codec, "simulation"), bridge / "model.pt")
    performances = [Performance(bpm=value).to_dict() for value in (100, 120, 140)]
    audios = [wave(codec, codec.default_target(), Performance.from_dict(value)) for value in performances]
    features = [frontend(audio) for audio in audios]
    recipe = {"state_id": "a", "target": codec.default_target().to_dict(), "performances": performances,
              "source_audio_sha256": [audio_float32_sha256(audio) for audio in audios], "source_feature_sha256": [tensor_digest(value) for value in features]}
    other = {**recipe, "state_id": "b"}
    cells = audit_cells([recipe], [recipe, other])
    updates = [{**row, "references": [0, 1]} for row in reciprocal_updates(1) + reciprocal_updates(1, seed=1)]
    plan = {"format": "software_simulation_not_native_quality_evidence", "batching": BatchConfig().to_dict(), "training": [recipe], "fit_audit_cells": cells,
            "regression_state_ids": ["a", "b"], "training_view_roles": ["original", "held", "alternate"], "pool_id": "simulation",
            "expected_renderer_identity": {"simulation": True}, "updates": updates,
            "optimizer": {"learning_rate": .001, "seed": 0, "gradient_norm_cap": 5.}, "prefix_check_updates": 2,
            "gates": {"six_groups_sound_ratio": .9, "original_32_mrstft_p90_ratio": 1.1},
            "limits": {**LIMITS, "optimizer_attempts": 5}, **profile_metadata(32768)}
    ctx = Inputs.__new__(Inputs)
    ctx.parent, ctx.plan, ctx.initial_path, ctx.model, ctx.frontend, ctx.codec = parent, plan, bridge / "model.pt", model, frontend, codec
    ctx.adapter = ProductionControlAdapter(codec, scales=SPEC["adapter_scales"])
    ctx.features = {(state, view): value for state in ("a", "b") for view, value in enumerate(features)}
    ctx.bytes = sum(fit.tensor_bytes(value) for value in ctx.features.values())
    with torch.no_grad():
        ctx.baselines = {key: model(batch_features([{"features": value}])) for key, value in ctx.features.items()}
    ctx.lookup = {(cell["recipe"]["state_id"], cell["input_view"], cell["playback_view"]): index for index, cell in enumerate(cells)}
    ctx.references = [ctx.lookup[(state, 0, 0)] for state in ("a", "b")]
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


def test_complete_direct_fit_replays_all_observations_and_rescores_finals(completed):
    ctx, output = completed
    result = verify(ctx, output)
    assert result["offline_training_replay_exact"] and result["comparison_waveforms_rescored"] == 24
    assert result["training_scalar_probes_replayed"] > 0 and not result["training_probe_audio_independently_rescored"]
    assert not result["qualified_for_release"] and not result["test_split_used"]
    assert all(all(groups.values()) for groups in result["result"]["later_gradient_access"].values())
    saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert all(len(row["estimates"]) == 4 and len(row["reference_penalized_losses"]) == 2 for row in saved["trace"])
    assert saved["trace"][0]["estimates"][0]["anchor"] != saved["trace"][1]["estimates"][0]["anchor"]


@pytest.mark.parametrize("change", ["probe", "waveform", "model", "prefix"])
def test_tampered_direct_evidence_is_refused(completed, tmp_path, change):
    ctx, source = completed
    output = tmp_path / "tampered"
    shutil.copytree(source, output)
    if change == "probe":
        path = output / "probes.jsonl"
        rows = fit.lines(path)
        rows[0]["loss"] += .1
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    elif change == "waveform":
        next((output / "comparison_audio").glob("*.pt")).write_bytes(b"changed")
    else:
        path = output / ("prefix-checkpoint.pt" if change == "prefix" else "checkpoint.pt")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        saved["models"]["spectral"]["correction.2.bias"][0] += .1
        torch.save(saved, path)
        if change == "prefix":
            proof = json.loads((output / "prefix-verification.json").read_text(encoding="utf-8"))
            proof["checkpoint_file_sha256"] = digest(path)
            write_json(output / "prefix-verification.json", proof)
    with pytest.raises((ValueError, RuntimeError), match="changed|reconstruct|reproduce"):
        verify(ctx, output)


def test_mid_probe_timeout_recovers_exact_state_with_all_attempts_charged(completed, tmp_path, monkeypatch):
    ctx, _source = completed
    output = tmp_path / "recovered"
    output.mkdir()
    write_json(output / "plan.json", ctx.plan)
    original, failed = FakeSession.render, False

    def render(session, candidate, performance, phase, label):
        nonlocal failed
        if not failed and label.startswith("probe:002:1:"):
            failed = True
            attempt = sum(row["event"] == "attempt" and row["phase"] == phase for row in session.events)
            _audio, normal = receipt(session.codec, session.plan, candidate, performance, phase, label, attempt)
            identity = {key: normal[key] for key in ("phase", "label", "attempt", "request_id", "source_sha256", "controlled_sha256")}
            session._append({"event": "attempt", **identity})
            session._append({"event": "outcome", **identity, "status": "error", "error_type": "RenderTimeout"})
            raise RuntimeError("simulated terminal RenderTimeout")
        return original(session, candidate, performance, phase, label)

    monkeypatch.setattr(FakeSession, "render", render)
    monkeypatch.setattr(fit, "PairedSession", FakeSession)
    with pytest.raises(RuntimeError, match="RenderTimeout"):
        fit.execute(ctx, output)
    assert torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)["step"] == 1
    result = fit.execute(ctx, output, resume=True)
    assert result["complete"] and len(fit.lines(output / "updates.jsonl")) == 5
    audit = verify(ctx, output)
    assert len(audit["recovered_timeouts"]) == 1 and audit["offline_training_replay_exact"]
    assert (output / "interruption" / "probes.jsonl").exists()


def test_failed_render_advances_neither_arm_and_candidate_ignores_parameter_labels():
    torch.set_num_threads(2)
    torch.manual_seed(12)
    base, codec = PilotInverse(), PilotCodec()
    model = BasicDirect(base)
    features = {"timbre": torch.rand(*TIMBRE_SHAPE), "pitch": torch.rand(PITCH_SIZE)}
    observations = [{"features": features, "target": codec.default_target(), "cells": [0, 1] if i < 2 else [0], "baseline": .2} for i in range(4)]

    def initialize():
        models = {arm: copy.deepcopy(model) for arm in ("parameter", "spectral")}
        return models, {arm: torch.optim.Adam(value.correction.parameters(), lr=.001) for arm, value in models.items()}

    models, opts = initialize()
    before = {arm: copy.deepcopy(value.state_dict()) for arm, value in models.items()}

    def fail(index, _patch):
        if index == 2:
            raise RuntimeError("native probe failed")
        return .3

    with pytest.raises(RuntimeError, match="native probe failed"):
        matched_step(models, opts, codec, observations, fail, 0, 48)
    assert all(equal_tree(before[arm], value.state_dict()) and not opts[arm].state for arm, value in models.items())
    models, opts = initialize()
    measured = matched_step(models, opts, codec, observations, lambda i, _patch: [.3, .4, .1, .4][i], 0, 48)
    assert measured["losses"]["spectral"] == pytest.approx(.25 * .3 + .5 * .4 + .125 * .1 + .125 * .6)
    altered = copy.deepcopy(observations)
    altered_target = PilotTarget(tuple([.8] * 9), (1, 1))
    for row in altered:
        row["target"] = altered_target
    other_models, other_opts = initialize()
    other = matched_step(other_models, other_opts, codec, altered, lambda i, _patch: [.3, .4, .1, .4][i], 0, 48)
    assert equal_tree(models["spectral"].state_dict(), other_models["spectral"].state_dict())
    assert measured["losses"]["parameter"] != other["losses"]["parameter"]


def test_storage_calculation_rejects_oversized_plan_and_limits_targets_to_four(completed):
    ctx, _output = completed
    bound = storage_bound(ctx.plan["fit_audit_cells"], ctx.codec.contract.tail_seconds)
    assert bound["target_cache_records"] == 4 and not bound["training_probe_audio_retained"]
    assert bound["resident_replay_upper_bound_bytes"] < LIMITS["replay_bytes"]
    with pytest.raises(ValueError, match="allocation"):
        storage_bound(ctx.plan["fit_audit_cells"] * 10000, ctx.codec.contract.tail_seconds)


def test_unverified_source_is_refused_before_render_or_model_loading(tmp_path):
    parent, bridge = tmp_path / "parent", tmp_path / "bridge"
    parent.mkdir()
    bridge.mkdir()
    write_json(parent / "verification.json", {"verified": False, "plan": {}})
    write_json(bridge / "summary.json", {})
    write_json(bridge / "plan.json", {})
    with pytest.raises(ValueError, match="qualified direct bridge"):
        make_plan(parent, bridge, tmp_path / "recent", tmp_path / "missing-cache", None)


def test_attempt_ledger_rejects_skipped_steps_and_excess_retries():
    plan = {"updates": [{"a": i} for i in range(3)], "limits": {"optimizer_attempts": 4}}
    charges = [{"step": i + 1, "attempt": i, "update": row} for i, row in enumerate(plan["updates"])]
    trace = [{**row} for row in charges]
    fit.require_charges(plan, charges, trace)
    with pytest.raises(ValueError, match="ledger changed"):
        fit.require_charges(plan, charges[1:], [])
    repeated = [{"step": 1, "attempt": i, "update": plan["updates"][0]} for i in range(3)]
    with pytest.raises(ValueError, match="interrupted attempt"):
        fit.require_charges(plan, repeated, [])
