from copy import deepcopy
import hashlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from obruxo_data.hashing import canonical_json
from obruxo_training.alpha import AlphaCodec, FAMILIES, import_wavetable, read_wavetable_bank
from obruxo_training.alpha_check import AlphaCheckSampler
from obruxo_training.cli import main
from obruxo_training.codec import PilotCodec, PilotTarget, load_codec
from obruxo_training.pilot import PilotPool, PilotSampler


@pytest.fixture(scope="module")
def bank():
    # Authored contract fixtures; actual DSP qualification uses the pinned installed factory bank.
    wave = PilotCodec().schema.load_init_document()["settings"]["wavetables"][0]
    references = []
    for index in range(2):
        source = canonical_json({**wave, "version": f"0.8.{index}"})
        references.append({"asset_id": f"authored-{index}", "provenance": "authored_reference",
                           "source_json": source, "source_sha256": hashlib.sha256(source.encode()).hexdigest()})
    return {"version": 1, "references": references}


@pytest.fixture(scope="module")
def alpha_pool(bank):
    return PilotPool.create(32, seed=3, codec=AlphaCodec(bank))


def test_alpha_roundtrip_and_nested_state_are_strict(alpha_pool):
    codec = alpha_pool.codec
    assert len(codec.continuous) == 19 and len(codec.categorical) == 9
    assert {codec.family(entry.target) for entry in alpha_pool.entries} == set(FAMILIES)
    for entry in alpha_pool.entries:
        raw = codec.decode(entry.target)
        target = codec.encode(raw)
        assert target.categorical == entry.target.categorical
        np.testing.assert_allclose(target.continuous, entry.target.continuous, atol=1e-14, rtol=0)
        assert len(codec.continuous_mask(target)) == 19 and len(codec.categorical_mask(target)) == 9
        left, right = json.loads(raw)["settings"], json.loads(codec.decode(target))["settings"]
        for name, *_ in codec.continuous:
            assert np.float32(left[name]) == np.float32(right[name])
        assert left["lfos"] == right["lfos"] and left["modulations"] == right["modulations"]
        assert left["wavetables"] == right["wavetables"]


def test_alpha_masks_and_canonicalizes_disabled_modules(alpha_pool):
    codec = alpha_pool.codec
    target = codec.canonical_target(PilotTarget((.7,) * 19, (0, 0, 0, 2, 1, 2, 1, 2, 0)))
    assert target.categorical == (0, 0, 0, 0, 1, 0, 0, 0, 0)
    assert codec.categorical_mask(target) == (True, True, True, False, True, False, False, False, True)
    assert codec.continuous_mask(target) == (True, True, False, False, False, True, True, True, True, True,
                                              False, False, False, False, False, False, False, False, False)
    defaults = codec.default_target()
    assert not codec.continuous_mask(defaults)[9]
    assert all(active or value == default for value, default, active in
               zip(target.continuous, defaults.continuous, codec.continuous_mask(target), strict=True))


@pytest.mark.parametrize("change", ["route", "extra_route", "shape", "asset", "fixed_effect", "inactive", "osc3"])
def test_alpha_rejects_unrepresented_state(alpha_pool, change):
    codec = alpha_pool.codec
    document = json.loads(codec.decode(codec.default_target()))
    settings = document["settings"]
    if change in {"route", "extra_route"}:
        settings["modulations"][int(change == "extra_route")] = {"source": "lfo_2", "destination": "osc_1_level"}
    elif change == "shape":
        settings["lfos"][0]["points"][1] = .9
    elif change == "asset":
        settings["wavetables"][0]["version"] = "0.8.9"
    elif change == "fixed_effect":
        settings["chorus_voices"] = 3.
    elif change == "inactive":
        settings["osc_2_level"] = .2
    else:
        settings["osc_3_on"] = 1.
    with pytest.raises(ValueError):
        codec.encode(canonical_json(document).encode())


@pytest.mark.parametrize("change", ["hash", "duplicate", "legacy", "provenance"])
def test_bank_rejects_unverified_or_ambiguous_references(bank, change):
    changed = deepcopy(bank)
    reference = changed["references"][0]
    if change == "hash":
        reference["source_sha256"] = "0" * 64
    elif change == "duplicate":
        changed["references"].append(deepcopy(reference))
    elif change == "provenance":
        reference["provenance"] = "guessed_from_name"
    else:
        source = json.loads(reference["source_json"])
        source.pop("version")
        reference["source_json"] = canonical_json(source)
        reference["source_sha256"] = hashlib.sha256(reference["source_json"].encode()).hexdigest()
    with pytest.raises(ValueError):
        AlphaCodec(changed)


def test_factory_import_does_not_relabel_unregistered_sources(bank):
    for reference in bank["references"]:
        raw = reference["source_json"].encode()
        assert import_wavetable(raw) == json.loads(raw)


def test_manifest_cli_embeds_verified_sources_and_rejects_escape(bank, tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    entries = []
    for index, reference in enumerate(bank["references"]):
        path = root / f"source-{index}.vitaltable"
        path.write_bytes(reference["source_json"].encode())
        entries.append({**{key: value for key, value in reference.items() if key != "source_json"}, "path": path.name})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "references": entries}), encoding="utf-8")
    output = tmp_path / "pool"
    assert main(["alpha-pool", "--wavetable-manifest", str(manifest), "--wavetable-root", str(root),
                 "--output", str(output), "--count", "5"]) == 0
    pool = PilotPool.load(output / "pool.json")
    assert pool.codec.to_dict()["wavetable_bank"] == bank
    assert {pool.codec.family(entry.target) for entry in pool.entries} == set(FAMILIES)
    outside = tmp_path / "outside.vitaltable"
    outside.write_bytes(bank["references"][0]["source_json"].encode())
    entries[0]["path"] = "../outside.vitaltable"
    manifest.write_text(json.dumps({"version": 1, "references": entries}), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        read_wavetable_bank(manifest, root)


def test_alpha_pool_and_codec_contracts_survive_serialization(alpha_pool, tmp_path):
    codec = alpha_pool.codec
    descriptor = json.loads(json.dumps(codec.to_dict()))
    assert load_codec(descriptor).codec_id == codec.codec_id
    descriptor["lfo_shapes"][0]["points"][0] = .7
    with pytest.raises(ValueError, match="changed codec"):
        load_codec(descriptor)
    assert codec.to_dict()["lfo_shapes"][0]["points"][0] == 0.
    path = tmp_path / "pool.json"
    alpha_pool.save(path)
    loaded = PilotPool.load(path)
    assert loaded.pool_id == alpha_pool.pool_id
    assert loaded.entries == alpha_pool.entries
    assert loaded.pool_id == PilotPool.create(32, seed=3, codec=codec).pool_id
    expanded = alpha_pool.expand(36)
    assert expanded.entries[:32] == alpha_pool.entries
    assert expanded.expansion_from(alpha_pool)["preserved_states"] == 32
    sampler = PilotSampler(loaded, policy="state_round_robin")
    assert all(sampler.recipe_at(index).split == "train" for index in range(sampler.epoch_size))
    with pytest.raises(ValueError, match="training cannot"):
        # Check split enforcement in the common recipe boundary through a held-out sampler.
        from obruxo_training.contracts import RenderRecipe
        from obruxo_training.splits import SplitLedger
        split = next(entry.split for entry in loaded.manifest.assignments if entry.split != "train")
        heldout = PilotSampler(loaded, split=split).recipe_at(0)
        request = heldout.render_request()
        RenderRecipe.create(request.preset_json.encode(), request.performance, schema=codec.schema, catalog=codec.catalog,
                            contract=codec.contract, ledger=SplitLedger(loaded.manifest), purpose="train", sequence_id=0)


def test_qualification_is_bounded_and_uses_only_training_lineage(alpha_pool):
    sampler = AlphaCheckSampler(alpha_pool)
    assert len(sampler.cases) == 12 and len(sampler.recipes) == 48
    assert all(recipe.split == "train" and recipe.performance_role == "reference" for recipe in sampler.recipes)
    assert len({recipe.lineage_id for recipe in sampler.recipes}) == 1
    assert all(case["controls_equal_at_float32"] and case["nested_state_equal"] for case in sampler.cases)
    assert all(case["state_ids"][0] != case["state_ids"][3] for case in sampler.cases)
    assert sampler.sampler_id == AlphaCheckSampler(alpha_pool).sampler_id
    for recipe in sampler.recipes:
        settings = json.loads(recipe.render_request().preset_json)["settings"]
        assert all(settings[f"osc_{index}_random_phase"] == 0 for index in range(1, 4))


@pytest.mark.parametrize("rejected", [False, True])
def test_qualification_fails_on_inaudible_interventions_and_retains_failed_cases(alpha_pool, tmp_path, monkeypatch, rejected):
    from obruxo_training import alpha_check

    class Stream:
        factory = object()

        def __init__(self, sampler, count, **kwargs):
            self.sampler, self.count = sampler, count

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def acknowledge(self, sequence):
            pass

        def __iter__(self):
            for index in range(self.count):
                failed = rejected and index == 0
                yield SimpleNamespace(recipe=self.sampler.recipe_at(index), audio=None if failed else np.ones((16, 2), dtype=np.float32),
                                      metadata={"status": "rejected" if failed else "accepted"})

    monkeypatch.setattr(alpha_check, "OnlineRenderStream", Stream)
    result = alpha_check.check_alpha_audio(alpha_pool, tmp_path / "check")
    assert not result["passed"] and result["passed_cases"] == 0
    assert result["requested"] == 48 and result["accepted"] == 48 - int(rejected)
    assert result["planned_cases"] == len(result["cases"]) == 12


def test_alpha_loss_masks_inactive_categorical_and_continuous_gradients(alpha_pool):
    torch = pytest.importorskip("torch")
    from obruxo_training.model import loss_terms
    codec = alpha_pool.codec
    target = codec.default_target()
    values = torch.full((1, 19), .9375, requires_grad=True)
    logits = [torch.zeros(1, len(choices), requires_grad=True) for _, choices in codec.categorical]
    loss_terms({"continuous": values, "categorical": logits}, [target], codec)["loss"].backward()
    assert [bool(value) for value in values.grad[0]] == list(codec.continuous_mask(target))
    assert [bool(value.grad.abs().sum()) for value in logits] == list(codec.categorical_mask(target))
    active = codec.sample_target(random.Random(0), 4)
    assert all(codec.categorical_mask(active))


def test_alpha_train_recovery_export_and_audio_only_inference(alpha_pool, tmp_path):
    torch = pytest.importorskip("torch")
    from obruxo_training.evaluate import load_evaluation_checkpoint
    from obruxo_training.features import tensor_digest
    from obruxo_training.trainer import TrainConfig, load_model, predict_audio, train_pilot
    from test_stream import FakeRendererFactory
    config = TrainConfig(mode="online", steps=4, replay_capacity=3, batch_size=2, checkpoint_every=1,
                         fresh_examples=3, render_budget=3, cpu_threads=2)

    def run(name, **kwargs):
        return train_pilot(alpha_pool, tmp_path / name, config=config, renderer_factory=FakeRendererFactory(), **kwargs)

    full = run("full")
    run("part", stop_after=2)
    resumed = run("resumed", resume=tmp_path / "part/checkpoint.pt")
    assert full["complete"] and resumed["complete"]
    assert full["model_state_sha256"] == resumed["model_state_sha256"]
    assert full["cursor"] == resumed["cursor"] and full["totals"]["accepted"] == 3
    model, frontend, codec = load_model(tmp_path / "full/model.pt")
    assert len(model.categorical) == 9 and model.continuous.out_features == 19
    assert codec.codec_id == alpha_pool.codec.codec_id
    assert tensor_digest(model.state_dict()) == full["model_state_sha256"]
    _, evaluated, _, _ = load_evaluation_checkpoint(tmp_path / "full/checkpoint.pt", alpha_pool)
    assert tensor_digest(evaluated.state_dict()) == full["model_state_sha256"]
    audio = np.tile((.1 * np.sin(np.arange(44100) * .05)).astype(np.float32)[:, None], (1, 2))
    predicted = predict_audio(model, frontend, codec, audio)
    assert codec.encode(codec.decode(predicted)).categorical == predicted.categorical
    saved = torch.load(tmp_path / "full/model.pt", weights_only=True)
    saved["model_spec"]["continuous_heads"] = 9
    torch.save(saved, tmp_path / "tampered.pt")
    with pytest.raises(ValueError, match="unsupported inference"):
        load_model(tmp_path / "tampered.pt")
