import base64
from copy import deepcopy
import json
import random
import wave

import numpy as np
import pytest

from obruxo_data.hashing import canonical_json, canonical_sha256
from obruxo_training.asset_codec import FactoryAssetCodec, validated_bank
from obruxo_training.codec import load_codec
from obruxo_training.factory_bank import IMPORT_POLICY, reference_sources, sample_from_wav
from obruxo_training.full_scalar import FullScalarCodec
from obruxo_training.identity import parse_document
from obruxo_training.pilot import PilotPool


@pytest.fixture(scope="module")
def bank():
    # Authored structural fixtures; real source fidelity is qualified separately with Vital.
    scalar = FullScalarCodec()
    settings = scalar.settings(scalar.default_target())
    wave_asset = deepcopy(settings["wavetables"][0])
    wave_asset["groups"][0]["components"][0]["keyframes"][0]["wave_data"] = base64.b64encode(np.sin(np.arange(2048) * .1).astype("<f4").tobytes()).decode()
    sample = deepcopy(settings["sample"])
    sample["sample_rate"] = 22050
    references = []
    for kind, asset in (("wavetable", wave_asset), ("sample", sample)):
        references.append({"asset_id": "authored-" + kind, "kind": kind, "provenance": "authored_reference", "path": kind,
                           "source_sha256": "1" * 64, "asset_sha256": canonical_sha256(asset), "asset": asset})
    value = {"version": 1, "policy": IMPORT_POLICY, "source_manifest_sha256": "2" * 64, "references": references}
    identity = canonical_sha256(value)
    return {"format": "obruxo_factory_asset_bank_v1", "bank_id": identity, "bank": value,
            "qualification": {"qualified": True, "bank_id": identity, "renderer_id": scalar.contract.renderer_id,
                              "source_unchanged": True, "references_passed": 2, "references_requested": 2, "plan_sha256": "3" * 64}}


@pytest.fixture(scope="module")
def codec(bank):
    return FactoryAssetCodec(bank)


@pytest.mark.parametrize("change", ["payload", "bank_id", "qualification_id", "partial", "unqualified", "renderer", "provenance"])
def test_changed_or_partially_qualified_factory_banks_are_refused(bank, change):
    changed = deepcopy(bank)
    if change == "payload":
        changed["bank"]["references"][0]["asset"]["name"] = "changed"
    elif change == "bank_id":
        changed["bank_id"] = "0" * 64
    elif change == "qualification_id":
        changed["qualification"]["bank_id"] = "0" * 64
    elif change == "partial":
        changed["qualification"]["references_passed"] = 1
    elif change == "unqualified":
        changed["qualification"]["qualified"] = False
    elif change == "renderer":
        changed["qualification"]["renderer_id"] = "other"
    else:
        changed["bank"]["references"][0]["provenance"] = "guessed_from_filename"
        identity = canonical_sha256(changed["bank"])
        changed["bank_id"] = changed["qualification"]["bank_id"] = identity
    with pytest.raises(ValueError):
        validated_bank(changed)


def test_asset_heads_roundtrip_all_numbered_slots_and_preserve_inactive_choices(codec):
    assert len(codec.continuous) == 424 and len(codec.categorical) == 410
    target = codec.default_target()
    target = type(target)(target.continuous, target.categorical[:-4] + (1, 1, 1, 1))
    raw = codec.decode(target)
    assert codec.encode(raw) == codec.canonical_target(target)
    settings = json.loads(raw)["settings"]
    assert settings["wavetables"][0] == settings["wavetables"][1] == settings["wavetables"][2] == codec._waves[1]
    assert settings["sample"] == codec._samples[1]
    assert codec.categorical_mask(target)[-4:] == (True, False, False, False)
    assert codec.catalog.identify("wavetable", parse_document(raw)["settings"]["wavetables"][2])


def test_unknown_assets_and_changed_curves_are_never_replaced_silently(codec):
    for field in ("sample", "lfos", "wavetables"):
        document = json.loads(codec.decode(codec.default_target()))
        if field == "sample":
            document["settings"][field]["sample_rate"] = 48000
        elif field == "lfos":
            document["settings"][field][0]["powers"][0] += .1
        else:
            document["settings"][field][0]["groups"][0]["components"][0]["keyframes"][0]["position"] = 1
        with pytest.raises(ValueError):
            codec.encode(canonical_json(document).encode())


def test_asset_renaming_uses_declared_canonical_metadata_and_same_dsp_content(codec):
    document = json.loads(codec.decode(codec.default_target()))
    document["settings"]["wavetables"][0]["name"] = "display-only rename"
    target = codec.encode(canonical_json(document).encode())
    assert target == codec.default_target()
    assert codec.to_dict()["asset_metadata_policy"] == "canonical_registered_names_and_authors"


def test_factory_pool_keeps_scalar_root_groups_through_reload_and_expansion(codec, tmp_path):
    scalar = PilotPool.create(24, seed=4, codec=FullScalarCodec())
    pool = PilotPool.create(24, seed=4, codec=codec)
    assert pool.manifest.parent.to_dict() == scalar.manifest.parent.to_dict()
    for index, entry in enumerate(pool.entries):
        expected = scalar.manifest.by_state[scalar.entries[index].state_id]
        actual = pool.manifest.by_state[entry.state_id]
        assert (actual.lineage_id, actual.split) == (expected.lineage_id, expected.split)
    path = tmp_path / "pool.json"
    pool.save(path)
    assert PilotPool.load(path).pool_id == pool.pool_id
    assert load_codec(codec.to_dict()).codec_id == codec.codec_id
    assert pool.expand(28).expansion_from(pool)["preserved_states"] == 24


def test_bank_timing_does_not_change_codec_identity(bank):
    changed = deepcopy(bank)
    changed["qualification"]["elapsed_seconds"] = 123.4
    assert FactoryAssetCodec(changed).codec_id == FactoryAssetCodec(bank).codec_id


def test_manifest_hashes_and_root_escape_are_checked_before_native_import(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    path = root / "asset.json"
    path.write_text("{}")
    import hashlib
    entry = {"asset_id": "fixture", "kind": "lfo", "path": path.name,
             "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "provenance": "factory_export"}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "references": [entry]}))
    assert len(reference_sources(manifest, root)[1]) == 1
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="hash changed"):
        reference_sources(manifest, root)
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    entry["path"] = "../outside.json"
    manifest.write_text(json.dumps({"version": 1, "references": [entry]}))
    with pytest.raises(ValueError, match="escapes"):
        reference_sources(manifest, root)


@pytest.mark.parametrize("channels", [1, 2])
def test_wav_import_preserves_pcm16_channels_frames_and_rate(tmp_path, channels):
    path = tmp_path / "fixture.wav"
    pcm = np.array([[-32768, 32767], [-123, 456], [0, 0], [32767, -32768]], dtype="<i2")[:, :channels].copy()
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(96000)
        output.writeframes(pcm.tobytes())
    sample = sample_from_wav(path)
    assert sample["length"] == 4 and sample["sample_rate"] == 96000
    np.testing.assert_array_equal(np.frombuffer(base64.b64decode(sample["samples"]), dtype="<i2"), pcm[:, 0])
    if channels == 2:
        np.testing.assert_array_equal(np.frombuffer(base64.b64decode(sample["samples_stereo"]), dtype="<i2"), pcm[:, 1])


def test_all_asset_families_are_sampled_without_changing_root_rng(codec):
    targets = [codec.sample_target(random.Random(index), index) for index in range(32)]
    assert any(codec.family(target) == "sampler" for target in targets)
    assert any(target.categorical[-4] == 1 for target in targets)
    assert any(target.categorical[-1] == 1 for target in targets)


def test_factory_asset_heads_receive_gradients_from_audio_features(codec):
    import torch
    from obruxo_training.features import PITCH_SIZE
    from obruxo_training.model import PilotInverse, loss_terms, model_spec
    torch.set_num_threads(1)
    model = PilotInverse(codec)
    assert model_spec(codec)["model"].startswith("factory_asset_")
    first = codec.default_target()
    first = type(first)(first.continuous, first.categorical[:-4] + (1, 0, 0, 0))
    second = codec.sample_target(random.Random(7), 7)
    second = type(second)(second.continuous, second.categorical[:-1] + (1,))
    features = {"timbre": torch.randn(2, 2, 96, 32), "pitch": torch.randn(2, PITCH_SIZE)}
    terms = loss_terms(model(features), [first, second], codec)
    terms["loss"].backward()
    assert torch.isfinite(terms["loss"])
    assert model.categorical[-4].weight.grad.count_nonzero() > 0
    assert model.categorical[-1].weight.grad.count_nonzero() > 0


def test_factory_pool_cli_embeds_the_qualified_bank_and_refuses_unqualified_content(bank, tmp_path):
    from obruxo_training.cli import main
    source, output = tmp_path / "bank.json", tmp_path / "pool"
    source.write_text(json.dumps(bank))
    assert main(["factory-pool", "--bank", str(source), "--output", str(output), "--count", "12"]) == 0
    pool = PilotPool.load(output / "pool.json")
    assert pool.codec.bank["bank_id"] == bank["bank_id"]
    assert pool.codec.contract.renderer_id == FullScalarCodec().contract.renderer_id
    changed = deepcopy(bank)
    changed["qualification"]["qualified"] = False
    source.write_text(json.dumps(changed))
    refused = tmp_path / "refused"
    with pytest.raises(SystemExit) as error:
        main(["factory-pool", "--bank", str(source), "--output", str(refused)])
    assert error.value.code == 1
    assert not refused.exists()


@pytest.mark.parametrize("qualified, exit_code", [(True, 0), (False, 2)])
def test_factory_bank_cli_propagates_qualification_failure(tmp_path, monkeypatch, qualified, exit_code):
    from obruxo_training.cli import main
    calls = []

    def qualify(root, output, **kwargs):
        calls.append((root, output, kwargs))
        return {"qualification": {"qualified": qualified}}

    monkeypatch.setattr("obruxo_training.factory_bank.qualify_factory_bank", qualify)
    root, output, manifest = tmp_path / "assets", tmp_path / "bank", tmp_path / "manifest.json"
    assert main(["factory-bank", "--factory-root", str(root), "--output", str(output),
                 "--manifest", str(manifest), "--timeout", "17"]) == exit_code
    assert calls[0][:2] == (root, output)
    assert calls[0][2]["manifest"] == manifest and calls[0][2]["timeout"] == 17
