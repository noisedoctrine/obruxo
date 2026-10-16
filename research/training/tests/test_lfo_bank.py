import json

import numpy as np
import pytest

from obruxo_data.hashing import canonical_json, canonical_sha256
from obruxo_training.full_scalar import FullScalarCodec
from obruxo_training.identity import PresetIdentity
from obruxo_training.lfo_bank import LfoBank, LfoCode, curve_to_lfo, fit_lfo_bank, sample_lfo
from obruxo_training.pilot import PilotPool
from obruxo_training.splits import FrozenVariantManifest


@pytest.fixture(scope="module")
def population():
    pool = PilotPool.create(32, codec=FullScalarCodec())
    raw, variants = {}, []
    parents = dict(pool.manifest.variants)
    for index, entry in enumerate(pool.entries):
        document = json.loads(pool.codec.decode(entry.target))
        rng = np.random.default_rng(index)
        for shape in document["settings"]["lfos"]:
            shape.update(num_points=4, points=np.column_stack(([0., .25, .75, 1.], rng.uniform(0, 1, 4))).reshape(-1).tolist(),
                         powers=[float(rng.uniform(-4, 4)), 0., 1., 0.], smooth=bool(index % 2))
        value = canonical_json(document).encode()
        state = PresetIdentity.from_bytes(value).state
        raw[state] = value
        variants.append((state, parents[entry.state_id]))
    manifest = FrozenVariantManifest(pool.manifest.parent, tuple(variants))
    train = [value for state, value in raw.items() if manifest.by_state[state].split == "train"]
    heldout = [value for state, value in raw.items() if manifest.by_state[state].split != "train"]
    assert train and heldout
    return manifest, train, heldout


@pytest.fixture(scope="module")
def bank(population):
    manifest, train, _ = population
    return fit_lfo_bank(train, manifest, maximum_curves=48)


def test_saved_bank_keeps_training_provenance_and_decodes_without_targets(bank, population):
    manifest, _, heldout = population
    saved = json.loads(json.dumps(bank.to_dict()))
    restored = LfoBank(saved)
    assert restored.bank_id == bank.bank_id
    assert restored.assets.base_dictionary.shape == (32, 97)
    assert len(saved["bank"]["fitted_rows"]) == 48
    assert all(manifest.by_state[row["state_id"]].split == "train" for row in saved["bank"]["fitted_rows"])
    original = np.stack([sample_lfo(shape) for shape in json.loads(heldout[0])["settings"]["lfos"]])
    codes, expected = bank.encode(original)
    actual = restored.decode([LfoCode.from_dict(code.to_dict()) for code in codes])
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (8, 97) and np.isfinite(actual).all()
    assert np.any(np.asarray([code.gains for code in codes]) != 0)
    for values in actual:
        shape = curve_to_lfo(values)
        assert shape["num_points"] == 97 and not shape["smooth"]
        np.testing.assert_allclose(sample_lfo(shape), values, atol=2e-6, rtol=0)


@pytest.mark.parametrize("which", ["heldout", "unknown", "duplicate", "empty"])
def test_fit_rejects_invalid_population_before_constructing_assets(population, monkeypatch, which):
    manifest, train, heldout = population

    def forbidden(*args, **kwargs):
        pytest.fail("dictionary fitting must not start on an invalid population")

    monkeypatch.setattr("obruxo_training.lfo_bank._construct_assets", forbidden)
    if which == "heldout":
        values = [train[0], heldout[0]]
    elif which == "unknown":
        document = json.loads(train[0])
        document["settings"]["osc_1_level"] = .001
        values = [canonical_json(document).encode()]
    else:
        values = [train[0], train[0]] if which == "duplicate" else []
    with pytest.raises(ValueError):
        fit_lfo_bank(values, manifest)


@pytest.mark.parametrize("change", ["payload", "policy", "heldout", "noop", "shape", "source"])
def test_bank_refuses_changed_payload_or_rehashed_invalid_contract(bank, change, population):
    document = bank.to_dict()
    if change in {"payload", "noop"}:
        document["bank"]["residual_dictionaries"][0][0][0] = .1
    elif change == "policy":
        document["bank"]["spec"]["phase_enabled"] = False
    elif change == "heldout":
        manifest, _, heldout = population
        assignment = manifest.by_state[PresetIdentity.from_bytes(heldout[0]).state]
        document["bank"]["fitted_rows"][0].update(state_id=assignment.state_id, lineage_id=assignment.lineage_id)
    elif change == "shape":
        document["bank"]["base_dictionary"].pop()
    else:
        document["bank"]["runtime_sources"] = {}
    if change != "payload":
        document["bank_id"] = canonical_sha256(document["bank"])
    with pytest.raises(ValueError):
        LfoBank(document)


@pytest.mark.parametrize("field,value", [("indices", [32] + [0] * 16), ("indices", [0] * 16),
                                        ("phases", [1.] * 17), ("phases", [float("nan")] * 17), ("gains", [2.01] * 16)])
def test_invalid_model_curve_codes_are_refused(field, value):
    values = {"indices": [0] * 17, "phases": [0.] * 17, "gains": [0.] * 16}
    values[field] = value
    with pytest.raises(ValueError):
        LfoCode.from_dict(values)


def test_bank_document_is_copied_and_grid_export_rejects_invalid_values(bank):
    saved = bank.to_dict()
    saved["bank"]["base_dictionary"][0][0] = 20
    assert bank.to_dict()["bank"]["base_dictionary"][0][0] <= 1
    for values in (np.zeros(96), np.full(97, np.inf), np.full(97, -.1)):
        with pytest.raises(ValueError):
            curve_to_lfo(values)


def test_fit_is_stable_under_input_order(bank, population):
    manifest, train, _ = population
    assert fit_lfo_bank(reversed(train), manifest, maximum_curves=48).bank_id == bank.bank_id


def test_lfo_fit_cli_writes_reloadable_assets_and_refuses_heldout_input(population, tmp_path):
    from obruxo_training.cli import main
    manifest, train, heldout = population
    split_path, preset, output = tmp_path / "splits.json", tmp_path / "train.vital", tmp_path / "bank"
    split_path.write_text(json.dumps(manifest.to_dict()))
    preset.write_bytes(train[0])
    assert main(["lfo-fit", "--presets", str(preset), "--split-manifest", str(split_path), "--output", str(output)]) == 0
    restored = LfoBank(json.loads((output / "bank.json").read_text()))
    assert restored.to_dict()["bank"]["source_presets"] == 1
    preset.write_bytes(heldout[0])
    refused = tmp_path / "refused"
    with pytest.raises(SystemExit) as error:
        main(["lfo-fit", "--presets", str(preset), "--split-manifest", str(split_path), "--output", str(refused)])
    assert error.value.code == 1 and not refused.exists()


def test_native_fixture_precision_is_fixed_before_identity_without_reassigning_roots():
    import importlib.util
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / "reports/scripts/run_lfo_bank_check.py"
    spec = importlib.util.spec_from_file_location("lfo_native_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    members, manifest = module.authored_population()
    original = PilotPool.create(256, codec=FullScalarCodec())
    assert manifest.parent.to_dict() == original.manifest.parent.to_dict()
    assert len(members) == 256
    for state, raw in members:
        assert state == PresetIdentity.from_bytes(raw).state
        for shape in json.loads(raw)["settings"]["lfos"]:
            for key in ("points", "powers"):
                assert shape[key] == np.asarray(shape[key], dtype=np.float32).tolist()
