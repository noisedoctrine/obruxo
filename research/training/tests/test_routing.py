from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from obruxo_data.hashing import canonical_json
from obruxo_training.codec import PilotTarget, load_codec
from obruxo_training.features import PITCH_SIZE
from obruxo_training.full_scalar import FullScalarCodec
from obruxo_training.lfo_bank import fit_lfo_bank
from obruxo_training.model import PilotInverse, batch_features, loss_terms, model_spec
from obruxo_training.pilot import PilotPool, PilotSampler
from obruxo_training.routing import FAMILIES, OrderedRoutingCodec
from obruxo_training.routing_check import check_routing, routing_cases
from test_factory_assets import bank  # noqa: F401
from test_lfo_bank import population  # noqa: F401


@pytest.fixture(scope="module")
def codec(bank, population):  # noqa: F811 - shared pytest fixtures
    manifest, train, _ = population
    return OrderedRoutingCodec(bank, fit_lfo_bank(train, manifest, maximum_curves=48).to_dict())


@pytest.fixture(scope="module")
def pool(codec):
    return PilotPool.create(32, codec=codec)


def document(codec):
    return json.loads(codec.decode(codec.default_target()))


def encode(codec, value):
    return codec.encode(canonical_json(value).encode())


def labels(codec, pool, targets):
    indices = {entry.target: index for index, entry in enumerate(pool.entries)}
    sources = [codec.source_preset(target, indices[target]) for target in targets]
    return {"curve_targets": torch.tensor(np.stack([codec.source_curves(raw) for raw in sources])),
            "route_curve_targets": torch.tensor(np.stack([codec.source_route_maps(raw) for raw in sources]))}


def test_ordered_heads_preserve_roots_original_sources_and_saved_targets(codec, pool, tmp_path):
    assert (len(codec.continuous), len(codec.categorical)) == (6896, 802)
    assert model_spec(codec)["model"].startswith("ordered_routing_")
    assert pool.manifest.parent.to_dict() == PilotPool.create(32, codec=FullScalarCodec()).manifest.parent.to_dict()
    assert {codec.family(entry.target) for entry in pool.entries} == set(FAMILIES)
    path = tmp_path / "pool.json"
    pool.save(path)
    restored = PilotPool.load(path)
    assert restored.pool_id == pool.pool_id and load_codec(codec.to_dict()).codec_id == codec.codec_id
    for index, entry in enumerate(pool.entries[:5]):
        assert codec.canonical_target(entry.target) == entry.target
        assert codec.decode(entry.target) == restored.codec.decode(entry.target)
        original = codec.source_preset(entry.target, index)
        assert encode(codec, json.loads(original)) == entry.target
        assert all(route["line_mapping"]["num_points"] == 3 for route in json.loads(original)["settings"]["modulations"] if "line_mapping" in route)
        assert all(route["line_mapping"]["num_points"] == 97 for route in codec.routes(entry.target) if "line_mapping" in route)


def test_every_slot_preserves_literal_references_and_all_five_controls(codec):
    value = document(codec)
    for slot in range(1, 65):
        value["settings"]["modulations"][slot - 1] = {"source": f"lfo_{(slot - 1) % 8 + 1}", "destination": f"modulation_{65 - slot}_amount"}
        value["settings"].update({f"modulation_{slot}_amount": .125, f"modulation_{slot}_power": -.25,
                                  f"modulation_{slot}_bipolar": float(slot % 2), f"modulation_{slot}_stereo": 1., f"modulation_{slot}_bypass": float(slot % 3 == 0)})
    target = encode(codec, value)
    result = json.loads(codec.decode(target))["settings"]
    assert result["modulations"] == value["settings"]["modulations"]
    for slot in range(1, 65):
        for field in ("amount", "power", "bipolar", "stereo", "bypass"):
            name = f"modulation_{slot}_{field}"
            assert result[name] == pytest.approx(value["settings"][name], abs=2e-7)
    changed = deepcopy(value)
    changed["settings"]["modulations"].reverse()
    assert encode(codec, changed).categorical != target.categorical


def test_full_source_and_destination_vocabulary_is_encodable_without_relabeling(codec):
    assert (len(codec.sources), len(codec.destinations)) == (32, 431)
    for start in range(0, len(codec.destinations), 64):
        value = document(codec)
        for slot, destination in enumerate(codec.destinations[start:start + 64]):
            value["settings"]["modulations"][slot] = {"source": codec.sources[slot % 32], "destination": destination}
        assert codec.routes(encode(codec, value)) == value["settings"]["modulations"]


def test_empty_slot_mapping_is_refused_and_latent_unused_predictions_stay_stored(codec):
    value = document(codec)
    value["settings"]["modulations"][0]["line_mapping"] = deepcopy(value["settings"]["lfos"][0])
    with pytest.raises(ValueError, match="empty connection"):
        encode(codec, value)
    target = codec.default_target()
    flags = list(target.categorical)
    flags[codec._curve_cat + 3] = 1
    changed = codec.canonical_target(PilotTarget(target.continuous, tuple(flags)))
    assert PilotTarget.from_dict(changed.to_dict()) == changed
    assert codec.routes(changed)[0] == {"source": "", "destination": ""}
    assert not codec.active_route_maps(changed)[0]


def test_amount_chains_and_modulator_cycles_reach_audible_routes_but_bypass_does_not(codec):
    value = document(codec)
    settings = value["settings"]
    routes = settings["modulations"]
    routes[0] = {"source": "lfo_1", "destination": "modulation_32_amount"}
    routes[31] = {"source": "lfo_2", "destination": "modulation_64_amount"}
    routes[63] = {"source": "lfo_3", "destination": "osc_1_pan"}
    settings.update(modulation_1_amount=.1, modulation_32_amount=0., modulation_64_amount=0.)
    target = encode(codec, value)
    assert [index for index, active in enumerate(codec.active_routes(target)) if active] == [0, 31, 63]
    assert codec.active_lfos(target) == (True, True, True, False, False, False, False, False)
    settings["modulation_32_bypass"] = 1.
    assert not any(codec.active_routes(encode(codec, value)))
    settings["modulation_32_bypass"] = 0.
    settings["modulation_64_amount"] = .1
    routes[0]["destination"], routes[31]["destination"] = "lfo_2_frequency", "lfo_1_frequency"
    routes[63]["source"] = "lfo_1"
    settings["modulation_32_amount"] = .1
    assert [index for index, active in enumerate(codec.active_routes(encode(codec, value))) if active] == [0, 31, 63]


def test_presence_classes_have_equal_total_weight_and_unused_structure_is_masked(codec, pool):
    for entry in pool.entries[:5]:
        target = entry.target
        weights, mask = codec.categorical_weights(target), codec.categorical_mask(target)
        classes = [0., 0.]
        for slot in range(64):
            offset = codec._curve_cat + slot * 4
            present = target.categorical[offset]
            classes[present] += weights[offset] * mask[offset]
            if not present:
                assert mask[offset:offset + 4] == (True, False, False, False)
        np.testing.assert_allclose(classes, [.5, .5])


def test_changed_original_map_or_connection_labels_are_refused(codec, pool):
    target = pool.entries[0].target
    flags = list(target.categorical)
    slot = next(slot for slot, route in enumerate(codec.routes(target)) if route["source"])
    offset = codec._curve_cat + slot * 4 + 1
    flags[offset] = (flags[offset] + 1) % len(codec.sources)
    with pytest.raises(ValueError, match="original ordered source"):
        codec.source_preset(PilotTarget(target.continuous, tuple(flags)), 0)
    continuous = list(target.continuous)
    continuous[codec._map_start + slot * 97 + 3] = .1234
    with pytest.raises(ValueError, match="original ordered source"):
        codec.source_preset(PilotTarget(tuple(continuous), target.categorical), 0)


def test_audio_features_train_connections_and_only_audible_maps(codec, pool):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    target = pool.entries[0].target
    model = PilotInverse(codec)
    prediction = model({"timbre": torch.randn(1, 2, 96, 32), "pitch": torch.randn(1, PITCH_SIZE)})
    expected = labels(codec, pool, [target])
    with pytest.raises(ValueError, match="original route maps"):
        loss_terms(prediction, [target], codec, curve_targets=expected["curve_targets"])
    terms = loss_terms(prediction, [target], codec, **expected)
    terms["loss"].backward()
    assert terms["route_map_mse"] > 0
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    for slot, active in enumerate(codec.active_routes(target)):
        offset = codec._curve_cat + slot * 4
        assert model.categorical[offset].weight.grad.count_nonzero()
        assert bool(model.categorical[offset + 1].weight.grad.count_nonzero()) == active
        assert bool(model.categorical[offset + 2].weight.grad.count_nonzero()) == active
        gradient = model.continuous.weight.grad[codec._map_start + slot * 97:codec._map_start + (slot + 1) * 97]
        assert bool(gradient.count_nonzero()) == codec.active_route_maps(target)[slot]


def test_routing_checkpoint_recovers_optimizer_rng_and_separate_map_labels(codec, tmp_path):
    from obruxo_training.stream import StreamConfig
    from obruxo_training.trainer import TrainConfig, replay_bytes, train_pilot, validate_replay
    from test_stream import FakeRendererFactory
    pool = PilotPool.create(12, codec=codec)
    config = TrainConfig(steps=3, replay_capacity=3, batch_size=2, render_budget=3, cpu_threads=1, checkpoint_every=1, sampling_policy="state_round_robin")

    def train(name, **kwargs):
        return train_pilot(pool, tmp_path / name, config=config, stream_config=StreamConfig(prefetch_clips=1), renderer_factory=FakeRendererFactory(), **kwargs)

    complete = train("complete")
    train("partial", stop_after=1)
    resumed = train("resumed", resume=tmp_path / "partial/checkpoint.pt")
    assert complete["model_state_sha256"] == resumed["model_state_sha256"] and complete["cursor"] == resumed["cursor"]
    left = torch.load(tmp_path / "complete/checkpoint.pt", weights_only=True)
    right = torch.load(tmp_path / "resumed/checkpoint.pt", weights_only=True)
    assert torch.equal(left["batch_rng"], right["batch_rng"]) and torch.equal(left["rng"]["torch"], right["rng"]["torch"])
    for index, state in left["optimizer"]["state"].items():
        assert all(torch.equal(value, right["optimizer"]["state"][index][name]) for name, value in state.items())
    assert set(batch_features(left["replay"])) == {"timbre", "pitch"}
    label_bytes = sum(item[name].numel() * item[name].element_size() for item in left["replay"] for name in ("curve_targets", "route_curve_targets"))
    features = [{"features": item["features"]} for item in left["replay"]]
    assert replay_bytes(left["replay"]) == label_bytes + replay_bytes(features)
    assert complete["route_curve_target_bytes"] == 3 * 64 * 97 * 4
    corrupted = deepcopy(left["replay"])
    corrupted[0]["route_curve_targets"][0, 0] += .001
    with pytest.raises(ValueError, match="original route maps differ"):
        validate_replay(corrupted, PilotSampler(pool, policy="state_round_robin"), left["cursor"], config)


def test_native_matrix_covers_full_vocab_all_slots_and_explicit_empty_mapping_rejection(codec):
    cases = routing_cases()
    names = {name for name, _, _ in cases}
    assert sum(repeats for _, _, repeats in cases) == 601
    assert {"source_" + name for name in codec.sources} <= names
    assert {"destination_" + name for name in codec.destinations} <= names
    assert {f"slot_{slot}" for slot in range(1, 65)} <= names
    assert {"empty_map", "self_amount", "modulator_cycle", "amount_cycle", "dense_64"} <= names


@pytest.mark.parametrize("expected_error", [True, False])
def test_native_check_accepts_only_the_documented_empty_slot_rejection(tmp_path, monkeypatch, expected_error):
    class Worker:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def render(self, request, **kwargs):
            route = json.loads(request.preset_json)["settings"]["modulations"][0]
            if not route["source"] and "line_mapping" in route:
                return None, {"status": "error", "message": "Vital did not load the requested modulations" if expected_error else "unrelated failure"}
            return None, {"status": "accepted", "qa": {"audio_float32_sha256": "1" * 64}}

    class Monitor:
        errors, peak_bytes = 0, 4096

        def start(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr("obruxo_training.routing_check.RenderWorker", Worker)
    monkeypatch.setattr("obruxo_training.routing_check.MemoryMonitor", Monitor)
    result = check_routing(tmp_path / "check", probe=True)
    assert result["passed"] == expected_error
    assert result["expected_rejected"] == (2 if expected_error else 0)
    assert result["attempted"] == 8 and result["accepted"] == 6


def test_routing_pool_cli_keeps_embedded_banks(codec, tmp_path):
    from obruxo_training.cli import main
    assets, curves, output = tmp_path / "assets.json", tmp_path / "curves.json", tmp_path / "pool"
    assets.write_text(json.dumps(codec.base.bank))
    curves.write_text(json.dumps(codec.lfo_bank.to_dict()))
    assert main(["routing-pool", "--asset-bank", str(assets), "--lfo-bank", str(curves), "--output", str(output), "--count", "12"]) == 0
    restored = PilotPool.load(output / "pool.json")
    assert isinstance(restored.codec, OrderedRoutingCodec) and restored.codec.codec_id == codec.codec_id
