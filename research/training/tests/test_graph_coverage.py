from dataclasses import replace
import hashlib
import json

import pytest
import torch

from obruxo_data.vital.full_contract import FullPresetContract
from obruxo_training.codec import load_codec
from obruxo_training.graph_codec import GraphCoverageCodec, coverage_routes, destination_schedule
from obruxo_training.graph_context import prepare_destination_context
from obruxo_training.pilot import PilotPool, PilotSampler, seeded_random
from obruxo_training.splits import FrozenMultiVariantManifest, FrozenVariantManifest, SplitManifest, build_splits
from test_curve_inputs import layout_codec  # noqa: F401
from test_combined import codec, expression  # noqa: F401
from test_factory_assets import bank  # noqa: F401
from test_lfo_bank import population  # noqa: F401


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def test_multiple_variants_keep_every_frozen_root_and_old_contract_unchanged():
    parent = build_splits([digest(index) for index in range(12)])
    pairs = tuple((digest(f"{root.state_id}-{variant}"), root.state_id) for root in parent.assignments for variant in range(4))
    saved = FrozenMultiVariantManifest(parent, pairs)
    assert SplitManifest.from_dict(saved.to_dict()) == saved
    assert len(saved.assignments) == 48 and len({entry.lineage_id for entry in saved.assignments}) == 12
    assert all(entry.split == parent.by_state[root].split for (state, root), entry in zip(saved.variants, saved.assignments, strict=True))
    with pytest.raises(ValueError, match="unique children"):
        FrozenVariantManifest(parent, pairs)
    with pytest.raises(ValueError, match="preserve every parent"):
        FrozenMultiVariantManifest(parent, tuple(pair for pair in pairs if pair[1] != parent.assignments[0].state_id))
    with pytest.raises(ValueError, match="unique children"):
        FrozenMultiVariantManifest(parent, pairs + (pairs[0],))
    single = FrozenVariantManifest(parent, tuple((digest(f"child-{root.state_id}"), root.state_id) for root in parent.assignments))
    assert single.to_dict()["version"] == 2 and SplitManifest.from_dict(single.to_dict()) == single


def test_referenced_connection_has_a_protected_audible_exit_and_sources_stay_present():
    settings = FullPresetContract().template()["settings"]
    sources = tuple(f"source-{index}" for index in range(32))
    for destination in ("modulation_1_amount", "modulation_32_power", "modulation_64_amount"):
        routes, selected, owners = coverage_routes(settings, sources, (destination,), 0, width=1)
        child = int(destination.split("_")[1]) - 1
        assert selected == (destination,) and child not in owners
        assert child + 32 not in owners if child < 32 else True
        assert routes[child]["destination"] in {"osc_1_pan", "osc_2_pan", "osc_3_pan", "sample_pan"}
        assert routes[child]["source"] == "lfo_2" and settings[f"modulation_{child + 1}_bypass"] == 0.
        # Each displaced synthetic source retains its other direct anchor.
        assert set(sources) - {sources[owners[0] % 32]} <= {route["source"] for route in routes}


def test_destination_schedule_covers_training_and_distributes_held_out_targets():
    roots = build_splits([digest(index) for index in range(256)])
    destinations = tuple(sorted(FullPresetContract().destinations))
    schedule = destination_schedule(roots, destinations, 3)
    assert schedule == destination_schedule(roots, destinations, 3)
    for split in ("train", "validation", "test"):
        targets = [name for entry in roots.assignments if entry.split == split for name in schedule[entry.state_id]]
        assert len(set(targets)) == min(len(targets), len(destinations))
        if split == "train":
            assert set(targets) == set(destinations)
        else:
            assert set(targets) != set(destinations[:len(targets)])
            assert len({name.split("_")[0] for name in targets}) >= 8
    assert any(schedule[entry.state_id] != destination_schedule(build_splits([digest(index) for index in range(256)], seed=1),
                                                              destinations, 3)[entry.state_id] for entry in roots.assignments)


def test_related_variants_share_selection_weight_and_bootstrap_unit():
    from obruxo_training.evaluate import curve_statistics

    def score(number):
        return {role: {"mrstft": number} for role in ("reference", "transfer")}

    def row(root, value):
        return {"lineage_id": root, "methods": {"model": score(value), "train_retrieval": score(2.)}}

    rows = [row("first", 0.)] * 4 + [row("second", 4.)]
    result = curve_statistics(rows, ["model"])
    assert result["bootstrap"]["unit"] == "lineage" and result["independent_lineages"] == 2 and result["planned_cases"] == 5
    model = result["checkpoints"]["model"]
    assert model["mean_reference_transfer_mrstft"] == 2. and model["against_train_retrieval"]["mean_delta"] == 0.
    assert model["against_train_retrieval"]["paired"] == 2
    assert result == curve_statistics([row("first", 0.), row("second", 4.)], ["model"]) | {"planned_cases": 5}
    rows[-1]["methods"]["model"].pop("transfer")
    assert curve_statistics(rows, ["model"])["selected"] is None
    with pytest.raises(ValueError, match="frozen lineage"):
        curve_statistics(rows + [{"methods": {}}], ["model"])


@pytest.fixture(scope="module")
def graph(layout_codec):  # noqa: F811
    return GraphCoverageCodec(layout_codec.base.bank, layout_codec.lfo_bank.to_dict())


@pytest.fixture(scope="module")
def pool(graph):
    return PilotPool.create(32, codec=graph)


def test_original_variants_remain_exact_and_graph_variants_are_active(pool):
    graph = pool.codec
    assert load_codec(graph.to_dict()).codec_id == graph.codec_id
    for index, entry in enumerate(pool.entries):
        raw = graph.source_preset(entry.target, index)
        assert graph.encode(raw) == entry.target
        if index % 4 == 0:
            root = index // 4
            target = graph.parent.sample_target(seeded_random(graph.parent.generator, graph.roots.seed, root), root)
            assert raw == graph.parent.source_preset(target, root)
        else:
            assert all(graph.routing.active_routes(graph._routing_target(entry.target)))
            assert set(graph.sources) <= {route["source"] for route in json.loads(raw)["settings"]["modulations"]}
    by_root = {}
    for entry in pool.manifest.assignments:
        by_root.setdefault(entry.lineage_id, set()).add(entry.split)
    assert len(by_root) == 8 and all(len(values) == 1 for values in by_root.values())


def test_pool_roundtrip_and_exact_training_recovery(pool, tmp_path):
    from obruxo_training.evaluate import curve_statistics, evaluate_pilot
    from obruxo_training.stream import StreamConfig
    from obruxo_training.trainer import TrainConfig, train_pilot
    from test_stream import FakeRendererFactory
    pool.save(tmp_path / "pool.json")
    assert PilotPool.load(tmp_path / "pool.json").pool_id == pool.pool_id
    assert PilotSampler(pool, policy="attack_compatible").epoch_size == pool.summary()["states_by_split"]["train"]
    config = TrainConfig(steps=2, batch_size=2, replay_capacity=4, render_budget=4, checkpoint_every=1, cpu_threads=2)

    def run(name, **kwargs):
        return train_pilot(pool, tmp_path / name, config=config, stream_config=StreamConfig(prefetch_clips=1), renderer_factory=FakeRendererFactory(), **kwargs)

    full = run("full")
    run("partial", stop_after=1)
    resumed = run("resumed", resume=tmp_path / "partial/checkpoint.pt")
    assert full["model_state_sha256"] == resumed["model_state_sha256"]
    saved = torch.load(tmp_path / "full/checkpoint.pt", weights_only=True)
    with pytest.raises(ValueError, match="frozen roots"):
        pool.codec.sample_target(seeded_random(pool.codec.generator, 1, 0), 0)
    invalid = replace(pool, manifest=build_splits([entry.state_id for entry in pool.entries]))
    with pytest.raises(ValueError, match="moved a root"):
        pool.codec.validate_manifest(invalid.manifest)
    assert saved["run_contract"]["pool_id"] == pool.pool_id
    evaluation = evaluate_pilot(pool, tmp_path / "full/checkpoint.pt", tmp_path / "evaluation", count=2, split="train",
                                renderer_factory=FakeRendererFactory())
    cases = evaluation["cases"]
    assert all(row["lineage_id"] == pool.manifest.by_state[row["state_id"]].lineage_id for row in cases)
    assert curve_statistics(cases, ["model"])["independent_lineages"] == len({row["lineage_id"] for row in cases})


def test_context_selects_rate_keytrack_and_formant_prerequisites(graph):
    for name, expected in (("lfo_1_frequency", 0.), ("lfo_1_tempo", 1.), ("lfo_1_keytrack_tune", 4.)):
        settings = graph.schema.template()["settings"]
        prepare_destination_context(settings, name, graph.base._waves, graph.base._samples)
        assert settings["lfo_1_sync"] == expected
        graph.schema.validate({**graph.schema.template(), "settings": settings}).require_valid()
    settings = graph.schema.template()["settings"]
    prepare_destination_context(settings, "filter_2_formant_spread", graph.base._waves, graph.base._samples)
    assert settings["filter_2_model"] == 5. and settings["filter_2_on"] == 1.
