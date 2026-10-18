"""Measure training-recipe coverage without claiming native or learned behavior."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from obruxo_data.hashing import canonical_sha256

from .codec import PilotTarget
from .full_scalar import EFFECTS, scalar_family
from .nested_codec import NestedStateCodec
from .performances import capabilities_for


def _scalar_view(codec, target):
    scalar = codec.base.scalar
    nc, nk = len(scalar.continuous), len(scalar.categorical)
    settings = scalar._values(PilotTarget(target.continuous[:nc], target.categorical[:nk]))
    names = [name for name, *_ in scalar.continuous] + list(scalar._categorical_controls)
    flags = codec.continuous_mask(target)[:nc] + codec.categorical_mask(target)[:nk]
    mask = {}
    for name, enabled in zip(names, flags, strict=True):
        if name in mask and mask[name] != bool(enabled):
            raise ValueError("heads for one scalar disagree on supervision eligibility")
        mask[name] = bool(enabled)
    if set(mask) != scalar.schema.scalar_names:
        raise ValueError("scalar supervision inventory is incomplete")
    return settings, mask


def _shape_stats():
    return {"supervised_states": 0, "shape_hashes": set(), "point_counts": set()}


def _record_shape(stats, shape, enabled):
    if not enabled:
        return
    stats["supervised_states"] += 1
    stats["shape_hashes"].add(canonical_sha256({key: value for key, value in shape.items() if key != "name"}))
    stats["point_counts"].add(shape["num_points"])


def _shape_report(stats):
    return {"supervised_states": stats["supervised_states"], "distinct_supervised_shapes": len(stats["shape_hashes"]),
            "supervised_point_counts": sorted(stats["point_counts"])}


def audit_capabilities(pool, *, progress=None):
    """Only frozen training states contribute; masks are declarations, not audibility."""
    codec = pool.codec
    from .curve_codec import CurveInputCodec
    from .graph_codec import GraphCoverageCodec
    from .corpus_codec import CorpusNativeCodec
    if not isinstance(codec, (NestedStateCodec, CurveInputCodec, GraphCoverageCodec, CorpusNativeCodec)):
        raise ValueError("coverage audit requires the complete nested-state codec")
    counts, values, stored = Counter(), defaultdict(set), defaultdict(set)
    names = sorted(codec.schema.scalar_names)
    lfos, warps, maps = ([_shape_stats() for _ in range(size)] for size in (8, 3, 64))
    asset_ids = [codec.base._wave_ids] * 3 + [codec.base._sample_ids]
    assets = [{"supervised_states": 0, "choices": set()} for _ in range(4)]
    seeds = [{"supervised_states": 0, "values": set()} for _ in range(3)]
    slots, sources, destinations, graphs, effect_sets = Counter(), set(), set(), set(), set()
    enabled_effects, trained_states = Counter(), 0
    for index, entry in enumerate(pool.entries):
        if pool.manifest.by_state[entry.state_id].split != "train":
            continue
        trained_states += 1
        settings, mask = _scalar_view(codec, entry.target)
        for name in names:
            stored[name].add(settings[name])
            if mask[name]:
                counts[name] += 1
                values[name].add(settings[name])
        document = json.loads(pool.source_preset(entry.target, index))["settings"]
        for stats, shape, enabled in zip(lfos, document["lfos"], codec.active_lfos(entry.target), strict=True):
            _record_shape(stats, shape, enabled)
        for stats, shape, enabled in zip(warps, document["custom_warps"], codec.active_warps(entry.target), strict=True):
            _record_shape(stats, shape, enabled)
        for stats, seed, enabled in zip(seeds, document["random_values"], codec.active_seeds(entry.target), strict=True):
            if enabled:
                stats["supervised_states"] += 1
                stats["values"].add(seed["seed"])
        offset = len(codec.base.scalar.categorical)
        asset_flags = codec.categorical_mask(entry.target)[offset:offset + 4]
        for stats, choice, enabled in zip(assets, entry.target.categorical[offset:offset + 4], asset_flags, strict=True):
            if enabled:
                stats["supervised_states"] += 1
                stats["choices"].add(choice)
        routes = document["modulations"]
        graphs.add(canonical_sha256([(slot, route["source"], route["destination"]) for slot, route in enumerate(routes, 1) if route["source"]]))
        for slot, (route, enabled) in enumerate(zip(routes, codec.routing.active_routes(codec._routing_target(entry.target)), strict=True)):
            if not enabled:
                continue
            slots[slot] += 1
            sources.add(route["source"])
            destinations.add(route["destination"])
            if "line_mapping" in route:
                _record_shape(maps[slot], route["line_mapping"], True)
        effects = tuple(effect for effect in EFFECTS if settings[effect + "_on"])
        effect_sets.add(effects)
        enabled_effects.update(effects)
        if progress and trained_states % 250 == 0:
            progress({"audited_training_states": trained_states})
    if not trained_states:
        raise ValueError("coverage audit needs at least one training state")
    rows = [{"name": name, "family": scalar_family(name), "supervised_states": counts[name],
             "distinct_stored_values": len(stored[name]), "distinct_supervised_values": len(values[name]),
             "status": "masked" if counts[name] == 0 else "constant" if len(values[name]) == 1 else "varied"} for name in names]
    families = {family: dict(Counter(row["status"] for row in rows if row["family"] == family)) for family in sorted({row["family"] for row in rows})}
    bank = codec.contract.performances()
    return {"audit": "nested_training_recipe_coverage_v1", "pool_id": pool.pool_id, "codec_id": codec.codec_id,
            "split_manifest_id": pool.manifest.manifest_id, "states_by_split": pool.summary()["states_by_split"], "audited_train_states": trained_states,
            **({"independent_roots_by_split": {split: sum(entry.split == split for entry in pool.manifest.parent.assignments)
                 for split in ("train", "validation", "test")}} if getattr(codec, "multiple_variants_per_root", False) else {}),
            "claim": "declared_output_and_training_supervision_coverage_only_not_native_audibility_or_learning",
            "source_sha256_lf": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "scalar_inventory_sha256": canonical_sha256(names), "scalar_count": len(rows), "scalar_status_counts": dict(Counter(row["status"] for row in rows)),
            "scalar_families": families, "scalars": rows,
            "structured": {"wavetables_and_sample": [{"slot": f"osc_{slot + 1}" if slot < 3 else "sample", "vocabulary_size": len(ids),
                "supervised_states": stats["supervised_states"], "distinct_supervised_choices": len(stats["choices"]),
                "supervised_asset_ids": [ids[index] for index in sorted(stats["choices"])]} for slot, (stats, ids) in enumerate(zip(assets, asset_ids, strict=True))],
                "lfos": [_shape_report(stats) for stats in lfos], "custom_warps": [_shape_report(stats) for stats in warps],
                "seed_slots": [{"supervised_states": stats["supervised_states"], "distinct_supervised_seeds": len(stats["values"])} for stats in seeds],
                "connections": {"supervised_slots": len(slots), "slot_state_counts": [slots[index] for index in range(64)],
                    "sources": sorted(sources), "destinations": sorted(destinations), "source_vocabulary_size": len(set(codec.routing.sources) - {""}),
                    "destination_vocabulary_size": len(set(codec.routing.destinations) - {""}), "distinct_stored_topologies": len(graphs),
                    "mapping_slots": [_shape_report(stats) for stats in maps]}},
            "effects": {"enabled_state_counts": dict(enabled_effects), "distinct_enabled_sets": len(effect_sets)},
            "performances": {"bank_id": codec.contract.performance_bank_id, "capabilities": capabilities_for(codec.contract.renderer_id).to_dict(),
                "reference": [entry.name for entry in bank if entry.role == "reference"], "transfer": [entry.name for entry in bank if entry.role == "transfer"]},
            "limits": {"mask_policy_is_not_audibility": True, "range_and_combination_completeness": False,
                "raw_corpus_layout_coverage": "not_audited", "render_attempts": 0, "optimizer_steps": 0, "final_test_audio": False, "model_promotion": False}}
