"""Bounded native qualification of the alpha vocabulary, using training lineage only."""

import json
from time import perf_counter

import numpy as np

from obruxo_data.hashing import canonical_sha256
from obruxo_data.render.qa import compare_audio

from .audit import write_json
from .codec import PilotTarget
from .contracts import RenderRecipe
from .identity import PresetIdentity
from .performances import initial_bank
from .splits import SplitLedger
from .stream import NativeRendererFactory, OnlineRenderStream, StreamConfig


# An intervention must exceed both a small absolute floor and measured repeat noise.
AUDIBILITY = {"metric": "log_spectral_rmse", "minimum": .01, "repeat_multiplier": 5.}


class AlphaCheckSampler:
    def __init__(self, pool):
        self.pool = pool
        codec = pool.codec
        if codec.generator != "independent_authored_alpha_v1":
            raise ValueError("alpha qualification requires an alpha pool")
        parent = next((entry for entry in pool.entries if pool.manifest.by_state[entry.state_id].split == "train"), None)
        if parent is None:
            raise ValueError("alpha qualification requires a training lineage")
        ledger, performance = SplitLedger(pool.manifest), initial_bank()[0].performance()
        self.recipes, self.cases = [], []

        def target(**settings):
            base = codec.default_target()
            values, categories = list(base.continuous), list(base.categorical)
            for index, (name, lo, hi, _) in enumerate(codec.continuous):
                if name in settings:
                    values[index] = (settings.pop(name) - lo) / (hi - lo)
            for index, (name, choices) in enumerate(codec.categorical):
                if name in settings:
                    categories[index] = choices.index(settings.pop(name))
            if settings:
                raise ValueError(f"unknown qualification control: {sorted(settings)}")
            return codec.canonical_target(PilotTarget(tuple(values), tuple(categories)))

        def add(label, source_settings, ablated_settings):
            source = codec.decode(target(**source_settings))
            decoded = codec.decode(codec.encode(source))
            ablated = codec.decode(target(**ablated_settings))
            left, right = json.loads(source)["settings"], json.loads(decoded)["settings"]
            scalar_equal = all(np.float32(left[name]) == np.float32(right[name]) for name in codec.schema.parameters)
            nested_equal = all(left[name] == right[name] for name in left if name not in codec.schema.parameters)
            case = {"label": label, "parent_state_id": parent.state_id, "controls_equal_at_float32": scalar_equal,
                    "nested_state_equal": nested_equal, "state_ids": []}
            for raw in (source, source, decoded, ablated):
                state = PresetIdentity.from_bytes(raw).state
                ledger.register_variant(state, parent.state_id)
                case["state_ids"].append(state)
                self.recipes.append(RenderRecipe.create(raw, performance, schema=codec.schema, catalog=codec.catalog,
                                                       contract=codec.contract, ledger=ledger, purpose="evaluate",
                                                       sequence_id=len(self.recipes)))
            self.cases.append(case)

        for asset in codec.asset_ids[1:]:
            add(f"osc1_asset:{asset}", {"osc_1_asset": asset, "osc_1_wave_frame": 127.5}, {"osc_1_wave_frame": 127.5})
        add("osc1_frame", {"osc_1_asset": codec.asset_ids[1], "osc_1_wave_frame": 255.}, {"osc_1_asset": codec.asset_ids[1]})
        for asset in codec.asset_ids:
            add(f"osc2_layer:{asset}", {"osc_2_on": 1, "osc_2_asset": asset, "osc_2_wave_frame": 127.5}, {})
        for transpose in (-12, 12):
            add(f"osc2_transpose:{transpose}", {"osc_2_on": 1, "osc_2_transpose": transpose}, {"osc_2_on": 1})
        moving = {"filter_1_on": 1, "filter_1_cutoff": 65., "lfo_1_filter_route": 1, "modulation_1_amount": .25}
        add("lfo_route", moving, {"filter_1_on": 1, "filter_1_cutoff": 65.})
        for shape in ("ramp_up", "ramp_down"):
            add(f"lfo_shape:{shape}", {**moving, "lfo_1_shape": shape}, moving)
        add("chorus", {"chorus_on": 1, "chorus_dry_wet": .35, "chorus_mod_depth": .6}, {})
        self.sampler_id = canonical_sha256({"check": "alpha_codec_native_v1", "pool_id": pool.pool_id,
                                           "cases": self.cases, "audibility": AUDIBILITY})

    def recipe_at(self, sequence_id):
        return self.recipes[sequence_id]


def check_alpha_audio(pool, output, *, renderer_factory=None, timeout=60., progress=None):
    sampler = AlphaCheckSampler(pool)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"complete": False})
    write_json(output / "plan.json", {"sampler_id": sampler.sampler_id, "cases": sampler.cases,
                                      "requested": len(sampler.recipes), "audibility": AUDIBILITY})
    started = perf_counter()
    cases, group, accepted = [], [], 0
    config = StreamConfig(prefetch_clips=1, timeout_seconds=timeout)
    with OnlineRenderStream(sampler, len(sampler.recipes), config=config, renderer_factory=renderer_factory or NativeRendererFactory()) as stream:
        for outcome in stream:
            group.append(outcome)
            accepted += outcome.audio is not None
            stream.acknowledge(outcome.recipe.sequence_id)
            if progress:
                progress(outcome.recipe.sequence_id + 1, len(sampler.recipes), outcome.metadata["status"])
            if len(group) != 4:
                continue
            case = {**sampler.cases[len(cases)], "outcomes": [item.metadata for item in group]}
            if all(item.audio is not None for item in group):
                source, repeat, decoded, ablated = (item.audio for item in group)
                case.update(source_repeat=compare_audio(source, repeat), decoded_comparison=compare_audio(source, decoded),
                            intervention=compare_audio(source, ablated))
                floor = max(AUDIBILITY["minimum"], AUDIBILITY["repeat_multiplier"] * case["source_repeat"][AUDIBILITY["metric"]])
                case.update(intervention_floor=floor, intervention_audible=case["intervention"][AUDIBILITY["metric"]] > floor)
                del source, repeat, decoded, ablated
            case["passed"] = (case["controls_equal_at_float32"] and case["nested_state_equal"]
                              and case.get("source_repeat", {}).get("within_tolerance", False)
                              and case.get("decoded_comparison", {}).get("within_tolerance", False)
                              and case.get("intervention_audible", False))
            cases.append(case)
            group.clear()
    result = {"version": 1, "check": "alpha_codec_native_v1", "pool_id": pool.pool_id, "codec_id": pool.codec.codec_id,
              "sampler_id": sampler.sampler_id, "requested": len(sampler.recipes), "accepted": accepted, "cases": cases,
              "passed_cases": sum(case["passed"] for case in cases), "planned_cases": len(sampler.cases),
              "audibility": AUDIBILITY, "elapsed_seconds": perf_counter() - started,
              "renderer_factory": type(stream.factory).__name__, "audio_files_written": 0,
              "passed": accepted == len(sampler.recipes) and len(cases) == len(sampler.cases) and all(case["passed"] for case in cases)}
    write_json(output / "summary.json", result)
    write_json(output / "status.json", {"complete": True, "passed": result["passed"]})
    return result
