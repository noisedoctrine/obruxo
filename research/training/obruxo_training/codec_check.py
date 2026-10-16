from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from obruxo_data.hashing import canonical_sha256
from obruxo_data.render.qa import compare_audio

from .audit import write_json
from .codec import CATEGORICAL, CONTINUOUS, PilotTarget, load_codec
from .contracts import RenderRecipe
from .identity import PresetIdentity
from .performances import initial_bank
from .reproducibility import FULL_PHASE_CONTROL, OSCILLATOR_PHASE_CONTROL, fixed_generator_phases, fixed_oscillator_phases
from .splits import SplitLedger
from .stream import NativeRendererFactory, OnlineRenderStream, StreamConfig


class CodecCheckSampler:
    def __init__(self, pool):
        self.pool = pool
        full = getattr(pool.codec, "balanced_loss", False)
        coverage = {}
        for entry in pool.entries:
            if pool.manifest.by_state[entry.state_id].split == "train":
                coverage.setdefault(pool.codec.family(entry.target) if full else entry.target.categorical, entry)
        if not coverage:
            raise ValueError("codec qualification requires authored training states")
        codec, ledger, performance = pool.codec, SplitLedger(pool.manifest), initial_bank()[0].performance()
        serialized = getattr(codec, "lossy_curves", False)
        restored = load_codec(codec.to_dict()) if serialized else None
        self.recipes, self.cases = [], []
        names = sorted(codec.schema.scalar_names) if full else [entry[0] for entry in CONTINUOUS + CATEGORICAL]
        for categories, entry in sorted(coverage.items()):
            source = codec.decode(entry.target)
            decoded = (restored.decode(PilotTarget.from_dict(entry.target.to_dict())) if serialized else codec.decode(codec.encode(source)))
            decoded_id = PresetIdentity.from_bytes(decoded).state
            ledger.register_variant(decoded_id, entry.state_id)
            left, right = json.loads(source)["settings"], json.loads(decoded)["settings"]
            equal = all(np.float32(left[name]) == np.float32(right[name]) for name in names)
            self.cases.append({"state_id": entry.state_id, **({"family": categories} if full else {"categories": list(categories)}),
                               "decoded_state_id": decoded_id, "controls_equal_at_float32": equal})
            if serialized:
                self.cases[-1]["serialized_target_and_decoder_exact"] = source == decoded
            for raw in (source, source, decoded):
                raw = fixed_generator_phases(raw) if serialized else fixed_oscillator_phases(raw)
                ledger.register_variant(PresetIdentity.from_bytes(raw).state, entry.state_id)
                self.recipes.append(RenderRecipe.create(raw, performance, schema=codec.schema, catalog=codec.catalog,
                                                       contract=codec.contract, ledger=ledger, purpose="evaluate",
                                                       sequence_id=len(self.recipes)))
        check = "lfo_serialized_target_native_v1" if serialized else "full_scalar_codec_native_v1" if full else "pilot_codec_native_v2"
        self.sampler_id = canonical_sha256({"check": check, "pool_id": pool.pool_id, "cases": self.cases,
                                           "phase_control": FULL_PHASE_CONTROL if serialized else OSCILLATOR_PHASE_CONTROL})

    def recipe_at(self, sequence_id):
        return self.recipes[sequence_id]


def check_codec_audio(pool, output: Path, *, renderer_factory=None, timeout: float = 60, progress=None) -> dict:
    if pool.codec.generator == "independent_authored_alpha_v1":
        from .alpha_check import check_alpha_audio
        return check_alpha_audio(pool, output, renderer_factory=renderer_factory, timeout=timeout, progress=progress)
    sampler = CodecCheckSampler(pool)
    full = getattr(pool.codec, "balanced_loss", False)
    serialized = getattr(pool.codec, "lossy_curves", False)
    config = StreamConfig(prefetch_clips=1, timeout_seconds=timeout)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"complete": False})
    cases, group, accepted = [], [], 0
    with OnlineRenderStream(sampler, len(sampler.recipes), config=config,
                             renderer_factory=renderer_factory or NativeRendererFactory(renderer_id=pool.codec.contract.renderer_id)) as stream:
        for outcome in stream:
            group.append(outcome)
            accepted += outcome.audio is not None
            stream.acknowledge(outcome.recipe.sequence_id)
            if progress:
                progress(outcome.recipe.sequence_id + 1, len(sampler.recipes), outcome.metadata["status"])
            if len(group) != 3:
                continue
            case = {**sampler.cases[len(cases)], "outcomes": [item.metadata for item in group]}
            if all(item.audio is not None for item in group):
                source, repeat, decoded = (item.audio for item in group)
                case.update(source_repeat=compare_audio(source, repeat), decoded_comparison=compare_audio(source, decoded))
                if full:
                    case.update(source_repeat_exact=np.array_equal(source, repeat), decoded_audio_exact=np.array_equal(source, decoded))
                del source, repeat, decoded
            cases.append(case)
            group.clear()
    expected_coverage = len(pool.codec.to_dict()["families"]) if full else len(CATEGORICAL[0][1]) * len(CATEGORICAL[1][1])
    result = {"version": 1 if full else 2, "check": "lfo_serialized_target_native_v1" if serialized else "full_scalar_codec_native_v1" if full else "pilot_codec_native_v2",
              "pool_id": pool.pool_id, "codec_id": pool.codec.codec_id,
              "phase_control": FULL_PHASE_CONTROL if serialized else OSCILLATOR_PHASE_CONTROL,
              "cases": cases, "covered_category_combinations": len(cases), "expected_category_combinations": expected_coverage,
              "requested": len(sampler.recipes), "accepted": accepted,
              "all_controls_equal_at_float32": all(case["controls_equal_at_float32"] for case in cases),
              "decoded_within_reference_tolerance": sum(case.get("decoded_comparison", {}).get("within_tolerance", False) for case in cases),
              "source_repeats_within_reference_tolerance": sum(case.get("source_repeat", {}).get("within_tolerance", False) for case in cases),
              "renderer_factory": type(stream.factory).__name__, "audio_files_written": 0}
    result["passed"] = (accepted == len(sampler.recipes) and len(cases) == expected_coverage
                        and result["all_controls_equal_at_float32"] and result["decoded_within_reference_tolerance"] == len(cases))
    if full:
        result["covered_families"] = [case["family"] for case in cases]
        result["expected_families"] = pool.codec.to_dict()["families"]
        result.pop("covered_category_combinations")
        result.pop("expected_category_combinations")
        result["passed"] &= all(case.get("source_repeat_exact") and case.get("decoded_audio_exact") for case in cases)
    if serialized:
        result["representation_roundtrip"] = "serialized_targets_only_raw_curve_reencoding_is_lossy"
        result["passed"] &= all(case["serialized_target_and_decoder_exact"] for case in cases)
    write_json(output / "summary.json", result)
    write_json(output / "status.json", {"complete": True, "passed": result["passed"]})
    return result
