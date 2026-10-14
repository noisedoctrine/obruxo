from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from obruxo_data.hashing import canonical_sha256
from obruxo_data.render.qa import compare_audio

from .audit import write_json
from .codec import CATEGORICAL, CONTINUOUS
from .contracts import RenderRecipe
from .identity import PresetIdentity
from .performances import initial_bank
from .splits import SplitLedger
from .stream import NativeRendererFactory, OnlineRenderStream, StreamConfig


class CodecCheckSampler:
    def __init__(self, pool):
        self.pool = pool
        coverage = {}
        for entry in pool.entries:
            if pool.manifest.by_state[entry.state_id].split == "train":
                coverage.setdefault(entry.target.categorical, entry)
        if not coverage:
            raise ValueError("codec qualification requires authored training states")
        codec, ledger, performance = pool.codec, SplitLedger(pool.manifest), initial_bank()[0].performance()
        self.recipes, self.cases = [], []
        names = [entry[0] for entry in CONTINUOUS + CATEGORICAL]
        for categories, entry in sorted(coverage.items()):
            source = codec.decode(entry.target)
            decoded = codec.decode(codec.encode(source))
            decoded_id = PresetIdentity.from_bytes(decoded).state
            ledger.register_variant(decoded_id, entry.state_id)
            left, right = json.loads(source)["settings"], json.loads(decoded)["settings"]
            equal = all(np.float32(left[name]) == np.float32(right[name]) for name in names)
            self.cases.append({"state_id": entry.state_id, "categories": list(categories),
                               "decoded_state_id": decoded_id, "controls_equal_at_float32": equal})
            for raw in (source, source, decoded):
                self.recipes.append(RenderRecipe.create(raw, performance, schema=codec.schema, catalog=codec.catalog,
                                                       contract=codec.contract, ledger=ledger, purpose="evaluate",
                                                       sequence_id=len(self.recipes)))
        self.sampler_id = canonical_sha256({"check": "pilot_codec_native_v1", "pool_id": pool.pool_id, "cases": self.cases})

    def recipe_at(self, sequence_id):
        return self.recipes[sequence_id]


def check_codec_audio(pool, output: Path, *, renderer_factory=None, timeout: float = 60, progress=None) -> dict:
    sampler = CodecCheckSampler(pool)
    config = StreamConfig(prefetch_clips=1, timeout_seconds=timeout)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"complete": False})
    cases, group, accepted = [], [], 0
    with OnlineRenderStream(sampler, len(sampler.recipes), config=config, renderer_factory=renderer_factory or NativeRendererFactory()) as stream:
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
                del source, repeat, decoded
            cases.append(case)
            group.clear()
    expected_coverage = len(CATEGORICAL[0][1]) * len(CATEGORICAL[1][1])
    result = {"version": 1, "check": "pilot_codec_native_v1", "pool_id": pool.pool_id, "codec_id": pool.codec.codec_id,
              "cases": cases, "covered_category_combinations": len(cases), "expected_category_combinations": expected_coverage,
              "requested": len(sampler.recipes), "accepted": accepted,
              "all_controls_equal_at_float32": all(case["controls_equal_at_float32"] for case in cases),
              "decoded_within_reference_tolerance": sum(case.get("decoded_comparison", {}).get("within_tolerance", False) for case in cases),
              "source_repeats_within_reference_tolerance": sum(case.get("source_repeat", {}).get("within_tolerance", False) for case in cases),
              "renderer_factory": type(stream.factory).__name__, "audio_files_written": 0}
    result["passed"] = (accepted == len(sampler.recipes) and len(cases) == expected_coverage
                        and result["all_controls_equal_at_float32"] and result["decoded_within_reference_tolerance"] == len(cases))
    write_json(output / "summary.json", result)
    write_json(output / "status.json", {"complete": True, "passed": result["passed"]})
    return result
