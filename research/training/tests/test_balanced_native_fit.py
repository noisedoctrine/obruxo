import copy
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from obruxo_data.render.qa import audio_float32_sha256
from obruxo_training.balanced_native_fit import compose_plan
from obruxo_training.features import PITCH_SIZE, TIMBRE_SHAPE
from obruxo_training.mixed_input_performances import MixedSampler
from obruxo_training.native_fit import FitRecords, audit_summary
from obruxo_training.native_fit_report import development_comparison
from obruxo_training.performances import initial_bank
from obruxo_training.pilot import PilotPool


@pytest.fixture(scope="module")
def plan():
    torch.set_num_threads(2)
    pool = PilotPool.create(512, seed=0)
    sampler = MixedSampler(pool)
    features = {"timbre": torch.zeros(*TIMBRE_SHAPE), "pitch": torch.zeros(PITCH_SIZE)}
    replay = []
    for index in range(sampler.epoch_size):
        recipe = sampler.recipe_at(index)
        entry = next(entry for entry in pool.entries if entry.state_id == recipe.state_id)
        replay.append({"state_id": recipe.state_id, "sequence_id": index, "recipe_id": recipe.recipe_id,
                       "target": entry.target.to_dict(), "audio_sha256": "0" * 64, "features": features})
    base = {"optimizer": {"learning_rate": .0001}, "limits": {"process_tree_bytes": 8 * 1024**3}, "gates": {}}
    return compose_plan(pool, base, {"replay": replay}, {"fixture": True})


def test_frozen_schedule_uses_every_recording_and_counts_all_queries(plan):
    assert len(plan["training"]) == 64 and len(plan["updates"]) == 256 and len(plan["validation"]) == 36
    assert plan["planned_maximum_requests"] == {"training_targets": 192, "gradient_probes": 5632, "fit_audit": 768,
                                               "validation_audit": 504, "total": 7096}
    assert sum(plan["limits"][name] for name in ("training_attempts", "qualification_attempts")) == 7120
    assert plan["storage_bound"]["estimated_archive_upper_bound_bytes"] < plan["limits"]["archive_bytes"]
    for index in range(64):
        updates = [row for row in plan["updates"] if row["state"] == index]
        assert Counter((row["input_view"], row["view"]) for row in updates) == Counter({(0, 1): 1, (1, 0): 1, (0, 2): 1, (2, 0): 1})
    for index, recipe in enumerate(plan["validation"][:12]):
        assert recipe["performances"] == [initial_bank()[offset + index % 4].performance().to_dict() for offset in (0, 4)]
    assert len(set(plan["validation_panels"]["counterbalanced"])) == 24


def quality_cases(plan):
    result = []
    for recipe in plan["validation"]:
        for view in (0, 1):
            value = 1.05 if "case_id" not in recipe else .1
            rows = {name: {"scores": {"spectral": score, "mrstft": score, "tail_error": .01}}
                    for name, score in (("initial", 1.), ("parameter", 1.), ("spectral", value), ("template", 2.), ("retrieval", 2.))}
            result.append({"state_id": recipe["state_id"], **({"case_id": recipe["case_id"]} if "case_id" in recipe else {}),
                           "view": view, "repeat_exact": True, "rows": rows})
    return result


def test_wider_improvements_cannot_hide_original_development_failure(plan):
    cases = quality_cases(plan)
    result = audit_summary(cases, plan, validation=True)
    assert all(row["pass"] for row in result["groups"].values())
    assert result["panels"]["counterbalanced"]["pass"] and not result["panels"]["original"]["pass"] and not result["pass"]
    assert not audit_summary(cases[:-1], plan, validation=True)["complete"]
    changed = copy.deepcopy(plan)
    changed["validation_panels"]["original"].append(changed["validation_panels"]["counterbalanced"][0])
    with pytest.raises(ValueError, match="panel membership"):
        audit_summary(cases, changed, validation=True)


def test_repeated_development_recordings_are_clustered_by_their_actual_source_states(plan):
    report = development_comparison(quality_cases(plan), plan["validation_panels"])
    assert report["panels_share_source_states"]
    assert all(row["independent_source_states"] == 12 for row in report["panels"].values())
    assert report["panels"]["original"]["groups"]["reference"]["initial"]["mean_delta"] > 0
    assert report["panels"]["counterbalanced"]["groups"]["reference"]["initial"]["mean_delta"] < 0


def test_training_record_preserves_and_reloads_each_actual_input_view(tmp_path):
    from obruxo_training.codec import PilotCodec
    codec = PilotCodec()
    calls = []

    def render(_candidate, _performance, _phase, label):
        calls.append(label)
        audio = np.full((8192, 2), .001 * (1 + int(label.rsplit(":", 1)[1])), dtype=np.float32)
        return audio, {"qa": {"audio_float32_sha256": audio_float32_sha256(audio)}}

    class Frontend:
        def __call__(self, audio):
            return {"timbre": torch.full(TIMBRE_SHAPE, float(audio.mean())), "pitch": torch.full((PITCH_SIZE,), float(audio.mean()))}

        def require_frozen(self):
            pass

    recipe = {"state_id": "fixture", "target": codec.default_target().to_dict(), "input_views": "all",
              "performances": [initial_bank()[view].performance().to_dict() for view in range(3)]}
    current = {"training": [recipe], "training_view_roles": ["original", "held", "alternate"],
               "limits": {"archive_bytes": 512 * 1024**2, "replay_bytes": 64 * 1024**2, "replay_records": 4}}
    session = SimpleNamespace(render=render, check_limits=lambda: None)
    record = FitRecords(tmp_path, current, Frontend(), session)[0]
    assert len(record["features"]) == len(record["targets"]) == 3 and len(set(record["audio_sha256"])) == 3
    assert [float(row["pitch"][0]) for row in record["features"]] == pytest.approx([.001, .002, .003])
    reloaded = FitRecords(tmp_path, current, Frontend(), session)[0]
    assert len(calls) == 3 and all(torch.equal(a["pitch"], b["pitch"]) for a, b in zip(record["features"], reloaded["features"], strict=True))
