import copy

import numpy as np
import pytest
import torch

from obruxo_training.codec import PilotCodec
from obruxo_training.native_gradient_screen import make_plan, qualification_summary, qualify_case
from obruxo_training.performances import initial_bank
from obruxo_training.pilot import PilotPool


@pytest.fixture(scope="module")
def plan():
    return make_plan(PilotPool.create(256, seed=0))


def test_plan_reserves_test_and_balances_imperfect_categories(plan):
    assert len(plan["cases"]) == 24 and not plan["test_split_used"]
    assert len({row["state_id"] for row in plan["cases"]}) == 12
    assert plan["limits"]["qualification_attempts"] == 24 * (22 + 4)
    changed = [sum(a != b for a, b in zip(row["source"]["categorical"], row["start"]["categorical"], strict=True))
               for row in plan["cases"]]
    assert changed.count(0) == 8 and changed.count(1) == 16
    assert len({str(row["performance"]) for row in plan["cases"]}) == 8


def test_gates_count_missing_duplicates_ties_and_both_metrics(plan):
    scores = {"start": {"mrstft": 2., "spectral": 2.}, "continuous": {"mrstft": 1., "spectral": 1.},
              "joint": {"mrstft": 1., "spectral": 1.}}
    cases = [{**row, "scores": copy.deepcopy(scores), "repeat_exact": True} for row in plan["cases"]]
    assert qualification_summary(cases, plan)["direction_gates_passed"]
    assert not qualification_summary(cases[:-1], plan)["direction_gates_passed"]
    assert not qualification_summary(cases[:-1] + [cases[0]], plan)["direction_gates_passed"]
    for row in cases:
        row["scores"]["joint"]["spectral"] = 3.
    assert not qualification_summary(cases, plan)["direction_gates_passed"]
    for row in cases:
        row["scores"]["joint"] = copy.deepcopy(scores["start"])
    assert not qualification_summary(cases, plan)["direction_gates_passed"]


def test_real_measurement_smoke_and_rejected_render():
    torch.set_num_threads(2)
    codec = PilotCodec()
    target = codec.default_target()
    case = {"index": 0, "role": "reference", "state_id": "fixture", "source": target.to_dict(),
            "start": target.to_dict(), "performance": initial_bank()[0].performance().to_dict()}

    class Session:
        def __init__(self):
            self.codec, self.calls, self.reject = codec, 0, False

        def render(self, candidate, performance, phase, label):
            self.calls += 1
            if self.reject:
                return None, {"status": "rejected"}
            wave = (np.sin(np.arange(16384) / 30) * .001 * (1 + sum(candidate.continuous))).astype(np.float32)
            return np.stack((wave, wave), axis=1), {"request_id": label, "qa": {"audio_float32_sha256": "fixture"}}

    session = Session()
    result = qualify_case(case, session)
    assert session.calls == result["estimate"]["queries"] + 4 <= 26
    assert result["repeat_exact"] and set(result["scores"]) == {"start", "continuous", "joint"}
    session.reject = True
    with pytest.raises(RuntimeError, match="rejected"):
        qualify_case(case, session)
