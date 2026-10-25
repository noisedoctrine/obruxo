import copy
import json

import pytest

from obruxo_training import protected_native_check as native
from obruxo_training.audit import write_json
from obruxo_training.protected_update_check import inspect_proposals
from test_basic_direct_fit import FakeSession, completed, receipt  # noqa: F401


def test_native_decisions_require_each_sound_preservation_progress_and_exact_repeats():
    jobs = [{"label": f"obs{i}", "step": 4, "observation": i, "before_sound": 1., "accepted_initial_sound": .8, "proposal_nonzero": True} for i in range(4)]
    jobs += [{**jobs[i], "label": f"repeat{i}", "repeat_of": jobs[i]["label"]} for i in (0, 3)]
    plan = {"jobs": jobs, "steps": [4], "gates": {**native.GATES, "minimum_protected_points": 1}}
    rows = [{"label": job["label"], "scores": {"spectral": .9}, "receipt": {"qa": {"audio_float32_sha256": "same"}}} for job in jobs]
    result = native.decisions(plan, rows)
    assert result["native_acceptance_screen_passed"] and result["accepted_points"] == 1
    changed = copy.deepcopy(rows)
    changed[2]["scores"]["spectral"] = 1.001
    assert not native.decisions(plan, changed)["native_acceptance_screen_passed"]
    changed = copy.deepcopy(rows)
    changed[-1]["receipt"]["qa"]["audio_float32_sha256"] = "different"
    assert not native.decisions(plan, changed)["native_acceptance_screen_passed"]
    with pytest.raises(ValueError, match="coverage"):
        native.decisions(plan, rows[:-1])


def test_retained_native_check_rescores_all_audio_refuses_tampering_and_second_execution(completed, tmp_path, monkeypatch):  # noqa: F811
    class TimeoutOnce(FakeSession):
        failures = 1

        def render(self, candidate, performance, phase, label):
            count = sum(row["event"] == "attempt" for row in self.events)
            if count >= self.failures:
                return super().render(candidate, performance, phase, label)
            _, result = receipt(self.codec, self.plan, candidate, performance, phase, label, count)
            identity = {key: result[key] for key in ("phase", "label", "attempt", "request_id", "source_sha256", "controlled_sha256")}
            self._append({"event": "attempt", **identity})
            result.pop("qa")
            result.update(status="error", error_type="RenderTimeout")
            self._append(result)
            return None, result

    inputs, source = completed
    screen = inspect_proposals(inputs, source)
    jobs = native.build_cases(inputs, source, screen, [4])
    jobs += [{**jobs[i], "label": label, "repeat_of": jobs[i]["label"]} for i, label in ((0, "repeat_first:after"), (-1, "repeat_last:after"))]
    plan = {"jobs": jobs, "steps": [4], "limits": native.LIMITS, "gates": native.GATES,
            **{key: inputs.plan[key] for key in ("render_profile", "render_phase_controls", "expected_renderer_identity")}}
    output = tmp_path / "native"
    output.mkdir()
    write_json(output / "plan.json", plan)
    monkeypatch.setattr(native, "PairedSession", TimeoutOnce)
    result = native.execute(plan, source, output)
    assert result["complete"] and result["native_attempts"] == 9 and result["optimizer_applications"] == 0
    proof = native.verify(plan, source, output)
    assert proof["verified"] and proof["comparison_waveforms_rescored"] == 8
    assert len(proof["recovered_timeouts"]) == 1 and proof["native_attempts"] == 9
    assert proof["result"]["repeat_exact"] == [True, True]
    assert not proof["result"]["native_acceptance_screen_passed"]  # One simulated point cannot meet the real nine-point gate.
    with pytest.raises(ValueError, match="second execution"):
        native.execute(plan, source, output)
    class TimeoutTwice(TimeoutOnce):
        failures = 2

    failed = tmp_path / "failed"
    failed.mkdir()
    write_json(failed / "plan.json", plan)
    monkeypatch.setattr(native, "PairedSession", TimeoutTwice)
    with pytest.raises(RuntimeError, match="incomplete"):
        native.execute(plan, source, failed)
    stopped = json.loads((failed / "summary.json").read_text(encoding="utf-8"))
    assert not stopped["complete"] and stopped["failure"] and stopped["native_attempts"] == 2
    path = output / "measurements.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["scores"]["mrstft"] += 1.
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="reproduced"):
        native.verify(plan, source, output)
