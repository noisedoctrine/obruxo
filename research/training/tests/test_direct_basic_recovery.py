import copy
from types import SimpleNamespace

import pytest

from obruxo_data.hashing import canonical_sha256
from obruxo_training.codec import PilotCodec, PilotTarget
from obruxo_training.direct_basic_audit import audit_ledger
from obruxo_training.direct_basic_fit import checkpoint, reconstruct
from obruxo_training.direct_basic_store import allowed_trial_rejection, measurement_query
from obruxo_training.paired_native_audit import equal_tree


class SyntheticStore:
    """Exercise the real orchestration with recorded scalar costs, never native audio."""
    def __init__(self, rows=None, *, fail_at=None):
        self.codec, self.rows, self.used = PilotCodec(), copy.deepcopy(rows or {}), set()
        self.offline, self.fail_at, self.fresh = False, fail_at, 0

    def check_limits(self):
        pass

    def cost(self, case, candidate, tag=None):
        query = {"case": case, "candidate": candidate.to_dict(), "tag": tag}
        key = canonical_sha256(query)
        self.used.add(key)
        if key not in self.rows:
            if self.offline:
                raise ValueError("missing synthetic measurement")
            self.fresh += 1
            if self.fresh == self.fail_at:
                raise RuntimeError("simulated partial update timeout")
            value = 1 + sum((value - .2) ** 2 for value in candidate.continuous) + .1 * (2 - candidate.categorical[0])
            self.rows[key] = {"query": query, "loss": value}
        return self.rows[key]["loss"]

    def evaluate(self, case, candidate, _role):
        return self.cost(case, candidate)

    def comparisons(self, case, candidate, stage):
        value = self.cost(case, candidate, stage)
        return [{"view": view, "scores": {"spectral": value, "mrstft": value}} for view in (1, 0)]


def synthetic_plan():
    initial = PilotTarget((.5,) * 9, (2, 1)).to_dict()
    return {"cases": [{"initial": initial, "training_index": index, "parameter": [{"scores": {"spectral": 2.}}] * 2}
                      for index in (2, 8, 11)], "limits": {"optimizer_attempts_per_case": 33}}


def write_measurements(output, store):
    import json
    (output / "measurements.jsonl").write_text("".join(json.dumps(row) + "\n" for row in store.rows.values()), encoding="utf-8")


def test_real_checkpoint_orchestration_recovers_partial_update_without_changing_trajectory(tmp_path):
    plan, interrupted = synthetic_plan(), SyntheticStore(fail_at=40)
    with pytest.raises(RuntimeError, match="timeout"):
        reconstruct(plan, interrupted, output=tmp_path)
    write_measurements(tmp_path, interrupted)
    saved = checkpoint(tmp_path, plan)
    assert saved and 0 < len(saved["trace"]) < 32
    recovered = SyntheticStore(interrupted.rows)
    result = reconstruct(plan, recovered, saved, output=tmp_path)
    whole = SyntheticStore()
    other = tmp_path / "uninterrupted"
    other.mkdir()
    expected = reconstruct(plan, whole, output=other)
    assert result["trace"] == expected["trace"]
    assert equal_tree(result["states"], expected["states"])
    assert result["cases"] == expected["cases"]
    assert len(recovered.rows) == len(whole.rows)
    write_measurements(tmp_path, recovered)
    final_checkpoint = checkpoint(tmp_path, plan)
    offline = SyntheticStore(recovered.rows)
    offline.offline = True
    verified = reconstruct(plan, offline, final_checkpoint)
    assert verified["trace"] == result["trace"] and offline.fresh == 0
    assert equal_tree(verified["states"], result["states"])


def test_checkpoint_rejects_changed_committed_measurements_before_recovery(tmp_path):
    plan, store = synthetic_plan(), SyntheticStore(fail_at=40)
    with pytest.raises(RuntimeError):
        reconstruct(plan, store, output=tmp_path)
    first = next(iter(store.rows.values()))
    first["loss"] += .01
    write_measurements(tmp_path, store)
    with pytest.raises(ValueError, match="committed measurements changed"):
        checkpoint(tmp_path, plan)


def test_replay_rejects_checkpoint_with_a_missing_committed_update(tmp_path):
    plan, store = synthetic_plan(), SyntheticStore(fail_at=100)
    with pytest.raises(RuntimeError):
        reconstruct(plan, store, output=tmp_path)
    write_measurements(tmp_path, store)
    saved = checkpoint(tmp_path, plan)
    assert len(saved["trace"]) > 2
    saved["trace"].pop(1)
    with pytest.raises(ValueError, match="contiguous update prefix"):
        reconstruct(plan, SyntheticStore(store.rows), saved, output=tmp_path)


def test_canonical_measurement_identity_deduplicates_inactive_values_but_keeps_repeats():
    codec = PilotCodec()
    a, b = PilotTarget((.5,) * 9, (0, 0)), PilotTarget((.5, .5, .9, .1, .8, .5, .5, .5, .5), (0, 0))
    assert measurement_query(codec, 0, 1, a) == measurement_query(codec, 0, 1, b)
    assert measurement_query(codec, 0, 1, a) != measurement_query(codec, 0, 1, a, "initial-repeat")


def test_only_predeclared_audio_trial_rejections_are_eligible():
    row = {"status": "rejected", "diagnostic_codes": ["audio.clipping"],
           "qa": {"finite": True, "actual_frames": 100, "expected_frames": 100, "actual_channels": 2, "expected_channels": 2}}
    assert allowed_trial_rejection(row)
    assert not allowed_trial_rejection({**row, "status": "error", "error_type": "RenderTimeout"})
    assert not allowed_trial_rejection({**row, "diagnostic_codes": ["audio.clipping", "vital.invalid_state"]})
    assert not allowed_trial_rejection({**row, "qa": {**row["qa"], "finite": False}})


def ledger_fixture(*, timeout=False):
    query = {"case": 0, "view": 1, "candidate": {}, "tag": None}
    request = {"phase": "training", "label": "direct:" + canonical_sha256(query), "request_id": "r", "source_sha256": "s", "controlled_sha256": "c"}
    events = []
    if timeout:
        events += [{"event": "attempt", "attempt": 0, **request}, {"event": "outcome", "attempt": 0, **request, "status": "error", "error_type": "RenderTimeout"}]
    ordinal = int(timeout)
    receipt = {"event": "outcome", "attempt": ordinal, **request, "status": "accepted"}
    events += [{"event": "attempt", "attempt": ordinal, **request}, receipt]
    rows = {canonical_sha256(query): {"query": query, "receipt": receipt}}
    plan = {"maximum_native_requests": 4608, "limits": {"training_attempts": 4584, "qualification_attempts": 432}}
    return SimpleNamespace(events=events, rows=rows, plan=plan)


def test_ledger_accounts_for_identical_timeout_recovery_and_rejects_missing_or_altered_results():
    fixture = ledger_fixture(timeout=True)
    result = audit_ledger(fixture.events, fixture.rows, fixture.plan)
    assert result["attempts"] == 2 and len(result["recovered_timeouts"]) == 1
    with pytest.raises(ValueError, match="unobserved"):
        audit_ledger(fixture.events[:-1], fixture.rows, fixture.plan)
    changed = copy.deepcopy(fixture.rows)
    next(iter(changed.values()))["receipt"]["request_id"] = "different"
    with pytest.raises(ValueError, match="foreign receipt"):
        audit_ledger(fixture.events, changed, fixture.plan)


def test_ledger_rejects_unaccounted_native_audio():
    fixture = ledger_fixture()
    with pytest.raises(ValueError, match="lacks a measurement"):
        audit_ledger(fixture.events, {}, fixture.plan)
