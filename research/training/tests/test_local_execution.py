import pytest

from obruxo_training import local_execution as execution


def test_active_request_is_system_only_and_restores_previous_state():
    calls = []
    def api(flags):
        calls.append(flags)
        return execution.CONTINUOUS
    with execution.active_computation(state_api=api) as report:
        assert report["requested"] and not report["released"]
    assert calls == [execution.CONTINUOUS | execution.SYSTEM_REQUIRED, execution.CONTINUOUS]
    assert report["released"] and not report["display_required"] and not report["persistent_settings_changed"]


def test_request_is_released_after_work_raises():
    calls = []
    def api(flags):
        calls.append(flags)
        return execution.CONTINUOUS
    with pytest.raises(RuntimeError, match="work failed"):
        with execution.active_computation(state_api=api):
            raise RuntimeError("work failed")
    assert calls[-1] == execution.CONTINUOUS


def test_failed_request_does_not_enter_computation():
    with pytest.raises(OSError, match="refused"):
        with execution.active_computation(state_api=lambda _flags: 0):
            pytest.fail("work entered despite a failed execution request")


def test_nested_request_restores_outer_hold():
    calls, state = [], execution.CONTINUOUS
    def api(flags):
        nonlocal state
        calls.append(flags)
        previous, state = state, flags
        return previous
    with execution.active_computation(state_api=api):
        with execution.active_computation(state_api=api):
            pass
        assert state == execution.CONTINUOUS | execution.SYSTEM_REQUIRED
    assert state == execution.CONTINUOUS
    assert len(calls) == 4


def test_other_platforms_need_no_power_api(monkeypatch):
    monkeypatch.setattr(execution, "execution_state_api", lambda: None)
    with execution.active_computation() as report:
        assert not report["requested"]
