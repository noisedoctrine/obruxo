import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).parents[1] / "reports/scripts/verify_basic_bootstrap.py"
spec = importlib.util.spec_from_file_location("bootstrap_audit", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_parallel_audit_waits_for_completed_status_and_never_uses_a_partial_summary(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"complete": True, "plan_id": "fixture"}))
    with pytest.raises(RuntimeError, match="has not completed"):
        audit.wait_for_native(tmp_path, 0)
    (tmp_path / "status.json").write_text(json.dumps({"complete": True}))
    assert audit.wait_for_native(tmp_path, 0) == {"complete": True, "plan_id": "fixture"}
    (tmp_path / "summary.json").write_text('{"complete":')
    with pytest.raises(RuntimeError, match="has not completed"):
        audit.wait_for_native(tmp_path, 0)


def test_parallel_audit_rejects_unbounded_waits(tmp_path):
    for seconds in (-1, 481, True):
        with pytest.raises(ValueError, match="integer"):
            audit.wait_for_native(tmp_path, seconds)
