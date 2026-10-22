import pytest

from obruxo_training.varied_bootstrap import run


def test_cached_comparison_refuses_missing_or_changed_cache_before_starting(tmp_path):
    plan = {"feature_cache": {"file_sha256": "frozen"}}
    with pytest.raises(ValueError, match="requires its frozen feature cache"):
        run(None, plan, tmp_path, tmp_path / "missing", execute=True)
    path = tmp_path / "changed.pt"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="feature cache changed"):
        run(None, plan, tmp_path, tmp_path / "changed", execute=True, feature_cache=path)
    assert not (tmp_path / "missing/run.json").exists() and not (tmp_path / "changed/run.json").exists()
