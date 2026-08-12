from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from obruxo_performance_benchmark.artifacts import (
    ArtifactError,
    ModelSpec,
    load_model_specs,
    verify_checkpoint,
)


def _spec(checkpoint_hash: str, size: int) -> ModelSpec:
    return ModelSpec(
        model_id="toy",
        family="toy",
        publication_year=2024,
        output_contract="note_events",
        source_repository="example/toy",
        source_revision="0123456789abcdef",
        checkpoint_repository="example/toy",
        checkpoint_revision="0123456789abcdef",
        checkpoint_path="model.pt",
        checkpoint_sha256=checkpoint_hash,
        checkpoint_size_bytes=size,
        code_license="MIT",
        weight_license="MIT",
        benchmark_dtype="float32",
        native_sample_rate=16000,
        environment="environment.yml",
        stock_inference={},
    )


def test_release_config_has_exact_candidate_set() -> None:
    path = Path(__file__).resolve().parents[2] / "config" / "models.yaml"
    specs = load_model_specs(path)
    assert list(specs) == [
        "basic_pitch",
        "timbre_trap_base",
        "ymt3_plus",
        "yptf_multi",
        "yptf_moe_multi",
        "muscriptor_small",
        "muscriptor_medium",
        "muscriptor_large",
    ]
    assert all(spec.is_fully_locked for spec in specs.values() if not spec.model_id.startswith("muscriptor_"))
    for model_id in ("muscriptor_small", "muscriptor_medium", "muscriptor_large"):
        spec = specs[model_id]
        assert spec.checkpoint_sha256 is None
        assert spec.checkpoint_size_bytes > 0
        assert spec.checkpoint_identity_status == "gated_digest_not_exposed_without_access"
        assert not spec.is_fully_locked


def test_checkpoint_verification_is_read_only(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    content = b"synthetic checkpoint"
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    before = (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes())
    verify_checkpoint(_spec(digest, len(content)), path)
    after = (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes())
    assert after == before
    with pytest.raises(ArtifactError, match="checkpoint_hash_mismatch"):
        verify_checkpoint(_spec("0" * 64, len(content)), path)
