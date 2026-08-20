from __future__ import annotations

import json
from pathlib import Path

from obruxo_performance_benchmark.artifacts import ModelSpec
from obruxo_performance_benchmark.evaluate import (
    _load_cached_pair,
    _pair_resume_identity,
)


class _Adapter:
    pass


class _ConfiguredAdapter:
    def __init__(self, segment_batch_size: int) -> None:
        self.segment_batch_size = segment_batch_size

    @property
    def inference_configuration(self) -> dict[str, int]:
        return {"segment_batch_size": self.segment_batch_size}


def _spec() -> ModelSpec:
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
        checkpoint_sha256="0" * 64,
        checkpoint_size_bytes=1,
        code_license="MIT",
        weight_license="MIT",
        benchmark_dtype="float32",
        native_sample_rate=16000,
        environment="environment.yml",
        stock_inference={"decoder": "stock"},
    )


def test_pair_cache_requires_exact_runtime_and_contract_identity(
    tmp_path: Path,
) -> None:
    spec = _spec()
    runtime = {"python": "3.12.13", "numpy": "2.4.6"}
    identity, details = _pair_resume_identity(
        spec,
        "full_precision",
        _Adapter(),
        "manifest",
        runtime,
        "pair-1",
        {"evaluate.py": "a"},
    )
    changed_identity, _ = _pair_resume_identity(
        spec,
        "full_precision",
        _Adapter(),
        "manifest",
        {"python": "3.12.14", "numpy": "2.4.6"},
        "pair-1",
        {"evaluate.py": "a"},
    )
    xpu_identity, _ = _pair_resume_identity(
        spec,
        "full_precision",
        _Adapter(),
        "manifest",
        runtime,
        "pair-1",
        {"evaluate.py": "a"},
        "xpu",
    )
    assert identity != changed_identity
    assert identity != xpu_identity
    assert details["backend_contract"] == {
        "route": "pytorch_cpu",
        "device": "cpu",
        "precision": "float32",
        "boundary": "full_clip_native_transcription",
    }
    path = tmp_path / "pair-1.json"
    path.write_text(
        json.dumps(
            {
                "pair_id": "pair-1",
                "resume_identity": details,
                "resume_identity_digest": identity,
                "status": "ok",
            }
        ),
        encoding="utf-8",
    )
    assert _load_cached_pair(path, identity, "pair-1", details) is not None
    assert _load_cached_pair(path, changed_identity, "pair-1", details) is None


def test_pair_cache_separates_internal_segment_batch_variants() -> None:
    spec = _spec()
    runtime = {"python": "3.12.13", "numpy": "2.4.6"}
    bsz8, details8 = _pair_resume_identity(
        spec,
        "full_precision",
        _ConfiguredAdapter(8),
        "manifest",
        runtime,
        "pair-1",
    )
    bsz32, details32 = _pair_resume_identity(
        spec,
        "full_precision",
        _ConfiguredAdapter(32),
        "manifest",
        runtime,
        "pair-1",
    )
    assert bsz8 != bsz32
    assert details8["adapter_configuration"] == {"segment_batch_size": 8}
    assert details32["adapter_configuration"] == {"segment_batch_size": 32}


def test_pair_cache_can_reuse_legacy_yourmt3_batch_rows_explicitly(
    tmp_path: Path,
) -> None:
    spec = _spec()
    runtime = {"python": "3.12.13", "numpy": "2.4.6"}
    legacy_identity, legacy_details = _pair_resume_identity(
        spec,
        "full_precision",
        _ConfiguredAdapter(8),
        "manifest",
        runtime,
        "pair-1",
    )
    current_identity, current_details = _pair_resume_identity(
        spec,
        "full_precision",
        _ConfiguredAdapter(64),
        "manifest",
        runtime,
        "pair-1",
    )
    path = tmp_path / "pair-1.json"
    path.write_text(
        json.dumps(
            {
                "pair_id": "pair-1",
                "resume_identity": legacy_details,
                "resume_identity_digest": legacy_identity,
                "status": "ok",
            }
        ),
        encoding="utf-8",
    )
    assert (
        _load_cached_pair(
            path,
            current_identity,
            "pair-1",
            current_details,
            allow_segment_batch_variant=True,
        )
        is not None
    )
    assert _load_cached_pair(path, current_identity, "pair-1", current_details) is None


def test_pair_cache_retries_prior_failure(tmp_path: Path) -> None:
    spec = _spec()
    runtime = {"python": "3.12.13", "numpy": "2.4.6"}
    identity, details = _pair_resume_identity(
        spec,
        "full_precision",
        _Adapter(),
        "manifest",
        runtime,
        "pair-1",
    )
    path = tmp_path / "pair-1.json"
    path.write_text(
        json.dumps(
            {
                "pair_id": "pair-1",
                "resume_identity": details,
                "resume_identity_digest": identity,
                "status": "runtime_failed",
                "failure_code": "transcription_runtime_error",
            }
        ),
        encoding="utf-8",
    )
    assert _load_cached_pair(path, identity, "pair-1", details) is None


def test_pair_cache_keeps_success_after_allowed_code_change(tmp_path: Path) -> None:
    spec = _spec()
    runtime = {"python": "3.12.13", "numpy": "2.4.6"}
    old_identity, old_details = _pair_resume_identity(
        spec,
        "full_precision",
        _ConfiguredAdapter(32),
        "manifest",
        runtime,
        "pair-1",
        {"evaluate.py": "old"},
    )
    new_identity, new_details = _pair_resume_identity(
        spec,
        "full_precision",
        _ConfiguredAdapter(32),
        "manifest",
        runtime,
        "pair-1",
        {"evaluate.py": "new"},
    )
    path = tmp_path / "pair-1.json"
    path.write_text(
        json.dumps(
            {
                "pair_id": "pair-1",
                "resume_identity": old_details,
                "resume_identity_digest": old_identity,
                "status": "ok",
            }
        ),
        encoding="utf-8",
    )
    assert (
        _load_cached_pair(
            path,
            new_identity,
            "pair-1",
            new_details,
            allow_segment_batch_variant=True,
        )
        is not None
    )
