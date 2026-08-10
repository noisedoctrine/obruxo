from __future__ import annotations

from pathlib import Path

import pytest
import torch
from obruxo_basic_pitch.model import BasicPitchICASSP2022
from obruxo_basic_pitch.weights import (
    SOURCE_GIT_BLOB_SHA1,
    SOURCE_SIZE_BYTES,
    import_onnx_state_dict,
    write_imported_checkpoint,
)

ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = ROOT / "outputs" / "nmp.onnx"
CHECKPOINT_PATH = ROOT / "artifacts" / "basic_pitch_icassp_2022.pt"
METADATA_PATH = ROOT / "artifacts" / "basic_pitch_icassp_2022.json"


def test_pinned_source_import_is_complete() -> None:
    assert ONNX_PATH.stat().st_size == SOURCE_SIZE_BYTES
    state, metadata = import_onnx_state_dict(ONNX_PATH)
    assert metadata.source_git_blob_sha1 == SOURCE_GIT_BLOB_SHA1
    assert set(state) == set(BasicPitchICASSP2022().state_dict())
    assert all(value.dtype in (torch.float32, torch.int64) for value in state.values())


def test_two_imports_are_elementwise_identical() -> None:
    first, _ = import_onnx_state_dict(ONNX_PATH)
    second, _ = import_onnx_state_dict(ONNX_PATH)
    assert first.keys() == second.keys()
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_checkpoint_strict_load_and_overwrite_guard() -> None:
    state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    BasicPitchICASSP2022().load_state_dict(state, strict=True)
    with pytest.raises(FileExistsError):
        write_imported_checkpoint(ONNX_PATH, CHECKPOINT_PATH, METADATA_PATH)


def test_source_identity_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    from obruxo_basic_pitch import weights

    monkeypatch.setattr(weights, "SOURCE_SIZE_BYTES", SOURCE_SIZE_BYTES + 1)
    with pytest.raises(ValueError, match="size"):
        import_onnx_state_dict(ONNX_PATH)

    monkeypatch.setattr(weights, "SOURCE_SIZE_BYTES", SOURCE_SIZE_BYTES)
    monkeypatch.setattr(weights, "SOURCE_GIT_BLOB_SHA1", "0" * 40)
    with pytest.raises(ValueError, match="SHA-1"):
        import_onnx_state_dict(ONNX_PATH)
