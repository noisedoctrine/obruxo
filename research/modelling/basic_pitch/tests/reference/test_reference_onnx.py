from __future__ import annotations

from pathlib import Path

from obruxo_basic_pitch.parity import assert_parity, compare_windows, synthetic_windows

ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = ROOT / "outputs" / "nmp.onnx"
CHECKPOINT_PATH = ROOT / "artifacts" / "basic_pitch_icassp_2022.pt"


def test_reference_onnx() -> None:
    if not ONNX_PATH.exists():
        raise FileNotFoundError(
            f"pinned public ONNX artifact is required at {ONNX_PATH}"
        )
    summary = compare_windows(ONNX_PATH, CHECKPOINT_PATH, synthetic_windows())
    assert_parity(summary)
