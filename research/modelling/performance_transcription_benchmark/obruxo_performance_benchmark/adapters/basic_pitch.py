"""Accessors for the landed #23/#24/#25 Basic Pitch seams."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import (
    ArtifactError,
    ArtifactUnavailable,
    ModelSpec,
    verify_checkpoint,
)
from ..types import NormalizedNote, TranscriptionOutput, rasterize_notes


def _basic_pitch_root() -> Path:
    root = Path(__file__).resolve().parents[3] / "basic_pitch"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def fixed_basic_pitch_contract() -> dict[str, Any]:
    _basic_pitch_root()
    from obruxo_basic_pitch.evaluation.runner import backend_contract

    return backend_contract()


def read_landed_baseline(manifest_path: Path) -> dict[str, Any]:
    """Read #25's stored baseline without re-running or re-scoring it."""
    manifest = Path(manifest_path).resolve(strict=True)
    output = manifest.parent
    run_path = output / "run.json"
    aggregate_path = output / "aggregates.json"
    if not run_path.is_file() or not aggregate_path.is_file():
        return {"status": "unavailable", "failure_code": "baseline_results_unavailable"}
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactUnavailable("landed Basic Pitch baseline is unreadable") from exc
    return {
        "status": str(run.get("status", "unavailable")),
        "failure_code": run.get("failure_code"),
        "pair_count": int(run.get("pair_count", 0)),
        "successful_pair_count": int(run.get("successful_pair_count", 0)),
        "failed_pair_count": int(run.get("failed_pair_count", 0)),
        "aggregate": aggregate,
        "backend": run.get("backend"),
        "run_identity": run.get("run_identity"),
    }


class BasicPitchAdapter:
    """Consume #25 for the full-precision baseline and run its graph for quantization."""

    def __init__(self, spec: ModelSpec, source_root: Path | None, checkpoint: Path | None) -> None:
        self.spec = spec
        self.source_root = None if source_root is None else Path(source_root)
        self.checkpoint = None if checkpoint is None else Path(checkpoint)
        self.model: Any | None = None
        self.bound_model: Any | None = None

    def preflight(self) -> None:
        root = _basic_pitch_root() if self.source_root is None else self.source_root.resolve(strict=True)
        metadata_path = root / "artifacts" / "basic_pitch_icassp_2022.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source = metadata["source"]
        except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
            raise ArtifactUnavailable("landed Basic Pitch metadata is unavailable") from exc
        if source.get("revision") != self.spec.source_revision or source.get("repository") != self.spec.source_repository:
            raise ArtifactError("Basic Pitch landed metadata does not match models.yaml")
        if self.checkpoint is not None:
            verify_checkpoint(self.spec, self.checkpoint)

    @property
    def active_model(self) -> Any | None:
        return self.bound_model if self.bound_model is not None else self.model

    def load(self, device: str = "cpu") -> None:
        self.preflight()
        import torch

        if device != "cpu":
            raise ArtifactUnavailable("Basic Pitch adapter quality/cost loading is CPU-only in #26")
        root = _basic_pitch_root() if self.source_root is None else self.source_root.resolve(strict=True)
        checkpoint = self.checkpoint or root / "artifacts" / "basic_pitch_icassp_2022.pt"
        from obruxo_basic_pitch.model import BasicPitchICASSP2022

        model = BasicPitchICASSP2022()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        self.model = model
        self.bound_model = model

    def bind_model(self, model: Any) -> None:
        import torch

        if not isinstance(model, torch.nn.Module):
            raise TypeError("Basic Pitch bound model must be a torch module")
        devices = {tensor.device.type for tensor in (*model.parameters(), *model.buffers())}
        if devices and devices != {"cpu"}:
            raise ValueError("Basic Pitch quantized model must remain on CPU")
        model.eval()
        self.bound_model = model

    def quantization_result(self) -> Any:
        if self.model is None:
            self.load()

        from ..quantization import quantize_dynamic_linear_int8

        return quantize_dynamic_linear_int8(self.model)

    def transcribe(self, audio: Path) -> TranscriptionOutput:
        if self.active_model is None:
            self.load()
        import torch
        from obruxo_basic_pitch.inference import prepare_wav, unwrap_window_outputs
        from obruxo_basic_pitch.postprocess import posteriorgrams_to_note_events

        prepared = prepare_wav(Path(audio))
        tensors = torch.from_numpy(prepared.windows)
        with torch.inference_mode():
            raw = self.active_model(tensors)
        posterior = unwrap_window_outputs(
            {name: value.detach().cpu().numpy() for name, value in raw.items()},
            original_sample_count=prepared.original_sample_count,
        )
        events = posteriorgrams_to_note_events(posterior)
        notes = tuple(
            NormalizedNote(
                event.start_time_s,
                event.end_time_s,
                event.pitch_midi,
                int(np.clip(np.rint(event.amplitude * 127.0), 0, 127)),
                None,
            )
            for event in events
            if event.end_time_s > event.start_time_s
        )
        from obruxo_basic_pitch.constants import ANNOTATIONS_FPS, AUDIO_SAMPLE_RATE

        frame_count = int(prepared.original_sample_count * ANNOTATIONS_FPS // AUDIO_SAMPLE_RATE)
        return TranscriptionOutput(notes, rasterize_notes(notes, frame_count))
