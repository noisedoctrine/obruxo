"""Lossless YourMT3+ event normalization and fixed stock inference settings."""

from __future__ import annotations

import inspect
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..artifacts import (
    ArtifactUnavailable,
    ModelSpec,
    verify_checkout,
    verify_checkpoint,
)
from ..types import NormalizedNote, TranscriptionOutput, rasterize_notes

YOURMT3_IDS = ("ymt3_plus", "yptf_multi", "yptf_moe_multi")


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def stock_inference_config(model_id: str, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if model_id not in YOURMT3_IDS:
        raise ValueError(f"not a YourMT3+ model: {model_id}")
    config = {
        "project": "2024",
        "precision": "float32",
        "deterministic": True,
        "decode": "stock_official_release",
        "checkpoint_selection": "locked_before_corpus_results",
    }
    if overrides:
        changed = {key: value for key, value in overrides.items() if config.get(key) != value}
        if changed:
            raise ValueError("YourMT3+ stock inference settings cannot be overridden")
    return config


def normalize_note_events(events: Any) -> tuple[NormalizedNote, ...]:
    """Normalize upstream event objects/dicts without adding an alternate decoder."""
    if events is None:
        return ()
    rows = events if isinstance(events, (list, tuple)) else list(events)
    starts: dict[int, dict[str, Any]] = {}
    result: list[NormalizedNote] = []
    for event in rows:
        kind = str(_field(event, "event_type", _field(event, "type", event.__class__.__name__))).casefold()
        if "progress" in kind or "tempo" in kind:
            continue
        onset_value = _field(event, "onset_seconds", _field(event, "onset", _field(event, "start")))
        offset_value = _field(event, "offset_seconds", _field(event, "offset", _field(event, "end")))
        pitch_value = _field(event, "midi_pitch", _field(event, "pitch"))
        if onset_value is not None and offset_value is not None and pitch_value is not None:
            onset = onset_value
            offset = offset_value
            pitch = _field(event, "midi_pitch", _field(event, "pitch"))
            result.append(
                NormalizedNote(
                    float(onset),
                    float(offset),
                    int(pitch),
                    None if _field(event, "velocity") is None else int(_field(event, "velocity")),
                    None if _field(event, "confidence") is None else float(_field(event, "confidence")),
                    _field(event, "instrument_or_program", _field(event, "instrument")),
                )
            )
            continue
        if "start" in kind or _field(event, "start_time") is not None:
            index = _field(event, "index", len(starts))
            starts[int(index)] = {
                "onset": _field(event, "start_time", _field(event, "onset")),
                "pitch": _field(event, "pitch", _field(event, "midi_pitch")),
                "instrument": _field(event, "instrument", _field(event, "program")),
                "velocity": _field(event, "velocity"),
            }
            continue
        if "end" in kind or _field(event, "end_time") is not None:
            start_event = _field(event, "start_event")
            index = _field(event, "start_event_index", _field(start_event, "index") if start_event is not None else None)
            if index is None or int(index) not in starts:
                raise ValueError("YourMT3+ end event does not reference a start event")
            start = starts[int(index)]
            result.append(
                NormalizedNote(
                    float(start["onset"]),
                    float(_field(event, "end_time")),
                    int(start["pitch"]),
                    None if start["velocity"] is None else int(start["velocity"]),
                    None,
                    start["instrument"],
                )
            )
    return tuple(result)


def write_temporary_midi(data: bytes, output_dir: Path, name: str = "upstream.mid") -> Path:
    root = Path(output_dir).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    destination = (root / name).resolve(strict=False)
    if destination.parent != root or destination.suffix.casefold() != ".mid":
        raise ValueError("temporary MIDI must remain directly under the provided output directory")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=".tmp", dir=root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


class YourMT3Adapter:
    def __init__(self, spec: ModelSpec, source_root: Path | None, checkpoint: Path | None) -> None:
        self.spec = spec
        self.source_root = None if source_root is None else Path(source_root)
        self.checkpoint = None if checkpoint is None else Path(checkpoint)
        self.inference = stock_inference_config(spec.model_id)
        self.model: Any | None = None
        self.bound_model: Any | None = None
        self._helper: Any | None = None

    def preflight(self) -> None:
        if self.source_root is None or self.checkpoint is None:
            raise ArtifactUnavailable("dependency_unavailable")
        verify_checkout(self.spec, self.source_root)
        verify_checkpoint(self.spec, self.checkpoint)

    @property
    def active_model(self) -> Any | None:
        return self.bound_model if self.bound_model is not None else self.model

    def _official_args(self) -> list[str]:
        checkpoint_path = Path(self.spec.checkpoint_path)
        checkpoint_name = checkpoint_path.name
        experiment = checkpoint_path.parent.parent.name
        args = [f"{experiment}@{checkpoint_name}", "-p", "2024", "-pr", "32", "-w", "false"]
        if self.spec.model_id == "yptf_multi":
            args += ["-tk", "mc13_full_plus_256", "-dec", "multi-t5", "-nl", "26", "-enc", "perceiver-tf", "-ac", "spec", "-hop", "300", "-atc", "1"]
        elif self.spec.model_id == "yptf_moe_multi":
            args += ["-tk", "mc13_full_plus_256", "-dec", "multi-t5", "-nl", "26", "-enc", "perceiver-tf", "-sqr", "1", "-ff", "moe", "-wf", "4", "-nmoe", "8", "-kmoe", "2", "-act", "silu", "-epe", "rope", "-rp", "1", "-ac", "spec", "-hop", "300", "-atc", "1"]
        return args

    def load(self, device: str = "cpu") -> None:
        self.preflight()
        if device != "cpu":
            raise ArtifactUnavailable("official YourMT3+ loader does not expose an approved XPU path")
        try:
            import torch

            root = self.source_root.resolve(strict=True)
            amt_src = root / "amt" / "src"
            if not amt_src.is_dir():
                raise ArtifactUnavailable("official YourMT3+ amt/src checkout is missing")
            for path in (root, amt_src):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            import model_helper
        except (ImportError, OSError) as exc:
            raise ArtifactUnavailable("dependency_unavailable") from exc
        if not Path(self.checkpoint).resolve().is_relative_to(root):
            raise ArtifactUnavailable("checkpoint must be inside the pinned official checkout for stock loading")
        args = self._official_args()
        loader = model_helper.load_model_checkpoint
        try:
            if "device" in inspect.signature(loader).parameters:
                model = loader(args=args, device=torch.device("cpu"))
            else:
                model = loader(args=args)
        except (ImportError, ModuleNotFoundError) as exc:
            raise ArtifactUnavailable("dependency_unavailable") from exc
        self.model = model
        self.bound_model = model
        self._helper = model_helper

    def bind_model(self, model: Any) -> None:
        self.bound_model = model
        if hasattr(model, "eval"):
            model.eval()

    @staticmethod
    def _frame_count(audio: Path) -> int:
        import sys

        root = Path(__file__).resolve().parents[2] / "basic_pitch"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from obruxo_basic_pitch.inference import prepare_wav

        return int(prepare_wav(Path(audio)).original_sample_count * 100 // 22050)

    def transcribe(self, audio: Path) -> TranscriptionOutput:
        if self.active_model is None or self._helper is None:
            self.load()
        import torch
        import torchaudio

        model = self.active_model
        helper = self._helper
        waveform, sample_rate = torchaudio.load(uri=str(Path(audio).resolve(strict=True)))
        waveform = torch.mean(waveform, dim=0).unsqueeze(0)
        target_rate = int(model.audio_cfg["sample_rate"])
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
        segments = helper.slice_padded_array(waveform, model.audio_cfg["input_frames"], model.audio_cfg["input_frames"])
        device = next(model.parameters()).device
        segments = torch.from_numpy(segments.astype("float32")).to(device).unsqueeze(1)
        with torch.inference_mode():
            predicted, _ = model.inference_file(bsz=8, audio_segments=segments)
        channels = int(model.task_manager.num_decoding_channels)
        starts = [model.audio_cfg["input_frames"] * i / target_rate for i in range(segments.shape[0])]
        notes_by_channel = []
        for channel in range(channels):
            channel_tokens = [array[:, channel, :] for array in predicted]
            zipped, _, _ = model.task_manager.detokenize_list_batches(channel_tokens, starts, return_events=True)
            notes, _ = helper.merge_zipped_note_events_and_ties_to_notes(zipped)
            notes_by_channel.append(notes)
        notes = normalize_note_events(helper.mix_notes(notes_by_channel))
        return TranscriptionOutput(notes, rasterize_notes(notes, self._frame_count(Path(audio))))
