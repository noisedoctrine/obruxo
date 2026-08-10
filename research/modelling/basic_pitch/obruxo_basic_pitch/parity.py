"""CPU float32 parity harness for the pinned Basic Pitch ONNX oracle."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from scipy import signal
from scipy.io import wavfile

from .constants import (
    AUDIO_N_SAMPLES,
    AUDIO_SAMPLE_RATE,
    FRAME_THRESHOLD,
    MODEL_ID,
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_CONTOUR,
    ONNX_OUTPUT_NOTE,
    ONNX_OUTPUT_ONSET,
    ONSET_THRESHOLD,
    SPOTIFY_ONNX_GIT_BLOB_SHA1,
)
from .model import BasicPitchICASSP2022
from .postprocess import NoteEvent, posteriorgrams_to_note_events


@dataclass(frozen=True)
class OutputParity:
    max_abs_error: float
    mean_abs_error: float
    rmse: float


@dataclass(frozen=True)
class ParitySummary:
    contour: OutputParity
    note: OutputParity
    onset: OutputParity
    note_threshold_disagreements: int
    onset_threshold_disagreements: int
    note_threshold_elements: int
    onset_threshold_elements: int
    onnx_event_count: int
    torch_event_count: int
    event_structure_disagreements: int
    amplitude_max_abs_error: float | None
    amplitude_mean_abs_error: float | None
    pitch_bend_element_disagreements: int
    synthetic_windows: int
    private_local_windows: int = 0


# These are the smallest fixed margins above the measured public synthetic
# suite maxima. The all-zero edge is retained because the ONNX Runtime CPU
# log kernel produces a one-ulp spread for the constant 1e-10 input.
ADOPTED_MAX_ABS_TOLERANCES = {
    "contour": 0.0205306,
    "note": 0.0038445,
    "onset": 0.2089347,
}


def synthetic_windows() -> np.ndarray:
    """Build the fixed public parity suite without storing audio artifacts."""
    samples = np.arange(AUDIO_N_SAMPLES, dtype=np.float32)
    seconds = samples / AUDIO_SAMPLE_RATE
    zeros = np.zeros_like(samples)
    impulses = np.zeros_like(samples)
    impulses[0] = 1.0
    impulses[AUDIO_N_SAMPLES // 2] = -0.75
    sinusoid = 0.5 * np.sin(2.0 * np.pi * 440.0 * seconds).astype(np.float32)
    chord = sum(
        0.2 * np.sin(2.0 * np.pi * frequency * seconds)
        for frequency in (220.0, 330.0, 440.0)
    ).astype(np.float32)
    noise = np.random.default_rng(20260810).standard_normal(AUDIO_N_SAMPLES).astype(np.float32) * 0.05
    return np.ascontiguousarray(np.stack((zeros, impulses, sinusoid, chord, noise), axis=0)[:, :, None])


def audio_to_windows(path: Path) -> np.ndarray:
    """Read one local WAV read-only and create the prescribed shared windows."""
    sample_rate, audio = wavfile.read(path)
    samples = np.asarray(audio)
    if samples.ndim == 2:
        samples = samples.astype(np.float32).mean(axis=1)
    else:
        samples = samples.astype(np.float32)
    if np.issubdtype(audio.dtype, np.integer):
        samples /= np.iinfo(audio.dtype).max
    if sample_rate != AUDIO_SAMPLE_RATE:
        gcd = np.gcd(sample_rate, AUDIO_SAMPLE_RATE)
        samples = signal.resample_poly(samples, AUDIO_SAMPLE_RATE // gcd, sample_rate // gcd).astype(np.float32)
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    padded = np.concatenate((np.zeros(3840, dtype=np.float32), samples))
    hop = AUDIO_N_SAMPLES - 30 * 256
    starts = list(range(0, max(1, padded.shape[0] - AUDIO_N_SAMPLES + 1), hop))
    if starts[-1] + AUDIO_N_SAMPLES < padded.shape[0]:
        starts.append(padded.shape[0] - AUDIO_N_SAMPLES)
    windows = []
    for start in starts:
        window = padded[start : start + AUDIO_N_SAMPLES]
        if window.shape[0] < AUDIO_N_SAMPLES:
            window = np.pad(window, (0, AUDIO_N_SAMPLES - window.shape[0]))
        windows.append(window)
    return np.ascontiguousarray(np.stack(windows, axis=0)[:, :, None], dtype=np.float32)


def _output_parity(reference: np.ndarray, candidate: np.ndarray) -> OutputParity:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    absolute = np.abs(difference)
    return OutputParity(
        max_abs_error=float(np.max(absolute)),
        mean_abs_error=float(np.mean(absolute)),
        rmse=float(np.sqrt(np.mean(difference * difference))),
    )


def _event_structure(event: NoteEvent) -> tuple[float, float, int, tuple[int, ...] | None]:
    return event.start_time_s, event.end_time_s, event.pitch_midi, event.pitch_bend


def _event_metrics(onnx_events: list[NoteEvent], torch_events: list[NoteEvent]) -> tuple[int, float | None, float | None, int]:
    if len(onnx_events) != len(torch_events):
        return 1, None, None, max(len(onnx_events), len(torch_events))
    structure_disagreements = sum(
        left != right for left, right in zip(map(_event_structure, onnx_events), map(_event_structure, torch_events))
    )
    amplitude_diffs = np.asarray([right.amplitude - left.amplitude for left, right in zip(onnx_events, torch_events)], dtype=np.float64)
    amplitude_abs = np.abs(amplitude_diffs)
    pitch_bend_disagreements = sum(
        (left.pitch_bend or ()) != (right.pitch_bend or ()) for left, right in zip(onnx_events, torch_events)
    )
    if amplitude_abs.size == 0:
        return structure_disagreements, 0.0, 0.0, int(pitch_bend_disagreements)
    return (
        structure_disagreements,
        float(np.max(amplitude_abs)),
        float(np.mean(amplitude_abs)),
        int(pitch_bend_disagreements),
    )


def compare_windows(onnx_path: Path, checkpoint_path: Path, windows: np.ndarray) -> ParitySummary:
    """Compare identical C-contiguous float32 windows through both model paths."""
    if windows.dtype != np.float32 or windows.ndim != 3 or windows.shape[1:] != (AUDIO_N_SAMPLES, 1):
        raise ValueError(f"expected CQT windows [N,{AUDIO_N_SAMPLES},1] float32, got {windows.shape} {windows.dtype}")
    if not windows.flags.c_contiguous:
        raise ValueError("parity windows must be C-contiguous")
    if windows.shape[0] == 0:
        raise ValueError("parity requires at least one window")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"unexpected ONNX Runtime providers: {session.get_providers()}")
    if session.get_inputs()[0].name != ONNX_INPUT_NAME:
        raise RuntimeError("unexpected ONNX Runtime input name")
    onnx_values = session.run(None, {ONNX_INPUT_NAME: windows})
    output_names = [output.name for output in session.get_outputs()]
    onnx_by_name = dict(zip(output_names, onnx_values, strict=True))
    onnx_outputs = {
        "note": np.asarray(onnx_by_name[ONNX_OUTPUT_NOTE]),
        "onset": np.asarray(onnx_by_name[ONNX_OUTPUT_ONSET]),
        "contour": np.asarray(onnx_by_name[ONNX_OUTPUT_CONTOUR]),
    }

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = BasicPitchICASSP2022()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    with torch.inference_mode():
        torch_values = model(torch.from_numpy(windows))
    torch_outputs = {name: value.detach().cpu().numpy() for name, value in torch_values.items()}

    output_metrics = {name: _output_parity(onnx_outputs[name], torch_outputs[name]) for name in ("contour", "note", "onset")}
    note_disagreements = int(np.count_nonzero((onnx_outputs["note"] >= FRAME_THRESHOLD) != (torch_outputs["note"] >= FRAME_THRESHOLD)))
    onset_disagreements = int(np.count_nonzero((onnx_outputs["onset"] >= ONSET_THRESHOLD) != (torch_outputs["onset"] >= ONSET_THRESHOLD)))

    onnx_events: list[NoteEvent] = []
    torch_events: list[NoteEvent] = []
    for index in range(windows.shape[0]):
        onnx_events.extend(posteriorgrams_to_note_events({name: value[index] for name, value in onnx_outputs.items()}))
        torch_events.extend(posteriorgrams_to_note_events({name: value[index] for name, value in torch_outputs.items()}))
    structure_disagreements, amplitude_max, amplitude_mean, pitch_bend_disagreements = _event_metrics(onnx_events, torch_events)
    return ParitySummary(
        contour=output_metrics["contour"],
        note=output_metrics["note"],
        onset=output_metrics["onset"],
        note_threshold_disagreements=note_disagreements,
        onset_threshold_disagreements=onset_disagreements,
        note_threshold_elements=int(onnx_outputs["note"].size),
        onset_threshold_elements=int(onnx_outputs["onset"].size),
        onnx_event_count=len(onnx_events),
        torch_event_count=len(torch_events),
        event_structure_disagreements=structure_disagreements,
        amplitude_max_abs_error=amplitude_max,
        amplitude_mean_abs_error=amplitude_mean,
        pitch_bend_element_disagreements=pitch_bend_disagreements,
        synthetic_windows=windows.shape[0],
    )


def summary_as_dict(summary: ParitySummary) -> dict[str, Any]:
    return asdict(summary)


def assert_parity(summary: ParitySummary) -> None:
    for name, tolerance in ADOPTED_MAX_ABS_TOLERANCES.items():
        observed = getattr(summary, name).max_abs_error
        if observed > tolerance:
            raise AssertionError(f"{name} max absolute error {observed} exceeds {tolerance}")
    if summary.note_threshold_disagreements or summary.onset_threshold_disagreements:
        raise AssertionError("threshold crossing disagreement detected")
    if summary.event_structure_disagreements or summary.pitch_bend_element_disagreements:
        raise AssertionError("stock note-event disagreement detected")


def write_reports(
    summary: ParitySummary,
    json_path: Path,
    markdown_path: Path,
    *,
    private_local_clips: int = 0,
    force: bool = False,
) -> None:
    json_path = json_path.resolve(strict=False)
    markdown_path = markdown_path.resolve(strict=False)
    workspace = Path(__file__).resolve().parents[1]
    reports = workspace / "reports"
    if not json_path.is_relative_to(reports) or not markdown_path.is_relative_to(reports):
        raise ValueError("parity reports must be written inside the approved reports directory")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    if not force and (json_path.exists() or markdown_path.exists()):
        raise FileExistsError("refusing to overwrite parity reports without force=True")
    report = {
        "format_version": 1,
        "model_id": MODEL_ID,
        "source_git_blob_sha1": SPOTIFY_ONNX_GIT_BLOB_SHA1,
        "runtime": {
            "torch_version": torch.__version__,
            "onnxruntime_version": ort.__version__,
            "device": "cpu",
            "precision": "float32",
        },
        "cases": {
            "synthetic_windows": summary.synthetic_windows,
            "private_local_clips": private_local_clips,
            "private_local_windows": summary.private_local_windows,
        },
        "outputs": {
            "contour": asdict(summary.contour),
            "note": asdict(summary.note),
            "onset": asdict(summary.onset),
        },
        "thresholds": {
            "note": FRAME_THRESHOLD,
            "onset": ONSET_THRESHOLD,
            "note_disagreements": summary.note_threshold_disagreements,
            "onset_disagreements": summary.onset_threshold_disagreements,
            "note_elements": summary.note_threshold_elements,
            "onset_elements": summary.onset_threshold_elements,
        },
        "events": {
            "onnx_count": summary.onnx_event_count,
            "torch_count": summary.torch_event_count,
            "structure_disagreements": summary.event_structure_disagreements,
            "pitch_bend_element_disagreements": summary.pitch_bend_element_disagreements,
            "amplitude_max_abs_error": summary.amplitude_max_abs_error,
            "amplitude_mean_abs_error": summary.amplitude_mean_abs_error,
        },
        "adopted_regression_tolerances": {
            "contour_max_abs": ADOPTED_MAX_ABS_TOLERANCES["contour"],
            "note_max_abs": ADOPTED_MAX_ABS_TOLERANCES["note"],
            "onset_max_abs": ADOPTED_MAX_ABS_TOLERANCES["onset"],
        },
        "known_non_equivalences": [
            {
                "case": "all_zero_window",
                "description": "ONNX Runtime CPU's float32 log kernel leaves a one-ulp spread at 1e-10; the native divide-by-zero-safe normalized log returns zero.",
                "threshold_disagreements": 0,
                "event_disagreements": 0,
            }
        ],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = "\n".join(
        [
            "# Basic Pitch ONNX parity",
            "",
            f"- Model: `{MODEL_ID}`",
            f"- ONNX blob: `{SPOTIFY_ONNX_GIT_BLOB_SHA1}`",
            f"- Windows: `{summary.synthetic_windows}` synthetic, `{private_local_clips}` private clips",
            "- Runtime: ONNX Runtime CPU and PyTorch CPU, float32",
            "",
            "| Output | Max absolute | Mean absolute | RMSE |",
            "| --- | ---: | ---: | ---: |",
            *[
                f"| {name} | {getattr(summary, name).max_abs_error:.9g} | {getattr(summary, name).mean_abs_error:.9g} | {getattr(summary, name).rmse:.9g} |"
                for name in ("contour", "note", "onset")
            ],
            "",
            f"Threshold disagreements: note `{summary.note_threshold_disagreements}`, onset `{summary.onset_threshold_disagreements}`.",
            f"Event counts: ONNX `{summary.onnx_event_count}`, PyTorch `{summary.torch_event_count}`; structural disagreements `{summary.event_structure_disagreements}`.",
            "",
            "The report contains aggregate values only; private validation identities are intentionally omitted.",
        ]
    )
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
