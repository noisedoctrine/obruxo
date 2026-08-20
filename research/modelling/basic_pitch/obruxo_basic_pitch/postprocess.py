"""A dependency-light port of Spotify's stock Basic Pitch note decoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .constants import (
    ANNOT_N_FRAMES,
    ANNOTATIONS_BASE_FREQUENCY,
    AUDIO_N_SAMPLES,
    AUDIO_SAMPLE_RATE,
    CONTOURS_BINS_PER_SEMITONE,
    FFT_HOP,
    FRAME_THRESHOLD,
    MIN_NOTE_LENGTH_FRAMES,
    N_FREQ_BINS_CONTOURS,
    ONSET_THRESHOLD,
)

MIDI_OFFSET = 21
MAX_FREQ_IDX = 87


@dataclass(frozen=True)
class StockDecoderSettings:
    onset_threshold: float = ONSET_THRESHOLD
    frame_threshold: float = FRAME_THRESHOLD
    min_note_length_frames: int = MIN_NOTE_LENGTH_FRAMES
    infer_onsets: bool = True
    melodia_trick: bool = True
    include_pitch_bends: bool = True
    multiple_pitch_bends: bool = False


@dataclass(frozen=True)
class NoteEvent:
    start_time_s: float
    end_time_s: float
    pitch_midi: int
    amplitude: float
    pitch_bend: tuple[int, ...] | None = None


def _midi_to_hz(pitch_midi: int) -> float:
    return 440.0 * 2.0 ** ((pitch_midi - 69.0) / 12.0)


def _hz_to_midi(frequency: float) -> float:
    return 69.0 + 12.0 * np.log2(frequency / 440.0)


def midi_pitch_to_contour_bin(pitch_midi: int) -> float:
    return (
        12.0
        * CONTOURS_BINS_PER_SEMITONE
        * np.log2(_midi_to_hz(pitch_midi) / ANNOTATIONS_BASE_FREQUENCY)
    )


def model_frames_to_time(n_frames: int) -> np.ndarray:
    original_times = np.arange(n_frames, dtype=np.float64) * FFT_HOP / AUDIO_SAMPLE_RATE
    window_numbers = np.floor(np.arange(n_frames, dtype=np.float64) / ANNOT_N_FRAMES)
    window_offset = (FFT_HOP / AUDIO_SAMPLE_RATE) * (
        ANNOT_N_FRAMES - AUDIO_N_SAMPLES / FFT_HOP
    ) + 0.0018
    return original_times - (window_offset * window_numbers)


def get_inferred_onsets(
    onsets: np.ndarray, frames: np.ndarray, n_diff: int = 2
) -> np.ndarray:
    diffs = []
    for n in range(1, n_diff + 1):
        frames_appended = np.concatenate(
            [np.zeros((n, frames.shape[1]), dtype=frames.dtype), frames]
        )
        diffs.append(frames_appended[n:, :] - frames_appended[:-n, :])
    frame_diff = np.min(diffs, axis=0)
    frame_diff[frame_diff < 0] = 0
    frame_diff[:n_diff, :] = 0
    maximum = np.max(frame_diff)
    if maximum != 0:
        frame_diff = np.max(onsets) * frame_diff / maximum
    else:
        frame_diff.fill(0)
    return np.maximum(onsets, frame_diff)


def constrain_frequency(
    onsets: np.ndarray,
    frames: np.ndarray,
    max_freq: float | None,
    min_freq: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if max_freq is not None:
        max_freq_idx = int(np.round(_hz_to_midi(max_freq) - MIDI_OFFSET))
        onsets[:, max_freq_idx:] = 0
        frames[:, max_freq_idx:] = 0
    if min_freq is not None:
        min_freq_idx = int(np.round(_hz_to_midi(min_freq) - MIDI_OFFSET))
        onsets[:, :min_freq_idx] = 0
        frames[:, :min_freq_idx] = 0
    return onsets, frames


def _clear_frequency_neighborhood(
    energy: np.ndarray, start: int, end: int, frequency: int
) -> None:
    energy[start:end, frequency] = 0
    if frequency < MAX_FREQ_IDX:
        energy[start:end, frequency + 1] = 0
    if frequency > 0:
        energy[start:end, frequency - 1] = 0


def output_to_notes_polyphonic(
    frames: np.ndarray,
    onsets: np.ndarray,
    onset_thresh: float,
    frame_thresh: float,
    min_note_len: int,
    infer_onsets: bool,
    max_freq: float | None = None,
    min_freq: float | None = None,
    melodia_trick: bool = True,
    energy_tol: int = 11,
) -> list[tuple[int, int, int, float]]:
    n_frames = frames.shape[0]
    onsets, frames = constrain_frequency(onsets, frames, max_freq, min_freq)
    if infer_onsets:
        onsets = get_inferred_onsets(onsets, frames)

    peak_threshold = np.zeros(onsets.shape, dtype=onsets.dtype)
    peaks = signal.argrelmax(onsets, axis=0)
    peak_threshold[peaks] = onsets[peaks]
    onset_indices = np.where(peak_threshold >= onset_thresh)
    onset_time_indices = onset_indices[0][::-1]
    onset_frequency_indices = onset_indices[1][::-1]

    remaining_energy = np.array(frames, copy=True)
    note_events: list[tuple[int, int, int, float]] = []
    for note_start_idx, frequency_idx in zip(
        onset_time_indices, onset_frequency_indices
    ):
        if note_start_idx >= n_frames - 1:
            continue
        i = note_start_idx + 1
        below_threshold = 0
        while i < n_frames - 1 and below_threshold < energy_tol:
            if remaining_energy[i, frequency_idx] < frame_thresh:
                below_threshold += 1
            else:
                below_threshold = 0
            i += 1
        i -= below_threshold
        if i - note_start_idx <= min_note_len:
            continue
        _clear_frequency_neighborhood(
            remaining_energy, note_start_idx, i, frequency_idx
        )
        note_events.append(
            (
                note_start_idx,
                i,
                int(frequency_idx) + MIDI_OFFSET,
                float(np.mean(frames[note_start_idx:i, frequency_idx])),
            )
        )

    if melodia_trick:
        while np.max(remaining_energy) > frame_thresh:
            i_mid, frequency_idx = np.unravel_index(
                np.argmax(remaining_energy), remaining_energy.shape
            )
            remaining_energy[i_mid, frequency_idx] = 0

            i = i_mid + 1
            below_threshold = 0
            while i < n_frames - 1 and below_threshold < energy_tol:
                if remaining_energy[i, frequency_idx] < frame_thresh:
                    below_threshold += 1
                else:
                    below_threshold = 0
                _clear_frequency_neighborhood(remaining_energy, i, i + 1, frequency_idx)
                i += 1
            i_end = i - 1 - below_threshold

            i = i_mid - 1
            below_threshold = 0
            while i > 0 and below_threshold < energy_tol:
                if remaining_energy[i, frequency_idx] < frame_thresh:
                    below_threshold += 1
                else:
                    below_threshold = 0
                _clear_frequency_neighborhood(remaining_energy, i, i + 1, frequency_idx)
                i -= 1
            i_start = i + 1 + below_threshold
            if i_end - i_start <= min_note_len:
                continue
            note_events.append(
                (
                    i_start,
                    i_end,
                    int(frequency_idx) + MIDI_OFFSET,
                    float(np.mean(frames[i_start:i_end, frequency_idx])),
                )
            )

    return note_events


def get_pitch_bends(
    contours: np.ndarray,
    note_events: list[tuple[int, int, int, float]],
    n_bins_tolerance: int = 25,
) -> list[tuple[int, int, int, float, tuple[int, ...]]]:
    window_length = n_bins_tolerance * 2 + 1
    frequency_gaussian = signal.windows.gaussian(window_length, std=5)
    result = []
    for start_idx, end_idx, pitch_midi, amplitude in note_events:
        frequency_idx = int(np.round(midi_pitch_to_contour_bin(pitch_midi)))
        frequency_start = max(frequency_idx - n_bins_tolerance, 0)
        frequency_end = min(N_FREQ_BINS_CONTOURS, frequency_idx + n_bins_tolerance + 1)
        left_crop = max(0, n_bins_tolerance - frequency_idx)
        right_crop = max(
            0, frequency_idx - (N_FREQ_BINS_CONTOURS - n_bins_tolerance - 1)
        )
        submatrix = (
            contours[start_idx:end_idx, frequency_start:frequency_end]
            * frequency_gaussian[left_crop : window_length - right_crop]
        )
        pitch_shift = n_bins_tolerance - left_crop
        bends = tuple((np.argmax(submatrix, axis=1) - pitch_shift).astype(int).tolist())
        result.append((start_idx, end_idx, pitch_midi, amplitude, bends))
    return result


def posteriorgrams_to_note_events(
    output: dict[str, np.ndarray],
    *,
    onset_threshold: float = ONSET_THRESHOLD,
    frame_threshold: float = FRAME_THRESHOLD,
    minimum_note_length_frames: int = MIN_NOTE_LENGTH_FRAMES,
) -> list[NoteEvent]:
    frames = np.array(output["note"], copy=True)
    onsets = np.array(output["onset"], copy=True)
    contours = np.asarray(output["contour"])
    estimated = output_to_notes_polyphonic(
        frames,
        onsets,
        onset_thresh=onset_threshold,
        frame_thresh=frame_threshold,
        infer_onsets=True,
        min_note_len=minimum_note_length_frames,
        melodia_trick=True,
    )
    with_bends = get_pitch_bends(contours, estimated)
    times = model_frames_to_time(contours.shape[0])
    return [
        NoteEvent(
            start_time_s=float(times[start]),
            end_time_s=float(times[end]),
            pitch_midi=int(pitch),
            amplitude=float(amplitude),
            pitch_bend=tuple(bends) if bends is not None else None,
        )
        for start, end, pitch, amplitude, bends in with_bends
    ]


def decode_notes(
    output: dict[str, np.ndarray], settings: StockDecoderSettings | None = None
) -> list[NoteEvent]:
    """Compatibility wrapper for callers that want an explicit settings object."""
    if settings is None:
        settings = StockDecoderSettings()
    if not settings.include_pitch_bends or settings.multiple_pitch_bends:
        raise ValueError(
            "the parity decoder exposes only the stock single-path settings"
        )
    return posteriorgrams_to_note_events(
        output,
        onset_threshold=settings.onset_threshold,
        frame_threshold=settings.frame_threshold,
        minimum_note_length_frames=settings.min_note_length_frames,
    )
