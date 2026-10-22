"""Conservative archive planning for full basic targets and retained source audits."""

from math import prod

from obruxo_data.midi import Performance, TempoMap

from .features import PITCH_SIZE, TIMBRE_SHAPE
from .spectral import ENVELOPE_HOP, FFT_SIZES, MAX_SAMPLES, SAMPLE_RATE


def target_tensor_bytes(samples):
    if type(samples) is not int or not 1 <= samples <= MAX_SAMPLES:
        raise ValueError("planned spectral target exceeds the qualified length")
    spectra = sum((2 * (size // 2 + 1) * 4 + 1) * (samples // (size // 4) + 1) for size in FFT_SIZES)
    envelope = (3 * 4 + 1) * (samples // ENVELOPE_HOP + 1)
    return spectra + envelope + 8


def storage_bound(training, validation, tail_seconds):
    def frames(recipe):
        result = []
        for value in recipe["performances"]:
            performance = Performance.from_dict(value)
            result.append(TempoMap.from_performance(performance).render_frame_count(performance.end_tick, tail_seconds, SAMPLE_RATE))
        return result
    train_frames = [count for recipe in training for count in frames(recipe)]
    audit_frames = train_frames + [count for recipe in validation for count in frames(recipe)]
    input_views = sum(len(recipe["performances"]) if recipe.get("input_views") == "all" else 1 for recipe in training)
    records = sum(target_tensor_bytes(count) for count in train_frames) + input_views * 4 * (prod(TIMBRE_SHAPE) + PITCH_SIZE)
    audit_audio = sum(audit_frames) * 2 * 4 * 2  # Stereo float32, current and previous atomic-save copies.
    serialization = len(training) * 128 * 1024 + len(audit_frames) * 256 * 1024
    reserve = 256 * 1024**2
    return {"format": "basic_native_fit_archive_bound_v1", "training_record_tensor_bytes": records,
            "audit_source_audio_bytes_with_previous_copies": audit_audio, "serialization_allowance_bytes": serialization,
            "checkpoint_export_ledger_temporary_reserve_bytes": reserve,
            "estimated_archive_upper_bound_bytes": records + audit_audio + serialization + reserve,
            "note": "planning allowance, not a measured peak; runtime archive cap and 32-MiB guard remain enforced"}
