from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile

import numpy as np

from .core import MODEL_REGISTRY
from .sensitivity import EXPECTED_VARIANTS, MODEL_ORDER


SEED = 161
TRIPLET_COUNT = 12
ROLE_NAMES = ("midi1_query", "midi2_retrieval", "midi3_sound_test")
PATTERN_ORDER = ("single_note_melody_bass", "chords_polyphonic", "repeating_sequence")
FFT_SIZES = (512, 2048, 8192)


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"issue161:{seed}:{value}".encode("utf-8")).hexdigest()


def classify_performance(row: dict) -> dict:
    """Classify only the frozen MIDI events and summary stats, never audio or encoder results."""
    from obruxo_data.midi import Performance

    performance = Performance.from_dict(row["excerpt"])
    notes = sorted(performance.note_spans(), key=lambda note: (note.start_tick, note.pitch, note.end_tick))
    if not notes:
        raise ValueError("selected MIDI excerpt has no complete notes")
    onset_groups = []
    for note in notes:
        if not onset_groups or onset_groups[-1][0] != note.start_tick:
            onset_groups.append((note.start_tick, [note.pitch]))
        else:
            onset_groups[-1][1].append(note.pitch)
    pitch_sequence = [int(np.median(pitches)) for _, pitches in onset_groups]
    pitch_classes = [tuple(sorted(pitch % 12 for pitch in pitches)) for _, pitches in onset_groups]
    repeating = any(
        count >= 2
        for width in (3, 4)
        for count in Counter(tuple(pitch_classes[start:start + width])
                             for start in range(max(0, len(pitch_classes) - width + 1))).values()
    )
    pitch_span = max(note.pitch for note in notes) - min(note.pitch for note in notes)
    monotonic_run = 1
    longest_run = 1
    direction = 0
    for left, right in zip(pitch_sequence, pitch_sequence[1:]):
        next_direction = 1 if right > left else -1 if right < left else 0
        if next_direction and next_direction == direction:
            monotonic_run += 1
        else:
            monotonic_run = 2 if next_direction else 1
            direction = next_direction
        longest_run = max(longest_run, monotonic_run)
    sequence = len(notes) >= 8 and len(onset_groups) >= 6 and (
        repeating or (pitch_span >= 12 and longest_run >= 5 and
                      sum(len(pitches) == 1 for _, pitches in onset_groups) / len(onset_groups) >= 0.8)
    )
    stats = row["stats"]
    texture = "chords_polyphonic" if stats.get("max_polyphony", 1) > 1 else "single_note_melody_bass"
    pattern = "repeating_sequence" if sequence else texture
    pattern_tags = [texture] + (["repeating_sequence"] if sequence else [])
    median_pitch = int(stats["pitch_median"])
    return {
        "midi_id": row["midi_id"], "excerpt_id": row["excerpt_id"], "pattern": pattern,
        "pattern_tags": pattern_tags, "texture": texture,
        "articulation": stats["articulation"], "register": "low" if median_pitch < 48 else "mid" if median_pitch < 68 else "high",
        "pitch_median": median_pitch, "pitch_min": int(stats["pitch_min"]), "pitch_max": int(stats["pitch_max"]),
        "pitch_span": int(stats["pitch_max"] - stats["pitch_min"]), "note_count": len(notes),
        "max_polyphony": int(stats["max_polyphony"]),
        "velocity_varied": stats["velocity_min"] != stats["velocity_max"],
        "velocity_min": int(stats["velocity_min"]), "velocity_max": int(stats["velocity_max"]),
        "repeated_onset_motif": bool(repeating), "longest_monotonic_note_run": int(longest_run),
    }


def _diversity_score(profile: dict, tag_counts: Counter, usage_count: int, seed: int) -> tuple:
    tags = (profile["articulation"], profile["register"], "velocity_varied" if profile["velocity_varied"] else "velocity_constant",
            "wide_pitch" if profile["pitch_span"] >= 24 else "mid_pitch" if profile["pitch_span"] >= 12 else "narrow_pitch")
    coverage = sum(1 / (1 + tag_counts[tag]) for tag in tags)
    return (-usage_count, coverage, _stable_key(seed, profile["midi_id"]))


def preselect_triplets(performance_rows: list[dict], *, seed: int = SEED, count: int = TRIPLET_COUNT) -> dict:
    """Freeze a balanced three-role sample from MIDI events before any encoder ranking."""
    if count != TRIPLET_COUNT:
        raise ValueError("issue #161 requires exactly 12 preselected MIDI triplets")
    profiles = [classify_performance(row) for row in performance_rows]
    if len({row["midi_id"] for row in profiles}) != len(profiles):
        raise ValueError("MIDI performance identities must be unique")
    by_pattern = {pattern: [row for row in profiles if pattern in row["pattern_tags"]] for pattern in PATTERN_ORDER}
    available = [pattern for pattern in PATTERN_ORDER if by_pattern[pattern]]
    if not available:
        raise ValueError("no classified MIDI performances are available")
    tag_counts: Counter = Counter()
    selected_ids: set[str] = set()
    usage = Counter()
    triplets = []
    for triplet_index in range(count):
        role_patterns = [available[(triplet_index + role) % len(available)] for role in range(3)]
        members = []
        for role, pattern in enumerate(role_patterns):
            candidates = [row for row in by_pattern[pattern] if row["midi_id"] not in {item["midi_id"] for item in members}]
            if not candidates:
                raise ValueError("could not select three distinct MIDI identities in one triplet")
            unused = [row for row in candidates if row["midi_id"] not in selected_ids]
            pool = unused or candidates
            chosen = max(pool, key=lambda row: _diversity_score(row, tag_counts, usage[row["midi_id"]], seed))
            members.append(chosen)
            selected_ids.add(chosen["midi_id"])
            usage[chosen["midi_id"]] += 1
            tag_counts.update((chosen["articulation"], chosen["register"],
                               "velocity_varied" if chosen["velocity_varied"] else "velocity_constant",
                               "wide_pitch" if chosen["pitch_span"] >= 24 else "mid_pitch" if chosen["pitch_span"] >= 12 else "narrow_pitch"))
        triplets.append({"triplet_index": triplet_index, "roles": [
            {**members[role], "pattern": role_patterns[role], "role": ROLE_NAMES[role]} for role in range(3)]})
    role_pattern_counts = {role: dict(Counter(item["pattern"] for triplet in triplets
                                               for item in triplet["roles"] if item["role"] == role))
                           for role in ROLE_NAMES}
    coverage = {
        "pool_performances": len(profiles), "triplets": len(triplets),
        "unique_selected_performances": len(selected_ids),
        "selected_performances_reused": sum(value > 1 for value in usage.values()),
        "available_pattern_counts": {pattern: len(by_pattern[pattern]) for pattern in PATTERN_ORDER},
        "selected_pattern_counts_by_role": role_pattern_counts,
        "selected_articulation_counts": dict(Counter(row["articulation"] for triplet in triplets for row in triplet["roles"])),
        "selected_register_counts": dict(Counter(row["register"] for triplet in triplets for row in triplet["roles"])),
        "selected_velocity_varied_count": sum(row["velocity_varied"] for triplet in triplets for row in triplet["roles"]),
        "selected_wide_pitch_span_count": sum(row["pitch_span"] >= 24 for triplet in triplets for row in triplet["roles"]),
        "unavailable_patterns": [pattern for pattern in PATTERN_ORDER if not by_pattern[pattern]],
    }
    return {"seed": seed, "selection_algorithm": "issue161_midi_event_stratified_greedy_v1",
            "selection_uses_encoder_results": False, "coverage": coverage, "triplets": triplets}


def verify_triplet_wavs(dataset_dir: Path, cache_audit: dict, plan_rows: list[dict], triplets: dict,
                        *, midi_ids_override: set[str] | None = None) -> tuple[dict, dict]:
    """Check selected matrix cells against accepted receipt, metadata provenance and exact WAV bytes."""
    from scipy.io import wavfile
    from .rendering import RENDER_POLICY_ID, QA_POLICY_ID, _receipt_matches_cached_audio, render_plan_identity
    from .sensitivity import _file_sha256

    dataset_dir = dataset_dir.resolve(strict=True)
    plan_by_request = {row["request_id"]: row for row in plan_rows}
    plan_by_pair = {(row["state_id"], row["midi_id"]): row for row in plan_rows
                    if row.get("source_kind") == "newly_rendered_original_preset"}
    audio_rows = {}
    preset_ids = cache_audit["preset_ids"]
    midi_ids = (midi_ids_override if midi_ids_override is not None else
                {item["midi_id"] for triplet in triplets["triplets"] for item in triplet["roles"]})
    for midi_id in sorted(midi_ids):
        for state_id in preset_ids:
            plan = plan_by_pair.get((state_id, midi_id))
            request_id = plan["request_id"] if plan else None
            receipt = cache_audit["receipts"].get(request_id) if request_id else None
            if (not plan or not receipt or not receipt.get("accepted") or receipt.get("qa_policy_id") != QA_POLICY_ID or
                    receipt.get("render_policy_id") != RENDER_POLICY_ID or receipt.get("plan_identity") != render_plan_identity(plan) or
                    receipt.get("source_kind") != "newly_rendered_original_preset"):
                raise ValueError("selected triplet matrix cell lacks an accepted frozen render receipt")
            for key in ("wav_relative_path", "metadata_relative_path"):
                relative = receipt.get(key)
                if not isinstance(relative, str) or "\\" in relative or PureWindowsPath(relative).drive:
                    raise ValueError("render receipt contains a non-portable relative path")
                relative_path = PurePosixPath(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts or not relative_path.parts:
                    raise ValueError("render receipt path escapes the frozen dataset")
            wav_path = dataset_dir.joinpath(*PurePosixPath(receipt["wav_relative_path"]).parts).resolve(strict=True)
            metadata_path = dataset_dir.joinpath(*PurePosixPath(receipt["metadata_relative_path"]).parts).resolve(strict=True)
            if not wav_path.is_relative_to(dataset_dir) or not metadata_path.is_relative_to(dataset_dir):
                raise ValueError("selected render path escapes the frozen dataset")
            audio_sha = _file_sha256(wav_path)
            if audio_sha != receipt.get("audio_sha256"):
                raise ValueError("selected triplet WAV bytes differ from the accepted receipt")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not _receipt_matches_cached_audio(receipt, request_id=request_id,
                    plan_identity=render_plan_identity(plan), renderer_id=plan["renderer_id"],
                    metadata=metadata, audio_sha256=audio_sha):
                raise ValueError("selected triplet render metadata differs from its accepted receipt")
            if plan.get("sample_rate") != 44_100 or plan.get("channels") != 2 or metadata.get("sample_rate") != 44_100:
                raise ValueError("selected render does not use the frozen stereo 44.1 kHz contract")
            frames = metadata.get("qa", {}).get("actual_frames")
            if not isinstance(frames, int) or frames < 8_192:
                raise ValueError("selected render has an invalid scoring-window length")
            audio_rows[(state_id, midi_id)] = {"path": wav_path, "receipt": receipt, "frames": frames,
                                                "qa_warnings": receipt.get("qa_warnings", [])}
    frame_counts = {midi_id: {audio_rows[(state_id, midi_id)]["frames"] for state_id in preset_ids}
                    for midi_id in midi_ids}
    if any(len(values) != 1 for values in frame_counts.values()):
        raise ValueError("same-MIDI scoring receipts do not share an exact sample-aligned window")
    warning_counts = Counter(warning for row in audio_rows.values() for warning in row["qa_warnings"])
    coverage = {"selected_matrix_wavs_verified": len(audio_rows), "midi3_scoring_wavs_loaded": 0,
                "selected_unique_midis": len(midi_ids), "qa_warning_counts": dict(sorted(warning_counts.items()))}
    return audio_rows, coverage


def load_scoring_midi(audio_rows: dict, preset_ids: list[str], midi3_id: str) -> list[np.ndarray]:
    """Load only one frozen MIDI-3 waveform matrix at a time to bound memory."""
    from scipy.io import wavfile

    samples = []
    expected_shape = None
    for state_id in preset_ids:
        item = audio_rows[(state_id, midi3_id)]
        rate, raw = wavfile.read(item["path"])
        if rate != 44_100 or raw.dtype != np.float32 or raw.ndim != 2 or raw.shape[1] != 2 or raw.shape[0] != item["frames"]:
            raise ValueError("selected WAV does not match its stereo float32 render receipt")
        if expected_shape is None:
            expected_shape = raw.shape
        elif raw.shape != expected_shape:
            raise ValueError("same-MIDI scoring WAVs do not share an exact sample-aligned window")
        if not np.isfinite(raw).all() or not float(np.square(raw.astype(np.float64)).mean()) > 0:
            raise ValueError("selected scoring WAV is nonfinite or silent")
        samples.append(raw)
    return samples


def raw_mrstft_matrix(audio_rows: list[np.ndarray], *, output_dir: Path, batch_size: int = 8,
                      pair_batch_size: int = 64, device_name: str = "cpu", progress=None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute raw and gain-matched aligned MR-STFT matrices from one set of WAV spectra."""
    import torch

    if len(audio_rows) < 2 or batch_size < 1 or pair_batch_size < 1:
        raise ValueError("MR-STFT matrix needs at least two recordings and a positive batch size")
    device = torch.device(device_name)
    if device.type == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("requested XPU is unavailable")
    shapes = {row.shape for row in audio_rows}
    if len(shapes) != 1 or next(iter(shapes))[1] != 2:
        raise ValueError("MR-STFT inputs must be sample-aligned stereo clips")
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    count, frames, _ = len(audio_rows), audio_rows[0].shape[0], audio_rows[0].shape[1]
    score_sum = np.zeros((count, count), dtype=np.float64)
    gain_score_sum = np.zeros((count, count), dtype=np.float64)
    rms_values = np.asarray([np.sqrt(np.mean(row.astype(np.float64) ** 2)) for row in audio_rows], dtype=np.float32)
    device_rms = torch.from_numpy(rms_values).to(device).clamp_min(1e-12)
    resolution_diagnostics = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for fft_size in FFT_SIZES:
        hop = fft_size // 4
        frame_count = frames // hop + 1
        dimensions = 2 * (fft_size // 2 + 1) * frame_count
        with tempfile.TemporaryDirectory(prefix=f"issue161-{fft_size}-", dir=output_dir) as temp:
            features = np.lib.format.open_memmap(Path(temp) / "magnitude.npy", mode="w+", dtype=np.float32,
                                                 shape=(count, dimensions))
            window = torch.hann_window(fft_size, dtype=torch.float32, device=device)
            for start in range(0, count, batch_size):
                audio = np.stack(audio_rows[start:start + batch_size])
                batch = torch.from_numpy(audio.copy()).to(device).transpose(1, 2).reshape(-1, frames)
                magnitude = torch.stft(batch, fft_size, hop_length=hop, window=window, return_complex=True).abs()
                flattened = magnitude.reshape(len(audio), 2, fft_size // 2 + 1, -1).reshape(len(audio), -1).cpu().numpy()
                if flattened.shape[1] != dimensions:
                    raise ValueError("STFT frame count differs from the fixed full-clip scoring window")
                features[start:start + len(audio)] = flattened
                del audio, batch, magnitude, flattened
            features.flush()
            device_features = torch.from_numpy(np.asarray(features)).to(device)
            resolution_scores = np.zeros((count, count), dtype=np.float64)
            resolution_gain_scores = np.zeros((count, count), dtype=np.float64)
            for reference_index in range(count):
                reference = device_features[reference_index].reshape(2, fft_size // 2 + 1, frame_count)
                reference_norm = torch.linalg.vector_norm(reference).clamp_min(1e-8)
                reference_log = torch.log(reference.clamp_min(1e-5))
                for start in range(0, count, pair_batch_size):
                    end = min(count, start + pair_batch_size)
                    candidates = device_features[start:end].reshape(end - start, 2, fft_size // 2 + 1, frame_count)
                    convergence = torch.linalg.vector_norm((candidates - reference).flatten(1), dim=1) / reference_norm
                    log_mae = torch.abs(torch.log(candidates.clamp_min(1e-5)) - reference_log).flatten(1).mean(dim=1)
                    resolution_scores[reference_index, start:end] = (convergence + log_mae).cpu().numpy()
                    scales = (device_rms[reference_index] / device_rms[start:end]).reshape(-1, 1, 1, 1)
                    matched_candidates = candidates * scales
                    matched_convergence = torch.linalg.vector_norm((matched_candidates - reference).flatten(1), dim=1) / reference_norm
                    matched_log_mae = torch.abs(torch.log(matched_candidates.clamp_min(1e-5)) - reference_log).flatten(1).mean(dim=1)
                    resolution_gain_scores[reference_index, start:end] = (matched_convergence + matched_log_mae).cpu().numpy()
                if progress and ((reference_index + 1) % 16 == 0 or reference_index + 1 == count):
                    progress(f"FFT {fft_size}: {reference_index + 1}/{count} target spectra compared")
            score_sum += resolution_scores / len(FFT_SIZES)
            gain_score_sum += resolution_gain_scores / len(FFT_SIZES)
            resolution_diagnostics[str(fft_size)] = {"fft_size": fft_size, "hop_length": hop,
                                                       "scored_ordered_pairs": count * count}
            del device_features, reference, reference_log, candidates, convergence, log_mae, matched_candidates, matched_convergence, matched_log_mae
            if device.type == "xpu":
                torch.xpu.synchronize()
            features.flush()
            features._mmap.close()
            del features
    np.fill_diagonal(score_sum, 0.0)
    np.fill_diagonal(gain_score_sum, 0.0)
    if not np.isfinite(score_sum).all() or not np.isfinite(gain_score_sum).all():
        raise ValueError("MR-STFT matrix contains nonfinite acoustic scores")
    return score_sum.astype(np.float32), gain_score_sum.astype(np.float32), {"metric": "mean_3_scale_spectral_convergence_plus_log_magnitude_mae",
        "fft_sizes": list(FFT_SIZES), "window": "hann", "hop_fraction": 0.25,
        "alignment": "sample zero; exact full-clip frame match; no shift, trim, or padding",
        "silence": "reject nonfinite or zero-energy clips; do not floor waveforms",
        "clipping": "retain accepted float32 samples unchanged; report receipt warnings",
        "scoring_window": "full saved stereo WAV including renderer tail", "device": device_name,
        "pair_batch_size": pair_batch_size, "resolutions": resolution_diagnostics}


def retrieve_alternatives(cache_audit: dict, triplets: dict) -> dict:
    """Rank all common 128-preset alternative pools using only MIDI 1/2 cached embeddings."""
    preset_ids = cache_audit["preset_ids"]
    family_by_state = cache_audit["family_by_state"]
    midi_ids = sorted({triplet["roles"][role]["midi_id"] for triplet in triplets["triplets"] for role in (0, 1)})
    midi_lookup = {value: index for index, value in enumerate(midi_ids)}
    state_lookup = {value: index for index, value in enumerate(preset_ids)}
    choices = {}
    for model_name in MODEL_ORDER:
        variants = EXPECTED_VARIANTS[model_name]
        dimensions = cache_audit["model_caches"][model_name]["dimensions_by_variant"]
        grids = {variant: np.empty((len(preset_ids), len(midi_ids), dimensions[variant]), dtype=np.float32)
                 for variant in variants}
        cache = cache_audit["model_caches"][model_name]
        for state_index, state_id in enumerate(preset_ids):
            for midi_id in midi_ids:
                request_id = cache_audit["request_by_pair"].get((state_id, midi_id))
                row = cache["rows_by_request"].get(request_id)
                if row is None:
                    raise ValueError("retrieval embedding cache misses a selected MIDI/state cell")
                with np.load(cache["directory"] / row["cache_path"], allow_pickle=False) as arrays:
                    for variant in variants:
                        grids[variant][state_index, midi_lookup[midi_id]] = arrays[variant]
        for variant, grid in grids.items():
            norms = np.linalg.norm(grid.astype(np.float64), axis=-1, keepdims=True)
            if not np.isfinite(grid).all() or np.any(norms <= 1e-12):
                raise ValueError("selected embedding matrix contains nonfinite or zero-norm values")
            grid /= norms.astype(np.float32)
            arm = f"{model_name}:{variant}"
            rows = []
            for triplet in triplets["triplets"]:
                midi1, midi2, midi3 = [item for item in triplet["roles"]]
                query_index, gallery_index = midi_lookup[midi1["midi_id"]], midi_lookup[midi2["midi_id"]]
                similarities = grid[:, query_index] @ grid[:, gallery_index].T
                for target_index, target_id in enumerate(preset_ids):
                    eligible = [candidate_index for candidate_index, candidate_id in enumerate(preset_ids)
                                if family_by_state[candidate_id] != family_by_state[target_id]]
                    ranked = sorted(eligible, key=lambda index: (-float(similarities[target_index, index]), preset_ids[index]))
                    chosen = ranked[0]
                    rows.append({"triplet_index": triplet["triplet_index"], "target_family_id": family_by_state[target_id],
                        "target_state_id": target_id, "candidate_state_id": preset_ids[chosen],
                        "candidate_rank": 1, "cosine_similarity": float(similarities[target_index, chosen]),
                        "eligible_candidates": len(eligible), "midi1_pattern": midi1["pattern"],
                        "midi2_pattern": midi2["pattern"], "midi3_pattern": midi3["pattern"],
                        "midi3_register": midi3["register"], "midi3_articulation": midi3["articulation"]})
            if len(rows) != 129 * TRIPLET_COUNT or any(row["eligible_candidates"] != 128 for row in rows):
                raise ValueError("retrieval episode denominator or candidate pool is incomplete")
            choices[arm] = rows
        del grids
    return choices


def challenger_reference_choices(preset_ids: list[str], family_by_state: dict, raw_matrices: dict[int, np.ndarray],
                                triplet_indices: list[int], *, seed: int = SEED) -> dict:
    """Select the MIDI-3 oracle and fixed random alternative only after retrieval is frozen."""
    result = {}
    for triplet_index in triplet_indices:
        matrix = raw_matrices[triplet_index]
        for target_index, target_id in enumerate(preset_ids):
            eligible = [candidate for candidate in preset_ids
                        if family_by_state[candidate] != family_by_state[target_id]]
            best = min(eligible, key=lambda candidate: (float(matrix[target_index, preset_ids.index(candidate)]), candidate))
            digest = _stable_key(seed, f"random:{triplet_index}:{target_id}")
            random_position = int(digest[:16], 16) % len(eligible)
            result[(triplet_index, target_id)] = {"oracle_state_id": best,
                "random_state_id": eligible[random_position], "oracle_error": float(matrix[target_index, preset_ids.index(best)])}
    return result


def score_secondary_metrics(choices: dict, references: dict, *, triplets: dict, preset_ids: list[str],
                            raw_matrices: dict[int, np.ndarray], gain_matrices: dict[int, np.ndarray],
                            audio_by_midi3: dict[str, list[np.ndarray]]) -> dict:
    """Reuse shared STFT matrices and compact WAV-derived RMS/envelopes for secondary scores."""
    triplet_midi3 = {row["triplet_index"]: row["roles"][2]["midi_id"] for row in triplets["triplets"]}
    state_index = {value: index for index, value in enumerate(preset_ids)}
    pairs = set()
    for rows in choices.values():
        pairs.update((row["triplet_index"], row["target_state_id"], row["candidate_state_id"]) for row in rows)
    for (triplet_index, target_id), row in references.items():
        pairs.add((triplet_index, target_id, row["oracle_state_id"]))
        pairs.add((triplet_index, target_id, row["random_state_id"]))
    groups = {}
    for triplet_index, target_id, candidate_id in sorted(pairs):
        groups.setdefault(triplet_index, []).append((target_id, candidate_id))
    result = {}
    for triplet_index, cases in groups.items():
        samples = audio_by_midi3[triplet_midi3[triplet_index]]
        windows = int(math.ceil(samples[0].shape[0] / 441))
        pad = windows * 441 - samples[0].shape[0]
        envelopes = []
        rms_values = []
        for audio in samples:
            values = audio.astype(np.float64)
            rms_values.append(float(np.sqrt(np.mean(values ** 2))))
            padded = np.pad(values, ((0, pad), (0, 0)))
            envelopes.append(np.sqrt(np.mean(padded.reshape(windows, 441, 2) ** 2, axis=1)))
        raw_matrix, gain_matrix = raw_matrices[triplet_index], gain_matrices[triplet_index]
        for target_id, candidate_id in cases:
            target_index, candidate_index = state_index[target_id], state_index[candidate_id]
            reference_env, candidate_env = envelopes[target_index], envelopes[candidate_index]
            env_denom = max(float(np.sqrt(np.mean(reference_env ** 2))), 1e-8)
            envelope_error = float(np.sqrt(np.mean((reference_env - candidate_env) ** 2)) / env_denom)
            rms_relative = abs(rms_values[target_index] - rms_values[candidate_index]) / max(rms_values[target_index], 1e-8)
            key = (triplet_index, target_id, candidate_id)
            result[key] = {"rms_matched_mrstft": float(gain_matrix[target_index, candidate_index]),
                "envelope_relative_rmse": envelope_error, "rms_relative": rms_relative,
                "raw_mrstft_check": float(raw_matrix[target_index, candidate_index])}
    return result


def _cluster_draws(family_count: int, triplet_count: int, *, seed: int, replicates: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (rng.integers(family_count, size=(replicates, family_count), dtype=np.int16),
            rng.integers(triplet_count, size=(replicates, triplet_count), dtype=np.int8))


def _bootstrap_summary(values: np.ndarray, family_draws: np.ndarray, triplet_draws: np.ndarray) -> dict:
    estimate = float(values.mean())
    draws = values[family_draws[:, :, None], triplet_draws[:, None, :]].mean(axis=(1, 2))
    return {"mean": estimate, "clustered_95_ci": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
            "family_clusters": values.shape[0], "triplet_clusters": values.shape[1]}


def build_results(choices: dict, raw_matrices: dict[int, np.ndarray], secondary: dict,
                  triplets: dict, existing_recall: dict, sensitivity: dict, *, seed: int = SEED,
                  bootstrap_replicates: int = 2_000) -> dict:
    arms = sorted(choices)
    family_ids = sorted({row["target_family_id"] for rows in choices.values() for row in rows})
    family_index = {value: index for index, value in enumerate(family_ids)}
    triplet_count = len(triplets["triplets"])
    if len(family_ids) != 129 or triplet_count != TRIPLET_COUNT:
        raise ValueError("Issue #161 report requires 129 families and 12 selected triplets")
    family_by_state = {row["target_state_id"]: row["target_family_id"]
                       for rows in choices.values() for row in rows}
    family_index_by_state = {state_id: family_index[family_id] for state_id, family_id in family_by_state.items()}
    preset_ids = sorted(family_by_state)
    state_index = {value: index for index, value in enumerate(preset_ids)}
    target_rows = choices[arms[0]]
    oracle = np.empty((len(family_ids), triplet_count), dtype=np.float64)
    random_errors = np.empty_like(oracle)
    oracle_choices = {}
    references = challenger_reference_choices(preset_ids, family_by_state, raw_matrices,
                                               sorted(raw_matrices), seed=seed)
    for triplet_index, matrix in raw_matrices.items():
        if matrix.shape != (129, 129):
            raise ValueError("MIDI 3 acoustic challenger matrix must be 129 by 129")
        for target_id in preset_ids:
            family_idx = family_index_by_state[target_id]
            target_state_idx = state_index[target_id]
            eligible = [candidate for candidate in preset_ids if family_by_state[candidate] != family_by_state[target_id]]
            reference = references[(triplet_index, target_id)]
            best_candidate = reference["oracle_state_id"]
            oracle[family_idx, triplet_index] = reference["oracle_error"]
            oracle_choices[(triplet_index, target_id)] = {"candidate_state_id": best_candidate,
                "error": reference["oracle_error"], "rank_by_error": 1, "candidate_count": len(eligible)}
            random_candidate = reference["random_state_id"]
            random_errors[family_idx, triplet_index] = float(matrix[target_state_idx, state_index[random_candidate]])
    results = {"schema": "issue161_audible_substitute_v1", "seed": seed, "triplets": triplet_count,
        "preset_families": len(family_ids), "episodes_per_arm": len(family_ids) * triplet_count,
        "eligible_alternatives_per_episode": 128, "random_baseline": _bootstrap_summary(random_errors,
            *_cluster_draws(len(family_ids), triplet_count, seed=seed, replicates=bootstrap_replicates)),
        "oracle": _bootstrap_summary(oracle, *_cluster_draws(len(family_ids), triplet_count, seed=seed, replicates=bootstrap_replicates)),
        "metric_contract": {"primary": "original-level MR-STFT = mean over 512/2048/8192 of spectral convergence + log-magnitude MAE",
            "secondary": ["RMS gain-matched MR-STFT", "10 ms stereo RMS-envelope relative RMSE", "global stereo RMS relative error"],
            "ties": "rank ties receive average rank; pairwise encoder win ties count one half",
            "uncertainty": "paired two-way bootstrap over 129 preset families and 12 fixed triplet clusters; descriptive"},
        "triplet_coverage": triplets["coverage"], "paired_comparisons_vs_cir": {}, "pairwise_wins": {}, "arms": {}}
    family_draws, triplet_draws = _cluster_draws(len(family_ids), triplet_count, seed=seed, replicates=bootstrap_replicates)
    choice_errors = {}
    choice_ranks = {}
    choice_excess = {}
    for arm in arms:
        raw = np.empty((len(family_ids), triplet_count), dtype=np.float64)
        matched = np.empty_like(raw)
        envelope = np.empty_like(raw)
        level = np.empty_like(raw)
        ranks = np.empty_like(raw)
        for row in choices[arm]:
            triplet_index, target_id = row["triplet_index"], row["target_state_id"]
            family_idx = family_index[row["target_family_id"]]
            case = (triplet_index, target_id, row["candidate_state_id"])
            secondary_row = secondary[case]
            matrix = raw_matrices[triplet_index]
            candidate_position = state_index[row["candidate_state_id"]]
            raw_value = float(matrix[state_index[target_id], candidate_position])
            eligible_errors = np.array([matrix[state_index[target_id], state_index[candidate]] for candidate in preset_ids
                                        if family_by_state[candidate] != family_by_state[target_id]], dtype=np.float64)
            less = int(np.sum(eligible_errors < raw_value - 1e-8))
            tied = int(np.sum(np.abs(eligible_errors - raw_value) <= 1e-8))
            ranks[family_idx, triplet_index] = less + (tied + 1) / 2
            raw[family_idx, triplet_index] = raw_value
            matched[family_idx, triplet_index] = secondary_row["rms_matched_mrstft"]
            envelope[family_idx, triplet_index] = secondary_row["envelope_relative_rmse"]
            level[family_idx, triplet_index] = secondary_row["rms_relative"]
        excess = raw - oracle
        choice_errors[arm], choice_ranks[arm], choice_excess[arm] = raw, ranks, excess
        family_means = raw.mean(axis=1)
        results["arms"][arm] = {
            "exact_preset_recall_at_1": existing_recall[arm],
            "issue157_preset_dominant_fraction": sensitivity[arm]["preset_dominant_fraction"],
            "original_level_mrstft": _bootstrap_summary(raw, family_draws, triplet_draws),
            "excess_over_oracle": _bootstrap_summary(excess, family_draws, triplet_draws),
            "gain_matched_mrstft": _bootstrap_summary(matched, family_draws, triplet_draws),
            "envelope_relative_rmse": _bootstrap_summary(envelope, family_draws, triplet_draws),
            "global_rms_relative_error": _bootstrap_summary(level, family_draws, triplet_draws),
            "rank_among_128_alternatives": {"mean": float(ranks.mean()), "median": float(np.median(ranks)),
                                               "oracle_first_fraction": float((ranks == 1).mean())},
            "family_mean_raw_error_quantiles": {str(q): float(np.quantile(family_means, q)) for q in (0.1, 0.5, 0.9)},
            "worst_decile_family_mean_raw_error": float(np.sort(family_means)[-max(1, math.ceil(len(family_means) * .1)):].mean()),
        }
    baseline_arm = "cir_ast:official_clip" if "cir_ast:official_clip" in arms else arms[0]
    for arm in arms:
        delta = choice_errors[arm] - choice_errors[baseline_arm]
        results["paired_comparisons_vs_cir"][arm] = _bootstrap_summary(delta, family_draws, triplet_draws)
    for left in arms:
        wins = {}
        for right in arms:
            delta = choice_errors[right] - choice_errors[left]
            wins[right] = float(np.mean(np.where(np.abs(delta) <= 1e-8, 0.5, delta > 0)))
        results["pairwise_wins"][left] = wins
    for role_key, field in (("midi1_pattern", "midi1_pattern"), ("midi2_pattern", "midi2_pattern"),
                            ("midi3_pattern", "midi3_pattern"), ("midi3_register", "midi3_register")):
        levels = sorted({row[field] for row in target_rows})
        results.setdefault("strata", {})[role_key] = {}
        for level_name in levels:
            triplets_for_level = sorted({row["triplet_index"] for row in target_rows if row[field] == level_name})
            selected_rows = [row for row in target_rows if row[field] == level_name]
            results["strata"][role_key][level_name] = {"episodes": len(selected_rows), "triplets": len(triplets_for_level),
                "arms": {arm: {"mean_original_level_mrstft": float(np.mean([
                                   choice_errors[arm][family_index[row["target_family_id"]], row["triplet_index"]]
                                   for row in selected_rows])),
                               "mean_excess_over_oracle": float(np.mean([
                                   choice_excess[arm][family_index[row["target_family_id"]], row["triplet_index"]]
                                   for row in selected_rows]))} for arm in arms}}
    # Private case details remain under ignored outputs and retain no waveform or embedding data.
    private = {"oracle": [{"triplet_index": triplet_index, "target_state_id": target_id, **value}
                            for (triplet_index, target_id), value in sorted(oracle_choices.items())],
               "arms": {arm: [{"triplet_index": row["triplet_index"], "target_family_id": row["target_family_id"],
                                "target_state_id": row["target_state_id"], "candidate_state_id": row["candidate_state_id"],
                                "raw_mrstft": float(choice_errors[arm][family_index[row["target_family_id"]], row["triplet_index"]]),
                                "excess_over_oracle": float(choice_excess[arm][family_index[row["target_family_id"]], row["triplet_index"]]),
                                "acoustic_rank": float(choice_ranks[arm][family_index[row["target_family_id"]], row["triplet_index"]]),
                                **secondary[(row["triplet_index"], row["target_state_id"], row["candidate_state_id"])]}
                               for row in rows] for arm, rows in choices.items()}}
    return results, private


def pairwise_win_matrix(choices: dict, raw_matrices: dict[int, np.ndarray], preset_ids: list[str], family_by_state: dict) -> dict:
    """Return paired win fractions from all 11 encoder choices on the same episodes."""
    arms = sorted(choices)
    state_index = {value: index for index, value in enumerate(preset_ids)}
    case_errors = {}
    for arm, rows in choices.items():
        case_errors[arm] = {(row["triplet_index"], row["target_state_id"]):
            float(raw_matrices[row["triplet_index"]][state_index[row["target_state_id"]], state_index[row["candidate_state_id"]]])
            for row in rows}
    return {left: {right: float(np.mean([0.5 if abs(case_errors[left][case] - case_errors[right][case]) <= 1e-8
                                          else case_errors[left][case] < case_errors[right][case]
                                          for case in case_errors[left]])) for right in arms} for left in arms}
