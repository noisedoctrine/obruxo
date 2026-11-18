from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[2]
sys.path[:0] = [str(PACKAGE), str(REPO / "research/data_generation"), str(REPO / "research/training")]

torch = pytest.importorskip("torch")

from frozen_retrieval.issue161_substitution import (build_results, preselect_triplets, raw_mrstft_matrix,
    active_rms_matched_mrstft_matrix)  # noqa: E402


def _profile(midi_id, pattern, index):
    texture = "chords_polyphonic" if pattern == "chords_polyphonic" else "single_note_melody_bass"
    return {"midi_id": midi_id, "excerpt_id": f"excerpt-{midi_id}", "pattern": pattern,
            "texture": texture, "pattern_tags": [pattern],
            "articulation": "held" if index % 2 else "phrased", "register": ("low", "mid", "high")[index % 3],
            "pitch_median": 42 + index % 40, "pitch_min": 24 + index % 20, "pitch_max": 72 + index % 30,
            "pitch_span": 24 if index % 2 else 12, "note_count": 12, "max_polyphony": 1 + index % 3,
            "velocity_varied": index % 2 == 0, "velocity_min": 40, "velocity_max": 90 if index % 2 == 0 else 40,
            "repeated_onset_motif": pattern == "repeating_sequence", "longest_monotonic_note_run": 5}


def test_triplets_are_reproducible_unique_and_rotate_available_patterns():
    profiles = {f"midi-{index}": _profile(f"midi-{index}", pattern, index)
                for pattern_index, pattern in enumerate(("single_note_melody_bass", "chords_polyphonic", "repeating_sequence"))
                for index in range(pattern_index * 12, (pattern_index + 1) * 12)}
    rows = [{"midi_id": midi_id} for midi_id in profiles]
    with patch("frozen_retrieval.issue161_substitution.classify_performance", side_effect=lambda row: profiles[row["midi_id"]]):
        first = preselect_triplets(rows, seed=161)
        second = preselect_triplets(list(reversed(rows)), seed=161)
    assert first == second
    assert first["coverage"]["unique_selected_performances"] == 36
    assert first["coverage"]["selected_performances_reused"] == 0
    for triplet in first["triplets"]:
        members = triplet["roles"]
        assert len({row["midi_id"] for row in members}) == 3
        assert {row["pattern"] for row in members} == set(profiles[row["midi_id"]]["pattern"] for row in members)
        assert len({row["pattern"] for row in members}) == 3
    for role_counts in first["coverage"]["selected_pattern_counts_by_role"].values():
        assert role_counts == {"single_note_melody_bass": 4, "chords_polyphonic": 4, "repeating_sequence": 4}


def test_triplet_selection_reports_pattern_types_missing_from_the_pool():
    profiles = {f"midi-{index}": _profile(f"midi-{index}", "single_note_melody_bass" if index < 30 else "chords_polyphonic", index)
                for index in range(50)}
    with patch("frozen_retrieval.issue161_substitution.classify_performance", side_effect=lambda row: profiles[row["midi_id"]]):
        selected = preselect_triplets([{"midi_id": midi_id} for midi_id in profiles])
    assert selected["coverage"]["unavailable_patterns"] == ["repeating_sequence"]
    assert all("repeating_sequence" not in [row["pattern"] for row in triplet["roles"]]
               for triplet in selected["triplets"])
    assert all(len({row["midi_id"] for row in triplet["roles"]}) == 3 for triplet in selected["triplets"])


@pytest.mark.parametrize("device_name", ["cpu", *( ["xpu"] if torch.xpu.is_available() else [])])
def test_raw_matrix_matches_existing_multiresolution_audio_utility(tmp_path, device_name):
    from obruxo_training.evaluate import audio_distance

    rng = np.random.default_rng(161)
    reference = (rng.normal(0, 0.15, (44_100, 2))).astype(np.float32)
    candidate = (reference * 0.62 + rng.normal(0, 0.025, reference.shape)).astype(np.float32)
    matrix, gain_matrix, contract = raw_mrstft_matrix([reference, candidate], output_dir=tmp_path,
                                                       batch_size=1, device_name=device_name)
    expected = audio_distance(reference, candidate)["mrstft"]
    reference_rms = np.sqrt(np.mean(reference.astype(np.float64) ** 2))
    candidate_rms = np.sqrt(np.mean(candidate.astype(np.float64) ** 2))
    gain_matched = audio_distance(reference, candidate * np.float32(reference_rms / candidate_rms))["mrstft"]
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 0
    assert matrix[0, 1] == pytest.approx(expected, abs=2e-5)
    assert gain_matrix[0, 1] == pytest.approx(gain_matched, abs=2e-5)
    assert not np.isclose(matrix[0, 1], matrix[1, 0])
    assert contract["fft_sizes"] == [512, 2048, 8192]
    assert contract["device"] == device_name
    assert contract["scoring_window"].startswith("full saved")


def test_active_rms_matrix_matches_reference_after_sliding_rms_gain(tmp_path):
    from obruxo_training.evaluate import audio_distance
    from frozen_retrieval.normalization import measure_active_rms

    rng = np.random.default_rng(164)
    reference = rng.normal(0, 0.12, (44_100, 2)).astype(np.float32)
    candidate = (reference * 0.55 + rng.normal(0, 0.03, reference.shape)).astype(np.float32)
    rms = [measure_active_rms(audio, 44_100)["active_rms"] for audio in (reference, candidate)]
    matrix, contract = active_rms_matched_mrstft_matrix([reference, candidate], rms, output_dir=tmp_path,
        batch_size=1, device_name="cpu")
    expected_candidate = candidate * np.float32(rms[0] / rms[1])
    expected = audio_distance(reference, expected_candidate)["mrstft"]
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 0
    assert matrix[0, 1] == pytest.approx(expected, abs=2e-5)
    assert contract["matching"].startswith("scale candidate B3")
    assert "sliding" in contract["active_rms"]


def test_aggregate_scores_each_arms_own_selected_candidate():
    states = [f"state-{index:03d}" for index in range(129)]
    families = {state: f"family-{index:03d}" for index, state in enumerate(states)}
    choices, matrices, secondary, triplets = {"arm_a": [], "arm_b": []}, {}, {}, {"triplets": [], "coverage": {}}
    for triplet_index in range(12):
        triplets["triplets"].append({"triplet_index": triplet_index, "roles": [
            {"midi_id": f"midi-{triplet_index}-{role}", "pattern": pattern, "register": "mid", "articulation": "held"}
            for role, pattern in enumerate(("melody", "chords", "repeating"))]})
        matrix = np.fromfunction(lambda i, j: np.where(i == j, 0, 1 + np.minimum(np.abs(i - j), 129 - np.abs(i - j))),
                                 (129, 129), dtype=float)
        matrices[triplet_index] = matrix.astype(np.float32)
        for index, target in enumerate(states):
            for arm, offset, pattern in (("arm_a", 1, "melody"), ("arm_b", 2, "chords")):
                candidate = states[(index + offset) % len(states)]
                row = {"triplet_index": triplet_index, "target_family_id": families[target],
                    "target_state_id": target, "candidate_state_id": candidate,
                    "midi1_pattern": pattern, "midi2_pattern": pattern, "midi3_pattern": pattern,
                    "midi3_register": "mid"}
                choices[arm].append(row)
                value = float(matrix[index, (index + offset) % len(states)])
                secondary[(triplet_index, target, candidate)] = {"rms_matched_mrstft": value,
                    "envelope_relative_rmse": value, "rms_relative": value, "raw_mrstft_check": value}
    recall = {"arm_a": 0.1, "arm_b": 0.2}
    sensitivity = {arm: {"preset_dominant_fraction": 0.3} for arm in choices}
    results, _ = build_results(choices, matrices, secondary, triplets, recall, sensitivity, bootstrap_replicates=20)
    assert results["arms"]["arm_a"]["original_level_mrstft"]["mean"] == pytest.approx(2.0)
    assert results["arms"]["arm_b"]["original_level_mrstft"]["mean"] == pytest.approx(3.0)
    assert results["pairwise_wins"]["arm_a"]["arm_b"] == 1.0
