from __future__ import annotations

import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from frozen_retrieval.sensitivity import (anchored_metrics_from_distances, cosine_distance_matrices,
                                          four_anchor_distances, normalize_embedding_grid,
                                          _paired_effect_summary, _sample_distinct_original_pairs,
                                          summarize_arm, variance_decomposition)


def test_all_four_anchored_orientations_match_straightforward_reference():
    generator = torch.Generator().manual_seed(157)
    embeddings = torch.rand((3, 4, 7), generator=generator)
    unit = normalize_embedding_grid(embeddings)[0]
    within, between = cosine_distance_matrices(unit)
    preset_pairs = list(itertools.combinations(range(3), 2))
    midi_pairs = list(itertools.combinations(range(4), 2))
    left_preset = torch.tensor([pair[0] for pair in preset_pairs]).repeat_interleave(len(midi_pairs))
    right_preset = torch.tensor([pair[1] for pair in preset_pairs]).repeat_interleave(len(midi_pairs))
    left_midi = torch.tensor([pair[0] for _ in preset_pairs for pair in midi_pairs])
    right_midi = torch.tensor([pair[1] for _ in preset_pairs for pair in midi_pairs])

    actual_midi, actual_preset = four_anchor_distances(within, between, left_preset, right_preset, left_midi, right_midi)
    reference_midi, reference_preset = [], []
    for (preset_a, preset_b), (midi_1, midi_2) in itertools.product(preset_pairs, midi_pairs):
        reference_midi.extend((within[preset_a, midi_1, midi_2].item(), within[preset_a, midi_1, midi_2].item(),
                               within[preset_b, midi_1, midi_2].item(), within[preset_b, midi_1, midi_2].item()))
        reference_preset.extend((between[midi_1, preset_a, preset_b].item(), between[midi_2, preset_a, preset_b].item(),
                                 between[midi_1, preset_a, preset_b].item(), between[midi_2, preset_a, preset_b].item()))

    np.testing.assert_allclose(actual_midi.numpy().reshape(-1), reference_midi, rtol=0, atol=1e-7)
    np.testing.assert_allclose(actual_preset.numpy().reshape(-1), reference_preset, rtol=0, atol=1e-7)


def test_worked_examples_and_tie_and_zero_distance_handling():
    midi = torch.tensor([[[0.10, 0.50, 0.10, 0.50]]])
    preset = torch.tensor([[[0.50, 0.10, 0.50, 0.10]]])
    metrics = anchored_metrics_from_distances(midi, preset)
    assert metrics["ratio"][0, 0, 0].item() == pytest.approx(0.2)
    assert metrics["difference"][0, 0, 0].item() == pytest.approx(-0.4)
    assert metrics["margin"][0, 0, 0].item() == pytest.approx(2 / 3, abs=1e-6)
    assert metrics["preset_dominant"][0, 0, 0].item()
    assert metrics["ratio"][0, 0, 1].item() == pytest.approx(5.0)
    assert metrics["difference"][0, 0, 1].item() == pytest.approx(0.4)
    assert metrics["margin"][0, 0, 1].item() == pytest.approx(-2 / 3, abs=1e-6)
    assert metrics["midi_dominant"][0, 0, 1].item()

    edge = anchored_metrics_from_distances(torch.tensor([0.2, 0.0, 0.5]), torch.tensor([0.0, 0.5, 0.5]))
    assert not edge["valid_ratio"][0]
    assert torch.isnan(edge["ratio"][0])
    assert edge["preset_dominant"][1]
    assert edge["tie"][2]
    assert not edge["preset_dominant"][2] and not edge["midi_dominant"][2]


def test_bootstrap_pair_sampling_excludes_duplicate_identities_and_uses_multiplicity_weights():
    resampled = np.array([0, 1, 0, 2])  # A, B, A, C
    pair_first, pair_second = np.triu_indices(3, k=1)
    first, second = _sample_distinct_original_pairs(
        resampled, pair_first, pair_second, sample_count=20_000, generator=np.random.default_rng(157))
    assert np.all(first != second)
    pair_counts = {(left, right): int(np.count_nonzero((first == left) & (second == right)))
                   for left, right in ((0, 1), (0, 2), (1, 2))}
    frequencies = np.asarray([pair_counts[(0, 1)], pair_counts[(0, 2)], pair_counts[(1, 2)]]) / len(first)
    np.testing.assert_allclose(frequencies, [0.4, 0.4, 0.2], atol=0.015)
    with pytest.raises(ValueError, match="at least two distinct source identities"):
        _sample_distinct_original_pairs(np.array([1, 1, 1]), pair_first, pair_second, sample_count=5,
                                        generator=np.random.default_rng(158))


def test_paired_effect_uses_full_cohort_estimates_and_bootstrap_only_for_interval():
    result = _paired_effect_summary(0.314, 0.171, np.array([0.25, 0.35]), np.array([0.21, 0.28]))
    assert result["difference"] == pytest.approx(0.143)
    assert result["difference"] != pytest.approx(np.mean(np.array([0.25, 0.35]) - np.array([0.21, 0.28])))
    assert result["confidence_interval_95"] == pytest.approx([0.04075, 0.06925])


def test_normalization_counts_zero_norm_and_rejects_nonfinite_vectors():
    with pytest.raises(ValueError, match="zero_norm=1"):
        normalize_embedding_grid(torch.tensor([[[0.0, 0.0]], [[1.0, 0.0]]]))
    with pytest.raises(ValueError, match="nonfinite=1"):
        normalize_embedding_grid(torch.tensor([[[float("nan"), 0.0]], [[1.0, 0.0]]]))
    unit, counts = normalize_embedding_grid(torch.tensor([[[3.0, 4.0]]]))
    assert counts == {"nonfinite_values": 0, "zero_norm_vectors": 0}
    torch.testing.assert_close(torch.linalg.vector_norm(unit, dim=-1), torch.ones((1, 1)))


def test_variance_decomposition_has_expected_zero_and_nonzero_interaction():
    low = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [2.0, 1.0]]])
    low_result = variance_decomposition(low)
    assert low_result["ss_total"] == pytest.approx(5.0)
    assert low_result["ss_components"]["preset"] == pytest.approx(4.0)
    assert low_result["ss_components"]["midi"] == pytest.approx(1.0)
    assert low_result["ss_components"]["interaction"] == pytest.approx(0.0, abs=1e-6)
    assert sum(low_result["shares"].values()) == pytest.approx(1.0)

    high = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [3.0, 0.0]]])
    high_result = variance_decomposition(high)
    assert high_result["ss_total"] == pytest.approx(7.5)
    assert high_result["ss_components"]["preset"] == pytest.approx(6.5)
    assert high_result["ss_components"]["midi"] == pytest.approx(0.5)
    assert high_result["ss_components"]["interaction"] == pytest.approx(0.5)
    assert sum(high_result["ss_components"].values()) == pytest.approx(high_result["ss_total"])

    constant = variance_decomposition(torch.ones((2, 3, 4)))
    assert constant["status"] == "zero_total_variation"
    assert constant["shares"] == {"preset": None, "midi": None, "interaction": None}


def test_chunked_exhaustive_summary_and_bootstrap(tmp_path):
    generator = torch.Generator().manual_seed(158)
    grid = normalize_embedding_grid(torch.rand((3, 4, 6), generator=generator))[0]
    rng = np.random.default_rng(159)
    preset_draws = np.tile(np.array([0, 0, 1], dtype=np.int16), (100, 1))
    midi_draws = np.tile(np.array([0, 0, 1, 2], dtype=np.int8), (100, 1))
    point, bootstrap = summarize_arm(grid, chunk_preset_pairs=2, temp_dir=tmp_path,
                                     bootstrap_replicates=100, bootstrap_combinations=1_024,
                                     preset_draws=preset_draws, midi_draws=midi_draws, device=torch.device("cpu"))
    assert point["unordered_2x2_combinations"] == 18
    assert point["anchored_observations"] == 72
    assert point["input_shape"] == [3, 4, 6]
    assert point["preset_dominant_fraction"] + point["midi_dominant_fraction"] + point["tie_fraction"] == pytest.approx(1.0)
    assert point["nonfinite_metric_values"] == 0
    assert point["variance_decomposition"]["status"] == "ok"
    assert len(bootstrap["relative_metrics"]["mean_margin"]) == 100
    assert len(bootstrap["variance_shares"]["interaction"]) == 100
