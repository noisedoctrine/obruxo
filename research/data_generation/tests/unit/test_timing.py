from __future__ import annotations

from fractions import Fraction

from obruxo_data.midi import Performance, TempoMap, round_fraction_half_even


def test_round_fraction_uses_half_even() -> None:
    assert round_fraction_half_even(Fraction(1, 2)) == 0
    assert round_fraction_half_even(Fraction(3, 2)) == 2
    assert round_fraction_half_even(Fraction(5, 2)) == 2


def test_tempo_map_boundaries_and_samples_are_exact() -> None:
    performance = Performance(ticks_per_beat=480, bpm=None)
    performance.add_tempo(120, tick=0)
    performance.add_tempo(60, tick=480)
    timing = TempoMap.from_performance(performance)
    assert timing.tick_to_seconds(480) == Fraction(1, 2)
    assert timing.tick_to_seconds(960) == Fraction(3, 2)
    assert timing.tick_to_sample(480, 44_100) == 22_050
    assert timing.tick_to_sample(960, 44_100) == 66_150


def test_end_frame_count_includes_explicit_end_and_tail() -> None:
    performance = Performance(ticks_per_beat=480, bpm=120, end_tick=960)
    timing = TempoMap.from_performance(performance)
    assert timing.render_frame_count(960, 2.0, 44_100) == 132_300


def test_adjacent_boundaries_neither_skip_nor_duplicate_frames() -> None:
    performance = Performance(ticks_per_beat=480, bpm=120)
    timing = TempoMap.from_performance(performance)
    boundaries = [timing.tick_to_sample(tick, 44_100) for tick in (0, 240, 480, 720, 960)]
    segment_lengths = [stop - start for start, stop in zip(boundaries, boundaries[1:])]
    assert sum(segment_lengths) == boundaries[-1]
    assert segment_lengths == [11_025] * 4
