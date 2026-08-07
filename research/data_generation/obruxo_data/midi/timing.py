from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from .events import EventKind, Performance


DEFAULT_TEMPO = 500_000


def round_fraction_half_even(value: Fraction) -> int:
    quotient, remainder = divmod(value.numerator, value.denominator)
    doubled = remainder * 2
    if doubled < value.denominator:
        return quotient
    if doubled > value.denominator:
        return quotient + 1
    return quotient if quotient % 2 == 0 else quotient + 1


@dataclass(frozen=True)
class TempoSegment:
    tick: int
    microseconds_per_beat: int
    elapsed_seconds: Fraction


@dataclass(frozen=True)
class TempoMap:
    ticks_per_beat: int
    segments: tuple[TempoSegment, ...]

    @classmethod
    def from_performance(cls, performance: Performance) -> "TempoMap":
        tempos = [event for event in performance.canonical_events() if event.kind == EventKind.TEMPO]
        by_tick: dict[int, int] = {0: DEFAULT_TEMPO}
        for event in tempos:
            by_tick[event.tick] = event.data[0]
        elapsed = Fraction(0)
        previous_tick = 0
        previous_tempo = by_tick[0]
        segments = []
        for tick, tempo in sorted(by_tick.items()):
            elapsed += Fraction((tick - previous_tick) * previous_tempo, performance.ticks_per_beat * 1_000_000)
            segments.append(TempoSegment(tick, tempo, elapsed))
            previous_tick = tick
            previous_tempo = tempo
        return cls(performance.ticks_per_beat, tuple(segments))

    def tick_to_seconds(self, tick: int) -> Fraction:
        if tick < 0:
            raise ValueError("tick must be non-negative")
        segment = self.segments[0]
        for candidate in self.segments[1:]:
            if candidate.tick > tick:
                break
            segment = candidate
        return segment.elapsed_seconds + Fraction(
            (tick - segment.tick) * segment.microseconds_per_beat,
            self.ticks_per_beat * 1_000_000,
        )

    def tick_to_sample(self, tick: int, sample_rate: int) -> int:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        return round_fraction_half_even(self.tick_to_seconds(tick) * sample_rate)

    def render_frame_count(self, end_tick: int, tail_seconds: float, sample_rate: int) -> int:
        if tail_seconds < 0:
            raise ValueError("tail_seconds must be non-negative")
        tail = Fraction(Decimal(str(tail_seconds)))
        return self.tick_to_sample(end_tick, sample_rate) + round_fraction_half_even(tail * sample_rate)
