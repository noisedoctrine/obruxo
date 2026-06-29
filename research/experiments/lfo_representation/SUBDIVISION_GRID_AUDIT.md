# Subdivision Grid Audit

This audit checks whether the fixed LFO sampling grids should be chosen only for computational convenience, or whether they should also respect common musical subdivisions.

The corpus answer is: yes, common factors matter. The active corpus is still dominated by power-of-two boundaries, but thirds/sixths/twelfths and other non-power-of-two positions are common enough that a pure `1024` grid is not musically neutral.

## Method

For every materially active routed LFO in `lfo_catalog.csv`, we parsed the serialized Vital point `x` coordinates.

We counted:

- interior boundary positions, excluding `0` and `1`;
- positive segment lengths between neighboring `x` values;
- zero-width segments, which indicate jumps/discontinuities;
- nearest rational denominators using `Fraction(x).limit_denominator(192)`;
- candidate grid coverage: whether a boundary denominator divides the grid size.

This is a heuristic audit. Odd high denominators can represent hand-drawn positions rather than intentional musical grids, so the cleanest signal is in common small denominators and denominator divisibility.

## Corpus scale

Materially active routed LFOs:

- active shapes: `14,900`
- point occurrences: `97,222`
- interior boundary occurrences: `58,434`
- positive segments: `54,655`
- zero-width jumps: `27,667`

## Most common interior boundary fractions

The top of the distribution is strongly power-of-two:

| Fraction | Occurrences |
|---|---:|
| `1/2` | 10,409 |
| `1/4` | 4,244 |
| `3/4` | 3,800 |
| `1/8` | 3,749 |
| `3/8` | 3,553 |
| `5/8` | 3,265 |
| `7/8` | 3,161 |
| sixteenth-family fractions | roughly 1,100–1,300 each |
| thirty-second-family fractions | roughly 440–480 each |

But triplet-related positions are present:

| Fraction | Occurrences |
|---|---:|
| `5/6` | 202 |
| `1/3` | 178 |
| `2/3` | 172 |
| `11/12` | 160 |
| `1/6` | 119 |
| `19/24` | 110 |
| `23/24` | 106 |
| `7/12` | 97 |
| `1/12` | 91 |
| `5/12` | 85 |

Fifth-family positions also appear:

| Fraction | Occurrences |
|---|---:|
| `2/5` | 61 |
| `1/5` | 55 |
| `3/5` | 53 |
| `4/5` | 52 |

## Top denominators

Interior boundary denominator counts:

| Denominator | Occurrences |
|---|---:|
| `8` | 13,728 |
| `2` | 10,409 |
| `16` | 9,999 |
| `4` | 8,044 |
| `32` | 7,270 |
| `12` | 433 |
| `64` | 409 |
| `24` | 377 |
| `3` | 350 |
| `6` | 321 |
| `5` | 221 |

Positive segment lengths show a similar pattern:

| Segment length | Occurrences |
|---|---:|
| `1/2` | 12,237 |
| `1/32` | 8,140 |
| `1/8` | 7,616 |
| `1/16` | 5,880 |
| `1/4` | 2,433 |
| `1/24` | 533 |
| `1/12` | 375 |
| `1/6` | 140 |
| `1/3` | 63 |

## Stratified signal

Triplet/non-power-of-two structure is not evenly distributed.

| Group | Shapes | Triplet-denominator shapes | Non-power-of-two-denominator shapes |
|---|---:|---:|---:|
| all active | 14,900 | 8.1% | 18.9% |
| stock-name hint true | 14,440 | 7.4% | 18.0% |
| stock-name hint false | 460 | 29.8% | 48.9% |
| smooth | 2,996 | 6.6% | 11.3% |
| continuous | 5,298 | 8.0% | 20.6% |
| discontinuous | 6,606 | 8.9% | 21.0% |

The custom-ish group is the loudest warning: almost half have at least one non-power-of-two rational boundary, and nearly a third have a denominator divisible by `3`.

## Candidate grid coverage

Coverage here means: among rationalized interior boundary denominator occurrences, what share have denominators that divide the candidate grid.

| Grid | Boundary denominator coverage |
|---:|---:|
| `128` | 85.6% |
| `192` | 88.2% |
| `384` | 88.3% |
| `960` | 89.1% |
| `1024` | 85.6% |
| `1536` | 88.3% |
| `1920` | 89.2% |

For custom-ish shapes only:

| Grid | Boundary denominator coverage |
|---:|---:|
| `128` / `1024` | 78.8% |
| `192` | 81.3% |
| `384` / `1536` | 81.4% |
| `960` | 82.0% |
| `1920` | 82.1% |

So the improvement is modest over the whole corpus, but more meaningful on the custom-ish subset.

## Interpretation

The current `1024` reference grid is computationally convenient and captures most power-of-two structure, but it systematically misses exact thirds, sixths, twelfths, and fifths.

This matters most for:

- gates;
- pulses;
- staircase shapes;
- discontinuities;
- user-edited/custom-ish LFOs;
- phase-factorized codebooks, where translated edge placement matters.

It matters less for smooth curves, because small boundary offsets smear less perceptually and geometrically.

## Recommendation

Use a musically composite grid in the next audit:

- feature/search grid: test `192` against current `128`;
- reference/evaluation grid: test `1920` against current `1024`;
- optionally include `384` and `1536` as middle-ground alternatives.

`192` is attractive because it preserves a compact feature grid while exactly representing common power-of-two and triplet subdivisions:

```text
192 = 64 × 3
```

`1920` is attractive as an evaluation grid because it covers powers of two through `64`, triplets/twelfths/twenty-fourths, and fifth/tenth/fifteenth/twentieth subdivisions:

```text
1920 = 2^7 × 3 × 5
```

The next experiment should not assume this improves final oracle error. It should measure:

1. whether selected codes change;
2. whether discontinuous/topology-tail RMSE changes;
3. whether phase estimates shift;
4. whether `192` features recover the same choices as `1024`/`1920` evaluation;
5. runtime/memory cost for exact XPU alignment.

