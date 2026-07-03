# Experiment 10 Plan: Subdivision And Direct Grid Audit

## Summary

Experiment 10 is reset. The previous raw point-grid version overloaded `N` as
both point count and grid quality, then accidentally tested inclusive grid slots
instead of the subdivision logic that mattered in Era 1.

Experiment 10 is a standalone corpus/grid audit. It is not a normal Era 2 model
experiment, and the shared model-runner CLI should not grow interfaces just to
support it.

The corrected experiment keeps three questions separate:

1. raw point-count coverage;
2. subdivision coverage of original x boundaries;
3. Era-1-style direct sampled-grid reconstruction.

The processed LFO corpus remains the source:

```text
raw Vital point set + occurrence count + dense 1920 reference
```

## Renderer Contract

Reference curves are rendered from the original Vital-ish point representation:

- original x/y points;
- original powers;
- original smooth flag;
- duplicate x positions preserved.

For raw-point and subdivision questions, the renderer must apply the true
segment shape. It must not silently replace every segment with linear
interpolation.

Direct sampled grids are different. A direct grid stores only y-values sampled
at fixed phases, so its decoder is periodic linear interpolation between those
y-values. That is not a claim about the source shape; it is the decoder contract
for the direct y-grid representation.

## Questions

### 1. Point-count coverage

For each point budget:

```text
24, 36, 48, 60, 72, 96, 100
```

report how many raw LFO shapes already fit:

```text
raw_num_points <= point_budget
```

This is corpus accounting only. Do not attach RMSE to over-budget shapes by
silently decimating them.

### 2. Subdivision boundary coverage

For each subdivision count:

```text
24,25,32,36,37,40,48,49,60,61,64,72,73,80,96,97,100
```

test how close each original interior x boundary is to the nearest grid point:

```text
nearest(k / subdivisions)
```

Report exact-hit coverage and nearest-distance statistics. This directly tests
the Era 1 claim that musically composite grids, especially multiples of 3, cover
real LFO boundaries better.

### 3. Direct sampled-grid reconstruction

Replicate the Era 1 direct-grid idea on the processed corpus:

```text
sample true raw curve at i / W for i in 0..W-1
store W y-values
decode by periodic linear interpolation
evaluate against true raw curve at 1920 samples
```

This answers whether a factor-of-3 direct grid can beat or match a higher
non-factor-of-3 grid, rather than merely beating a smaller grid.

The explicit comparisons are:

```text
24 vs 25
24 vs 32
36 vs 37
36 vs 40
48 vs 49
48 vs 64
60 vs 61
60 vs 64
72 vs 73
72 vs 80
96 vs 97
96 vs 100
```

A factor-of-3 grid only gets credit if it beats or matches the higher-capacity
non-factor-of-3 comparator on direct-grid P95 RMSE.

## Outputs

Experiment 10 writes:

```text
research/experiments/lfo_representation/era2/artifacts/experiment_10/subdivision_grid/
```

Expected files:

- `manifest.json`
- `point_budget_summary.csv`
- `subdivision_summary.csv`
- `direct_grid_summary.csv`
- `factor3_comparisons.csv`
- `summary.csv`
- `EXPERIMENT_10_SUBDIVISION_GRID_REPORT.md`

## Non-Goals

Experiment 10 does not:

- choose residual atoms;
- calculate model prediction head budget;
- decide Experiment 11 rows;
- test runtime topology;
- decimate raw point sets;
- optimize y-values by least squares.

The deprecated least-squares fixed-basis preflight should not be used for
`96` vs `100` decisions.

## Command

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment10_grid_audit.py
```

The command is standalone by design. It may use the processed corpus and shared
LFO parsing/rendering utilities, but it does not define the Era 2 runtime model
interface.
