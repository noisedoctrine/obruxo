# Experiment 10 Plan: Control-Point X Grid Audit

## Summary

Experiment 10 is a standalone corpus/grid audit. It is not a normal Era 2 model
experiment, and it should not shape the shared model-runner interface.

The central question is now:

```text
How well can a fixed x-grid place the original ordered Vital LFO control-point x positions?
```

This is not a curve reconstruction test. It ignores y values and does not draw
straight lines, Beziers, power curves, or any other segment between predicted
points.

## Naming Contract

Use subdivision counts as the public experiment variable:

```text
subdivision_count = number of x-grid subdivisions
control_point_count = subdivision_count + 1
grid_point_count = implementation field name for control_point_count
```

The default sweep covers every even `subdivision_count` from 8 through 100.
This intentionally includes `subdivision_count = 100`, which implies
`control_point_count = 101`. That row exceeds Vital's 100-point limit by one
control point, but it is useful for the requested subdivision-based audit.

`W` is reserved for residual-layer atom choices in Era 2 model experiments.
Experiment 10 must not use `W` for grid size.

Factor language applies to the inferred subdivision count:

```text
subdivision_count = 96
control_point_count = 97
96 is divisible by 2 and 3
```

## Questions

### 1. Source Point-Count Frequency

Vital LFOs already have an upper bound of 100 points, so "coverage under 100"
is not useful. The useful corpus fact is the frequency by source point count,
reported two ways:

- deduplicated LFO corpus: each unique raw LFO shape counts once;
- LFO corpus: each unique shape is weighted by its occurrence count.

Output:

```text
point_count_frequency.csv
plots/experiment10_point_count_frequency.png
```

### 2. Control-Point X Placement

For each `subdivision_count`, infer:

```text
control_point_count = subdivision_count + 1
```

Then score each true ordered control point:

```text
x_pred_i = nearest fixed grid point to x_true_i
error_i = abs(x_pred_i - x_true_i)
```

For uniform grids, the fixed grid points are:

```text
k / subdivision_count
```

Repeated grid points are allowed because Vital LFOs can use duplicate x
positions for discontinuities.

Report all-point and interior-only statistics. Interior-only stats matter most
because endpoints at 0 and 1 are usually fixed and can otherwise make the grid
look artificially good.

Also report the fraction of LFOs whose maximum control-point x error is at most
0.001:

```text
max_i(abs(x_pred_i - x_true_i)) <= 0.001
```

Report that fraction two ways:

- deduplicated LFO corpus;
- occurrence-weighted LFO corpus.

Output:

```text
control_point_x_summary.csv
control_point_x_lattice_frequency.csv
plots/experiment10_control_point_x_lattice_frequency.png
plots/experiment10_control_point_x_p95.png
plots/experiment10_control_point_x_median.png
plots/experiment10_lfo_pass_rate_0p001.png
plots/experiment10_nonuniform_delta.png
plots/experiment10_factor3_checks.png
```

The markdown report should be findings-first. It should start from the observed
periodicity in the plots, then use the x-lattice frequency graph to explain why
the corpus behaves as mostly dyadic rather than generally factor-3 dominated.

## Decision Carried Forward

Experiment 10 settles the Era 2 default LFO x-grid:

```text
subdivision_count = 96
control_point_count = 97
```

This fixed uniform grid stays under Vital's 100-point limit and performs well
because it matches the corpus's dominant dyadic x-position structure while also
supporting factor-3 subdivisions. Future model experiments should not spend
model prediction head budget on x-position prediction or variable grid spacing.

### 3. Global Non-Uniform Grid

Add fixed global non-uniform x-grids as a baseline against uniform grids. These
grids are learned offline from corpus control-point x positions and stored in
the decoder. The deployed model would still predict only a grid slot; it would
not predict the grid locations.

Report two learned global grids:

- `global_quantile / deduplicated`: learned from unique LFO shapes with equal
  point weight;
- `global_quantile / occurrence_weighted`: learned from point positions weighted
  by LFO occurrence count.

Output:

```text
global_nonuniform_grids.json
plots/experiment10_nonuniform_delta.png
```

### 4. Factor Checks

Compare grid point counts whose inferred subdivision count is divisible by 3
against higher grid point counts whose inferred subdivision count is not
divisible by 3.

Example:

```text
25 grid points -> 24 subdivisions
26 grid points -> 25 subdivisions
33 grid points -> 32 subdivisions
```

The comparison should be read as:

```text
Does a factor-3 subdivision grid beat or match a higher point-count non-factor-3 grid?
```

Output:

```text
factor3_grid_point_comparisons.csv
```

## Outputs

Experiment 10 writes data artifacts to:

```text
research/experiments/lfo_representation/era2/artifacts/experiment_10/control_point_x_grid/
```

Expected artifact files:

- `manifest.json`
- `point_count_frequency.csv`
- `control_point_x_lattice_frequency.csv`
- `control_point_x_summary.csv`
- `global_nonuniform_grids.json`
- `factor3_grid_point_comparisons.csv`
- `summary.csv`
- `plots/experiment10_point_count_frequency.png`
- `plots/experiment10_control_point_x_lattice_frequency.png`
- `plots/experiment10_control_point_x_median.png`
- `plots/experiment10_control_point_x_p95.png`
- `plots/experiment10_lfo_pass_rate_0p001.png`
- `plots/experiment10_nonuniform_delta.png`
- `plots/experiment10_factor3_checks.png`

Experiment 10 writes the markdown report to:

```text
research/experiments/lfo_representation/era2/reports/EXPERIMENT_10_CONTROL_POINT_X_GRID_REPORT.md
```

The report uses local image copies under:

```text
research/experiments/lfo_representation/era2/reports/images/experiment_10/
```

## Non-Goals

Experiment 10 does not:

- choose residual atoms;
- calculate model prediction head budget;
- test runtime topology;
- predict y values;
- render curves between predicted points;
- evaluate sampled y-grid reconstruction;
- test per-LFO adaptive grids.

Per-LFO adaptive grids are a different representation family because they
require additional runtime predictions.

## Command

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment10_grid_audit.py
```
