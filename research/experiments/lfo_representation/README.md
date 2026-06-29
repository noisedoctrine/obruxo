# LFO Representation Research

This directory is for choosing a compact, model-predictable representation of
Vital-style custom LFO curves.

The central question is:

> What should the downstream model emit so that it can reconstruct useful
> variable-length LFO shapes without directly predicting raw editor geometry?

The work here is still representation research. Most experiments are oracle
encoders: they measure how good a compact code can be when selected perfectly.
That is the right ceiling test before training an audio-to-preset model.

## Current State

The current working direction is a structured residual representation:

```text
base:
  32-way categorical choice
  = 15 provisional stock shapes + 1 no-op + 16 observed medoids
  + continuous circular phase

residual stack:
  W-way categorical residual atom per residual layer
  + continuous circular phase per residual layer
  + optional scalar families under test

decoder:
  additive reconstruction
  explicit no-op behavior
  decoder clipping policy under test
  final output clipped to [0, 1]
```

`W` is residual codebook width. `D` is the actual number of residual layers.
Depth is not counted in hidden pairs or named-layer bundles.

The current construction recipe to carry forward is
`topology_balanced_common_then_tail`. Topology is not a separate model-facing
layer concept; it is a residual-codebook construction policy that tries to give
common structure and topology-specific tail structure fair coverage.

## Current Design Decisions

- **Stock-only is rejected.** The nearest-stock oracle is far too weak.
- **Direct grids remain the baseline.** They are simple and strong enough that
  every structured code must justify itself against them.
- **Residual stacking is the main representation family.** It gives progressive
  refinement and natural no-op paths.
- **Continuous phase is mandatory.** Low-cardinality phase bins were not enough;
  phase factorization is one of the biggest wins in the sequence.
- **No-op codes are part of the design, not a convenience.** They let solved
  shapes stop accumulating residual noise.
- **Depth is currently the main quality lever.** Experiment 8 strongly favored
  narrow/deep stacks over wide/shallow stacks at similar output-head cost.
- **Offset is not carried forward by default.** In Experiment 8 it degraded P95
  at the tested anchor.
- **Gain is not globally settled.** It was critical in earlier phase-aware
  experiments, but in the Experiment 8 W12D16 modifier screen it tied phase-only
  unless paired with the promising clipping policy.
- **Per-layer clipping is promising.** Experiment 8 showed a zero-output-cost
  improvement for intermediate `[-1, 1]` clipping in the phase+gain anchor.
- **XPU is the default target for new oracle runs.** CPU remains a correctness
  fallback, but the production labeling path should avoid repeated XPU/CPU
  handoff.

## Current Best References

From Experiment 8's cheap 120-point, beam-4 screen:

| Role | Config | Head outputs | Median RMSE | P95 RMSE | Notes |
|---|---:|---:|---:|---:|---|
| Compact deep reference | `W8D32 phase_only final_only` | 321 | 0.000920 | 0.012003 | Strong parameter-efficiency point |
| Best quality in screen | `W16D32 phase_only final_only` | 577 | 0.000651 | 0.010222 | Best median and P95 |
| Modifier anchor | `W12D16 phase_only final_only` | 241 | 0.001958 | 0.026819 | Gain tied, offset worsened P95 |
| Clipping anchor | `W12D16 phase_gain intermediate_m11_final_01` | 258 | 0.001759 | 0.023425 | Best cheap decoder-policy signal |

Interpretation:

- `W8D32` is the parameter-efficient deep reference.
- `W16D32` is the current quality reference.
- The next useful work is not just "more width"; it is understanding decoder
  hygiene, normalization, gain/offset scope, and whether deeper narrow stacks
  stay learnable.

## Output-Head Accounting

The model-facing output burden is not the number of serialized fields. A
categorical code index means the model emits a softmax over that codebook.

Use:

```text
head_outputs = 32 + W*D + (D + 1) * (I_phase + I_gain + I_offset)
```

where:

- `32` is the base categorical choice;
- `W*D` is the total residual categorical vocabulary emitted across layers;
- each enabled scalar family costs one scalar for the base and one per residual
  layer.

In the current experiments phase is assumed enabled:

```text
phase-only baseline = 33 + D(W + 1)
optional gain       = +(D + 1)
optional offset     = +(D + 1)
```

Cost is analytic. Quality value must be measured with controlled empirical
contrasts.

## Evidence Trail

### Experiments 1-5

Experiments 1-5 established the representation family:

- The corpus has about 14.9k materially active routed LFO instances.
- Stock-only reconstruction failed even under an oracle.
- Direct grids are strong and remain the sanity-check baseline.
- Pure categorical residual stacks improved over stock-only but needed scalar
  factors.
- Frequency-first construction and exact no-op branches improved residual use.
- Phase factorization was a major win; phase should be continuous.
- Exact per-code phase/gain alignment is necessary before comparing candidate
  atoms.
- XPU exact alignment matched CPU numerically for the tested path and was much
  faster.

The consolidated report is `EXPERIMENTS_1_5_CONSOLIDATED_REPORT.md`.

### Experiment 6

Experiment 6 moved from "can this representation work?" to "which generation
recipe should produce the codebooks?" It introduced stronger reporting around
threshold coverage, editor-node preservation, complexity accounting, and
candidate construction recipes.

Details live in `experiment6_codebook_selection/`.

### Experiment 7A

Experiment 7A tested construction-policy variants and led us away from
frequency-first as the carry-forward policy. The important communication cleanup
from this phase was nomenclature: count real residual layers, not named bundles.

### Experiment 8

Experiment 8 replaced the expanded 7B idea with a cheap, interaction-aware
screen. It fixed the current planning baseline:

- use `topology_balanced_common_then_tail`;
- phase is always enabled;
- test `W` and `D` as output-head tradeoffs;
- test gain/offset separately from phase;
- test cheap clipping policies without broad grid explosion.

The full generated report is
`artifacts/additive_finalization_8_screen/EXPERIMENT_8_FINDINGS.md`.

### Experiment 9

Experiment 9 is the next quick fixed-budget screen at `W8D16`. It is meant to
answer questions that Experiment 8 exposed rather than expanding the size grid:

- where gain/offset should apply: base, residuals, or both;
- whether residual range normalization makes those scalars useful;
- which synth-style decoder hygiene policy is the best cheap baseline;
- whether data-derived snap anchors help final output cheaply.

Experiment 9 should record both train and validation metrics, because worse
validation under more degrees of freedom may reflect the construction/decoder
objective rather than normal overfitting.

## Open Questions

- **Training loss vs validation loss.** We need train-side reconstruction metrics
  to tell whether a richer decoder is failing to fit even the training targets
  or merely failing to generalize.
- **Residual normalization semantics.** Range-normalized residual targets may
  make gain/offset meaningful, but the clipping order must stay consistent:
  current clipped prefix, compute residual, optionally normalize, select shifted
  code, denormalize correction, add correction, apply decoder hygiene.
- **Clipping policy.** `[-1, 1]` intermediate clipping helped once. Experiment 9
  checks whether that was the right policy or just one member of a broader
  decoder-hygiene family.
- **Snap anchors.** Synth workflows often imply values near rails, halves,
  quarters, thirds, and two-thirds may be intentional. Experiment 9 tests a
  small final-output-only snap screen with data-derived snap Schwarzschild
  radii.
- **Canonical stock shapes.** The 15 stock entries are still provisional corpus
  modes and should eventually be replaced with controlled Vital saves.
- **Sparse Vital refit.** The current metrics mostly score sampled curves. We
  still need to prove decoded curves can become valid editable Vital point/power
  state without losing the benefit.
- **Rendered-audio relevance.** Geometry RMSE is not the final perceptual
  metric. Destination, modulation amount, rate, and context determine audibility.
- **Learnability.** Oracle paths can be multi-modal. A learned predictor should
  be judged by decoded reconstruction and rendered behavior, not only exact code
  identity.
- **Grid resolution.** Cheap screens use low resolution deliberately. Final
  audits may need musically composite grids, especially for triplet-like or
  discontinuous custom shapes.

## Operations

Run commands from this directory:

```powershell
cd research\experiments\lfo_representation
```

Use the existing `py312` conda environment:

```powershell
conda run -n py312 python run_experiment.py --help
```

### Experiment 8

Run:

```powershell
.\start_experiment8_with_monitor.cmd --beam-width 4 --align-device xpu --cache-every 1 --seed 7267 --refresh-seconds 30
```

Status:

```powershell
conda run -n py312 python .\experiment8.py status
```

Analysis/report:

```powershell
conda run -n py312 python .\experiment8.py analysis
```

Monitor only:

```powershell
.\open_experiment8_monitor.cmd -RefreshSeconds 30
```

### Experiment 9

Run:

```powershell
.\start_experiment9_with_monitor.cmd --beam-width 4 --align-device xpu --cache-every 1 --seed 7267 --refresh-seconds 30
```

Status:

```powershell
conda run -n py312 python .\experiment9.py status
```

Analysis/report:

```powershell
conda run -n py312 python .\experiment9.py analysis
```

Monitor only:

```powershell
.\open_experiment9_monitor.cmd -RefreshSeconds 30
```

### Legacy Experiments

The original runner remains available for older stages:

```powershell
conda run -n py312 python run_experiment.py catalog
conda run -n py312 python run_experiment.py codebook
conda run -n py312 python run_experiment.py benchmark
conda run -n py312 python run_experiment.py experiment2
conda run -n py312 python run_experiment.py experiment3
conda run -n py312 python run_experiment.py experiment4
conda run -n py312 python run_experiment.py experiment5
```

Generated files are written under `artifacts/`. That directory is intentionally
ignored by git.

## Report Index

- `FINDINGS.md`: initial corpus and Experiment 1 oracle baselines.
- `EXPERIMENT_2_FINDINGS.md`: stacked categorical residual codebooks.
- `EXPERIMENT_3_FINDINGS.md`: frequency-first residual peeling.
- `EXPERIMENT_4_FINDINGS.md`: phase factorization and mixed dictionaries.
- `EXPERIMENT_5_FINDINGS.md`: exact phase-alignment oracle.
- `EXPERIMENTS_1_5_CONSOLIDATED_REPORT.md`: design synthesis through
  Experiment 5.
- `experiment6_codebook_selection/`: Experiment 6 plan and runner notes.
- `EXPERIMENT_8_PLAN.md`: cheap size/modifier/clipping screen plan.
- `artifacts/additive_finalization_8_screen/EXPERIMENT_8_FINDINGS.md`: generated
  Experiment 8 report.
- `artifacts/additive_finalization_9_screen/EXPERIMENT_9_FINDINGS.md`: generated
  Experiment 9 report once jobs complete.
