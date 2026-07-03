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
- **Depth is currently the main quality lever.** Experiment 8 suggests deeper
  residual stacks are more valuable than simply widening the residual alphabet,
  but this remains a learnability question for the downstream model.
- **Width buys coverage, but with diminishing returns.** The useful question is
  marginal value per output head, not the absolute best oracle RMSE.
- **Offset is not carried forward by default.** In Experiment 8 it degraded P95
  at the tested anchor.
- **Gain is not globally settled.** It was critical in earlier phase-aware
  experiments, but in the Experiment 8 W12D16 modifier screen it tied phase-only
  unless paired with the promising clipping policy.
- **Decoder hygiene matters.** Per-layer clipping gave a cheap improvement in
  Experiment 8, so Experiment 9 tests clipping, limiting, normalization, and
  snap behavior directly.
- **XPU is the default target for new oracle runs.** CPU remains a correctness
  fallback, but the production labeling path should avoid repeated XPU/CPU
  handoff.

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

The consolidated report is
[experiments-01-05-consolidated-report.md](./reports/experiments-01-05-consolidated-report.md).

### [Experiment 6](./reports/experiment-06-findings.md)

Experiment 6 moved from "can this representation work?" to "which generation
recipe should produce the codebooks?" It introduced stronger reporting around
threshold coverage, editor-node preservation, complexity accounting, and
candidate construction recipes.

Carry forward:

- compare structured codebooks against direct grids, not just against stock
  shapes;
- evaluate median and tail behavior separately;
- track model-output cost and decoder storage separately;
- treat editor-node preservation as a distinct concern from sampled-curve RMSE.

Report:
[experiment-06-findings.md](./reports/experiment-06-findings.md).

### [Experiment 7A](./reports/experiment-07a-findings.md)

Experiment 7A tested construction-policy variants and led us away from
frequency-first as the carry-forward policy. The important communication cleanup
from this phase was nomenclature: count real residual layers, not named bundles.

Carry forward:

- use `topology_balanced_common_then_tail` as the construction recipe to test
  next;
- do not let frequency-first dominate plots or conclusions when it is plainly a
  poor policy;
- keep "shared" and "topology" as codebook-construction details, not hidden
  model-facing layer types;
- evaluate modifier effects only through controlled contrasts.

Report:
[experiment-07a-findings.md](./reports/experiment-07a-findings.md).

### Experiment 8

Experiment 8 replaced the expanded 7B idea with a cheap, interaction-aware
screen. It fixed the current planning baseline:

- use `topology_balanced_common_then_tail`;
- phase is always enabled;
- test `W` and `D` as output-head tradeoffs;
- test gain/offset separately from phase;
- test cheap clipping policies without broad grid explosion.

Carry forward:

- depth appears to be the dominant quality lever among the tested size knobs;
- width has value, but its marginal value must be judged against the added
  categorical output burden;
- gain and offset have equal structural cost but different empirical value;
- per-layer clipping is a zero-output-cost decoder policy worth isolating;
- no single oracle "best" config should be treated as final, because deeper
  oracle stacks can always keep improving.

Report:
[experiment-08-findings.md](./reports/experiment-08-findings.md).

### Experiment 9

Experiment 9 is the next quick screen. Its main decoder/modifier jobs stay fixed
at `W8D16`, and it adds a small equivalent-output-budget check for narrow
residual widths rather than expanding the full size grid:

- where gain/offset should apply: base, residuals, or both;
- whether residual range normalization makes those scalars useful;
- which synth-style decoder hygiene policy is the best cheap baseline;
- whether data-derived snap anchors help final output cheaply.
- whether W4/W6 narrow-deep stacks are more parameter-efficient than reused W8
  reference budgets from Experiment 8.

Experiment 9 should record both train and validation metrics, because worse
validation under more degrees of freedom may reflect the construction/decoder
objective rather than normal overfitting. Alongside median and P95 RMSE, it
tracks `perfect_lfo_rate_eps_0.02`: the share of LFOs whose entire sampled curve
has `max_abs_error <= 0.02`.

Report:
[experiment-09-findings.md](./reports/experiment-09-findings.md).

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

## Output-Head Accounting

The model-facing output burden is not the number of serialized fields. A
categorical code index means the model emits a softmax over that codebook.

Use:

```text
head_outputs = 32 + sum(layer_codebook_size) + (D + 1) * (I_phase + I_gain + I_offset)
```

For shared residual layers, `layer_codebook_size = W`. For topology-conditioned
residual layers, deployment should flatten the topology-specific dictionaries
and use `layer_codebook_size = 3W`; the model predicts one categorical code, not
a topology label plus a second code.

In the current experiments phase is assumed enabled:

```text
phase-only baseline = 32 + sum(layer_codebook_size) + (D + 1)
optional gain       = +(D + 1)
optional offset     = +(D + 1)
```

Cost is analytic. Quality value must be measured with controlled empirical
contrasts.

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

- [experiment-01-findings.md](./reports/experiment-01-findings.md): initial
  corpus and Experiment 1 oracle baselines.
- [experiment-02-findings.md](./reports/experiment-02-findings.md): stacked
  categorical residual codebooks.
- [experiment-03-findings.md](./reports/experiment-03-findings.md):
  frequency-first residual peeling.
- [experiment-04-findings.md](./reports/experiment-04-findings.md): phase
  factorization and mixed dictionaries.
- [experiment-05-findings.md](./reports/experiment-05-findings.md): exact
  phase-alignment oracle.
- [experiment-06-findings.md](./reports/experiment-06-findings.md): codebook
  selection and parameter-efficiency screen.
- [experiment-07a-findings.md](./reports/experiment-07a-findings.md):
  construction-policy and modifier screen.
- [experiment-08-findings.md](./reports/experiment-08-findings.md): cheap
  size/modifier/clipping screen.
- [experiment-09-findings.md](./reports/experiment-09-findings.md): quick
  decoder/modifier hygiene screen.
- [experiments-01-05-consolidated-report.md](./reports/experiments-01-05-consolidated-report.md):
  design synthesis through Experiment 5.
- [experiment6_codebook_selection/](./experiment6_codebook_selection/):
  Experiment 6 plan and runner notes.
- `EXPERIMENT_8_PLAN.md`: cheap size/modifier/clipping screen plan.

Rendered report images live under [reports/images/](./reports/images/), with
one subfolder per experiment. The raw run outputs still live under `artifacts/`,
which is ignored by git.
