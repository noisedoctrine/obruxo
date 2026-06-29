# Experiment 6 Plan: Codebook-Generation Approach Selection

## Purpose

Experiment 6 will choose the best approach for generating the eventual production LFO codebook. It will not yet create the final frozen production codebook.

The experiment should answer:

1. Which codebook-generation families form the useful quality/complexity frontier?
2. How much do factor-of-3 musical grids improve reconstruction and selected paths?
3. Which approaches preserve original editor nodes, not just average sampled-curve shape?
4. Which approaches are easiest for a later model to predict?
5. Which recipe should be used later to precompute the production codebook?

The output is a decision packet: tables, plots, manifests, and a findings report. We will review those together before locking the final recipe.

## Non-goals

- Do not precompute the final production codebook.
- Do not implement rendered-audio evaluation here.
- Do not fit dense reconstructions back into sparse Vital node/power state here, unless a later explicit scope expansion adds it.
- Do not select a winner by one scalar metric alone.

## Candidate families

### Direct-grid baselines

Include factor-of-3 grids, not only powers of two:

| Candidate | Reason |
|---|---|
| Grid32 | Existing compact baseline |
| Grid48 | Compact grid with factor-of-3 support |
| Grid64 | Existing strong baseline |
| Grid96 | Mid-size factor-of-3 baseline |
| Grid128 | Larger power-of-two baseline |
| Grid192 | Larger musically composite baseline |

These direct grids are not expected to be the final structured codebook, but they are necessary sanity checks.

### Structured codebook families

Use the Experiment 5 exact per-code alignment oracle as the reference. Evaluate:

- phase shared residual stacks;
- phase topology/switch residual stacks;
- additive shared + topology residual stacks;
- additive widths: K4+K4, K8+K8, K12+K12, K16+K16;
- residual depths: 2, 3, 4, and 5 where feasible;
- 32-way base as the default base family:
  - 15 provisional stock shapes;
  - 17 observed medoids.

The final report must label the stock-15 geometries as provisional.

### Sampling grids

Compare numerical convenience against musical subdivision coverage.

Feature/search grids:

- 128;
- 192;
- optionally 384 for finalist audits.

Evaluation/reference grids:

- 1024;
- 1920.

Reasoning:

```text
128 and 1024 = power-of-two grids
192 = 64 x 3
1920 = 2^7 x 3 x 5
```

Earlier subdivision analysis showed that custom-ish LFOs have meaningful non-power-of-two and triplet structure. Experiment 6 should measure whether that changes selected code paths and reconstruction quality.

## Oracle and labeling policy

Use Experiment 5's exact XPU per-code alignment as the oracle.

Routine evaluation:

- beam width 64;
- full held-out corpus.

Finalist evaluation:

- beam width 128;
- full held-out corpus.

Tail audit:

- beam width 256;
- deterministic 10% held-out subset;
- only for finalists or suspicious tail cases.

Every code must receive its own best phase/gain before code selection.

No-op residual codes must be exact and must leave the prefix unchanged.

## Reconstruction metrics

For every candidate recipe, report:

- RMSE median, mean, P90, P95, P99;
- max-error median, P95, P99;
- derivative RMSE;
- topology-specific metrics:
  - smooth;
  - continuous;
  - discontinuous;
- custom-ish subset metrics:
  - `stock_name_hint=False`;
- discontinuous/gate/pulse-heavy subset metrics.

## Threshold coverage metrics

Report the percentage of presets with full-curve RMSE below each threshold:

| Threshold | Meaning |
|---:|---|
| `1e-6` | effectively exact |
| `0.005` | below 0.5% normalized amplitude RMSE |
| `0.01` | below 1% |
| `0.02` | below 2% |
| `0.05` | below 5% |
| `0.10` | below 10% |

Report these globally and by topology/custom-ish status.

Threshold coverage must be monotonic:

```text
exact <= 0.5% <= 1% <= 2% <= 5% <= 10%
```

## Editor-node preservation metrics

Average curve RMSE can hide visible local errors at original Vital nodes. Experiment 6 should therefore report node-anchor preservation.

For each original LFO node:

1. evaluate the original curve at that node's phase;
2. evaluate the reconstructed curve at the same phase;
3. compute absolute node error.

For every preset, compute:

- maximum node error;
- mean node error;
- node error P95 within that preset.

Report:

- `% presets where no node exceeds 0.005`;
- `% presets where no node exceeds 0.01`;
- `% presets where no node exceeds 0.02`;
- `% presets where no node exceeds 0.05`;
- `% presets where no node exceeds 0.10`;
- node max-error median/P95/P99.

Duplicate-x discontinuities need two probes:

- right-hand boundary value at `x`;
- left-limit value immediately before the jump.

This metric answers:

> Does the reconstruction preserve the visible/editor-intent anchors, even when global RMSE looks good?

## Complexity metrics

Report complexity separately from quality.

For each candidate:

- dense model output dimensions;
- categorical logits;
- continuous scalar count;
- effective index bits;
- total number of stored codes;
- stored floats;
- estimated stored bytes;
- number of decoder branches;
- topology dependency;
- oracle labeling runtime;
- peak memory if available.

Do not collapse these into a single "complexity" number in the main tables.

## Model-selection diagnostics

Generate Pareto plots for:

- median RMSE vs dense outputs;
- P95 RMSE vs dense outputs;
- exact reconstruction share vs dense outputs;
- RMSE threshold coverage vs dense outputs;
- node-anchor threshold coverage vs dense outputs;
- P95 RMSE vs stored bytes;
- P95 RMSE vs effective bits;
- learnability oracle gap vs dense outputs.

Also compute pseudo-AIC/BIC-style diagnostics:

```text
AIC_dense = n * log(SSE / n) + 2 * dense_outputs
BIC_dense = n * log(SSE / n) + log(n) * dense_outputs
```

Also compute variants using:

- effective index bits;
- stored floats;
- stored bytes.

These are comparative diagnostics, not literal likelihood claims. The report must label them accordingly.

## Learnability probe

Because the later model must predict the representation, Experiment 6 should include a lightweight dense-curve learnability gate.

Inputs:

- dense sampled LFO curve;
- same train/eval author split as oracle evaluation.

Targets:

- base code;
- residual code choices;
- topology branch/code where applicable;
- residual gains;
- circular phases.

Models:

- circular-padding 1D CNN;
- parameter-matched MLP;
- direct-grid regression baselines for Grid64, Grid96, and Grid192.

Report:

- predicted reconstruction RMSE;
- oracle gap;
- code accuracy as diagnostic only;
- phase error in degrees, modulo symmetry where possible;
- gain error;
- topology errors;
- model parameter count;
- training stability and NaN checks.

Primary learnability metric:

```text
decoded predicted reconstruction error - oracle reconstruction error
```

Exact code accuracy is secondary because multiple code/phase/gain paths may reconstruct the same curve.

## Outputs

Write artifacts under:

```text
artifacts/codebook_selection/
```

Expected outputs:

- candidate recipe manifests;
- temporary evaluated codebooks with provenance;
- per-shape oracle paths;
- per-shape predicted paths for learnability probes;
- summary metrics by recipe/grid/topology/custom-ish status;
- utilization/dead-code reports;
- full-curve RMSE threshold tables;
- editor-node threshold tables;
- Pareto plots;
- pseudo-AIC/BIC diagnostic tables and plots;
- runtime/memory summaries;
- `EXPERIMENT_6_FINDINGS.md`.

## Acceptance checks

- Author train/evaluation partitions do not overlap.
- Every learned code retains observed training provenance.
- No-op residuals leave the prefix unchanged.
- Added residual layers do not worsen accepted oracle RMSE.
- Beam-128 is not worse than beam-64 on finalist audits.
- Exact XPU alignment agrees with CPU reference within Experiment 5 tolerances.
- Fixed seeds reproduce candidate summaries.
- No NaNs or infinite metrics.
- RMSE threshold coverage is monotonic.
- Node-anchor threshold coverage is monotonic.
- Duplicate-x node probes are included in node-anchor metrics.
- Pseudo-AIC/BIC diagnostics use consistent held-out `n` and SSE definitions.
- Learnability probes report oracle gap separately from raw code accuracy.

## Decision packet

The findings report should not auto-pick a winner without context. It should present:

1. efficient-frontier candidates;
2. quality/complexity curves;
3. threshold coverage curves;
4. node-preservation curves;
5. learnability curves;
6. custom-ish and discontinuous-tail comparisons;
7. a short recommendation section with 2-3 viable recipes.

The final discussion should decide which recipe to use for the later production codebook precompute.

