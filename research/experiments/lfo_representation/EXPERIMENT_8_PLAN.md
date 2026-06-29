# Experiment 8 Plan: Low-Cost Residual Scaling Screen

## Summary

Experiment 8 replaces the expanded 7B plan. It is a cheap, interaction-aware
screening experiment for the next residual representation pass.

The goal is not to pick the final best model directly. The goal is to decide
which residual sizes, gain/offset modifiers, and clipping policy are worth
carrying into a more expensive follow-up run.

This is a structured sequential design:

1. Run a coarse screen at very low resolution.
2. Analyze effect sizes and interactions.
3. Promote only the useful dimensions into the next full experiment.

Do not use Bayesian optimization, Optuna, random search, or a full factorial
grid at this stage.

## Research Questions

The Experiment 8 report should open with these questions:

- How much quality do we gain from wider residual layers?
- How much quality do we gain from more residual layers?
- Where is the useful `W x D` size band?
- Which settings are most parameter-efficient for the downstream model to
  predict?
- Do gain and/or offset help once phase alignment is always enabled?
- Does inter-layer clipping help in the phase+gain setting?
- Which settings should move into the full follow-up experiment?

Use labels like `W16D24`, where:

- `W` is residual codebook width.
- `D` is the actual number of residual layers.
- `D` is never a pair count.

## Fixed Screen Settings

- Construction recipe: `topology_balanced_common_then_tail`
- Resolution: `120`
- Beam width: `4`
- Dataset: deterministic random 1/3 train and 1/3 validation sample
- Sampling happens after the existing author split
- Sample seed: `7267`
- Default clipping: `final_only`
- Phase alignment: always enabled
- Cache keys include resolution, sample hash, `W`, `D`, modifier label, and clipping policy

Persist the shared sample at:

`artifacts/additive_finalization_8_screen/screen_sample_indices.npz`

## Modifier Labels

Phase is always enabled, so it should not be described as an optional modifier
axis.

Public label | Internal policy
--- | ---
`phase_only` | `none`
`phase_gain` | `base_gain`
`phase_offset` | `global_offset`
`phase_gain_offset` | `base_gain_global_offset`

## Size Screen

The size screen tests residual width and residual depth with:

- Modifier: `phase_only`
- Clipping: `final_only`
- Widths: `8, 16, 24, 32`
- Depths: `4, 8, 12, 16, 20, 24, 28, 32`
- Include every pair where `128 <= W * D <= 576`

Grid:

| Width \ Depth | D4 | D8 | D12 | D16 | D20 | D24 | D28 | D32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W8 |  |  |  | x | x | x | x | x |
| W16 |  | x | x | x | x | x | x | x |
| W24 |  | x | x | x | x | x |  |  |
| W32 | x | x | x | x |  |  |  |  |

Size jobs: `21`

## Modifier Screen

The modifier screen tests gain and offset only at the anchor size:

- Size: `W12D16`
- Clipping: `final_only`
- Modifiers:
  - `phase_only`
  - `phase_gain`
  - `phase_offset`
  - `phase_gain_offset`

Modifier jobs: `4`

## Clipping Screen

The clipping screen is standalone, not crossed with the full size grid.

- Size: `W12D16`
- Modifier: `phase_gain`
- Compare:
  - `final_only`
  - `intermediate_m11_final_01`

Only one additional job is needed because `W12D16 phase_gain final_only` is
already included in the modifier screen.

Clipping jobs: `1` additional

## Total Screen

- Size jobs: `21`
- Modifier jobs: `4`
- Additional clipping jobs: `1`
- Total: `26` jobs

## Clipping Policies

`final_only` is the current Experiment 7 behavior:

- Residual layers accumulate into an unclipped running state.
- Scoring and final output clip reconstruction to `[0, 1]`.

`intermediate_m11_final_01` is the new candidate behavior:

- After each residual layer, clamp the running residual state to `[-1, 1]`.
- Final reconstruction is still clipped to `[0, 1]`.

This policy must be applied consistently in training, beam evaluation, decoding,
metrics, checkpoint metadata, and cache keys.

## Memory Guard

Add a preflight memory estimator before running jobs. This is required even
though resolution `120` and the 1/3 sample should be small, because prior XPU
runs failed from memory pressure and expensive copies.

The estimator should include:

- sampled train and validation arrays
- base dictionary
- residual dictionaries
- training-stage scratch
- eval beam scratch
- cached encoding arrays
- temporary shifted/addition/candidate tensors

Estimate peak eval scratch from:

`batch_size * beam_width * W * resolution * 4 bytes * tensor_multiplier`

Estimate training scratch from:

`train_stage_batch_size * W * resolution * 4 bytes * tensor_multiplier`

Use a conservative default memory budget of `4096 MB`.

If the estimate exceeds the budget:

- refuse to run
- print the estimated peak memory
- print recommended lower `--batch-size`
- print recommended lower `--train-stage-batch-size`

The report should also record estimated memory per job so we can see whether
runtime and memory scale with `W`, `D`, or both.

## Parameter Efficiency

Parameter efficiency is a first-class result. It means the number and type of
values the downstream model must predict for each LFO, not just the stored
codebook size.

For each job, report the prediction burden:

- discrete base index
- base phase
- one residual atom index per residual layer
- one residual phase per residual layer
- one residual gain per residual layer, if the current representation requires it
- optional base gain
- optional global offset

The analysis should distinguish:

- categorical outputs: code indices the model must classify
- continuous outputs: phases, gains, and offsets the model must regress
- total dense outputs: existing scalar count used for rough comparison
- active residual outputs: non-noop residual layers actually used on validation

Use these to plot quality against prediction burden, not only quality against
stored codebook size.

## Implementation Notes

Add commands:

- `experiment8_screen`
- `experiment8_screen_status`
- `experiment8_screen_analysis`

Write clean output columns:

- `residual_width`
- `residual_depth`
- `modifier_label`
- `residual_clip_policy`
- `sample_fraction`
- `sample_hash`
- `estimated_peak_memory_mb`
- `categorical_outputs`
- `continuous_outputs`
- `predicted_outputs`
- `active_residual_outputs_median`
- `active_residual_outputs_p95`

Keep old compatibility fields readable where existing code/checkpoints need
them:

- `k`
- `d`
- `named_depth`
- `construction_strategy`
- `modifier_policy`

Hide pair-count terminology from reports, plots, commands, and progress output.
Internally map public `D` to the existing half-depth loop only where necessary.

Also fix the current grouped-run stock-loading bug in the Experiment 7/8 path so
non-quick grouped runs cannot reference `stock` before assignment.

## Analysis Outputs

`experiment8_screen_analysis` should produce:

- median RMSE heatmap over `W/D`
- P95 RMSE heatmap over `W/D`
- runtime heatmap over `W/D`
- estimated-memory heatmap over `W/D`
- storage table over `W/D`
- parameter-efficiency plots: median/P95 RMSE vs predicted outputs
- modifier comparison at `W12D16`
- clipping comparison for `W12D16 phase_gain`
- recommendation table for the next full experiment

Show median and tail metrics side by side. Tail-only plots are not sufficient.

Follow-up selection should be based on the screen evidence, not fixed numeric
thresholds.

The analysis should report:

- the `W/D` elbow region for median RMSE and P95 RMSE
- whether the same `W/D` region looks good for both central and tail behavior
- gain/offset deltas at `W12D16`, including runtime, storage, and prediction
  burden changes
- clipping deltas for `W12D16 phase_gain`, including runtime, memory, and
  prediction burden changes
- whether an accuracy gain is still attractive after accounting for what the
  model has to predict
- whether each observed effect is large relative to nearby screen-to-screen
  variation

The recommendation table should classify each dimension as:

- `carry_forward`
- `drop`
- `uncertain_needs_targeted_rerun`

Each recommendation must include the observed metric deltas that justify it.

## Commands

Start the screen in the background and open a monitor shell:

```powershell
cd research\experiments\lfo_representation
.\start_experiment8_with_monitor.cmd --beam-width 4 --align-device xpu --cache-every 1 --seed 7267 --refresh-seconds 30
```

Check status manually:

```powershell
cd research\experiments\lfo_representation
conda run -n py312 python .\experiment8.py status
```

Run analysis:

```powershell
cd research\experiments\lfo_representation
conda run -n py312 python .\experiment8.py analysis
```

## Tests

Add tests for:

- scheduler emits exactly these `26` jobs
- size screen contains exactly the `W/D` pairs where `128 <= W * D <= 576`
- 1/3 sample is deterministic and shared by all jobs
- sample hash is included in cache keys
- `D` means actual residual-layer count
- public labels use forms like `W12D16`
- `phase_only` maps to internal `none`
- clipping cache keys differ from final-only cache keys
- memory estimator rejects oversized settings
- old 7A and early 7B artifacts remain readable

Run:

```powershell
conda run -n py312 python -m unittest tests.test_alignment5 tests.test_experiment7
```
