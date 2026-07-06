# Experiment 12: PhaseAndResidualGain Screening Read

## Main Findings

This report filters Experiment 12 to the `PhaseAndResidualGain` rows. `IndicesOnly` remains useful as a diagnostic baseline in the full report, but it is poor enough that it visually crowds the decision read for Experiment 13. Under the default construction policy, `PhaseAndResidualGain` reaches median RMSE `0.0409`, validation P95 `0.0580`, and node-max P95 `0.1830` at `193` model prediction head outputs.

Construction policy is the most important process-like variable in the run. `CommonCaseRepair` is the median and strict-perfect outlier: with `PhaseAndResidualGain`, it reaches median RMSE `0.0034` and strict perfect-LFO rate `0.1651`, but its P95 is `0.1280`. `FinishRepairRescue` is the cleaner balanced construction candidate: median `0.0087`, strict perfect rate `0.1277`, P95 `0.0511`, and node-max P95 `0.1500`.

End-of-layer normalization is a real free decoder-policy lever. `LayerClip0To1` has the best validation P95 in the run at `0.0509`, while `LayerCenterPreserveClip` is essentially tied on P95 at `0.0510` and has the best node-max P95 among the layer-normalization rows at `0.1563`. The soft-clip variants, bounded residual step, and overshoot-penalty/no-clip variant are weaker in this screen.

`no_damage_policy` and duplicate suppression are mostly flat. They do not move quality enough to justify treating them as primary Experiment 13 axes unless the grid has spare room. Duplicate suppression is especially weak here: quality is identical to the default row while construction time is higher.

The report contains `36` rows from the current `36`-row `PhaseAndResidualGainOnly` view. Every row keeps `W=8`, `D=16`, `control_point_count=97`, flat-categorical per-residual-layer addressing, and one required `NoOpAtom` per residual layer. This is still a screening read, not an automatic winner selection: median RMSE, strict perfect-LFO rate, P95 RMSE, and node-max P95 disagree in meaningful ways.

## Why This Happens

The scalar result is expected. Residual atoms need phase and scale invariance: a useful residual shape may be shifted in cycle phase or appear at a different amplitude. `IndicesOnly` can only choose an atom slot, so it often needs later layers to compensate for a phase or amplitude mismatch. `PhaseAndResidualGain` gives the decoder the missing alignment degrees of freedom directly.

`NoOpAtom` changes how atom usage should be read. Explicit no-op handling appears correctly applied in the encoder: atom index `0` keeps the prefix unchanged and resets phase/gain to `0`. The apparent no-op collapse under `PhaseAndResidualGain` is probably not a broken no-op path. An active atom can receive a near-zero optimized gain, which behaves like an implicit no-op while not counting as `NoOpAtom`. For that reason the report now tracks both explicit no-op usage and effective no-op usage, where effective no-op means explicit no-op or `abs(gain) <= 1e-4`.

The construction-policy split is a finish-vs-repair tradeoff. `CommonCaseRepair` spends atoms on residuals that many LFOs share, so it strongly improves the median and creates many exact reconstructions. It leaves some hard cases under-repaired, which is why its P95 stays high. `FinishRepairRescue` mixes finishing behavior with broader repair and later hard-case rescue, so it gives up some perfect-rate upside for a much better tail.

`utility_candidate_budget` is an offline construction knob. Each residual layer needs seven active atoms plus the required no-op. For each active atom slot, the constructor does not score every possible residual in the corpus as a candidate atom. `CandidateBudgetN` is the number of candidate residual shapes it pulls into that scoring round before choosing the next atom. It changes oracle construction work and atom quality; it is not model prediction head budget and does not change deployed runtime outputs.

Layer clipping helps because residual additions can overshoot the legal LFO y range before the final decoder clip. Hard clipping after each layer can stop overshoot from propagating through later residual choices. This is a decoder/free policy: it changes deterministic reconstruction behavior and adds zero model prediction head outputs. It should not be confused with oracle/offline construction work or with deployed runtime inputs.

Soft clipping is different from hard clipping because the sigmoid-style transform is not identity-preserving inside the valid range. It slightly compresses values even when they were already good, which is hostile to perfect reconstruction. That explains why `LayerSoftClip0To1` and `LayerSoftClipNeg0p1To1p1` show high no-op/dead-usage behavior and poor perfect reconstruction despite reducing boundary violations.

## Experiment 13 Candidate Read

This section is manual selection guidance, not an automatic ranking. The right Experiment 13 grid should preserve candidates that win different co-primary metrics.

- `path_search_policy`: keep both `Beam4Path` and `Beam8Path` unless grid size must shrink. `Beam8Path` is modestly better on P95 (`0.0567` vs `0.0580`) but costs more encoding time.
- `construction_policy`: shortlist `FinishRepairRescue`, `CommonCaseRepair`, and `FamilyBalancedRepair` or `ShapeClusterRepair`. `FinishRepairRescue` is the balanced choice; `CommonCaseRepair` is the median/perfect-rate stress test.
- `utility_candidate_budget`: shortlist `CandidateBudget48`, `CandidateBudget24`, and `CandidateBudget12`. `CandidateBudget8` is cheap, but under `PhaseAndResidualGain` it is less compelling on tail quality.
- `layer_normalization_policy`: shortlist `LayerClip0To1`, `LayerCenterPreserveClip`, and `LayerClipNeg0p1To1p1`. Treat soft clips, `BoundedResidualStep`, and `OvershootPenaltyNoClip` as weak unless a later run gives them a different role.
- `no_damage_policy`: if keeping three values, use `NoDamageOff`, `LateLayerNoDamage`, and `LateLayerNoDamageAndPerfectLocking`. The variable looks low-impact in this run.
- `atom_preprocessing_policy`: shortlist `EnergyNormalizedAtoms`, `RawAtoms`, and `CenteredEnergyNormalizedAtoms`. Keep the warning that centered normalization hurts `IndicesOnly` badly.
- `duplicate_suppression_policy`: keep both only if Experiment 13 budget allows. Current quality metrics are identical, while duplicate suppression costs more construction time.

## Independent Variable Chapters

### Path Search Policy

This family asks whether the decoder should keep a wider path beam while choosing atom sequences. The family plot shows `Beam8Path` buys a small P95 improvement over `Beam4Path`, but the duration row shows the expected encoding-time cost. It is worth keeping both only if Experiment 13 can afford the extra rows.

![Path Search Policy metrics and diagnostics](./images/experiment_12_phase_gain/experiment12_path_search_policy_family.png)

### Construction Policy

This is the most important process-like family. The family plot shows why there is no single automatic winner: `CommonCaseRepair` dominates median and strict-perfect behavior, while `FinishRepairRescue` gives the better balanced tail and node-max result. This family should get real width in Experiment 13.

![Construction Policy metrics and diagnostics](./images/experiment_12_phase_gain/experiment12_construction_policy_family.png)

### Utility Candidate Budget

In plain language, this is how many residual-shape candidates the offline constructor bothers to score before choosing the next atom. The family plot shows diminishing returns rather than a clean monotonic curve. `CandidateBudget48` is the best quality candidate under `PhaseAndResidualGain`, but `CandidateBudget24` and `CandidateBudget12` remain useful cost controls.

![Utility Candidate Budget metrics and diagnostics](./images/experiment_12_phase_gain/experiment12_utility_candidate_budget_family.png)

### Layer Normalization Policy

This family tests decoder/free end-of-layer state policies. The family plot shows hard clipping is genuinely useful for the tail: `LayerClip0To1` and `LayerCenterPreserveClip` are the clean candidates. The soft-clipping rows are visibly poor because the transform compresses already-valid values instead of preserving exact reconstructions.

![Layer Normalization Policy metrics and diagnostics](./images/experiment_12_phase_gain/experiment12_layer_normalization_policy_family.png)

### No Damage Policy

This family tests whether late layers should be prevented from making an already-good reconstruction worse. The family plot is mostly flat, which means the required `NoOpAtom` already handles much of the safety behavior. Keep this axis small in Experiment 13.

![No Damage Policy metrics and diagnostics](./images/experiment_12_phase_gain/experiment12_no_damage_policy_family.png)

### Atom Preprocessing Policy

This family tests whether residual atoms should be normalized before being put into layer dictionaries. `EnergyNormalizedAtoms` is a plausible keeper because it is competitive under `PhaseAndResidualGain`; `CenteredEnergyNormalizedAtoms` is riskier because the `IndicesOnly` plot shows a clear degradation.

![Atom Preprocessing Policy metrics and diagnostics](./images/experiment_12_phase_gain/experiment12_atom_preprocessing_policy_family.png)

### Duplicate Suppression Policy

This family tests whether phase/scale-near-duplicate atoms should be removed during construction. The quality plot is essentially unchanged, while the diagnostic plot shows extra construction cost. Keep both only if Experiment 13 has room; otherwise this is a lower-priority axis.

![Duplicate Suppression Policy metrics and diagnostics](./images/experiment_12_phase_gain/experiment12_duplicate_suppression_policy_family.png)

## Global Plot Notes

Lower is better for validation P95, validation median, max-point error, overshoot, and runtime. Higher is better for strict perfect-LFO rate.

### Validation P95 By Row

![Validation P95](./images/experiment_12_phase_gain/experiment12_validation_p95_by_row.png)

The x-axis is the screened `PhaseAndResidualGain` row; the y-axis is validation P95 RMSE, where lower is better. With the weak `IndicesOnly` rows removed, the important structure is the spread among process variables: hard layer clipping and balanced construction sit at the low end, while finish-only and soft-clipping policies remain visibly weak.

### Validation Median By Row

![Validation median](./images/experiment_12_phase_gain/experiment12_validation_median_by_row.png)

The x-axis is the screened row; the y-axis is validation median RMSE, where lower is better. The plot has a small cluster near zero plus a broader band of ordinary rows. `CommonCaseRepair` creates the clearest near-zero median bar, while finish-only and soft-clip rows remain visibly high. This is why median remains co-primary instead of being folded into P95.

### P95 Vs Model Prediction Head Budget

![P95 vs head outputs](./images/experiment_12_phase_gain/experiment12_p95_vs_head_outputs.png)

The x-axis is deployed model prediction head budget; the y-axis is validation P95 RMSE. All rows in this report use the same `193`-head scalar schema, so the vertical spread is the point: process variables and decoder/free policies still move quality even when model prediction head budget is fixed.

### Residual Gain Usage

![Scalar usage](./images/experiment_12_phase_gain/experiment12_scalar_usage.png)

The x-axis is the screened row; the y-axis is residual-gain absolute P95. Higher is not automatically better here: it means the optimized residual scalar is being used more strongly. Most `PhaseAndResidualGain` rows form a moderate band, while a few construction/normalization rows spike close to the gain bounds. Those spikes are diagnostics for aggressive correction or overshoot compensation, not quality wins by themselves.

### Atom Usage

![Atom usage](./images/experiment_12_phase_gain/experiment12_atom_usage.png)

The x-axis is the screened row; the y-axis is median residual-layer dead-atom rate. Lower means more dictionary slots are used, but this is diagnostic rather than a direct objective. The tallest spikes line up with policies that collapse much of the residual ladder into no-op or unused active atoms, especially finish-heavy and soft/bounded normalization variants. That pattern helps explain why some policies look safe but do not repair the tail well.

### Runtime

![Runtime](./images/experiment_12_phase_gain/experiment12_runtime_by_row.png)

The x-axis is the screened row; the y-axis is row elapsed seconds, where lower is faster. Most rows sit in a broad middle band, but a few construction-heavy rows stand out as clear runtime outliers. This is oracle construction and encoding runtime on the current implementation, not deployed model runtime. It matters for experiment velocity and for sizing Experiment 13, but not for the model prediction head budget.

## Grouped Evidence Tables

Co-primary metrics: `validation_median_rmse`, `validation_strict_perfect_lfo_rate`, `validation_p95_rmse`, and `validation_node_max_error_p95`. The tables are grouped by screened variable. This report shows only `PhaseAndResidualGain` rows.

### `path_search_policy`

| Value | Median RMSE | Perfect Rate | P95 RMSE | Node Max P95 | Construct s | Encode s | NoOp Median | Effective NoOp | Overshoot Rate |
|---|---|---|---|---|---|---|---|---|---|
| `Beam4Path` | 0.0409 | 0.0087 | 0.0580 | 0.1830 | 22.5015 | 6.0689 | 0.0087 | n/a | 0.1779 |
| `Beam8Path` | 0.0406 | 0.0087 | 0.0567 | 0.1970 | 17.5831 | 11.8806 | 0.0087 | n/a | 0.1772 |

### `construction_policy`

| Value | Median RMSE | Perfect Rate | P95 RMSE | Node Max P95 | Construct s | Encode s | NoOp Median | Effective NoOp | Overshoot Rate |
|---|---|---|---|---|---|---|---|---|---|
| `BestOverallRepair` | 0.0409 | 0.0087 | 0.0580 | 0.1830 | 17.7621 | 6.4351 | 0.0087 | n/a | 0.1779 |
| `CommonCaseRepair` | 0.0034 | 0.1651 | 0.1280 | 0.3891 | 19.5441 | 7.2809 | 0.0109 | n/a | 0.1297 |
| `HardCaseRepair` | 0.0451 | 0.0087 | 0.0606 | 0.1837 | 19.2506 | 8.5541 | 0.0087 | n/a | 0.1790 |
| `FamilyBalancedRepair` | 0.0406 | 0.0044 | 0.0546 | 0.1684 | 18.2513 | 6.2020 | 0.0044 | n/a | 0.1701 |
| `ShapeClusterRepair` | 0.0395 | 0.0087 | 0.0582 | 0.1846 | 20.0846 | 7.2252 | 0.0087 | n/a | 0.1761 |
| `FinishMoreLfos` | 0.2303 | 0.0044 | 0.4094 | 1.0000 | 18.5619 | 6.1920 | 1.0000 | n/a | 0.0000 |
| `FinishAndRepair` | 0.0398 | 0.0087 | 0.0660 | 0.1994 | 18.1996 | 6.2228 | 0.0103 | n/a | 0.1712 |
| `AlternatingFinishRepair` | 0.0511 | 0.0087 | 0.0674 | 0.2530 | 19.5840 | 7.3052 | 0.0106 | n/a | 0.1664 |
| `FinishRepairRescue` | 0.0087 | 0.1277 | 0.0511 | 0.1500 | 19.1698 | 7.1602 | 0.0097 | n/a | 0.1405 |
| `MetricBalancedRepair` | 0.0446 | 0.0087 | 0.0611 | 0.1868 | 19.6579 | 7.1354 | 0.0087 | n/a | 0.1750 |
| `TuneAtomsAfterUse` | 0.0165 | 0.0044 | 0.0899 | 0.3051 | 25.3123 | 6.3464 | 0.0044 | n/a | 0.1426 |
| `PathAwareRepair` | 0.0414 | 0.0087 | 0.0581 | 0.1810 | 18.0874 | 6.2295 | 0.0087 | n/a | 0.1748 |

### `utility_candidate_budget`

| Value | Median RMSE | Perfect Rate | P95 RMSE | Node Max P95 | Construct s | Encode s | NoOp Median | Effective NoOp | Overshoot Rate |
|---|---|---|---|---|---|---|---|---|---|
| `CandidateBudget8` | 0.0484 | 0.0087 | 0.0619 | 0.2875 | 14.5487 | 6.1955 | 0.0087 | n/a | 0.1791 |
| `CandidateBudget12` | 0.0402 | 0.0087 | 0.0624 | 0.1960 | 22.0455 | 8.3261 | 0.0087 | n/a | 0.1830 |
| `CandidateBudget24` | 0.0409 | 0.0087 | 0.0580 | 0.1830 | 20.1140 | 6.7153 | 0.0087 | n/a | 0.1779 |
| `CandidateBudget48` | 0.0390 | 0.0087 | 0.0561 | 0.1803 | 25.7701 | 6.6890 | 0.0087 | n/a | 0.1744 |

### `layer_normalization_policy`

| Value | Median RMSE | Perfect Rate | P95 RMSE | Node Max P95 | Construct s | Encode s | NoOp Median | Effective NoOp | Overshoot Rate |
|---|---|---|---|---|---|---|---|---|---|
| `FinalClipOnly` | 0.0409 | 0.0087 | 0.0580 | 0.1830 | 19.6227 | 6.7517 | 0.0087 | n/a | 0.1779 |
| `LayerClip0To1` | 0.0401 | 0.0087 | 0.0509 | 0.1641 | 19.5482 | 6.9451 | 0.0087 | n/a | 0.0000 |
| `LayerClipNeg0p1To1p1` | 0.0408 | 0.0087 | 0.0545 | 0.1795 | 18.9346 | 6.7073 | 0.0087 | n/a | 0.1595 |
| `LayerClipNeg1To1` | 0.0383 | 0.0087 | 0.0582 | 0.2821 | 19.0457 | 6.8430 | 0.0087 | n/a | 0.0578 |
| `LayerCenterPreserveClip` | 0.0390 | 0.0087 | 0.0510 | 0.1563 | 21.2112 | 7.7517 | 0.0087 | n/a | 0.0000 |
| `LayerSoftClip0To1` | 0.1187 | 0.0044 | 0.1978 | 0.7510 | 19.3392 | 8.1602 | 0.9717 | n/a | 0.0000 |
| `LayerSoftClipNeg0p1To1p1` | 0.1188 | 0.0087 | 0.2199 | 0.7676 | 21.5257 | 8.2844 | 0.9782 | n/a | 0.3528 |
| `OvershootPenaltyNoClip` | 0.0445 | 0.0087 | 0.0638 | 0.1974 | 22.0703 | 8.4366 | 0.0044 | n/a | 0.1133 |
| `BoundedResidualStep` | 0.0429 | 0.0087 | 0.1303 | 0.5158 | 21.8500 | 7.9036 | 0.1128 | n/a | 0.0001 |

### `no_damage_policy`

| Value | Median RMSE | Perfect Rate | P95 RMSE | Node Max P95 | Construct s | Encode s | NoOp Median | Effective NoOp | Overshoot Rate |
|---|---|---|---|---|---|---|---|---|---|
| `NoDamageOff` | 0.0409 | 0.0087 | 0.0580 | 0.1830 | 18.2375 | 6.9013 | 0.0087 | n/a | 0.1779 |
| `LateLayerNoDamage` | 0.0409 | 0.0087 | 0.0580 | 0.1830 | 27.3323 | 7.0330 | 0.0087 | n/a | 0.1779 |
| `PerfectLocking` | 0.0409 | 0.0087 | 0.0580 | 0.1830 | 18.3230 | 6.7490 | 0.0087 | n/a | 0.1779 |
| `LateLayerNoDamageAndPerfectLocking` | 0.0409 | 0.0087 | 0.0580 | 0.1830 | 18.0694 | 6.9222 | 0.0087 | n/a | 0.1779 |

### `atom_preprocessing_policy`

| Value | Median RMSE | Perfect Rate | P95 RMSE | Node Max P95 | Construct s | Encode s | NoOp Median | Effective NoOp | Overshoot Rate |
|---|---|---|---|---|---|---|---|---|---|
| `RawAtoms` | 0.0409 | 0.0087 | 0.0580 | 0.1830 | 18.2910 | 6.9124 | 0.0087 | n/a | 0.1779 |
| `EnergyNormalizedAtoms` | 0.0389 | 0.0087 | 0.0590 | 0.1804 | 18.8382 | 8.7149 | 0.0087 | n/a | 0.1827 |
| `CenteredEnergyNormalizedAtoms` | 0.0402 | 0.0044 | 0.0677 | 0.2147 | 21.8282 | 7.1741 | 0.0056 | n/a | 0.1531 |

### `duplicate_suppression_policy`

| Value | Median RMSE | Perfect Rate | P95 RMSE | Node Max P95 | Construct s | Encode s | NoOp Median | Effective NoOp | Overshoot Rate |
|---|---|---|---|---|---|---|---|---|---|
| `DuplicateSuppressionOff` | 0.0409 | 0.0087 | 0.0580 | 0.1830 | 18.3394 | 7.1177 | 0.0087 | n/a | 0.1779 |
| `PhaseScaleDuplicateSuppression` | 0.0409 | 0.0087 | 0.0580 | 0.1830 | 26.8901 | 7.1420 | 0.0087 | n/a | 0.1779 |


## Fixed Contract

| Variable | Fixed Value |
|---|---|
| `base_dictionary_size` | `32` |
| `residual_width` | `8` |
| `reserved_atom` | `NoOpAtom` |
| `active_atoms_per_layer` | `7` |
| `residual_depth` | `16` |
| `control_point_count` | `97` |
| `runtime_interface` | `FlatCategoricalPerResidualLayer` |
| `dictionary_scope` | `PerResidualLayer` |
| `runtime_topology` | `None` |

## Screening Variables

| Variable | Values |
|---|---|
| `path_search_policy` | `Beam4Path`, `Beam8Path` |
| `construction_policy` | `BestOverallRepair`, `FamilyBalancedRepair`, `FinishMoreLfos`, `FinishAndRepair`, `AlternatingFinishRepair`, `FinishRepairRescue`, `CommonCaseRepair`, `HardCaseRepair`, `MetricBalancedRepair`, `ShapeClusterRepair`, `TuneAtomsAfterUse`, `PathAwareRepair` |
| `utility_candidate_budget` | `CandidateBudget8`, `CandidateBudget12`, `CandidateBudget24`, `CandidateBudget48` |
| `layer_normalization_policy` | `FinalClipOnly`, `LayerClip0To1`, `LayerClipNeg0p1To1p1`, `LayerClipNeg1To1`, `LayerSoftClip0To1`, `LayerSoftClipNeg0p1To1p1`, `LayerCenterPreserveClip`, `OvershootPenaltyNoClip`, `BoundedResidualStep` |
| `no_damage_policy` | `NoDamageOff`, `LateLayerNoDamage`, `PerfectLocking`, `LateLayerNoDamageAndPerfectLocking` |
| `atom_preprocessing_policy` | `RawAtoms`, `EnergyNormalizedAtoms`, `CenteredEnergyNormalizedAtoms` |
| `duplicate_suppression_policy` | `DuplicateSuppressionOff`, `PhaseScaleDuplicateSuppression` |

## Method Notes

- `W=8` means eight atom choices per residual layer.
- `D=16` means sixteen residual layers.
- `control_point_count=97` is fixed decoder geometry.
- The indices-only baseline has `head_outputs = 32 + 16 * 8 = 160`.
- `PhaseAndResidualGain` has `head_outputs = 32 + 16 * 8 + 17 phase_scalars + 16 residual_gain_scalars = 193`.
- Every residual layer reserves `Atom0 = NoOpAtom`, leaving seven active repair atoms.
- PascalCase is used for variable values in reports and artifacts; variable field names remain implementation-friendly.
- Offline/oracle construction may use corpus residuals to build atoms. Deployed runtime still uses flat categorical per-residual-layer atom selection and does not receive topology or corpus metadata.
- Decoder/free policies such as layer clipping change reconstruction deterministically and add zero model prediction head outputs.

## Run And Artifact Notes

Full run command:

```powershell
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment12_component_ladder.py --mkl-threading-layer SEQUENTIAL --native-threads 1 run --async --backend xpu --metadata .\datasets\presetshare\raw\presetshare_vital_metadata.csv --corpus-sample-fraction 1.0 --monitor-refresh-seconds 15
```

Regenerate this report from completed artifacts:

```powershell
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment12_component_ladder.py --mkl-threading-layer SEQUENTIAL --native-threads 1 analyze --run-dir .\research\experiments\lfo_representation\era2\artifacts\experiment_12\component_ladder
```

- Completed rows in this report view: `36/36`.
- CSV artifacts live under `research/experiments/lfo_representation/era2/artifacts/experiment_12/component_ladder/`.
- Report images live under `C:\Users\angert\Documents\projects\OBRUXO\research\experiments\lfo_representation\era2\reports\images\experiment_12_phase_gain`.
- XPU acceleration was added for optimized phase/gain lattice alignment during the run work. Treat that as workflow/runtime context only; it is not a model-quality variable.

