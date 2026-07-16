# Experiment 13A: Complete Fixed-W8D16 Strategy Grid

> **13A complete · 90/90 rows.** This report is authoritative for the unfiltered `AllResiduals` strategy grid. The automatic epsilon selector did not pass, no epsilon is frozen, and Experiment 13B has not run; therefore this is not the final AllResiduals-versus-UnresolvedOnly report.

## Main Findings

Layer-wise clipping is the clearest global result. `LayerClip0To1` improves validation P95 RMSE in `45/45` matched pairs, with median delta `-0.0069147758` and range `-0.020405114` to `-0.00089984387`. It is a decoder-free constraint and changes no model prediction heads.

A larger repair shortlist is a secondary, mixed lever. `CandidateBudget48` improves `25/42`, worsens `13`, and ties `4` P95 comparisons; its median effect is only `-0.00043479912`. More offline search is not a guaranteed quality win.

`TwoPhase` improves `21/36` schedule pairs with median P95 delta `-0.0011320245`. The slight aggregate edge remains family-dependent, so schedule should remain an interaction rather than a universal default.

The quality frontier has three distinct jobs rather than one universal winner: `AllClusterMeans + LayerClip0To1` gives the lowest P95 RMSE; `DiverseCoverageHardRepairTwoPhase + CandidateBudget48 + LayerClip0To1` gives the best median RMSE and node-max P95; and the clipped `CommonCaseRepair + CandidateBudget24` anchor preserves the highest strict-perfect rate.

The automatic epsilon rule did not pass. All candidates satisfy the retired unexplained-energy limits, but the best early/middle median reconstructed fraction is only `2.054%`, below the required `5%`. The prescribed `0.001` versus `0.0025` restricted pilot is therefore required before 13B.

![Experiment 13A co-primary quality frontier](images/experiment_13/13a/co_primary_pareto.png)

The x-axis is validation median RMSE and the y-axis is validation P95 RMSE; lower-left is better. Outlined points remain non-dominated after strict-perfect rate and node-max P95 are also considered. The plot is navigation across tradeoffs, not a scalar leaderboard.

## Four Co-Primary Validation Metrics

| Co-primary metric | Better | Best value | Strategy row |
| --- | --- | ---: | --- |
| Median RMSE | lower | 0.008189681 | `x13a_diverse_coverage_hard_repair_two_phase_candidate_budget48_layer_clip0_to1` |
| Strict-perfect LFO rate | higher | 1.246% | `x13a_common_case_repair_candidate_budget24_final_clip_only` |
| P95 RMSE | lower | 0.037600964 | `x13a_all_cluster_means_null_layer_clip0_to1` |
| Node-max error P95 | lower | 0.10100925 | `x13a_diverse_coverage_hard_repair_two_phase_candidate_budget48_layer_clip0_to1` |

| Pareto strategy | Median RMSE ↓ | Strict-perfect ↑ | P95 RMSE ↓ | Node-max P95 ↓ |
| --- | ---: | ---: | ---: | ---: |
| `x13a_all_cluster_means_null_layer_clip0_to1` | 0.012555717 | 0.810% | 0.037600964 | 0.11244357 |
| `x13a_diverse_coverage_hard_repair_two_phase_candidate_budget48_layer_clip0_to1` | 0.008189681 | 0.810% | 0.037883807 | 0.10100925 |
| `x13a_common_case_repair_candidate_budget24_layer_clip0_to1` | 0.035652492 | 1.246% | 0.047089189 | 0.14439343 |

Strict-perfect rate has only two observed values across the 90 rows. RMSE improvements therefore must not be described as automatically improving exact finishes at the fixed `1e-5` threshold.

## Matched Policy Effects

| Matched factor | Metric | Right wins / left wins / ties | Median right-minus-left delta |
| --- | --- | ---: | ---: |
| LayerClip0To1 vs FinalClipOnly | Median RMSE | 44 / 1 / 0 | -0.0025784886 |
| LayerClip0To1 vs FinalClipOnly | Strict-perfect LFO rate | 0 / 0 / 45 | +0.00000 pp |
| LayerClip0To1 vs FinalClipOnly | P95 RMSE | 45 / 0 / 0 | -0.0069147758 |
| LayerClip0To1 vs FinalClipOnly | Node-max error P95 | 45 / 0 / 0 | -0.030743808 |
| CandidateBudget48 vs CandidateBudget24 | Median RMSE | 30 / 8 / 4 | -0.00059053162 |
| CandidateBudget48 vs CandidateBudget24 | Strict-perfect LFO rate | 0 / 6 / 36 | +0.00000 pp |
| CandidateBudget48 vs CandidateBudget24 | P95 RMSE | 25 / 13 / 4 | -0.00043479912 |
| CandidateBudget48 vs CandidateBudget24 | Node-max error P95 | 26 / 12 / 4 | -0.0049522445 |
| TwoPhase vs Interleaved | Median RMSE | 32 / 4 / 0 | -0.0021913247 |
| TwoPhase vs Interleaved | Strict-perfect LFO rate | 0 / 0 / 36 | +0.00000 pp |
| TwoPhase vs Interleaved | P95 RMSE | 21 / 15 / 0 | -0.0011320245 |
| TwoPhase vs Interleaved | Node-max error P95 | 25 / 11 / 0 | -0.0075820833 |

Negative RMSE and node-max deltas favor the policy named before `vs`; positive strict-perfect deltas favor it. These matched comparisons isolate one design factor while holding the others fixed.

### Layer normalization

![Matched normalization P95 deltas](images/experiment_13/13a/normalization_p95_deltas.png)

Every bar is below zero: clipping after each residual layer consistently prevents physical-range overshoot from accumulating into the validation tail.

### Candidate budget

![Matched candidate-budget P95 deltas](images/experiment_13/13a/candidate_budget_p95_deltas.png)

Bars fall on both sides of zero. CandidateBudget48 can find better observed repairs, but later slots and Beam4 encoding frequently compensate for the smaller shortlist.

### Layer schedule

![Matched schedule P95 deltas](images/experiment_13/13a/schedule_p95_deltas.png)

The signs remain mixed. TwoPhase works especially well for some diversity-aware and robust prototype families, while other families benefit from earlier repair interleaving.

## Construction-Family Interpretation

Pure cluster prototypes are competitive at the tail: `AllClusterMeans + LayerClip0To1` is the P95 leader. The best median and node-max row instead combines diverse broad coverage with hard-tail repair, supporting a mechanism in which dissimilar population prototypes remove reusable structure before observed examples address the remaining difficult cases. The CommonCaseRepair anchor retains the strict-perfect lead, showing that finishing behavior is not captured by aggregate RMSE alone.

The ten lowest-P95 rows are:

1. `x13a_all_cluster_means_null_layer_clip0_to1` — P95 `0.037600964`, median `0.012555717`, strict-perfect `0.810%`, node-max P95 `0.11244357`.
2. `x13a_diverse_coverage_hard_repair_two_phase_candidate_budget48_layer_clip0_to1` — P95 `0.037883807`, median `0.008189681`, strict-perfect `0.810%`, node-max P95 `0.10100925`.
3. `x13a_diverse_coverage_hard_repair_two_phase_candidate_budget24_layer_clip0_to1` — P95 `0.038598217`, median `0.0082718218`, strict-perfect `0.810%`, node-max P95 `0.11814735`.
4. `x13a_cluster_mean_hard_repair_interleaved_candidate_budget48_layer_clip0_to1` — P95 `0.038626019`, median `0.014098603`, strict-perfect `0.810%`, node-max P95 `0.11522713`.
5. `x13a_cluster_mean_global_repair_interleaved_candidate_budget48_layer_clip0_to1` — P95 `0.040312462`, median `0.013645125`, strict-perfect `0.810%`, node-max P95 `0.11487934`.
6. `x13a_diverse_coverage_hard_repair_interleaved_candidate_budget24_layer_clip0_to1` — P95 `0.040468305`, median `0.010938331`, strict-perfect `0.810%`, node-max P95 `0.11869422`.
7. `x13a_trimmed_mean_global_repair_two_phase_candidate_budget48_layer_clip0_to1` — P95 `0.041253194`, median `0.012876331`, strict-perfect `0.810%`, node-max P95 `0.1255862`.
8. `x13a_diverse_coverage_hard_repair_interleaved_candidate_budget48_layer_clip0_to1` — P95 `0.041508205`, median `0.0098211784`, strict-perfect `0.810%`, node-max P95 `0.12005837`.
9. `x13a_trimmed_mean_global_repair_two_phase_candidate_budget24_layer_clip0_to1` — P95 `0.041652802`, median `0.012870583`, strict-perfect `0.810%`, node-max P95 `0.12339681`.
10. `x13a_all_cluster_means_null_final_clip_only` — P95 `0.042315565`, median `0.013672336`, strict-perfect `0.810%`, node-max P95 `0.13565724`.

## Partial-Codebook Progression

![Partial-codebook progression](images/experiment_13/13a/partial_codebook_progression.png)

Move left to right as each residual layer gains another active atom; lower validation P95 is better. The early slope measures capacity efficiency, while late flattening shows diminishing returns. The first few atoms carry most of the quality gain, but family curves continue to separate through slot seven, so this fixed-W8 design does not support removing late slots without a separate head-budget experiment.

## Eligibility Calibration and Gate Result

Completed-layer and slot quantiles show how the reconstruction-error threshold required to cover a fixed curve percentile falls as codebook construction proceeds. Coverage plots invert the question: higher reconstructed fraction means more training curves would be retired at a fixed epsilon.

![Completed-layer epsilon quantiles](images/experiment_13/13a/layer_epsilon_quantiles.png)

![Slot-level epsilon quantiles](images/experiment_13/13a/slot_epsilon_quantiles.png)

![Completed-layer reconstructed fractions](images/experiment_13/13a/completed_layer_coverage.png)

![Slot-level reconstructed fractions](images/experiment_13/13a/slot_coverage.png)

The retirement plots ask whether excluding more LFOs would abandon meaningful unexplained residual energy. The desired direction is lower-right: more LFOs retired with less unexplained energy. The energy safety criteria pass, but the coverage criterion does not, so these plots motivate the pilot rather than an epsilon override.

![Retired fraction versus unexplained energy](images/experiment_13/13a/retired_fraction_vs_energy.png)

![Incoming versus unexplained retired energy](images/experiment_13/13a/incoming_vs_unexplained_energy.png)

## Training-Data Scaling Ablation

The preserved full-training prefix supplies `39` matched rows with identical validation membership. It is a non-random execution-order prefix, so this is a bounded method-level ablation rather than a balanced estimate over all construction families.

| Co-primary metric | Median 50%-minus-100% delta | 50% better / 100% better / ties |
| --- | ---: | ---: |
| Median RMSE | +0.002360452 | 13 / 26 / 0 |
| Strict-perfect LFO rate | +0.37383178 pp | 33 / 6 / 0 |
| P95 RMSE | +0.0011711642 | 17 / 22 / 0 |
| Node-max error P95 | -0.0025390834 | 27 / 12 / 0 |

The 50%-training run has modestly worse median and P95 RMSE on the matched prefix, while strict-perfect rate and node-max P95 improve on most rows. This mixed direction argues against describing the sample reduction as uniformly harmful or uniformly beneficial. Runtime is excluded because the legacy fragment includes Modern Standby and a superseded execution implementation.

## Same-Run Runtime Diagnostics

![Experiment 13A same-run oracle construction time](images/experiment_13/13a/oracle_runtime.png)

This chart compares rows only inside the optimized train-50% run. Median oracle construction time is `211.639` seconds and the maximum is `344.741` seconds. The scale is continuous because this run contains no host-sleep outliers. These timings support within-run cost comparisons only.

## Practical Takeaways

- Keep `LayerClip0To1` as the default normalization candidate for the eventual paired analysis.
- Carry all three Pareto strategies into the 13B interpretation; no scalar winner represents all four quality objectives.
- Treat CandidateBudget48 and TwoPhase as interaction-dependent choices, not unconditional defaults.
- Run the prescribed restricted epsilon pilot before any full Experiment 13B launch.
- Do not compare legacy and optimized wall-clock timings or claim a general 50%-training scaling law from the 39-row prefix.

## Method Notes and Generated Artifacts

The completed source run is `../artifacts/experiment_13/strategy_grid_train50_val100_exactopt_v1` relative to this report. The scaling baseline is `../artifacts/experiment_13/strategy_grid_train100_val100_interrupted_39rows_20260716`. Derived analysis tables, report images, and the interactive payload are written outside both source runs.

All rows preserve W8D16, 32 base choices, one no-op plus seven active atoms per residual layer, 97 control points, PhaseAndResidualGain scalars, Beam4 encoding, and 193 model prediction outputs. Codebook construction is offline/oracle work; topology is not a deployed runtime input.
