# Experiment 12 Plan: Fixed-W8D16 Screening Grid

## Summary

Experiment 12 is now a fixed-W8D16 screening run. It keeps the runtime shape
fixed and varies prediction-head-free process variables one at a time.

```text
base_dictionary_size = 32
residual_width = 8
reserved_atom = NoOpAtom
active_atoms_per_layer = 7
residual_depth = 16
control_point_count = 97
runtime_interface = FlatCategoricalPerResidualLayer
dictionary_scope = PerResidualLayer
runtime_topology = None
```

Variable names may remain implementation-friendly. Variable values in docs,
reports, and artifacts are PascalCase.

## Run Command

Run the full Experiment 12 screening grid:

```powershell
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment12_component_ladder.py --mkl-threading-layer SEQUENTIAL --native-threads 1 run --async --backend xpu --metadata .\datasets\presetshare\raw\presetshare_vital_metadata.csv --corpus-sample-fraction 1.0 --monitor-refresh-seconds 30
```

Run the tiny plumbing check:

```powershell
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment12_component_ladder.py --mkl-threading-layer SEQUENTIAL --native-threads 1 run --backend auto --metadata .\datasets\presetshare\raw\presetshare_vital_metadata.csv --smoke
```

Regenerate analytics and the canonical report:

```powershell
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment12_component_ladder.py --mkl-threading-layer SEQUENTIAL --native-threads 1 analyze --run-dir .\research\experiments\lfo_representation\era2\artifacts\experiment_12\component_ladder
```

## Scalar Contexts

Every screened candidate is tested once in each scalar context:

```text
ScalarSchema = IndicesOnly
ScalarSchema = PhaseAndResidualGain
```

`IndicesOnly` has:

```text
head_outputs = 32 + 16 * 8 = 160
```

`PhaseAndResidualGain` has:

```text
head_outputs = 32 + 16 * 8 + 17 phase_scalars + 16 residual_gain_scalars = 193
```

## Screening Grid

Stable defaults while screening one variable at a time:

```text
path_search_policy = Beam4Path
construction_policy = BestOverallRepair
utility_candidate_budget = CandidateBudget24
layer_normalization_policy = FinalClipOnly
no_damage_policy = NoDamageOff
atom_preprocessing_policy = RawAtoms
duplicate_suppression_policy = DuplicateSuppressionOff
```

Candidate values:

| Variable | Candidate Values |
|---|---|
| `path_search_policy` | `Beam4Path`, `Beam8Path` |
| `construction_policy` | `BestOverallRepair`, `FamilyBalancedRepair`, `FinishMoreLfos`, `FinishAndRepair`, `AlternatingFinishRepair`, `FinishRepairRescue`, `CommonCaseRepair`, `HardCaseRepair`, `MetricBalancedRepair`, `ShapeClusterRepair`, `TuneAtomsAfterUse`, `PathAwareRepair` |
| `utility_candidate_budget` | `CandidateBudget8`, `CandidateBudget12`, `CandidateBudget24`, `CandidateBudget48` |
| `layer_normalization_policy` | `FinalClipOnly`, `LayerClip0To1`, `LayerClipNeg0p1To1p1`, `LayerClipNeg1To1`, `LayerSoftClip0To1`, `LayerSoftClipNeg0p1To1p1`, `LayerCenterPreserveClip`, `OvershootPenaltyNoClip`, `BoundedResidualStep` |
| `no_damage_policy` | `NoDamageOff`, `LateLayerNoDamage`, `PerfectLocking`, `LateLayerNoDamageAndPerfectLocking` |
| `atom_preprocessing_policy` | `RawAtoms`, `EnergyNormalizedAtoms`, `CenteredEnergyNormalizedAtoms` |
| `duplicate_suppression_policy` | `DuplicateSuppressionOff`, `PhaseScaleDuplicateSuppression` |

`GreedyPath` and `OutlierChaser` are not main-grid candidates.

## Report Requirements

The report must group rows by `screening_variable` and show both scalar contexts
side by side. It must not auto-rank winners.

Co-primary metrics:

```text
validation_median_rmse
validation_strict_perfect_lfo_rate
validation_p95_rmse
validation_node_max_error_p95
```

Secondary diagnostics:

```text
oracle_construction_time
validation_encoding_time
residual_layer_no_op_usage_rate_median
residual_layer_usage_entropy_median
duplicate_atom_rate
validation_overshoot_rate_before_final_clip
```

## Outputs

The standalone runner writes generated artifacts under:

```text
era2/artifacts/experiment_12/component_ladder/
```

Key CSV artifacts:

```text
summary.csv
screening_results.csv
component_deltas.csv
budget_accounting.csv
scalar_usage.csv
atom_usage_diagnostics.csv
```

The canonical report is:

```text
era2/reports/EXPERIMENT_12_W8D16_COMPONENT_LADDER_REPORT.md
```

Experiment 13 should be planned after manually choosing the top candidates from
the grouped Experiment 12 report.
