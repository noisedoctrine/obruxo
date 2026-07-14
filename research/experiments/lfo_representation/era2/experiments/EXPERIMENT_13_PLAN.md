# Experiment 13 Plan: Fixed-W8D16 Residual Construction Strategy Grid

## Summary

Experiment 13 is the strategy-focused stacked grid that follows Experiment 12.
It keeps the runtime representation fixed and spends the experimental width on
how the seven active atoms in each residual-layer codebook are constructed.

The main question is:

```text
Can broad population-derived prototype atoms, mixed with targeted repair atoms,
produce stronger W8D16 reconstruction than observed-residual-only construction?
```

Experiment 12 showed that mixed construction roles can materially change
common-case quality, strict-perfect coverage, and tail quality. Experiment 13
therefore treats construction strategy as the primary axis rather than crossing
all Experiment 12 process variables equally.

The fixed runtime contract remains:

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

This experiment changes offline codebook construction only. It does not change
the deployed target schema or model prediction head budget.

## Construction Semantics

Each residual layer contains one no-op atom and seven active atoms. An LFO
selects one atom from that layer. The construction order of the seven active
atoms affects the greedy codebook-building process, but it does not make the
decoder apply all seven atoms sequentially.

For one residual layer:

```text
layer_residual_i = target_i - prefix_i
```

After each active atom slot is constructed, the current per-LFO error is the
best error available from the no-op plus all atoms selected so far in that
layer. Later atom slots should therefore focus on residual patterns that remain
poorly covered by the partial codebook.

Broad atoms may be synthesized from many residuals. They are not required to
equal any observed training residual. Repair atoms remain observed residual
examples selected by an explicit utility objective.

All synthesized broad atoms must account for phase and residual gain before
combining residuals. Raw averaging of differently shifted residuals is not a
valid implementation.

## Fixed Settings

The main grid fixes:

```text
scalar_schema = PhaseAndResidualGain
path_search_policy = Beam4Path
no_damage_policy = NoDamageOff
atom_preprocessing_policy = RawAtoms
duplicate_suppression_policy = DuplicateSuppressionOff
main_epsilon = 0.02
```

The fixed W8D16 prediction-head budget is:

```text
head_outputs = 32 + 16 * 8 + 17 phase_scalars + 16 residual_gain_scalars
head_outputs = 193
```

Every main-grid row must preserve this budget.

## Primary Questions

Experiment 13 should answer:

1. Do synthesized broad atoms improve reconstruction over atoms copied only from
   observed residuals?
2. Is an interleaved broad/repair schedule better than a two-phase schedule?
3. Should construction use all residuals or hard-exclude LFOs already within
   epsilon?
4. Which broad prototype family works best: aligned mean, trimmed mean, aligned
   median, cluster mean, dominant direction, or diversity-aware coverage?
5. Which repair objective pairs best with broad prototypes: global improvement,
   strict finishing, or hard-tail improvement?
6. Does `LayerClip0To1` help these strategy families consistently?
7. Does increasing repair candidate breadth from 24 to 48 still matter once
   broad atoms are synthesized?
8. How sensitive is unresolved-only construction to the epsilon threshold?

## Residual Population Policies

Every applicable construction policy is tested under two residual-population
policies.

### `AllResiduals`

All training residuals remain eligible during atom construction. Policy-specific
weights may depend on current error, but no residual is hard-excluded because it
is already within epsilon.

Plain-language interpretation:

> Keep learning from the full population, including curves that are already
> reconstructed well.

### `UnresolvedOnly`

Only LFOs whose current best reconstruction is outside epsilon remain eligible:

```text
resolved_i = max(abs(target_i - current_reconstruction_i)) <= epsilon
eligible_i = not resolved_i
```

The mask is recomputed after every active atom slot.

Plain-language interpretation:

> Once a curve is good enough, stop allowing it to consume later codebook
> capacity.

The main grid uses:

```text
epsilon = 0.02
```

The epsilon test uses maximum absolute error across the complete sampled curve,
not RMSE or MSE.

If no eligible residuals remain, the remaining active atom slots should be
filled with no-op atoms and recorded as early completion rather than treated as
a failure.

## Reusable Broad-Atom Builders

### `BroadMean`

Start from a deterministic seed, align the eligible residuals to the current
prototype with the existing phase-and-residual-gain oracle, undo those fitted
transformations, and update the prototype with the aligned least-squares mean.
Repeat for a fixed small number of iterations or until convergence.

With fixed phase and gain assignments, the update is:

```text
atom = sum_i(weight_i * gain_i * inverse_shift(residual_i, phase_i))
       / sum_i(weight_i * gain_i^2)
```

The implementation should use current reconstruction error as a soft weight so
later broad slots emphasize residuals that remain poorly represented.

Plain-language interpretation:

> Average the shared correction after lining the residuals up properly.

### `TrimmedMean`

Build an aligned mean, discard the 10% of eligible residuals with the largest
alignment error to the current prototype, and recompute the aligned mean.

Plain-language interpretation:

> Learn the common broad correction without letting a few unusual residuals
> pull the average too far.

### `AlignedMedian`

After phase/gain alignment, compute a coordinate-wise weighted median in the
prototype frame, then refit the prototype scale.

Plain-language interpretation:

> Use the middle correction at each point so outliers have less influence than
> they do in an average.

### `ClusterMean`

Form deterministic phase/gain-invariant residual clusters, construct an aligned
mean for each cluster, and choose the cluster prototype with the greatest
remaining corpus utility. Recompute clustering after each broad slot.

Plain-language interpretation:

> Find a common family of mistakes, make one representative correction for that
> family, then repeat for the remaining families.

### `DominantDirection`

Phase-canonicalize the eligible residuals, compute the leading weighted
principal direction, choose its sign deterministically, and fit its useful
scale through the existing gain oracle.

Plain-language interpretation:

> Find the strongest recurring direction in which the current reconstruction is
> wrong.

### `DiverseCoverage`

Generate several synthesized prototype proposals from deterministic residual
partitions. Score each proposal by the number of eligible LFOs it improves by a
meaningful amount, total improvement, and a penalty for phase/scale similarity
to already selected atoms.

Plain-language interpretation:

> Help many curves, but avoid spending several atom slots on nearly the same
> broad correction.

The internal prototype proposal count is part of this algorithm, not
`utility_candidate_budget`. Its effective utility candidate budget is `Null`.

## Reusable Repair-Atom Builders

Repair atoms are selected from observed residuals.

### `GlobalRepair`

Choose the observed residual candidate with the greatest summed improvement
across the eligible population.

Plain-language interpretation:

> Pick the concrete correction that removes the most error overall.

### `FinishRepair`

First maximize the number of eligible LFOs moved from outside epsilon to within
epsilon. Break ties with total improvement.

Plain-language interpretation:

> Spend the repair slot finishing curves that are already close.

### `HardRepair`

Choose the candidate with the greatest summed improvement over the worst 10% of
current eligible reconstruction losses.

Plain-language interpretation:

> Spend the repair slot on the difficult tail.

## Mixed Slot Schedules

Every new broad-plus-repair recipe is tested with both schedules below.

### `Interleaved`

```text
slot 1 = Broad
slot 2 = Repair
slot 3 = Broad
slot 4 = Repair
slot 5 = Broad
slot 6 = Repair
slot 7 = Broad
```

Plain-language interpretation:

> Alternate broad coverage with targeted cleanup.

### `TwoPhase`

```text
slot 1 = Broad
slot 2 = Broad
slot 3 = Broad
slot 4 = Broad
slot 5 = Repair
slot 6 = Repair
slot 7 = Repair
```

Plain-language interpretation:

> Establish broad coverage first, then use the remaining slots for cleanup.

Both schedules use four broad slots and three repair slots. This isolates order
without changing the broad/repair ratio.

## Construction Policies

### Existing Experiment 12 anchors

Keep these observed-residual policies as historical anchors:

```text
CommonCaseRepair
FinishRepairRescue
FamilyBalancedRepair
```

Each anchor is crossed with both residual-population policies, both repair
candidate budgets, and both layer-normalization values.

### New mixed prototype/repair recipes

Each recipe below receives both `Interleaved` and `TwoPhase` variants:

| Recipe | Broad builder | Repair builder | Simple interpretation |
|---|---|---|---|
| `BroadMeanGlobalRepair` | `BroadMean` | `GlobalRepair` | Make a broad shared correction, then remove the largest remaining error. |
| `BroadMeanFinishRepair` | `BroadMean` | `FinishRepair` | Make broad progress, then finish curves near epsilon. |
| `BroadMeanHardRepair` | `BroadMean` | `HardRepair` | Make broad progress, then rescue the difficult tail. |
| `TrimmedMeanGlobalRepair` | `TrimmedMean` | `GlobalRepair` | Use a robust broad average, then optimize total cleanup. |
| `AlignedMedianGlobalRepair` | `AlignedMedian` | `GlobalRepair` | Use a robust middle-shaped prototype, then optimize total cleanup. |
| `ClusterMeanGlobalRepair` | `ClusterMean` | `GlobalRepair` | Cover common residual families, then remove the largest remaining error. |
| `ClusterMeanHardRepair` | `ClusterMean` | `HardRepair` | Cover common residual families, then rescue difficult leftovers. |
| `DominantDirectionGlobalRepair` | `DominantDirection` | `GlobalRepair` | Remove the largest shared error direction, then clean up globally. |
| `DiverseCoverageHardRepair` | `DiverseCoverage` | `HardRepair` | Cover several distinct broad problems, then repair the tail. |

PascalCase row identifiers should append the schedule, for example:

```text
BroadMeanGlobalRepairInterleaved
BroadMeanGlobalRepairTwoPhase
ClusterMeanHardRepairInterleaved
ClusterMeanHardRepairTwoPhase
```

### Pure-prototype controls

Test whether observed-residual repair atoms are needed at all:

```text
AllBroadAlignedMeans
AllClusterMeans
AllDominantDirections
```

All seven active slots are synthesized. These policies use
`utility_candidate_budget = Null`.

## Utility Candidate Budget

The main grid recognizes:

```text
CandidateBudget24
CandidateBudget48
Null
```

`CandidateBudget24` and `CandidateBudget48` apply only to repair slots that
select observed residual examples.

For a mixed row:

```text
row utility_candidate_budget = CandidateBudget48
effective slot budgets = [Null, 48, Null, 48, Null, 48, Null]
```

for the interleaved schedule, or:

```text
effective slot budgets = [Null, Null, Null, Null, 48, 48, 48]
```

for the two-phase schedule.

Broad synthesized slots always use `Null`. Pure-prototype rows use `Null` for
all seven slots.

Artifacts must record both the row-level budget and the effective budget for
each slot.

## Layer Normalization

Only two values remain in the main grid:

```text
FinalClipOnly
LayerClip0To1
```

Interpretation:

- `FinalClipOnly`: no per-layer clipping;
- `LayerClip0To1`: clip the running reconstruction to `[0, 1]` after each layer.

This is the only secondary process axis retained because `LayerClip0To1`
performed strongly in Experiment 12 and remains simple to interpret.

## Main Grid

Fixed axes:

```text
scalar_schema = PhaseAndResidualGain
path_search_policy = Beam4Path
no_damage_policy = NoDamageOff
atom_preprocessing_policy = RawAtoms
duplicate_suppression_policy = DuplicateSuppressionOff
epsilon = 0.02
```

Crossed axes:

```text
construction_policy
residual_population_policy = AllResiduals | UnresolvedOnly
utility_candidate_budget = CandidateBudget24 | CandidateBudget48 | Null
layer_normalization_policy = FinalClipOnly | LayerClip0To1
```

The 21 policies containing repair slots receive:

```text
21 construction policies
* 2 residual-population policies
* 2 repair candidate budgets
* 2 layer-normalization policies
= 168 rows
```

The three pure-prototype policies receive:

```text
3 construction policies
* 2 residual-population policies
* 1 Null candidate budget
* 2 layer-normalization policies
= 12 rows
```

Main-grid total:

```text
168 + 12 = 180 rows
```

The 21 repair-containing policies are:

```text
3 existing anchors
+ 9 mixed recipes * 2 schedules
= 21
```

## Epsilon Sensitivity Aside

Do not multiply the full grid by epsilon. Run a focused aside using:

```text
BroadMeanGlobalRepairInterleaved
BroadMeanGlobalRepairTwoPhase
ClusterMeanHardRepairTwoPhase
FinishRepairRescue
```

Fixed aside settings:

```text
residual_population_policy = UnresolvedOnly
utility_candidate_budget = CandidateBudget48
layer_normalization_policy = LayerClip0To1
scalar_schema = PhaseAndResidualGain
path_search_policy = Beam4Path
no_damage_policy = NoDamageOff
atom_preprocessing_policy = RawAtoms
duplicate_suppression_policy = DuplicateSuppressionOff
```

Test:

```text
epsilon = 0.01
epsilon = 0.02
epsilon = 0.04
```

This defines 12 comparison rows. The four `epsilon = 0.02` rows may reuse exact
main-grid rows, so only eight additional executions are required.

The aside should be reported separately and should not determine the main
strategy ranking.

## Co-Primary Metrics

Retain the Experiment 12 co-primary metrics:

```text
validation_median_rmse
validation_strict_perfect_lfo_rate
validation_p95_rmse
validation_node_max_error_p95
```

Also report:

```text
validation_p99_rmse
validation_max_rmse
validation_max_abs_error_p95
```

## Strategy Diagnostics

Experiment 13 must record enough information to explain why a policy works.

Per row and per active atom slot:

```text
slot_index
slot_role
atom_source_kind
effective_candidate_budget
eligible_residual_count_before
eligible_residual_count_after
resolved_lfo_rate_before
resolved_lfo_rate_after
training_median_rmse_before
training_median_rmse_after
training_p95_rmse_before
training_p95_rmse_after
newly_resolved_lfo_count
assigned_residual_count
atom_phase_scale_similarity_to_previous_max
prototype_iteration_count
prototype_converged
prototype_population_size
repair_source_dataset_index
```

For partial-codebook validation, evaluate the row using:

```text
NoOpAtom + first 1 active atom per layer
NoOpAtom + first 2 active atoms per layer
...
NoOpAtom + all 7 active atoms per layer
```

Write:

```text
active_atom_count
validation_median_rmse
validation_strict_perfect_lfo_rate
validation_p95_rmse
validation_node_max_error_p95
```

This progression is required because a central hypothesis is that broad-first
strategies reduce error quickly with the earliest codebook slots.

## Report Requirements

The report should be findings-first and focus on paired comparisons:

1. prototype-containing policies versus observed-residual anchors;
2. `Interleaved` versus `TwoPhase`;
3. `AllResiduals` versus `UnresolvedOnly`;
4. `CandidateBudget24` versus `CandidateBudget48`;
5. `FinalClipOnly` versus `LayerClip0To1`;
6. broad-builder families;
7. repair objectives;
8. partial-codebook progression from one through seven active atoms.

The report should not collapse the result into one automatic scalar ranking.
It should identify Pareto candidates across the co-primary metrics and explain
the tradeoffs.

The report must clearly distinguish:

```text
observed residual atom
aligned mean prototype
trimmed mean prototype
aligned median prototype
cluster mean prototype
dominant direction prototype
diversity-aware coverage prototype
```

## Outputs

The standalone runner should write generated artifacts under:

```text
era2/artifacts/experiment_13/strategy_grid/
```

Required files:

```text
manifest.json
summary.csv
strategy_results.csv
slot_progression.csv
partial_codebook_validation.csv
atom_construction.csv
atom_assignments.csv
candidate_search_diagnostics.csv
epsilon_sensitivity.csv
budget_accounting.csv
failures.csv
run_status.json
run_events.jsonl
```

The canonical report should be:

```text
era2/reports/EXPERIMENT_13_W8D16_STRATEGY_GRID_REPORT.md
```

Local report images should be written under:

```text
era2/reports/images/experiment_13/
```

## Manifest Fields

Preserve the Experiment 12 fields and add strategy-specific fields:

```text
experiment_id
scalar_schema
path_search_policy
construction_policy
construction_family
slot_schedule
residual_population_policy
epsilon
utility_candidate_budget
effective_candidate_budget_by_slot
layer_normalization_policy
no_damage_policy
atom_preprocessing_policy
duplicate_suppression_policy
reserved_atom
active_atoms_per_layer
broad_atom_builder
repair_atom_builder
prototype_uses_observed_residual_value
runtime_topology
head_outputs_actual
```

## Run Command

Experiment 13 requires a dedicated runner. The intended command shape is:

```powershell
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment13_strategy_grid.py --mkl-threading-layer SEQUENTIAL --native-threads 1 run --async --backend xpu --metadata .\datasets\presetshare\raw\presetshare_vital_metadata.csv --corpus-sample-fraction 1.0 --monitor-refresh-seconds 30
```

Tiny plumbing check:

```powershell
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment13_strategy_grid.py --mkl-threading-layer SEQUENTIAL --native-threads 1 run --backend auto --metadata .\datasets\presetshare\raw\presetshare_vital_metadata.csv --smoke
```

Regenerate analytics and the canonical report:

```powershell
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment13_strategy_grid.py --mkl-threading-layer SEQUENTIAL --native-threads 1 analyze --run-dir .\research\experiments\lfo_representation\era2\artifacts\experiment_13\strategy_grid
```

The runner is not implemented by this plan.

## Test Requirements

Tests should verify:

- the fixed W8D16 runtime contract and 193-head accounting;
- `Atom0 = NoOpAtom` for every residual layer;
- exact seven-slot `Interleaved` and `TwoPhase` schedules;
- AllResiduals and UnresolvedOnly mask behavior;
- epsilon uses complete-curve maximum absolute error;
- the unresolved mask is recomputed after every slot;
- broad atoms can differ from every observed residual;
- phase/gain-aligned prototype updates do not raw-average shifted residuals;
- repair candidate budgets apply only to repair slots;
- pure-prototype rows require `Null`;
- deterministic construction under a fixed seed;
- empty unresolved populations terminate with remaining no-op atoms;
- the main grid contains exactly 180 rows;
- partial-codebook validation covers active atom counts 1 through 7;
- no topology appears in runtime targets, decoder lookup, loss, or head accounting.

## Deferred Backlog

Do not cross these in Experiment 13:

```text
Beam8Path
LateLayerNoDamage
LateLayerNoDamageAndPerfectLocking
EnergyNormalizedAtoms
CenteredEnergyNormalizedAtoms
PhaseScaleDuplicateSuppression
```

They remain valid follow-up axes after the construction strategy is selected.

## Non-Goals

Experiment 13 does not:

- vary residual width or depth;
- compare scalar schemas;
- compare Beam4 and Beam8;
- screen no-damage policies;
- screen atom preprocessing;
- screen duplicate suppression;
- change the flat-categorical runtime interface;
- add runtime topology;
- predict the x grid;
- train the audio-to-patch model;
- establish that epsilon 0.02 is uniquely correct;
- treat oracle construction time or codebook storage as model prediction-head
  cost.

The experiment is an oracle reconstruction and codebook-construction study. Its
result should choose a stronger construction recipe before returning to
representation scaling or learned-model predictability.
