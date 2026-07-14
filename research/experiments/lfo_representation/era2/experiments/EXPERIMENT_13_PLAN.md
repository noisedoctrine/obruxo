# Experiment 13 Plan: Fixed-W8D16 Residual Construction Strategy Grid

## Summary

Experiment 13 tests how the seven active atoms in each residual-layer codebook
should be constructed. The runtime representation remains fixed. The experiment
compares broad population-derived prototypes, targeted observed-residual repair
atoms, and different schedules for combining those roles.

Experiment 13 must run in two ordered phases:

1. **Experiment 13A — calibration and unfiltered construction.** Run every
   `AllResiduals` row. Do not exclude any LFO because it is already within an
   epsilon threshold. Measure the complete distribution of reconstruction error
   after every residual layer and after every active atom slot inside each layer.
2. **Experiment 13B — filtered construction.** Use the 13A measurements to choose
   one prespecified global epsilon, then run the paired `UnresolvedOnly` rows.
   These rows recompute the eligible population after every active atom slot.

**Experiment 13B must not begin until the Experiment 13A calibration report and
threshold-selection artifact have been written.** Epsilon is therefore not a
free constant selected before the experiment. It is a construction-control
parameter calibrated from 13A and then fixed for 13B.

The main research question is:

> Can broad population-derived prototype atoms, mixed with targeted repair atoms,
> produce stronger W8D16 reconstruction than observed-residual-only construction?

The epsilon-specific question is:

> At each layer and atom slot, how much of the training population is already
> reconstructed well enough that removing it from subsequent atom construction
> preserves useful residual-error mass while reducing the population that later
> slots must consider?

## Fixed Runtime Contract

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
scalar_schema = PhaseAndResidualGain
path_search_policy = Beam4Path
no_damage_policy = NoDamageOff
atom_preprocessing_policy = RawAtoms
duplicate_suppression_policy = DuplicateSuppressionOff
```

The prediction-head budget is fixed:

$$
H = 32 + 16\cdot 8 + 17 + 16 = 193.
$$

Experiment 13 changes offline codebook construction only. It does not change the
model-facing target schema, runtime decoder interface, or head accounting.

## Core Reconstruction Notation

For LFO $i$ entering residual layer $d$, let the incoming residual be

$$
r_{i,d} = y_i - \hat y_{i,d-1}.
$$

Let $A_{d,s}=\{a_{d,0},\ldots,a_{d,s}\}$ be the partial codebook after active
slot $s$, with $a_{d,0}=0$ as the no-op atom. Let $S_\phi(a)$ circularly shift
atom $a$ by phase $\phi$. The partial-codebook maximum-absolute residual error is

$$
E_{i,d,s}
=
\min_{a\in A_{d,s},\,\phi,\,g}
\left\|r_{i,d}-gS_\phi(a)\right\|_\infty.
$$

The full-curve error after completing residual layer $d$ is

$$
G_{i,d}=\left\|y_i-\hat y_{i,d}\right\|_\infty.
$$

Epsilon membership must always use these complete-curve maximum-absolute errors.
RMSE or MSE may still be used for candidate utility scoring, but they must not be
substituted for the epsilon test.

## Ordered Execution Contract

### Experiment 13A — AllResiduals calibration

13A runs all construction strategies with

```text
residual_population_policy = AllResiduals
```

For every training residual $i$, layer $d$, and active slot $s$,

$$
\operatorname{eligible}^{13A}_{i,d,s}=1.
$$

No epsilon changes construction in 13A. Epsilon values are used only for
measurement and plotting.

13A has four jobs:

1. establish the unfiltered quality baseline for every strategy;
2. measure how rapidly errors fall through the residual ladder;
3. estimate how many LFOs would be retired by candidate epsilon values;
4. choose one global epsilon for the paired 13B run using a rule written below.

### Experiment 13B — UnresolvedOnly construction

13B runs the paired construction rows with

```text
residual_population_policy = UnresolvedOnly
```

Given the selected global threshold $\epsilon^*$, eligibility is

$$
\operatorname{eligible}^{13B}_{i,d,s}
=
\mathbf{1}\!\left[E_{i,d,s}>\epsilon^*\right].
$$

The mask must be recomputed after every active atom slot, not only after a whole
residual layer. A newly added atom can resolve additional residuals, and those
residuals must stop influencing later broad or repair slots in the same layer.

If no eligible residuals remain, all remaining active slots in that layer are
filled with no-op atoms and the row records early completion.

13B must use one fixed epsilon across all rows. Do not choose a different epsilon
for each construction strategy; doing so would destroy the paired comparison.

## Experiment 13A Calibration Measurements

### Layer-level error distribution

After the base dictionary and after each of the 16 completed residual layers,
record the distribution of $G_{i,d}$ on both training and validation splits.
Layer 0 means base reconstruction before residual layers.

For each layer, record at least these quantiles:

```text
50th percentile
25th percentile
10th percentile
5th percentile
2nd percentile
1st percentile
0.1st percentile, only when the split contains at least 1000 LFOs
```

For percentile $p$, define

$$
Q^{\mathrm{global}}_d(p)=\operatorname{Quantile}_i(G_{i,d},p).
$$

Interpretation: $Q^{\mathrm{global}}_d(0.10)$ is the epsilon that would classify
approximately 10% of LFOs as reconstructed after layer $d$.

### Slot-level error distribution

The filtered policy operates inside each residual layer, so layer-level data is
not sufficient. After every active atom slot, record the training distribution
of $E_{i,d,s}$:

$$
Q^{\mathrm{slot}}_{d,s}(p)=\operatorname{Quantile}_i(E_{i,d,s},p).
$$

This is the primary calibration dataset for 13B. It estimates how many residuals
would stop influencing slots $s+1,\ldots,7$ under a given epsilon.

### Fixed-epsilon coverage curves

For each candidate threshold $\epsilon$ and checkpoint $(d,s)$, record

$$
F_{d,s}(\epsilon)
=
\frac{1}{N}\sum_{i=1}^N
\mathbf{1}\!\left[E_{i,d,s}\leq\epsilon\right].
$$

Initially evaluate at least:

```text
0.001
0.0025
0.005
0.01
0.02
```

These are observational thresholds in 13A. They do not alter construction.

### Retired residual-error mass

A threshold may retire many easy residuals while discarding little remaining
error, or it may retire fewer but still important residuals. Count alone is not
enough. For each checkpoint and epsilon, record the retired residual-energy share

$$
M_{d,s}(\epsilon)
=
\frac{
\sum_{i:E_{i,d,s}\leq\epsilon}\lVert r_{i,d}\rVert_2^2
}{
\sum_i\lVert r_{i,d}\rVert_2^2
}.
$$

Also record the retained share $1-M_{d,s}(\epsilon)$.

The desired regime is a high retired-LFO fraction with a low retired-error-mass
fraction.

## Epsilon Selection Rule

The selection rule must be applied to training data only. Validation data is for
reporting and must not determine $\epsilon^*$.

Use the following default rule unless the 13A run reveals a numerical pathology:

1. evaluate candidate thresholds `0.001`, `0.0025`, `0.005`, `0.01`, and `0.02`;
2. aggregate slot-level measurements across all 13A rows;
3. choose the largest threshold satisfying both:

$$
\operatorname{Median}_{\text{rows, slots}}M_{d,s}(\epsilon)\leq 0.01,
$$

and

$$
Q_{0.95,\,\text{rows, slots}}\!\left(M_{d,s}(\epsilon)\right)\leq 0.05;
$$

4. require that the chosen threshold retires a nontrivial population at some
   early or middle checkpoints; if every candidate satisfying the energy rule
   retires less than 1% of LFOs everywhere, select the largest satisfying value
   and record that filtering has little practical effect;
5. write the chosen threshold and all supporting statistics to
   `epsilon_selection.json` before starting 13B.

If no candidate threshold satisfies the energy rule, do not silently loosen it.
Record the failure and run a small 13B pilot at the two tightest thresholds before
proceeding.

## Required Calibration Plots

13A must produce the following plots before 13B begins.

### Epsilon quantiles by completed layer

- x-axis: base/layer index `0..16`;
- y-axis: maximum-absolute-error epsilon;
- lines: 50th, 25th, 10th, 5th, 2nd, 1st, and eligible 0.1st percentiles.

### Epsilon quantiles by atom slot

For each residual layer or a clearly documented aggregation across layers:

- x-axis: active atom slot `0..7`;
- y-axis: $E_{i,d,s}$;
- lines: the same quantiles.

### Fraction reconstructed by layer

- x-axis: base/layer index `0..16`;
- y-axis: fraction with $G_{i,d}\leq\epsilon$;
- one line per candidate epsilon.

### Fraction reconstructed by atom slot

- x-axis: active atom slot `0..7`;
- y-axis: fraction with $E_{i,d,s}\leq\epsilon$;
- one line per candidate epsilon.

### Retired count versus retired error mass

- x-axis: retired LFO fraction $F_{d,s}(\epsilon)$;
- y-axis: retired residual-energy share $M_{d,s}(\epsilon)$;
- one point per candidate epsilon and checkpoint.

Detailed row-level curves remain in artifacts. The report should show median
curves and percentile bands across rows, plus grouped views by strategy family.

## Residual Population Policies

### `AllResiduals`

**Common intuitive description**

> Keep learning from every residual, including ones already reconstructed well.

**Technical description**

No residual is hard-excluded. Strategy-specific soft weights may still emphasize
larger current errors.

**Mathematical formulation**

$$
\operatorname{eligible}_{i,d,s}=1.
$$

### `UnresolvedOnly`

**Common intuitive description**

> Once a residual is within the calibrated tolerance, stop allowing it to consume
> later codebook capacity.

**Technical description**

Use the globally selected $\epsilon^*$ and recompute eligibility after every atom
slot from the current partial codebook.

**Mathematical formulation**

$$
\operatorname{eligible}_{i,d,s}
=
\mathbf{1}\!\left[E_{i,d,s}>\epsilon^*\right].
$$

## Shared Alignment and Utility Definitions

For proposal atom $a$ and residual $r_i$, fit phase and gain by

$$
(\phi_i(a),g_i(a))
=
\arg\min_{\phi,g}
\ell\!\left(r_i,gS_\phi(a)\right).
$$

Let $L_i^{(s)}$ be the current scalar construction loss before slot $s$, and
$L_i(a)$ the best loss with proposal $a$. Candidate improvement is

$$
\Delta_i(a)=\max\!\left(0,L_i^{(s)}-L_i(a)\right).
$$

The scalar construction loss must be recorded. Strict epsilon membership remains
based on $E_{i,d,s}$, independent of this utility loss.

## Broad-Atom Builders

### `BroadMean`

**Common intuitive description**

> Line up the residuals, account for their fitted sizes, and average the shared
> correction.

**Technical description**

Initialize deterministically. Alternate phase/gain alignment with a weighted
least-squares prototype update. Use current reconstruction error as a soft weight.
Stop at a fixed iteration cap or convergence tolerance.

**Mathematical formulation**

For fixed assignments $(\phi_i,g_i)$ and weights $w_i\geq0$, minimize

$$
\min_a
\sum_i e_iw_i
\left\|r_i-g_iS_{\phi_i}(a)\right\|_2^2,
$$

where $e_i$ is the eligibility indicator. The update is

$$
a
=
\frac{
\sum_i e_iw_ig_iS_{-\phi_i}(r_i)
}{
\sum_i e_iw_ig_i^2
}.
$$

### `TrimmedMean`

**Common intuitive description**

> Average the common pattern without allowing a small number of unusual residuals
> to pull the prototype away from the majority.

**Technical description**

Run the aligned-mean update, discard the worst 10% of eligible aligned losses,
and recompute from the retained 90%. Repeat until convergence or the iteration
cap. Trimming happens after alignment.

**Mathematical formulation**

If $T$ is the retained lowest-loss 90%, then

$$
a
=
\frac{
\sum_{i\in T}w_ig_iS_{-\phi_i}(r_i)
}{
\sum_{i\in T}w_ig_i^2
}.
$$

### `AlignedMedian`

**Common intuitive description**

> Use the middle aligned correction at each control point so isolated extremes
> have less influence.

**Technical description**

Undo fitted phase and numerically safe gain, compute a coordinate-wise weighted
median, refit phase/gain, and iterate. Define and record a gain floor.

**Mathematical formulation**

$$
z_i=rac{S_{-\phi_i}(r_i)}{\max(|g_i|,g_{\min})},
$$

and for control point $t$,

$$
a[t]=\operatorname{WeightedMedian}_i\!\left(z_i[t];e_iw_i\right).
$$

### `ClusterMean`

**Common intuitive description**

> Find a major family of remaining mistakes, average that family into one broad
> correction, then search for another family at the next broad slot.

**Technical description**

Cluster eligible residuals using a phase/gain-invariant distance. Build an aligned
mean for each viable cluster. Score all cluster prototypes against the complete
eligible population and select the highest-utility proposal. Recluster at every
broad slot.

**Mathematical formulation**

$$
d(r_i,r_j)
=
\min_{\phi,g}
\left\|r_i-gS_\phi(r_j)\right\|_2^2.
$$

For cluster $C_k$,

$$
a_k
=
\arg\min_a
\sum_{i\in C_k}w_i
\min_{\phi,g}
\left\|r_i-gS_\phi(a)\right\|_2^2.
$$

Choose $a^*=\arg\max_{a_k}\operatorname{Utility}(a_k)$.

### `DominantDirection`

**Common intuitive description**

> Find the strongest recurring direction in which many current reconstructions
> are wrong, even if no single residual is the ideal example.

**Technical description**

Canonicalize phase and sign, compute the leading weighted uncentered principal
direction, choose sign deterministically, and let the gain oracle fit magnitude.
Alternate canonicalization and direction estimation.

**Mathematical formulation**

For canonical residuals $z_i=S_{-\phi_i}(r_i)$,

$$
C=\sum_i e_iw_iz_iz_i^\top,
$$

and

$$
a=\arg\max_{\lVert a\rVert_2=1}a^\top Ca.
$$

### `DiverseCoverage`

**Common intuitive description**

> Help many residuals, but avoid spending several broad slots on nearly the same
> correction.

**Technical description**

Generate deterministic synthesized proposals. Score each by coverage, total
improvement, and phase/scale-invariant dissimilarity from previously selected
broad atoms.

**Mathematical formulation**

$$
\operatorname{coverage}(a)
=
\sum_i e_i\mathbf{1}[\Delta_i(a)\geq\delta_{\min}],
$$

$$
\operatorname{improvement}(a)
=
\sum_i e_iw_i\Delta_i(a),
$$

$$
\operatorname{sim}(a,b)
=
\max_\phi
\frac{|\langle a,S_\phi(b)\rangle|}
{\lVert a\rVert_2\lVert b\rVert_2},
$$

and

$$
\operatorname{score}(a)
=
\alpha\operatorname{coverage}(a)
+
\beta\operatorname{improvement}(a)
-
\lambda\max_{b\in B}\operatorname{sim}(a,b).
$$

## Repair-Atom Builders

Repair atoms are selected from observed eligible residuals.

### `GlobalRepair`

**Common intuitive description**

> Pick the concrete residual example that removes the most total error.

**Technical description**

Score each candidate against every eligible residual and maximize summed positive
improvement.

$$
a^*=\arg\max_a\sum_i e_i\Delta_i(a).
$$

### `FinishRepair`

**Common intuitive description**

> Prefer the repair that moves the greatest number of residuals under the current
> epsilon threshold.

**Technical description**

Maximize newly resolved residual count; break ties with total improvement. In
13A this builder must use observational thresholds only for diagnostics and keep
its historical Experiment 12 semantics for anchor comparability. In 13B it uses
$\epsilon^*$.

$$
\operatorname{finish}(a)
=
\sum_i e_i
\mathbf{1}[E_i^{\text{before}}>\epsilon^*]
\mathbf{1}[E_i(a)\leq\epsilon^*].
$$

### `HardRepair`

**Common intuitive description**

> Spend the repair slot on the worst remaining residuals.

**Technical description**

Define the eligible hard set as the worst 10% by current scalar loss and maximize
summed improvement on that set.

$$
H=\{i:L_i^{(s)}\geq Q_{0.90}(L^{(s)})\},
$$

$$
a^*=\arg\max_a\sum_{i\in H}e_i\Delta_i(a).
$$

## Slot Schedules

### `Interleaved`

```text
Broad, Repair, Broad, Repair, Broad, Repair, Broad
```

> Alternate population-wide coverage with targeted cleanup.

### `TwoPhase`

```text
Broad, Broad, Broad, Broad, Repair, Repair, Repair
```

> Establish broad coverage first, then spend the remaining slots on cleanup.

Both schedules use four broad and three repair slots, isolating order rather than
role count.

## Construction Policies

### Experiment 12 anchors

Retain these exact observed-residual policies as historical anchors:

```text
CommonCaseRepair
FinishRepairRescue
FamilyBalancedRepair
```

Their existing implementation semantics should be preserved. The only new change
in 13B is the eligibility mask applied to the population used for construction.

### New mixed recipes

Each recipe receives `Interleaved` and `TwoPhase` variants:

| Recipe | Broad builder | Repair builder | Intention |
|---|---|---|---|
| `BroadMeanGlobalRepair` | `BroadMean` | `GlobalRepair` | Shared average plus total cleanup. |
| `BroadMeanFinishRepair` | `BroadMean` | `FinishRepair` | Shared average plus threshold finishing. |
| `BroadMeanHardRepair` | `BroadMean` | `HardRepair` | Shared average plus tail rescue. |
| `TrimmedMeanGlobalRepair` | `TrimmedMean` | `GlobalRepair` | Robust average plus total cleanup. |
| `AlignedMedianGlobalRepair` | `AlignedMedian` | `GlobalRepair` | Robust median prototype plus total cleanup. |
| `ClusterMeanGlobalRepair` | `ClusterMean` | `GlobalRepair` | Common-family prototypes plus total cleanup. |
| `ClusterMeanHardRepair` | `ClusterMean` | `HardRepair` | Common-family prototypes plus tail rescue. |
| `DominantDirectionGlobalRepair` | `DominantDirection` | `GlobalRepair` | Shared error direction plus total cleanup. |
| `DiverseCoverageHardRepair` | `DiverseCoverage` | `HardRepair` | Distinct broad coverage plus tail rescue. |

### Pure-prototype controls

```text
AllBroadAlignedMeans
AllClusterMeans
AllDominantDirections
```

All seven active slots are synthesized. Their candidate budget is `Null`.

## Candidate Budget and Normalization

```text
utility_candidate_budget = CandidateBudget24 | CandidateBudget48 | Null
layer_normalization_policy = FinalClipOnly | LayerClip0To1
```

Candidate budgets apply only to observed-residual repair slots. Synthesized broad
slots always use `Null`.

Examples:

```text
Interleaved with CandidateBudget48:
[Null, 48, Null, 48, Null, 48, Null]

TwoPhase with CandidateBudget48:
[Null, Null, Null, Null, 48, 48, 48]
```

## Grid Size and Phase Split

The full paired design remains 180 rows:

```text
21 repair-containing policies
* 2 population policies
* 2 repair candidate budgets
* 2 normalization policies
= 168 rows

3 pure-prototype policies
* 2 population policies
* 1 Null budget
* 2 normalization policies
= 12 rows
```

Execution split:

```text
Experiment 13A: 90 AllResiduals rows
Experiment 13B: 90 UnresolvedOnly rows
```

13A and 13B rows must share a stable `pair_id` so each filtered row can be
compared directly with its unfiltered counterpart.

## Required Artifacts

Write under:

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
layer_epsilon_quantiles.csv
slot_epsilon_quantiles.csv
epsilon_coverage.csv
retired_error_mass.csv
epsilon_selection.json
experiment13a_status.json
experiment13b_status.json
budget_accounting.csv
failures.csv
run_events.jsonl
```

`epsilon_selection.json` must contain:

```text
candidate_epsilons
selection_rule_version
selected_epsilon
training_statistics_used
median_retired_error_mass
p95_retired_error_mass
retired_lfo_fraction_summary
selection_timestamp
13a_run_identity
```

## Strategy Diagnostics

Per row, residual layer, and active atom slot record:

```text
phase = 13A | 13B
pair_id
slot_index
slot_role
atom_source_kind
effective_candidate_budget
eligible_residual_count_before
eligible_residual_count_after
resolved_lfo_rate_before
resolved_lfo_rate_after
newly_resolved_lfo_count
training_median_rmse_before
training_median_rmse_after
training_p95_rmse_before
training_p95_rmse_after
prototype_iteration_count
prototype_converged
prototype_population_size
repair_source_dataset_index
```

For 13A, `eligible_residual_count_after` remains the full count; separately record
counterfactual resolved fractions for every candidate epsilon.

## Co-Primary Metrics

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

## Report Requirements

The report must be findings-first and explicitly separated into:

1. **13A construction results** — strategy performance without filtering;
2. **13A epsilon calibration** — quantile, coverage, and retired-error-mass plots;
3. **epsilon decision** — exact rule, selected value, and supporting statistics;
4. **13B filtered results** — paired comparison against 13A;
5. **strategy interactions** — schedule, builder, budget, and normalization effects.

The report must not imply that 13A and 13B were run simultaneously. It must make
clear that 13A informed one global threshold and that 13B used that frozen value.

Do not collapse results into one automatic scalar ranking. Report Pareto
candidates across the co-primary metrics.

## Run Commands

The dedicated runner should expose explicit phases:

```powershell
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment13_strategy_grid.py --mkl-threading-layer SEQUENTIAL --native-threads 1 run-13a --async --backend xpu --metadata .\datasets\presetshare\raw\presetshare_vital_metadata.csv --corpus-sample-fraction 1.0 --monitor-refresh-seconds 30
```

After 13A analysis writes `epsilon_selection.json`:

```powershell
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment13_strategy_grid.py --mkl-threading-layer SEQUENTIAL --native-threads 1 run-13b --async --backend xpu --metadata .\datasets\presetshare\raw\presetshare_vital_metadata.csv --epsilon-selection .\research\experiments\lfo_representation\era2\artifacts\experiment_13\strategy_grid\epsilon_selection.json --monitor-refresh-seconds 30
```

The 13B command must fail if the selection artifact is absent, malformed, or does
not identify a completed 13A run.

## Test Requirements

Tests must verify:

- the fixed W8D16 runtime contract and 193-head accounting;
- 13A contains exactly the 90 `AllResiduals` rows;
- 13B contains exactly the 90 paired `UnresolvedOnly` rows;
- 13B cannot start before a valid completed 13A selection artifact exists;
- all pairs share identical settings except population policy and frozen epsilon;
- epsilon uses complete-curve maximum absolute error;
- slot-level eligibility is recomputed after every active atom slot;
- 13A construction never changes because of an observational epsilon;
- layer and slot quantiles are calculated correctly;
- candidate-epsilon coverage fractions are correct;
- retired residual-energy shares are correct;
- the epsilon selection rule is deterministic;
- broad atoms can differ from every observed residual;
- phase/gain-aligned builders do not raw-average shifted residuals;
- candidate budgets apply only to repair slots;
- pure-prototype policies require `Null`;
- empty 13B eligible populations terminate with remaining no-op atoms;
- no topology appears in runtime targets, loss, decoder lookup, or head accounting.

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

## Non-Goals

Experiment 13 does not:

- vary residual width or depth;
- compare scalar schemas;
- compare Beam4 and Beam8;
- screen no-damage, preprocessing, or duplicate-suppression policies;
- change the flat-categorical runtime interface;
- add runtime topology;
- train the audio-to-patch model;
- claim that any epsilon is perceptually final;
- use validation data to select epsilon;
- allow per-strategy epsilon selection;
- treat oracle construction time or codebook storage as prediction-head cost.

The experiment remains an oracle reconstruction and codebook-construction study.
Its output should identify both a stronger construction strategy and whether
residual-population filtering is useful enough to retain in later scaling work.
