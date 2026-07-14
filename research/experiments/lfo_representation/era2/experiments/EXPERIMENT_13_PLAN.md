# Experiment 13 Plan: Fixed-W8D16 Residual Construction Strategy Grid

## Summary

Experiment 13 tests how the seven active atoms in each residual-layer codebook
should be constructed. The runtime representation stays fixed. The experiment
compares broad population-derived prototypes, targeted observed-residual repair
atoms, and different schedules for combining those roles.

Experiment 13 must run in two ordered phases:

1. **Experiment 13A — calibration and unfiltered construction.** Run every
   `AllResiduals` row. Do not remove any LFO because it is already within an
   epsilon threshold. Measure reconstruction-error distributions after every
   residual layer and every active atom slot.
2. **Experiment 13B — filtered construction.** Use the 13A measurements to choose
   one global epsilon, then run the paired `UnresolvedOnly` rows. Recompute the
   eligible population after every active atom slot.

**Experiment 13B must not start until 13A has completed and written both its
calibration report and `epsilon_selection.json`.** Epsilon is not a free constant
chosen before the experiment. It is calibrated from 13A and then frozen for 13B.

The main research question is:

> Can broad population-derived prototype atoms, mixed with targeted repair atoms,
> produce stronger W8D16 reconstruction than observed-residual-only construction?

The epsilon question is:

> At each layer and atom slot, how much of the training population is already
> reconstructed well enough that removing it from later construction preserves
> nearly all useful residual-error mass?

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
model-facing target schema, decoder interface, or head accounting.

## Core Reconstruction Notation

For LFO $i$ entering residual layer $d$, define the incoming residual as

$$
r_{i,d}=y_i-\hat y_{i,d-1}.
$$

Let $A_{d,s}=\{a_{d,0},\ldots,a_{d,s}\}$ be the partial layer codebook after
active slot $s$, with $a_{d,0}=0$ as the no-op atom. Let $S_\phi(a)$ circularly
shift atom $a$ by phase $\phi$. The partial-codebook maximum-absolute residual
error is

$$
E_{i,d,s}
=
\min_{a\in A_{d,s},\,\phi,\,g}
\left\|r_{i,d}-gS_\phi(a)\right\|_\infty.
$$

The full-curve error after residual layer $d$ is

$$
G_{i,d}=\left\|y_i-\hat y_{i,d}\right\|_\infty.
$$

Epsilon membership must use maximum absolute error. RMSE or MSE may be used for
candidate utility scoring, but not as a substitute for $E_{i,d,s}$ or $G_{i,d}$.

## Ordered Execution Contract

### Experiment 13A — `AllResiduals`

13A runs all strategies with

```text
residual_population_policy = AllResiduals
```

For every training residual, layer, and slot,

$$
\operatorname{eligible}^{13A}_{i,d,s}=1.
$$

No epsilon changes construction in 13A. Candidate epsilon values are used only
for measurement and plotting.

13A must:

1. establish the unfiltered quality baseline for every strategy;
2. measure how errors fall through the residual ladder;
3. estimate how many LFOs each epsilon would retire at each checkpoint;
4. measure how much residual-error mass those retired LFOs contain;
5. select one global epsilon for 13B using the fixed rule below.

### Experiment 13B — `UnresolvedOnly`

13B runs paired rows with

```text
residual_population_policy = UnresolvedOnly
```

Given the selected threshold $\epsilon^*$,

$$
\operatorname{eligible}^{13B}_{i,d,s}
=
\mathbf{1}\!\left[E_{i,d,s}>\epsilon^*\right].
$$

Recompute this mask after every active atom slot, not only after a complete
residual layer. If no eligible residuals remain, fill all remaining active slots
in that layer with no-op atoms and record early completion.

All 13B rows must use the same frozen $\epsilon^*$. Do not select a separate
threshold per strategy.

## Experiment 13A Calibration Measurements

### Layer-level distributions

After the base dictionary and after each of the 16 completed residual layers,
record $G_{i,d}$ on both training and validation splits. Layer 0 is the base-only
reconstruction.

Record these quantiles:

```text
50th percentile
25th percentile
10th percentile
5th percentile
2nd percentile
1st percentile
0.1st percentile, only when the split has at least 1000 LFOs
```

For percentile $p$,

$$
Q^{\mathrm{global}}_d(p)
=
\operatorname{Quantile}_i\!\left(G_{i,d},p\right).
$$

$Q^{\mathrm{global}}_d(0.10)$ is the epsilon that would classify approximately
10% of LFOs as reconstructed after layer $d$.

### Slot-level distributions

Because filtering acts inside each layer, layer-level measurements are not enough.
After every active atom slot, record the training distribution of $E_{i,d,s}$:

$$
Q^{\mathrm{slot}}_{d,s}(p)
=
\operatorname{Quantile}_i\!\left(E_{i,d,s},p\right).
$$

This is the primary calibration dataset for 13B.

### Fixed-epsilon coverage

For each candidate threshold and checkpoint,

$$
F_{d,s}(\epsilon)
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}\!\left[E_{i,d,s}\leq\epsilon\right].
$$

Evaluate at least:

```text
0.001
0.0025
0.005
0.01
0.02
```

These thresholds are observational in 13A and do not alter construction.

### Retired residual-error mass

For each checkpoint and epsilon, record the fraction of residual energy held by
LFOs that would be retired:

$$
M_{d,s}(\epsilon)
=
\frac{
\sum_{i:E_{i,d,s}\leq\epsilon}\left\|r_{i,d}\right\|_2^2
}{
\sum_i\left\|r_{i,d}\right\|_2^2
}.
$$

Also record the retained fraction $1-M_{d,s}(\epsilon)$. The desirable regime is
a high retired-LFO fraction with a low retired-error-mass fraction.

## Epsilon Selection Rule

Use training data only. Validation data must not influence $\epsilon^*$.

1. Evaluate `0.001`, `0.0025`, `0.005`, `0.01`, and `0.02`.
2. Aggregate slot-level measurements across all 13A rows.
3. Choose the largest epsilon satisfying both

$$
\operatorname{Median}_{\text{rows, slots}}
M_{d,s}(\epsilon)
\leq 0.01,
$$

and

$$
Q_{0.95,\,\text{rows, slots}}
\!\left(M_{d,s}(\epsilon)\right)
\leq 0.05.
$$

4. Require that the selected threshold retires a nontrivial population at at
   least one early or middle checkpoint. If every qualifying threshold retires
   less than 1% everywhere, choose the largest qualifying value and record that
   filtering has little practical effect.
5. Write the decision and supporting statistics to `epsilon_selection.json`.

If no threshold satisfies the rule, do not silently loosen it. Record the failure
and run a small 13B pilot at the two tightest thresholds before proceeding.

## Required Calibration Plots

13A must produce these plots before 13B starts:

1. **Epsilon quantiles by completed layer** — layer `0..16` on the x-axis,
   maximum-absolute-error epsilon on the y-axis, with quantile lines.
2. **Epsilon quantiles by atom slot** — slot `0..7` on the x-axis, $E_{i,d,s}$
   on the y-axis, with quantile lines.
3. **Fraction reconstructed by layer** — one line per candidate epsilon.
4. **Fraction reconstructed by atom slot** — one line per candidate epsilon.
5. **Retired count versus retired error mass** — retired fraction
   $F_{d,s}(\epsilon)$ on the x-axis and $M_{d,s}(\epsilon)$ on the y-axis.

Detailed row-level curves remain in artifacts. The report should show median
curves and percentile bands across rows, plus grouped views by strategy family.

## Residual Population Policies

### `AllResiduals`

**Common intuitive description**

> Keep learning from every residual, including residuals already reconstructed
> well.

**Technical description**

No residual is hard-excluded. Strategy-specific soft weights may still emphasize
larger errors.

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
slot.

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

Let $L_i^{(s)}$ be the current scalar construction loss before slot $s$, and let
$L_i(a)$ be the best loss using proposal $a$. Candidate improvement is

$$
\Delta_i(a)
=
\max\!\left(0,L_i^{(s)}-L_i(a)\right).
$$

The construction loss must be recorded. Epsilon membership remains based on
$E_{i,d,s}$.

## Broad-Atom Builders

### `BroadMean`

**Common intuitive description**

> Line up the residuals, account for their fitted sizes, and average the shared
> correction.

**Technical description**

Initialize deterministically. Alternate phase/gain alignment with a weighted
least-squares prototype update. Use current error as a soft weight. Stop at a
fixed iteration cap or convergence tolerance.

**Mathematical formulation**

For fixed $(\phi_i,g_i)$ and weights $w_i\geq0$, minimize

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

> Average the common pattern without allowing a few unusual residuals to pull the
> prototype away from the majority.

**Technical description**

Run the aligned-mean step, remove the worst 10% of eligible aligned losses, and
recompute from the retained 90%. Repeat until convergence or the iteration cap.

**Mathematical formulation**

If $T$ is the retained lowest-loss 90%,

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
z_i
=
\frac{S_{-\phi_i}(r_i)}{\max\!\left(|g_i|,g_{\min}\right)}.
$$

For control point $t$,

$$
a[t]
=
\operatorname{WeightedMedian}_i
\!\left(z_i[t];e_iw_i\right).
$$

### `ClusterMean`

**Common intuitive description**

> Find a major family of remaining mistakes, average that family into one broad
> correction, then search for another family at the next broad slot.

**Technical description**

Cluster eligible residuals with a phase/gain-invariant distance. Build an aligned
mean for each viable cluster, score each prototype against the complete eligible
population, select the highest-utility prototype, and recluster at every broad
slot.

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

Choose

$$
a^*=\arg\max_{a_k}\operatorname{Utility}(a_k).
$$

### `DominantDirection`

**Common intuitive description**

> Find the strongest recurring direction in which many current reconstructions
> are wrong, even if no single residual is the ideal example.

**Technical description**

Canonicalize phase and sign, compute the leading weighted uncentered principal
direction, choose sign deterministically, and let the gain oracle fit magnitude.

**Mathematical formulation**

For $z_i=S_{-\phi_i}(r_i)$,

$$
C=\sum_i e_iw_iz_iz_i^\top,
$$

and

$$
a
=
\arg\max_{\left\|a\right\|_2=1}
a^\top Ca.
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
\sum_i e_i\mathbf{1}\!\left[\Delta_i(a)\geq\delta_{\min}\right],
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
\frac{
\left|\left\langle a,S_\phi(b)\right\rangle\right|
}{
\left\|a\right\|_2\left\|b\right\|_2
},
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

**Mathematical formulation**

$$
a^*
=
\arg\max_a
\sum_i e_i\Delta_i(a).
$$

### `FinishRepair`

**Common intuitive description**

> Prefer the repair that moves the most residuals under the current threshold.

**Technical description**

Maximize newly resolved residual count and break ties with total improvement. For
13B, use $\epsilon^*$. Existing Experiment 12 anchors retain their historical
13A semantics for comparability.

**Mathematical formulation**

$$
\operatorname{finish}(a)
=
\sum_i e_i
\mathbf{1}\!\left[E_i^{\mathrm{before}}>\epsilon^*\right]
\mathbf{1}\!\left[E_i(a)\leq\epsilon^*\right].
$$

### `HardRepair`

**Common intuitive description**

> Spend the repair slot on the worst remaining residuals.

**Technical description**

Define the hard set as the worst 10% of eligible residuals by current scalar loss
and maximize summed improvement on that set.

**Mathematical formulation**

$$
H
=
\left\{i:L_i^{(s)}\geq Q_{0.90}\!\left(L^{(s)}\right)\right\},
$$

$$
a^*
=
\arg\max_a
\sum_{i\in H}e_i\Delta_i(a).
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

Both schedules use four broad and three repair slots.

## Construction Policies

### Experiment 12 anchors

Retain these exact observed-residual policies as historical anchors:

```text
CommonCaseRepair
FinishRepairRescue
FamilyBalancedRepair
```

Preserve their existing implementation semantics. The only new behavior in 13B
is the eligibility mask applied to the construction population.

### New mixed recipes

Each recipe receives `Interleaved` and `TwoPhase` variants:

| Recipe | Broad builder | Repair builder | Intention |
|---|---|---|---|
| `BroadMeanGlobalRepair` | `BroadMean` | `GlobalRepair` | Shared average plus total cleanup. |
| `BroadMeanFinishRepair` | `BroadMean` | `FinishRepair` | Shared average plus threshold finishing. |
| `BroadMeanHardRepair` | `BroadMean` | `HardRepair` | Shared average plus tail rescue. |
| `TrimmedMeanGlobalRepair` | `TrimmedMean` | `GlobalRepair` | Robust average plus total cleanup. |
| `AlignedMedianGlobalRepair` | `AlignedMedian` | `GlobalRepair` | Robust median plus total cleanup. |
| `ClusterMeanGlobalRepair` | `ClusterMean` | `GlobalRepair` | Family prototypes plus total cleanup. |
| `ClusterMeanHardRepair` | `ClusterMean` | `HardRepair` | Family prototypes plus tail rescue. |
| `DominantDirectionGlobalRepair` | `DominantDirection` | `GlobalRepair` | Shared direction plus total cleanup. |
| `DiverseCoverageHardRepair` | `DiverseCoverage` | `HardRepair` | Distinct coverage plus tail rescue. |

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

```text
Interleaved, CandidateBudget48:
[Null, 48, Null, 48, Null, 48, Null]

TwoPhase, CandidateBudget48:
[Null, Null, Null, Null, 48, 48, 48]
```

## Grid Size and Phase Split

The complete paired design remains 180 rows:

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

Every pair must share a stable `pair_id`.

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

## Per-Slot Diagnostics

Record:

```text
phase = 13A | 13B
pair_id
residual_layer
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

For 13A, the actual eligible count remains the full count. Separately record the
counterfactual resolved fraction for every candidate epsilon.

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

The report must be findings-first and divided into:

1. 13A construction results;
2. 13A epsilon calibration;
3. the epsilon decision and exact supporting statistics;
4. 13B paired filtered results;
5. strategy interactions.

Do not imply that 13A and 13B ran simultaneously. State clearly that 13A chose
one global threshold and 13B used the frozen value. Do not collapse results into
one automatic scalar ranking.

## Run Commands

The runner should expose explicit phases:

```powershell
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment13_strategy_grid.py --mkl-threading-layer SEQUENTIAL --native-threads 1 run-13a --async --backend xpu --metadata .\datasets\presetshare\raw\presetshare_vital_metadata.csv --corpus-sample-fraction 1.0 --monitor-refresh-seconds 30
```

After `epsilon_selection.json` exists:

```powershell
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment13_strategy_grid.py --mkl-threading-layer SEQUENTIAL --native-threads 1 run-13b --async --backend xpu --metadata .\datasets\presetshare\raw\presetshare_vital_metadata.csv --epsilon-selection .\research\experiments\lfo_representation\era2\artifacts\experiment_13\strategy_grid\epsilon_selection.json --monitor-refresh-seconds 30
```

The 13B command must fail if the selection artifact is absent, malformed, or not
linked to a completed 13A run.

## Test Requirements

Tests must verify:

- fixed W8D16 runtime contract and 193-head accounting;
- exactly 90 13A and 90 paired 13B rows;
- 13B cannot start before a valid completed 13A selection artifact exists;
- all pairs differ only in population policy and frozen epsilon behavior;
- epsilon uses complete-curve maximum absolute error;
- eligibility is recomputed after every active atom slot;
- observational epsilon values never change 13A construction;
- layer and slot quantiles are correct;
- coverage fractions and retired energy shares are correct;
- epsilon selection is deterministic;
- broad atoms may differ from every observed residual;
- aligned builders do not raw-average shifted residuals;
- candidate budgets apply only to repair slots;
- pure-prototype policies require `Null`;
- empty 13B populations terminate with remaining no-op atoms;
- no topology appears in runtime targets, loss, decoder lookup, or accounting.

## Deferred Backlog

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
- compare scalar schemas or beam widths;
- screen no-damage, preprocessing, or duplicate-suppression policies;
- change the flat-categorical runtime interface;
- add runtime topology;
- train the audio-to-patch model;
- claim any epsilon is perceptually final;
- use validation data to select epsilon;
- allow per-strategy epsilon selection;
- treat oracle construction time or storage as prediction-head cost.

The experiment remains an oracle reconstruction and codebook-construction study.
Its output should identify both a stronger construction strategy and whether
residual-population filtering is useful enough to retain in later scaling work.
