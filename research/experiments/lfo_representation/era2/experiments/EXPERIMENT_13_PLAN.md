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

Experiment 13 must run in two ordered phases:

1. **Experiment 13A — unfiltered construction and calibration.** Run the 90
   `AllResiduals` rows. Candidate eligibility thresholds are observational only
   and must not change construction.
2. **Experiment 13B — filtered construction.** Select one global eligibility
   epsilon from completed 13A training artifacts, freeze it, and run the 90 paired
   `UnresolvedOnly` rows.

**Experiment 13B must not begin until Experiment 13A has completed and written a
valid threshold-selection artifact.** The paired phases are sequential, not one
simultaneous 180-row run.

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

For LFO `i` entering residual layer `d`, let:

```text
r_i,d = target_i - prefix_i,d-1
```

Let `A_d,s = {a_d,0, ..., a_d,s}` be the partial codebook after slot `s`, where
`a_d,0 = 0` is the no-op atom. Let `S_phi(a)` circularly phase-shift atom `a` by
`phi`. The current best layer reconstruction error is:

```text
E_i,d,s = min over a in A_d,s, phi, g of
          max_abs(r_i,d - g * S_phi(a))
```

The implementation may also retain the corresponding RMSE or MSE for utility
scoring, but eligibility membership and finish membership must use
complete-curve maximum absolute error. Later atom slots should focus on residual
patterns that remain poorly covered by the partial codebook.

Experiment 13 uses two deliberately separate thresholds:

```text
finish_threshold
    fixed construction threshold used by finish-oriented atom objectives

eligibility_epsilon
    threshold selected from Experiment 13A and used only by the Experiment 13B
    UnresolvedOnly population mask
```

Candidate eligibility epsilons must not alter 13A construction. Separating these
thresholds avoids making finish-oriented 13A rows depend on the value that 13A is
supposed to calibrate.

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
finish_threshold = 1e-5
eligibility_epsilon = selected_after_experiment_13a
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
3. Does excluding LFOs already within the calibrated eligibility epsilon improve
   later atom construction?
4. Which broad prototype family works best: aligned mean, trimmed mean, aligned
   median, cluster mean, dominant direction, or diversity-aware coverage?
5. Which repair objective pairs best with broad prototypes: global improvement,
   strict finishing, or hard-tail improvement?
6. Does `LayerClip0To1` help these strategy families consistently?
7. Does increasing repair candidate breadth from 24 to 48 still matter once
   broad atoms are synthesized?
8. How quickly does unfiltered construction move the training population through
   candidate eligibility thresholds?

## Residual Population Policies

Every applicable construction policy is tested under two residual-population
policies.

### `AllResiduals`

**Common intuitive description**

> Keep learning from the full population, including curves that are already
> reconstructed well.

**Technical description**

All training residuals remain eligible during atom construction. Policy-specific
weights may depend on current error, but no residual is hard-excluded because it
is already within a candidate eligibility epsilon. In Experiment 13A, every
candidate eligibility epsilon is observational only.

**Mathematical formulation**

```text
eligible_i^(s) = 1
```

for every residual `i` and active atom slot `s`.

### `UnresolvedOnly`

**Common intuitive description**

> Once a curve is good enough, stop allowing it to consume later codebook
> capacity.

**Technical description**

Only LFOs whose current best reconstruction is outside the globally selected
`eligibility_epsilon` remain eligible. Experiment 13B loads that frozen value
from the completed Experiment 13A selection artifact. The mask must be
recomputed after every active atom slot because a newly added atom can resolve
additional LFOs.

**Mathematical formulation**

```text
resolved_i,d,s = E_i,d,s <= eligibility_epsilon
eligible_i,d,s = 1 - resolved_i,d,s
```

All Experiment 13B rows use the same frozen `eligibility_epsilon`. If no
eligible residuals remain, the remaining active atom slots should be filled with
no-op atoms and recorded as early completion rather than treated as a failure.

## Shared Alignment and Utility Definitions

For an atom proposal `a` and residual `r_i`, define its best fitted phase and
gain as:

```text
(phi_i(a), g_i(a)) = argmin over phi, g of
                     loss(r_i, g * S_phi(a))
```

For utility scoring, let:

```text
L_i^(s) = current scalar loss for residual i before slot s
L_i(a)  = best scalar loss using proposal a
Delta_i(a) = max(0, L_i^(s) - L_i(a))
```

The scalar loss used by a utility policy should be recorded. The default should
match the Experiment 12 construction loss so comparisons remain grounded.
Eligibility remains a maximum-error test against `eligibility_epsilon`, while
finish-oriented construction remains a separate maximum-error test against the
fixed `finish_threshold`. Neither test may be implemented by comparing MSE with
a squared maximum-error threshold.

## Reusable Broad-Atom Builders

### `BroadMean`

**Common intuitive description**

> Line up the remaining mistakes, undo their fitted sizes, and average the shared
> correction.

**Technical description**

Start from a deterministic seed atom. Repeatedly align every eligible residual
to the current atom with the phase-and-residual-gain oracle, map each residual
back into the atom's canonical phase frame, and compute the weighted
least-squares prototype. Use current reconstruction error as a soft weight so
later broad slots emphasize residuals that remain poorly represented. Stop at a
fixed iteration cap or when atom change falls below a recorded tolerance.

The seed should be deterministic, for example the highest-weight eligible
residual after canonical phase normalization. The implementation must record the
seed rule, iteration count, and convergence status.

**Mathematical formulation**

For fixed assignments `(phi_i, g_i)` and nonnegative weights `w_i`, solve:

```text
min_a sum_i eligible_i * w_i * ||r_i - g_i S_phi_i(a)||_2^2
```

The least-squares update is:

```text
a = [sum_i eligible_i * w_i * g_i * S_-phi_i(r_i)]
    / [sum_i eligible_i * w_i * g_i^2]
```

Then refit `(phi_i, g_i)` against the updated atom and alternate.

### `TrimmedMean`

**Common intuitive description**

> Build the shared average, but do not let a small number of unusual residuals
> drag it away from the common pattern.

**Technical description**

Run the `BroadMean` alignment step, calculate each eligible residual's fitted
error to the current prototype, exclude the worst 10% by that fitted error, and
recompute the weighted least-squares mean from the retained 90%. Repeat the
align-trim-update cycle until convergence or the iteration cap.

Trimming must occur after phase/gain fitting, not on raw residual distance. Ties
at the trimming boundary should be resolved deterministically.

**Mathematical formulation**

Let `T` be the retained set containing the lowest 90% of aligned losses:

```text
T = lowest_90_percent_i loss(r_i, g_i S_phi_i(a))
```

Then update:

```text
a = [sum_{i in T} w_i * g_i * S_-phi_i(r_i)]
    / [sum_{i in T} w_i * g_i^2]
```

### `AlignedMedian`

**Common intuitive description**

> At each curve position, use the middle aligned correction rather than the
> average, so isolated extremes have less influence.

**Technical description**

Align eligible residuals into a common prototype frame. Undo fitted phase and,
where numerically safe, fitted gain. Compute a coordinate-wise weighted median
across the canonicalized residuals. Refit the prototype scale through the normal
gain oracle and repeat alignment plus median update until stable.

The implementation must define gain-floor handling so very small fitted gains do
not amplify noise when residuals are normalized back into prototype space.

**Mathematical formulation**

Let:

```text
z_i = S_-phi_i(r_i) / max(|g_i|, gain_floor)
```

For each control point `t`:

```text
a[t] = weighted_median_i(z_i[t], weight = eligible_i * w_i)
```

A deterministic global sign convention must be applied before the next
alignment iteration.

### `ClusterMean`

**Common intuitive description**

> Find one major family of remaining mistakes, average that family into a broad
> fix, then look for another family at the next broad slot.

**Technical description**

Construct deterministic phase/gain-invariant clusters over the eligible
residuals. Each cluster receives an aligned-mean prototype. Evaluate every
cluster prototype against the complete eligible population and choose the
prototype with the highest configured broad-slot utility. Recompute clustering
from the updated residual state at every broad slot.

The implementation should use a fixed cluster count rule or deterministic model
selection rule, record cluster sizes, reject clusters below a minimum size, and
avoid selecting the same cluster prototype repeatedly through a similarity
check internal to this builder.

**Mathematical formulation**

Using phase/gain-invariant distance:

```text
d(r_i, r_j) = min over phi, g ||r_i - g S_phi(r_j)||_2^2
```

partition eligible residuals into clusters `C_1, ..., C_K`. For each cluster,
solve:

```text
a_k = argmin_a sum_{i in C_k} w_i * min_{phi,g}
                    ||r_i - g S_phi(a)||_2^2
```

Then choose:

```text
a* = argmax_{a_k} Utility(a_k)
```

### `DominantDirection`

**Common intuitive description**

> Find the strongest recurring direction in which many current reconstructions
> are wrong, even when no single residual is a perfect representative.

**Technical description**

Canonicalize eligible residuals into a common phase and sign frame. Centering is
not applied unless explicitly required by the chosen PCA formulation, because
the zero vector has semantic meaning as no correction. Compute the leading
weighted principal direction, choose its sign deterministically, normalize it,
and let the residual-gain oracle determine per-LFO magnitude during evaluation.

Because phase canonicalization and principal-direction estimation depend on one
another, alternate them for a fixed number of iterations. Record explained
weighted energy and convergence.

**Mathematical formulation**

For canonicalized residuals `z_i = S_-phi_i(r_i)`, form:

```text
C = sum_i eligible_i * w_i * z_i z_i^T
```

Choose:

```text
a = argmax_{||a||_2 = 1} a^T C a
```

which is the leading eigenvector of `C`. Choose the sign using a deterministic
rule such as positive correlation with the highest-weight canonical residual.

### `DiverseCoverage`

**Common intuitive description**

> Help many remaining curves, but avoid spending several broad slots on nearly
> the same kind of correction.

**Technical description**

Generate a deterministic set of synthesized proposals from partitions or seeds
of the eligible residual population. Score each proposal by coverage, total
improvement, and dissimilarity from broad atoms already selected in the layer.
Coverage means the number or weighted share of eligible residuals improved by at
least a configured meaningful-improvement threshold. Similarity must be
phase-and-scale invariant.

The proposal count and partition rule are internal algorithm parameters and must
be fixed in the plan implementation, not crossed as `utility_candidate_budget`.

**Mathematical formulation**

Define:

```text
coverage(a) = sum_i eligible_i * 1[Delta_i(a) >= delta_min]
improvement(a) = sum_i eligible_i * w_i * Delta_i(a)
similarity(a, b) = max over phi of
                   |<a, S_phi(b)>| / (||a||_2 ||b||_2)
```

For previously selected broad atoms `B`, score:

```text
score(a) = alpha * coverage(a)
         + beta  * improvement(a)
         - lambda * max_{b in B} similarity(a, b)
```

Choose the highest-scoring proposal with deterministic tie-breaking.

## Reusable Repair-Atom Builders

Repair atoms are selected from observed eligible residuals. A repair candidate
may be phase-canonicalized for storage only if that canonicalization preserves
the same runtime phase/gain semantics and is applied consistently.

### `GlobalRepair`

**Common intuitive description**

> Pick the concrete residual example that removes the most total error across
> the remaining population.

**Technical description**

Build a deterministic shortlist of 24 or 48 observed eligible residuals. Evaluate
each candidate against every eligible target residual with best phase and gain.
Choose the candidate with the greatest weighted summed improvement. Candidate
shortlisting and final utility evaluation must remain separate and be recorded.

**Mathematical formulation**

For candidate set `C`:

```text
a* = argmax_{a in C} sum_i eligible_i * w_i * Delta_i(a)
```

### `FinishRepair`

**Common intuitive description**

> Use the repair slot to push as many almost-correct curves as possible across
> the fixed strict-finish threshold.

**Technical description**

For every observed residual candidate, count eligible LFOs that are outside the
fixed `finish_threshold` before the candidate and within it after adding the
candidate to the partial codebook. Maximize that newly finished count first.
Break ties with weighted total scalar-loss improvement, then deterministic
candidate order.

This construction objective is identical in 13A and its paired 13B row. It must
compute finishing with maximum absolute error, not MSE compared with
`finish_threshold^2`.

**Mathematical formulation**

```text
finish_i(a) = 1[E_i(A_s) > finish_threshold
                  and E_i(A_s union {a}) <= finish_threshold]
```

Choose lexicographically:

```text
argmax_a (
    sum_i eligible_i * finish_i(a),
    sum_i eligible_i * w_i * Delta_i(a)
)
```

### `HardRepair`

**Common intuitive description**

> Spend the repair slot on the worst remaining curves instead of letting easy
> cases dominate the decision.

**Technical description**

Rank eligible residuals by their current reconstruction loss, define the hard
tail as the worst 10%, and choose the observed residual candidate with the
largest weighted summed improvement on that tail. Recompute the hard-tail set at
every repair slot. Break ties with improvement over the full eligible population.

**Mathematical formulation**

Let `H_s` be the eligible residuals at or above the 90th percentile of current
loss. Choose lexicographically:

```text
argmax_a (
    sum_{i in H_s} w_i * Delta_i(a),
    sum_i eligible_i * w_i * Delta_i(a)
)
```

## Mixed Slot Schedules

Every new broad-plus-repair recipe is tested with both schedules below.

### `Interleaved`

**Common intuitive description**

> Alternate a broad population-level fix with targeted cleanup.

**Technical description**

Construct slots in this order:

```text
slot 1 = Broad
slot 2 = Repair
slot 3 = Broad
slot 4 = Repair
slot 5 = Broad
slot 6 = Repair
slot 7 = Broad
```

Every slot uses the residual state and eligible population produced by all
previous slots.

### `TwoPhase`

**Common intuitive description**

> Establish broad coverage first, then spend the remaining slots cleaning up
> what broad prototypes could not solve.

**Technical description**

Construct slots in this order:

```text
slot 1 = Broad
slot 2 = Broad
slot 3 = Broad
slot 4 = Broad
slot 5 = Repair
slot 6 = Repair
slot 7 = Repair
```

Both schedules use four broad slots and three repair slots. This isolates order
without changing the broad/repair ratio.

## Construction Policies

### Existing Experiment 12 anchors

Keep these observed-residual policies as historical anchors. Their exact
Experiment 12 role schedules and aggregation semantics should be preserved,
except that the `UnresolvedOnly` variant applies the dynamic eligibility mask
defined above and every finish test must be corrected to use complete-curve
maximum absolute error against the fixed `finish_threshold`.

#### `CommonCaseRepair`

**Common intuitive description**

> Prefer atoms that improve the typical residual, even if they do not target the
> worst outliers.

**Technical description**

Select each observed residual atom by the existing Experiment 12 common-case
utility, applied to the current eligible population. Preserve its original
aggregation and tie-breaking rules so the row remains a valid anchor.

**Mathematical formulation**

Use the exact Experiment 12 score. If that implementation is median-oriented,
record it explicitly as a robust central utility such as:

```text
score(a) = median over eligible i of Delta_i(a)
```

Do not silently substitute this illustrative form for the actual existing score.

#### `FinishRepairRescue`

**Common intuitive description**

> Finish easy near-perfect cases first, improve common cases in the middle, and
> reserve late capacity for difficult leftovers.

**Technical description**

Preserve the exact Experiment 12 role schedule and role-specific scoring. Apply
roles to the current eligible population and recompute all masks and hard-tail
sets after each slot.

**Mathematical formulation**

Represent the existing schedule as slot-specific utility functions:

```text
a_s = argmax_{a in C_s} U_role(s)(a)
```

where `U_role(s)` is the Experiment 12 finish, common-case, or rescue score for
that slot. The implementation must document the exact slot-to-role mapping.

#### `FamilyBalancedRepair`

**Common intuitive description**

> Avoid letting a large residual family dominate; choose atoms that improve
> several families more evenly.

**Technical description**

Preserve the Experiment 12 family construction or grouping and its family-level
utility aggregation. Recompute family statistics over the current eligible
population while keeping the family definition itself stable.

**Mathematical formulation**

If families are `F_1, ..., F_K`, use the exact existing score. A representative
form is:

```text
score(a) = aggregate_k [ mean_{i in F_k and eligible} Delta_i(a) ]
```

where `aggregate_k` is the Experiment 12 balancing operator. Record the actual
operator rather than replacing it with an undocumented mean.

### New mixed prototype/repair recipes

Each recipe below receives both `Interleaved` and `TwoPhase` variants. The
schedule determines which builder runs in each slot; the builder definitions
above determine the actual optimization.

#### `BroadMeanGlobalRepair`

**Common intuitive description**

> Repeatedly combine a shared average correction with a concrete example that
> removes the most total remaining error.

**Technical description**

Broad slots use `BroadMean`. Repair slots use `GlobalRepair`. This is the most
direct test of whether smooth population-level corrections and sharp observed
examples complement one another.

**Mathematical formulation**

```text
Broad slot:  solve weighted aligned least squares for a
Repair slot: maximize sum_i eligible_i * w_i * Delta_i(a)
```

#### `BroadMeanFinishRepair`

**Common intuitive description**

> Make broad progress, then use concrete repairs to push near-correct curves
> inside the fixed strict-finish threshold.

**Technical description**

Broad slots use `BroadMean`. Repair slots use `FinishRepair`. This policy tests
whether broad prototypes create a large population of almost-solved residuals
that targeted finishing can convert into strict-perfect reconstructions. The
finish objective uses the fixed `finish_threshold`, not the calibrated eligibility
epsilon.

**Mathematical formulation**

```text
Broad slot:  solve weighted aligned least squares for a
Repair slot: lexicographically maximize newly_resolved_count, total_improvement
```

#### `BroadMeanHardRepair`

**Common intuitive description**

> Use broad averages for general progress, then use concrete examples to rescue
> the worst remaining cases.

**Technical description**

Broad slots use `BroadMean`. Repair slots use `HardRepair`. This separates broad
central coverage from explicit tail control.

**Mathematical formulation**

```text
Broad slot:  solve weighted aligned least squares for a
Repair slot: maximize improvement over current worst-loss decile
```

#### `TrimmedMeanGlobalRepair`

**Common intuitive description**

> Use an outlier-resistant broad average, then choose the concrete correction
> that removes the most total residual error.

**Technical description**

Broad slots use `TrimmedMean`. Repair slots use `GlobalRepair`. This tests
whether ordinary aligned means are being distorted by a small number of unusual
residuals.

**Mathematical formulation**

```text
Broad slot:  aligned least squares over retained lowest-loss 90%
Repair slot: maximize weighted summed Delta_i(a)
```

#### `AlignedMedianGlobalRepair`

**Common intuitive description**

> Build a very robust middle-shaped prototype, then use concrete examples for
> global cleanup.

**Technical description**

Broad slots use `AlignedMedian`. Repair slots use `GlobalRepair`. This is a
stronger robustness test than trimming because each control point uses a median
rather than a mean.

**Mathematical formulation**

```text
Broad slot:  coordinate-wise weighted median in canonical phase/gain frame
Repair slot: maximize weighted summed Delta_i(a)
```

#### `ClusterMeanGlobalRepair`

**Common intuitive description**

> Cover one common family of mistakes at a time, then choose concrete corrections
> that remove the most error left across all families.

**Technical description**

Broad slots use `ClusterMean`. Repair slots use `GlobalRepair`. Clustering is
recomputed before every broad slot from the current eligible residuals.

**Mathematical formulation**

```text
Broad slot:  choose highest-utility aligned cluster prototype a_k
Repair slot: maximize weighted summed Delta_i(a)
```

#### `ClusterMeanHardRepair`

**Common intuitive description**

> Cover the main residual families broadly, then use concrete examples to rescue
> the difficult leftovers.

**Technical description**

Broad slots use `ClusterMean`. Repair slots use `HardRepair`. This policy tests a
coarse family-coverage phase combined with explicit tail protection.

**Mathematical formulation**

```text
Broad slot:  choose highest-utility aligned cluster prototype a_k
Repair slot: maximize improvement over current worst-loss decile
```

#### `DominantDirectionGlobalRepair`

**Common intuitive description**

> Remove the strongest shared direction of error, then clean up with the best
> concrete residual examples.

**Technical description**

Broad slots use `DominantDirection`. Repair slots use `GlobalRepair`. The broad
atom may not resemble any single residual; it represents a high-energy shared
correction direction.

**Mathematical formulation**

```text
Broad slot:  leading eigenvector of weighted canonical residual covariance
Repair slot: maximize weighted summed Delta_i(a)
```

#### `DiverseCoverageHardRepair`

**Common intuitive description**

> Spend broad slots on several different widely useful corrections, then spend
> repair slots on the worst cases still left.

**Technical description**

Broad slots use `DiverseCoverage`. Repair slots use `HardRepair`. The broad score
must penalize phase/scale similarity to earlier broad atoms in the same layer.

**Mathematical formulation**

```text
Broad slot:  maximize alpha*coverage + beta*improvement - lambda*similarity
Repair slot: maximize improvement over current worst-loss decile
```

PascalCase row identifiers append the schedule, for example:

```text
BroadMeanGlobalRepairInterleaved
BroadMeanGlobalRepairTwoPhase
ClusterMeanHardRepairInterleaved
ClusterMeanHardRepairTwoPhase
```

### Pure-prototype controls

#### `AllBroadAlignedMeans`

**Common intuitive description**

> Build the entire active codebook from broad aligned averages and never copy an
> individual residual as an atom.

**Technical description**

All seven active slots run `BroadMean` sequentially. The eligible population and
current losses are updated after every slot. Each new mean must be constructed
against the current partial codebook, not precomputed once from the original
layer residuals.

**Mathematical formulation**

For every slot `s = 1..7`:

```text
a_s = argmin_a sum_i eligible_i^(s) * w_i^(s)
                 * min_{phi,g} ||r_i - g S_phi(a)||_2^2
```

using the alternating `BroadMean` solver.

#### `AllClusterMeans`

**Common intuitive description**

> Fill the whole codebook with representative prototypes of residual families.

**Technical description**

All seven active slots use `ClusterMean`. Recluster the current eligible
residual population at every slot and select the highest-utility cluster
prototype not already adequately represented by the partial codebook.

**Mathematical formulation**

For every slot, form clusters `C_k`, solve aligned prototype `a_k` for each, and
choose `argmax_k Utility(a_k)`.

#### `AllDominantDirections`

**Common intuitive description**

> Fill the whole codebook with successive shared directions of residual error.

**Technical description**

All seven active slots use `DominantDirection`. After selecting each direction,
update current best reconstruction losses before estimating the next direction.
This is not a one-shot seven-component PCA decomposition unless that produces
identical greedy semantics.

**Mathematical formulation**

For every slot, recompute the weighted canonical residual covariance under the
current partial codebook and select its leading direction.

All pure-prototype policies use `utility_candidate_budget = Null`.

## Utility Candidate Budget

The main grid recognizes:

```text
CandidateBudget24
CandidateBudget48
Null
```

`CandidateBudget24` and `CandidateBudget48` apply only to repair slots that
select observed residual examples. Broad synthesized slots always use `Null`.
Artifacts must record both the row-level budget and the effective budget for
each slot.

For example:

```text
Interleaved, CandidateBudget48:
[Null, 48, Null, 48, Null, 48, Null]

TwoPhase, CandidateBudget48:
[Null, Null, Null, Null, 48, 48, 48]
```

## Layer Normalization

Only two values remain in the main grid:

```text
FinalClipOnly
LayerClip0To1
```

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
finish_threshold = 1e-5
```

Crossed axes:

```text
construction_policy
residual_population_policy = AllResiduals | UnresolvedOnly
utility_candidate_budget = CandidateBudget24 | CandidateBudget48 | Null
layer_normalization_policy = FinalClipOnly | LayerClip0To1
```

The logical design remains paired across the population-policy axis, but
execution is ordered:

```text
Experiment 13A = 90 AllResiduals rows
Experiment 13B = 90 paired UnresolvedOnly rows
```

Every pair must share a stable `pair_id`. Paired rows must match on construction
policy, slot schedule, repair candidate budget, layer normalization, fixed
settings, seed rules, and `finish_threshold`. They differ only in experiment
phase, residual-population behavior, and the presence of the frozen
`eligibility_epsilon` mask.

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
main-grid rows, so only eight additional executions are required. The aside
should be reported separately and should not determine the main strategy
ranking.

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
prototype_objective_before
prototype_objective_after
prototype_seed_rule
cluster_count
cluster_size
explained_weighted_energy
repair_source_dataset_index
repair_shortlist_rule
repair_utility_score
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

The report should not collapse the result into one automatic scalar ranking. It
should identify Pareto candidates across the co-primary metrics and explain the
tradeoffs.

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
- `AllResiduals` and `UnresolvedOnly` mask behavior;
- epsilon uses complete-curve maximum absolute error;
- the unresolved mask is recomputed after every slot;
- broad atoms can differ from every observed residual;
- phase/gain-aligned prototype updates do not raw-average shifted residuals;
- each broad builder follows its documented objective and deterministic tie rules;
- repair candidate budgets apply only to repair slots;
- finish scoring uses maximum absolute epsilon rather than MSE `<= epsilon^2`;
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
- establish a perceptually final eligibility epsilon;
- use validation results to select the eligibility epsilon;
- allow strategy-, layer-, or slot-specific eligibility epsilons;
- treat oracle construction time or codebook storage as model prediction-head
  cost.

The experiment is an oracle reconstruction and codebook-construction study. Its
result should choose a stronger construction recipe before returning to
representation scaling or learned-model predictability.
