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
2. **Experiment 13B — filtered construction and epsilon sweep.** Run each of
   the 45 `UnresolvedOnly` rows paired to the 13A `LayerClip0To1` strategies at
   eligibility epsilons `1e-2`, `1e-3`, and `1e-4`, producing 135 rows. The
   `FinalClipOnly` variants are not repeated because 13A favored layer-wise
   clipping in all 45 matched P95 comparisons.

**Experiment 13B must not begin until Experiment 13A has completed and written a
valid threshold-selection artifact.** The artifact records the completed 13A
calibration result, but it no longer needs to select one passing epsilon. The
three-value 13B sweep is exploratory and was fixed after 13A completed; `1e-4`
was not part of the original 13A calibration candidate set.

## Operator Quick Start

Run these commands from the repository root in PowerShell. Experiment 13A,
epsilon calibration, and Experiment 13B are ordered; do not start 13B until 13A
is complete and the matching `epsilon_selection.json` has been written.

Define the paths once for the current shell:

```powershell
$runner = ".\research\experiments\lfo_representation\era2\code\experiment13_strategy_grid.py"
$metadata = ".\datasets\presetshare\raw\presetshare_vital_metadata.csv"
$runDir = ".\research\experiments\lfo_representation\era2\artifacts\experiment_13\strategy_grid_train50_val100_exactopt_v1"
$cacheDir = ".\research\experiments\lfo_representation\era2\artifacts\experiment_13\cache_exactopt_v1"
$legacyRun = ".\research\experiments\lfo_representation\era2\artifacts\experiment_13\strategy_grid_train100_val100_interrupted_39rows_20260716"
$selection = "$runDir\epsilon_selection.json"
```

### Build the provisional report from the preserved fragment

This command reads the immutable 39-row legacy Experiment 13A archive and
writes all derived CSVs, plots, Markdown, and the self-contained interactive
HTML report outside that source directory. The result is explicitly
provisional: it cannot select the eligibility epsilon or stand in for the
canonical report from completed 13A and 13B phases.

```powershell
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment13_strategy_grid.py analyze-partial `
  --run-dir .\research\experiments\lfo_representation\era2\artifacts\experiment_13\strategy_grid_train100_val100_interrupted_39rows_20260716 `
  --analysis-output-dir .\research\experiments\lfo_representation\era2\artifacts\experiment_13\analysis_legacy39_provisional `
  --report-path .\research\experiments\lfo_representation\era2\reports\EXPERIMENT_13_W8D16_STRATEGY_GRID_PROVISIONAL.md `
  --html-report-path .\research\experiments\lfo_representation\era2\reports\EXPERIMENT_13_W8D16_STRATEGY_GRID_PROVISIONAL.html `
  --image-dir .\research\experiments\lfo_representation\era2\reports\images\experiment_13\provisional
```

Preview the interactive report without a build step:

```powershell
conda run --no-capture-output -n py312 python -m http.server 8765 `
  --directory .\research\experiments\lfo_representation\era2\reports
```

Then open
`http://localhost:8765/EXPERIMENT_13_W8D16_STRATEGY_GRID_PROVISIONAL.html`.
The report is one HTML file with inline CSS, JavaScript, and compact data; only
the pinned ECharts renderer is loaded from jsDelivr.

### Build the complete Experiment 13A report

After all 90 Experiment 13A rows complete, record the official automatic
epsilon-selection result and generate the complete AllResiduals report. This
report is authoritative for 13A but remains separate from the canonical paired
report, which is still gated on the complete fixed epsilon sweep in 13B.

```powershell
conda run --no-capture-output -n py312 python $runner --mkl-threading-layer SEQUENTIAL --native-threads 1 select-epsilon --run-dir $runDir

conda run --no-capture-output -n py312 python $runner --mkl-threading-layer SEQUENTIAL --native-threads 1 replay-strict-thresholds `
  --run-dir $runDir `
  --output-dir .\research\experiments\lfo_representation\era2\artifacts\experiment_13\analysis_train50_val100_13a_complete `
  --cache-dir .\research\experiments\lfo_representation\era2\artifacts\experiment_13\cache_exactopt_v1 `
  --backend xpu

conda run --no-capture-output -n py312 python $runner analyze-13a `
  --run-dir $runDir `
  --analysis-output-dir .\research\experiments\lfo_representation\era2\artifacts\experiment_13\analysis_train50_val100_13a_complete `
  --report-path .\research\experiments\lfo_representation\era2\reports\EXPERIMENT_13_W8D16_STRATEGY_GRID_13A_REPORT.md `
  --html-report-path .\research\experiments\lfo_representation\era2\reports\EXPERIMENT_13_W8D16_STRATEGY_GRID_13A_REPORT.html `
  --image-dir .\research\experiments\lfo_representation\era2\reports\images\experiment_13\13a `
  --scaling-baseline-run $legacyRun `
  --strict-thresholds-path .\research\experiments\lfo_representation\era2\artifacts\experiment_13\analysis_train50_val100_13a_complete\strict_perfect_threshold_sweep.csv
```

The threshold replay uses the saved codebooks and deterministic validation
sample only. It does not reconstruct dictionaries, repeat candidate search, or
rerun training encoding. The report exposes `1e-2`, `1e-3`, `1e-4`, and `1e-5` controls;
each requires RMSE at most one tenth of the selected tolerance and maximum
absolute point error at most the selected tolerance. `1e-5` is the original
strict-perfect definition. A log-scale sensitivity plot shows construction-family
median and in-scope envelope behavior across all four tolerance tuples.

Preview it with the same local HTTP server command above, then open
`http://localhost:8765/EXPERIMENT_13_W8D16_STRATEGY_GRID_13A_REPORT.html`.
The training-data scaling section uses only quality metrics from the 39 matched
legacy rows with identical validation membership. It never compares legacy and
optimized runtime.

### 1. Run the tests

Set conservative native-thread limits before importing NumPy, then run either
the Experiment 13 tests or the complete Era 2 suite:

```powershell
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

# Experiment 13 execution and reporting
conda run --no-capture-output -n py312 python -B -m unittest discover -v -s .\research\experiments\lfo_representation\era2\tests -p 'test_strategy_grid*.py'

# Complete Era 2 suite
conda run --no-capture-output -n py312 python -B -m unittest discover -v -s .\research\experiments\lfo_representation\era2\tests -p 'test_*.py'
```

### 2. Run a safe smoke check

Use a dedicated smoke directory. A fresh non-resume 13A run resets aggregate
state in its output directory, so a smoke check must not target `$runDir`.

```powershell
$smokeDir = ".\research\experiments\lfo_representation\era2\artifacts\experiment_13\strategy_grid_exactopt_smoke"
conda run --no-capture-output -n py312 python $runner --mkl-threading-layer SEQUENTIAL --native-threads 1 run-13a --output-dir $smokeDir --cache-dir $cacheDir --backend auto --metadata $metadata --smoke
```

The runner acquires a scoped Windows system-required execution state for every
compute command and restores it on exit. PowerToys Awake may remain enabled as
an independent safeguard; the runner never controls PowerToys.

Before launching the long run, verify representative optimized rows against
the frozen 100%-training legacy artifacts:

```powershell
$equivalenceDir = ".\research\experiments\lfo_representation\era2\artifacts\experiment_13\equivalence_exactopt_v1"
$equivalenceRows = "x13a_finish_repair_rescue_candidate_budget24_final_clip_only,x13a_broad_mean_finish_repair_interleaved_candidate_budget24_final_clip_only,x13a_broad_mean_hard_repair_two_phase_candidate_budget48_final_clip_only,x13a_trimmed_mean_global_repair_interleaved_candidate_budget24_final_clip_only"
conda run --no-capture-output -n py312 python $runner --mkl-threading-layer SEQUENTIAL --native-threads 1 verify-equivalence --baseline-run $legacyRun --output-dir $equivalenceDir --cache-dir $cacheDir --metadata $metadata --backend xpu --rows $equivalenceRows --chunk-size 256
```

### 3. Run Experiment 13A

The primary run uses a deterministic, topology-stratified 50% training sample
and the complete validation split. `--async` starts the worker in the
background. The command below suppresses the automatic monitor so it can be
opened explicitly ten seconds after the worker starts.

```powershell
conda run --no-capture-output -n py312 python $runner --mkl-threading-layer SEQUENTIAL --native-threads 1 run-13a --output-dir $runDir --cache-dir $cacheDir --async --no-monitor-window --backend xpu --metadata $metadata --train-sample-fraction 0.5 --validation-sample-fraction 1.0 --sample-seed 13 --verify-optimized-kernels first-use --chunk-size 256
Start-Sleep -Seconds 10
conda run --no-capture-output -n py312 python $runner monitor --run-dir $runDir --monitor-refresh-seconds 30
```

Resume the same 13A run after an interruption:

```powershell
conda run --no-capture-output -n py312 python $runner --mkl-threading-layer SEQUENTIAL --native-threads 1 run-13a --output-dir $runDir --cache-dir $cacheDir --async --backend xpu --metadata $metadata --train-sample-fraction 0.5 --validation-sample-fraction 1.0 --sample-seed 13 --verify-optimized-kernels first-use --chunk-size 256 --monitor-refresh-seconds 30 --resume
```

Request a safe cancellation at the next layer or slot checkpoint:

```powershell
conda run --no-capture-output -n py312 python $runner cancel --run-dir $runDir --reason "operator-requested safe cancellation"
```

Omit `--async` to run in the foreground. Add `--no-monitor-window` to keep the
background runner but suppress the separate Windows monitor.

### 4. Check status and events

The async command opens a live monitor automatically. For a one-time status
check from any shell, run:

```powershell
conda run --no-capture-output -n py312 python $runner status --run-dir $runDir
```

For a simple in-terminal monitor, press Ctrl+C to stop this loop:

```powershell
while ($true) {
    Clear-Host
    conda run --no-capture-output -n py312 python $runner status --run-dir $runDir
    Start-Sleep -Seconds 30
}
```

The status command reports `not_started`, `running`, `partial`, `blocked`,
`failed`, or `complete` for each phase and shows recent structured events.

### 5. Record the completed 13A epsilon calibration

Run selection only after 13A is complete:

```powershell
conda run --no-capture-output -n py312 python $runner --mkl-threading-layer SEQUENTIAL --native-threads 1 select-epsilon --run-dir $runDir
```

The command may legitimately record `selection_passed = false`. That result is
retained as calibration provenance and does not block the fixed three-epsilon
13B sweep.

### 6. Run Experiment 13B

Experiment 13B contains 135 rows: the 45 `LayerClip0To1` strategies at `1e-2`,
`1e-3`, and `1e-4`. The command does not schedule `FinalClipOnly` variants. The
three thresholds are fixed by the implementation and need no CLI argument.

```powershell
conda run --no-capture-output -n py312 python $runner --mkl-threading-layer SEQUENTIAL --native-threads 1 run-13b --output-dir $runDir --cache-dir $cacheDir --async --backend xpu --metadata $metadata --train-sample-fraction 0.5 --validation-sample-fraction 1.0 --sample-seed 13 --verify-optimized-kernels first-use --epsilon-selection $selection --chunk-size 256 --monitor-refresh-seconds 30
```

Resume the same 13B sweep after an interruption:

```powershell
conda run --no-capture-output -n py312 python $runner --mkl-threading-layer SEQUENTIAL --native-threads 1 run-13b --output-dir $runDir --cache-dir $cacheDir --async --backend xpu --metadata $metadata --train-sample-fraction 0.5 --validation-sample-fraction 1.0 --sample-seed 13 --verify-optimized-kernels first-use --epsilon-selection $selection --chunk-size 256 --monitor-refresh-seconds 30 --resume
```

The 13B command fails closed when the calibration artifact is missing,
malformed, stale, or incompatible with the completed 13A run. A valid
`selection_passed = false` result is accepted because epsilon is now an
explicit sweep axis rather than a selected scalar.

### 7. Generate analytics and the canonical report

Run this after both phases complete. It can also be rerun to regenerate the
derived CSVs, plots, and report from the retained run artifacts.

```powershell
conda run --no-capture-output -n py312 python $runner --mkl-threading-layer SEQUENTIAL --native-threads 1 analyze --run-dir $runDir

conda run --no-capture-output -n py312 python $runner analyze-scaling --full-run $legacyRun --sampled-run $runDir --output-dir "$runDir\training_data_scaling_ablation"
```

### Useful diagnostic controls

- `--rows <comma-separated-row-ids>` limits 13A or 13B to named rows for a
  diagnostic run; do not treat a partial run as phase completion.
- `--chunk-size <count>` changes scoring batch size; the default is `256`.
- `--backend auto|numpy|xpu` chooses the numerical backend.
- `--train-sample-fraction` and `--validation-sample-fraction` control the two
  splits independently. The primary run uses `0.5` and `1.0` respectively.
- `--corpus-sample-fraction` remains a deprecated compatibility alias that sets
  both split fractions and cannot be mixed with the split-specific flags.
- `--sample-seed 13` freezes deterministic hash-ranked sample membership.
- `--cache-dir` caches the parsed dataset, base dictionary, and base alignments.
- Run `conda run --no-capture-output -n py312 python $runner <command> --help`
  for the complete options of any subcommand.

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

For LFO $i$ entering residual layer $d$, let:

```math
r_{i,d}=y_i-\hat y_{i,d-1}.
```

Let $A_{d,s}=\{a_{d,0},\ldots,a_{d,s}\}$ be the partial codebook after
slot $s$, where $a_{d,0}=0$ is the no-op atom. Let $S_\phi(a)$ circularly
phase-shift atom $a$ by $\phi$. The current best layer reconstruction error is:

```math
E_{i,d,s}
=
\min_{a\in A_{d,s},\,\phi,\,g}
\left\|r_{i,d}-gS_\phi(a)\right\|_\infty.
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
supposed to calibrate. In formulas, $\tau_{\mathrm{finish}}$ denotes the fixed
`finish_threshold` and $\epsilon^*$ denotes the selected
`eligibility_epsilon`.

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

The fixed finish threshold carries forward Experiment 12's intended
maximum-error finishing boundary while correcting its MSE-versus-maximum-error
implementation mismatch. It is a construction objective and does not redefine
the report's joint strict-perfect metric.

The fixed W8D16 prediction-head budget is:

```math
H=32+16\cdot8+17+16=193.
```

Every main-grid row must preserve this budget.

## Primary Questions

Experiment 13 should answer:

1. Do synthesized broad atoms improve reconstruction over atoms copied only from
   observed residuals?
2. Is an interleaved broad/repair layer schedule better than a two-phase layer schedule?
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

```math
\mathrm{eligible}^{13A}_{i,d,s}=1
```

for every residual $i$, residual layer $d$, and active atom slot $s$.

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

```math
\mathrm{resolved}^{13B}_{i,d,s}
=
\mathbf{1}\!\left[E_{i,d,s}\leq\epsilon_j\right],
```

```math
\mathrm{eligible}^{13B}_{i,d,s}
=
1-\mathrm{resolved}^{13B}_{i,d,s}.
```

For each clipped strategy, Experiment 13B runs independent rows at
`eligibility_epsilon` values `1e-2`, `1e-3`, and `1e-4`. If no
eligible residuals remain, the remaining active atom slots should be filled with
no-op atoms and recorded as early completion rather than treated as a failure.

## Shared Alignment and Utility Definitions

For an atom proposal $a$ and residual $r_i$, define its best fitted phase and
gain as:

```math
(\phi_i(a),g_i(a))
=
\arg\min_{\phi,g}
\ell\!\left(r_i,gS_\phi(a)\right).
```

For utility scoring, let $L_i^{(s)}$ be the current scalar loss before slot $s$
and $L_i(a)$ the best scalar loss using proposal $a$. Define:

```math
\Delta_i(a)=\max\!\left(0,L_i^{(s)}-L_i(a)\right).
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

For fixed assignments $(\phi_i,g_i)$ and nonnegative weights $w_i$, solve:

```math
\min_a
\sum_i e_iw_i
\left\|r_i-g_iS_{\phi_i}(a)\right\|_2^2,
```

where $e_i$ is the current eligibility indicator. The least-squares update is:

```math
a
=
\frac{
\sum_i e_iw_ig_iS_{-\phi_i}(r_i)
}{
\sum_i e_iw_ig_i^2
}.
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

Let $T$ be the retained set containing the lowest 90% of aligned losses:

```math
T
=
\mathrm{Lowest}_{90\%}
\left\{
 i:e_i=1
 \;\middle|\;
 \ell\!\left(r_i,g_iS_{\phi_i}(a)\right)
\right\}.
```

Then update:

```math
a
=
\frac{
\sum_{i\in T}w_ig_iS_{-\phi_i}(r_i)
}{
\sum_{i\in T}w_ig_i^2
}.
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

```math
z_i
=
\frac{S_{-\phi_i}(r_i)}{\max(|g_i|,g_{\min})}.
```

For each control point $t$:

```math
a[t]
=
\mathrm{WeightedMedian}_i
\!\left(z_i[t];\,e_iw_i\right).
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

Using the phase/gain-invariant distance:

```math
d(r_i,r_j)
=
\min_{\phi,g}
\left\|r_i-gS_\phi(r_j)\right\|_2^2,
```

partition eligible residuals into clusters $C_1,\ldots,C_K$. For each cluster,
solve:

```math
a_k
=
\arg\min_a
\sum_{i\in C_k}w_i
\min_{\phi,g}
\left\|r_i-gS_\phi(a)\right\|_2^2.
```

Then choose:

```math
a^*=\arg\max_{a_k}\mathrm{Utility}(a_k).
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

For canonicalized residuals $z_i=S_{-\phi_i}(r_i)$, form:

```math
C=\sum_i e_iw_i z_i z_i^\top.
```

Choose:

```math
a
=
\arg\max_{\|a\|_2=1}
a^\top Ca.
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

```math
\mathrm{coverage}(a)
=
\sum_i e_i\mathbf{1}\!\left[\Delta_i(a)\geq\delta_{\min}\right],
```

```math
\mathrm{improvement}(a)
=
\sum_i e_iw_i\Delta_i(a),
```

and

```math
\mathrm{similarity}(a,b)
=
\max_\phi
\frac{|\langle a,S_\phi(b)\rangle|}
{\|a\|_2\|b\|_2}.
```

For previously selected broad atoms $B$, score:

```math
\mathrm{score}(a)
=
\alpha\mathrm{coverage}(a)
+\beta\mathrm{improvement}(a)
-\lambda\max_{b\in B}\mathrm{similarity}(a,b).
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

For candidate set $\mathcal C$:

```math
a^*
=
\arg\max_{a\in\mathcal C}
\sum_i e_iw_i\Delta_i(a).
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

For candidate $a$, define the candidate-after error

```math
E^+_{i,d,s}(a)
=
\min_{b\in A_{d,s}\cup\{a\},\,\phi,\,g}
\left\|r_{i,d}-gS_\phi(b)\right\|_\infty.
```

Then define

```math
\mathrm{finish}_{i,d,s}(a)
=
\mathbf{1}\!\left[
E_{i,d,s}>\tau_{\mathrm{finish}}
\;\land\;
E^+_{i,d,s}(a)\leq\tau_{\mathrm{finish}}
\right].
```

Choose lexicographically:

```math
\arg\max_a
\left(
\sum_i e_i\mathrm{finish}_{i,d,s}(a),
\sum_i e_iw_i\Delta_i(a)
\right).
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

Let $H_s$ be the eligible residuals at or above the 90th percentile of
current loss. Choose lexicographically:

```math
\arg\max_a
\left(
\sum_{i\in H_s}w_i\Delta_i(a),
\sum_i e_iw_i\Delta_i(a)
\right).
```

## Mixed Layer Schedules

Every new broad-plus-repair recipe is tested with both layer schedules below.
The schedule assigns one construction role to each of the 16 residual layers.
Within a Broad layer, all seven active atom slots use the recipe's broad builder.
Within a Repair layer, all seven active atom slots use the recipe's repair
builder. `Atom0` remains `NoOpAtom` in every layer and is not part of the
Broad/Repair schedule.

Active atoms are still constructed sequentially. Every active atom slot uses the
residual state and eligible population produced by all preceding layers and by
all preceding active atom slots in the current layer.

### `Interleaved`

**Common intuitive description**

> Alternate broad population-level layers with targeted repair layers throughout
> the residual stack.

**Technical description**

Assign residual-layer roles in this order:

```text
residual layer 1  = Broad
residual layer 2  = Repair
residual layer 3  = Broad
residual layer 4  = Repair
residual layer 5  = Broad
residual layer 6  = Repair
residual layer 7  = Broad
residual layer 8  = Repair
residual layer 9  = Broad
residual layer 10 = Repair
residual layer 11 = Broad
residual layer 12 = Repair
residual layer 13 = Broad
residual layer 14 = Repair
residual layer 15 = Broad
residual layer 16 = Repair
```

### `TwoPhase`

**Common intuitive description**

> Establish broad coverage throughout the first half of the residual stack, then
> use the second half for targeted repair.

**Technical description**

Assign residual-layer roles in this order:

```text
residual layers 1 through 8  = Broad
residual layers 9 through 16 = Repair
```

Both schedules use eight Broad layers and eight Repair layers. Because each
layer contains seven active atom slots, both schedules allocate 56 active atom
construction positions to Broad builders and 56 to Repair builders. This
isolates role ordering across residual depth without changing the Broad/Repair
allocation.

## Construction Policies

### Existing Experiment 12 anchors

Keep these observed-residual policies as historical anchors. Their exact
Experiment 12 role schedules and aggregation semantics should be preserved,
except that the `UnresolvedOnly` variant applies the dynamic eligibility mask
defined above and every finish test must be corrected to use complete-curve
maximum absolute error against the fixed `finish_threshold`.

These anchors use `layer_schedule = AnchorNative`. They do not inherit the mixed
`Interleaved` or `TwoPhase` layer schedules. Their existing Experiment 12
slot-role schedules remain active inside every residual layer.

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

```math
\mathrm{score}(a)
=
\mathrm{Median}_{i:e_i=1}\Delta_i(a).
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

```math
a_s
=
\arg\max_{a\in\mathcal C_s}U_{\mathrm{role}(s)}(a).
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

If families are $F_1,\ldots,F_K$, use the exact existing score. A
representative form is:

```math
\mathrm{score}(a)
=
\mathrm{Aggregate}_k
\left[
\mathrm{Mean}_{i\in F_k,\,e_i=1}\Delta_i(a)
\right].
```

where `aggregate_k` is the Experiment 12 balancing operator. Record the actual
operator rather than replacing it with an undocumented mean.

### New mixed prototype/repair recipes

Each recipe below receives both `Interleaved` and `TwoPhase` variants. The
layer schedule determines which builder runs in each residual layer; the builder
definitions above determine the optimization used for all seven active atom slots
in that layer.

#### `BroadMeanGlobalRepair`

**Common intuitive description**

> Repeatedly combine a shared average correction with a concrete example that
> removes the most total remaining error.

**Technical description**

In Broad-designated layers, all seven active atom slots use `BroadMean`. In
Repair-designated layers, all seven active atom slots use `GlobalRepair`. This is
the most direct test of whether smooth population-level corrections and sharp
observed examples complement one another.

**Mathematical formulation**

```text
Broad-layer active slot:  solve weighted aligned least squares for a
Repair-layer active slot: maximize sum_i eligible_i * w_i * Delta_i(a)
```

#### `BroadMeanFinishRepair`

**Common intuitive description**

> Make broad progress, then use concrete repairs to push near-correct curves
> inside the fixed strict-finish threshold.

**Technical description**

In Broad-designated layers, all seven active atom slots use `BroadMean`. In
Repair-designated layers, all seven active atom slots use `FinishRepair`. This
policy tests whether broad prototypes create a large population of almost-solved
residuals that targeted finishing can move across the construction finish
criterion. The finish objective uses the fixed `finish_threshold`, not the
calibrated eligibility epsilon.

**Mathematical formulation**

```text
Broad-layer active slot:  solve weighted aligned least squares for a
Repair-layer active slot: lexicographically maximize newly_resolved_count, total_improvement
```

#### `BroadMeanHardRepair`

**Common intuitive description**

> Use broad averages for general progress, then use concrete examples to rescue
> the worst remaining cases.

**Technical description**

In Broad-designated layers, all seven active atom slots use `BroadMean`. In
Repair-designated layers, all seven active atom slots use `HardRepair`. This
separates broad central coverage from explicit tail control.

**Mathematical formulation**

```text
Broad-layer active slot:  solve weighted aligned least squares for a
Repair-layer active slot: maximize improvement over current worst-loss decile
```

#### `TrimmedMeanGlobalRepair`

**Common intuitive description**

> Use an outlier-resistant broad average, then choose the concrete correction
> that removes the most total residual error.

**Technical description**

In Broad-designated layers, all seven active atom slots use `TrimmedMean`. In
Repair-designated layers, all seven active atom slots use `GlobalRepair`. This
tests whether ordinary aligned means are being distorted by a small number of
unusual residuals.

**Mathematical formulation**

```text
Broad-layer active slot:  aligned least squares over retained lowest-loss 90%
Repair-layer active slot: maximize weighted summed Delta_i(a)
```

#### `AlignedMedianGlobalRepair`

**Common intuitive description**

> Build a very robust middle-shaped prototype, then use concrete examples for
> global cleanup.

**Technical description**

In Broad-designated layers, all seven active atom slots use `AlignedMedian`. In
Repair-designated layers, all seven active atom slots use `GlobalRepair`. This is
a stronger robustness test than trimming because each control point uses a
median rather than a mean.

**Mathematical formulation**

```text
Broad-layer active slot:  coordinate-wise weighted median in canonical phase/gain frame
Repair-layer active slot: maximize weighted summed Delta_i(a)
```

#### `ClusterMeanGlobalRepair`

**Common intuitive description**

> Cover one common family of mistakes at a time, then choose concrete corrections
> that remove the most error left across all families.

**Technical description**

In Broad-designated layers, all seven active atom slots use `ClusterMean`. In
Repair-designated layers, all seven active atom slots use `GlobalRepair`.
Clustering is recomputed before every active atom slot in a Broad-designated
layer from the current eligible residuals.

**Mathematical formulation**

```text
Broad-layer active slot:  choose highest-utility aligned cluster prototype a_k
Repair-layer active slot: maximize weighted summed Delta_i(a)
```

#### `ClusterMeanHardRepair`

**Common intuitive description**

> Cover the main residual families broadly, then use concrete examples to rescue
> the difficult leftovers.

**Technical description**

In Broad-designated layers, all seven active atom slots use `ClusterMean`. In
Repair-designated layers, all seven active atom slots use `HardRepair`. This
policy tests a coarse family-coverage phase combined with explicit tail
protection.

**Mathematical formulation**

```text
Broad-layer active slot:  choose highest-utility aligned cluster prototype a_k
Repair-layer active slot: maximize improvement over current worst-loss decile
```

#### `DominantDirectionGlobalRepair`

**Common intuitive description**

> Remove the strongest shared direction of error, then clean up with the best
> concrete residual examples.

**Technical description**

In Broad-designated layers, all seven active atom slots use
`DominantDirection`. In Repair-designated layers, all seven active atom slots use
`GlobalRepair`. The broad atom may not resemble any single residual; it
represents a high-energy shared correction direction.

**Mathematical formulation**

```text
Broad-layer active slot:  leading eigenvector of weighted canonical residual covariance
Repair-layer active slot: maximize weighted summed Delta_i(a)
```

#### `DiverseCoverageHardRepair`

**Common intuitive description**

> Spend Broad layers on several different widely useful corrections, then use
> Repair layers for the worst cases still left.

**Technical description**

In Broad-designated layers, all seven active atom slots use `DiverseCoverage`.
In Repair-designated layers, all seven active atom slots use `HardRepair`. The
broad score must penalize phase/scale similarity to earlier broad atoms in the
same layer.

**Mathematical formulation**

```text
Broad-layer active slot:  maximize alpha*coverage + beta*improvement - lambda*similarity
Repair-layer active slot: maximize improvement over current worst-loss decile
```

PascalCase row identifiers append the schedule, for example:

```text
BroadMeanGlobalRepairInterleaved
BroadMeanGlobalRepairTwoPhase
ClusterMeanHardRepairInterleaved
ClusterMeanHardRepairTwoPhase
```

### Pure-prototype controls

These controls use `layer_schedule = AllBroad`. All 16 residual layers are
Broad-designated, and all seven active atom slots in every layer use the
control's broad builder.

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

For every slot $s=1,\ldots,7$:

```math
a_s
=
\arg\min_a
\sum_i e_i^{(s)}w_i^{(s)}
\min_{\phi,g}
\left\|r_i-gS_\phi(a)\right\|_2^2.
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

`CandidateBudget24` and `CandidateBudget48` apply to every active atom slot in
a Repair-designated layer. Every active atom slot in a Broad-designated layer
uses `Null`. For the Experiment 12 anchors, the row-level candidate budget
applies wherever the preserved native slot role selects an observed residual
example. Pure-prototype policies always use `Null`.

Artifacts must record the row-level budget, a 16-element
effective-candidate-budget vector by residual layer, and a nested `16 x 7`
effective-candidate-budget matrix by active atom slot. For example, the
16-element effective layer budgets for `CandidateBudget48` are:

```text
Interleaved:
[Null, 48, Null, 48, Null, 48, Null, 48,
 Null, 48, Null, 48, Null, 48, Null, 48]

TwoPhase:
[Null, Null, Null, Null, Null, Null, Null, Null,
 48, 48, 48, 48, 48, 48, 48, 48]
```

Within each Repair-designated layer, all seven active atom slots use the listed
budget. Within each Broad-designated layer, all seven active atom slots use
`Null`.

## Layer Normalization

Experiment 13A evaluates both values:

```text
FinalClipOnly
LayerClip0To1
```

- `FinalClipOnly`: no per-layer clipping;
- `LayerClip0To1`: clip the running reconstruction to `[0, 1]` after each layer.

This is the only secondary process axis retained because `LayerClip0To1`
performed strongly in Experiment 12 and remains simple to interpret. The
completed 13A grid then found a P95 improvement in all 45 matched comparisons,
so Experiment 13B is locked to `LayerClip0To1` and does not repeat
`FinalClipOnly`.

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

Phase-specific crossed axes:

```text
construction_policy
residual_population_policy = AllResiduals | UnresolvedOnly
utility_candidate_budget = CandidateBudget24 | CandidateBudget48 | Null
layer_normalization_policy =
  13A: FinalClipOnly | LayerClip0To1
  13B: LayerClip0To1
```

The logical design remains paired across the population-policy axis, but
execution is ordered:

```text
Experiment 13A = 90 AllResiduals rows
Experiment 13B = 45 paired UnresolvedOnly + LayerClip0To1 strategies * 3 epsilons = 135 rows
```

Every 13B row must share a stable `pair_id` with its clipped 13A counterpart;
`row_id` additionally encodes its epsilon and remains unique.
Paired rows must match on construction
policy, layer schedule, repair candidate budget, layer normalization, fixed
settings, seed rules, and `finish_threshold`. They differ only in experiment
phase, residual-population behavior, and the fixed experimental
`eligibility_epsilon` mask. The 45 `FinalClipOnly` rows remain 13A-only
normalization controls.

The 21 repair-containing construction policies receive:

```text
13A: 21 policies * 2 repair budgets * 2 normalization policies = 84 rows
13B: 21 policies * 2 repair budgets * 1 normalization policy * 3 epsilons = 126 rows
= 210 rows
```

The three pure-prototype policies receive:

```text
13A: 3 policies * 1 Null budget * 2 normalization policies = 6 rows
13B: 3 policies * 1 Null budget * 1 normalization policy * 3 epsilons = 9 rows
= 15 rows
```

Main-grid total:

```text
210 + 15 = 225 rows
```

The 21 repair-containing policies are:

```text
3 existing anchors
+ 9 mixed recipes * 2 schedules
= 21
```

## Experiment 13A Calibration and Epsilon Selection

Experiment 13A performs unfiltered construction and measures the natural error
trajectory. Candidate eligibility epsilons are counterfactual measurements only.
They must not alter candidate generation, candidate scoring, prototype fitting,
clustering, residual weights, early termination, or any other construction
behavior.

### Completed-layer full-curve error

After the base dictionary and after every completed residual layer, record the
maximum-absolute full-curve error:

```math
G_{i,d}
=
\left\|y_i-\hat y_{i,d}\right\|_\infty.
```

Layer `d = 0` is the base-dictionary reconstruction before any residual layer.
Record completed-layer distributions for both training and validation splits.
Validation data is report-only and must not influence epsilon selection.

### Slot-level partial-codebook error

Inside each residual layer, record `E_i,d,s` after:

```text
slot 0 = NoOpAtom only
slot 1 = first active atom
...
slot 7 = all active atoms
```

Slot-level training measurements are the primary calibration data because
`UnresolvedOnly` changes the population inside a layer. Slot 7 is retained for
diagnostics, but it is not a decision checkpoint because no later active slot in
that layer can be affected by retiring a residual after slot 7.

For each residual, also retain the best fitted unexplained residual vector:

```math
u_{i,d,s}
=
r_{i,d}-g_{i,d,s}S_{\phi_{i,d,s}}(a_{i,d,s}).
```

where the atom, phase, and gain are the minimizers used to compute `E_i,d,s`.

### Required quantiles

For every relevant `G_i,d` and `E_i,d,s` distribution, record:

```text
50th percentile
25th percentile
10th percentile
5th percentile
2nd percentile
1st percentile
0.1st percentile when sample_count >= 1000
```

Interpretation:

For percentile $p$, define the completed-layer and slot-level quantiles:

```math
Q^{\mathrm{global}}_d(p)
=
\mathrm{Quantile}_i\!\left(G_{i,d},p\right),
```

and

```math
Q^{\mathrm{slot}}_{d,s}(p)
=
\mathrm{Quantile}_i\!\left(E_{i,d,s},p\right).
```

Thus $Q^{\mathrm{slot}}_{d,s}(0.10)$ is the eligibility epsilon that would
classify approximately 10% of training residuals as resolved at layer $d$, slot
$s$, on the unfiltered 13A trajectory.

### Candidate epsilon coverage

Measure at least these fixed candidate values:

```text
0.001
0.0025
0.005
0.01
0.02
```

For each candidate epsilon and checkpoint, record:

```math
F_{d,s}(\epsilon)
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}\!\left[E_{i,d,s}\leq\epsilon\right],
```

and the counterfactual eligible fraction $1-F_{d,s}(\epsilon)$.

Also record completed-layer coverage:

```math
F^{\mathrm{global}}_d(\epsilon)
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}\!\left[G_{i,d}\leq\epsilon\right].
```

These statistics describe what filtering would have done; they do not change
13A construction.

### Retired error mass

Count alone is insufficient. Record two energy views for every slot checkpoint
and candidate epsilon.

Incoming residual-energy share:

```math
M^{\mathrm{incoming}}_{d,s}(\epsilon)
=
\frac{
\sum_{i:E_{i,d,s}\leq\epsilon}\|r_{i,d}\|_2^2
}{
\sum_i\|r_{i,d}\|_2^2
}.
```

Current unexplained-error-energy share:

```math
M^{\mathrm{unexplained}}_{d,s}(\epsilon)
=
\frac{
\sum_{i:E_{i,d,s}\leq\epsilon}\|u_{i,d,s}\|_2^2
}{
\sum_i\|u_{i,d,s}\|_2^2
}.
```

The incoming-energy view is diagnostic. Epsilon selection must use the
unexplained-error view because it measures error that would actually stop
influencing later atom construction. A curve may have a large incoming residual
while already being fitted almost exactly by the partial codebook.

When an energy denominator is zero, define the corresponding retired-energy
fraction as `0.0`, record `zero_total_energy = true`, and exclude that checkpoint
from quantile aggregation rather than producing `NaN` or infinity.

Also record the corresponding retained fractions. The desired regime is:

```text
high retired-LFO fraction
low retired unexplained-error-energy fraction
```

### Deterministic selection checkpoint set

The selection calculation uses training data only and aggregates over:

```text
all 90 completed Experiment 13A rows
residual layers 1 through 16
decision slots 0 through 6
```

Slot 7 and all validation measurements are excluded from selection. Let
$`\mathcal{S}_{\mathrm{valid}}`$ denote the subset of this row/layer/slot set with
nonzero unexplained-error energy, and let $`\mathcal{R}_{13A}`$ denote the 90
completed 13A rows.

Define $\mathcal{S}_{\mathrm{early\text{-}middle}}$ as:

```text
residual layers 1 through 12
slots 0 through 5
```

### Deterministic selection rule

Evaluate the candidate epsilons in ascending order and choose the largest value
that satisfies all three conditions:

1. The median unexplained retired-error fraction across all selection rows and
   checkpoints satisfies

```math
\mathrm{Median}_{(\rho,d,s)\in\mathcal{S}_{\mathrm{valid}}}
M^{\mathrm{unexplained}}_{\rho,d,s}(\epsilon)
\leq0.01.
```

2. Its 95th percentile satisfies

```math
\mathrm{Quantile}_{0.95,
(\rho,d,s)\in\mathcal{S}_{\mathrm{valid}}}
\!\left(M^{\mathrm{unexplained}}_{\rho,d,s}(\epsilon)\right)
\leq0.05.
```

3. At least one early-or-middle checkpoint satisfies

```math
\exists(d,s)\in\mathcal{S}_{\mathrm{early\text{-}middle}}:
\mathrm{Median}_{\rho\in\mathcal{R}_{13A}}
F_{\rho,d,s}(\epsilon)
\geq0.05.
```

This rule is an operational calibration policy, not a claim of theoretical or
perceptual optimality. Do not choose an epsilon by visual inspection.

Write the automatic result, exact supporting statistics, checkpoint definition,
candidate set, and selection-rule version to `epsilon_selection.json` before
Experiment 13B begins. This preserves the 13A calibration decision even when it
does not select a passing value.

If no candidate satisfies all three conditions, write
`selection_passed = false` and do not silently relax the rule. The fixed 13B
sweep may still proceed because it evaluates three explicit thresholds rather
than claiming that the calibration selected one. The legacy restricted-pilot
and override interfaces remain available for compatibility, but they are not
part of the main sweep workflow.

### Required calibration plots

The Experiment 13A calibration report must contain at least:

1. completed-layer epsilon quantiles for layers `0..16`;
2. slot-level epsilon quantiles for slots `0..7`;
3. completed-layer reconstructed fractions for each candidate epsilon;
4. slot-level reconstructed fractions for each candidate epsilon;
5. retired-LFO fraction versus retired unexplained-error-energy fraction;
6. incoming-energy and unexplained-energy retirement shown separately.

Detailed per-row and per-layer values must remain available in artifacts even
when the report shows median curves, percentile bands, or strategy-family
aggregations.

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
experiment_phase
row_id
pair_id
residual_layer
slot_index
layer_role
slot_role
atom_source_kind
effective_candidate_budget
finish_threshold
selected_eligibility_epsilon
eligibility_selection_rule_version
eligible_residual_count_before
eligible_residual_count_after
resolved_lfo_rate_before
resolved_lfo_rate_after
newly_eligibility_resolved_lfo_count
newly_finish_threshold_lfo_count
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
counterfactual_resolved_fraction_by_candidate_epsilon
counterfactual_incoming_retired_energy_fraction_by_candidate_epsilon
counterfactual_unexplained_retired_energy_fraction_by_candidate_epsilon
```

For mixed recipes, `layer_role` is `Broad` or `Repair` and is shared by all
seven active atom slots in the residual layer. The Experiment 12 anchors use
`AnchorNative` as the layer role and preserve their native per-slot roles.
Pure-prototype controls use `Broad` for every residual layer. `slot_role` records
the builder-specific or preserved anchor role used for that active atom slot.

For 13A, `eligible_residual_count_before` and
`eligible_residual_count_after` must describe the actual unfiltered construction
population. Counterfactual filtered counts and fractions must use separate fields
and must never be presented as actual 13A eligibility.

`newly_finish_threshold_lfo_count` records the fixed construction finish
criterion. `newly_eligibility_resolved_lfo_count` records crossings of the frozen
13B eligibility epsilon. Preserve `newly_resolved_lfo_count` as a compatibility
field, but define it as an alias of `newly_eligibility_resolved_lfo_count` in 13B
and `0` in unfiltered 13A; do not use it for finish-objective accounting.

For partial-codebook validation, evaluate the row using:

```text
NoOpAtom + first 1 active atom per layer
NoOpAtom + first 2 active atoms per layer
...
NoOpAtom + all 7 active atoms per layer
```

Write:

```text
experiment_phase
row_id
pair_id
finish_threshold
selected_eligibility_epsilon
active_atom_count
validation_median_rmse
validation_strict_perfect_lfo_rate
validation_p95_rmse
validation_node_max_error_p95
```

This progression is required to distinguish stack-level effects of early
Broad-designated layers from within-layer efficiency as each codebook grows from
one through seven active atoms.

## Report Requirements

The report should be findings-first and follow the actual execution order:

1. Experiment 13A unfiltered construction results;
2. Experiment 13A epsilon calibration;
3. the automatic eligibility-calibration result, fixed 13B epsilon sweep, and
   exact supporting training statistics;
4. Experiment 13B paired filtered results;
5. `AllResiduals` versus `UnresolvedOnly` paired effects;
6. prototype-containing policies versus observed-residual anchors;
7. `Interleaved` versus `TwoPhase`;
8. `CandidateBudget24` versus `CandidateBudget48`;
9. the completed 13A `FinalClipOnly` versus `LayerClip0To1` evidence that fixed
   13B to `LayerClip0To1`;
10. broad-builder and repair-objective interactions;
11. partial-codebook progression from one through seven active atoms.

The report must state that 13A completed first, the eligibility epsilon was
selected from training calibration data and frozen, and 13B then used that
value. It must not imply that the two phases ran simultaneously.

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
layer_epsilon_quantiles.csv
slot_epsilon_quantiles.csv
epsilon_coverage.csv
retired_error_mass.csv
epsilon_selection.json
experiment13a_status.json
experiment13b_status.json
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


### Calibration artifact schemas

`layer_epsilon_quantiles.csv` must contain:

```text
experiment_phase
row_id
pair_id
dataset_split
residual_layer
percentile
epsilon_value
sample_count
```

`slot_epsilon_quantiles.csv` must contain:

```text
experiment_phase
row_id
pair_id
residual_layer
active_atom_slot
percentile
epsilon_value
sample_count
```

For completed-layer records in `epsilon_coverage.csv`, set
`active_atom_slot = Null`. Slot-level records use slots `0..7`.

`epsilon_coverage.csv` must contain:

```text
experiment_phase
row_id
pair_id
dataset_split
residual_layer
active_atom_slot
epsilon
resolved_count
resolved_fraction
counterfactual_eligible_count
counterfactual_eligible_fraction
```

`retired_error_mass.csv` must contain:

```text
experiment_phase
row_id
pair_id
residual_layer
active_atom_slot
epsilon
retired_lfo_count
retired_lfo_fraction
incoming_retired_energy
incoming_retired_energy_fraction
unexplained_retired_energy
unexplained_retired_energy_fraction
retained_unexplained_energy_fraction
zero_total_energy
```

When automatic selection fails, `selected_epsilon` remains `Null`. The 13B
sweep values are recorded separately and must not be represented as a selection
override.

`epsilon_selection.json` must contain at least:

```text
candidate_epsilons
selection_rule_version
selection_checkpoint_definition
selected_epsilon
training_statistics_used
median_unexplained_retired_energy_fraction
p95_unexplained_retired_energy_fraction
retired_lfo_fraction_summary
selection_timestamp
experiment13a_run_identity
selection_passed
selection_override
selection_override_rationale
selection_override_timestamp
pilot_evidence
selection_notes
```

## Manifest Fields

Preserve the Experiment 12 fields and add strategy-specific fields:

```text
experiment_id
experiment_phase
row_id
pair_id
experiment13a_run_identity
scalar_schema
path_search_policy
construction_policy
construction_family
layer_schedule
residual_population_policy
finish_threshold
eligibility_epsilon
eligibility_selection_rule_version
eligibility_epsilon_sweep
eligibility_epsilon_sweep_version
utility_candidate_budget
effective_candidate_budget_by_layer
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

## Run Commands

The implemented runner and complete ordered command sequence are documented in
[Operator Quick Start](#operator-quick-start) near the top of this plan. That
section is the canonical operator runbook; command examples should be maintained
there rather than duplicated here.

## Test Requirements

Tests should verify:

- the fixed W8D16 runtime contract and 193-head accounting;
- `Atom0 = NoOpAtom` for every residual layer;
- the exact 16-layer `Interleaved` and `TwoPhase` schedules;
- `Interleaved` assigns Broad to odd-numbered layers and Repair to even-numbered
  layers;
- `TwoPhase` assigns Broad to layers 1 through 8 and Repair to layers 9 through
  16;
- both mixed schedules contain exactly eight Broad layers and eight Repair
  layers;
- all seven active atom slots in a mixed-policy layer inherit that layer's Broad
  or Repair role;
- exactly 90 13A `AllResiduals` rows and 135 paired 13B `UnresolvedOnly` rows;
- the complete executed design contains exactly 225 rows;
- every 13B row uses `LayerClip0To1`, carries one of the three fixed epsilon
  values, and has exactly one 13A counterpart with identical paired settings;
- the 45 `FinalClipOnly` rows remain 13A-only normalization controls and are not
  scheduled in 13B;
- `AllResiduals` and `UnresolvedOnly` mask behavior;
- candidate eligibility epsilons never change 13A construction;
- finish and eligibility thresholds are separate and use maximum absolute error;
- validation data cannot affect eligibility-epsilon selection;
- the unresolved mask is recomputed after every active atom slot;
- broad atoms can differ from every observed residual;
- phase/gain-aligned prototype updates do not raw-average shifted residuals;
- each broad builder follows its documented objective and deterministic tie rules;
- repair candidate budgets apply to all seven active atom slots in
  Repair-designated layers, while Broad-designated layers use `Null`;
- Experiment 12 anchors preserve their native slot-role schedules under the
  `AnchorNative` layer schedule;
- pure-prototype controls use the `AllBroad` layer schedule;
- finish scoring uses maximum absolute error against `finish_threshold` rather
  than MSE `<= finish_threshold^2`;
- finish-threshold crossings and eligibility-epsilon crossings are recorded in
  separate diagnostic fields;
- layer and slot quantiles match direct NumPy calculations;
- coverage fractions, incoming energy shares, and unexplained energy shares are
  correct, including zero-denominator handling;
- the selection checkpoint set excludes slot 7 and validation data;
- epsilon selection is deterministic, while 13B independently covers all 45
  clipped strategies at `1e-2`, `1e-3`, and `1e-4`;
- the pilot command is restricted to the prespecified rows and two tightest
  candidate epsilons;
- 13B cannot start without a valid completed 13A calibration artifact, but a
  matching artifact with `selection_passed = false` is accepted;
- pure-prototype rows require `Null`;
- deterministic construction under a fixed seed;
- empty unresolved populations terminate with remaining no-op atoms;
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
