# LFO Era 2

This is the clean workspace for the next LFO representation research phase.

Era 2 starts from the design contract in
[reports/LFO_ERA2_DESIGN.md](./reports/LFO_ERA2_DESIGN.md). The key point is that
topology may help offline codebook construction, but topology must not select
atoms after the codebook exists.

## Workspace

- [experiments/](./experiments/): Era 2 experiment plans, including
  [EXPERIMENT_10_PLAN.md](./experiments/EXPERIMENT_10_PLAN.md),
  [EXPERIMENT_11_PLAN.md](./experiments/EXPERIMENT_11_PLAN.md),
  [EXPERIMENT_12_PLAN.md](./experiments/EXPERIMENT_12_PLAN.md),
  [EXPERIMENT_13_PLAN.md](./experiments/EXPERIMENT_13_PLAN.md), and
  [ERA2_CORE_FRAMEWORK_PLAN.md](./experiments/ERA2_CORE_FRAMEWORK_PLAN.md).
- [reports/](./reports/): Era 2 research notes, design contracts, and future
  result writeups.
- [artifacts/](./artifacts/): future generated outputs.
- [notes/](./notes/): working notes and sketches.
- [code/](./code/): compact Era 2 framework code plus standalone audit scripts.
- [tests/](./tests/): focused framework tests.

## Core Framework

The Era 2 framework starts fresh from the model-facing contract. It is
intentionally small and split around the concepts that matter for Experiment 11
and later:

```text
offline oracle construction -> reconstruction assets -> runtime interface -> decoder policy -> accounting/metrics
```

Settled LFO-grid choice after Experiment 10:

```text
control_point_count = 97
```

This fixed uniform x lattice is decoder-owned. It has 96 derived subdivisions,
which are useful for alignment discussion, but future row configuration should
use the 97 control-point count because that determines vector shapes. Era 2
model rows should not spend model prediction head budget on x-coordinate
prediction, grid selection, or variable grid spacing.

Run the framework smoke path with:

```text
python .\research\experiments\lfo_representation\era2\code\run_era2.py smoke-flat
```

Build the processed LFO corpus with raw point sets and dense 1920-sample
references:

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\run_era2.py --mkl-threading-layer SEQUENTIAL --native-threads 1 build-lfo-corpus
```

Use `--no-capture-output` for long-running commands. Plain `conda run` can
buffer child-process stdout until the command exits, which makes progress
checkpoints look like they are not printing.

Run Experiment 10, the standalone control-point x-grid audit:

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment10_grid_audit.py
```

  Defaults:

  ```text
  subdivision_counts = 8,10,12,...,100
  control_point_count = subdivision_count + 1
  ```

  This experiment reports source point-count frequency and ordered control-point
  x placement against every even `subdivision_count` from 8 through 100. The
  CSV keeps `grid_point_count` as the implementation field for
  `control_point_count`. The metric scores x-axis placement only, so y values
  and segment connection rules are intentionally out of scope. It also reports
  the fraction of LFOs whose maximum control-point x error is at most `0.001`,
  plus fixed global non-uniform quantile grids learned offline from
  deduplicated and occurrence-weighted corpus x positions. Supporting CSV files
  include point-count frequency, x-lattice frequency, control-point x summary,
  factor-3 comparisons, and global non-uniform grid definitions. `W` remains
  reserved for residual-layer atom choices in model experiments. The generated
  markdown report is written to
  `era2/reports/EXPERIMENT_10_CONTROL_POINT_X_GRID_REPORT.md` and embeds
  graph-first local plot copies under `era2/reports/images/experiment_10/`.
  The same plots are also retained with the CSV artifacts under
  `era2/artifacts/experiment_10/control_point_x_grid/plots/`.

Run Experiment 11, the topology-free flat-categorical residual run. By
default this uses the full corpus:

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\run_era2.py --mkl-threading-layer SEQUENTIAL --native-threads 1 run-screen --async --screen experiment11 --backend xpu --metadata .\datasets\presetshare\raw\presetshare_vital_metadata.csv --corpus-sample-fraction 1.0 --oracle-phase-search-policy continuous
```

`--oracle-phase-search-policy continuous` is the canonical Experiment 11
default. It estimates continuous phase targets on the oracle side using the
FFT/lattice method. The deployed model still emits one continuous phase scalar
for the base and one per residual layer, so changing oracle search details does
not change `head_outputs`. Use `--oracle-phase-search-policy grid
--oracle-phase-candidate-count <N>` only for an explicit discrete phase search
ablation.

The wrapper-level runtime flags are applied before NumPy/SciPy/PyTorch imports.
`--native-threads 1` sets `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, and
`MKL_NUM_THREADS` together. The individual flags are also available:
`--openblas-threads`, `--omp-threads`, and `--mkl-threads`.

`run-screen --async` starts the runner in the background, opens a monitor
window, prints a `Started async Experiment 11 run` message with the run
directory and log paths, then immediately returns to the shell. The monitor
refreshes every 30 seconds by default. The same is true when continuing a run:

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\run_era2.py --mkl-threading-layer SEQUENTIAL --native-threads 1 run-screen --async --screen experiment11 --backend xpu --metadata .\datasets\presetshare\raw\presetshare_vital_metadata.csv --corpus-sample-fraction 1.0 --oracle-phase-search-policy continuous --run-dir <run_dir> --resume
```

Use `--monitor-refresh-seconds <seconds>` to change the monitor refresh rate.
Use `--no-monitor-window` to start the background runner without opening the
monitor. Foreground `run-screen` remains available by omitting `--async`.

Run the tiny Experiment 11 plumbing check with:

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\run_era2.py --mkl-threading-layer SEQUENTIAL --native-threads 1 run-screen --async --screen experiment11 --backend auto --metadata .\datasets\presetshare\raw\presetshare_vital_metadata.csv --smoke --oracle-phase-search-policy continuous
```

Use `--corpus-sample-fraction <0..1>` for a deterministic corpus fraction.
`--smoke` is only a fixed tiny plumbing test and should not be treated as an
Experiment 11 result.

Run artifacts are written under:

```text
era2/artifacts/experiment_11/runs/<run_id>/
```

Those run directories are provenance and raw artifact storage. The user-facing
Experiment 11 writeup is the single canonical report:

```text
era2/reports/EXPERIMENT_11_FLAT_CATEGORICAL_REPORT.md
```

Its local plot copies are written under
`era2/reports/images/experiment_11/`.

Attach to an existing run from another terminal:

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\run_era2.py status --run-dir <run_dir> --watch 5
```

Regenerate analytics:

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\run_era2.py analyze --run-dir <run_dir>
```

This refreshes run-local CSV analytics and rewrites the canonical Experiment 11
report in `era2/reports/`. Run-local analytics also include
`budget_projections.csv`, which records formula-only alternate indexing budget
views such as binary path addressing. These are not reconstruction-quality rows.

Run the standalone W8D16 deviation audit when comparing Era 2 behavior against
Era 1 W8D16 anchors:

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment11_w8d16_deviation_audit.py --metadata .\datasets\presetshare\raw\presetshare_vital_metadata.csv --output-dir .\research\experiments\lfo_representation\era2\artifacts\experiment_11\w8d16_deviation_audit --backend xpu --corpus-sample-fraction 1.0 --chunk-size 256
```

This is a diagnostic, not a new Experiment 11 profile. It fixes `W=8`,
`D=16`, writes supporting CSV artifacts under
`era2/artifacts/experiment_11/w8d16_deviation_audit/`, and inserts a W8D16
section into the canonical Experiment 11 report.

Run the RMSE gap forensic audit when investigating why Era 2 reconstruction is
far worse than Era 1:

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment11_rmse_gap_audit.py --mkl-threading-layer SEQUENTIAL --native-threads 1 --row-limit 256
```

This is not a screen. It checks same-array metrics, circular shift, phase/gain
alignment, and replay of an Era 1 W8D16 checkpoint. The report is written to
`era2/reports/EXPERIMENT_11_RMSE_GAP_FORENSIC_AUDIT.md`, with CSV probes under
`era2/artifacts/experiment_11/rmse_gap_audit/`.

Run Experiment 12, the fixed-W8D16 screening grid:

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment12_component_ladder.py --mkl-threading-layer SEQUENTIAL --native-threads 1 run --async --backend xpu --metadata .\datasets\presetshare\raw\presetshare_vital_metadata.csv --corpus-sample-fraction 1.0 --monitor-refresh-seconds 30
```

Experiment 12 is standalone. It fixes `W=8`, `D=16`, and
`control_point_count=97`, reserves `Atom0 = NoOpAtom` in every residual layer,
and screens prediction-head-free process variables one at a time. Variable
values are PascalCase in docs, reports, and artifacts. Every screened value is
tested under `IndicesOnly` and `PhaseAndResidualGain` scalar schemas. The
screening report is grouped by variable and does not auto-rank winners.

Run the tiny Experiment 12 smoke path with:

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment12_component_ladder.py --mkl-threading-layer SEQUENTIAL --native-threads 1 run --backend auto --metadata .\datasets\presetshare\raw\presetshare_vital_metadata.csv --smoke
```

Regenerate Experiment 12 analytics and the canonical report with:

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment12_component_ladder.py --mkl-threading-layer SEQUENTIAL --native-threads 1 analyze --run-dir .\research\experiments\lfo_representation\era2\artifacts\experiment_12\component_ladder
```

Experiment 12 artifacts are written under
`era2/artifacts/experiment_12/component_ladder/`. The user-facing report is
`era2/reports/EXPERIMENT_12_W8D16_COMPONENT_LADDER_REPORT.md`, with plots under
`era2/reports/images/experiment_12/`.

Experiment 13 is implemented at `era2/code/experiment13_strategy_grid.py`. It
executes the paired 90-row 13A and 90-row 13B grids, including synthesized broad
atoms, observed-residual repair atoms, resumable row artifacts, phase/slot
calibration, deterministic epsilon selection, the restricted fallback pilot,
paired analysis, and the canonical report required by
[EXPERIMENT_13_PLAN.md](./experiments/EXPERIMENT_13_PLAN.md). The Windows
`--async` path launches a background runner, captures stdout/stderr under the
Experiment 13 `launcher_logs` directory, and opens a live status/event monitor.
Experiment 13 also supports independent deterministic train/validation sample
fractions, a versioned dataset/base-stage cache, exact optimized kernels,
slot/layer wall and CPU timings, safe `cancel`, standalone `monitor`, legacy
`verify-equivalence`, and matched `analyze-scaling` commands. Compute workers
hold a scoped Windows system-required execution state without controlling
PowerToys Awake. Use `status` to inspect `not_started`, `running`, `partial`,
`blocked`, `failed`, `cancelled`, and `complete` phase states.

If automatic epsilon selection does not pass, run the restricted pilot and then
record the explicit decision with `override-epsilon --selected-epsilon 0.001
--rationale "..."`. The override command validates that the chosen value is
covered by completed pilot artifacts before it updates the frozen selection and
unblocks `run-13b`.

Run tests with:

```text
conda run -n py312 python -m unittest discover research\experiments\lfo_representation\era2\tests
```

Use a Python environment with NumPy installed. PyTorch/XPU is optional and only
used by the `auto` backend for larger batched scoring workloads when available.

The smoke path writes generated artifacts under
[artifacts/smoke_flat/](./artifacts/smoke_flat/).

Experiment 11 runtime paths remain topology-free. Topology labels may appear
only in analysis-only bucket metrics; they are not model inputs, targets, loss
fields, decoder lookup keys, or model prediction head budget terms.
Experiment 11 rows should also keep the fixed 97-control-point x lattice
constant and spend the model prediction head budget only on base choice,
residual atom choice, and enabled scalars such as phase.
Topology may still be used later as an offline construction signal if the
resulting target schema and decoder lookup stay topology-free.
