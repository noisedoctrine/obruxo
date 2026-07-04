# LFO Era 2

This is the clean workspace for the next LFO representation research phase.

Era 2 starts from the design contract in
[reports/LFO_ERA2_DESIGN.md](./reports/LFO_ERA2_DESIGN.md). The key point is that
topology may help offline codebook construction, but topology must not select
atoms after the codebook exists.

## Workspace

- [experiments/](./experiments/): Era 2 experiment plans, including
  [EXPERIMENT_10_PLAN.md](./experiments/EXPERIMENT_10_PLAN.md),
  [EXPERIMENT_11_PLAN.md](./experiments/EXPERIMENT_11_PLAN.md), and
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

Run Experiment 11, the topology-free flat-categorical residual screen:

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\run_era2.py --mkl-threading-layer SEQUENTIAL --native-threads 1 run-screen --async --screen experiment11 --profile quick --backend auto --metadata .\datasets\presetshare\raw\presetshare_vital_metadata.csv
```

The wrapper-level runtime flags are applied before NumPy/SciPy/PyTorch imports.
`--native-threads 1` sets `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, and
`MKL_NUM_THREADS` together. The individual flags are also available:
`--openblas-threads`, `--omp-threads`, and `--mkl-threads`.

`run-screen --async` starts the runner in the background, opens a monitor
window, prints a `Started async Experiment 11 run` message with the run
directory and log paths, then immediately returns to the shell. The monitor
refreshes every 30 seconds by default. The same is true when continuing a run:

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\run_era2.py --mkl-threading-layer SEQUENTIAL --native-threads 1 run-screen --async --screen experiment11 --profile quick --backend auto --metadata .\datasets\presetshare\raw\presetshare_vital_metadata.csv --run-dir <run_dir> --resume
```

Use `--monitor-refresh-seconds <seconds>` to change the monitor refresh rate.
Use `--no-monitor-window` to start the background runner without opening the
monitor. Foreground `run-screen` remains available by omitting `--async`.

Run the larger Experiment 11 screen on XPU with:

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\run_era2.py --mkl-threading-layer SEQUENTIAL --native-threads 1 run-screen --async --screen experiment11 --profile screen --backend xpu --metadata .\datasets\presetshare\raw\presetshare_vital_metadata.csv
```

Run artifacts are written under:

```text
era2/artifacts/experiment_11/runs/<run_id>/
```

Attach to an existing run from another terminal:

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\run_era2.py status --run-dir <run_dir> --watch 5
```

Regenerate analytics:

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\run_era2.py analyze --run-dir <run_dir>
```

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
