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

Run Experiment 10, the standalone subdivision/direct-grid audit:

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment10_grid_audit.py
```

Defaults:

```text
point_budgets = 24,36,48,60,72,96,100
subdivisions = 24,25,32,36,37,40,48,49,60,61,64,72,73,80,96,97,100
dense_resolution = 1920
```

This experiment reports raw point-count coverage, subdivision coverage of
original x boundaries, and Era-1-style direct sampled-grid reconstruction. It
tests whether factor-of-3 grids beat or match higher non-factor-of-3 grids
rather than only beating smaller grids. It is intentionally outside the shared
Era 2 model-runner CLI.

Run Experiment 11, the topology-free flat-categorical residual screen:

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\run_era2.py --mkl-threading-layer SEQUENTIAL run-screen --screen experiment11 --profile quick --backend auto
```

The wrapper-level runtime flags are applied before NumPy/SciPy/PyTorch imports.
`--native-threads 1` sets `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, and
`MKL_NUM_THREADS` together. The individual flags are also available:
`--openblas-threads`, `--omp-threads`, and `--mkl-threads`.

`run-screen` prints live status automatically while it runs. The same is true
when continuing a run:

```text
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\run_era2.py run-screen --screen experiment11 --profile quick --backend auto --run-dir <run_dir> --resume
```

Use `--no-monitor` only for scripted runs where stdout should stay quiet.

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
