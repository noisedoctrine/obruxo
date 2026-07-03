# LFO Era 2

This is the clean workspace for the next LFO representation research phase.

Era 2 starts from the design contract in
[reports/LFO_ERA2_DESIGN.md](./reports/LFO_ERA2_DESIGN.md). The key point is that
topology may help offline codebook construction, but topology must not select
atoms after the codebook exists.

## Workspace

- [experiments/](./experiments/): Era 2 experiment plans, including
  [EXPERIMENT_10_PLAN.md](./experiments/EXPERIMENT_10_PLAN.md) and
  [ERA2_CORE_FRAMEWORK_PLAN.md](./experiments/ERA2_CORE_FRAMEWORK_PLAN.md).
- [reports/](./reports/): Era 2 research notes, design contracts, and future
  result writeups.
- [artifacts/](./artifacts/): future generated outputs.
- [notes/](./notes/): working notes and sketches.
- [code/](./code/): compact Era 2 framework code.
- [tests/](./tests/): focused framework tests.

## Core Framework

The Era 2 code starts fresh from the model-facing contract. It is intentionally
small and split around the concepts that matter for Experiment 10 and later:

```text
offline oracle construction -> reconstruction assets -> runtime interface -> decoder policy -> accounting/metrics
```

Run the framework smoke path with:

```text
python .\research\experiments\lfo_representation\era2\code\run_era2.py smoke-flat
```

Run Experiment 10 with the local `py312` environment:

```text
$env:MKL_THREADING_LAYER='SEQUENTIAL'
conda run -n py312 python .\research\experiments\lfo_representation\era2\code\run_era2.py run-screen --screen experiment10 --profile quick --backend auto
```

Estimate the theoretical fixed-grid ceiling before running Experiment 10:

```text
conda run -n py312 python .\research\experiments\lfo_representation\era2\code\run_era2.py grid-ceiling
```

Defaults:

```text
atom_grid_points = 24,36,48,60,72,96,100
dense_points = 1920
```

This audit asks how well a fixed `N`-point LFO grid can reproduce the corpus in
the best case. It changes atom dimensionality, not `D`, `W`, atom selection, or
model prediction head budget.

`run-screen` prints live status automatically while it runs. The same is true
when continuing a run:

```text
conda run -n py312 python .\research\experiments\lfo_representation\era2\code\run_era2.py run-screen --screen experiment10 --profile quick --backend auto --run-dir <run_dir> --resume
```

Use `--no-monitor` only for scripted runs where stdout should stay quiet.

Run artifacts are written under:

```text
era2/artifacts/experiment_10/runs/<run_id>/
```

Attach to an existing run from another terminal:

```text
conda run -n py312 python .\research\experiments\lfo_representation\era2\code\run_era2.py status --run-dir <run_dir> --watch 5
```

Regenerate analytics:

```text
conda run -n py312 python .\research\experiments\lfo_representation\era2\code\run_era2.py analyze --run-dir <run_dir>
```

Run tests with:

```text
conda run -n py312 python -m unittest discover research\experiments\lfo_representation\era2\tests
```

Use a Python environment with NumPy installed. PyTorch/XPU is optional and only
used by the `auto` backend for larger batched scoring workloads when available.

The smoke path writes generated artifacts under
[artifacts/smoke_flat/](./artifacts/smoke_flat/).

Experiment 10 runtime paths remain topology-free. Topology labels may appear
only in analysis-only bucket metrics; they are not model inputs, targets, loss
fields, decoder lookup keys, or model prediction head budget terms.
