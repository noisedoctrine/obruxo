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

Run tests with:

```text
python -m unittest discover research\experiments\lfo_representation\era2\tests
```

Use a Python environment with NumPy installed. PyTorch/XPU is optional and only
used by the `auto` backend for larger batched scoring workloads when available.

The smoke path writes generated artifacts under
[artifacts/smoke_flat/](./artifacts/smoke_flat/).
