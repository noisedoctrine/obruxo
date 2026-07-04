# Era 2 Core Framework Plan

## Summary

Build a small, self-contained Era 2 Python framework under
`research/experiments/lfo_representation/era2/code/` plus focused tests under
`research/experiments/lfo_representation/era2/tests/`.

This first pass is framework plus smoke validation. It does not run the full
Experiment 11 screen. It creates the reusable pieces needed for Experiment 11
and later work, then proves the topology-free flat-categorical path with a tiny
deterministic smoke run.

The framework is a clean break from Era 1 code structure. Stable math ideas are
allowed to carry forward, but the Era 2 boundaries are new:

```text
offline oracle construction -> reconstruction assets -> runtime interface -> decoder policy -> accounting/metrics
```

## Core Interfaces

- `RuntimeInterfaceSpec`
  Defines what the deployed model must emit. The first framework pass supports
  budget specs for `flat_categorical`, `basis_coefficients`, `path_address`, and
  `continuous_address`.

- `ModelPredictionHeadBudget`
  Computes base-selection outputs, residual atom-selection outputs,
  categorical outputs, continuous outputs, scalar outputs, and total
  `head_outputs_actual`.

- `TopologyContract`
  Validates the no-runtime-topology rule. Clean Era 2 rows may use topology
  during construction, but topology must not appear in runtime inputs, targets,
  loss, decoder lookup, or head accounting.

- `ExperimentRowManifest`
  Records the Experiment 11-era row fields: construction identity, runtime
  interface, decoder policy, dictionary scope, output counts, topology flags,
  storage counts, fixed LFO x-grid identity, and method-specific parameters.

- `ReconstructionAssets`
  Stores decoder-side assets: base dictionary, residual-layer dictionaries, and
  future basis/tree/address assets. Era 2 LFO curves use fixed uniform
  `control_point_count=97` decoder geometry. The derived 96 subdivisions can be
  discussed for lattice alignment, but the control-point count determines vector
  shape. Public terminology uses residual layer, not stage.

- `OracleEncoding`
  Stores model-facing targets only. For the flat smoke path this means base
  index/phase and residual-layer atom indices/phases. It must not carry
  topology.

## XPU Policy

Use XPU acceleration only where it helps. Era 1 showed that Intel XPU is useful
for large batched oracle alignment/scoring, while small jobs and smoke tests are
better left on NumPy/CPU because launch overhead and memory pressure dominate.

The framework should therefore include an optional backend that:

- uses NumPy by default for small workloads;
- uses `torch.xpu` only when PyTorch and an XPU device are available;
- switches to XPU for large batched nearest-code scoring;
- supports explicit `--backend numpy|xpu|auto`;
- keeps conservative chunking and CPU fallback;
- never makes XPU availability required for tests.

## Implementation Shape

Create a compact package:

- `lfo_era2/curve.py`
  Pure NumPy curve utilities: resampling, phase shifting, synthetic smoke
  curves.

- `lfo_era2/accelerator.py`
  Optional NumPy/XPU backend for batched nearest-code scoring.

- `lfo_era2/accounting.py`
  Model prediction head budget formulas for flat categorical, basis
  coefficients, path address, and continuous address.

- `lfo_era2/contracts.py`
  Topology contract validation and public-key terminology checks.

- `lfo_era2/assets.py`
  Dataclasses for runtime specs, decoder policy, reconstruction assets, and
  oracle encodings.

- `lfo_era2/flat.py`
  Minimal topology-free flat-categorical construction, encoding, and decoding.

- `lfo_era2/metrics.py`
  Reconstruction metrics and simple atom-usage diagnostics.

- `lfo_era2/manifest.py`
  JSON/CSV-ready manifest and summary helpers.

- `lfo_era2/cli.py`
  Thin CLI with a `smoke-flat` command.

## Smoke Path

The smoke command writes an artifact bundle under
`era2/artifacts/smoke_flat/`:

- `manifest.json`
- `summary.csv`
- `targets_schema.json`
- `topology_contract.json`

The smoke path should prove:

- the deployed target schema is topology-free;
- the flat-categorical formula is exactly
  `head_outputs = 32 + D * W + (D + 1)`;
- the fixed 97-control-point x lattice adds zero model prediction head outputs;
- artifacts separate `oracle_construction_id`, `runtime_interface_id`, and
  `decoder_policy_id`;
- topology flags pass the Era 2 contract;
- decoding works from `OracleEncoding` without topology lookup.

## Test Plan

Run:

```text
python -m unittest discover research/experiments/lfo_representation/era2/tests
```

Tests should cover:

- accounting formulas for all accepted runtime interfaces;
- topology contract pass/fail behavior;
- manifest required fields;
- flat smoke target schema;
- decoder reconstruction without topology;
- public manifest/schema keys do not use `stage`.

## Assumptions

- No full Experiment 11 screen in this first pass.
- No SciPy, plotting, multiprocessing, or background monitors.
- NumPy is the only required third-party dependency for the framework tests.
- PyTorch/XPU is optional and used only when available and beneficial.
- Basis/path/continuous interfaces get accounting support now; working
  reconstruction implementations come later.
