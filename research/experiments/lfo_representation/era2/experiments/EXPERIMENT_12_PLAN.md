# Experiment 12 Plan: W8D16 First-Principles Component Ladder

## Summary

Experiment 12 resets the W8D16 question from first principles. It fixes the
flat-categorical runtime shape:

```text
B = 32
W = 8
D = 16
control_point_count = 97
runtime_interface_id = flat_categorical_per_residual_layer
```

`W8D16` fixes atom choices and residual-layer count. It does not fix scalar
families or final `head_outputs`. The baseline is indices-only:

```text
head_outputs = 32 + 16 * 8 = 160
```

Phase, residual-layer gain, beam search, utility construction, and
topology-balanced offline construction are added as explicit components.

## Design Rules

- Phase is model-facing only when enabled. It adds `D + 1 = 17` scalar outputs.
- Optimized per-sample residual-layer gain is model-facing. It adds `D = 16`
  scalar outputs.
- Beam search and construction policy are oracle-side components. They may
  change quality, but they add zero model prediction head outputs.
- Topology may be used only during offline atom construction. It must not enter
  runtime inputs, targets, loss, decoder lookup, or head accounting.
- Exact piecewise-continuous phase is out of scope for this first ladder; the
  phase component uses the current FFT/lattice oracle search.

## Rows

Single-component rows:

```text
x12_c0_indices_only                         heads=160
x12_add_phase                               heads=177
x12_add_residual_gain                       heads=176
x12_add_beam4                               heads=160
x12_add_utility_construction                heads=160
x12_add_topology_balanced_utility_construction heads=160
```

Interaction and cumulative rows:

```text
x12_phase_gain                              heads=193
x12_phase_beam4                             heads=177
x12_gain_beam4                              heads=176
x12_phase_gain_beam4                        heads=193
x12_phase_gain_beam4_utility                heads=193
x12_phase_gain_beam4_topology_balanced_utility heads=193
```

There is no runtime-topology row in Experiment 12.

## Outputs

The standalone runner writes generated artifacts under:

```text
era2/artifacts/experiment_12/component_ladder/
```

The canonical report is:

```text
era2/reports/EXPERIMENT_12_W8D16_COMPONENT_LADDER_REPORT.md
```

The report should be findings-first and focus on RMSE behavior and component
interaction. Budget is recorded, but it is not the primary explanatory lens.
