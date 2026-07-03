# Experiment 11 Plan: Method-Neutral Era 2 Accounting

## Summary

Experiment 11 should establish the Era 2 experiment accounting frame, not
prematurely choose a runtime atom-selection winner.

The core split is:

```text
offline oracle construction -> method-neutral reconstruction assets -> runtime model-choice interface -> model prediction head budget
```

This keeps topology, oracle search, atom/basis construction, and decoder-policy
decisions separate from the deployed model interface. The output of Experiment
11 should make future rows comparable across flat categorical, basis
coefficients, path addressing, and continuous address without mixing oracle
construction with model-choice accounting.

## Experiment 11 Goal

Experiment 11 has two jobs:

1. Define the row manifest and metrics needed for Era 2 comparisons.
2. Run or prepare the first small, clean, topology-free flat-categorical
   baseline screen to validate that accounting path.

It is not a broad addressing-method shootout. The point is to make sure each
future experiment records the parameters needed to estimate model prediction
head budget under different runtime choice methods while keeping the oracle path
independent.

## Core Design Rule

Every row must separate three identities:

```text
oracle_construction_id
runtime_interface_id
decoder_policy_id
```

`oracle_construction_id` says how offline reconstruction assets are built.
Topology may appear here only as an offline construction signal.

`runtime_interface_id` says what the deployed audio model must emit. Topology
must not appear here.

`decoder_policy_id` says which deterministic reconstruction choices are applied
with zero or explicit model-head cost.

## No-Runtime-Topology Contract

Every clean Era 2 row must satisfy:

```text
topology_used_at_runtime = false
topology_used_in_targets = false
topology_used_in_loss = false
topology_used_in_decoder_lookup = false
topology_used_in_head_accounting = false
```

The only topology flag that may be true in a clean row is:

```text
topology_used_in_construction = true
```

That flag means topology helped build or balance offline assets. It must not
change the model-facing target schema, decoder lookup, loss, masks, or
`head_outputs` formula.

## Required Row Parameters

Every Experiment 11-era row should log:

```text
experiment_id
oracle_construction_id
runtime_interface_id
decoder_policy_id
base_dictionary_size
D
scalar_families
scalar_outputs
categorical_outputs
continuous_outputs
head_outputs_formula
head_outputs_actual
dictionary_scope
codebook_storage_count
oracle_construction_time
oracle_encoding_time
topology_used_in_construction
topology_used_at_runtime
topology_used_in_targets
topology_used_in_loss
topology_used_in_decoder_lookup
topology_used_in_head_accounting
```

## Runtime-Specific Parameters

Flat categorical rows:

```text
addressing_scheme = flat_categorical
W_by_residual_layer
residual_atom_selection_outputs = sum(W_d)
```

Basis-coefficient rows:

```text
addressing_scheme = basis_coefficients
basis_count = P
coefficient_constraint
basis_construction_policy
coefficient_target_policy
continuous_basis_outputs = D * P
```

Path-address rows:

```text
addressing_scheme = path_address
branch_factors
path_length
leaf_capacity
reachable_atom_count
unused_leaf_count
tree_build_policy
path_loss_policy
head_sharing_policy
```

Continuous-address rows:

```text
addressing_scheme = continuous_address
address_dim = E
codebook_size
embedding_training_policy
distance_metric
nearest_neighbor_policy
```

## Required Metrics

Reconstruction quality:

```text
median_rmse
p90_rmse
p95_rmse
p99_rmse
max_rmse
strict_perfect_lfo_rate
node_max_error_median
node_max_error_p95
```

Model-budget metrics:

```text
head_outputs_actual
categorical_outputs
continuous_outputs
scalar_outputs
outputs_per_residual_layer
quality_per_100_head_outputs
```

Oracle/runtime separability diagnostics:

```text
oracle_quality_metrics
runtime_interface_budget_metrics
decoder_policy_cost
topology_contract_pass
```

Offline stratification metrics:

```text
topology_bucket_rmse_median
topology_bucket_rmse_p95
worst_topology_bucket
topology_p95_gap
```

These topology metrics are analysis-only. They must not imply runtime topology
conditioning.

## Learnability Proxies

Flat categorical:

```text
atom_usage_entropy_by_layer
dead_atom_rate
dominant_atom_share_by_layer
```

Basis coefficients:

```text
coefficient_mean_abs
coefficient_sparsity_rate
coefficient_p95_abs
basis_usage_energy_share
```

Path address:

```text
branch_usage_entropy_by_level
dead_branch_rate
invalid_leaf_rate
```

Continuous address:

```text
nearest_neighbor_margin_median
nearest_neighbor_margin_p95
embedding_norm_p95
```

## First Screen

The first Experiment 11 screen should be a small, fast, topology-free
flat-categorical baseline screen.

It should validate:

- the manifest fields;
- the no-runtime-topology contract;
- clean `head_outputs` accounting;
- oracle/runtime separation;
- report tables and plots.

Do not include basis coefficients, path addressing, or continuous address in
the first screen except as formula-only accounting rows. Full reconstruction
runs for those methods belong after the flat manifest/accounting path is proven
clean.

## First-Screen Row Shape

Use phase-only flat categorical residual layers:

```text
head_outputs = 32 + D * W + (D + 1)
```

Model-facing targets:

```text
base_index
base_phase
residual_layer_1_index
residual_layer_1_phase
...
residual_layer_D_index
residual_layer_D_phase
```

Budget bands should be internal Era 2 bands, not inherited Era 1 row names:

```text
small:  ~256-384 head_outputs
medium: ~512-640 head_outputs
large:  ~960-1152 head_outputs
```

Rows should emphasize narrow/deep flat-categorical settings, for example:

```text
W4 deep rows
W6 medium-depth rows
W8 reference rows
```

Exact row values should be chosen so `head_outputs_actual` lands near those
budget bands and all rows share the same runtime contract.

## What Experiment 11 Should Not Do

Experiment 11 should not:

- reuse topology-conditioned Era 1 target schemas;
- include topology in decoder lookup;
- compare by codebook storage as the primary capacity axis;
- treat Era 1 W/D names as baselines;
- promote gain or offset into the first screen;
- include snap as a default decoder policy;
- treat basis coefficients, path addressing, or continuous address as mature
  reconstruction contenders before the flat baseline is audited.

Era 1 rows may be cited as historical context or budget anchors only.

## Assumptions

- Flat categorical remains the first runnable baseline because it is the
  easiest runtime interface to audit.
- Basis coefficients are the first genuinely distinct non-tree follow-up
  candidate.
- Path addressing is the first structured discrete follow-up.
- Continuous address is later because embedding geometry and nearest-neighbor
  target generation add more ambiguity.
- The primary scarce resource is the model prediction head budget, not oracle
  search time, codebook storage, or serialized field count.
