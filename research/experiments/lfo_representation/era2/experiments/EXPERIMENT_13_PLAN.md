# Experiment 13 Plan: Fixed-W8D16 Stacked Grid

## Summary

Experiment 13 is the large stacked grid that follows Experiment 12. It keeps the
same fixed W8D16 runtime contract:

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

Experiment 13 should not pick candidates automatically. Read the Experiment 12
grouped report and manually choose the values to stack.

## Run Command

Experiment 13 requires a dedicated stacked-grid runner after manual candidate
selection. The intended command shape is:

```powershell
conda run --no-capture-output -n py312 python .\research\experiments\lfo_representation\era2\code\experiment13_stacked_grid.py --mkl-threading-layer SEQUENTIAL --native-threads 1 run --async --backend xpu --metadata .\datasets\presetshare\raw\presetshare_vital_metadata.csv --corpus-sample-fraction 1.0 --selection .\research\experiments\lfo_representation\era2\experiments\experiment13_selection.json --monitor-refresh-seconds 30
```

The selection file should contain the manually chosen PascalCase values from
the Experiment 12 report. The Experiment 13 runner and selection-file schema
are not implemented yet.

## Manual Selection Inputs

Choose:

```text
Top 3 construction_policy values
Top 3 utility_candidate_budget values
Top 3 layer_normalization_policy values
Top 3 no_damage_policy values
Top 3 atom_preprocessing_policy values
Top 2 duplicate_suppression_policy values, if both remain useful
Beam4Path and/or Beam8Path
IndicesOnly and/or PhaseAndResidualGain
```

Selection should consider all co-primary metrics:

```text
validation_median_rmse
validation_strict_perfect_lfo_rate
validation_p95_rmse
validation_node_max_error_p95
```

## Full Combination Size

Maximum expected size if both scalar schemas, both beam values, three values for
five variables, and two duplicate-suppression values are retained:

```text
2 * 2 * 3 * 3 * 3 * 3 * 3 * 2 = 1944 rows
```

If that is too large after seeing Experiment 12 runtimes, reduce first by
dropping the weaker scalar schema or weaker beam width. Do not drop
construction or normalization candidates first.

## Output Expectations

Experiment 13 should preserve the same manifest fields as Experiment 12:

```text
screening_variable
screening_value
scalar_schema
path_search_policy
construction_policy
utility_candidate_budget
layer_normalization_policy
no_damage_policy
atom_preprocessing_policy
duplicate_suppression_policy
reserved_atom
active_atoms_per_layer
```

The report should focus on interactions between selected values rather than
single-variable screening.
