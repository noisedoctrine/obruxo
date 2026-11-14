# Balanced-bank supervised learnability check — completed

## Question

Does short training against source-setting labels give the shared predictor a useful start, and does it make the real Vital sound closer?

## Fixed setup and result

- Same 14-output model and eleven adjustable settings as the balanced sound-training check; all remaining fields stay at the reviewed template.
- 48 CPU Adam updates at batch size four, using the frozen balanced schedule and two original reference examples per update. Loss used codec source-setting labels only, with no audio-quality term.
- Evaluation: all 18 balanced training performances rendered once in Vital. They are training examples, not held-out test data.
- Label error fell, but missed its pre-set fitting target. Mean audio scores worsened across all three source groups, so the candidate was not admitted.
- Budget: 20 minutes and 4 GiB; no reserved/test examples; fixed phases; qualified Vital renderer.

## Reports

- [Full results and separate control tables for both versions](BALANCED_SUPERVISED_CONTROL_RESULTS.md)
- [Compact metrics, learning curve and all 18 sound scores](data/balanced_supervised_control_20261114_results.json)
- [Exact field-by-field inventory for both versions](control_scope/balanced_supervised_control_20261114.jsonl)
- [Run script](scripts/run_balanced_supervised_control.py)

A separate [four-schedule comparison](PRESET_LOSS_SCHEDULE_RESULTS.md) already tests fading, fixed label/audio mixtures, a paper-style fixed weight and abrupt warm-start removal. Its raw and gain-matched winners differ. Do not repeat it; continue with the current balanced-bank sound-feedback problem.
