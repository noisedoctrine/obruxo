# Experiment reporting

Every training or model-comparison experiment report must show a reader-facing table of the Vital controls that experiment can change.

- Give each compared experiment version its own table, introduced by a heading. Do not use an experiment-version/arm column.
- List controls explicitly, including each non-trainable or fixed control in the experiment's declared control inventory. Do not collapse them into an “all other controls” row.
- Put the visual status column immediately after the control name, using ✅ when the experiment can change that control and ❌ when it cannot. The next column is the enabled count out of the total controls named in that row.
- Follow with the update method, output representation and size (continuous values or categorical scores), training signal and gradient path, and the range/choices or fixed value.
- Count Vital controls, not dense output slots. For example, one unison-voice control with three category scores counts as one control and three outputs.
- Distinguish shared-model training from direct patch optimization, source-label warm starts from audio-loss training, fixed template values from omitted fields, and trainable heads from frozen heads.
- A machine-readable inventory may generate the table, but the complete control-by-control table must appear in or be directly visible from the experiment report.
