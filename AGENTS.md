# Experiment reporting

Every training or model-comparison report must show a compact, reader-facing summary of the Vital controls it can change.

- Give each compared experiment version its own table, introduced by a heading. Do not use a version/arm column.
- Group summary rows by meaningful control family or capability so readers can scan them quickly. Include fixed and disabled controls as named groups; do not hide them in a broad “other” or “everything else” row.
- Put the ✅/❌ status immediately after the family name, followed by the enabled/total instance count. Use ✅ when at least one instance is changeable and ❌ when none are; the count shows partial coverage (for example, `1 / 3`).
- Keep dense output counts separate from control counts. For example, one unison control represented by three category scores counts as one enabled control and three outputs.
- Include output representation/size and the update method or training signal in the summary where they clarify what the model does.
- Maintain a linked, machine-readable per-control inventory (CSV, JSONL, SQLite, or equivalent) with one record per exact Vital field/instance, including fixed and disabled controls. Record the experiment/version, status, output count, update method, supervision/loss path, ranges/choices, fixed value/source, masks and relevant notes. The report summary must reconcile with this inventory.
- Distinguish shared-model training from direct patch optimization, source-label warm starts from audio-loss training, fixed template values from omitted fields, and trainable heads from frozen heads.
