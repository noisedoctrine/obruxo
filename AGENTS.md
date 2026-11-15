# Research findings and experiment reporting

## Reader-facing findings first

Every empirical training, model-comparison or diagnostic report, and its PR summary, must let a reader answer **what did we learn, how strong is the evidence, and what should we do next?** without reading implementation details.

1. **Lead with the scientific answer.** Begin with one or two plain-language sentences stating the research question, the main measured finding (or that no empirical result exists), and the practical decision it supports. Say what a metric *means* before elaborating on experimental machinery.
2. **Show the primary results immediately.** Put a compact, cross-arm comparison table near the top (one row per compared model/condition; columns for the common decision-driving metrics). Report clear percentages or units and a meaningful matched baseline. Put sample sizes, denominators, gallery size, and the *independent diversity* of inputs nearby: e.g., 1,548 queries from 129 presets × **12 distinct MIDI performances** is not 1,548 independent performances.
3. **Unfold complexity in layers.** After the headline table, state the most important performance/difficulty breakdowns and a brief interpretation in ordinary language; follow with uncertainty, caveats, controls, and detailed diagnostics. Explain what each result establishes **and does not establish**. Distinguish exploratory comparisons from independently confirmed findings; specify what clustering an interval covers and what it does not.
4. **Keep research and operations visually separate.** Do not interleave retrieval/quality results with render QA, installation and checkpoint availability, cache reuse, profiling, test counts, memory, elapsed time, provenance or implementation changes. Put those in a later **Execution and reproducibility** section or appendix. Summarize essential dataset/QA coverage once near the research results only when needed to interpret the denominator. Do not treat engineering success or render QA as evidence of model quality.
5. **Make missing evidence conspicuous.** List models/conditions not run, small or biased samples, untested performance dimensions, and missing downstream reconstruction separately. Never promote a best-*tested* result to best overall. End with a short, evidence-limited recommendation or the next experiment needed to decide.
6. **Keep the first screen scannable.** Avoid per-arm repeating large metric/control tables, verbose provenance strings, raw debug logs, and giant inventories before the primary result. Put detail after the comparison or link to reproducible machine-readable artifacts. A table shared across model arms is encouraged for **research metrics**; the per-version table rule below applies to **Vital control-scope accounting**, not to primary metric comparisons.

## Vital-control scope reporting

Every training or model-comparison report must make the changeable Vital-control scope clear, but it must not displace the scientific results.

- For experiments that train, predict or edit Vital controls, give each compared version its own **control-scope** table, introduced by a heading; do not use a version/arm column in those detailed scope tables. Keep the **cross-arm research-results table** separate and first.
- Group scope rows by meaningful control family or capability. Include fixed and disabled groups as named groups; do not hide them in a broad “other” row.
- Put the ✅/❌ status immediately after the family name, followed by the enabled/total instance count. Use ✅ when at least one instance is changeable and ❌ when none are; the count shows partial coverage (for example, `1 / 3`).
- Keep dense output counts separate from control counts. For example, one unison control represented by three category scores counts as one enabled control and three outputs.
- Include output representation/size and the update method or training signal where they clarify what the model does.
- Maintain a linked, machine-readable per-control inventory (CSV, JSONL, SQLite or equivalent) for actual control-changing/prediction experiments, one record per exact Vital field/instance including fixed and disabled controls. Record experiment/version, status, output count, update method, supervision/loss path, ranges/choices, fixed value/source, masks and relevant notes. Scope tables must reconcile with this inventory.
- For frozen *representation-only* benchmarks with **no predicted or edited controls**, state that once, give a brief shared frozen-control summary, and link any applicable canonical inventory. Do not repeat the same zero-output, 979-control accounting inside every encoder's research-results table.
- Distinguish shared-model training from direct patch optimization, source-label warm starts from audio-loss training, fixed template values from omitted fields, and trainable heads from frozen heads.
