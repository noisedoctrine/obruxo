# LFO Era 1

This folder preserves the first LFO representation research arc.

It contains the historical implementation, run scripts, tests, generated
artifacts, reports, audits, and planning notes for Experiments 1-9. These files
remain useful as evidence, but they are not the starting point for Era 2 code.

## Contents

- [code/](./code/): historical Python package, experiment runners, report
  generators, monitor scripts, and launch scripts.
- [tests/](./tests/): tests for the historical implementation.
- [reports/](./reports/): experiment findings, consolidated reports, and report
  images.
- [audits/](./audits/): conceptual audits of the Era 1 representation and
  accounting assumptions.
- [plans/](./plans/): historical experiment plans and research notes.
- [artifacts/](./artifacts/): generated run outputs.

## Boundary

Era 1 code may be inspected for algorithms and historical behavior. New Era 2
work should not import it by default. Any reuse should be explicit, audited, and
checked against the Era 2 no-runtime-topology contract.
