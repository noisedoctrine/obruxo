# LFO Representation Research

This folder is split by research era.

Era 1 is the preserved evidence base: historical code, reports, audits, plans,
plots, and generated artifacts from Experiments 1-9.

Era 2 is the active research direction. Its framework code starts fresh from the
clarified model-facing contract instead of copying the Era 1 file structure.

## Where Things Live

- [era1/](./era1/): historical implementation, reports, audits, plans, tests,
  and artifacts.
- [era1/reports/EXPERIMENTS_6_TO_9_CONSOLIDATED_REPORT.md](./era1/reports/EXPERIMENTS_6_TO_9_CONSOLIDATED_REPORT.md):
  consolidated Experiments 6-9 findings.
- [era1/audits/lfo_experiments_1to9_audit_results.md](./era1/audits/lfo_experiments_1to9_audit_results.md):
  conceptual audit of the Era 1 topology/accounting ambiguity.
- [era2/](./era2/): clean Era 2 research workspace.
- [era2/reports/LFO_ERA2_DESIGN.md](./era2/reports/LFO_ERA2_DESIGN.md):
  Era 2 research priors and design contract.
- [era2/experiments/ERA2_CORE_FRAMEWORK_PLAN.md](./era2/experiments/ERA2_CORE_FRAMEWORK_PLAN.md):
  implementation plan for the Era 2 core framework.
- [era2/code/](./era2/code/): compact framework package for Experiment 10 and
  later Era 2 work.

## Era 2 Contract

The deployed model receives audio-derived features and emits reconstruction
codes. It must not receive topology, predict topology, or use topology to
select atoms at runtime.

Topology may still be used, if useful, as an offline codebook-construction
signal. Once the codebook exists, the model-facing path must be topology-free.

The main capacity axis for Era 2 is the **model prediction head budget**:
categorical logits plus continuous scalar outputs required from the deployed
model for the LFO component.

## Current Folder Policy

- Do not add new Era 2 code under [era1/code/](./era1/code/).
- Do not import Era 1 code into new Era 2 work unless that dependency is
  deliberate and audited.
- Keep new Era 2 framework code under [era2/code/](./era2/code/).
- Keep generated Era 2 outputs under [era2/artifacts/](./era2/artifacts/).
- Keep Era 2 writeups under [era2/reports/](./era2/reports/) or
  [era2/notes/](./era2/notes/), depending on whether they are polished research
  notes/results or working notes.
