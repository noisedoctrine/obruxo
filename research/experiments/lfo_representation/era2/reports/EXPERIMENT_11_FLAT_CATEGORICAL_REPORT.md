# Experiment 11 Flat-Categorical Report

## Main Findings

This run completed `6` topology-free flat-categorical rows.
Corpus mode: smoke=`False`, requested sample fraction=`1.0`.
Effective dataset: train=`13295`, validation=`1605`, total active LFO rows=`14900`.
The LFO vector shape recorded by completed rows is `97` control points.
Topology contract passed for `6/6` completed rows.
Active oracle phase search was recorded for `0/6` completed rows.

Best validation P95: `w8_d72` at `0.2842312455177307` with `681` head outputs.
Best validation median: `w4_d120` at `0.21102221310138702` with `633` head outputs.
The Pareto frontier has `5` row(s) when sorting by model prediction head budget and validation P95.

![Validation P95 vs model prediction head budget](./images/experiment_11/validation_p95_vs_head_outputs.png)

![Validation median vs model prediction head budget](./images/experiment_11/validation_median_vs_head_outputs.png)

Important caveat: `6` row(s) count phase scalar outputs but record only one oracle phase candidate. Treat those rows as framework/readiness runs, not fair quality comparisons against phase-active Era 1 rows.

## Best Rows By Validation P95

| row | budget band | head outputs | validation p95 RMSE | validation median RMSE | elapsed seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| w8_d72 | medium | 681 | 0.2842312455177307 | 0.21235518157482147 | 54.956672500120476 |
| w4_d120 | medium | 633 | 0.29377031326293945 | 0.21102221310138702 | 53.365473200101405 |
| w6_d80 | medium | 593 | 0.29824167490005493 | 0.21798233687877655 | 50.83388559985906 |
| w8_d28 | small | 285 | 0.31310710310935974 | 0.2189859002828598 | 25.258332999888808 |
| w6_d32 | small | 257 | 0.31825578212738037 | 0.21776773035526276 | 22.296140299877152 |
| w4_d48 | small | 273 | 0.3273541331291199 | 0.22117942571640015 | 22.556008500047028 |

## Budget Band Read

| budget band | rows | best validation P95 | min head outputs | max head outputs |
| --- | ---: | ---: | ---: | ---: |
| medium | 3 | 0.2842312455177307 | 593.0 | 681.0 |
| small | 3 | 0.31310710310935974 | 257.0 | 285.0 |

## Frontier Read

Lower validation P95 is better. `head_outputs_actual` is the model prediction head budget; the fixed x lattice is decoder-owned and does not add outputs.
Oracle phase-search resolution is also not part of this budget: the deployed model emits one continuous phase scalar per base/residual layer either way.

![Validation P95 by row](./images/experiment_11/validation_p95_by_row.png)

| row | head outputs | validation p95 RMSE | budget band |
| --- | ---: | ---: | --- |
| w6_d32 | 257 | 0.31825578212738037 | small |
| w8_d28 | 285 | 0.31310710310935974 | small |
| w6_d80 | 593 | 0.29824167490005493 | medium |
| w4_d120 | 633 | 0.29377031326293945 | medium |
| w8_d72 | 681 | 0.2842312455177307 | medium |

## Budget Projection Notes

Run-local `analytics/budget_projections.csv` includes formula-only views for alternate dictionary addressing strategies, currently including binary path addressing over the same residual-layer leaf capacity. These rows are budget views, not quality claims: changing atom indexing changes the learning problem and may require a different dictionary organization.

## Runtime And Readiness Notes

- Run id: `run_20260704_205908`
- Screen: `experiment11`
- Smoke: `False`
- Corpus sample fraction requested: `1.0`
- Topology may be used for offline construction, but runtime topology is not part of inputs, targets, loss, decoder lookup, or model prediction head budget.
- Any topology bucket metrics are analysis-only.
- `oracle_phase_search_policy` and `oracle_phase_candidate_count` describe oracle target generation, not deployed head-output cost.
- CSV analytics remain in the run artifact directory. This markdown file is the canonical Experiment 11 report.

![Runtime vs model prediction head budget](./images/experiment_11/runtime_vs_head_outputs.png)
