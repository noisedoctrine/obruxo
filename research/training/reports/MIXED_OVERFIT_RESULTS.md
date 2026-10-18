# Mixed-source fit and phase-output correction

The matched eight-example check found a phase-learning defect. A sigmoid output cannot cross its zero boundary when the shortest cyclic error points through that boundary. The new opt-in periodic output wraps an unbounded phase modulo one. A focused gradient test reproduces the old trap and verifies the correction. Existing saved model behavior remains unchanged.

Both runs reuse the same five authored and three original-corpus recordings, labels and frozen decoder. They start fresh with seed 0 and use the same 1,000 updates, batch size eight, AdamW settings and losses. All nine original attempts remain acknowledged, including the earlier clipping rejection; no source audio is rendered again. The [aggregate receipt](data/mixed_overfit_summary.json) and [corrected qualification plan](data/mixed_overfit_plan.json) record the comparison.

| Final training measurement | Original sigmoid output | Periodic phase output |
|---|---:|---:|
| Continuous MAE | 0.032943 | 0.018386 |
| Active categorical accuracy | 99.494% | 99.747% |
| Total loss | 0.025640 | 0.016421 |
| Original-LFO reconstruction MSE | 0.012631 | 0.009724 |
| Optimizer updates | 1,000 | 1,000 |

The unchanged strict gate still fails because categorical accuracy must be exactly 100%. Both numerical MAE and total-loss reduction pass their limits. This is progress on fitting training examples, with no validation result or model promotion. Do not infer sound-quality gains from these parameter errors.

Perfect saved labels produce LFO reconstruction MSE 0.001550 on this batch, below the training gate's loss limit. Keep the existing immutable factory/LFO dictionaries for the first mixed-data baseline; a decoder refit is not needed to explain this fit failure. This does not establish a representation ceiling for all training states. All patch families and their supervision remain represented. The small batch has no active oscillator-seed labels, so it cannot qualify complete seed learning.

The original model's sixteen native requests verify all 775 scalar fields and six nested families. All eight fixed-phase prediction/repeat pairs have identical audio hashes. Seven predictions pass audio QA; one clips on both attempts. The original helper's repeat flag required QA acceptance, so its recorded repeat gate is conservatively false; the aggregate receipt separately verifies exact hashes from the retained native records, including that clipped pair. The corrected helper records repeatability independently from audio QA.

The periodic model passes all sixteen native requests, full readback and all eight exact repeat pairs. It halves the remaining active LFO code errors from sixteen to eight; all other active categorical families have zero errors. Eleven of twelve qualification gates pass; strict fitting is the only remaining failure. Learning takes 425.5 seconds (7.1 CPU minutes), with a 1.60 GiB training peak and 1.79 GiB qualification peak, below the 8 GiB limit. Fitting performs zero source renders and zero feature extraction. These figures exclude the earlier source collection and metadata loading before the trainer's timer starts.

Run instructions and artifact compatibility are in [MIXED_OVERFIT.md](../MIXED_OVERFIT.md). All 79 selected local checks pass, covering verified rejected-attempt reuse, phase gradients, unchanged legacy heads/contracts, export/evaluation loading, optional candidate learning and exact recovery. GitHub Actions remains disabled. Final-test audio remains reserved. The next task is complete fitting of the remaining LFO labels, followed by the approximately 32-example proof, mixed-data scaling and held-out sound-quality gates.
