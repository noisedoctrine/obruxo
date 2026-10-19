# Local basic helper: completed, not qualified

The frozen A1 run completed on 2026-10-19. **The helper is not reliable enough to guide inverse-model training.** It predicts held-out spectra better than the simple forward baselines, but fails the declared direction and real-improvement gates. The artifact remains marked unqualified; ordinary loading for inverse training refuses it. No inverse model was trained or promoted.

[Frozen card](BASIC_SPECTRAL_HELPER_CARD.md) · [Machine-readable evidence](data/basic_spectral_helper_summary.json)

## Actual sound results

| Check | Reference | Transfer | Required |
|---|---:|---:|---|
| Mean helper forward error | 0.575 | 0.596 | Reference ≤90% of best baseline; transfer no worse |
| Best baseline forward error | 0.903 | 0.930 | Train-only five-neighbor interpolation |
| Continuous-change ranking agreement | 13/24 (54%) | 17/24 (71%) | ≥70% each |
| Category-change ranking agreement | 10/12 (83%) | 10/12 (83%) | ≥70% each |
| Pan-change ranking agreement | 9/12 (75%) | 9/12 (75%) | ≥70% each |
| Active-detune ranking agreement | 5/8 (63%) | 7/8 (88%) | ≥70% each |
| Continuous-guided cases improved | 7/12 (58%) | 8/12 (67%) | ≥60% each |
| Joint-guided cases improved | 6/12 (50%) | 6/12 (50%) | ≥60% each |

All 240 qualification requests passed audio QA. All 24 phase-fixed repeats were sample-exact. There were no ranking ties; four inactive-detune cases per role were reported separately, as declared before collection.

Mean native MRSTFT change was negative for continuous-only guidance (−0.183 reference, −0.198 transfer) and joint guidance (−0.110, −0.039). Mean changes in the new spectral objective were also negative. However, every paired family bootstrap interval crossed zero, and the required case-level success rates failed. These are mixed development results, not confirmed reconstruction gains.

In **21 of 48 guided trials**, the helper predicted lower error while real Vital error increased. Joint guidance changed hard categories in two reference and five transfer cases. Better average forward prediction therefore did not establish trustworthy gradients. This result supports targeting the learned response to control changes before increasing model size or patch coverage; it does not identify a single proven cause.

## Compute and data integrity

The run used 1,024 training attempts, with 1,004 accepted records (98.05%), then all 12,000 declared helper updates. Twenty training attempts were retained as failures: 16 returned an incompatible audio shape and four clipped. No failures were silently replaced. All 16 shape failures used a two-beat performance at 150 BPM with a two-second tail; the duration boundary needs a separate renderer regression check. This issue does not explain away the failed qualification: all 240 qualification buffers passed shape/QA checks.

Elapsed time was **4,416.96 seconds (73.62 minutes)**. Peak sampled process-tree RSS was **1,610,149,888 bytes (1.50 GiB)** with zero sampling errors. The complete retained run occupied **1,237,486,627 bytes (1.15 GiB)**, including features, evaluation tensors and checkpoint backups. These fit the frozen four-hour, 8-GiB RAM and 2-GiB archive limits. Training audio was not archived. No final-test states were used.

The helper has 1,413,033 parameters. The patch prediction contract remains nine continuous values and five category scores: **14 dense outputs**, reducing to 11 codec inputs after category selection. The 30-update synthetic timing probe in the card is separate from these 12,000 real-data learning updates.

## Verification and next work

Seven focused tests pass: exact optimizer/minibatch recovery, preserved source splits, paired training recipes, exact sampled-frequency values/times, changed-feature rejection, input gradients with frozen helper weights, and refusal of unqualified artifacts. Ruff and whitespace checks pass. The evidence publication script recomputes the qualification decision from saved cases and checks request/repeat counts and plan identity.

Keep this artifact disconnected from inverse training. First reproduce and fix the duration-boundary rendering rejection without altering these historical receipts. Then declare a bounded comparison aimed at learning local control-change effects, using aligned paired measurements and independent control perturbations. Keep the same small patch vocabulary and admission gates. A revised helper must earn qualification before A2 starts; final listening and independent confirmation remain later work. Colab remains deferred.
