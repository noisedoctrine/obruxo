# Paired synthesis-identity feasibility screen

Measured locally on 2026-10-16. The paired sampler, contrastive decoder-representation objective, QA-qualified feature cache and held-out performance diagnostics are implemented. [Method and commands](../PAIRED_IDENTITY.md), [final plan](data/identity_screen_plan.json) and [aggregate evidence](data/identity_screen_summary.json) accompany this report.

The identity arm's mean native reconstruction distance was **+8.25% (higher)** than the matched supervised control: **1.731555 → 1.874434**. Identity-minus-control was **+0.142878**, with a descriptive 95% state-bootstrap interval **[-0.254230, +0.505310]** and 3/10 state wins. These are ten balanced development states and one learning seed. No model is promoted.

## Frozen comparison

This fixed direct-contrastive setting showed no held-out benefit in either identity retrieval or native reconstruction. The identity weight remains zero by default. The result does not settle other identity-learning designs; the next planned screen moves to an audio-based objective rather than tuning this setting.

Both arms use the broader alpha codec (19 continuous controls, nine categorical heads), the same 126 retained audio features from 63 training states, identical initial weights and identical paired minibatches. Each runs 1,000 CPU updates with 16 views per batch, learning rate 0.001 and four threads. The sole learning difference is contrastive weight 0 versus 1 at temperature 0.1. The objective aligns the 128-dimensional decoder representation across two distinct reference performances, with other training states as negatives. IDs never enter the network.

The panel freezes two validation states per synthesis family before native outcomes. Both models share source audio, fixed-template, training-only retrieval and source-repeat baselines. Each prediction is rendered under reference and reserved transfer performances. The primary metric equally weights those two performances within each state, then weights states equally. The paired bootstrap uses 2,000 state resamples with seed 0; it is descriptive and does not account for future repeated development decisions.

| Synthesis family | States | Control | Identity | Identity − control |
|---|---:|---:|---:|---:|
| chorus | 2 | 2.2686 | 1.7684 | -0.5002 |
| filtered | 2 | 1.7354 | 1.7566 | +0.0212 |
| layered | 2 | 1.8257 | 2.1594 | +0.3337 |
| modulated | 2 | 1.8621 | 1.9463 | +0.0842 |
| wavetable | 2 | 0.9660 | 1.7415 | +0.7755 |

## Cross-performance representation

Bidirectional retrieval matches each reference representation to its own reserved-transfer representation among ten validation states (chance 10%). Accuracy was **25% → 20%**. Matched-minus-different-state cosine margin was **0.1526 → 0.1290**. Normalized representation spread was **0.0627 → 0.0651**. Both diagnostics retain a 10-state denominator. These measures describe identity consistency; native reconstruction remains the primary outcome.

## Native QA and compute

The source schedule attempted 128 views for 64 states. One view clipped (19 samples; peak 1.6138) and was rejected. Its accepted partner was excluded too, leaving 126 views in 63 complete pairs. The original accepted-prefix collector stopped at its fixed attempt budget without an optimizer update. Before training either arm, the protocol was amended to persist every scheduled outcome and retain complete QA-qualified pairs. The original and amended plans are archived with the final plan. The old renders and features were recovered; no training view was rendered again.

Both trained arms were reused for evaluation after an orchestration fix selected the trained control checkpoint as the retrieval source. Runtime code, arm settings, features and model weights remained fixed. The final plan pins the reused artifact hashes. No native model-comparison outcome had been observed before this fix.

All **120/120 comparison renders** passed. Across source collection and comparison, **247/248 native attempts** passed; the one failure and one accepted orphan remain explicit. All five synthesis families remain in training. No audio archive, paid compute or final-test rendering was used. The alpha's 31 test states and the pilot's separate final test reserve remain untouched.

Retained features occupy **4.65 MiB**, within the 8 MiB cap. Peak sampled process-tree RSS across the learning arms was **0.71 GiB**. The recorded resumed-workflow segments total **6.77 minutes**; this excludes the original source rendering and pre-run source/contract verification. Original render wall times sum to **5.91 minutes**, and original feature extraction used 27.96 seconds. These are observed local CPU costs, not a dedicated throughput benchmark. Colab native rendering remains unqualified.

## Verification and next work

The full local suite passed 154 tests before the audited-cache addition. The final paired/cache suite passed all 11 tests, including QA-broken pair exclusion, content tamper detection, source-cache recovery and training without new renders; focused cache/recovery checks also passed after validation deduplication. Existing feature-seed, resume and inference checks were rerun. Ruff and whitespace checks pass. GitHub Actions remains disabled.

The next independent direction is an audio-based training objective. Identity-weight sweeps, further seed replication and detailed head tuning remain deferred. The current screen is a feasibility measurement on an authored vocabulary, not release evidence for arbitrary Vital presets.
