# Fresh basic inverse trial on the settled renderer

Declared 2026-10-21 before execution. This is a fresh A2 trial after correcting the native startup transient, not a continuation of the stopped cold-renderer run. Execute only after the [settled numerical-gradient qualification](SETTLED_NATIVE_GRADIENT_CARD.md) passes and its publication audit reconstructs all requests and updates. The [original matched protocol](NATIVE_INVERSE_FIT_CARD.md) supplies the unchanged design and acceptance thresholds below.

Frozen plan ID: `7fc5c19575d3869d3b82d9790b2c89d50c0e92f69ce702971d1446a69c8e5801`. The [settled-gradient publication audit](SETTLED_NATIVE_GRADIENT_RESULTS.md) passes; no old target record or partial sound-trained checkpoint is imported.

## What starts fresh

Use the same 404-training-state pool, historical selected seed-zero inference model and matching 16,000-update training checkpoint. Freeze their byte hashes, model digest, codec, frontend and train-only retrieval memory in a new plan. Both arms start from the same 355,806-parameter model with fresh matched Adam states and a fresh seed-zero Gumbel stream. The prediction head remains **14 values**: nine bounded continuous controls and category widths three and two. Neither learned forward helper is used.

Every native source, gradient probe and final hard prediction uses the qualified **32,768-frame silent lead**, with all MIDI shifted by that exact count and only the added lead removed. Oscillator 1–3 and sampler random phase stay fixed under the existing policy. Strict audio QA is unchanged. There is no clipping penalty or amplitude normalization. Old cold targets, old probe losses and the 34-update sound-trained prefix cannot enter this run. Targets render when first needed; retain bounded checksummed measurements rather than training WAVs.

## Matched learning and decisions

Keep the original 32 category-balanced training states, two deterministic training performances per state and 128 matched updates across four epochs. Both arms infer from the first performance's audio; the training objective alternates performances. Adam uses learning rate 0.0001, batch one, float32 CPU, two Torch threads and gradient-norm cap five. The control arm uses masked parameter MSE/cross-entropy. The spectral arm uses only the fixed A0 spectral/envelope/stereo loss, with zero parameter anchor, native finite differences and conditional categorical losses. Hard Gumbel-softmax temperature falls from one to 0.5; native audits and exports use raw-logit argmax.

Save both models, both optimizer states, RNG and trace after every committed update. Persist optimizer charges and native attempts before execution. Exact resume requires this new plan, source/profile, retained targets and original deadline. A stopped run keeps its spent attempts and committed progress. Final checkpoints only: no sweep, native checkpoint selection, old-prefix warm start or automatic extension.

After fitting, audit all 32 training states under both performances, including sample-exact source repeats. Separately for each performance group, the spectral arm must have mean primary loss at most **90% of the historical starting model**, no worse than the matched parameter arm, and strictly lower mean legacy MRSTFT than the starting model. Incomplete or rejected cases fail. If either fit group fails, stop before validation.

Only after both fit groups pass, compare the first 12 category-balanced development states under reference and transfer MIDI. Infer each method's patch once from reference audio and reuse it unchanged for transfer. Compare the starting model, both trained arms, fixed template and unchanged train-only retrieval. Require exact repeats of all 24 source observations. Separately for reference and transfer, the spectral arm must have mean native MRSTFT at most **90% of both trained parameter arm and starting model**, beat template/retrieval, have mean primary spectral loss no worse than the parameter arm, and keep 90th-percentile tail error within **110% of the starting model**. These are the original gates, with no threshold revision after the rendering correction.

## Local cost and publication

Caps remain **3,328 total native attempts** (2,896 training and 432 audit), **132 charged matched optimizer attempts**, **four hours elapsed**, **8 GiB process-tree RAM** and **2 GiB retained evidence**. Replay is bounded by 512 records/512 MiB; use one fresh native worker per request. Worst-case planned requests remain 64 targets + 2,816 probes + 256 fit-audit requests + 168 conditional development requests = **3,304**. All rejected/interrupted attempts count. The new gradient qualification is additional compute, reported separately; no prior failed run is relabelled within this trial's budget.

Freeze the new plan and commit this card before `--execute`. Independently replay the actual matched updates, optimizers and Gumbel RNG from logged native losses. Reconstruct hard patches/requests, verify cached inputs/targets against source-repeat audio, and recompute all final gates before publication. Candidate audit audio is discarded, so distinguish evaluator receipts from independent waveform rescoring. Retain either outcome. A pass permits fresh confirmation and second-seed/listening/bundle work; it does not complete basic reconstruction or authorize wider prediction heads. No test states, Colab work or GitHub Actions changes.

```powershell
python research/training/reports/scripts/run_native_inverse_fit.py --pool BASIC_POOL --initial HISTORICAL_SELECTED_MODEL --retrieval MATCHING_TRAINING_CHECKPOINT --qualification SETTLED_GRADIENT_SUMMARY --output NEW_DIRECTORY
```

The first invocation only freezes the plan. Add `--execute` after recording its identity here and committing. Use `--execute --resume` only for that same attempted run and its original limits.
