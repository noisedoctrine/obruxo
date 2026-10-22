# Basic sound training with fixed category-score scales

Declared 2026-10-21 before execution. The [previous matched trial](SETTLED_NATIVE_INVERSE_RESULTS.md) learned its training sounds but missed the held-out improvement target. The [completed derivative check](CATEGORY_SIGNAL_RESULTS.md) identified saturated category scores and verified fixed training-only scales. This trial tests whether that repair improves actual reconstruction. It does not expand the prediction task.

Frozen plan ID: `b93183419006684feaa0dd593842a1b27511528bda2f9c6c21a4dd18dddb891b`. Compared with the previous settled plan, only the format, pinned implementation sources and explicit category-training metadata change. All other fields, including source states, MIDI, updates, initial weights, retrieval, renderer, optimizer, limits and gates, are identical.

Execution is [complete and independently replayed](CALIBRATED_NATIVE_INVERSE_RESULTS.md): training fit passes, but held-out gains of 7.23% reference / 4.24% transfer miss the unchanged target. This status note does not revise the frozen protocol below.

## Training change

Divide the unison logits by **20.31771924164867** and filter logits by **20.337628037582693** before both arms' training losses. The constants come from the initial model's 32 training-reference observations: float64 RMS after removing each row's class mean, floored at one. Labels and validation measurements do not enter calibration. Bind the completed diagnosis, pool, initial weights, qualification and Gumbel implementation by hash; recompute the constants from the saved training outputs before admission.

The parameter arm uses these scaled logits for cross-entropy. The sound arm uses them for straight-through Gumbel-softmax, with the existing temperature schedule from 1.0 to 0.5 and conditional native losses. Scaling changes stochastic training choices and their gradients. Raw-score argmax still determines every native audit prediction and exported patch; all initial hard predictions remain unchanged. Log gradients at the raw model outputs and scaled scores separately. Save calibration identity in checkpoints and diagnostic exports, and require it on resume and independent replay.

Keep the **14 dense outputs**: nine bounded continuous controls plus unison logits of width three and filter logits of width two. There are 355,806 trainable parameters. Both arms start from the same historical selected seed-zero weights, with fresh Adam states and the same seed-zero sampler. Do not initialize from either previous sound-trained arm. No learned forward helper, new controls, parameter anchor in the sound arm, or production probability sampling is introduced.

## Unchanged comparison

The [settled matched card](SETTLED_NATIVE_INVERSE_CARD.md) supplies the exact protocol. Use the same 32 category-balanced training states, two performances per state, 128 matched updates per arm, four epochs, batch one, Adam learning rate 0.0001, gradient-norm cap five, float32 CPU and two Torch threads. Input always comes from the reference performance; the target performance alternates. Render all 64 targets fresh when first needed and retain bounded measurements. No parent cache import is implemented.

Every native request uses the qualified 32,768-frame silent lead. Shift all MIDI by that count and trim only the added lead. Keep oscillator 1–3 and sampler random phase fixed, strict audio QA, and the unchanged A0 spectral/envelope/stereo loss. No clipping penalty or waveform normalization is enabled.

Audit all 32 training states under both performances with exact source repeats. Separately in each performance group, require mean primary sound loss at most **90% of the starting model**, no worse than the parameter arm, and strictly better mean legacy MRSTFT than the starting model. Stop before validation if either fit group fails.

Only after both fit groups pass, evaluate the same 12 development states under reference and transfer MIDI. Infer once from reference audio and reuse the patch for transfer. Separately in each group, require mean native MRSTFT at most **90% of both the starting model and trained parameter arm**, better than template and unchanged training-only retrieval, mean primary loss no worse than the parameter arm, and 90th-percentile tail error within **110% of the starting model**. Require all cases and sample-exact repeats of all 24 source observations. These thresholds have not changed in response to previous results. No test states are used.

## Resources, recovery and publication

Caps: **3,328 charged native attempts** (2,896 training, 432 audit), **132 charged matched optimizer attempts**, **four hours elapsed**, **8 GiB process-tree RAM**, **2 GiB retained evidence** and replay bounded by 512 records/512 MiB. The worst-case planned allocation remains 64 targets + 2,816 probes + 256 fit audit requests + 168 conditional development requests = **3,304**. Use one fresh native worker per request. Earlier experiments are separately reported costs.

Checkpoint both models, both optimizers, RNG and trace after every committed update; persist charges before execution. An interruption retains its receipts, spent attempts, original start time and deadline. Resume only the same frozen plan; do not reset budgets, drop failed attempts, change scales, select an intermediate checkpoint or extend fitting automatically. The machine is on mains power at launch; no power settings are changed.

Commit this card and implementation before execution. Independently replay every actual update, both optimizer states and Gumbel RNG; reconstruct hard patches/requests and rescore retained targets and inputs against source-repeat audio. Recompute final gates and publish either outcome. Candidate audit audio is discarded: distinguish evaluator receipts from independent waveform rescoring. Diagnostic exports remain unqualified for release.

```powershell
python research/training/reports/scripts/run_native_inverse_fit.py --pool BASIC_POOL --initial HISTORICAL_SELECTED_MODEL --retrieval MATCHING_TRAINING_CHECKPOINT --qualification SETTLED_GRADIENT_SUMMARY --category-calibration research/training/reports/data/category_signal_summary.json --output NEW_DIRECTORY
```

The first invocation freezes only. Add `--execute` after committing; use `--execute --resume` only for this same attempted run within its original limits. Supply the same `--category-calibration` to the full verifier or interrupted-prefix auditor. Preserve the frozen training source for historical reproduction.

A pass permits fresh-family/performance confirmation, second seed, actual blind listening and basic bundle checks. It does not complete Stage A or authorize broader heads. All work stays local; Colab and GitHub Actions remain deferred.
