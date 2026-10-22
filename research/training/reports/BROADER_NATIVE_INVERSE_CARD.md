# Broader basic sound training at the same update count

Declared 2026-10-21 before execution. The [calibrated 32-state trial](CALIBRATED_NATIVE_INVERSE_RESULTS.md) learns its fitted sounds but misses held-out quality. The [saved-data diagnosis](BASIC_FIT_COVERAGE_RESULTS.md) finds uneven coverage in that fine-tuning subset. This comparison tests broader training coverage while retaining the **14-output, 355,806-parameter** model. Coverage is a hypothesis, not an established cause.

Frozen plan ID: `8f2a9c858d8c0d2ca7d69deebc1c2a5f2f5d45e4e040619fde7de1d1574f2f53`. The initial card and implementation were committed before execution.

The [completed, independently replayed outcome](BROADER_NATIVE_INVERSE_RESULTS.md) passes full/original-subset training fit, but misses held-out quality: reference improves 5.82% and transfer is 0.24% worse than the starting model. This status note does not change the frozen protocol below.

## Data and selection

Preserve the original 32 states in their exact order, with identical targets and both MIDI recipes. Add 32 from the existing 404-state **training split only**. Keep all prior split assignments. Neither development/test target values nor sound scores enter selection. Development membership and performances remain byte-equivalent to the previous plan.

Across the six source-category combinations, the final counts are 11, 11, 11, 11, 10 and 10 in lexicographic category order. At each addition, choose the category with the largest unfilled quota, resolving category ties lexicographically. Within it, select the candidate whose nearest already-selected training target is farthest away. Distance is mean squared difference over source-active normalized controls within the same category; square root is reported for readability. Resolve exact distance ties by state ID. Record every addition, its nearest selected neighbor and distance in the plan. Inactive controls do not influence selection. This training-data rule is not a perceptual objective or an inference-time search.

Infer input features from each state's reference performance and alternate the two training target performances as before. Use **two passes over 64 states**, one per target performance, for the same **128 updates per arm**; the previous comparison used four passes over 32. Shuffle each pass with the same seed-zero generator. Every state/performance pair receives one matched update. Generate 128 fresh targets on demand; import no parent target cache or sound-trained weights.

## Unchanged learning and sound checks

Both arms start from the historical selected initial weights, with fresh matched Adam states and saved seed-zero Gumbel randomness. Keep batch one, learning rate 0.0001, gradient-norm cap five, float32 CPU and two Torch threads. Preserve the checked category scales in both training losses, temperature 1.0 to 0.5, zero parameter anchor in the sound arm, and raw-score argmax for hard predictions/exports. No outputs, assets or learned helper are added.

Every source, probe and audit uses the qualified 32,768-frame silent lead and fixed oscillator/sampler random-phase policy. Preserve strict audio QA, exact note/tail frames and the original spectral/envelope/stereo objective. Do not normalize amplitude or enable the investigated clipping penalty.

Audit all 64 states under both performances, with exact source repeats. Separately for each performance, require mean primary sound loss at most **90% of the starting model**, no worse than the parameter arm, and strictly better mean legacy MRSTFT than the starting model. **Apply those same fit gates separately to the retained original 32 states.** Both populations must pass before development scoring; newly added cases cannot dilute an original-subset failure.

Then score the unchanged 12 development states under reference and transfer MIDI, inferring each patch once from reference audio. In each performance group require mean native MRSTFT at most **90% of both starting model and parameter arm**, better than template/training-only retrieval, mean primary sound loss no worse than the parameter arm, and 90th-percentile tail error within **110% of the starting model**. Require every case and exact source repeats. Thresholds are unchanged; final-test states remain reserved.

## Explicit local allocation

The native cap increases by exactly 320 requests for the additional targets and fit audits. Worst-case planned use is **3,624**: 128 targets + 2,816 probes + 512 fit-audit requests + 168 conditional development requests. Charge at most **3,648 attempts**, split into 2,960 training and 688 audit attempts, preserving the original 24-attempt headroom. All failures/retries count. Keep **132 charged matched optimizer attempts**, **four hours elapsed**, **8 GiB process-tree RAM**, **2 GiB retained evidence**, 512-record/512-MiB replay and one fresh native worker per request. Previous runs remain separate reported costs.

The frozen storage preflight accounts for exact spectral tensor layouts and feature sizes, retained stereo source audio with both current/previous atomic-save copies, serialization allowances and a 256-MiB reserve for checkpoints, exports, ledgers and temporary files. Its planning bound is **1,665,817,728 bytes (1.55 GiB)**, below the unchanged 2-GiB cap plus runtime guard. This is an allowance, not a measured peak; actual runtime accounting still stops at the original cap. Three layout checks compare byte formulas against real `SpectralTargets` tensors.

The laptop reports battery power at preparation (97%). The user explicitly authorizes starting on battery with checkpoints. No power settings are changed. Any host sleep still counts toward the original elapsed deadline; checkpointing does not reset budgets or erase failed attempts.

Save both models, both Adam states, RNG and trace after every committed update. Persist native/optimizer charges before execution. Exact resume requires the same selection, plan, calibration, source/profile, target hashes and original deadline. Preserve any interruption and all spent attempts. Use final checkpoints only, with no sweep, checkpoint selection, automatic extension or budget reset.

## Publication

Commit this card and implementation before execution. Independently replay all actual updates, optimizers and Gumbel RNG, reconstruct native requests/hard patches, rescore retained targets/input features from exact source-repeat audio, and recompute both complete-population and original-subset gates. Candidate audit audio is discarded as declared; distinguish evaluator receipts from independent waveform rescoring. Publish either outcome and keep diagnostic exports unqualified for release.

Twenty focused checks pass across training-only selection, deterministic farthest additions, exact target byte layouts, original-subset failure guards, both training losses and model/optimizer/Gumbel recovery. The real frozen plan verifies identical original recipes, validation, initial weights, calibration and optimizer, with exactly one update per new state/performance pair. Ruff and whitespace checks pass.

```powershell
python research/training/reports/scripts/run_native_inverse_fit.py --pool BASIC_POOL --initial HISTORICAL_SELECTED_MODEL --retrieval MATCHING_TRAINING_CHECKPOINT --qualification SETTLED_GRADIENT_SUMMARY --category-calibration research/training/reports/data/category_signal_summary.json --training-states 64 --output NEW_DIRECTORY
```

The first invocation freezes only; add `--execute` after committing. Resume only the same attempted run with `--execute --resume` and the same flags. Full/prefix auditors recover the state count from the frozen plan and require the identical calibration file. Historical trials require their pinned source. Keep PRs ready for review; never merge automatically. GitHub Actions stays disabled and Colab deferred.

A passing comparison permits independent-family/performance confirmation, a second seed, actual blind listening and basic bundle checks. It does not complete Stage A or authorize wider synthesis/modulation heads.
