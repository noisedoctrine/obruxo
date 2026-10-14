# 205-versus-404-state training coverage experiment

Completed 2026-10-13. The 404-state, one-reference-view model at 16,000 updates was selected on the original 21 validation cases, with a primary MR-STFT distance of **0.9124**, versus **0.9870** for the incumbent and **0.9906** for the matched 205-state one-view control. That is a 7.6% lower mean validation score than the incumbent on this fixed small case set. Expanded 16K beat the incumbent on only **8/21 states**, so the mean gain does not show broad statewise dominance. The paired state bootstrap is descriptive; it is not adjusted for selecting among checkpoints.

The 404-state arm completed 16,000 updates with all 404 accepted training-state/reference recipes available before update one. It narrowly missed the trainer’s tiny-overfit MAE cutoff (0.05253 vs 0.05), while categorical accuracy was 1.0 and replay loss fell from 1.8766 to 0.00597. This is a learning-loop gate result, not a release threshold; the declared full-case native validation proceeded because the planned training and coverage artifacts were complete.

The [frozen experiment plan](data/state_coverage_plan.json) was written before native training and evaluation (SHA-256 `eebef985adb7182f3adcece3b148d51f29a7c6e7e43aa7cd13828337e25a7246`). The [machine-readable results](data/state_coverage_summary.json) contain checkpoint identities, paired comparisons, diagnostics, QA denominators and resource measurements. The [native evaluation output](../outputs/state-coverage-validation-20261013/summary.json) retains all case-level measurements. The [comparison plot](images/state_coverage_comparison.png) shows training fit and primary validation scores.

## Design and fixed controls

The original pool remains the evaluation pool: 205 training states, the same 21 validation state/reference/transfer pairs, and the same 205-record training-only retrieval bank. All 64 test states in the expanded pool remain reserved. The expanded training pool has 404 states and preserves all prior state assignments. The expanded arm and cached 205-state controls use the same vocabulary, architecture, frozen frontend, loss, AdamW learning rate 0.001, seed 0, batch size 32 and float32 CPU training. Each state contributes one reference performance. The full replay was held in memory before update one, so the comparison does not change gradual warmup.

The expanded model checkpoints were taken at 4,000, 8,000 and 16,000 updates. Native warmup had a 480-attempt cap; it used 404 accepted renders with no rejection, error or held-out recipe. The existing 205-state one-view checkpoints were reused from the verified cached-feature curve, and the incumbent two-view checkpoint remained in the candidate set. The experiment did not extend its training budget based on partial validation.

The expanded arm freshly re-rendered the shared 205 parent training states. A post-warmup comparison paired those records with the cached 205-state replay: **205/205** had the same reference-performance ID, identical audio SHA-256, and exactly equal frontend tensors. The expanded arm also added the 199 new training states. The 404-record replay used 15,618,640 bytes, below the 32 MiB cap.

## Validation results

The primary score is the equal-weight mean of reference and transfer MR-STFT for each state; lower is better. Selection requires all 21 states in both roles. Candidate order breaks ties.

| Candidate | Updates | Primary mean | Reference mean | Transfer mean | Validation active-control MAE |
|---|---:|---:|---:|---:|---:|
| incumbent | 4,000 | 0.9870 | 0.9307 | 1.0432 | 0.1938 |
| one_4000 | 4,000 | 1.0786 | 1.0214 | 1.1359 | 0.2110 |
| expanded_4000 | 4,000 | 1.0324 | 0.9738 | 1.0909 | 0.1927 |
| one_8000 | 8,000 | 1.0222 | 0.9453 | 1.0990 | 0.2065 |
| expanded_8000 | 8,000 | 1.0451 | 0.9635 | 1.1267 | 0.2052 |
| one_16000 | 16,000 | 0.9906 | 0.9162 | 1.0649 | 0.2004 |
| expanded_16000 | 16,000 | 0.9124 | 0.8304 | 0.9943 | 0.1884 |

The selected 404-state 16K checkpoint is lower by **0.0746** than the incumbent (paired 95% interval for expanded minus incumbent: [-0.2984, +0.1070]). Against the matched 205-state 16K control, the paired delta is **-0.0782** (95% interval [-0.2398, +0.0676]). **Both intervals include zero.** The selected expanded 16K model wins on 8/21 states against the incumbent and 12/21 against the matched 205-state 16K control. These results use one seed and 21 validation states; the bootstrap does not adjust for checkpoint selection or multiple comparisons.

## Paired planned comparisons

Each delta is left candidate minus right candidate; negative favors the left candidate. Intervals use 2,000 paired state-bootstrap resamples with seed 0.

| Contrast | Mean delta | 95% interval | Wins / paired |
|---|---:|---:|---:|
| expanded 4000 minus one 4000 | -0.0462 | [-0.2016, +0.0904] | 11/21 |
| expanded 8000 minus one 8000 | +0.0230 | [-0.1404, +0.1657] | 6/21 |
| expanded 16000 minus one 16000 | -0.0782 | [-0.2398, +0.0676] | 12/21 |
| expanded 8000 minus expanded 4000 | +0.0127 | [-0.1154, +0.1471] | 11/21 |
| expanded 16000 minus expanded 4000 | -0.1200 | [-0.3099, +0.0566] | 11/21 |
| one 4000 minus incumbent | +0.0917 | [-0.0111, +0.1899] | 6/21 |
| expanded 4000 minus incumbent | +0.0454 | [-0.1767, +0.2201] | 7/21 |
| one 8000 minus incumbent | +0.0352 | [-0.0857, +0.1625] | 9/21 |
| expanded 8000 minus incumbent | +0.0581 | [-0.1627, +0.2314] | 5/21 |
| one 16000 minus incumbent | +0.0036 | [-0.1240, +0.1352] | 11/21 |
| expanded 16000 minus incumbent | -0.0746 | [-0.2984, +0.1070] | 8/21 |
| incumbent minus train retrieval | -0.2876 | [-0.4916, -0.0526] | 17/21 |
| one 4000 minus train retrieval | -0.1959 | [-0.4034, +0.0456] | 14/21 |
| expanded 4000 minus train retrieval | -0.2422 | [-0.4910, +0.0617] | 16/21 |
| one 8000 minus train retrieval | -0.2524 | [-0.4037, -0.0889] | 20/21 |
| expanded 8000 minus train retrieval | -0.2295 | [-0.4579, +0.0068] | 13/21 |
| one 16000 minus train retrieval | -0.2840 | [-0.4779, -0.0857] | 18/21 |
| expanded 16000 minus train retrieval | -0.3622 | [-0.5566, -0.1623] | 18/21 |

The `Validation active-control MAE` column is computed from each predicted patch against its validation target. Training replay fit is separate:

| Candidate | Updates | Training replay active-control MAE |
|---|---:|---:|
| incumbent | 4,000 | 0.1282 |
| one_4000 | 4,000 | 0.0753 |
| expanded_4000 | 4,000 | 0.1367 |
| one_8000 | 8,000 | 0.0292 |
| expanded_8000 | 8,000 | 0.0918 |
| one_16000 | 16,000 | 0.0225 |
| expanded_16000 | 16,000 | 0.0525 |

Selected-versus-incumbent parameter and level diagnostics are:

| Diagnostic | Incumbent | Expanded 16K | Change |
|---|---:|---:|---:|
| Active-control MAE | 0.1938 | 0.1884 | -0.0054 |
| Envelope attack MAE | 0.1682 | 0.1369 | -0.0313 |
| Envelope decay MAE | 0.2315 | 0.2374 | +0.0059 |
| Envelope sustain MAE | 0.2041 | 0.1977 | -0.0064 |
| Envelope release MAE | 0.3325 | 0.3321 | -0.0004 |
| Oscillator level MAE | 0.1839 | 0.1522 | -0.0317 |
| Unison categorical accuracy | 0.810 | 0.571 | -0.238 |
| Filter categorical accuracy | 1.000 | 1.000 | +0.000 |
| Reference RMS-relative error | 0.2867 | 0.2142 | -0.0724 |
| Transfer RMS-relative error | 0.2632 | 0.2146 | -0.0486 |

These diagnostics do not replace the audio metric; all validation metrics cover only the same 21 cases.

## Native and artifact verification

All **462/462** planned native evaluation renders passed; all seven candidates are scorable on **21/21** states for both roles. The 205-record retrieval digest remains `1831f6a3aeedfcd00991ab98fa8ee106bb4c70970d0c2686a3709e7b6b60d6ac`. All **147/147** predicted `.vital` files pass codec and Vital schema validation. The [selected inference bundle](../outputs/state-coverage-validation-20261013/selected-model.pt) has the expanded training-pool ID and weights identical to the selected 16K checkpoint. No training or evaluation WAV archive was written. The evaluation status is complete and the final test split was not evaluated.

Training used 21.31 CPU minutes across the fresh warmup and two resume segments. Evaluation took 16.29 minutes; combined recorded wall time on CPU was 37.60 minutes; sampled training process-tree peak memory was 882 MiB. The 16K training CLI returned its intended nonzero `SystemExit(2)` because the tiny-overfit continuous MAE was 0.05253. The PowerShell execution layer reported exit code 1; the completed 16K checkpoint, `model.pt`, and `status.json` confirm training reached the planned step. The validation command returned shell exit code 0.

## Limits and next step

This is a single-seed result on 21 authored/generated validation cases under one local renderer. Unison categorical accuracy declined from 17/21 for the incumbent to 12/21 for expanded 16K, alongside improved envelope attack, oscillator-level and RMS-relative diagnostics and slightly worse envelope decay. The two training arms differ in population and minibatch membership; all initial features were present, but replay ordering differs. The shared 205-state audio and feature contents matched exactly in this run, though they were independently rendered for the expanded arm. The score is a spectral distance rather than a calibrated perceptual measure. It does not qualify a release, establish broad generalization, or use the final test split.

## Reproduction

The paths below are the recorded run paths. To reproduce the work, use fresh empty output directories; the trainer does not overwrite existing runs. Run from the repository root with the pinned local Python environment. The first command performs the full 404-render warmup before update one; resume into distinct output directories for the later checkpoints. The commands use the ignored local `tmp/run_training_native.py` launcher, which adds `research/training`, `research/data_generation`, and the pinned local `research/data_generation/outputs/smoke-deps` native runtime packages to `sys.path` before invoking the documented training CLI. A configured environment with those packages on its import path can invoke `research/training/run.py` instead.

```powershell
python tmp/run_training_native.py train --pool research/training/outputs/authored-expanded-20261013/pool.json --output research/training/outputs/state-coverage-expanded-4000-20261013 --mode overfit --steps 16000 --stop-after 4000 --seed 0 --replay-capacity 404 --replay-mib 32 --batch-size 32 --learning-rate 0.001 --sampling-policy state_round_robin --render-budget 480 --checkpoint-every 4000 --cpu-threads 4 --timeout 30
python tmp/run_training_native.py train --pool research/training/outputs/authored-expanded-20261013/pool.json --resume research/training/outputs/state-coverage-expanded-4000-20261013/checkpoint.pt --output research/training/outputs/state-coverage-expanded-8000-20261013 --mode overfit --steps 16000 --stop-after 8000 --seed 0 --replay-capacity 404 --replay-mib 32 --batch-size 32 --learning-rate 0.001 --sampling-policy state_round_robin --render-budget 480 --checkpoint-every 4000 --cpu-threads 4 --timeout 30
python tmp/run_training_native.py train --pool research/training/outputs/authored-expanded-20261013/pool.json --resume research/training/outputs/state-coverage-expanded-8000-20261013/checkpoint.pt --output research/training/outputs/state-coverage-expanded-16000-20261013 --mode overfit --steps 16000 --seed 0 --replay-capacity 404 --replay-mib 32 --batch-size 32 --learning-rate 0.001 --sampling-policy state_round_robin --render-budget 480 --checkpoint-every 4000 --cpu-threads 4 --timeout 30
```

The fixed-case comparison command was:

```powershell
python tmp/run_training_native.py compare --pool research/training/outputs/authored-pilot-v2-20261005/pool.json --checkpoints incumbent=research/training/outputs/two-view-train-20261010/checkpoint.pt one_4000=research/training/outputs/cached-curve-one-step4000-20261011/checkpoint.pt expanded_4000=research/training/outputs/state-coverage-expanded-4000-20261013/checkpoint.pt one_8000=research/training/outputs/cached-curve-one-step8000-20261011/checkpoint.pt expanded_8000=research/training/outputs/state-coverage-expanded-8000-20261013/checkpoint.pt one_16000=research/training/outputs/cached-curve-one-step16000-20261011/checkpoint.pt expanded_16000=research/training/outputs/state-coverage-expanded-16000-20261013/checkpoint.pt --checkpoint-pools expanded_4000=research/training/outputs/authored-expanded-20261013/pool.json expanded_8000=research/training/outputs/authored-expanded-20261013/pool.json expanded_16000=research/training/outputs/authored-expanded-20261013/pool.json --retrieval-checkpoint research/training/outputs/full-pilot-step4000-20261009/checkpoint.pt --output research/training/outputs/state-coverage-validation-20261013 --count 21 --timeout 30
```

The launcher adds the same import paths described above. This command binds expanded checkpoints to their training pool while evaluation and retrieval stay pinned to the parent pool. The 21-case, 462-render summary is linked above.

The planned follow-up is complete: see the [two-seed fixed-case replication report](STATE_COVERAGE_REPLICATION_RESULTS.md). Its paired results vary by learning seed, so the original seed-0 selection remains the current candidate and the replication candidates are not promoted. The 64 test states remain reserved. Broader envelope/level improvement and renderer/platform qualification remain separate work.
