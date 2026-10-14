# First audio-to-patch trainer

The trainer is a bounded, supervised bootstrap for the [pilot vocabulary](ONLINE_PILOT.md), with nine continuous controls and two categorical heads. It accepts audio only, combines a stereo spectral CNN with frozen Basic Pitch dense features, and exports a complete `.vital` patch. It is a pipeline proof, not the finished broad synthesizer model. The completed [205-versus-404-state discovery](reports/STATE_COVERAGE_RESULTS.md) is followed by a fixed-case [two-seed replication](reports/STATE_COVERAGE_REPLICATION_RESULTS.md). Replication deltas vary by seed (+0.0026 and -0.0405) and both state-bootstrap intervals include zero; no replication candidate is promoted, and the seed-0 selected bundle remains current. The [cached-feature report](reports/CACHED_CURVE_RESULTS.md) and [two-view report](reports/TWO_VIEW_RESULTS.md) remain historical controls. The [full-pilot report](reports/FULL_PILOT_RESULTS.md) records the earlier learning curve, and the [first-trainer report](reports/FIRST_TRAINER_RESULTS.md) preserves the tiny-overfit proof.

## Environment and commands

The authored training contract and native factory pin historical renderer `state-verified-v2`, including its original header adapter, scalar tolerance and provenance. The general data-generation default is `state-verified-v3`; changing that default does not migrate training checkpoints. The [renderer documentation](../data_generation/README.md) describes both profiles and the exact retained-training replay qualification.

GitHub Actions remains disabled for this repository by owner choice. Run the checks locally; workflow files describe available CI commands but are not evidence of a successful hosted run. With the native dependencies available on `PYTHONPATH`, enable the official-plugin boundary sweep with `OBRUXO_VALIDATE_NATIVE=1` and run:

```sh
python -m pytest research/training/tests research/data_generation/tests/unit research/data_generation/tests/integration research/modelling/simple_training/test_pipeline.py research/modelling/simple_training/test_preset_contract.py -q -rs
```

Use the existing Python 3.12 native rendering environment and its installed PyTorch. Training was measured on the local CPU with PyTorch 2.12.1+xpu; no environment replacement is needed. For a separate CPU environment, install `requirements-trainer.txt` and a CPU PyTorch wheel. CI pins `torch==2.12.1` from the [official CPU wheel index](https://download.pytorch.org/whl/cpu/torch/). Native synthesis additionally requires the reviewed plugin and the [renderer dependencies](../data_generation/README.md#setup). No network access is needed once dependencies, source code and the existing Basic Pitch checkpoint are available.

Run these from the repository root; each output directory must be new:

```sh
python research/training/run.py pilot --output research/training/outputs/pilot-v2 --count 256
python research/training/run.py train --pool research/training/outputs/pilot-v2/pool.json --output research/training/outputs/overfit-part1 --steps 800 --checkpoint-every 100 --stop-after 400
python research/training/run.py train --pool research/training/outputs/pilot-v2/pool.json --output research/training/outputs/overfit-complete --steps 800 --checkpoint-every 100 --resume research/training/outputs/overfit-part1/checkpoint.pt
python research/training/run.py evaluate --pool research/training/outputs/pilot-v2/pool.json --checkpoint research/training/outputs/overfit-complete/checkpoint.pt --output research/training/outputs/validation --count 4
python research/training/run.py infer --model research/training/outputs/overfit-complete/model.pt --audio input-stereo-44100.wav --output research/training/outputs/predicted.vital
```

`infer` accepts 44.1 kHz stereo WAV, from 2,048 samples through 16 seconds; integer PCM is converted to float32. It does not consume MIDI or a source preset. No native plugin is required for prediction/export. The limited codec fixes all unsupported controls and assets to its declared template. The output file must not already exist.

For fresh audio interleaved with updates:

```sh
python research/training/run.py train --pool research/training/outputs/pilot-v2/pool.json --output research/training/outputs/online --mode online --steps 400 --replay-capacity 32 --batch-size 16 --render-budget 512
```

The online mode warms up with `batch_size` accepted examples, then by default adds one fresh accepted render before each subsequent update and evicts the oldest feature example at capacity. Failed/rejected renders consume the attempt budget and remain in the log. Overfit mode fills the bounded replay once, closes the renderer and reuses those exact features. Neither mode creates a WAV training corpus. Replay defaults to 32 examples and an 8 MiB feature limit; prefetch is one clip, at most 32 MiB, with one isolated native worker. The pilot enforces maximum replay counts/bytes and a total render-attempt budget.

## Full authored-split learning curve

`--sampling-policy state_round_robin` visits every training state once per epoch and rotates its reference performance on successive epochs. Four epochs cover all four reference performances per state. The default `pairs` schedule and existing default checkpoint identities remain compatible. A QA rejection remains a rejected attempt; always check actual distinct-state coverage rather than assuming accepted-count equals coverage.

`--learning-seed-override N` seeds model initialization, the global learning RNGs and minibatch draws with `N`; the recipe sampler continues to use `--seed`. Omitting the override preserves legacy run contracts, while changing an explicit override is rejected on resume.

In online mode, `--updates-per-render N` schedules a refresh every N optimizer updates after warmup. `--fresh-examples N` closes the renderer after N accepted examples and continues optimization from the bounded feature replay. Both settings are part of the recovery contract, including an interrupted refresh before its optimizer update.

The [declared experiment](reports/data/full_pilot_plan.json) uses all 205 training states, one reference view each, with checkpoints at 1,000, 2,000 and 4,000 updates:

```sh
python research/training/run.py train \
  --pool research/training/outputs/pilot-v2/pool.json \
  --output research/training/outputs/full-step1000 --mode online --steps 4000 \
  --replay-capacity 205 --replay-mib 16 --batch-size 32 --learning-rate 0.001 \
  --sampling-policy state_round_robin --updates-per-render 4 --fresh-examples 205 \
  --render-budget 256 --checkpoint-every 1000 --cpu-threads 4 --timeout 30 --stop-after 1000
```

Repeat the same configuration into a new `full-step2000` output with `--resume research/training/outputs/full-step1000/checkpoint.pt --stop-after 2000`. Then resume that checkpoint into `full-step4000`, omitting `--stop-after`. Keep `--steps 4000` in every segment. Resumed segments reuse the existing features after the fresh-example target is met.

```sh
python research/training/run.py evaluate-curve \
  --pool research/training/outputs/pilot-v2/pool.json \
  --checkpoints research/training/outputs/full-step1000/checkpoint.pt \
    research/training/outputs/full-step2000/checkpoint.pt \
    research/training/outputs/full-step4000/checkpoint.pt \
  --output research/training/outputs/full-validation --count 21 --timeout 30
```

The curve evaluator requires increasing steps, matching contracts/renderers and identical retained retrieval features, targets and audio identities. It renders each validation source and each baseline once per role, then compares every checkpoint against those shared buffers. Three checkpoints and 21 states require at most 294 renders. Audio stays in memory for one case; no audio archive is written.

`plan.json` fixes checkpoint hashes, selected recipes, retrieval identity and metric before evaluation. Selection uses the lowest mean of reference and transfer MR-STFT distance, equally weighted per state, and requires every planned state to be scorable in both roles. Ties choose the earlier step. The selected checkpoint becomes `selected-model.pt`; failed cases cannot disappear from denominators. Paired bootstrap intervals compare each checkpoint with retrieval over states, using 2,000 resamples and seed 0. These are descriptive validation intervals, unadjusted for checkpoint selection. This selection does not qualify a release or access the final test split.

For an interrupted curve, repeat the same command with `--resume` and the same output directory. `progress.json` commits complete cases atomically, including hashes of their results, prediction files and render-log prefix. Resume verifies the plan and these hashes, preserves complete failures, and restarts the entire unfinished case so every checkpoint still shares the same reference buffers. Uncommitted renders stay in the log and increase actual attempts beyond the uninterrupted budget. Recorded elapsed time excludes lost work after the last saved boundary; summed logged render wall time is reported separately. A malformed log or changed artifact fails recovery rather than silently skipping data.

## Reusing features for a controlled two-performance run

`--seed-replay CHECKPOINT` starts a new training run with verified features from an earlier checkpoint. It resets model weights, optimizer state, RNG state and feature-use counts. The source must retain a complete accepted recipe prefix, with matching pool, sampler, frozen frontend and stream contracts. Evicted records, QA holes or incompatible features are rejected. The source checkpoint hash, recipe cursor and renderer identity become part of the new run's recovery contract.

The [two-view experiment plan](reports/data/two_view_plan.json) reuses the first 205 training views, then renders one additional reference performance for each state during optimization. The matched control starts with the same features and initialization, but adds no views. Both runs use 4,000 updates, batch size 32, learning rate 0.001 and the same replay limits. This control accounts for making all 205 initial features available before the first update; the original full-pilot run streamed those views in gradually.

```sh
python research/training/run.py train \
  --pool research/training/outputs/pilot-v2/pool.json \
  --seed-replay research/training/outputs/full-step4000/checkpoint.pt \
  --output research/training/outputs/two-view --mode online --steps 4000 --seed 0 \
  --replay-capacity 410 --replay-mib 32 --batch-size 32 --learning-rate 0.001 \
  --sampling-policy state_round_robin --updates-per-render 4 --fresh-examples 410 \
  --render-budget 480 --checkpoint-every 1000 --cpu-threads 4 --timeout 30
```

For the control, change only the output directory to `single-view-control` and `--fresh-examples` to `205`. The accepted-example target and recipe budget include the seeded prefix: a target of 410 requires 205 additional accepted views, and a budget of 480 permits at most 275 additional attempts. The summary separates seeded features from newly accepted renders and feature-extraction time. Replay is capped at 512 examples and 256 MiB; these runs use a 410-example/32 MiB limit and retain about 15.12 MiB of features at full coverage. No seed audio is re-rendered or archived.

Resume a seeded run with its own `--resume CHECKPOINT` and unchanged configuration, omitting `--seed-replay`. The recovery checkpoint already embeds the seed provenance and retained features.

Use named comparison when training configurations or replay populations differ:

```sh
python research/training/run.py compare \
  --pool research/training/outputs/pilot-v2/pool.json \
  --checkpoints previous_selected=research/training/outputs/full-step4000/checkpoint.pt \
    single_view_control=research/training/outputs/single-view-control/checkpoint.pt \
    two_view=research/training/outputs/two-view/checkpoint.pt \
  --retrieval-checkpoint research/training/outputs/full-step4000/checkpoint.pt \
  --output research/training/outputs/two-view-validation --count 21 --timeout 30
```

`compare` independently validates each checkpoint and requires matching inference contracts and native renderer identity. The explicitly pinned retrieval checkpoint supplies the same 205 training records for every candidate, even when a candidate was trained with 410. Source audio and all baselines are shared within each case. Selection uses the same complete-case primary metric as the learning curve; input order breaks ties. The plan records all checkpoint hashes and the retrieval source, and `--resume` uses the same checked recovery protocol. The selected candidate is exported as `selected-model.pt`.

The [measured run](reports/TWO_VIEW_RESULTS.md) selected the two-view model: 8.5% lower primary validation distance than the matched control and 5.2% lower than the previous selected model. The paired intervals against both models include zero. Unison accuracy improves, while envelope and signal-level gaps remain. Both new training arms plus evaluation took 22.92 recorded CPU minutes, excluding setup and artifact verification.

## Longer learning curves from cached features

The [cached-curve plan](reports/data/cached_curve_plan.json) fixes 4,000, 8,000 and 16,000-update checkpoints for both 205-feature and 410-feature arms. Both start with all their features available and reset learning state. This makes the within-arm curve a comparison of optimization duration on a fixed population. The previous selected two-view model, which introduced its second views during training, remains a separate comparator.

Start an explicit new 16,000-update run from features. A 4,000-update checkpoint's original training contract cannot be silently extended with `--resume`:

```sh
python research/training/run.py train \
  --pool research/training/outputs/pilot-v2/pool.json \
  --seed-replay research/training/outputs/two-view/checkpoint.pt \
  --output research/training/outputs/cached-two-4000 --mode online --steps 16000 --seed 0 \
  --replay-capacity 410 --replay-mib 32 --batch-size 32 --learning-rate 0.001 \
  --sampling-policy state_round_robin --updates-per-render 4 --fresh-examples 410 \
  --render-budget 480 --checkpoint-every 4000 --cpu-threads 4 --timeout 30 --stop-after 4000
```

Resume that run into a new `cached-two-8000` output with `--resume .../cached-two-4000/checkpoint.pt --stop-after 8000`, omitting `--seed-replay` and retaining `--steps 16000` and the other settings. Resume again into `cached-two-16000`, omitting `--stop-after`. For the matched single-view arm, use `full-step4000/checkpoint.pt` as the feature seed, set `--fresh-examples 205`, and use `cached-one-*` output directories. No new render is needed because each arm's seed already meets its accepted-example target.

Pass all six checkpoints and the incumbent to `compare`, in the declared order `incumbent`, `one_4000`, `two_4000`, `one_8000`, `two_8000`, `one_16000`, `two_16000`. Keep `--retrieval-checkpoint` pinned to the original `full-step4000/checkpoint.pt`. Seven candidates and 21 states require 462 uninterrupted native evaluation renders. Neither training features nor evaluation state selection change in response to partial results.

The [completed curve](reports/CACHED_CURVE_RESULTS.md) retains the incumbent at primary distance 0.9870. One-performance scores fall from 1.0786 to 0.9906; two-performance scores fall from 1.2592 to 0.9937, with little further improvement after 8,000 updates. The longer two-performance checkpoints improve over their cached 4,000-update control, but neither arm beats the incumbent. These are single-seed validation results. All 462 evaluation renders and 147 predicted-preset checks pass; the selected export is byte-identical to the previous bundle. Training segments take 4.94 CPU minutes with zero new synthesis, evaluation 18.94 minutes, and sampled training memory peaks at about 716 MiB.

## Comparing expanded training pools

`expand-pilot --pool PARENT --count N --output NEW_DIRECTORY` creates a larger authored population using the parent's seed and vocabulary. It verifies the complete ordered parent prefix, target values and every frozen split assignment, then writes `pool.json`, `summary.json` and `expansion.json`. The [verified 512-state pool](reports/POOL_EXPANSION_READINESS.md) has 404 train, 44 validation and 64 test states. All original states keep their assignments, and the additional held-out states remain excluded from training.

Start fresh training with the expanded pool. Cross-pool resume and feature seeding are intentionally rejected because sampler/recipe identities differ; same-pool recovery and feature reuse remain supported. A 404-example replay holds one reference view per expanded training state in about 14.90 MiB. The 512-example cap cannot hold two views of every expanded state. Audio is still rendered on demand and discarded after feature extraction.

For comparison, keep `--pool` set to the original evaluation/retrieval pool, and supply `--checkpoint-pools NAME=PATH` for each expanded candidate. Its checkpoint and replay are validated against that declared training pool; the pool must prove a strict extension preserving all original splits. The existing inference and renderer compatibility checks still apply. See the [complete command](reports/POOL_EXPANSION_READINESS.md#fixed-validation-across-training-pools).

New comparison plans use `pilot_real_render_comparison_v2`, pinning training-pool bindings, candidate tie order and validation reference/transfer pairs. Changing them invalidates resume. V1 comparison plans need their earlier implementation for recovery or a new output directory; completed historical reports and bundles remain usable. The selected bundle records the winning model's training pool, while the report separately identifies the evaluation pool.

### Evaluating every newly added validation state

Add `--validation-pool EXPANDED_POOL` to `compare` to evaluate only states added to the validation split by a verified prefix expansion of `--pool`. `--count` must equal the entire added validation population; partial selection is rejected. The original parent states and every test state remain excluded. Each checkpoint still loads under its declared training pool, and the retrieval bank stays pinned to `--pool` and `--retrieval-checkpoint`.

For the frozen 256-to-512-state expansion, use `--validation-pool research/training/outputs/authored-expanded-20261013/pool.json --count 23` with the existing checkpoint and retrieval arguments. The plan records the expansion proof, excluded parent states and exact reference/transfer pairs. Resume rejects any change to this selection. Seven candidates require 506 native renders, with no optimizer steps or audio archive. Freeze candidates and analysis rules before rendering; automatic best-checkpoint export alone is not a promotion decision.

## Model and loss

The float32 CPU frontend preserves stereo and uses a 2,048-point STFT with a 441-sample hop, 96 unit-area mel bands through 22.05 kHz, fixed `log1p` scaling and 32 time bins. The separate Basic Pitch path uses its native mono/22.05 kHz preparation and dense note/onset/contour outputs, pooled to eight time bins. Audio duration is included. No normalization statistics are fitted from held-out data.

A small two-layer CNN and pitch adapter feed a 128-wide fusion MLP: 355,806 trainable parameters. Basic Pitch parameters and running buffers are frozen and fingerprinted. Training uses AdamW, float32, gradient clipping and finite-gradient/weight checks. The loss is masked normalized-control MSE plus two categorical cross-entropies. Inactive detune/filter controls receive no continuous supervision. Perceptual credit assignment, stochastic hypotheses and broader synthesis controls remain later work.

The tiny overfit gate requires active-control MAE below 0.05, 100% categorical accuracy and total loss below 5% of its initial value. A completed overfit command returns status 2 if this gate fails. This is a learning-loop gate, not a release-quality threshold. Default training is 400 steps; the measured 32-example proof used 800.

## Recovery and artifacts

`checkpoint.pt` is saved atomically and round-trip checked with weights-only loading. `checkpoint.previous.pt` retains the preceding recovery point. A checkpoint contains model, optimizer, global step, RNG states, batch sampler RNG, frozen weights, bounded replay features/targets/audio hashes, acknowledged render cursor, counters and the pending-render phase. Scheduler and precision scaler are explicitly absent in this constant-rate float32 trainer. Per-render metadata are logged before cursor acknowledgement; the next checkpoint persists that acknowledgement and its matching features together at an optimizer boundary. Producer-queued audio is never considered consumed.

Resume requires identical pool, codec, renderer, frontend, model, stream and training configuration, including the original total `--steps`. `--stop-after` limits one segment without changing that contract. An interruption between ingestion and the optimizer does not ingest a second replacement view. Exact numerical recovery is verified on the same CPU/runtime with retained features. Newly rendered Vital audio is only recipe-equivalent; cross-device or cross-version bit equivalence is not promised. `--device cpu`, `--device cuda` and `--device xpu` are explicit choices with no fallback. Only CPU training has been qualified in this increment.

Completion also produces `model.pt`, embedding the trained weights, frozen Basic Pitch weights, frontend description and codec contracts for offline inference within this repository. Keep the existing Basic Pitch licensing notices with redistributed artifacts. Completion means the requested optimizer steps finished; it does not promote a model to production. `summary.json` records losses, memory, timing, gradients, diversity counts and overfit status. All artifacts stay under ignored `outputs/` unless explicitly exported.

## Evaluation and compute limits

`evaluate` selects fixed unique validation states, predicts one patch from one reference audio clip, and renders that same prediction under reference and reserved transfer performances. Known MIDI is used only to generate/compare evaluation audio. Comparators are the fixed pilot template, nearest neighbor from the checkpoint's **training replay only**, and source repeats. The metric is stereo multi-resolution STFT distance, not a calibrated perceptual score. Reports retain planned, scorable and failed denominators. `--split train` inspects retained training examples; final test states are intentionally unavailable in this development CLI.

The corrected 32-example proof needed roughly 78 seconds across its two run segments and under 1 GB sampled process-tree RSS. CPU optimizer work took about six seconds; native rendering dominates. The bounded online smoke achieved about 0.43 accepted fresh clips/second. Colab is unnecessary for this stage. A faster GPU alone will not remove the renderer bottleneck. Linux/Colab needs its own reviewed Vital binary, asset/state fidelity checks and interrupted-session qualification before native training there. Pipeline/checkpoint portability does not imply the Windows plugin runs on Linux.

The completed [discovery](reports/STATE_COVERAGE_RESULTS.md) and [two-seed replication](reports/STATE_COVERAGE_REPLICATION_RESULTS.md) use the original 21 validation pairs and 205-record retrieval comparator, with all 64 expanded-pool test states reserved. Replication expanded-minus-205 deltas are +0.0026 and -0.0405; both paired state-bootstrap intervals include zero. The tiny-overfit continuous MAE gate remains a training-loop diagnostic (seed-0 expanded miss: 0.00253). No replication candidate is promoted, and the seed-0 selected inference bundle remains current. Reproduction commands and complete diagnostics are in the replication report.
