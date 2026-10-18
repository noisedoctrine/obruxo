# Sound prediction before audio-based inverse training

This documents the historical alpha helper, which [failed native qualification](reports/SOUND_PROXY_RESULTS.md). The active [training plan](../modelling/TRAINING_PLAN.md#sound-loss-training-path) and [basic-stage protocol](STAGED_RECONSTRUCTION.md) now require a new small-scope spectral helper before inverse integration. The old artifact remains unqualified. Its implementation and original experiment protocol follow.

The inverse model needs a way to relate changing its predicted patch to changing the sound. Vital cannot supply those gradients. This increment built a separate small forward model and checked its advice against real Vital renders before allowing it to guide inverse training.

The helper takes canonical patch controls, hard categorical choices, and the actual notes, velocities and timing. It predicts fixed multiscale spectral, stereo envelope and side-energy measurements. It receives no patch IDs, performance IDs, audio encoder states or validation labels as network inputs. The inverse model remains audio-only. Known notes belong to the training/evaluation forward path.

`SoundFeatures` uses three FFT sizes (512/2048/8192), 24 unit-area mel bands per channel, eight spectral time bins, 32 stereo RMS envelope bins and 32 side RMS bins: 1,248 float32 values. The time axis is fixed at 16 seconds; shorter audio is zero padded, preserving absolute note and release timing. Gain is retained. Low-frequency bands without FFT samples remain explicitly zero. Measurements have no fitted statistics. Training-only coordinate standard deviations (floor 0.05) scale the regression error; the five measurement groups receive equal weight.

The helper is a 256/256-unit MLP with positive outputs. Inactive patch controls are canonicalized and masked, and their input gradients are zero. Categories stay hard and supervised; this increment does not pretend that blended categories are playable patches. `freeze()` fixes parameters and clears their gradients while retaining derivatives through active continuous inputs.

## Run the fixed feasibility check

From the repository root, with the qualified data-generation environment:

```sh
python research/training/run.py proxy-screen \
  --pool /path/to/authored-alpha/pool.json \
  --output research/training/outputs/sound-proxy-screen
```

The output directory must be new. An optional `--plugin-path` selects the pinned installed Vital binary. The default plan freezes source hashes, runtime, pool, model, measurements, training recipes, validation cases, baselines and gates **before rendering**:

- 64 training states, chosen in family round-robin order, with all four reference performances: 256 planned native training attempts. Only accepted audio contributes measurements. At least 80% of attempts and every planned synthesis family must survive QA. Failures are recorded without replacement.
- 4,000 CPU updates, batch size 32, learning rate 0.001, four CPU threads and an 8 MiB retained-measurement limit. Audio is rendered on demand during ingestion, measured and discarded. Several learning updates reuse each measurement; there is no audio archive.
- Ten validation states, balanced across the five alpha families. Each uses one reference and one reserved performance. The final test split stays reserved.
- Six planned renders per performance: source, small and large continuous changes, one categorical change, one helper-guided update from the large-change patch, and a repeated source. Total: 120 planned validation attempts. A failed source makes its remaining comparisons unscored, not successful.
- Three train-only forward baselines: global mean, nearest patch/performance, and interpolation of the five nearest records. Ties in retrieval use original training-record order.

The guided update is a qualification probe, not a shipped inference-refinement feature. It uses the target recording's measurements and evaluator notes, without source patch labels in its update. It takes 16 Adam steps on continuous controls, keeps categories fixed and stays within 0.15 normalized units of its starting controls. Real Vital must verify any proposed improvement.

## Qualification and artifacts

The initial gates require complete accepted validation renders, reference regression error at least 10% below each baseline, transfer regression error no worse than each baseline, at least 60% native candidate-ranking agreement on at least half the planned comparisons, and native guided improvements beyond repeat noise on at least 60% of cases with a negative mean error change. These are conservative feasibility gates, not calibrated release-quality thresholds. Repeated performances of one patch are not independent validation samples. Results are descriptive and do not promote a final model.

Native multiresolution STFT error, repeat-render noise and compact measurement error are reported separately. Better helper regression alone cannot pass. A helper that recommends changes which sound worse fails qualification even if its own predicted error improves.

Files include `plan.json`, `renders.jsonl`, `checkpoint.pt`, complete case JSON, `summary.json` and `proxy.pt`. The checkpoint contains only bounded measurements, recipes/outcomes, learning state and completed comparisons. The export embeds the codec and model state for offline loading. `load_proxy()` defaults to refusing an export that failed native qualification; diagnostic loading requires explicit `require_qualified=False`. No inverse-training connection is enabled by this increment.

## Recovery and compute

Repeat the same command with `--resume` and the original arguments. Changed source, runtime, pool or plan is rejected. Collection recovers after each acknowledged native outcome; optimizer state and minibatch RNG recover at configured training checkpoints. Completed evaluation cases are reused. An interrupted, incomplete case regenerates its transient audio; every extra attempt stays in the append-only log. Twelve recovery attempts are reserved in the frozen maximum of 388. Exhausting that cap stops the run; recovery never silently grants more compute.

Feature hashes, training recipes, audio hashes and committed QA outcomes are checked on recovery. Rejected audio never becomes a regression target. Native renderer identity must stay fixed. Output checkpoints retain a previous recovery point.

Local CPU is the default. The model and retained measurements are portable to a compatible free Colab optimizer environment, but Linux/Colab native Vital rendering remains unqualified. This increment assumes no paid compute or cloud storage.
