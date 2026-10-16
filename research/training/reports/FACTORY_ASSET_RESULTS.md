# Factory asset qualification and learned selection

Completed locally on 2026-10-16 with the pinned Windows Vital/DawDreamer renderer. All 44 factory references qualified, and the expanded codec passed exact native checks. The model trained, exported and restored successfully. Its complete-audio validation gate **failed on clipping**, and its strict tiny-overfit gate remains false. This is implemented asset support with recorded limitations; no model is promoted.

## Factory source fidelity

| Reference family | Qualified | Native renders | Sound comparisons |
|---|---:|---:|---|
| Wavetables | 19/19 | 133/133 accepted | 57 original/converted frame comparisons; 45 exact |
| Samples | 10/10 | 40/40 accepted | 10/10 repeated middle-pitch comparisons exact |
| LFO shapes | 15/15 | 30/30 accepted | 15/15 repeated modulation comparisons exact |

Legacy wavetable imports are explicitly re-exported through Vital, given the reviewed 1.5.5 header, checked for stable structure, and compared at frames 0/128/256 with fixed oscillator and sampler phases. The worst relative RMSE is `0.0000170261` and worst absolute peak error `0.00000244379`, below the predeclared `0.001` and `0.0001` limits. Conversion is therefore audio-equivalent within those limits, not universally byte/audio-exact. Sample imports preserve mono/stereo PCM16 frames and original rates; native sample readback retains the documented four-sample prefix/one-PCM16-step policy.

Qualification took 273.95 seconds and peaked at 416.03 MiB process memory, with no sampling errors and unchanged source hashes. The bank ID is `1f512c5134c259cbbcdb9c4df1d14fe2b40dc0290fcfdb604c740efc54112e40`. The [plan](data/factory_bank_plan.json) pins references, source hashes, renderer, phases, limits and budgets. The [sanitized qualification](data/factory_bank_summary.json) retains per-reference comparisons and request/audio hashes without asset payloads.

## Training and codec integration

| Measurement | Result |
|---|---|
| Output vocabulary | 20 wavetable choices for each of three numbered oscillators; 11 sampler choices |
| Model | 812,982 trainable parameters; audio input only; frozen Basic Pitch |
| Total outputs | 424 continuous values, 410 categorical heads |
| Frozen authored roots | 256: 205 train, 27 validation, 24 reserved test |
| Native codec check | 18/18 accepted; six families; exact source-repeat and decoded audio; all 775 controls equal at float32 |
| Training | 1,000 CPU updates, batch 16, four threads, 64 retained feature examples |
| Training renders | 65 attempts, 64 accepted, one clipping rejection, zero errors |
| Retained features | 2,474,240 bytes; no audio archive |
| Training/collection time | 341.01 seconds, including 12.17 seconds of feature extraction |
| Peak training/process memory | 980.20 MiB, below 8 GiB; no memory sampling errors |
| Export/reload | Model weights and codec identity match |

Loss over the complete retained set fell from `0.448865` to `0.002181`, continuous MAE from `0.062392` to `0.009824`, and categorical accuracy rose from `93.3927%` to `99.9730%`. The strict gate requires 100% categorical accuracy, so it remains false. Aggregate accuracy includes fixed controls and does not establish broad learning.

Training-set active asset choices were predicted correctly in 52/53 oscillator-1 cases, 8/8 oscillator-2 cases, 1/1 oscillator-3 case and 11/11 sampler cases. The retained set covers all 20 oscillator-1 choices but only seven of the eleven sampler choices. Inactive choices are preserved and masked. The sparse third-oscillator coverage and incomplete sample coverage remain curriculum work; these figures do not establish generalization or learning every asset in every slot.

## Native validation failure remains visible

The fixed six-state panel spans oscillator, filter, envelope, voice, effect and sampler families, each with reference and reserved-transfer performance. It planned 60 native renders; 56 were attempted and 54 accepted. One source transfer render clipped at peak `1.53360`, so four dependent renders were skipped. A model reference render also clipped at peak `1.36530`. Both failures occurred on the same oscillator-family case. There were no native state-readback failures. No failed case was replaced, and clipping thresholds were unchanged.

| Method | Reference mean sound distance (scored/planned) | Transfer mean sound distance (scored/planned) |
|---|---:|---:|
| Model | 2.22937 (5/6) | 1.80348 (5/6) |
| Training-only retrieval | 1.78973 (6/6) | 1.78279 (5/6) |
| Fixed template | 16.98311 (6/6) | 17.04369 (5/6) |
| Source repeat | 0 (6/6) | 0 (5/6) |

The means omit unscorable cases and have unequal reference coverage, so they are not a complete paired comparison or evidence of model superiority. This expanded task also differs from the earlier scalar-only task. The pipeline report passes seven of eight checks and fails `native_validation_complete`; neither pipeline completion nor model quality is claimed. Output headroom and the sparse sampler curriculum must be addressed before broad quality qualification. Further small-panel tuning is deferred while missing curve/routing capabilities are implemented.

The [integration plan](data/factory_asset_plan.json), [summary with failures and coverage](data/factory_asset_summary.json) and [exact codec checks](data/factory_asset_codec_summary.json) preserve the result. Training package and runner source hashes match the completed run. The 84-check compatibility command passed. Three additional CLI checks passed, including a focused rerun after correcting a test to expect the existing `SystemExit(1)` failure contract. Ruff and branch whitespace checks passed.

All original scalar lineage/split assignments survive the asset mutations. Final-test metadata is retained for split auditing, but final-test audio was never rendered. Generated banks, pools and models remain private local outputs; no third-party asset payloads or audio are committed. GitHub Actions stays disabled, no paid compute was used, and Linux/Colab remains unqualified.

LFO references are qualified for later work; their shapes, custom warps, seeds and ordered modulation connections are not yet learned outputs. See the [delivery status](../TRAINING_STATUS.md) and [reproduction guide](../FACTORY_ASSETS.md).
