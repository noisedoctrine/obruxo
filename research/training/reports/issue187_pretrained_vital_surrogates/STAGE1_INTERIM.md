# Issue 187 — Stage 1 results

On the 129-preset retrieval panel, the carried-forward CIR and CLAP representations substantially outperform the three tested audio encoders; SAME-L is the strongest of those three, but it does not beat the existing baselines. The pipeline selected SAME-L for Stage 2 because its codec passed the prespecified reconstruction/gradient readiness gate and it has a compatible pretrained generator, not because retrieval established it as the best audio representation.

Recall@1 is the percentage of queries whose exact target preset ranks first in a 129-candidate gallery. Each condition contains 6,450 queries from 129 preset families × 50 distinct MIDI performances. Those repeated queries are clustered, so the intervals below use a two-way bootstrap over preset family and MIDI performance (2,000 replicates); the crossed-cluster 95% intervals quantify uncertainty across this panel, not across all possible instruments or MIDI.

## Retrieval results

| Encoder | Common gallery Recall@1 (95% CI) | Mixed gallery Recall@1 (95% CI) |
|---|---:|---:|
| CIR — carried-forward baseline | 81.0% [74.7, 86.6] | 80.9% [77.1, 84.4] |
| CLAP — carried-forward baseline | 59.4% [50.8, 67.7] | 55.0% [50.2, 59.8] |
| Vital JEPA V7 — carried-forward baseline | 29.1% [23.0, 35.6] | 29.5% [25.2, 33.8] |
| Stable Audio Open Small | 39.1% [31.7, 47.0] | 39.0% [33.7, 44.3] |
| Stable Audio SAME-S | 40.4% [32.9, 48.3] | 40.6% [35.1, 45.9] |
| Stable Audio SAME-L | 41.7% [33.9, 49.6] | 42.1% [36.6, 47.6] |

SAME-L exceeds Open Small by 2.59 points in the common gallery (paired 95% CI −0.79 to +6.09) and 3.13 points in the mixed gallery (+0.25 to +6.17). SAME-L exceeds SAME-S by 1.24 points (common: +0.40 to +2.23; mixed: +0.71 to +2.51). The small SAME-L/Open Small common-gallery difference remains uncertain, and even its mixed-gallery interval is a narrow panel-specific advantage. CIR and CLAP remain clearly stronger on both conditions. These results support carrying CIR/CLAP forward as retrieval references; they do not establish a universal ranking or downstream audio-to-control quality.

## Codec reconstruction and Stage 1 gate

Codec checks used 256 clips from 64 patch roots and 16 MIDI performances. MR-STFT and spectral convergence are reconstruction errors (lower is better), not retrieval or control-recovery metrics.

| Codec | MR-STFT | Spectral convergence | Spectral loss |
|---|---:|---:|---:|
| Stable Audio Open Small | 1.055 | 0.2233 | 0.2014 |
| Stable Audio SAME-S | 2.994 | 0.1837 | 0.2621 |
| Stable Audio SAME-L | 2.224 | 0.1262 | 0.2041 |
| SAME-L embedded generator codec | 2.392 | 0.1292 | 0.2104 |

The operational Stage 1 eligibility gate required spectral loss below the matched 8,000-update direct baseline (0.5552), finite MR-STFT, and a finite nonzero decoder gradient. The compatible generator codecs passed that gate; SAME-L had lower spectral loss than SAME-S, so the runner selected `same-l` and used the generator’s embedded codec revision `d4d3579395a8ac0df4ab13b5b1d7138ce7290f81`. Open Small had lower standalone spectral loss, but no compatible generator was registered for it. This is a compatibility/readiness choice, not a retrieval-quality claim. Stage 2 did not produce a trained adapter result.

## DAC disposition and scope

There was no standalone DAC retrieval encoder in this run. DAC appeared only as a codec compatibility probe (decode/gradient checking); it was not a carried-forward retrieval baseline, candidate training trunk, or selected model. The carried-forward representations were CIR, CLAP, and Vital JEPA V7. This report contains no predicted or edited Vital controls, so the control scope is zero.

## Execution and reproducibility

The existing worker completed Stage 1 and stopped during Stage 2 decoder-latent caching at 1,536/2,174 items, leaving 638 cache items unprocessed. `expanded-0514:original` failed the strict native-render receipt comparison. The original and retry records have the same request ID, renderer ID, engine fingerprint, Vital preset SHA, duration (180,075 frames), and peak; waveform SHA-256 differs (`963b69ae…` versus `d0cb5d26…`), while RMS differs by only 2.45e−13. The original receipt describes Vital determinism as numeric-tolerance only, not bitwise. However, the native readback path also differs (11 verified loaded scalars through `cached_basic_template` versus 772 through `full_state_json`), so the precise cause is not isolated. The original waveform is unavailable in the source pair artifacts, which contain measurements and targets but no audio. Therefore the blocked latent cannot be regenerated compatibly from current artifacts. The receipt guard is correct: do not bypass it or rerender the full corpus. Preserve the 1,536 valid cached latents; recovery requires locating the exact original audio or receipt-compatible source, then continuing the remaining cache work.

Existing Stage 1 per-episode results and the Issue 167 v7 baseline were reused; no retrieval or codec evaluation was repeated to prepare this report. The machine-readable metrics and bootstrap intervals are in [STAGE1_INTERIM.json](STAGE1_INTERIM.json). Stage 2 formal training, Stage 3 inversion, and control recovery remain untested.
