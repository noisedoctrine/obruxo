# Issue 187 — Stage 1: pretrained audio representation quality

## Research question and findings

Can a pretrained codec preserve the sound of a Vital patch well enough to build a differentiable surrogate? On the 256-clip panel, Open Small leads the primary log-mel reconstruction metric and CIR embedding similarity among the four Stable Audio codecs. Medium's embedded SAME-L leads stereo phase coherence and better preserves level and sample-by-sample waveform structure. The metrics support different choices depending on which errors matter.

| Codec | MEL-log1p ↓ (95% CI) | STFT-log1p ↓ | CIR cosine ↑ | SI-SDR ↑ (dB) | CCPC ↑ (%) | Legacy MR-STFT ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Open Small | **0.002237 [0.001622, 0.002954]** | **0.002657** | **0.775** | 7.62 | 94.04 | **1.055** |
| SAME-S | 0.003515 [0.002544, 0.004657] | 0.004370 | 0.446 | 10.54 | 95.97 | 2.994 |
| SAME-L standalone | 0.002300 [0.001673, 0.003021] | 0.003128 | 0.586 | **14.98** | 97.26 | 2.224 |
| Medium embedded SAME-L | 0.002504 [0.001816, 0.003311] | 0.003269 | 0.586 | 14.44 | **97.31** | 2.392 |
| DAC (two independent mono channels) | Pending | — | — | — | — | — |

Each completed cell covers 256 clips from 64 patch roots and 16 distinct MIDI performances. ViSQOL is unavailable on this host. The primary metric measures how closely the reconstructed sound follows the reference energy across mel frequency bands. CIR measures similarity in a frozen timbre representation. SI-SDR measures waveform fidelity after a fitted gain and CCPC compares stereo phase coherence. Retrieval remains a separate endpoint.

| Paired difference, candidate minus Open Small | MEL-log1p ↓ | CIR cosine ↑ | SI-SDR ↑ (dB) | CCPC ↑ (pp) |
|---|---:|---:|---:|---:|
| SAME-L standalone | +0.000063 [−0.000127, +0.000277] | −0.189 | +7.36 [6.45, 8.30] | +3.22 [1.84, 4.91] |
| Medium embedded SAME-L | +0.000267 [+0.000044, +0.000528] | −0.188 [−0.218, −0.157] | +6.82 [5.91, 7.75] | +3.27 [1.88, 4.95] |

MEL-log1p and legacy MR-STFT disagree about which of Open Small and standalone SAME-L wins on 133 of 256 clips. STFT-log1p agrees with MEL-log1p on 196, and CIR cosine agrees on 133. Open Small's mean MEL advantage over standalone SAME-L is uncertain. Its advantage over the actual Medium codec is about 12% and the paired interval excludes zero. These intervals condition on the observed MIDI panel and cannot establish listener preference.

The [per-clip results](STAGE1_RESCORE_CASES.csv) and [paired intervals](STAGE1_RESCORE.json) preserve the different rankings. The following examples use original playback levels and the same reference for every codec. They are for review and have no listener ratings.

| Clip and what to compare | Native | Open Small | SAME-L | Medium embedded |
|---|---|---|---|---|
| 062-0 original. SAME-L has lower MEL error, while legacy MR-STFT prefers Open Small. Compare sustained tone and stereo image. | [Play](listening/mel_favors_same_l-reference.flac) | [Play](listening/mel_favors_same_l-open-small.flac) | [Play](listening/mel_favors_same_l-same-l.flac) | [Play](listening/mel_favors_same_l-same-l-embedded.flac) |
| 061-1 perturbed. Open Small has lower MEL error and its RMS differs by 30.4% versus 1.8% for SAME-L. Compare timbre and overall level separately. | [Play](listening/mel_favors_open_small-reference.flac) | [Play](listening/mel_favors_open_small-open-small.flac) | [Play](listening/mel_favors_open_small-same-l.flac) | [Play](listening/mel_favors_open_small-same-l-embedded.flac) |
| 058-1 perturbed. Typical example from the cases where both metrics disagree in the same direction. SAME-L leads MEL, Open Small leads legacy MR-STFT and CIR, and Medium is between them on MEL. | [Play](listening/mel_and_mrstft_disagree-reference.flac) | [Play](listening/mel_and_mrstft_disagree-open-small.flac) | [Play](listening/mel_and_mrstft_disagree-same-l.flac) | [Play](listening/mel_and_mrstft_disagree-same-l-embedded.flac) |

**Trunk recommendation for this experiment:** qualify the Medium embedded codec for the matched decoder and generator arms. It has gains on waveform fidelity, stereo phase and level, a slightly faster measured decoder, and 1,536 existing exact-revision training latent pairs. Advancing Open Small would require recovering original audio for those 3,072 clips as well, leaving nearly the whole training corpus to replay. Open Small is the strongest measured choice on the primary reconstruction metric. The Medium choice reflects the study constraints and leaves Open Small's acoustic advantage visible.

## 1. How accurately do the codecs reconstruct Vital audio?

Each frozen audio encoder converts a native Vital recording into a sequence of latent vectors (a compressed representation of the sound). Its frozen decoder reconstructs a stereo waveform from those vectors. The reconstructed recording is compared with the original at its original level, with no adapter involved.

### Spectral accuracy

| Error metric (lower is better) | Open Small | SAME-S | SAME-L |
|---|---:|---:|---:|
| MR-STFT | **1.055** | 2.994 | 2.224 |
| Spectral convergence | 0.223 | 0.184 | **0.126** |
| Log-magnitude MAE | **0.179** | 0.340 | 0.282 |
| Shared spectral loss | **0.2014** | 0.2621 | 0.2041 |
| General octave-band energy (dB) | **7.65** | 14.45 | 12.32 |
| MIDI-centered octave energy (dB) | **7.87** | 12.30 | 9.63 |

MR-STFT combines spectral convergence (relative magnitude-spectrum error) and absolute log-magnitude error over FFT sizes 512, 2,048 and 8,192, using the existing waveform evaluation. The shared spectral loss is the equal-weight combination of spectral convergence and log-magnitude MAE at the sampled 2,048-point frames used in #183/#185; it is a separate metric from full-clip MR-STFT.

SAME-L has the lowest spectral convergence error, which gives more weight to strong spectral components. Open Small has the lowest log-magnitude error, which is more sensitive to weaker components, and its MR-STFT is less than half SAME-L's. Their shared spectral losses are nearly identical because the two components trade off.

Octave error compares the amount of energy in frequency bands against the native recording. The MIDI-centered version measures bands relative to the played pitch on monophonic frames. Open Small has the smallest error in both tests. This matters because incorrect octave balance was one of the audible problems that motivated #183 and this research. The aggregate scores do not establish which individual harmonics are wrong.

### Envelope, level and stereo accuracy

| Error metric (lower is better) | Direct MLP (#185, 8k) | Open Small | SAME-S | SAME-L |
|---|---:|---:|---:|---:|
| Attack envelope (dB) | 6.67 | 2.74 | 2.24 | **1.91** |
| Decay envelope (dB) | 5.28 | 1.98 | 1.19 | **1.03** |
| Sustain envelope (dB) | 7.25 | 2.47 | 2.24 | **1.77** |
| Release envelope (dB) | 33.24 | **21.57** | 26.38 | 23.97 |
| Relative RMS level error | — | 13.7% | 8.6% | **7.0%** |
| Stereo side-to-mid ratio error (dB) | — | 1.87 | **1.10** | 1.60 |

SAME-L best preserves the attack, decay and sustain phases and overall level; Open Small best preserves the release. All three codecs have lower envelope errors than the MLP reference. These phases are defined by MIDI timing, not by the exact envelope settings inside Vital. Release remains difficult, and its dB error is sensitive to very quiet frames.

Relative RMS error measures the difference in overall signal level relative to the original. Stereo side-to-mid error measures the difference in the ratio of stereo-difference energy to common-channel energy; it does not establish complete stereo waveform fidelity.

The legacy values here preserve the original aggregation. Confidence intervals in the rescore use paired patch-root resampling over the same clips. The original direct MLP predicts magnitudes from patch and MIDI and is a common spectral reference. It cannot be scored on native waveform metrics.

### Reconstruction cost and latent size

| Metric | Open Small | SAME-S | SAME-L |
|---|---:|---:|---:|
| Mean encoding time (ms/clip) | 400 | **54** | 390 |
| Mean decoding time (ms/clip) | 427 | **52** | 393 |
| Mean latent storage (KiB/clip) | **19.9** | 40.3 | 40.0 |
| Finite, nonzero decoder-input gradient | Pass | Pass | Pass |

SAME-S decodes about eight times faster than Open Small on the measured XPU workload. This could be valuable when repeatedly evaluating a candidate patch during optimization. Open Small stores roughly half as many latent bytes per clip, although storage alone does not establish how difficult those latents will be to predict. The timings are per-clip averages, not matched sampled-frame inference timings for the MLP, and no backward latency or peak-memory comparison is reported here.

All three decoders passed a gradient smoke check. This establishes that gradients can reach their input latents; it does not show that gradients through a trained adapter will lead to better real Vital patches.

## 2. Can the latent representation identify the same patch across MIDI performances?

For example, if the query is a bass patch playing one melody, the test asks whether the representation ranks that same patch playing a different melody above 128 other patches.

The common-MIDI condition plays all 129 comparison patches using the same different performance. The mixed-MIDI condition uses varying comparison performances. Recall@K is the percentage of queries for which the exact target patch appears among the first K results; higher is better.

| Encoder / representation | Common Recall@1 (95% CI) | Common Recall@5 | Common Recall@10 | Mixed Recall@1 (95% CI) | Mixed Recall@5 | Mixed Recall@10 |
|---|---:|---:|---:|---:|---:|---:|
| CIR | **81.0% [74.7, 86.6]** | 93.6% | 96.5% | **80.9% [77.1, 84.4]** | 93.9% | 96.9% |
| CLAP | 59.4% [50.8, 67.7] | 79.9% | 87.4% | 55.0% [50.2, 59.8] | 75.8% | 83.7% |
| SAME-L | 41.7% [33.9, 49.6] | 62.5% | 70.8% | 42.1% [36.6, 47.6] | 61.1% | 69.4% |
| SAME-S | 40.4% [32.9, 48.3] | 61.1% | 70.1% | 40.6% [35.1, 45.9] | 59.8% | 68.5% |
| Open Small | 39.1% [31.7, 47.0] | 58.8% | 66.8% | 39.0% [33.7, 44.3] | 56.7% | 64.8% |
| Vital JEPA V7 (original pooling) | 29.1% [23.0, 35.6] | 54.1% | 67.0% | 29.5% [25.2, 33.8] | 49.9% | 61.6% |

The comparison reuses the Issue 167 baseline, including JEPA V7's original pooling. Issue #169 subsequently found an exploratory 56.5% common Recall@1 for top-seven-token mean pooling on this cohort. See the [representation ranking and its held-out evaluation caveats](../../../modelling/frozen_encoder_retrieval/reports/VITAL_JEPA_V7_ALL_REPRESENTATIONS.md).

The older Issue 152 DAC masked-mean result was 31.4% common Recall@1 (95% CI 24.9–38.7%), 52.2% Recall@5, and 32.9% mixed Recall@1. Its 129-way galleries and 6,450 queries have the same scale as Issue 167. The published episode snapshot differs, so the DAC result is historical context outside the matched table. No retrieval was repeated. [Frozen encoder retrieval report](../../../modelling/frozen_encoder_retrieval/reports/FROZEN_ENCODER_RETRIEVAL_RESULTS.md)

| Paired Recall@1 difference (percentage points) | Common MIDI, estimate [95% CI] | Mixed MIDI, estimate [95% CI] |
|---|---:|---:|
| SAME-L − Open Small | +2.59 [−0.79, +6.09] | +3.13 [+0.25, +6.17] |
| SAME-L − SAME-S | +1.24 [+0.40, +2.23] | +1.57 [+0.71, +2.51] |
| SAME-S − Open Small | +1.35 [−2.06, +4.84] | +1.57 [−1.38, +4.59] |

SAME-L beats SAME-S in both conditions on this panel. Its advantage over Open Small is small and the common-MIDI interval includes zero. CIR and CLAP remain substantially stronger retrieval references. All three new codec point estimates change little between the common and mixed conditions, but this is not a direct test of MIDI invariance.

For these new representations, the runner concatenates the mean and standard deviation of each latent channel over time, normalizes the resulting vector and ranks candidates by cosine similarity. This is a simple fixed pooling test, not an exhaustive comparison of latent readouts. The audio codecs are intended to preserve the entire recording, including pitch, timing and performance. Low patch recall alone is therefore not grounds to reject a decoder for forward synthesis, and high recall does not establish that a representation can be predicted from controls.

## 3. Which candidate should proceed to adapter training?

| Candidate | Reason to consider it | Limitation |
|---|---|---|
| Open Small | Lowest MEL, STFT, and octave errors, strongest CIR match, compact latents | Full training would require 3,072 more original waveforms |
| SAME-L | Lowest attack/decay/sustain and level errors; registered generator | Higher MR-STFT and octave error than Open Small |
| SAME-S | Much faster encode/decode | Highest MR-STFT and shared spectral loss of the three |

The operational gate required shared spectral loss below the #185 direct MLP's 8,000-update value of 0.5552, finite MR-STFT and finite nonzero decoder-input gradients. The compatible generator codecs passed the gate. Among the registered candidates, the runner chose SAME-L over SAME-S and selected the Stable Audio 3 Medium generator's embedded codec (revision `d4d3579395a8ac0df4ab13b5b1d7138ce7290f81`).

| Codec revision used for reconstruction | MR-STFT ↓ | Spectral convergence ↓ | Shared spectral loss ↓ |
|---|---:|---:|---:|
| Standalone SAME-L | 2.224 | 0.1262 | 0.2041 |
| Medium generator's embedded codec | 2.392 | 0.1292 | 0.2104 |

The embedded codec is slightly worse on the reported metrics; do not substitute the standalone codec's other metrics or timings for the embedded revision. The intended Stage 2 comparison must use the same actual decoder in both adapter arms.

The pinned Open Small checkpoint contains both its 64-channel VAE and an RF denoiser that consumes the same latent grid. Its DiT takes 768D prompt tokens and duration conditioning, so an adapter can replace those tokens through the separate Stable Audio Tools wrapper. The missing registration was an implementation gap. The connector is present for future qualification, and this run qualifies Medium only. The model cards put Open Small, SAME and Medium under Stability AI community terms. Deployment and redistribution are outside this local experiment. [Open Small model card](https://huggingface.co/stabilityai/stable-audio-open-small) · [Medium model card](https://huggingface.co/stabilityai/stable-audio-3-medium).

## Execution and reproducibility

- **Control scope:** zero predicted or edited Vital controls. This is a frozen representation benchmark; all source patches and MIDI performances are fixed and no model weights are trained.
- **Reconstruction panel:** 128 original/perturbed pairs (256 clips), 64 patch roots and 16 MIDI performances from the frozen #185 validation data. The octave metric covers 5,409 of 8,192 candidate frames; the MIDI-centered metric covers 2,804 of 2,804 active monophonic candidate frames. Envelope frame counts are 1,114 attack, 1,314 decay, 1,472 sustain and 1,914 release. Open Small uses seeded VAE sampling.
- **Rescore definitions:** 44.1 kHz float32 stereo at native level with sample alignment and centered zero padding. Periodic Hann windows overlap 75%. STFT-log1p averages the absolute difference of raw magnitude `log1p` values over channels, frames and bins, then equally over FFT sizes 128–4,096. MEL-log1p applies the same loss after the paper's `64 / (FFT / 2 + 1)` scaling of 64 unit-height Slaney mel bands at FFT sizes 1,024, 2,048 and 4,096. The paper does not specify the filter convention or padding. CCPC compares left/right phase differences, using a weighted mean resultant length per frame and averaging frames by their weights. Joint geometric cross-spectrum weights give zero-energy bins no influence. FFT sizes are 512, 1,024, 2,048 and 4,096. SI-SDR uses mean-centered channels and a fitted gain per channel, excludes silent channels and bounds singularities to ±100 dB. Phase agreement and SI-SDR are undefined for a silent pair. The local results have no normalized-away gain adjustment for the spectral metrics. [SAME §5.3](https://arxiv.org/html/2605.18613) · [Back to Ear §4.4.1](https://arxiv.org/html/2509.14912v2#S4.SS4.SSS1).
- **CIR and ViSQOL:** locked CIR `official_clip` (768D, checkpoint SHA-256 `b08996e4…`) uses the original retrieval preprocessing: average stereo to mono, resample to 16 kHz, and keep level. CIR cosine is within the same recording and its reconstruction. The official ViSQOL audio mode would resample the stereo mean to 48 kHz. The official binding, CLI and Windows Bazel build tools are absent in this environment. ViSQOL has 0 valid cases. [ViSQOL audio-mode documentation](https://github.com/google/visqol).
- **Rescore provenance:** eight saved waveforms per Stable Audio candidate were reused. The other 248 per candidate used the pinned checkpoint and original deterministic seed assignment. All originals were checked against their native render receipts. Uncertainty is a 5,000-replicate paired bootstrap over the 64 patch roots, using seed 187 and keeping both roles and all clips of a sampled root together. These are validation panel estimates at the fixed observed performances.
- **Retrieval panel:** 6,450 recordings and 6,450 queries per condition, 12,900 episodes in total. These are not independent samples. The new Recall@1 and paired-difference intervals use a two-way bootstrap over preset family and MIDI performance (2,000 replicates, seed 187). Baseline intervals use the existing Issue 167 bootstrap (2,000 replicates, seed 152). The intervals cover variation over the sampled preset/performance dimensions, not training seeds or all possible sounds.
- **DAC:** the matched reconstruction pass uses the pinned 44.1 kHz mono RVQ checkpoint independently for left and right as a two-item batch. It decodes the sum of the quantized code vectors, not the discrete indices or the encoder's raw output. Its joint stereo model and pretrained generator are absent. The previous isolated decode gradient probe followed the continuous quantized embedding through the decoder. [XPU setup and checkpoint revisions](XPU_SETUP.md).
- **Stage 2 audio recovery:** the previous worker stopped at 1,536/2,174 cached pairs on `expanded-0514:original`. Running it on a new engine selected full JSON readback (772 verified scalars) and produced a different waveform hash. Its original receipt records it as job 2 on a reused engine (11 verified scalars through the native parameter getter). Replaying `expanded-0512:original` first reproduced both original hashes and renderer provenance, including `963b69ae…` for the failure. The worker-prefix replay then recovered all 1,019 other missing training waveforms with exact matches and no errors. The other 128 pending pairs already have receipt-verified validation audio. The 1,536 legacy latent files were preserved and remain subject to the selected codec revision and input receipt audit. [Two-job replay evidence](STAGE2_AUDIO_RECOVERY_PROBE.json)

[Machine-readable Stage 1 results and uncertainty](STAGE1_INTERIM.json) · [Issue 167 retrieval baseline](../../../modelling/frozen_encoder_retrieval/reports/issue167_v7_results.json) · [Issue 185 direct MLP baseline](../issue185_direct_flow_mlp/REPORT.md) · [Stage 2 preparation and outstanding integration](STAGE2_PREPARATION.md).

## Decision and remaining questions

Open Small best preserves the measured mel spectrum and CIR timbre embedding. Medium is the one trunk recommended for qualification in the matched adapter experiment because its exact generator and training cache fit the study constraints. This report has no independent confirmation panel or listening ratings. [Machine-readable selection](STAGE1_SELECTION.json)
