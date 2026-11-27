# Issue 187 — Stage 1: pretrained audio representation quality

The revised reconstruction rescore is in progress. The tables below preserve the original Stage 1 results and the earlier provisional trunk choice.

## Research question and findings

Can a pretrained audio model preserve the sound of a Vital patch accurately enough to serve as the basis of a differentiable surrogate? If it can, we could train a smaller network to translate patch parameters and MIDI into the model's latent representation instead of learning audio synthesis from scratch.

Stage 1 tests two properties without training anything: how accurately each audio encoder and decoder reconstruct an existing recording, and how well the encoder identifies the same patch when the MIDI performance changes.

**Open Small has the lowest multi-resolution spectral reconstruction error and octave error. SAME-L has better level and attack/decay/sustain accuracy, while SAME-S decodes considerably faster.** All three standalone codecs reconstruct the supplied audio with lower shared spectral loss than our existing forward MLP achieves when predicting a spectrum from patch and MIDI. This establishes useful reconstruction potential, not an improvement in end-to-end surrogate performance.

| Pretrained audio codec | MR-STFT ↓ | Shared spectral loss ↓ | Octave error (dB) ↓ | Decode time (ms) ↓ | Cross-MIDI Recall@1 ↑ |
|---|---:|---:|---:|---:|---:|
| Open Small | **1.055** | **0.2014** | **7.65** | 427 | 39.1% |
| SAME-S | 2.994 | 0.2621 | 14.45 | **52** | 40.4% |
| SAME-L | 2.224 | 0.2041 | 12.32 | 393 | **41.7%** |
| Existing direct MLP (#185) | — | 0.5552 | 14.03 | — | — |

Reconstruction uses 256 clips from 64 patch roots and 16 distinct MIDI performances. Retrieval uses 6,450 queries per condition from 129 preset families and 50 performances. These are different panels and measure different properties. The MLP row is a reference from the 8,000-update checkpoint, not a matched encode/decode arm: it receives patch and MIDI rather than the correct audio and predicts magnitudes rather than a waveform.

The runner provisionally selected SAME-L for Stage 2 because it passed the reconstruction and gradient checks and has a registered compatible generator. This is a practical selection, not evidence that SAME-L is the best reconstruction model. Before adapter training, Open Small's exclusion deserves reconsideration.

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

These reconstruction comparisons are descriptive means over the same panel. Paired reconstruction confidence intervals and formal listening ratings are not available in the committed Stage 1 results. In particular, the small shared-loss difference between Open Small and SAME-L is not a demonstrated statistically reliable advantage.

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
| SAME-L | 41.7% [33.9, 49.6] | — | — | 42.1% [36.6, 47.6] | — | — |
| SAME-S | 40.4% [32.9, 48.3] | 61.1% | 70.1% | 40.6% [35.1, 45.9] | 59.8% | 68.5% |
| Open Small | 39.1% [31.7, 47.0] | 58.8% | 66.8% | 39.0% [33.7, 44.3] | 56.7% | 64.8% |
| Vital JEPA V7 (original pooling) | 29.1% [23.0, 35.6] | 54.1% | 67.0% | 29.5% [25.2, 33.8] | 49.9% | 61.6% |

Dashes indicate values not published in the current Stage 1 snapshot, not zero recall. The comparison reuses the Issue 167 baseline, including JEPA V7's original pooling. Issue #169 subsequently found an exploratory 56.5% common Recall@1 for top-seven-token mean pooling on this cohort; see the [representation ranking and its held-out evaluation caveats](../../../modelling/frozen_encoder_retrieval/reports/VITAL_JEPA_V7_ALL_REPRESENTATIONS.md). The new codecs therefore should not be described as outperforming the best tested JEPA V7 representation.

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
| Open Small | Lowest MR-STFT and octave error; compact latents | Slow decoding relative to SAME-S; generator connector not registered |
| SAME-L | Lowest attack/decay/sustain and level errors; registered generator | Higher MR-STFT and octave error than Open Small |
| SAME-S | Much faster encode/decode | Highest MR-STFT and shared spectral loss of the three |

The operational gate required shared spectral loss below the #185 direct MLP's 8,000-update value of 0.5552, finite MR-STFT and finite nonzero decoder-input gradients. The compatible generator codecs passed the gate. Among the registered candidates, the runner chose SAME-L over SAME-S and selected the Stable Audio 3 Medium generator's embedded codec (revision `d4d3579395a8ac0df4ab13b5b1d7138ce7290f81`).

| Codec revision used for reconstruction | MR-STFT ↓ | Spectral convergence ↓ | Shared spectral loss ↓ |
|---|---:|---:|---:|
| Standalone SAME-L | 2.224 | 0.1262 | 0.2041 |
| Medium generator's embedded codec | 2.392 | 0.1292 | 0.2104 |

The embedded codec is slightly worse on the reported metrics; do not substitute the standalone codec's other metrics or timings for the embedded revision. The intended Stage 2 comparison must use the same actual decoder in both adapter arms.

Open Small had lower standalone MR-STFT and shared spectral loss, but the runner registers only Stable Audio 3 Small Music and Medium as generators. The lack of an Open Small connector is an implementation limitation, not a demonstrated incompatibility of the model architecture. Check the feasibility, licensing and cost of that connector before treating SAME-L as the final choice. Open Small is the leading reconstruction-quality candidate from Stage 1; SAME-L is a reasonable practical alternative for a matched decoder/generator comparison.

## Execution and reproducibility

- **Control scope:** zero predicted or edited Vital controls. This is a frozen representation benchmark; all source patches and MIDI performances are fixed and no model weights are trained.
- **Reconstruction panel:** 128 original/perturbed pairs (256 clips), 64 patch roots and 16 MIDI performances from the frozen #185 validation data. The octave metric covers 5,409 of 8,192 candidate frames; the MIDI-centered metric covers 2,804 of 2,804 active monophonic candidate frames. Envelope frame counts are 1,114 attack, 1,314 decay, 1,472 sustain and 1,914 release. Open Small uses seeded VAE sampling.
- **Retrieval panel:** 6,450 recordings and 6,450 queries per condition, 12,900 episodes in total. These are not independent samples. The new Recall@1 and paired-difference intervals use a two-way bootstrap over preset family and MIDI performance (2,000 replicates, seed 187). Baseline intervals use the existing Issue 167 bootstrap (2,000 replicates, seed 152). The intervals cover variation over the sampled preset/performance dimensions, not training seeds or all possible sounds.
- **DAC:** only a mono decode/gradient compatibility probe was performed. The requested quantitative reconstruction benchmark is missing; no new DAC retrieval test was run or prior DAC retrieval score carried into this table. DAC cannot be ranked on reconstruction from these results. See [XPU setup and checkpoint revisions](XPU_SETUP.md).
- **Stage 2 audio recovery:** the previous worker stopped at 1,536/2,174 cached pairs on `expanded-0514:original`. Running it on a new engine selected full JSON readback (772 verified scalars) and produced a different waveform hash. Its original receipt records it as job 2 on a reused engine (11 verified scalars through the native parameter getter). Replaying `expanded-0512:original` first on the same isolated worker reproduced exact original audio hashes and renderer provenance for both jobs, including `963b69ae…` for the failure. See the [two-job replay evidence](STAGE2_AUDIO_RECOVERY_PROBE.json). This verifies the first failure; the remaining missing audio requires its own receipt checks.

[Machine-readable Stage 1 results and uncertainty](STAGE1_INTERIM.json) · [Issue 167 retrieval baseline](../../../modelling/frozen_encoder_retrieval/reports/issue167_v7_results.json) · [Issue 185 direct MLP baseline](../issue185_direct_flow_mlp/REPORT.md) · [Stage 2 preparation and outstanding integration](STAGE2_PREPARATION.md).

## Decision and remaining questions

Stage 1 supports continuing the pretrained-surrogate investigation: the tested codecs preserve enough acoustic detail to justify a bounded adapter experiment, especially Open Small and SAME-L. It has not established that a patch-and-MIDI adapter can predict those latents accurately, that a full generator improves on a decoder alone, or that either yields better gradients for real Vital patch optimization.

Resolve the candidate-selection and cache issues before formal Stage 2 training. Keep the existing direct MLP as the baseline, use the agreed matched adapter budget, and judge the final surrogate by actual Vital reconstruction after patch optimization in Stage 3. Do not reopen the main encoder-selection research on the basis of these retrieval results or treat this research as a prerequisite for #174.

This report expands the recorded Stage 1 findings without repeating retrieval, codec reconstruction or training. No independent confirmation/test panel, formal adapter learning curves, or Stage 3 results are available yet.
