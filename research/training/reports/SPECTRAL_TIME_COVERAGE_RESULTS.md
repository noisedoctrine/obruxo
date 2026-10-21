# Training frames miss some short sound changes

The frozen local check completed on 2026-10-20. **Some attack/release changes are present in whole recordings but absent from their saved training frames.** The helper also fits these responses poorly when evaluated over the complete recording. Sparse sampling is a demonstrated weakness, not an established sole cause of the failed helper.

[Frozen card](SPECTRAL_TIME_COVERAGE_CARD.md) · [Verified machine-readable evidence](data/spectral_time_coverage_summary.json) · [Previous paired result](PAIRED_SPECTRAL_HELPER_RESULTS.md)

## What the check established

The check selected three existing training pairs per control by first/middle/last pair ID, without ranking by model error. All **36 native requests passed**. Every rerender matched its historical training audio hash; all 12 original-repeat pairs were exact, and all stored spectral/envelope frames were reproduced exactly. These observations rule out changed rendering or corrupt saved measurements for the inspected pairs.

There are concrete gaps in time coverage:

- Release pair 268 has a full-recording mean log-envelope change of 0.002625, but its 16 saved envelope frames show **zero change**. Its FFT-2048 mean change is 0.0004233 in the full recording versus 0.0000000295 in the sample—about 0.007% as large. The helper receives almost no response evidence from those measurements.
- Release pairs 16 and 501 have nonzero full FFT-2048 changes but **zero change** at the sampled times for that resolution. Other resolutions or envelope frames can still carry signal; this does not mean every measurement of those pairs is silent.
- Attack pair 496 has zero sampled envelope change despite a full envelope peak response of 0.1592. Its nearest saved envelope frame is 0.534 seconds from that peak. Attack pair 244 similarly misses the change at FFT-2048, though its sampled envelope does capture it.

The fixed sample can also overrepresent a brief change. For example, release pair 16's sampled mean envelope change is 0.0290 versus 0.00567 over the full recording. Thus the issue includes inconsistent weighting over time, not only missing all signal. These are descriptive examples from 12 chosen training pairs, not estimates of prevalence in the full population.

## Response fit over complete recordings

Lower signed-change error is better. The no-change baseline predicts that moving a control has no effect. Each control uses the existing paired-loss normalization, so compare helper and baseline within a row rather than compare magnitudes across controls.

| Control | Full helper error | Full no-change error | Full pairs beating no change | Sampled helper error | Sampled no-change error |
|---|---:|---:|---:|---:|---:|
| Attack | 0.06525 | 0.04999 | 0/3 | 0.14471 | 0.12780 |
| Decay | 0.13163 | 0.10494 | 1/3 | 0.12582 | 0.09674 |
| Release | 0.04266 | 0.03303 | 0/3 | 0.05616 | 0.04616 |
| Detune | 0.59702 | 0.61154 | 3/3 | 0.61076 | 0.62035 |

The helper fails the no-change comparison for every selected attack/release pair under full measurements as well as sampled measurements. That does not show that denser supervision cannot help: it was trained on the sparse frames. It does show that its current response is weak beyond an evaluation-sampling artifact. Detune has a small advantage in these training pairs; this does not overturn its failed transfer qualification.

## Cost, checks and decision

Elapsed time was **61.32 seconds**, peak process-tree RSS **906,551,296 bytes (0.84 GiB)**, with zero memory-sampling errors. The complete new plan, receipts and reports occupy **234,739 bytes**. All fit the frozen 36-attempt, 15-minute, 8-GiB RAM and 10-MiB storage limits. There were zero optimizer updates, no new training tensor/audio archive and no validation or test use.

Two focused synthetic tests pass, covering a missed response peak, invalid-time exclusion and the exact-response/no-change distinction. Ruff and whitespace checks pass. Publication verifies the frozen plan ID, selected pair membership, accepted request counts, case/receipt audio hashes and all repeat/frame checks. The runner was committed before execution as `78cbc93`; reproducing this diagnostic requires the frozen parent source modules in its plan.

Both helpers remain unqualified. The next implementation should preserve envelope time coverage and deliberately sample spectral frames around note starts and ends, alongside broad time coverage. First check that this revised measurement selection captures the known missed changes within the existing memory budget; do not launch another 12,000-update fit merely because the sampler code exists. Keep model size, output scope and native qualification thresholds unchanged. If response fitting remains weak with adequate supervision, investigate the restricted differentiable-synth/native-optimization alternative separately. Colab stays deferred.
