# Sound-prediction helper: native feasibility result

The helper does not pass the frozen native qualification gate. It is not connected to inverse training.

The helper predicts fixed sound measurements from patch controls and actual notes. The check asks whether those predictions generalize and whether its suggested changes improve real Vital audio. This is a single-seed screen on the authored alpha vocabulary, not a finished-model quality claim.

[Frozen plan](data/sound_proxy_plan.json) · [Aggregate evidence](data/sound_proxy_summary.json) · [Protocol](../AUDIO_PROXY.md)

## Held-out prediction

Errors below use equal measurement-group weights and training-only coordinate scales. They are helper-regression errors, separate from native MR-STFT sound distance. Lower is better.

| Performance | Scored/planned | Helper | Training mean | Nearest training | Five-neighbor interpolation | Gate |
|---|---:|---:|---:|---:|---:|---|
| reference | 10/10 | 0.0645 | 0.1610 | 0.1308 | 0.1128 | pass |
| transfer | 10/10 | 0.4504 | 0.4224 | 0.4397 | 0.4212 | fail |

## Advice checked against Vital

Candidate rankings agree with real renders on **35/60 scored comparisons**; 0 comparisons are ties under the frozen repeat-noise rule, out of 60 planned. Missing comparisons stay in the planned denominator.

Helper-guided updates improve real sound beyond source-repeat noise on **0/20 scored performances**, out of 20 planned. Mean guided-minus-starting native MR-STFT change: **0.9194017792702652**. Negative means improvement. The paired-state bootstrap interval is **[0.6507037287636195, 1.181922041243427]**; it is descriptive and resamples patches, not individual correlated performances.

The update lowered the helper's predicted error on 20/20 cases. Real error decreased on 0 and increased on 20. Prediction accuracy and useful input gradients are separate requirements; this result diagnoses this helper and bounded update, not the underlying need for audio-based training.

| Synthesis family | Forward scored/planned | Helper error | Nearest-training error | Native guided-minus-starting change |
|---|---:|---:|---:|---:|
| wavetable | 4/4 | 0.2305 | 0.2041 | 0.3417 |
| chorus | 4/4 | 0.4851 | 0.4994 | 1.2078 |
| modulated | 4/4 | 0.2697 | 0.2614 | 0.6446 |
| filtered | 4/4 | 0.0404 | 0.0956 | 0.9876 |
| layered | 4/4 | 0.2615 | 0.3658 | 1.4154 |

## Data, compute and limits

Native QA accepted **374/376 attempts**. Outcomes by phase: `{'training': {'accepted': 254, 'rejected': 2}, 'validation': {'accepted': 120}}`. Rejected diagnostics: `{'audio.clipping': 2, 'vital.runtime.canonicalization': 2, 'vital.runtime.numeric_canonicalization': 2, 'vital.runtime.version_canonicalization': 2}`. Accepted training data contains 254 measurements from 64 patches, covering {'chorus': 12, 'filtered': 13, 'layered': 13, 'modulated': 13, 'wavetable': 13}.

The helper has 426,464 parameters. Retained measurements occupy 1.21 MiB; sampled peak process-tree memory is 1.12 GiB. The CPU run took 18.85 minutes after pool/plan preparation. Summed native render wall time was 17.85 minutes. Four CPU threads, float32, no paid compute and no audio archive were used.

Training uses only the four reference performances. Reserved performances change register, velocity, timing and/or note combinations; their input range is broader than this small training bank. Reference and transfer results therefore remain separate. This screen cannot establish performance across the full Vital vocabulary or real recordings. Linux/Colab native rendering remains unqualified.

The complete local training/renderer suite passes **164 tests**, including fixed measurement behavior, active/inactive input gradients, exact optimizer recovery, QA rejection, native collection recovery, artifact loading, data tampering and qualification gates. Ruff and whitespace checks are recorded separately in the PR. GitHub Actions stays disabled.

These are development validation patches, including patches used in earlier screens. The interval does not account for repeated development decisions. The final test reserve is untouched. The inverse model and selected release weights are unchanged. A passed helper gate permits a later inverse-training comparison; it does not establish that comparison's outcome.

## Next planned capability

Implement bounded candidate improvement checked directly with Vital. This also provides the training plan's fallback route to better audio targets without relying on this helper's gradients. Larger proxy runs, tuning and seed replication remain deferred while that broad direction is screened. Audio-based training remains a requirement; this failed helper/update is not evidence against it.
