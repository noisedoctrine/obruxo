# Basic-model sound diagnosis

Completed 2026-10-19 under the [declared basic-stage protocol](../STAGED_RECONSTRUCTION.md). **The strongest direction in these five cases is unison/stereo reconstruction.** Replacing the predicted unison voices, detune and pan with the source values reduces spectral error in all five cases, by 26.68% on average. This is an oracle diagnostic using known settings, not a learned model improvement or generalization result.

No model was trained or promoted. The output vocabulary stays at nine continuous controls and two categorical choices. No final-test state was used.

## Frozen comparison

The [earlier audit](BASIC_RECONSTRUCTION_AUDIT.md) selected cases 7, 1, 8, 22 and 14 at five error quantiles of the already-used 23-case development panel. The unchanged 404-state/16K-update checkpoint, source pool, case files, saved prediction files and accepted render receipts were verified before execution. Predictions were reused; no checkpoint was reselected and no new prediction was inferred from the controlled diagnostic audio.

Each case was rendered under its recorded reference and transfer performances with seven methods: source, source repeat, saved prediction, and separate envelope, level, filter and unison/stereo corrections. Each correction copies only its named control group from the source and keeps the rest of the prediction unchanged. Unit checks cover both enable/disable directions and legal inactive defaults.

Every diagnostic copy fixes oscillator 1–3 random phase and sampler random phase to zero, preserving other phase controls and seeds. Every request uses a fresh native process. These explicit controls differ from an unrestricted reconstruction task; the diagnostic scores must not replace the historical validation scores.

The [machine-readable report](data/basic_sound_diagnosis_summary.json) includes the pre-render plan, script/input identities, all request/audio hashes, per-case results and aggregate measurements.

## What changes the sound error?

Lower spectral error is better. The primary score equally weights reference and transfer within each case, then weights the five cases equally. Groups are corrected separately, so their changes are not additive.

| Method | Primary error | Change from prediction | Cases improved | Reference | Transfer |
|---|---:|---:|---:|---:|---:|
| Saved prediction | 0.8628 | — | — | 0.8183 | 0.9073 |
| Source envelope settings | 0.8783 | +1.80% | 4/5 | 0.8102 | 0.9463 |
| Source oscillator level | 0.8115 | -5.95% | 4/5 | 0.7539 | 0.8690 |
| Source filter settings | 0.8669 | +0.48% | 2/5 | 0.8156 | 0.9182 |
| Source unison/detune/pan | **0.6326** | **-26.68%** | **5/5** | **0.6326** | **0.6327** |

| Stored case | Prediction | Envelope correction | Unison/stereo correction |
|---:|---:|---:|---:|
| 7 | 0.4454 | 0.4031 | 0.4398 |
| 1 | 0.5954 | 0.4352 | 0.5355 |
| 8 | 0.8594 | 1.8180 | 0.6866 |
| 22 | 1.0750 | 0.5793 | 0.7691 |
| 14 | 1.3388 | 1.1559 | 0.7322 |

The envelope correction worsens case 8 enough to outweigh its other improvements. This illustrates why copying more exact source controls is not itself a quality objective: predicted controls can compensate for one another. The result does not establish that envelope learning is unimportant or that the test is defective.

Unison/stereo correction also lowers the stereo-energy diagnostic from roughly 0.110 to 0.006 on reference performances and 0.003 on transfers. Level correction improves relative RMS and envelope error. These auxiliary metrics are descriptive; they are not calibrated listening scores. The combined unison/stereo intervention cannot distinguish which of voice count, detune or pan is responsible for each improvement.

The next training investigation should test whether better audio-derived stereo/width and detuning evidence improves this same compact model. Keep its outputs fixed, compare against the unchanged baseline, and declare the experiment budget before training. This result does not justify adding oscillators, curves, routes or seed heads.

## Native correctness, compute and listening

All **70/70 requests passed audio QA**. All **10/10 source-repeat pairs were exactly equal** with the four random-phase controls fixed. No failure, replacement request or budget extension occurred during the native run. Renderer provenance matches the historical accepted requests.

The diagnostic took **105.28 seconds** after input verification, including fresh worker startup, native rendering, metric calculation and WAV output. It rendered 290.73 seconds of audio. Recorded native render time was 55.53 seconds: about 5.24 times real time for that component, or 2.76 times real time across the diagnostic loop. Peak sampled process-tree memory was **802.09 MiB**, within the 8 GiB budget; the 15-minute cap was not approached.

The first launcher exited before rendering because its WAV library could not load a DLL. The runner uses the installed SciPy float32 WAV writer instead; no dependency installation or extra native request was needed. The timing above covers the completed diagnostic, not that startup failure or input verification.

Seventy float32 listening clips are retained locally under `outputs/basic-sound-diagnosis-20261019/`, named `case-NNN-reference-METHOD.wav` or `case-NNN-transfer-METHOD.wav`. They are evaluation artifacts, not a pre-rendered training corpus. Listener assessment remains pending; no claim of audible preference is made from the metrics alone.

Local verification: three focused tests pass for control isolation, inactive legality and level/stereo metric behavior. All 70 saved float32 clips round-trip with the exact native audio hashes. Ruff and whitespace checks pass. The preceding scope PR passed 32 training/recovery tests and reloaded the historical checkpoint unchanged.

## Reproduction

Run `reports/scripts/diagnose_basic_sound.py` with `--pool`, `--evaluation`, `--checkpoint` and a new `--output` directory. Use the historical expanded pool, `fresh-validation-20261014` evaluation directory and `state-coverage-expanded-16000-20261013/checkpoint.pt`. The runner verifies their identities and writes its plan before rendering. It refuses an existing output directory and does not silently retry failed requests.

Publish a completed run using `reports/scripts/summarize_basic_diagnosis.py --source RUN/summary.json --output NEW_SUMMARY.json`. Publication requires all requests, exact repeats and resource checks to pass. Presets, audio and weights remain private local artifacts; the public report contains their identities and measurements.
