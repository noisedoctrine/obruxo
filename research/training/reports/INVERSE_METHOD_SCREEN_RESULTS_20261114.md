# Inverse-method development screen results — 14 November 2026

## Question

Can local patch refinement improve the sound score enough to be useful after the audio-only model has produced a starting patch? The screen compared direct edits scored by Vital, a small local fit to Vital scores, gradient-based patch adjustment through the neural sound copy, and per-clip tuning of a temporary model copy (test-time training, or TTT).

## What was compared

The starting predictions came from the selected checkpoint of the [completed source-label schedule comparison](PRESET_LOSS_SCHEDULE_RESULTS.md), including its linear 90-to-0 decay version. The screen covered 21 development sound families and varied oscillator 1 level, amplitude-envelope attack, and the model's predicted category choices. The methods were not given source-preset parameter values. The reported quality measure was gain-matched spectral error; a lower change is better.

The first comparison supplied the correct note pattern to control patch scoring. A follow-up supplied no source MIDI and estimated note events from audio. Tempo and velocity were fixed benchmark defaults in that follow-up.

## Results

| Method | Result on the 21-family panel | What it establishes |
|---|---|---|
| Render direct patch candidates in Vital and keep the best | Improved 20/21 families; mean paired error change **−0.379**, 95% interval **[−0.532, −0.235]** | Direct Vital scoring can select a better patch on this development panel when note events are correct. |
| Fit a small curve to Vital scores and choose an edit | Improved 20/21; mean paired error change **−0.094**, 95% interval **[−0.137, −0.055]** | The local fit also moved scores in the right direction here. It reused candidate renders, so this is not an independent evaluation of the fitted choice. |
| Use gradients through the neural sound copy to adjust patch values | Inconclusive | This screen did not establish a reliable gain from proxy gradients for patch refinement. Vital itself does not provide gradients. |
| Fine-tune a temporary copy of the model for each clip (TTT) | Inconclusive | This screen did not establish a reliable gain from per-clip model tuning. |

These are development-screen outcomes, not evidence that either successful method improves audio-only reconstruction on new sounds.

## Audio-derived note timing check

In the follow-up, Basic Pitch identified the correct pitch in all 21 clips, but its note starts averaged **27 ms late** and note lengths averaged **64 ms too long**. For the same frozen predicted patch at pitch 55, the score increased from **1.312** with the correct note event to **2.494** with the estimated event. This shows that timing errors can overwhelm the patch comparison even when pitch is right. The follow-up therefore cannot fairly rank the patch-refinement methods.

## Limits and record gaps

- The 21 families had already informed checkpoint and proxy selection. Neither comparison is independent promotion evidence.
- The first comparison used correct note events, so it tests patch scoring under controlled performance rather than the audio-only inference path.
- The second comparison removed source MIDI but had the measured onset and duration errors above; tempo and velocity also remained fixed defaults.
- The durable repository record does not contain the per-family score table, the exact candidate edit list and proposal rule, or all run settings. The aggregate outcomes above are recorded, but the experiment is not fully reproducible from this repository. In particular, the candidate-search result should not be treated as a reusable recipe until those details are recovered or a new frozen run records them.
- Vital is non-differentiable. The gradient-based method used the neural sound copy, so its outcome is about that proxy-gradient path, not gradients through real Vital.

## Decision

Keep direct Vital candidate search, local score fitting, proxy-based patch adjustment, and per-clip TTT deferred until the base model passes its reconstruction-quality gate. This screen does not select a winning inference-time method. It also does not decide whether the neural sound copy should provide audio-loss gradients during shared model training; that is a separate core-training question.
