# Balanced-bank supervised control: results

## Question

Can the current small predictor learn the source control settings on the same balanced sounds used for the completed sound-training check, and does that help the resulting audio?

## What was compared

Both versions use the same 14-output model and the same eleven adjustable Vital settings. The first is the starting prediction before these diagnostic updates. The second received 48 updates against codec control labels. Each update used a four-example batch from the frozen balanced schedule, including two original reference examples. The loss compared predicted settings with source settings; it did not use audio-quality loss. The 18 balanced clips are training examples, not a held-out test.

### Starting prediction before diagnostic updates

Green means the prediction head exists. This version had zero optimizer updates in this diagnostic. All other listed settings stay at the reviewed template values.

| Control family / capability | On? | Enabled / total | Head outputs | Treatment in this version |
|---|:---:|---:|---:|---|
| Oscillator level | ✅ | 1 / 3 | 1 continuous | Prediction head is present; 0 updates in this diagnostic. Oscillator 1 only; 0.15–0.35 stored units |
| Oscillator pan | ✅ | 1 / 3 | 1 continuous | Prediction head is present; 0 updates in this diagnostic. Oscillator 1 only; −0.7–0.7 |
| Unison detune | ✅ | 1 / 3 | 1 continuous | Prediction head is present; 0 updates in this diagnostic. Oscillator 1 only; 0.5–2.5, masked at one voice |
| Unison voice count | ✅ | 1 / 3 | 3 scores | Prediction head is present; 0 updates in this diagnostic. Oscillator 1 chooses 1, 2, or 4 voices |
| Oscillator on/off | ❌ | 0 / 3 | None | Fixed off/on values from template |
| Oscillator pitch: transpose, tune, and quantize | ❌ | 0 / 9 | None | Fixed by template |
| Oscillator phase and random phase | ❌ | 0 / 6 | None | Fixed phases for repeatable renders |
| Oscillator distortion amount, type, phase, and spread | ❌ | 0 / 12 | None | Fixed by template |
| Wavetable frame selection | ❌ | 0 / 3 | None | Fixed frame values |
| Wavetable assets | ❌ | 0 / 3 | None | Template assets retained |
| Wavetable frame spread | ❌ | 0 / 3 | None | Fixed by template |
| Unison blend | ❌ | 0 / 3 | None | Fixed by template |
| Oscillator detune range and power | ❌ | 0 / 6 | None | Fixed by template |
| Spectral morph type, amount, and spread | ❌ | 0 / 9 | None | Fixed by template |
| Spectral unison and stack style | ❌ | 0 / 6 | None | Fixed by template |
| Oscillator stereo spread | ❌ | 0 / 3 | None | Fixed by template |
| Oscillator destination and MIDI tracking | ❌ | 0 / 6 | None | Fixed by template |
| Oscillator interpolation and 2D view | ❌ | 0 / 6 | None | Fixed by template |
| Custom oscillator curves | ❌ | 0 / 3 | None | No curve heads; template curves retained |
| Filter cutoff | ✅ | 1 / 2 | 1 continuous | Prediction head is present; 0 updates in this diagnostic. Filter 1 only; 50–120 stored units, masked while off |
| Filter resonance | ✅ | 1 / 2 | 1 continuous | Prediction head is present; 0 updates in this diagnostic. Filter 1 only; 0.1–0.6, masked while off |
| Filter on/off | ✅ | 1 / 2 | 2 scores | Prediction head is present; 0 updates in this diagnostic. Filter 1 chooses off or on |
| Filter model and style | ❌ | 0 / 4 | None | Fixed by template |
| Filter drive, mix, blend, and blend transpose | ❌ | 0 / 8 | None | Fixed by template |
| Filter formant controls | ❌ | 0 / 10 | None | Position, spread, transpose, and resonance fixed |
| Filter input and key tracking | ❌ | 0 / 4 | None | Fixed by template |
| Envelope attack/decay/sustain/release | ✅ | 4 / 24 | 4 continuous | Prediction head is present; 0 updates in this diagnostic. Envelope 1 only; other envelope instances fixed |
| Envelope delay and hold | ❌ | 0 / 12 | None | Fixed by template |
| Envelope curve powers | ❌ | 0 / 18 | None | Fixed by template |
| LFO rate and timing controls | ❌ | 0 / 48 | None | Eight LFOs remain at template settings |
| LFO phase, smoothing, stereo, and key tracking | ❌ | 0 / 48 | None | Eight LFOs remain at template settings |
| LFO curves | ❌ | 0 / 8 | None | Curves remain at template settings |
| Modulation route sources | ❌ | 0 / 64 | None | All source fields fixed |
| Modulation route destinations | ❌ | 0 / 64 | None | All destination fields fixed |
| Per-route amount, power, polarity, stereo, and bypass | ❌ | 0 / 320 | None | All five controls for each of 64 routes fixed |
| Per-route input mapping curves | ❌ | 0 / 64 | None | Linear default; no mapping heads |
| Sampler controls | ❌ | 0 / 11 | None | Fixed by template |
| Sample asset | ❌ | 0 / 1 | None | Fixed by template |
| Chorus controls | ❌ | 0 / 12 | None | Fixed by template |
| Compressor controls | ❌ | 0 / 20 | None | Fixed by template |
| Delay controls | ❌ | 0 / 12 | None | Fixed by template |
| Distortion controls | ❌ | 0 / 8 | None | Fixed by template |
| EQ controls | ❌ | 0 / 13 | None | Fixed by template |
| Flanger controls | ❌ | 0 / 9 | None | Fixed by template |
| Phaser controls | ❌ | 0 / 10 | None | Fixed by template |
| Reverb controls | ❌ | 0 / 13 | None | Fixed by template |
| Filter-FX controls | ❌ | 0 / 15 | None | Fixed by template |
| Randomizer controls | ❌ | 0 / 32 | None | Fixed by template |
| Macro controls | ❌ | 0 / 4 | None | Fixed by template |
| Voice controls | ❌ | 0 / 5 | None | Fixed by template |
| Portamento controls | ❌ | 0 / 4 | None | Fixed by template |
| Pitch controls | ❌ | 0 / 2 | None | Fixed by template |
| Stereo controls | ❌ | 0 / 2 | None | Fixed by template |
| Polyphony | ❌ | 0 / 1 | None | Fixed by template |
| Master volume | ❌ | 0 / 1 | None | Fixed by template |
| Tempo, bypass, effect order, legato, mod wheel, MPE, oversampling, velocity, and view settings | ❌ | 0 / 9 | None | Fixed by template |

### Prediction after 48 source-label updates

The same eleven settings are predicted after 48 Adam updates. Every other listed setting stays at the reviewed template value. **The optimizer used source-setting labels only; no sound loss was used.**

| Control family / capability | On? | Enabled / total | Head outputs | Treatment in this version |
|---|:---:|---:|---:|---|
| Oscillator level | ✅ | 1 / 3 | 1 continuous | 48 source-label updates; audio quality loss was not used. Oscillator 1 only; 0.15–0.35 stored units |
| Oscillator pan | ✅ | 1 / 3 | 1 continuous | 48 source-label updates; audio quality loss was not used. Oscillator 1 only; −0.7–0.7 |
| Unison detune | ✅ | 1 / 3 | 1 continuous | 48 source-label updates; audio quality loss was not used. Oscillator 1 only; 0.5–2.5, masked at one voice |
| Unison voice count | ✅ | 1 / 3 | 3 scores | 48 source-label updates; audio quality loss was not used. Oscillator 1 chooses 1, 2, or 4 voices |
| Oscillator on/off | ❌ | 0 / 3 | None | Fixed off/on values from template |
| Oscillator pitch: transpose, tune, and quantize | ❌ | 0 / 9 | None | Fixed by template |
| Oscillator phase and random phase | ❌ | 0 / 6 | None | Fixed phases for repeatable renders |
| Oscillator distortion amount, type, phase, and spread | ❌ | 0 / 12 | None | Fixed by template |
| Wavetable frame selection | ❌ | 0 / 3 | None | Fixed frame values |
| Wavetable assets | ❌ | 0 / 3 | None | Template assets retained |
| Wavetable frame spread | ❌ | 0 / 3 | None | Fixed by template |
| Unison blend | ❌ | 0 / 3 | None | Fixed by template |
| Oscillator detune range and power | ❌ | 0 / 6 | None | Fixed by template |
| Spectral morph type, amount, and spread | ❌ | 0 / 9 | None | Fixed by template |
| Spectral unison and stack style | ❌ | 0 / 6 | None | Fixed by template |
| Oscillator stereo spread | ❌ | 0 / 3 | None | Fixed by template |
| Oscillator destination and MIDI tracking | ❌ | 0 / 6 | None | Fixed by template |
| Oscillator interpolation and 2D view | ❌ | 0 / 6 | None | Fixed by template |
| Custom oscillator curves | ❌ | 0 / 3 | None | No curve heads; template curves retained |
| Filter cutoff | ✅ | 1 / 2 | 1 continuous | 48 source-label updates; audio quality loss was not used. Filter 1 only; 50–120 stored units, masked while off |
| Filter resonance | ✅ | 1 / 2 | 1 continuous | 48 source-label updates; audio quality loss was not used. Filter 1 only; 0.1–0.6, masked while off |
| Filter on/off | ✅ | 1 / 2 | 2 scores | 48 source-label updates; audio quality loss was not used. Filter 1 chooses off or on |
| Filter model and style | ❌ | 0 / 4 | None | Fixed by template |
| Filter drive, mix, blend, and blend transpose | ❌ | 0 / 8 | None | Fixed by template |
| Filter formant controls | ❌ | 0 / 10 | None | Position, spread, transpose, and resonance fixed |
| Filter input and key tracking | ❌ | 0 / 4 | None | Fixed by template |
| Envelope attack/decay/sustain/release | ✅ | 4 / 24 | 4 continuous | 48 source-label updates; audio quality loss was not used. Envelope 1 only; other envelope instances fixed |
| Envelope delay and hold | ❌ | 0 / 12 | None | Fixed by template |
| Envelope curve powers | ❌ | 0 / 18 | None | Fixed by template |
| LFO rate and timing controls | ❌ | 0 / 48 | None | Eight LFOs remain at template settings |
| LFO phase, smoothing, stereo, and key tracking | ❌ | 0 / 48 | None | Eight LFOs remain at template settings |
| LFO curves | ❌ | 0 / 8 | None | Curves remain at template settings |
| Modulation route sources | ❌ | 0 / 64 | None | All source fields fixed |
| Modulation route destinations | ❌ | 0 / 64 | None | All destination fields fixed |
| Per-route amount, power, polarity, stereo, and bypass | ❌ | 0 / 320 | None | All five controls for each of 64 routes fixed |
| Per-route input mapping curves | ❌ | 0 / 64 | None | Linear default; no mapping heads |
| Sampler controls | ❌ | 0 / 11 | None | Fixed by template |
| Sample asset | ❌ | 0 / 1 | None | Fixed by template |
| Chorus controls | ❌ | 0 / 12 | None | Fixed by template |
| Compressor controls | ❌ | 0 / 20 | None | Fixed by template |
| Delay controls | ❌ | 0 / 12 | None | Fixed by template |
| Distortion controls | ❌ | 0 / 8 | None | Fixed by template |
| EQ controls | ❌ | 0 / 13 | None | Fixed by template |
| Flanger controls | ❌ | 0 / 9 | None | Fixed by template |
| Phaser controls | ❌ | 0 / 10 | None | Fixed by template |
| Reverb controls | ❌ | 0 / 13 | None | Fixed by template |
| Filter-FX controls | ❌ | 0 / 15 | None | Fixed by template |
| Randomizer controls | ❌ | 0 / 32 | None | Fixed by template |
| Macro controls | ❌ | 0 / 4 | None | Fixed by template |
| Voice controls | ❌ | 0 / 5 | None | Fixed by template |
| Portamento controls | ❌ | 0 / 4 | None | Fixed by template |
| Pitch controls | ❌ | 0 / 2 | None | Fixed by template |
| Stereo controls | ❌ | 0 / 2 | None | Fixed by template |
| Polyphony | ❌ | 0 / 1 | None | Fixed by template |
| Master volume | ❌ | 0 / 1 | None | Fixed by template |
| Tempo, bypass, effect order, legato, mod wheel, MPE, oversampling, velocity, and view settings | ❌ | 0 / 9 | None | Fixed by template |

The exact inventory includes all 979 Vital fields for each version: [machine-readable control inventory](control_scope/balanced_supervised_control_20261114.jsonl). The [Stage A master inventory](../../modelling/control_scope/stage_a_controls.jsonl) and [current family summary](../../modelling/RESEARCH_TRACKER.md#current-stage-a-control-scope) define the unchanged field set. The [compact results data](data/balanced_supervised_control_20261114_results.json) contains the complete 48-step learning curve and all 18 per-clip audio scores.

## Result

The model moved closer to the source settings, but still missed the pre-set label-fit target:

| Measure | Before updates | After 48 updates | Outcome |
|---|---:|---:|---|
| Setting loss | 2.323 | 1.738 | Down 25.2% |
| Continuous-setting error | 0.353 | 0.246 | Down 30.3%; target < 0.06 |
| Correct categorical choices | 19/36 | 22/36 | Final target: 36/36 |

The decisive result is the actual Vital sound. Averaged over the 18 matching performances, both audio scores got worse after label training (lower is better):

| Sound group | Spectral score before → after | Change | MR-STFT score before → after | Change |
|---|---:|---:|---:|---:|
| MIDI 84 | 0.426 → 0.558 | +30.8% | 1.061 → 1.218 | +14.9% |
| **All 18 clips** | **0.408 → 0.527** | **+29.1%** | **1.079 → 1.295** | **+20.0%** |

Individual results were mixed: 7/18 improved on the spectral score and 9/18 improved on MR-STFT. But every source group’s mean worsened on both. The model is **not admitted**. This is not a test of generalization because the 18 clips were used in fitting.

## Run details

- 48 updates; 1.6 seconds of model fitting; 18 successful Vital renders; about 103.3 seconds total.
- CPU, two threads; peak measured process memory 1.28 GiB, below the 4-GiB cap.
- Vital 1.6.4 / DawDreamer 0.8.3; 32,768-frame preroll; all oscillator and sampler random phases fixed at zero.
- The first render launch lacked the pinned Vita path and produced no audio. It was corrected; the saved 48-update candidate was reused, and the final comparison contains 18 accepted renders.
- No reserved/test examples were used. No model was selected for use or release.

## What this means for the plan

Matching source settings is not enough: settings can differ while making equivalent sounds, and getting closer to the original numbers did not improve the audio here. Keep source-setting loss as a possible temporary warm start, then let sound quality decide. The agreed four-schedule comparison already exists on a different 172-sound training bank, with real Vital scoring on 21 held-out sounds at one pitch. See [the schedule results](PRESET_LOSS_SCHEDULE_RESULTS.md). That result is not a duplicate of this label-only check: it used sound-proxy loss, while this check used source-setting labels alone on the balanced 18-clip bank. Do not rerun the four schedules. The next A2 work should test the sound-feedback path on the current balanced setup, preserve the same 14 outputs, and score real Vital sound across the declared pitches.
