# Source-label schedules tested against audio quality

## Question

Does gradually reducing the source-setting loss help more than keeping it on, or turning it off after a short warm start?

## What was compared

All four versions started from the same random weights and received the same 600 batches of 64 audio examples from 172 training sound families. They used the same fixed neural audio proxy and 256 training audio views. The model had **355,806 trainable parameters**, **14 dense outputs**, and **11 decoded Vital settings**. The fixed Vital template supplied every other setting.

The proxy estimated how predicted patches would sound and supplied gradients during training. **Vital itself was not differentiated.** The final comparison rendered each version through real Vital on 21 held-out validation sound families, each at MIDI 55. There were 105 Vital renders total: 21 source recordings plus 84 predicted patches. The final test was not used.

Each version has its own table. ✅ indicates that its prediction head includes the setting; ❌ indicates that the setting stayed fixed to the reviewed template. The exact per-field inventory records all settings and training paths for all four versions: [machine-readable inventory](control_scope/preset_loss_schedule_comparison_20261114.jsonl). The [Stage A family scope](../../modelling/RESEARCH_TRACKER.md#current-stage-a-control-scope) and [master per-field inventory](../../modelling/control_scope/stage_a_controls.jsonl) define the shared fields. [Saved results, all case scores, schedule histories, hashes and paired comparisons](data/preset_loss_schedule_comparison_20261114_results.json).

### 90% label, fading to zero

| Control family / capability | On? | Enabled / total | Head outputs | Treatment in this version |
|---|:---:|---:|---:|---|
| Oscillator level | ✅ | 1 / 3 | 1 continuous | 90% label, fading to zero; 600 updates. Oscillator 1 only; 0.15–0.35 stored units |
| Oscillator pan | ✅ | 1 / 3 | 1 continuous | 90% label, fading to zero; 600 updates. Oscillator 1 only; −0.7–0.7 |
| Unison detune | ✅ | 1 / 3 | 1 continuous | 90% label, fading to zero; 600 updates. Oscillator 1 only; 0.5–2.5, masked at one voice |
| Unison voice count | ✅ | 1 / 3 | 3 scores | 90% label, fading to zero; 600 updates. Oscillator 1 chooses 1, 2, or 4 voices |
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
| Filter cutoff | ✅ | 1 / 2 | 1 continuous | 90% label, fading to zero; 600 updates. Filter 1 only; 50–120 stored units, masked while off |
| Filter resonance | ✅ | 1 / 2 | 1 continuous | 90% label, fading to zero; 600 updates. Filter 1 only; 0.1–0.6, masked while off |
| Filter on/off | ✅ | 1 / 2 | 2 scores | 90% label, fading to zero; 600 updates. Filter 1 chooses off or on |
| Filter model and style | ❌ | 0 / 4 | None | Fixed by template |
| Filter drive, mix, blend, and blend transpose | ❌ | 0 / 8 | None | Fixed by template |
| Filter formant controls | ❌ | 0 / 10 | None | Position, spread, transpose, and resonance fixed |
| Filter input and key tracking | ❌ | 0 / 4 | None | Fixed by template |
| Envelope attack/decay/sustain/release | ✅ | 4 / 24 | 4 continuous | 90% label, fading to zero; 600 updates. Envelope 1 only; other envelope instances fixed |
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

### Fixed source loss + audio loss (lambda 1)

| Control family / capability | On? | Enabled / total | Head outputs | Treatment in this version |
|---|:---:|---:|---:|---|
| Oscillator level | ✅ | 1 / 3 | 1 continuous | Fixed source loss + audio loss (lambda 1); 600 updates. Oscillator 1 only; 0.15–0.35 stored units |
| Oscillator pan | ✅ | 1 / 3 | 1 continuous | Fixed source loss + audio loss (lambda 1); 600 updates. Oscillator 1 only; −0.7–0.7 |
| Unison detune | ✅ | 1 / 3 | 1 continuous | Fixed source loss + audio loss (lambda 1); 600 updates. Oscillator 1 only; 0.5–2.5, masked at one voice |
| Unison voice count | ✅ | 1 / 3 | 3 scores | Fixed source loss + audio loss (lambda 1); 600 updates. Oscillator 1 chooses 1, 2, or 4 voices |
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
| Filter cutoff | ✅ | 1 / 2 | 1 continuous | Fixed source loss + audio loss (lambda 1); 600 updates. Filter 1 only; 50–120 stored units, masked while off |
| Filter resonance | ✅ | 1 / 2 | 1 continuous | Fixed source loss + audio loss (lambda 1); 600 updates. Filter 1 only; 0.1–0.6, masked while off |
| Filter on/off | ✅ | 1 / 2 | 2 scores | Fixed source loss + audio loss (lambda 1); 600 updates. Filter 1 chooses off or on |
| Filter model and style | ❌ | 0 / 4 | None | Fixed by template |
| Filter drive, mix, blend, and blend transpose | ❌ | 0 / 8 | None | Fixed by template |
| Filter formant controls | ❌ | 0 / 10 | None | Position, spread, transpose, and resonance fixed |
| Filter input and key tracking | ❌ | 0 / 4 | None | Fixed by template |
| Envelope attack/decay/sustain/release | ✅ | 4 / 24 | 4 continuous | Fixed source loss + audio loss (lambda 1); 600 updates. Envelope 1 only; other envelope instances fixed |
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

### 90% label / 10% audio, fixed

| Control family / capability | On? | Enabled / total | Head outputs | Treatment in this version |
|---|:---:|---:|---:|---|
| Oscillator level | ✅ | 1 / 3 | 1 continuous | 90% label / 10% audio, fixed; 600 updates. Oscillator 1 only; 0.15–0.35 stored units |
| Oscillator pan | ✅ | 1 / 3 | 1 continuous | 90% label / 10% audio, fixed; 600 updates. Oscillator 1 only; −0.7–0.7 |
| Unison detune | ✅ | 1 / 3 | 1 continuous | 90% label / 10% audio, fixed; 600 updates. Oscillator 1 only; 0.5–2.5, masked at one voice |
| Unison voice count | ✅ | 1 / 3 | 3 scores | 90% label / 10% audio, fixed; 600 updates. Oscillator 1 chooses 1, 2, or 4 voices |
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
| Filter cutoff | ✅ | 1 / 2 | 1 continuous | 90% label / 10% audio, fixed; 600 updates. Filter 1 only; 50–120 stored units, masked while off |
| Filter resonance | ✅ | 1 / 2 | 1 continuous | 90% label / 10% audio, fixed; 600 updates. Filter 1 only; 0.1–0.6, masked while off |
| Filter on/off | ✅ | 1 / 2 | 2 scores | 90% label / 10% audio, fixed; 600 updates. Filter 1 chooses off or on |
| Filter model and style | ❌ | 0 / 4 | None | Fixed by template |
| Filter drive, mix, blend, and blend transpose | ❌ | 0 / 8 | None | Fixed by template |
| Filter formant controls | ❌ | 0 / 10 | None | Position, spread, transpose, and resonance fixed |
| Filter input and key tracking | ❌ | 0 / 4 | None | Fixed by template |
| Envelope attack/decay/sustain/release | ✅ | 4 / 24 | 4 continuous | 90% label / 10% audio, fixed; 600 updates. Envelope 1 only; other envelope instances fixed |
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

### 100 label updates, then audio only

| Control family / capability | On? | Enabled / total | Head outputs | Treatment in this version |
|---|:---:|---:|---:|---|
| Oscillator level | ✅ | 1 / 3 | 1 continuous | 100 label updates, then audio only; 600 updates. Oscillator 1 only; 0.15–0.35 stored units |
| Oscillator pan | ✅ | 1 / 3 | 1 continuous | 100 label updates, then audio only; 600 updates. Oscillator 1 only; −0.7–0.7 |
| Unison detune | ✅ | 1 / 3 | 1 continuous | 100 label updates, then audio only; 600 updates. Oscillator 1 only; 0.5–2.5, masked at one voice |
| Unison voice count | ✅ | 1 / 3 | 3 scores | 100 label updates, then audio only; 600 updates. Oscillator 1 chooses 1, 2, or 4 voices |
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
| Filter cutoff | ✅ | 1 / 2 | 1 continuous | 100 label updates, then audio only; 600 updates. Filter 1 only; 50–120 stored units, masked while off |
| Filter resonance | ✅ | 1 / 2 | 1 continuous | 100 label updates, then audio only; 600 updates. Filter 1 only; 0.1–0.6, masked while off |
| Filter on/off | ✅ | 1 / 2 | 2 scores | 100 label updates, then audio only; 600 updates. Filter 1 chooses off or on |
| Filter model and style | ❌ | 0 / 4 | None | Fixed by template |
| Filter drive, mix, blend, and blend transpose | ❌ | 0 / 8 | None | Fixed by template |
| Filter formant controls | ❌ | 0 / 10 | None | Position, spread, transpose, and resonance fixed |
| Filter input and key tracking | ❌ | 0 / 4 | None | Fixed by template |
| Envelope attack/decay/sustain/release | ✅ | 4 / 24 | 4 continuous | 100 label updates, then audio only; 600 updates. Envelope 1 only; other envelope instances fixed |
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

## Results

Lower scores are better. Raw MR-STFT includes loudness; the gain-matched score removes one best-fit gain per render and is **not LUFS normalization**.

| Schedule | Raw MR-STFT | Gain-matched MR-STFT | Relative level error |
|---|---:|---:|---:|
| 90% label, fading to zero | 1.1358 | 1.3120 | 0.2382 |
| Fixed source loss + audio loss (lambda 1) | 1.1809 | 1.2472 | 0.3043 |
| 90% label / 10% audio, fixed | 1.3281 | 1.2838 | 0.3949 |
| 100 label updates, then audio only | 1.3540 | 1.3407 | 0.2686 |

The fading schedule had the best raw score and level match. After gain matching, the **fixed paper-style loss** had the best average score. In a post-hoc paired comparison on the same 21 families, the fading schedule's raw MR-STFT was 0.045 lower on average (12/21 wins; 95% interval −0.057 to +0.147). That interval includes no difference. For gain-matched MR-STFT, the paper-style score was 0.065 lower (12/21 wins; interval −0.269 to +0.130), also inconclusive.

Against the abrupt “100 label updates, then audio only” version, fading lowered raw MR-STFT by 0.218 on average, winning 17/21 families (95% interval −0.442 to −0.058). But after gain matching, the average advantage shrank to 0.029 and its interval included no difference. This points to output level as a major part of the raw-score gain; it does not establish a timbre-quality winner.

The paper-style comparison holds `source-setting loss + audio-proxy loss` at coefficient 1. Its losses were adapted to this small model and are **not a full reproduction of the paper**. It began with an 83.2% source-loss share, not exactly 90%. This was a held-out development screen at one pitch per family; these results and checkpoints later informed other development screens, so the 21 families are now development data rather than independent confirmation. It is not independent confirmation or a model release result. The separate [balanced-bank label-only check](BALANCED_SUPERVISED_CONTROL_RESULTS.md) asked a different question and does not duplicate this four-version comparison.

## What follows

Do not rerun the same loss-schedule comparison. The existing result does not settle a winner because raw and gain-matched scores disagree and the direct fade-versus-paper intervals include no difference. Keep this as supporting evidence for the warm-start idea, not a reason to add a preset loss permanently. The next A2 work should use the already established small balanced bank and test whether the model's audio-feedback path can improve real sound across its declared pitches while preserving earlier cases. Keep all 14 outputs and existing sound-quality gates; do not expand Vital controls.

The run used 2,400 total optimizer updates and 105 final Vital renders. Its retained summary does not include a reliable total runtime or RAM peak. The original runner, feature/proxy checkpoints and detailed render outputs remain in the ignored local run directory. Their exact hashes and all 21 per-family scores are retained in the linked JSON. A clean-clone rerun would need those source/data artifacts promoted or regenerated; no large model/audio archive is added to this PR.
