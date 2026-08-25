# Vital Component-by-Component Value Distributions

This companion report uses the **file weighted** aggregate (9,618 parsed presets). It is organized by the pinned Vital component registry rather than pooling unrelated parameter names.

The row-oriented exports are the complete machine-readable detail: [`vital_usage_categorical_values.csv`](vital_usage_categorical_values.csv) contains one row per categorical value, and [`vital_usage_continuous_bins.csv`](vital_usage_continuous_bins.csv) contains all 64 histogram bins for each continuous parameter, with both file-weighted and exact-deduplicated counts.

Categorical values are Vital's raw numeric ordinals. Labels are included when the pinned atlas provides an option list. Their denominator is the observed value count under the existing policy that fills missing common scalar keys with the atlas default.

Continuous parameters include raw-value min/max/mean/stddev, histogram quantiles, default and zero prevalence, and dominant bins. Atlas-backed controls use 64 bins in normalized control position, preserving the parameter's scale metadata; version-introduced controls without atlas bounds use adaptive raw-value bins. 16 atlas-backed controls also use complete raw-value bins because their corpus observations exceed the pinned bounds; their partial normalized histograms remain in the JSON. For an `Exponential` parameter, the raw Vital value is already the logarithmic storage domain.

## Global controls

Voice, performance, macro, routing, and other controls that do not belong to a repeated component.

### Global controls

28 scalar parameters: 14 categorical/enum and 14 continuous. component-level enable/routing state is not applicable.

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `bypass` | 9,618 | 1 | 0: 9,618 (100.0%) |
| `effect_chain_order` | 9,618 | 2,730 | 0: 3,355 (34.9%); 1.512e+04: 127 (1.3%); 2.57e+05: 115 (1.2%) |
| `legato` | 9,618 | 2 | 0 (Off): 7,843 (81.5%); 1 (On): 1,775 (18.5%) |
| `mpe_enabled` | 9,618 | 2 | 0 (Off): 9,574 (99.5%); 1 (On): 44 (0.5%) |
| `oversampling` | 9,618 | 4 | 1 (2x): 9,231 (96.0%); 2 (4x): 207 (2.2%); 0 (1x): 128 (1.3%) |
| `pitch_bend_range` | 9,618 | 21 | 2: 8,861 (92.1%); 12: 378 (3.9%); 0: 90 (0.9%) |
| `polyphony` | 9,618 | 29 | 8: 6,007 (62.5%); 1: 2,680 (27.9%); 12: 158 (1.6%) |
| `portamento_force` | 9,618 | 2 | 0 (Off): 8,546 (88.9%); 1 (On): 1,072 (11.1%) |
| `portamento_scale` | 9,618 | 2 | 0 (Off): 9,438 (98.1%); 1 (On): 180 (1.9%) |
| `stereo_mode` | 9,618 | 2 | 0 (SPREAD): 9,581 (99.6%); 1 (ROTATE): 37 (0.4%) |
| `view_spectrogram` | 9,618 | 3 | 0 (Off): 8,921 (92.8%); 1 (On): 696 (7.2%); 2: 1 (0.0%) |
| `voice_override` | 9,618 | 2 | 0 (Kill): 9,557 (99.4%); 1 (Steal): 61 (0.6%) |
| `voice_priority` | 9,618 | 5 | 4 (Round Robin): 9,535 (99.1%); 0 (Newest): 44 (0.5%); 1 (Oldest): 17 (0.2%) |
| `voice_transpose` | 9,618 | 65 | 0: 9,164 (95.3%); -12: 101 (1.1%); 12: 65 (0.7%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `beats_per_minute` | 9,618 | 0.16667–7 | 1.4428 / 2.15163 / 2.91047 | 18.31982% | 0% | 1.98177–2.08854 raw: 1,957 (20.3%) |
| `macro_control_1` | 9,618 | 0–1 | 0.00113 / 0.01127 / 0.9911 | 67.78956% | 67.78956% | 0–0.01562 raw / 0–0.01562 norm: 6,669 (69.3%) |
| `macro_control_2` | 9,618 | 0–1 | 0.00101 / 0.01008 / 0.98674 | 76.0969% | 76.0969% | 0–0.01562 raw / 0–0.01562 norm: 7,450 (77.5%) |
| `macro_control_3` | 9,618 | 0–1 | 0.0009442 / 0.00944 / 0.84014 | 81.96091% | 81.96091% | 0–0.01562 raw / 0–0.01562 norm: 7,957 (82.7%) |
| `macro_control_4` | 9,618 | 0–1 | 0.0009066 / 0.00907 / 0.69325 | 85.24641% | 85.24641% | 0–0.01562 raw / 0–0.01562 norm: 8,287 (86.2%) |
| `mod_wheel` | 9,618 | 0–1 | 0.0008394 / 0.00839 / 0.26602 | 92.83635% | 92.83635% | 0–0.01562 raw / 0–0.01562 norm: 8,951 (93.1%) |
| `pitch_wheel` | 9,618 | -1–1 | 0.00145 / 0.01559 / 0.02973 | 99.02267% | 99.02267% | 0–0.03125 raw / 0.5–0.51562 norm: 9,564 (99.4%) |
| `portamento_slope` | 9,618 | -8–8 | 0.00217 / 0.12606 / 0.24995 | 90.47619% | 90.47619% | 0–0.25 raw / 0.5–0.51562 norm: 8,733 (90.8%) |
| `portamento_time` | 9,618 | -10–4 | -9.98469 / -9.84691 / -2.49671 | 70.77355% | 0.0104% | -10–-9.78125 raw / 0–0.01562 norm: 6,871 (71.4%) |
| `stereo_routing` | 9,618 | 0–1 | 0.65505 / 0.99147 / 0.99915 | 91.38074% | 1.50759% | 0.98438–1 raw / 0.98438–1 norm: 8,812 (91.6%) |
| `velocity_track` | 9,618 | -1–1 | 0.00144 / 0.01645 / 0.25211 | 93.51216% | 93.51216% | 0–0.03125 raw / 0.5–0.51562 norm: 9,006 (93.6%) |
| `voice_amplitude` | 9,618 | 1–1 | 0.98516 / 0.99219 / 0.99922 | 100% | 0% | 0.98438–1 raw / 0.98438–1 norm: 9,618 (100.0%) |
| `voice_tune` | 9,618 | -1–1 | 0.00139 / 0.01562 / 0.02985 | 98.79393% | 98.79393% | 0–0.03125 raw / 0.5–0.51562 norm: 9,505 (98.8%) |
| `volume` | 9,618 | 0–7399 | 4396 / 5563 / 6526 | 55.48971% | 0.40549% | 5465–5665 raw / 0.85938–0.875 norm: 5,606 (58.3%) |

## Oscillators

The three wavetable oscillators, including oscillator routing, spectral morph controls, and nested wavetable-editor state.

### Oscillator 1

30 scalar parameters: 12 categorical/enum and 18 continuous. enabled 95.9% (9,223/9,618); changed 98.0% (9,427/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `osc_1_destination` | 9,618 | 5 | 0 (FILTER 1): 7,545 (78.4%); 2 (FILTER 1+2): 775 (8.1%); 3 (EFFECTS): 730 (7.6%) |
| `osc_1_distortion_type` | 9,618 | 13 | 0 (None): 5,712 (59.4%); 7 (FM <- Osc): 1,289 (13.4%); 1 (Sync): 707 (7.4%) |
| `osc_1_midi_track` | 9,618 | 2 | 1 (On): 9,493 (98.7%); 0 (Off): 125 (1.3%) |
| `osc_1_on` | 9,618 | 2 | 1 (On): 9,223 (95.9%); 0 (Off): 395 (4.1%) |
| `osc_1_smooth_interpolation` | 9,618 | 2 | 0 (Off): 9,014 (93.7%); 1 (On): 604 (6.3%) |
| `osc_1_spectral_morph_type` | 9,618 | 16 | 0 (None): 6,385 (66.4%); 2 (Formant Scale): 532 (5.5%); 3 (Harmonic Stretch): 401 (4.2%) |
| `osc_1_spectral_unison` | 9,618 | 3 | 1 (On): 9,598 (99.8%); 2: 11 (0.1%); 0 (Off): 9 (0.1%) |
| `osc_1_stack_style` | 9,618 | 13 | 0 (Unison): 9,132 (94.9%); 1 (Center Drop 12): 85 (0.9%); 3 (Octave): 76 (0.8%) |
| `osc_1_transpose` | 9,618 | 85 | 0: 4,971 (51.7%); -12: 1,550 (16.1%); -24: 1,453 (15.1%) |
| `osc_1_transpose_quantize` | 9,618 | 117 | 0: 9,366 (97.4%); 1: 58 (0.6%); 4096: 16 (0.2%) |
| `osc_1_unison_voices` | 9,618 | 16 | 1: 4,606 (47.9%); 16: 864 (9.0%); 2: 748 (7.8%) |
| `osc_1_view_2d` | 9,618 | 3 | 1 (On): 8,796 (91.5%); 0 (Off): 705 (7.3%); 2: 117 (1.2%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `osc_1_detune_power` | 9,618 | -5–5 | 0.9313 / 1.48377 / 1.62457 | 88.9582% | 0.3847% | 1.40625–1.5625 raw / 0.64062–0.65625 norm: 8,575 (89.2%) |
| `osc_1_detune_range` | 9,618 | 0–48 | 1.5229 / 1.87322 / 2.22354 | 95.61239% | 0.76939% | 1.5–2.25 raw / 0.03125–0.04688 norm: 9,265 (96.3%) |
| `osc_1_distortion_amount` | 9,618 | 0–1 | 0.00771 / 0.50556 / 0.69191 | 59.84612% | 9.28467% | 0.5–0.51562 raw / 0.5–0.51562 norm: 5,903 (61.4%) |
| `osc_1_distortion_phase` | 9,618 | 0–1 | 0.50046 / 0.50782 / 0.51518 | 95.06134% | 0.66542% | 0.5–0.51562 raw / 0.5–0.51562 norm: 9,188 (95.5%) |
| `osc_1_distortion_spread` | 9,618 | -0.5–0.5 | 0.0005282 / 0.00788 / 0.01524 | 95.55001% | 95.55001% | 0–0.01562 raw / 0.5–0.51562 norm: 9,195 (95.6%) |
| `osc_1_frame_spread` | 9,618 | -128–128 | 0.1401 / 2.0357 / 3.9313 | 94.72863% | 94.72863% | 0–4 raw / 0.5–0.51562 norm: 9,132 (94.9%) |
| `osc_1_level` | 9,618 | 0–1 | 0.06066 / 0.69787 / 0.9829 | 44.61426% | 17.58162% | 0.69597–0.70711 raw / 0.48438–0.5 norm: 4,361 (45.3%) |
| `osc_1_pan` | 9,618 | -1–1 | 0.0007176 / 0.01538 / 0.03003 | 95.49802% | 95.49802% | 0–0.03125 raw / 0.5–0.51562 norm: 9,226 (95.9%) |
| `osc_1_phase` | 9,618 | 0–1 | 0.00654 / 0.50651 / 0.51508 | 81.76336% | 11.81119% | 0.5–0.51562 raw / 0.5–0.51562 norm: 7,886 (82.0%) |
| `osc_1_random_phase` | 9,618 | 0–1 | 0.00291 / 0.98875 / 0.99887 | 69.25556% | 26.39842% | 0.98438–1 raw / 0.98438–1 norm: 6,678 (69.4%) |
| `osc_1_spectral_morph_amount` | 9,618 | 0–1 | 0.0099 / 0.50632 / 0.77261 | 63.5891% | 6.96611% | 0.5–0.51562 raw / 0.5–0.51562 norm: 6,264 (65.1%) |
| `osc_1_spectral_morph_phase` | 7,517 | 0–1 | 0.50056 / 0.50772 / 0.51489 | — | 0.439% | 0.5–0.51562 raw: 7,375 (98.1%) |
| `osc_1_spectral_morph_spread` | 9,618 | -0.5–0.5 | 0.0004611 / 0.00783 / 0.0152 | 95.31088% | 95.31088% | 0–0.01562 raw / 0.5–0.51562 norm: 9,178 (95.4%) |
| `osc_1_stereo_spread` | 9,618 | 0–1 | 0.84039 / 0.99175 / 0.99917 | 94.57268% | 1.56997% | 0.98438–1 raw / 0.98438–1 norm: 9,104 (94.7%) |
| `osc_1_tune` | 9,618 | -1–1 | 0.0005405 / 0.01555 / 0.03056 | 92.97151% | 92.97151% | 0–0.03125 raw / 0.5–0.51562 norm: 9,011 (93.7%) |
| `osc_1_unison_blend` | 9,618 | 0–1 | 0.79706 / 0.80464 / 0.81221 | 92.71158% | 0.53026% | 0.79688–0.8125 raw / 0.79688–0.8125 norm: 8,927 (92.8%) |
| `osc_1_unison_detune` | 9,618 | 0–10 | 0.67664 / 3.81937 / 4.61748 | 40.96486% | 10.76107% | 4.33013–4.50694 raw / 0.1875–0.20312 norm: 4,021 (41.8%) |
| `osc_1_wave_frame` | 9,618 | 0–256 | 0.33515 / 3.35145 / 208.19167 | 58.43211% | 58.43211% | 0–4 raw / 0–0.01562 norm: 5,739 (59.7%) |

### Oscillator 2

30 scalar parameters: 12 categorical/enum and 18 continuous. enabled 80.3% (7,721/9,618); changed 89.6% (8,617/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `osc_2_destination` | 9,618 | 6 | 1 (FILTER 2): 4,702 (48.9%); 0 (FILTER 1): 2,211 (23.0%); 2 (FILTER 1+2): 1,546 (16.1%) |
| `osc_2_distortion_type` | 9,618 | 13 | 0 (None): 6,902 (71.8%); 1 (Sync): 591 (6.1%); 7 (FM <- Osc): 401 (4.2%) |
| `osc_2_midi_track` | 9,618 | 2 | 1 (On): 9,530 (99.1%); 0 (Off): 88 (0.9%) |
| `osc_2_on` | 9,618 | 2 | 1 (On): 7,721 (80.3%); 0 (Off): 1,897 (19.7%) |
| `osc_2_smooth_interpolation` | 9,618 | 2 | 0 (Off): 9,204 (95.7%); 1 (On): 414 (4.3%) |
| `osc_2_spectral_morph_type` | 9,618 | 16 | 0 (None): 7,092 (73.7%); 5 (Smear): 381 (4.0%); 7 (Low Pass): 355 (3.7%) |
| `osc_2_spectral_unison` | 9,618 | 3 | 1 (On): 9,600 (99.8%); 0 (Off): 10 (0.1%); 2: 8 (0.1%) |
| `osc_2_stack_style` | 9,618 | 13 | 0 (Unison): 9,277 (96.5%); 1 (Center Drop 12): 69 (0.7%); 10 (Odd Harmonics): 61 (0.6%) |
| `osc_2_transpose` | 9,618 | 90 | 0: 4,744 (49.3%); -12: 1,582 (16.4%); -24: 990 (10.3%) |
| `osc_2_transpose_quantize` | 9,618 | 95 | 0: 9,431 (98.1%); 1: 34 (0.4%); 4096: 9 (0.1%) |
| `osc_2_unison_voices` | 9,618 | 17 | 1: 5,558 (57.8%); 2: 698 (7.3%); 16: 589 (6.1%) |
| `osc_2_view_2d` | 9,618 | 3 | 1 (On): 9,111 (94.7%); 0 (Off): 451 (4.7%); 2: 56 (0.6%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `osc_2_detune_power` | 9,618 | -5–5 | 1.4085 / 1.48459 / 1.56067 | 92.2749% | 0.20794% | 1.40625–1.5625 raw / 0.64062–0.65625 norm: 8,887 (92.4%) |
| `osc_2_detune_range` | 9,618 | 0–48 | 1.52778 / 1.8734 / 2.21902 | 97.36952% | 0.42628% | 1.5–2.25 raw / 0.03125–0.04688 norm: 9,391 (97.6%) |
| `osc_2_distortion_amount` | 9,618 | 0–1 | 0.01147 / 0.50667 / 0.67113 | 71.03348% | 6.16552% | 0.5–0.51562 raw / 0.5–0.51562 norm: 6,923 (72.0%) |
| `osc_2_distortion_phase` | 9,618 | 0–1 | 0.50049 / 0.5078 / 0.51511 | 95.93471% | 0.56145% | 0.5–0.51562 raw / 0.5–0.51562 norm: 9,246 (96.1%) |
| `osc_2_distortion_spread` | 9,618 | -0.5–0.5 | 0.0006244 / 0.00783 / 0.01504 | 97.46309% | 97.46309% | 0–0.01562 raw / 0.5–0.51562 norm: 9,380 (97.5%) |
| `osc_2_frame_spread` | 9,618 | -128–128 | 0.16198 / 2.01715 / 3.87232 | 96.86005% | 96.86005% | 0–4 raw / 0.5–0.51562 norm: 9,331 (97.0%) |
| `osc_2_level` | 9,618 | 0–1 | 0.05203 / 0.56857 / 0.80872 | 34.63298% | 23.49761% | 0.69597–0.70711 raw / 0.48438–0.5 norm: 3,376 (35.1%) |
| `osc_2_pan` | 9,618 | -1–1 | 0.0009115 / 0.01564 / 0.03037 | 95.10293% | 95.10293% | 0–0.03125 raw / 0.5–0.51562 norm: 9,183 (95.5%) |
| `osc_2_phase` | 9,618 | 0–1 | 0.00755 / 0.50668 / 0.51508 | 83.59326% | 10.06446% | 0.5–0.51562 raw / 0.5–0.51562 norm: 8,046 (83.7%) |
| `osc_2_random_phase` | 9,618 | 0–1 | 0.00345 / 0.98946 / 0.99894 | 73.97588% | 22.29154% | 0.98438–1 raw / 0.98438–1 norm: 7,129 (74.1%) |
| `osc_2_spectral_morph_amount` | 9,618 | 0–1 | 0.013 / 0.50686 / 0.73549 | 70.50322% | 5.44812% | 0.5–0.51562 raw / 0.5–0.51562 norm: 6,910 (71.8%) |
| `osc_2_spectral_morph_phase` | 7,517 | 0–1 | 0.48655 / 0.50412 / 0.51452 | — | 0.05321% | 0.5–0.51562 raw: 5,078 (67.6%) |
| `osc_2_spectral_morph_spread` | 9,618 | -0.5–0.5 | 0.0006057 / 0.00783 / 0.01505 | 97.23435% | 97.23435% | 0–0.01562 raw / 0.5–0.51562 norm: 9,361 (97.3%) |
| `osc_2_stereo_spread` | 9,618 | 0–1 | 0.98462 / 0.99191 / 0.99919 | 96.50655% | 1.15409% | 0.98438–1 raw / 0.98438–1 norm: 9,284 (96.5%) |
| `osc_2_tune` | 9,618 | -1–1 | 0.0004638 / 0.01569 / 0.03091 | 91.74465% | 91.74465% | 0–0.03125 raw / 0.5–0.51562 norm: 8,884 (92.4%) |
| `osc_2_unison_blend` | 9,618 | 0–1 | 0.79729 / 0.80466 / 0.81203 | 95.36286% | 0.41589% | 0.79688–0.8125 raw / 0.79688–0.8125 norm: 9,177 (95.4%) |
| `osc_2_unison_detune` | 9,618 | 0–10 | 0.77374 / 4.35766 / 4.5056 | 53.04637% | 8.56727% | 4.33013–4.50694 raw / 0.1875–0.20312 norm: 5,157 (53.6%) |
| `osc_2_wave_frame` | 9,618 | 0–256 | 0.31204 / 3.12038 / 200.55333 | 63.16282% | 63.16282% | 0–4 raw / 0–0.01562 norm: 6,164 (64.1%) |

### Oscillator 3

30 scalar parameters: 12 categorical/enum and 18 continuous. enabled 53.7% (5,162/9,618); changed 62.2% (5,984/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `osc_3_destination` | 9,618 | 5 | 3 (EFFECTS): 6,054 (62.9%); 0 (FILTER 1): 1,848 (19.2%); 1 (FILTER 2): 946 (9.8%) |
| `osc_3_distortion_type` | 9,618 | 13 | 0 (None): 7,959 (82.8%); 1 (Sync): 401 (4.2%); 7 (FM <- Osc): 225 (2.3%) |
| `osc_3_midi_track` | 9,618 | 2 | 1 (On): 9,542 (99.2%); 0 (Off): 76 (0.8%) |
| `osc_3_on` | 9,618 | 2 | 1 (On): 5,162 (53.7%); 0 (Off): 4,456 (46.3%) |
| `osc_3_smooth_interpolation` | 9,618 | 2 | 0 (Off): 9,309 (96.8%); 1 (On): 309 (3.2%) |
| `osc_3_spectral_morph_type` | 9,618 | 16 | 0 (None): 7,936 (82.5%); 7 (Low Pass): 317 (3.3%); 2 (Formant Scale): 199 (2.1%) |
| `osc_3_spectral_unison` | 9,618 | 3 | 1 (On): 9,602 (99.8%); 0 (Off): 11 (0.1%); 2: 5 (0.1%) |
| `osc_3_stack_style` | 9,618 | 13 | 0 (Unison): 9,371 (97.4%); 1 (Center Drop 12): 41 (0.4%); 10 (Odd Harmonics): 41 (0.4%) |
| `osc_3_transpose` | 9,618 | 83 | 0: 6,110 (63.5%); -12: 1,117 (11.6%); -24: 864 (9.0%) |
| `osc_3_transpose_quantize` | 9,618 | 73 | 0: 9,472 (98.5%); 1: 27 (0.3%); 4096: 8 (0.1%) |
| `osc_3_unison_voices` | 9,618 | 16 | 1: 7,039 (73.2%); 16: 488 (5.1%); 2: 375 (3.9%) |
| `osc_3_view_2d` | 9,618 | 3 | 1 (On): 9,330 (97.0%); 0 (Off): 252 (2.6%); 2: 36 (0.4%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `osc_3_detune_power` | 9,618 | -5–5 | 1.41099 / 1.48453 / 1.55807 | 95.43564% | 0.15596% | 1.40625–1.5625 raw / 0.64062–0.65625 norm: 9,195 (95.6%) |
| `osc_3_detune_range` | 9,618 | 0–48 | 1.53144 / 1.87342 / 2.2154 | 98.50281% | 0.34311% | 1.5–2.25 raw / 0.03125–0.04688 norm: 9,491 (98.7%) |
| `osc_3_distortion_amount` | 9,618 | 0–1 | 0.0992 / 0.50724 / 0.52799 | 82.81347% | 3.52464% | 0.5–0.51562 raw / 0.5–0.51562 norm: 8,021 (83.4%) |
| `osc_3_distortion_phase` | 9,618 | 0–1 | 0.5006 / 0.5078 / 0.515 | 97.46309% | 0.3639% | 0.5–0.51562 raw / 0.5–0.51562 norm: 9,392 (97.7%) |
| `osc_3_distortion_spread` | 9,618 | -0.5–0.5 | 0.0006848 / 0.00783 / 0.01497 | 98.36764% | 98.36764% | 0–0.01562 raw / 0.5–0.51562 norm: 9,465 (98.4%) |
| `osc_3_frame_spread` | 9,618 | -128–128 | 0.18206 / 2.01504 / 3.84801 | 98.12851% | 98.12851% | 0–4 raw / 0.5–0.51562 norm: 9,444 (98.2%) |
| `osc_3_level` | 9,618 | 0–1 | 0.06315 / 0.69798 / 0.7897 | 51.98586% | 15.71013% | 0.69597–0.70711 raw / 0.48438–0.5 norm: 5,027 (52.3%) |
| `osc_3_pan` | 9,618 | -1–1 | 0.00105 / 0.01552 / 0.02999 | 96.90164% | 96.90164% | 0–0.03125 raw / 0.5–0.51562 norm: 9,346 (97.2%) |
| `osc_3_phase` | 9,618 | 0–1 | 0.01161 / 0.50711 / 0.51498 | 89.17654% | 6.56062% | 0.5–0.51562 raw / 0.5–0.51562 norm: 8,587 (89.3%) |
| `osc_3_random_phase` | 9,618 | 0–1 | 0.00501 / 0.99052 / 0.99905 | 82.38719% | 15.38781% | 0.98438–1 raw / 0.98438–1 norm: 7,925 (82.4%) |
| `osc_3_spectral_morph_amount` | 9,618 | 0–1 | 0.09313 / 0.50711 / 0.56496 | 80.60927% | 3.36868% | 0.5–0.51562 raw / 0.5–0.51562 norm: 7,810 (81.2%) |
| `osc_3_spectral_morph_phase` | 7,517 | 0–1 | 0.50046 / 0.50767 / 0.51487 | — | 0.07982% | 0.5–0.51562 raw: 7,336 (97.6%) |
| `osc_3_spectral_morph_spread` | 9,618 | -0.5–0.5 | 0.0006531 / 0.00781 / 0.01497 | 98.1701% | 98.1701% | 0–0.01562 raw / 0.5–0.51562 norm: 9,446 (98.2%) |
| `osc_3_stereo_spread` | 9,618 | 0–1 | 0.98486 / 0.99203 / 0.9992 | 98.03493% | 0.64462% | 0.98438–1 raw / 0.98438–1 norm: 9,429 (98.0%) |
| `osc_3_tune` | 9,618 | -1–1 | 0.0008608 / 0.01548 / 0.0301 | 95.86193% | 95.86193% | 0–0.03125 raw / 0.5–0.51562 norm: 9,252 (96.2%) |
| `osc_3_unison_blend` | 9,618 | 0–1 | 0.79747 / 0.80472 / 0.81197 | 96.97442% | 0.24953% | 0.79688–0.8125 raw / 0.79688–0.8125 norm: 9,331 (97.0%) |
| `osc_3_unison_detune` | 9,618 | 0–10 | 0.95373 / 4.38868 / 4.5012 | 69.82741% | 6.22791% | 4.33013–4.50694 raw / 0.1875–0.20312 norm: 6,760 (70.3%) |
| `osc_3_wave_frame` | 9,618 | 0–256 | 0.26618 / 2.66178 / 160.63 | 74.28779% | 74.28779% | 0–4 raw / 0–0.01562 norm: 7,226 (75.1%) |

**Nested wavetable-editor analysis**

Active oscillator slots contain **15,869** named stock/unresolved slots and **6,236** named non-stock/unresolved slots. The semantic non-init comparison flags **22,106** active slots as different from the canonical init descriptor; that is not a claim that their embedded payload is user-authored.

Wavetable editor component types are counted as sanitized active-component occurrences; waveform/keyframe payloads are not retained:

| Editor component type | Occurrences |
|---|---:|
| `Audio File Source` | 6,642 |
| `Frequency Filter` | 603 |
| `Line Source` | 1,986 |
| `Phase Shift` | 193 |
| `Slew Limiter` | 109 |
| `Wave Folder` | 592 |
| `Wave Source` | 13,637 |
| `Wave Warp` | 345 |
| `Wave Window` | 326 |

## Sampler

The sample oscillator's scalar controls and the sanitized stock/non-stock/unresolved content classification.

### Sampler

11 scalar parameters: 8 categorical/enum and 3 continuous. enabled 35.4% (3,405/9,618); changed 48.7% (4,684/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `sample_bounce` | 9,618 | 2 | 0 (Off): 9,312 (96.8%); 1 (On): 306 (3.2%) |
| `sample_destination` | 9,618 | 5 | 3 (EFFECTS): 7,103 (73.9%); 0 (FILTER 1): 1,246 (13.0%); 1 (FILTER 2): 785 (8.2%) |
| `sample_keytrack` | 9,618 | 2 | 0 (Off): 8,279 (86.1%); 1 (On): 1,339 (13.9%) |
| `sample_loop` | 9,618 | 2 | 1 (On): 8,943 (93.0%); 0 (Off): 675 (7.0%) |
| `sample_on` | 9,618 | 2 | 0 (Off): 6,213 (64.6%); 1 (On): 3,405 (35.4%) |
| `sample_random_phase` | 9,618 | 2 | 0 (Off): 9,114 (94.8%); 1 (On): 504 (5.2%) |
| `sample_transpose` | 9,618 | 87 | 0: 8,440 (87.8%); 12: 238 (2.5%); -12: 182 (1.9%) |
| `sample_transpose_quantize` | 9,618 | 28 | 0: 9,570 (99.5%); 1: 11 (0.1%); 4096: 6 (0.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `sample_level` | 9,618 | 0–1 | 0.05421 / 0.69793 / 0.70694 | 55.42732% | 18.94365% | 0.69597–0.70711 raw / 0.48438–0.5 norm: 5,346 (55.6%) |
| `sample_pan` | 9,618 | -1–1 | 0.00137 / 0.01557 / 0.02978 | 98.66916% | 98.66916% | 0–0.03125 raw / 0.5–0.51562 norm: 9,518 (99.0%) |
| `sample_tune` | 9,618 | -1–1 | 0.00143 / 0.01565 / 0.02987 | 98.74194% | 98.74194% | 0–0.03125 raw / 0.5–0.51562 norm: 9,510 (98.9%) |

**Nested sample-content analysis**

| Sample state | Presets |
|---|---:|
| `disabled` | 6,213 |
| `named_nonstock_or_unresolved_content` | 869 |
| `named_stock_or_unresolved_content` | 2,536 |

## Filters

The two main filters and their per-filter model, style, cutoff, resonance, drive, blend, and routing controls.

### Filter 1

16 scalar parameters: 4 categorical/enum and 12 continuous. enabled 74.0% (7,118/9,618); changed 81.0% (7,793/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `filter_1_filter_input` | 9,618 | 2 | 0 (Off): 9,287 (96.6%); 1 (On): 331 (3.4%) |
| `filter_1_model` | 9,618 | 8 | 0 (Analog): 6,971 (72.5%); 6 (Comb): 667 (6.9%); 3 (Digital): 558 (5.8%) |
| `filter_1_on` | 9,618 | 2 | 1 (On): 7,118 (74.0%); 0 (Off): 2,500 (26.0%) |
| `filter_1_style` | 9,618 | 6 | 0 (12dB): 6,278 (65.3%); 1 (24dB): 2,515 (26.1%); 2 (Notch Blend): 275 (2.9%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `filter_1_blend` | 9,618 | 0–2 | 0.00197 / 0.0197 / 1.9756 | 78.49865% | 78.49865% | 0–0.03125 raw / 0–0.01562 norm: 7,626 (79.3%) |
| `filter_1_blend_transpose` | 9,618 | 0–84 | 42.02709 / 42.65113 / 43.27517 | 94.52069% | 0.85257% | 42–43.3125 raw / 0.5–0.51562 norm: 9,102 (94.6%) |
| `filter_1_cutoff` | 9,618 | 8–136 | 16.3425 / 61.22698 / 122.62308 | 27.71886% | 0% | 60–62 raw / 0.40625–0.42188 norm: 2,824 (29.4%) |
| `filter_1_drive` | 9,618 | 0–20 | 0.02124 / 0.21242 / 16.90264 | 72.37471% | 72.37471% | 0–0.3125 raw / 0–0.01562 norm: 7,074 (73.5%) |
| `filter_1_formant_resonance` | 9,618 | 0.3–1 | 0.84731 / 0.85232 / 0.85733 | 98.21169% | 0% | 0.84688–0.85781 raw / 0.78125–0.79688 norm: 9,447 (98.2%) |
| `filter_1_formant_spread` | 9,618 | -1–1 | 0.00137 / 0.01559 / 0.02982 | 98.84591% | 98.84591% | 0–0.03125 raw / 0.5–0.51562 norm: 9,507 (98.8%) |
| `filter_1_formant_transpose` | 9,618 | -12–12 | 0.01374 / 0.1862 / 0.35867 | 97.82699% | 97.82699% | 0–0.375 raw / 0.5–0.51562 norm: 9,410 (97.8%) |
| `filter_1_formant_x` | 9,618 | 0–1 | 0.50055 / 0.5078 / 0.51505 | 96.89125% | 0.39509% | 0.5–0.51562 raw / 0.5–0.51562 norm: 9,328 (97.0%) |
| `filter_1_formant_y` | 9,618 | 0–1 | 0.5006 / 0.50781 / 0.51501 | 97.50468% | 0.56145% | 0.5–0.51562 raw / 0.5–0.51562 norm: 9,386 (97.6%) |
| `filter_1_keytrack` | 9,618 | -1–1 | 0.00126 / 0.01881 / 0.98795 | 80.08942% | 80.08942% | 0–0.03125 raw / 0.5–0.51562 norm: 7,705 (80.1%) |
| `filter_1_mix` | 9,618 | 0–1 | 0.70299 / 0.9915 / 0.99915 | 91.14161% | 1.51799% | 0.98438–1 raw / 0.98438–1 norm: 8,838 (91.9%) |
| `filter_1_resonance` | 9,618 | 0–1 | 0.00287 / 0.36555 / 0.77753 | 29.4136% | 25.88896% | 0.5–0.51562 raw / 0.5–0.51562 norm: 2,897 (30.1%) |

### Filter 2

16 scalar parameters: 4 categorical/enum and 12 continuous. enabled 44.6% (4,294/9,618); changed 52.3% (5,035/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `filter_2_filter_input` | 9,618 | 2 | 0 (Off): 8,131 (84.5%); 1 (On): 1,487 (15.5%) |
| `filter_2_model` | 9,618 | 8 | 0 (Analog): 7,575 (78.8%); 6 (Comb): 592 (6.2%); 3 (Digital): 364 (3.8%) |
| `filter_2_on` | 9,618 | 2 | 0 (Off): 5,324 (55.4%); 1 (On): 4,294 (44.6%) |
| `filter_2_style` | 9,618 | 6 | 0 (12dB): 7,382 (76.8%); 1 (24dB): 1,405 (14.6%); 2 (Notch Blend): 283 (2.9%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `filter_2_blend` | 9,618 | 0–2 | 0.00197 / 0.01971 / 1.97852 | 78.70659% | 78.70659% | 0–0.03125 raw / 0–0.01562 norm: 7,623 (79.3%) |
| `filter_2_blend_transpose` | 9,618 | 0–84 | 42.02355 / 42.64547 / 43.2674 | 94.92618% | 0.99813% | 42–43.3125 raw / 0.5–0.51562 norm: 9,133 (95.0%) |
| `filter_2_cutoff` | 9,618 | 8–136 | 30.22833 / 61.16829 / 111.72041 | 54.32522% | 0% | 60–62 raw / 0.40625–0.42188 norm: 5,342 (55.5%) |
| `filter_2_drive` | 9,618 | 0–20 | 0.01851 / 0.18512 / 12.28217 | 83.72843% | 83.72843% | 0–0.3125 raw / 0–0.01562 norm: 8,117 (84.4%) |
| `filter_2_formant_resonance` | 9,618 | 0.3–1 | 0.84731 / 0.85232 / 0.85732 | 98.34685% | 0% | 0.84688–0.85781 raw / 0.78125–0.79688 norm: 9,459 (98.3%) |
| `filter_2_formant_spread` | 9,618 | -1–1 | 0.00135 / 0.01556 / 0.02978 | 98.91869% | 98.91869% | 0–0.03125 raw / 0.5–0.51562 norm: 9,514 (98.9%) |
| `filter_2_formant_transpose` | 9,618 | -12–12 | 0.01463 / 0.18672 / 0.35882 | 98.04533% | 98.04533% | 0–0.375 raw / 0.5–0.51562 norm: 9,430 (98.0%) |
| `filter_2_formant_x` | 9,618 | 0–1 | 0.50057 / 0.50777 / 0.51498 | 97.50468% | 0.40549% | 0.5–0.51562 raw / 0.5–0.51562 norm: 9,386 (97.6%) |
| `filter_2_formant_y` | 9,618 | 0–1 | 0.50061 / 0.50779 / 0.51497 | 97.89977% | 0.49906% | 0.5–0.51562 raw / 0.5–0.51562 norm: 9,420 (97.9%) |
| `filter_2_keytrack` | 9,618 | -1–1 | 0.00137 / 0.01765 / 0.98412 | 86.31732% | 86.31732% | 0–0.03125 raw / 0.5–0.51562 norm: 8,304 (86.3%) |
| `filter_2_mix` | 9,618 | 0–1 | 0.68185 / 0.99153 / 0.99915 | 91.77584% | 1.65315% | 0.98438–1 raw / 0.98438–1 norm: 8,869 (92.2%) |
| `filter_2_resonance` | 9,618 | 0–1 | 0.00439 / 0.50428 / 0.68237 | 56.13433% | 16.95779% | 0.5–0.51562 raw / 0.5–0.51562 norm: 5,454 (56.7%) |

## Envelopes

Six envelopes, with quartic-time controls, sustain, and operational routing prevalence.

### Envelope 1

9 scalar parameters: 0 categorical/enum and 9 continuous. routed 15.1% (1,451/9,618); changed 0.0% (0/9,618).

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `env_1_attack` | 9,618 | 0–2.37842 | 0.40377 / 0.71801 / 0.90941 | 46.11146% | 12.94448% | 0–0.8409 raw / 0–0.01562 norm: 9,046 (94.1%) |
| `env_1_attack_power` | 9,618 | -20–20 | 0.0084 / 0.32762 / 2.31503 | 87.46101% | 87.46101% | 0–0.625 raw / 0.5–0.51562 norm: 8,473 (88.1%) |
| `env_1_decay` | 9,618 | 0–2.37842 | 0.58183 / 0.92252 / 1.23055 | 46.60012% | 1.99626% | 0.8409–1 raw / 0.01562–0.03125 norm: 6,043 (62.8%) |
| `env_1_decay_power` | 9,618 | -20–20 | -3.62057 / -2.17764 / 0.77096 | 79.49678% | 0.84217% | -2.5–-1.875 raw / 0.4375–0.45312 norm: 7,826 (81.4%) |
| `env_1_delay` | 9,618 | 0–0.77379 | 0.23648 / 0.42052 / 0.49372 | 97.75421% | 97.75421% | 0–0.5 raw / 0–0.01562 norm: 9,610 (99.9%) |
| `env_1_hold` | 9,618 | 0–1.41421 | 0.2385 / 0.42412 / 0.49794 | 93.85527% | 93.85527% | 0–0.5 raw / 0–0.01562 norm: 9,288 (96.6%) |
| `env_1_release` | 9,618 | 0–2.37842 | 0.42524 / 0.75619 / 1.22996 | 39.23893% | 7.49636% | 0–0.8409 raw / 0–0.01562 norm: 7,353 (76.5%) |
| `env_1_release_power` | 9,618 | -20–20 | -2.4946 / -2.18223 / -1.43472 | 88.86463% | 0.7174% | -2.5–-1.875 raw / 0.4375–0.45312 norm: 8,659 (90.0%) |
| `env_1_sustain` | 9,618 | 0–1 | 0.00503 / 0.9861 / 0.99861 | 55.67686% | 14.83676% | 0.98438–1 raw / 0.98438–1 norm: 5,407 (56.2%) |

### Envelope 2

9 scalar parameters: 0 categorical/enum and 9 continuous. routed 30.4% (2,926/9,618); changed 0.0% (0/9,618).

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `env_2_attack` | 9,618 | 0–2.37842 | 0.40127 / 0.71358 / 0.83778 | 77.89561% | 7.39239% | 0–0.8409 raw / 0–0.01562 norm: 9,273 (96.4%) |
| `env_2_attack_power` | 9,618 | -20–20 | 0.01711 / 0.31356 / 0.61001 | 94.65585% | 94.65585% | 0–0.625 raw / 0.5–0.51562 norm: 9,124 (94.9%) |
| `env_2_decay` | 9,618 | 0–2.37842 | 0.5797 / 0.91341 / 1.07899 | 63.9842% | 0.97733% | 0.8409–1 raw / 0.01562–0.03125 norm: 6,833 (71.0%) |
| `env_2_decay_power` | 9,618 | -20–20 | -3.39369 / -2.19617 / -1.87808 | 87.44022% | 0.64462% | -2.5–-1.875 raw / 0.4375–0.45312 norm: 8,503 (88.4%) |
| `env_2_delay` | 9,618 | 0–1.41421 | 0.23651 / 0.42058 / 0.49378 | 98.67956% | 98.67956% | 0–0.5 raw / 0–0.01562 norm: 9,605 (99.9%) |
| `env_2_hold` | 9,618 | 0–1.41421 | 0.237 / 0.42145 / 0.4948 | 97.7854% | 97.7854% | 0–0.5 raw / 0–0.01562 norm: 9,526 (99.0%) |
| `env_2_release` | 9,618 | 0–2.37842 | 0.40721 / 0.72413 / 1.04545 | 78.43627% | 4.01331% | 0–0.8409 raw / 0–0.01562 norm: 8,744 (90.9%) |
| `env_2_release_power` | 9,618 | -20–19.18002 | -2.47232 / -2.18565 / -1.89897 | 97.73342% | 0.16635% | -2.5–-1.875 raw / 0.4375–0.45312 norm: 9,435 (98.1%) |
| `env_2_sustain` | 9,618 | 0–1 | 0.00354 / 0.98798 / 0.9988 | 64.95113% | 21.28301% | 0.98438–1 raw / 0.98438–1 norm: 6,253 (65.0%) |

### Envelope 3

9 scalar parameters: 0 categorical/enum and 9 continuous. routed 11.8% (1,135/9,618); changed 0.0% (0/9,618).

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `env_3_attack` | 9,618 | 0–2.02273 | 0.39941 / 0.71027 / 0.83389 | 91.12082% | 3.04637% | 0–0.8409 raw / 0–0.01562 norm: 9,447 (98.2%) |
| `env_3_attack_power` | 9,618 | -19.04001–15.53999 | 0.02601 / 0.31323 / 0.60045 | 97.84779% | 97.84779% | 0–0.625 raw / 0.5–0.51562 norm: 9,417 (97.9%) |
| `env_3_decay` | 9,618 | 0–2.37842 | 0.70343 / 0.92362 / 0.99631 | 84.90331% | 0.34311% | 0.8409–1 raw / 0.01562–0.03125 norm: 8,401 (87.3%) |
| `env_3_decay_power` | 9,618 | -20–14.98 | -2.483 / -2.18964 / -1.89628 | 95.62279% | 0.15596% | -2.5–-1.875 raw / 0.4375–0.45312 norm: 9,220 (95.9%) |
| `env_3_delay` | 9,618 | 0–1.41421 | 0.23656 / 0.42068 / 0.4939 | 99.15783% | 99.15783% | 0–0.5 raw / 0–0.01562 norm: 9,596 (99.8%) |
| `env_3_hold` | 9,618 | 0–1.41421 | 0.23665 / 0.42083 / 0.49408 | 99.20981% | 99.20981% | 0–0.5 raw / 0–0.01562 norm: 9,582 (99.6%) |
| `env_3_release` | 9,618 | 0–2.37842 | 0.4014 / 0.71381 / 0.83805 | 91.15201% | 2.17301% | 0–0.8409 raw / 0–0.01562 norm: 9,261 (96.3%) |
| `env_3_release_power` | 9,618 | -20–14.68004 | -2.47044 / -2.18737 / -1.90429 | 99.22021% | 0.0104% | -2.5–-1.875 raw / 0.4375–0.45312 norm: 9,555 (99.3%) |
| `env_3_sustain` | 9,618 | 0–1 | 0.00788 / 0.9908 / 0.99908 | 84.9137% | 9.48222% | 0.98438–1 raw / 0.98438–1 norm: 8,169 (84.9%) |

### Envelope 4

9 scalar parameters: 0 categorical/enum and 9 continuous. routed 3.8% (368/9,618); changed 0.0% (0/9,618).

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `env_4_attack` | 9,618 | 0–1.89539 | 0.39836 / 0.7084 / 0.8317 | 96.83926% | 0.92535% | 0–0.8409 raw / 0–0.01562 norm: 9,547 (99.3%) |
| `env_4_attack_power` | 9,618 | -20–8.04 | 0.02763 / 0.31178 / 0.59592 | 98.94989% | 98.94989% | 0–0.625 raw / 0.5–0.51562 norm: 9,519 (99.0%) |
| `env_4_decay` | 9,618 | 0–2.00304 | 0.84372 / 0.92815 / 0.99433 | 94.64546% | 0.10397% | 0.8409–1 raw / 0.01562–0.03125 norm: 9,193 (95.6%) |
| `env_4_decay_power` | 9,618 | -20–18.28 | -2.47421 / -2.18862 / -1.90304 | 98.43003% | 0.07278% | -2.5–-1.875 raw / 0.4375–0.45312 norm: 9,471 (98.5%) |
| `env_4_delay` | 9,618 | 0–0.8409 | 0.23646 / 0.42049 / 0.49368 | 99.66729% | 99.66729% | 0–0.5 raw / 0–0.01562 norm: 9,613 (99.9%) |
| `env_4_hold` | 9,618 | 0–1.05511 | 0.23649 / 0.42055 / 0.49374 | 99.69848% | 99.69848% | 0–0.5 raw / 0–0.01562 norm: 9,608 (99.9%) |
| `env_4_release` | 9,618 | 0–2.37842 | 0.39897 / 0.70948 / 0.83297 | 96.54814% | 1.06051% | 0–0.8409 raw / 0–0.01562 norm: 9,489 (98.7%) |
| `env_4_release_power` | 9,618 | -20–4.74531 | -2.46915 / -2.18737 / -1.90559 | 99.79206% | 0% | -2.5–-1.875 raw / 0.4375–0.45312 norm: 9,599 (99.8%) |
| `env_4_sustain` | 9,618 | 0–1 | 0.4952 / 0.99174 / 0.99917 | 94.59347% | 3.6598% | 0.98438–1 raw / 0.98438–1 norm: 9,099 (94.6%) |

### Envelope 5

9 scalar parameters: 0 categorical/enum and 9 continuous. routed 1.5% (145/9,618); changed 0.0% (0/9,618).

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `env_5_attack` | 9,618 | 0–1.57574 | 0.39794 / 0.70764 / 0.83081 | 98.68996% | 0.43668% | 0–0.8409 raw / 0–0.01562 norm: 9,588 (99.7%) |
| `env_5_attack_power` | 9,618 | -19.4–14.82 | 0.03 / 0.31237 / 0.59474 | 99.59451% | 99.59451% | 0–0.625 raw / 0.5–0.51562 norm: 9,579 (99.6%) |
| `env_5_decay` | 9,618 | 0–2.37842 | 0.84814 / 0.92964 / 0.99404 | 97.7854% | 0.13516% | 0.8409–1 raw / 0.01562–0.03125 norm: 9,430 (98.0%) |
| `env_5_decay_power` | 9,618 | -20–20 | -2.47153 / -2.18819 / -1.90485 | 99.23061% | 0.0104% | -2.5–-1.875 raw / 0.4375–0.45312 norm: 9,546 (99.3%) |
| `env_5_delay` | 9,618 | 0–0.98885 | 0.2365 / 0.42057 / 0.49377 | 99.84404% | 99.84404% | 0–0.5 raw / 0–0.01562 norm: 9,606 (99.9%) |
| `env_5_hold` | 9,618 | 0–0.99437 | 0.23648 / 0.42052 / 0.49372 | 99.80245% | 99.80245% | 0–0.5 raw / 0–0.01562 norm: 9,610 (99.9%) |
| `env_5_release` | 9,618 | 0–1.89601 | 0.39811 / 0.70796 / 0.83118 | 98.70035% | 0.42628% | 0–0.8409 raw / 0–0.01562 norm: 9,571 (99.5%) |
| `env_5_release_power` | 9,618 | -16.28–1.42 | -2.46892 / -2.18747 / -1.90601 | 99.90643% | 0.0104% | -2.5–-1.875 raw / 0.4375–0.45312 norm: 9,610 (99.9%) |
| `env_5_sustain` | 9,618 | 0–1 | 0.98483 / 0.99201 / 0.9992 | 97.83739% | 1.4764% | 0.98438–1 raw / 0.98438–1 norm: 9,410 (97.8%) |

### Envelope 6

9 scalar parameters: 0 categorical/enum and 9 continuous. routed 0.6% (53/9,618); changed 0.0% (0/9,618).

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `env_6_attack` | 9,618 | 0–1.25094 | 0.39778 / 0.70737 / 0.83049 | 99.30339% | 0.21834% | 0–0.8409 raw / 0–0.01562 norm: 9,603 (99.8%) |
| `env_6_attack_power` | 9,618 | -12.30001–13.86 | 0.03064 / 0.3123 / 0.59396 | 99.84404% | 99.84404% | 0–0.625 raw / 0.5–0.51562 norm: 9,603 (99.8%) |
| `env_6_decay` | 9,618 | 0–1.4079 | 0.85007 / 0.93018 / 0.99372 | 99.23061% | 0.0104% | 0.8409–1 raw / 0.01562–0.03125 norm: 9,555 (99.3%) |
| `env_6_decay_power` | 9,618 | -11.91353–8.62 | -2.46959 / -2.1877 / -1.9058 | 99.74007% | 0% | -2.5–-1.875 raw / 0.4375–0.45312 norm: 9,595 (99.8%) |
| `env_6_delay` | 9,618 | 0–0.57901 | 0.23644 / 0.42045 / 0.49363 | 99.91682% | 99.91682% | 0–0.5 raw / 0–0.01562 norm: 9,617 (100.0%) |
| `env_6_hold` | 9,618 | 0–1.22729 | 0.23644 / 0.42046 / 0.49364 | 99.96881% | 99.96881% | 0–0.5 raw / 0–0.01562 norm: 9,616 (100.0%) |
| `env_6_release` | 9,618 | 0–2.37842 | 0.39778 / 0.70737 / 0.83049 | 99.55292% | 0.17675% | 0–0.8409 raw / 0–0.01562 norm: 9,603 (99.8%) |
| `env_6_release_power` | 9,618 | -2–0.64 | -2.46874 / -2.18743 / -1.90613 | 99.96881% | 0% | -2.5–-1.875 raw / 0.4375–0.45312 norm: 9,615 (100.0%) |
| `env_6_sustain` | 9,618 | 0–1 | 0.98505 / 0.99213 / 0.99921 | 99.29299% | 0.60304% | 0.98438–1 raw / 0.98438–1 norm: 9,550 (99.3%) |

## LFOs

Eight drawable LFOs, including rate/sync controls and nested custom-shape prevalence.

### LFO 1

12 scalar parameters: 5 categorical/enum and 7 continuous. routed 62.0% (5,960/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `lfo_1_keytrack_transpose` | 9,618 | 25 | -12: 9,449 (98.2%); 0: 54 (0.6%); -60: 22 (0.2%) |
| `lfo_1_smooth_mode` | 9,618 | 2 | 1 (On): 9,556 (99.4%); 0 (Off): 62 (0.6%) |
| `lfo_1_sync` | 9,618 | 5 | 1 (Tempo): 8,097 (84.2%); 0 (Seconds): 1,087 (11.3%); 4 (Keytrack): 160 (1.7%) |
| `lfo_1_sync_type` | 9,618 | 7 | 0 (Trigger): 6,374 (66.3%); 2 (Envelope): 1,780 (18.5%); 1 (Sync): 1,324 (13.8%) |
| `lfo_1_tempo` | 9,618 | 13 | 7 (1/2): 4,716 (49.0%); 8 (1/4): 1,138 (11.8%); 6 (1/1): 922 (9.6%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `lfo_1_delay_time` | 9,618 | 0–4 | 0.00316 / 0.03159 / 0.06002 | 98.33645% | 98.33645% | 0–0.0625 raw / 0–0.01562 norm: 9,513 (98.9%) |
| `lfo_1_fade_time` | 9,618 | 0–8 | 0.00626 / 0.06264 / 0.11902 | 99.75047% | 99.75047% | 0–0.125 raw / 0–0.01562 norm: 9,595 (99.8%) |
| `lfo_1_frequency` | 9,618 | -7–9 | -0.70509 / 1.12287 / 1.67341 | 86.73321% | 0.13516% | 1–1.25 raw / 0.5–0.51562 norm: 8,378 (87.1%) |
| `lfo_1_keytrack_tune` | 9,618 | -1–0.26 | 0.00154 / 0.01562 / 0.0297 | 99.84404% | 99.84404% | 0–0.03125 raw / 0.5–0.51562 norm: 9,606 (99.9%) |
| `lfo_1_phase` | 9,618 | 0–1 | 0.0007997 / 0.008 / 0.01519 | 96.71449% | 96.71449% | 0–0.01562 raw / 0–0.01562 norm: 9,395 (97.7%) |
| `lfo_1_smooth_time` | 9,618 | -10–4 | -9.94793 / -7.51329 / -6.63464 | 69.70264% | 0.03119% | -7.59375–-7.375 raw / 0.17188–0.1875 norm: 6,738 (70.1%) |
| `lfo_1_stereo` | 9,618 | -0.5–0.5 | 0.0006987 / 0.00797 / 0.01524 | 96.64171% | 96.64171% | 0–0.01562 raw / 0.5–0.51562 norm: 9,299 (96.7%) |

### LFO 2

12 scalar parameters: 5 categorical/enum and 7 continuous. routed 39.8% (3,826/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `lfo_2_keytrack_transpose` | 9,618 | 24 | -12: 9,506 (98.8%); 0: 36 (0.4%); 12: 25 (0.3%) |
| `lfo_2_smooth_mode` | 9,618 | 2 | 1 (On): 9,568 (99.5%); 0 (Off): 50 (0.5%) |
| `lfo_2_sync` | 9,618 | 5 | 1 (Tempo): 8,629 (89.7%); 0 (Seconds): 695 (7.2%); 2 (Tempo Dotted): 115 (1.2%) |
| `lfo_2_sync_type` | 9,618 | 7 | 0 (Trigger): 7,360 (76.5%); 1 (Sync): 1,131 (11.8%); 2 (Envelope): 1,029 (10.7%) |
| `lfo_2_tempo` | 9,618 | 14 | 7 (1/2): 6,373 (66.3%); 6 (1/1): 689 (7.2%); 8 (1/4): 638 (6.6%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `lfo_2_delay_time` | 9,618 | 0–4 | 0.00314 / 0.03144 / 0.05974 | 99.14743% | 99.14743% | 0–0.0625 raw / 0–0.01562 norm: 9,559 (99.4%) |
| `lfo_2_fade_time` | 9,618 | 0–4.08 | 0.00626 / 0.06262 / 0.11897 | 99.79206% | 99.79206% | 0–0.125 raw / 0–0.01562 norm: 9,599 (99.8%) |
| `lfo_2_frequency` | 9,618 | -7–9 | 1.00059 / 1.12242 / 1.24426 | 92.13974% | 0.09357% | 1–1.25 raw / 0.5–0.51562 norm: 8,880 (92.3%) |
| `lfo_2_keytrack_tune` | 9,618 | -0.03631–1 | 0.00155 / 0.01562 / 0.02969 | 99.94801% | 99.94801% | 0–0.03125 raw / 0.5–0.51562 norm: 9,612 (99.9%) |
| `lfo_2_phase` | 9,618 | 0–1 | 0.0007911 / 0.00791 / 0.01503 | 98.34685% | 98.34685% | 0–0.01562 raw / 0–0.01562 norm: 9,497 (98.7%) |
| `lfo_2_smooth_time` | 9,618 | -10–4 | -9.92372 / -7.50148 / -7.37975 | 80.66126% | 0.02079% | -7.59375–-7.375 raw / 0.17188–0.1875 norm: 7,777 (80.9%) |
| `lfo_2_stereo` | 9,618 | -0.5–0.5 | 0.0007287 / 0.0079 / 0.01507 | 98.05573% | 98.05573% | 0–0.01562 raw / 0.5–0.51562 norm: 9,431 (98.1%) |

### LFO 3

12 scalar parameters: 5 categorical/enum and 7 continuous. routed 23.7% (2,282/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `lfo_3_keytrack_transpose` | 9,618 | 18 | -12: 9,543 (99.2%); 12: 18 (0.2%); 0: 14 (0.1%) |
| `lfo_3_smooth_mode` | 9,618 | 2 | 1 (On): 9,564 (99.4%); 0 (Off): 54 (0.6%) |
| `lfo_3_sync` | 9,618 | 5 | 1 (Tempo): 8,955 (93.1%); 0 (Seconds): 480 (5.0%); 2 (Tempo Dotted): 73 (0.8%) |
| `lfo_3_sync_type` | 9,618 | 7 | 0 (Trigger): 8,216 (85.4%); 1 (Sync): 748 (7.8%); 2 (Envelope): 598 (6.2%) |
| `lfo_3_tempo` | 9,618 | 13 | 7 (1/2): 7,646 (79.5%); 6 (1/1): 397 (4.1%); 8 (1/4): 378 (3.9%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `lfo_3_delay_time` | 9,618 | 0–4 | 0.00314 / 0.03135 / 0.05957 | 99.55292% | 99.55292% | 0–0.0625 raw / 0–0.01562 norm: 9,585 (99.7%) |
| `lfo_3_fade_time` | 9,618 | 0–4.98899 | 0.00626 / 0.06257 / 0.11887 | 99.87523% | 99.87523% | 0–0.125 raw / 0–0.01562 norm: 9,607 (99.9%) |
| `lfo_3_frequency` | 9,618 | -7–9 | 1.00326 / 1.122 / 1.24073 | 94.72863% | 0.02079% | 1–1.25 raw / 0.5–0.51562 norm: 9,112 (94.7%) |
| `lfo_3_keytrack_tune` | 9,618 | 0–0.98294 | 0.00156 / 0.01563 / 0.02969 | 99.95841% | 99.95841% | 0–0.03125 raw / 0.5–0.51562 norm: 9,615 (100.0%) |
| `lfo_3_phase` | 9,618 | 0–0.95735 | 0.0007857 / 0.00786 / 0.01493 | 99.2722% | 99.2722% | 0–0.01562 raw / 0–0.01562 norm: 9,563 (99.4%) |
| `lfo_3_smooth_time` | 9,618 | -10–4 | -9.89471 / -7.49681 / -7.38331 | 86.57725% | 0.02079% | -7.59375–-7.375 raw / 0.17188–0.1875 norm: 8,341 (86.7%) |
| `lfo_3_stereo` | 9,618 | -0.5–0.5 | 0.0007564 / 0.00789 / 0.01502 | 98.55479% | 98.55479% | 0–0.01562 raw / 0.5–0.51562 norm: 9,479 (98.6%) |

### LFO 4

12 scalar parameters: 5 categorical/enum and 7 continuous. routed 13.6% (1,307/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `lfo_4_keytrack_transpose` | 9,618 | 12 | -12: 9,562 (99.4%); 0: 16 (0.2%); -5: 8 (0.1%) |
| `lfo_4_smooth_mode` | 9,618 | 2 | 1 (On): 9,567 (99.5%); 0 (Off): 51 (0.5%) |
| `lfo_4_sync` | 9,618 | 5 | 1 (Tempo): 9,203 (95.7%); 0 (Seconds): 273 (2.8%); 2 (Tempo Dotted): 68 (0.7%) |
| `lfo_4_sync_type` | 9,618 | 6 | 0 (Trigger): 8,746 (90.9%); 1 (Sync): 515 (5.4%); 2 (Envelope): 324 (3.4%) |
| `lfo_4_tempo` | 9,618 | 13 | 7 (1/2): 8,421 (87.6%); 5 (2/1): 231 (2.4%); 6 (1/1): 229 (2.4%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `lfo_4_delay_time` | 9,618 | 0–4 | 0.00314 / 0.03136 / 0.05959 | 99.55292% | 99.55292% | 0–0.0625 raw / 0–0.01562 norm: 9,583 (99.6%) |
| `lfo_4_fade_time` | 9,618 | 0–8 | 0.00625 / 0.06251 / 0.11877 | 99.95841% | 99.95841% | 0–0.125 raw / 0–0.01562 norm: 9,615 (100.0%) |
| `lfo_4_frequency` | 9,618 | -7–9 | 1.00621 / 1.1227 / 1.23918 | 96.56893% | 0.02079% | 1–1.25 raw / 0.5–0.51562 norm: 9,288 (96.6%) |
| `lfo_4_keytrack_tune` | 9,618 | -0.02651–1 | 0.00154 / 0.01562 / 0.02969 | 99.89603% | 99.89603% | 0–0.03125 raw / 0.5–0.51562 norm: 9,608 (99.9%) |
| `lfo_4_phase` | 9,618 | 0–1 | 0.0007854 / 0.00785 / 0.01492 | 99.24101% | 99.24101% | 0–0.01562 raw / 0–0.01562 norm: 9,566 (99.5%) |
| `lfo_4_smooth_time` | 9,618 | -10–4 | -9.85994 / -7.49366 / -7.38476 | 90.30984% | 0% | -7.59375–-7.375 raw / 0.17188–0.1875 norm: 8,693 (90.4%) |
| `lfo_4_stereo` | 9,618 | -0.5–0.5 | 0.0007692 / 0.00785 / 0.01494 | 99.23061% | 99.23061% | 0–0.01562 raw / 0.5–0.51562 norm: 9,544 (99.2%) |

### LFO 5

12 scalar parameters: 5 categorical/enum and 7 continuous. routed 7.7% (742/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `lfo_5_keytrack_transpose` | 9,618 | 12 | -12: 9,546 (99.3%); 0: 24 (0.2%); 12: 18 (0.2%) |
| `lfo_5_smooth_mode` | 9,618 | 2 | 1 (On): 9,576 (99.6%); 0 (Off): 42 (0.4%) |
| `lfo_5_sync` | 9,618 | 5 | 1 (Tempo): 9,350 (97.2%); 0 (Seconds): 169 (1.8%); 4 (Keytrack): 56 (0.6%) |
| `lfo_5_sync_type` | 9,618 | 6 | 0 (Trigger): 9,121 (94.8%); 1 (Sync): 308 (3.2%); 2 (Envelope): 178 (1.9%) |
| `lfo_5_tempo` | 9,618 | 13 | 7 (1/2): 8,842 (91.9%); 4 (4/1): 161 (1.7%); 6 (1/1): 159 (1.7%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `lfo_5_delay_time` | 9,618 | 0–3.5 | 0.00313 / 0.03127 / 0.05941 | 99.91682% | 99.91682% | 0–0.0625 raw / 0–0.01562 norm: 9,611 (99.9%) |
| `lfo_5_fade_time` | 9,618 | 0–3.26849 | 0.00626 / 0.06256 / 0.11886 | 99.88563% | 99.88563% | 0–0.125 raw / 0–0.01562 norm: 9,608 (99.9%) |
| `lfo_5_frequency` | 9,618 | -7–9 | 1.00889 / 1.12378 / 1.23867 | 97.82699% | 0.0104% | 1–1.25 raw / 0.5–0.51562 norm: 9,417 (97.9%) |
| `lfo_5_keytrack_tune` | 9,618 | -0.01044–0 | 0.00156 / 0.01562 / 0.02968 | 99.9896% | 99.9896% | 0–0.03125 raw / 0.5–0.51562 norm: 9,617 (100.0%) |
| `lfo_5_phase` | 9,618 | 0–0.87411 | 0.0007826 / 0.00783 / 0.01487 | 99.76087% | 99.76087% | 0–0.01562 raw / 0–0.01562 norm: 9,600 (99.8%) |
| `lfo_5_smooth_time` | 9,618 | -10–4 | -9.81927 / -7.49113 / -7.38516 | 92.81555% | 0% | -7.59375–-7.375 raw / 0.17188–0.1875 norm: 8,933 (92.9%) |
| `lfo_5_stereo` | 9,618 | -0.5–0.5 | 0.0007661 / 0.00782 / 0.01488 | 99.6361% | 99.6361% | 0–0.01562 raw / 0.5–0.51562 norm: 9,583 (99.6%) |

### LFO 6

12 scalar parameters: 5 categorical/enum and 7 continuous. routed 4.5% (434/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `lfo_6_keytrack_transpose` | 9,618 | 8 | -12: 9,575 (99.6%); 12: 13 (0.1%); 0: 9 (0.1%) |
| `lfo_6_smooth_mode` | 9,618 | 2 | 1 (On): 9,570 (99.5%); 0 (Off): 48 (0.5%) |
| `lfo_6_sync` | 9,618 | 5 | 1 (Tempo): 9,460 (98.4%); 0 (Seconds): 100 (1.0%); 4 (Keytrack): 36 (0.4%) |
| `lfo_6_sync_type` | 9,618 | 4 | 0 (Trigger): 9,311 (96.8%); 1 (Sync): 170 (1.8%); 2 (Envelope): 136 (1.4%) |
| `lfo_6_tempo` | 9,618 | 13 | 7 (1/2): 9,167 (95.3%); 4 (4/1): 92 (1.0%); 6 (1/1): 85 (0.9%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `lfo_6_delay_time` | 9,618 | 0–4 | 0.00313 / 0.03127 / 0.05941 | 99.91682% | 99.91682% | 0–0.0625 raw / 0–0.01562 norm: 9,611 (99.9%) |
| `lfo_6_fade_time` | 9,618 | 0–2.80814 | 0.00625 / 0.06251 / 0.11877 | 99.96881% | 99.96881% | 0–0.125 raw / 0–0.01562 norm: 9,615 (100.0%) |
| `lfo_6_frequency` | 9,618 | -5.695–5.83389 | 1.01063 / 1.12455 / 1.23847 | 98.73155% | 0.0104% | 1–1.25 raw / 0.5–0.51562 norm: 9,497 (98.7%) |
| `lfo_6_keytrack_tune` | 9,618 | 0–0.04604 | 0.00156 / 0.01563 / 0.02969 | 99.96881% | 99.96881% | 0–0.03125 raw / 0.5–0.51562 norm: 9,615 (100.0%) |
| `lfo_6_phase` | 9,618 | 0–0.61549 | 0.0007816 / 0.00782 / 0.01485 | 99.91682% | 99.91682% | 0–0.01562 raw / 0–0.01562 norm: 9,613 (99.9%) |
| `lfo_6_smooth_time` | 9,618 | -10–4 | -7.59296 / -7.48949 / -7.38602 | 95.06134% | 0% | -7.59375–-7.375 raw / 0.17188–0.1875 norm: 9,149 (95.1%) |
| `lfo_6_stereo` | 9,618 | -0.5–0.5 | 0.0007776 / 0.00782 / 0.01486 | 99.83365% | 99.83365% | 0–0.01562 raw / 0.5–0.51562 norm: 9,602 (99.8%) |

### LFO 7

12 scalar parameters: 5 categorical/enum and 7 continuous. routed 2.4% (227/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `lfo_7_keytrack_transpose` | 9,618 | 5 | -12: 9,599 (99.8%); 0: 11 (0.1%); 12: 5 (0.1%) |
| `lfo_7_smooth_mode` | 9,618 | 2 | 1 (On): 9,575 (99.6%); 0 (Off): 43 (0.4%) |
| `lfo_7_sync` | 9,618 | 5 | 1 (Tempo): 9,535 (99.1%); 0 (Seconds): 46 (0.5%); 4 (Keytrack): 26 (0.3%) |
| `lfo_7_sync_type` | 9,618 | 7 | 0 (Trigger): 9,446 (98.2%); 1 (Sync): 100 (1.0%); 2 (Envelope): 63 (0.7%) |
| `lfo_7_tempo` | 9,618 | 13 | 7 (1/2): 9,364 (97.4%); 5 (2/1): 59 (0.6%); 4 (4/1): 47 (0.5%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `lfo_7_delay_time` | 9,618 | 0–4 | 0.00313 / 0.03126 / 0.0594 | 99.94801% | 99.94801% | 0–0.0625 raw / 0–0.01562 norm: 9,613 (99.9%) |
| `lfo_7_fade_time` | 9,618 | 0–4.92575 | 0.00625 / 0.06252 / 0.11879 | 99.95841% | 99.95841% | 0–0.125 raw / 0–0.01562 norm: 9,614 (100.0%) |
| `lfo_7_frequency` | 9,618 | -5.66682–6.1275 | 1.01137 / 1.12449 / 1.23761 | 99.37617% | 0% | 1–1.25 raw / 0.5–0.51562 norm: 9,564 (99.4%) |
| `lfo_7_keytrack_tune` | 9,618 | 0–0.02457 | 0.00156 / 0.01562 / 0.02968 | 99.9896% | 99.9896% | 0–0.03125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `lfo_7_phase` | 9,618 | 0–0.50179 | 0.0007816 / 0.00782 / 0.01485 | 99.94801% | 99.94801% | 0–0.01562 raw / 0–0.01562 norm: 9,613 (99.9%) |
| `lfo_7_smooth_time` | 9,618 | -10–4 | -7.59022 / -7.48814 / -7.38606 | 96.40258% | 0% | -7.59375–-7.375 raw / 0.17188–0.1875 norm: 9,274 (96.4%) |
| `lfo_7_stereo` | 9,618 | -0.08791–0.5 | 0.0007799 / 0.00781 / 0.01485 | 99.93762% | 99.93762% | 0–0.01562 raw / 0.5–0.51562 norm: 9,613 (99.9%) |

### LFO 8

12 scalar parameters: 5 categorical/enum and 7 continuous. routed 1.3% (122/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `lfo_8_keytrack_transpose` | 9,618 | 5 | -12: 9,593 (99.7%); 12: 15 (0.2%); 0: 8 (0.1%) |
| `lfo_8_smooth_mode` | 9,618 | 2 | 1 (On): 9,617 (100.0%); 0 (Off): 1 (0.0%) |
| `lfo_8_sync` | 9,618 | 5 | 1 (Tempo): 9,556 (99.4%); 4 (Keytrack): 25 (0.3%); 0 (Seconds): 24 (0.2%) |
| `lfo_8_sync_type` | 9,618 | 6 | 0 (Trigger): 9,535 (99.1%); 1 (Sync): 44 (0.5%); 2 (Envelope): 35 (0.4%) |
| `lfo_8_tempo` | 9,618 | 13 | 7 (1/2): 9,466 (98.4%); 6 (1/1): 37 (0.4%); 5 (2/1): 30 (0.3%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `lfo_8_delay_time` | 9,618 | 0–4 | 0.00313 / 0.03126 / 0.05939 | 99.95841% | 99.95841% | 0–0.0625 raw / 0–0.01562 norm: 9,615 (100.0%) |
| `lfo_8_fade_time` | 9,618 | 0–2.30681 | 0.00625 / 0.0625 / 0.11875 | 99.9896% | 99.9896% | 0–0.125 raw / 0–0.01562 norm: 9,617 (100.0%) |
| `lfo_8_frequency` | 9,618 | -7–6.5733 | 1.01147 / 1.12454 / 1.23762 | 99.48014% | 0.0104% | 1–1.25 raw / 0.5–0.51562 norm: 9,568 (99.5%) |
| `lfo_8_keytrack_tune` | 9,618 | 0–0 | 0.00156 / 0.01562 / 0.02968 | 100% | 100% | 0–0.03125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `lfo_8_phase` | 9,618 | 0–0.625 | 0.0007815 / 0.00781 / 0.01485 | 99.95841% | 99.95841% | 0–0.01562 raw / 0–0.01562 norm: 9,614 (100.0%) |
| `lfo_8_smooth_time` | 9,618 | -10–1.32804 | -7.58822 / -7.48717 / -7.38611 | 97.35912% | 0% | -7.59375–-7.375 raw / 0.17188–0.1875 norm: 9,368 (97.4%) |
| `lfo_8_stereo` | 9,618 | 0–0.5 | 0.0007813 / 0.00781 / 0.01485 | 99.96881% | 99.96881% | 0–0.01562 raw / 0.5–0.51562 norm: 9,616 (100.0%) |

**Nested drawable-shape analysis**

Across the eight drawable LFO objects, **15,871** slots have a non-default shape and **12,470** of those are operationally routed. Display names are ignored; the comparison uses the serialized drawable fields.

## Random sources

Four random modulation sources, including style, rate, sync, stereo, and keytracking controls.

### Random source 1

8 scalar parameters: 6 categorical/enum and 2 continuous. routed 18.3% (1,756/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `random_1_keytrack_transpose` | 9,618 | 9 | -12: 9,604 (99.9%); 0: 4 (0.0%); -13: 3 (0.0%) |
| `random_1_stereo` | 9,618 | 2 | 0 (Off): 9,144 (95.1%); 1 (On): 474 (4.9%) |
| `random_1_style` | 9,618 | 4 | 0 (Perlin): 8,850 (92.0%); 1 (Sample & Hold): 495 (5.1%); 2 (Sine Interpolate): 182 (1.9%) |
| `random_1_sync` | 9,618 | 5 | 1 (Tempo): 9,070 (94.3%); 0 (Seconds): 483 (5.0%); 2 (Tempo Dotted): 32 (0.3%) |
| `random_1_sync_type` | 9,618 | 2 | 0 (Off): 9,019 (93.8%); 1 (On): 599 (6.2%) |
| `random_1_tempo` | 9,618 | 13 | 8 (1/4): 8,435 (87.7%); 9 (1/8): 222 (2.3%); 7 (1/2): 199 (2.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `random_1_frequency` | 9,618 | -7–9 | 1.00401 / 1.12225 / 1.2405 | 95.06134% | 0.02079% | 1–1.25 raw / 0.5–0.51562 norm: 9,150 (95.1%) |
| `random_1_keytrack_tune` | 9,618 | -1–0.56223 | 0.00155 / 0.01562 / 0.02969 | 99.93762% | 99.93762% | 0–0.03125 raw / 0.5–0.51562 norm: 9,612 (99.9%) |

### Random source 2

8 scalar parameters: 6 categorical/enum and 2 continuous. routed 7.9% (757/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `random_2_keytrack_transpose` | 9,618 | 2 | -12: 9,617 (100.0%); 0: 1 (0.0%) |
| `random_2_stereo` | 9,618 | 2 | 0 (Off): 9,457 (98.3%); 1 (On): 161 (1.7%) |
| `random_2_style` | 9,618 | 4 | 0 (Perlin): 9,264 (96.3%); 1 (Sample & Hold): 217 (2.3%); 2 (Sine Interpolate): 104 (1.1%) |
| `random_2_sync` | 9,618 | 4 | 1 (Tempo): 9,322 (96.9%); 0 (Seconds): 273 (2.8%); 2 (Tempo Dotted): 15 (0.2%) |
| `random_2_sync_type` | 9,618 | 2 | 0 (Off): 9,398 (97.7%); 1 (On): 220 (2.3%) |
| `random_2_tempo` | 9,618 | 13 | 8 (1/4): 9,133 (95.0%); 7 (1/2): 96 (1.0%); 6 (1/1): 86 (0.9%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `random_2_frequency` | 9,618 | -7–9 | 1.00748 / 1.12312 / 1.23875 | 97.25515% | 0% | 1–1.25 raw / 0.5–0.51562 norm: 9,356 (97.3%) |
| `random_2_keytrack_tune` | 9,618 | 0–0 | 0.00156 / 0.01562 / 0.02968 | 100% | 100% | 0–0.03125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |

### Random source 3

8 scalar parameters: 6 categorical/enum and 2 continuous. routed 3.2% (309/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `random_3_keytrack_transpose` | 9,618 | 2 | -12: 9,617 (100.0%); 28: 1 (0.0%) |
| `random_3_stereo` | 9,618 | 2 | 0 (Off): 9,560 (99.4%); 1 (On): 58 (0.6%) |
| `random_3_style` | 9,618 | 4 | 0 (Perlin): 9,487 (98.6%); 1 (Sample & Hold): 85 (0.9%); 2 (Sine Interpolate): 29 (0.3%) |
| `random_3_sync` | 9,618 | 5 | 1 (Tempo): 9,505 (98.8%); 0 (Seconds): 97 (1.0%); 2 (Tempo Dotted): 10 (0.1%) |
| `random_3_sync_type` | 9,618 | 2 | 0 (Off): 9,532 (99.1%); 1 (On): 86 (0.9%) |
| `random_3_tempo` | 9,618 | 12 | 8 (1/4): 9,410 (97.8%); 6 (1/1): 38 (0.4%); 7 (1/2): 35 (0.4%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `random_3_frequency` | 9,618 | -4.52356–9 | 1.01073 / 1.12436 / 1.23798 | 98.97068% | 0.02079% | 1–1.25 raw / 0.5–0.51562 norm: 9,522 (99.0%) |
| `random_3_keytrack_tune` | 9,618 | 0–0 | 0.00156 / 0.01562 / 0.02968 | 100% | 100% | 0–0.03125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |

### Random source 4

8 scalar parameters: 6 categorical/enum and 2 continuous. routed 1.4% (139/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `random_4_keytrack_transpose` | 9,618 | 1 | -12: 9,618 (100.0%) |
| `random_4_stereo` | 9,618 | 2 | 0 (Off): 9,589 (99.7%); 1 (On): 29 (0.3%) |
| `random_4_style` | 9,618 | 4 | 0 (Perlin): 9,550 (99.3%); 1 (Sample & Hold): 54 (0.6%); 2 (Sine Interpolate): 12 (0.1%) |
| `random_4_sync` | 9,618 | 3 | 1 (Tempo): 9,556 (99.4%); 0 (Seconds): 52 (0.5%); 2 (Tempo Dotted): 10 (0.1%) |
| `random_4_sync_type` | 9,618 | 2 | 0 (Off): 9,579 (99.6%); 1 (On): 39 (0.4%) |
| `random_4_tempo` | 9,618 | 11 | 8 (1/4): 9,525 (99.0%); 7 (1/2): 20 (0.2%); 0 (Freeze): 16 (0.2%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `random_4_frequency` | 9,618 | -3.64–9 | 1.01167 / 1.12474 / 1.2378 | 99.48014% | 0% | 1–1.25 raw / 0.5–0.51562 norm: 9,569 (99.5%) |
| `random_4_keytrack_tune` | 9,618 | 0–0 | 0.00156 / 0.01562 / 0.02968 | 100% | 100% | 0–0.03125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |

## Effects

Each effect block separately, rather than pooling chorus, delay, distortion, EQ, and reverb controls.

### Chorus effect

12 scalar parameters: 4 categorical/enum and 8 continuous. enabled 50.3% (4,835/9,618); changed 62.2% (5,981/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `chorus_on` | 9,618 | 2 | 1 (On): 4,835 (50.3%); 0 (Off): 4,783 (49.7%) |
| `chorus_sync` | 9,618 | 4 | 1 (Tempo): 9,378 (97.5%); 0 (Seconds): 213 (2.2%); 2 (Tempo Dotted): 16 (0.2%) |
| `chorus_tempo` | 9,618 | 11 | 4 (4/1): 7,470 (77.7%); 0 (Freeze): 848 (8.8%); 3 (8/1): 464 (4.8%) |
| `chorus_voices` | 9,618 | 5 | 4: 7,196 (74.8%); 2: 996 (10.4%); 1: 992 (10.3%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `chorus_cutoff` | 9,618 | 8–136 | 56.16286 / 61.3522 / 123.64074 | 64.97193% | 0% | 60–62 raw / 0.40625–0.42188 norm: 6,306 (65.6%) |
| `chorus_delay_1` | 9,618 | -10–-5.63784 | -9.57734 / -9.00969 / -7.48784 | 76.88709% | 0% | -9.04578–-8.97762 raw: 7,450 (77.5%) |
| `chorus_delay_2` | 9,618 | -10–-5.47292 | -9.52171 / -6.99919 / -6.6567 | 75.54585% | 0% | -7.02911–-6.95837 raw: 7,287 (75.8%) |
| `chorus_dry_wet` | 9,618 | 0–1 | 0.0084 / 0.50119 / 0.51557 | 48.5236% | 8.71283% | 0.5–0.51562 raw / 0.5–0.51562 norm: 4,700 (48.9%) |
| `chorus_feedback` | 9,618 | -0.95–0.95584 | -0.0143 / 0.40103 / 0.47376 | 68.0287% | 9.9189% | 0.39004–0.41982 raw: 6,557 (68.2%) |
| `chorus_frequency` | 9,618 | -6–3 | -3.04127 / -2.97627 / -2.91126 | 97.26554% | 0% | -3.04688–-2.90625 raw / 0.32812–0.34375 norm: 9,362 (97.3%) |
| `chorus_mod_depth` | 9,618 | 0–1 | 0.13676 / 0.50714 / 0.86177 | 68.17426% | 2.03785% | 0.5–0.51562 raw / 0.5–0.51562 norm: 6,600 (68.6%) |
| `chorus_spread` | 9,618 | 0–1 | 0.0844 / 0.98853 / 0.99885 | 67.72718% | 3.54544% | 0.98438–1 raw / 0.98438–1 norm: 6,551 (68.1%) |

### Compressor effect

20 scalar parameters: 2 categorical/enum and 18 continuous. enabled 70.1% (6,738/9,618); changed 77.8% (7,482/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `compressor_enabled_bands` | 9,618 | 4 | 0 (Multiband): 8,301 (86.3%); 3 (Single Band): 763 (7.9%); 2 (High Band): 343 (3.6%) |
| `compressor_on` | 9,618 | 2 | 1 (On): 6,738 (70.1%); 0 (Off): 2,880 (29.9%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `compressor_attack` | 9,618 | 0–1 | 0.14103 / 0.50921 / 0.98894 | 53.98212% | 2.59929% | 0.5–0.51562 raw / 0.5–0.51562 norm: 5,247 (54.6%) |
| `compressor_band_gain` | 9,618 | -30–30 | 6.03057 / 11.73646 / 18.42309 | 70.31607% | 0.14556% | 11.25–12.1875 raw / 0.6875–0.70312 norm: 7,016 (72.9%) |
| `compressor_band_lower_ratio` | 9,618 | -1.00811–1.01154 | 0.26348 / 0.80474 / 0.82196 | 81.51383% | 0.62383% | 0.79064–0.8222 raw: 7,930 (82.4%) |
| `compressor_band_lower_threshold` | 9,618 | -80–-1 | -50.18304 / -35.64308 / -27.68048 | 72.59305% | 0% | -36.25–-35 raw / 0.54688–0.5625 norm: 7,086 (73.7%) |
| `compressor_band_upper_ratio` | 9,618 | -0.57592–1.02341 | 0.42287 / 0.85865 / 0.87267 | 79.70472% | 0.31192% | 0.84849–0.87348 raw: 7,717 (80.2%) |
| `compressor_band_upper_threshold` | 9,618 | -79–-1 | -33.38292 / -24.37536 / -16.25932 | 69.91058% | 0% | -25–-23.75 raw / 0.6875–0.70312 norm: 6,907 (71.8%) |
| `compressor_high_gain` | 9,618 | -30–30 | 3.36121 / 16.31965 / 19.82474 | 69.4947% | 0.29112% | 15.9375–16.875 raw / 0.76562–0.78125 norm: 6,782 (70.5%) |
| `compressor_high_lower_ratio` | 9,618 | -1.01698–1.0283 | 0.30967 / 0.79413 / 0.84044 | 81.79455% | 0.44708% | 0.77264–0.8046 raw: 5,398 (56.1%) |
| `compressor_high_lower_threshold` | 9,618 | -79–-1 | -53.97147 / -34.47253 / -31.23503 | 73.09212% | 0% | -35–-33.75 raw / 0.5625–0.57812 norm: 7,113 (74.0%) |
| `compressor_high_upper_ratio` | 9,618 | -0.58412–1.02264 | 0.47588 / 1.00728 / 1.0211 | 80.5053% | 0.28072% | 0.99754–1.02264 raw: 7,860 (81.7%) |
| `compressor_high_upper_threshold` | 9,618 | -79–0.3046 | -40.93187 / -30.00044 / -20.09142 | 69.8482% | 0.0104% | -30.67376–-29.43463 raw: 6,628 (68.9%) |
| `compressor_low_gain` | 9,618 | -30–30 | 1.16611 / 16.32296 / 19.20798 | 71.81327% | 0.19755% | 15.9375–16.875 raw / 0.76562–0.78125 norm: 7,001 (72.8%) |
| `compressor_low_lower_ratio` | 9,618 | -1.33529–1.55634 | 0.30157 / 0.80872 / 0.83235 | 85.51674% | 0.65502% | 0.78825–0.83343 raw: 8,276 (86.0%) |
| `compressor_low_lower_threshold` | 9,618 | -79–-1 | -54.80757 / -34.45507 / -32.65804 | 78.06197% | 0% | -35–-33.75 raw / 0.5625–0.57812 norm: 7,564 (78.6%) |
| `compressor_low_upper_ratio` | 9,618 | -0.02553–1.0283 | 0.53418 / 0.90361 / 0.91256 | 84.32106% | 0.24953% | 0.89657–0.91304 raw: 7,962 (82.8%) |
| `compressor_low_upper_threshold` | 9,618 | -79–0.25473 | -39.79739 / -27.64353 / -21.51998 | 76.75192% | 0.0104% | -28.22744–-26.98908 raw: 7,250 (75.4%) |
| `compressor_mix` | 9,618 | 0–1 | 0.17143 / 0.98949 / 0.99895 | 73.94469% | 2.59929% | 0.98438–1 raw / 0.98438–1 norm: 7,147 (74.3%) |
| `compressor_release` | 9,618 | 0–1 | 0.15223 / 0.50728 / 0.98616 | 57.49636% | 2.06904% | 0.5–0.51562 raw / 0.5–0.51562 norm: 5,579 (58.0%) |

### Delay effect

12 scalar parameters: 6 categorical/enum and 6 continuous. enabled 40.6% (3,908/9,618); changed 49.9% (4,804/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `delay_aux_sync` | 9,618 | 4 | 1 (Tempo): 8,646 (89.9%); 2 (Tempo Dotted): 651 (6.8%); 0 (Seconds): 217 (2.3%) |
| `delay_aux_tempo` | 9,618 | 9 | 9 (2/1): 8,041 (83.6%); 8 (4/1): 895 (9.3%); 10 (1/1): 367 (3.8%) |
| `delay_on` | 9,618 | 2 | 0 (Off): 5,710 (59.4%); 1 (On): 3,908 (40.6%) |
| `delay_style` | 9,618 | 4 | 0 (Mono): 6,391 (66.4%); 2 (Ping Pong): 2,027 (21.1%); 1 (Stereo): 838 (8.7%) |
| `delay_sync` | 9,618 | 4 | 1 (Tempo): 8,521 (88.6%); 2 (Tempo Dotted): 547 (5.7%); 0 (Seconds): 430 (4.5%) |
| `delay_tempo` | 9,618 | 9 | 9 (2/1): 7,533 (78.3%); 8 (4/1): 1,069 (11.1%); 10 (1/1): 445 (4.6%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `delay_aux_frequency` | 9,618 | -2–9 | 1.96064 / 2.04019 / 2.11975 | 97.10959% | 0.08318% | 1.95312–2.125 raw / 0.35938–0.375 norm: 9,350 (97.2%) |
| `delay_dry_wet` | 9,618 | 0–1 | 0.00736 / 0.33181 / 0.35847 | 57.66272% | 9.96049% | 0.32812–0.34375 raw / 0.32812–0.34375 norm: 5,602 (58.2%) |
| `delay_feedback` | 9,618 | -1–1 | 0.11327 / 0.51209 / 0.60469 | 68.36141% | 0.39509% | 0.5–0.53125 raw / 0.75–0.76562 norm: 6,666 (69.3%) |
| `delay_filter_cutoff` | 9,618 | 8–136 | 57.748 / 61.31522 / 105.61522 | 67.48804% | 0% | 60–62 raw / 0.40625–0.42188 norm: 6,532 (67.9%) |
| `delay_filter_spread` | 9,618 | 0–1 | 0.0153 / 0.98876 / 0.99887 | 69.33874% | 4.80349% | 0.98438–1 raw / 0.98438–1 norm: 6,684 (69.5%) |
| `delay_frequency` | 9,618 | -2–9 | 1.96046 / 2.04187 / 2.12327 | 94.88459% | 0.07278% | 1.95312–2.125 raw / 0.35938–0.375 norm: 9,137 (95.0%) |

### Distortion effect

8 scalar parameters: 3 categorical/enum and 5 continuous. enabled 65.0% (6,253/9,618); changed 76.4% (7,348/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `distortion_filter_order` | 9,618 | 3 | 0 (None): 7,299 (75.9%); 1 (Pre): 1,262 (13.1%); 2 (Post): 1,057 (11.0%) |
| `distortion_on` | 9,618 | 2 | 1 (On): 6,253 (65.0%); 0 (Off): 3,365 (35.0%) |
| `distortion_type` | 9,618 | 6 | 0 (Soft Clip): 6,340 (65.9%); 1 (Hard Clip): 1,418 (14.7%); 5 (Down Sample): 731 (7.6%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `distortion_drive` | 9,618 | -30–30 | -15.60298 / 1.71607 / 29.06473 | 28.70659% | 28.70659% | 0–0.9375 raw / 0.5–0.51562 norm: 2,948 (30.7%) |
| `distortion_filter_blend` | 9,618 | 0–2 | 0.00178 / 0.01783 / 1.41585 | 87.40902% | 87.40902% | 0–0.03125 raw / 0–0.01562 norm: 8,427 (87.6%) |
| `distortion_filter_cutoff` | 9,618 | 8–136 | 41.61852 / 81.0359 / 127.54186 | 69.0996% | 0% | 80–82 raw / 0.5625–0.57812 norm: 6,714 (69.8%) |
| `distortion_filter_resonance` | 9,618 | 0–1 | 0.00991 / 0.50563 / 0.51535 | 71.93803% | 7.3404% | 0.5–0.51562 raw / 0.5–0.51562 norm: 6,961 (72.4%) |
| `distortion_mix` | 9,618 | 0–1 | 0.01423 / 0.98928 / 0.99893 | 72.01081% | 5.16739% | 0.98438–1 raw / 0.98438–1 norm: 7,009 (72.9%) |

### Eq effect

13 scalar parameters: 4 categorical/enum and 9 continuous. enabled 64.3% (6,186/9,618); changed 69.3% (6,668/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `eq_band_mode` | 9,618 | 2 | 0 (Shelf): 9,385 (97.6%); 1 (Notch): 233 (2.4%) |
| `eq_high_mode` | 9,618 | 2 | 0 (Shelf): 8,213 (85.4%); 1 (Low Pass): 1,405 (14.6%) |
| `eq_low_mode` | 9,618 | 2 | 0 (Shelf): 7,013 (72.9%); 1 (High Pass): 2,605 (27.1%) |
| `eq_on` | 9,618 | 2 | 1 (On): 6,186 (64.3%); 0 (Off): 3,432 (35.7%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `eq_band_cutoff` | 9,618 | 8–136 | 44.71702 / 80.63489 / 103.02203 | 46.58973% | 0% | 80–82 raw / 0.5625–0.57812 norm: 4,689 (48.8%) |
| `eq_band_gain` | 9,618 | -15–15 | -11.00657 / 0.24053 / 8.61348 | 49.89603% | 49.89603% | 0–0.46875 raw / 0.5–0.51562 norm: 4,914 (51.1%) |
| `eq_band_resonance` | 9,618 | 0–1 | 0.10173 / 0.43989 / 0.58692 | 66.9162% | 5.12581% | 0.43301–0.45069 raw / 0.1875–0.20312 norm: 6,486 (67.4%) |
| `eq_high_cutoff` | 9,618 | 8–136 | 75.57368 / 101.18302 / 129.21442 | 45.83073% | 0% | 100–102 raw / 0.71875–0.73438 norm: 4,699 (48.9%) |
| `eq_high_gain` | 9,618 | -15–19.74391 | -14.21283 / 0.46249 / 8.6167 | 50.68621% | 50.68621% | 0.20046–0.74334 raw: 4,990 (51.9%) |
| `eq_high_resonance` | 9,618 | 0–1 | 0.10519 / 0.31745 / 0.37144 | 78.23872% | 5.10501% | 0.30619–0.33072 raw / 0.09375–0.10938 norm: 7,619 (79.2%) |
| `eq_low_cutoff` | 9,618 | 8–136 | 18.97234 / 41.08741 / 62.18485 | 45.5812% | 0% | 40–42 raw / 0.25–0.26562 norm: 4,656 (48.4%) |
| `eq_low_gain` | 9,618 | -15–16.13977 | -14.74816 / -0.20485 / 6.93772 | 51.55958% | 51.55958% | -0.40323–0.08332 raw: 4,931 (51.3%) |
| `eq_low_resonance` | 9,618 | 0–1 | 0.12009 / 0.31751 / 0.35823 | 80.25577% | 3.6702% | 0.30619–0.33072 raw / 0.09375–0.10938 norm: 7,802 (81.1%) |

### Filter Fx effect

15 scalar parameters: 3 categorical/enum and 12 continuous. enabled 35.5% (3,417/9,618); changed 48.4% (4,653/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `filter_fx_model` | 9,618 | 8 | 0 (Analog): 7,929 (82.4%); 6 (Comb): 418 (4.3%); 3 (Digital): 373 (3.9%) |
| `filter_fx_on` | 9,618 | 2 | 0 (Off): 6,201 (64.5%); 1 (On): 3,417 (35.5%) |
| `filter_fx_style` | 9,618 | 6 | 0 (12dB): 7,736 (80.4%); 1 (24dB): 1,367 (14.2%); 2 (Notch Blend): 186 (1.9%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `filter_fx_blend` | 9,618 | 0–2 | 0.0018 / 0.01796 / 1.79978 | 86.68122% | 86.68122% | 0–0.03125 raw / 0–0.01562 norm: 8,367 (87.0%) |
| `filter_fx_blend_transpose` | 9,618 | 0–84 | 42.04453 / 42.65653 / 43.26854 | 96.42337% | 0.57184% | 42–43.3125 raw / 0.5–0.51562 norm: 9,281 (96.5%) |
| `filter_fx_cutoff` | 9,618 | 8–136 | 27.99211 / 61.21136 / 134.38305 | 56.26949% | 0% | 60–62 raw / 0.40625–0.42188 norm: 5,474 (56.9%) |
| `filter_fx_drive` | 9,618 | 0–20 | 0.0181 / 0.18104 / 8.48931 | 85.24641% | 85.24641% | 0–0.3125 raw / 0–0.01562 norm: 8,300 (86.3%) |
| `filter_fx_formant_resonance` | 9,618 | 0.12309–1 | 0.84957 / 0.85595 / 0.86234 | 98.28447% | 0% | 0.84928–0.86298 raw: 9,288 (96.6%) |
| `filter_fx_formant_spread` | 9,618 | -1–1 | 0.00128 / 0.01553 / 0.02978 | 98.66916% | 98.66916% | 0–0.03125 raw / 0.5–0.51562 norm: 9,490 (98.7%) |
| `filter_fx_formant_transpose` | 9,618 | -12–12 | 0.01563 / 0.18699 / 0.35834 | 98.47162% | 98.47162% | 0–0.375 raw / 0.5–0.51562 norm: 9,471 (98.5%) |
| `filter_fx_formant_x` | 9,618 | 0–1 | 0.50061 / 0.50778 / 0.51494 | 98.02454% | 0.29112% | 0.5–0.51562 raw / 0.5–0.51562 norm: 9,434 (98.1%) |
| `filter_fx_formant_y` | 9,618 | 0–1 | 0.50061 / 0.50778 / 0.51495 | 98.02454% | 0.39509% | 0.5–0.51562 raw / 0.5–0.51562 norm: 9,434 (98.1%) |
| `filter_fx_keytrack` | 9,618 | -1–1 | 0.00125 / 0.01685 / 0.97617 | 90.15388% | 90.15388% | 0–0.03125 raw / 0.5–0.51562 norm: 8,671 (90.2%) |
| `filter_fx_mix` | 9,618 | 0–1 | 0.25962 / 0.99119 / 0.99912 | 88.19921% | 3.6806% | 0.98438–1 raw / 0.98438–1 norm: 8,528 (88.7%) |
| `filter_fx_resonance` | 9,618 | 0–1 | 0.00408 / 0.50381 / 0.56484 | 57.88106% | 17.81036% | 0.5–0.51562 raw / 0.5–0.51562 norm: 5,610 (58.3%) |

### Flanger effect

10 scalar parameters: 3 categorical/enum and 7 continuous. enabled 11.9% (1,148/9,618); changed 21.3% (2,049/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `flanger_on` | 9,618 | 2 | 0 (Off): 8,470 (88.1%); 1 (On): 1,148 (11.9%) |
| `flanger_sync` | 9,618 | 4 | 1 (Tempo): 9,467 (98.4%); 0 (Seconds): 113 (1.2%); 2 (Tempo Dotted): 22 (0.2%) |
| `flanger_tempo` | 9,618 | 11 | 4 (4/1): 8,673 (90.2%); 0 (Freeze): 465 (4.8%); 3 (8/1): 86 (0.9%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `flanger_center` | 9,618 | 8–136 | 51.775 / 64.98172 / 68.83235 | 87.22188% | 0% | 64–66 raw / 0.4375–0.45312 norm: 8,425 (87.6%) |
| `flanger_depth` | 2 | 0.0003139–0.037 | 0.0003426 / 0.0006005 / 0.0008584 | — | 0% | 0.0003139–0.000887 raw: 1 (50.0%) |
| `flanger_dry_wet` | 9,618 | -0.00278–0.74048 | 0.05822 / 0.50118 / 0.50756 | 84.43543% | 3.3271% | 0.4966–0.50821 raw: 7,872 (81.8%) |
| `flanger_feedback` | 9,618 | -1–1 | 0.22905 / 0.51495 / 0.53117 | 86.48368% | 1.12289% | 0.5–0.53125 raw / 0.75–0.76562 norm: 8,338 (86.7%) |
| `flanger_frequency` | 9,618 | -5–2 | 1.89464 / 1.94454 / 1.99444 | 98.59638% | 0.05199% | 1.89062–2 raw / 0.98438–1 norm: 9,485 (98.6%) |
| `flanger_mod_depth` | 9,618 | 0–1 | 0.35198 / 0.50775 / 0.56655 | 88.31358% | 1.65315% | 0.5–0.51562 raw / 0.5–0.51562 norm: 8,506 (88.4%) |
| `flanger_phase_offset` | 9,618 | 0–1 | 0.28258 / 0.33586 / 0.34366 | 90.04991% | 2.23539% | 0.32812–0.34375 raw / 0.32812–0.34375 norm: 8,673 (90.2%) |

### Phaser effect

10 scalar parameters: 3 categorical/enum and 7 continuous. enabled 15.2% (1,458/9,618); changed 24.3% (2,341/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `phaser_on` | 9,618 | 2 | 0 (Off): 8,160 (84.8%); 1 (On): 1,458 (15.2%) |
| `phaser_sync` | 9,618 | 4 | 1 (Tempo): 9,467 (98.4%); 0 (Seconds): 108 (1.1%); 2 (Tempo Dotted): 27 (0.3%) |
| `phaser_tempo` | 9,618 | 12 | 3 (8/1): 8,446 (87.8%); 0 (Freeze): 572 (5.9%); 4 (4/1): 157 (1.6%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `phaser_blend` | 9,618 | 0–2 | 1.00034 / 1.01547 / 1.0306 | 92.95072% | 2.15221% | 1–1.03125 raw / 0.5–0.51562 norm: 8,940 (93.0%) |
| `phaser_center` | 9,618 | 8–136 | 60.76944 / 80.95543 / 81.99811 | 85.89104% | 0% | 80–82 raw / 0.5625–0.57812 norm: 8,301 (86.3%) |
| `phaser_dry_wet` | 9,618 | 0–1 | 0.05691 / 0.99048 / 0.99905 | 81.89852% | 4.16927% | 0.98438–1 raw / 0.98438–1 norm: 7,893 (82.1%) |
| `phaser_feedback` | 9,618 | 0–1 | 0.2712 / 0.50765 / 0.60999 | 83.70763% | 2.1938% | 0.5–0.51562 raw / 0.5–0.51562 norm: 8,071 (83.9%) |
| `phaser_frequency` | 9,618 | -5–2 | -3.02644 / -2.97646 / -2.92649 | 98.47162% | 0% | -3.03125–-2.92188 raw / 0.28125–0.29688 norm: 9,472 (98.5%) |
| `phaser_mod_depth` | 9,618 | 0–48 | 9.0925 / 24.35845 / 24.74884 | 86.33812% | 3.07756% | 24–24.75 raw / 0.5–0.51562 norm: 8,314 (86.4%) |
| `phaser_phase_offset` | 9,618 | 0–1 | 0.10501 / 0.33574 / 0.34373 | 87.8665% | 3.41027% | 0.32812–0.34375 raw / 0.32812–0.34375 norm: 8,458 (87.9%) |

### Reverb effect

13 scalar parameters: 1 categorical/enum and 12 continuous. enabled 67.9% (6,532/9,618); changed 73.5% (7,074/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `reverb_on` | 9,618 | 2 | 1 (On): 6,532 (67.9%); 0 (Off): 3,086 (32.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `reverb_chorus_amount` | 9,618 | 0–1 | 0.12812 / 0.23488 / 0.48607 | 80.29736% | 3.46226% | 0.21651–0.25 raw / 0.04688–0.0625 norm: 7,813 (81.2%) |
| `reverb_chorus_frequency` | 9,618 | -8–3 | -6.15208 / -2.08032 / -1.98771 | 83.20857% | 0.03119% | -2.15625–-1.98438 raw / 0.53125–0.54688 norm: 8,032 (83.5%) |
| `reverb_decay_time` | 9,618 | -6–6 | -3.17414 / 0.12792 / 2.82004 | 46.51695% | 46.51695% | 0–0.1875 raw / 0.5–0.51562 norm: 4,571 (47.5%) |
| `reverb_delay` | 9,618 | 0–0.3 | 0.0002791 / 0.00279 / 0.06819 | 83.34373% | 83.34373% | 0–0.00469 raw / 0–0.01562 norm: 8,077 (84.0%) |
| `reverb_dry_wet` | 9,618 | 0–1 | 0.00722 / 0.25802 / 0.62161 | 37.64816% | 10.13724% | 0.25–0.26562 raw / 0.25–0.26562 norm: 3,736 (38.8%) |
| `reverb_high_shelf_cutoff` | 9,618 | 0–128 | 87.3663 / 91.28644 / 126.13598 | 66.10522% | 0.17675% | 90–92 raw / 0.70312–0.71875 norm: 6,490 (67.5%) |
| `reverb_high_shelf_gain` | 9,618 | -6–0 | -5.93011 / -0.99264 / -0.07193 | 66.04284% | 6.46704% | -1.03125–-0.9375 raw / 0.82812–0.84375 norm: 6,405 (66.6%) |
| `reverb_low_shelf_cutoff` | 9,618 | 0–128 | 0.16149 / 1.61495 / 55.35375 | 61.64483% | 61.64483% | 0–2 raw / 0–0.01562 norm: 5,955 (61.9%) |
| `reverb_low_shelf_gain` | 9,618 | -6–0 | -5.977 / -0.07559 / -0.00757 | 61.91516% | 61.91516% | -0.09375–0 raw / 0.98438–1 norm: 5,965 (62.0%) |
| `reverb_pre_high_cutoff` | 9,618 | 0–128 | 91.57925 / 110.96168 / 116.05349 | 84.16511% | 0.32231% | 110–112 raw / 0.85938–0.875 norm: 8,141 (84.6%) |
| `reverb_pre_low_cutoff` | 9,618 | 0–128 | 0.1346 / 1.34598 / 65.49683 | 73.9031% | 73.9031% | 0–2 raw / 0–0.01562 norm: 7,145 (74.3%) |
| `reverb_size` | 9,618 | 0–1 | 0.17116 / 0.50819 / 0.88529 | 60.10605% | 2.00665% | 0.5–0.51562 raw / 0.5–0.51562 norm: 5,832 (60.6%) |

## Modulation matrix

All 64 matrix slots, with per-slot scalar distributions plus the route source/destination and remap analysis.

### Modulation slot 1

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 90.8% (8,734/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_1_bipolar` | 9,618 | 2 | 0 (Off): 8,945 (93.0%); 1 (On): 673 (7.0%) |
| `modulation_1_bypass` | 9,618 | 2 | 0 (Off): 9,552 (99.3%); 1 (On): 66 (0.7%) |
| `modulation_1_stereo` | 9,618 | 2 | 0 (Off): 9,556 (99.4%); 1 (On): 62 (0.6%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_1_amount` | 9,618 | -1–1 | -0.30374 / 0.34168 / 0.99093 | 13.25639% | 13.25639% | 0.96875–1 raw / 0.98438–1 norm: 1,660 (17.3%) |
| `modulation_1_power` | 9,618 | -10–10 | 0.01455 / 0.15636 / 0.29818 | 99.05386% | 99.05386% | 0–0.3125 raw / 0.5–0.51562 norm: 9,536 (99.1%) |
| `modulation_1_ramp_down` | 287 | -10–-1.98779 | -9.99374 / -9.9374 / -9.88107 | — | 0% | -10–-9.87481 raw: 286 (99.7%) |
| `modulation_1_ramp_up` | 287 | -10–4 | -9.98906 / -9.89062 / -9.79219 | — | 0% | -10–-9.78125 raw: 286 (99.7%) |

### Modulation slot 2

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 83.2% (7,998/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_2_bipolar` | 9,618 | 2 | 0 (Off): 9,006 (93.6%); 1 (On): 612 (6.4%) |
| `modulation_2_bypass` | 9,618 | 2 | 0 (Off): 9,578 (99.6%); 1 (On): 40 (0.4%) |
| `modulation_2_stereo` | 9,618 | 2 | 0 (Off): 9,560 (99.4%); 1 (On): 58 (0.6%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_2_amount` | 9,618 | -1–1 | -0.40269 / 0.25994 / 0.98853 | 17.9975% | 17.9975% | 0–0.03125 raw / 0.5–0.51562 norm: 1,852 (19.3%) |
| `modulation_2_power` | 9,618 | -9.73686–10 | 0.0143 / 0.15594 / 0.29758 | 99.24101% | 99.24101% | 0–0.3125 raw / 0.5–0.51562 norm: 9,548 (99.3%) |
| `modulation_2_ramp_down` | 287 | -10–-3.49008 | -9.99491 / -9.94914 / -9.90337 | — | 0% | -10–-9.89828 raw: 286 (99.7%) |
| `modulation_2_ramp_up` | 287 | -10–0.4112 | -9.99187 / -9.91866 / -9.84546 | — | 0% | -10–-9.83733 raw: 286 (99.7%) |

### Modulation slot 3

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 75.3% (7,238/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_3_bipolar` | 9,618 | 2 | 0 (Off): 9,086 (94.5%); 1 (On): 532 (5.5%) |
| `modulation_3_bypass` | 9,618 | 2 | 0 (Off): 9,558 (99.4%); 1 (On): 60 (0.6%) |
| `modulation_3_stereo` | 9,618 | 2 | 0 (Off): 9,572 (99.5%); 1 (On): 46 (0.5%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_3_amount` | 9,618 | -1–1 | -0.38862 / 0.18857 / 0.98641 | 25.20274% | 25.20274% | 0–0.03125 raw / 0.5–0.51562 norm: 2,518 (26.2%) |
| `modulation_3_power` | 9,618 | -10–10 | 0.01455 / 0.15612 / 0.29769 | 99.30339% | 99.30339% | 0–0.3125 raw / 0.5–0.51562 norm: 9,553 (99.3%) |
| `modulation_3_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_3_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 4

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 67.3% (6,475/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_4_bipolar` | 9,618 | 2 | 0 (Off): 9,142 (95.1%); 1 (On): 476 (4.9%) |
| `modulation_4_bypass` | 9,618 | 2 | 0 (Off): 9,573 (99.5%); 1 (On): 45 (0.5%) |
| `modulation_4_stereo` | 9,618 | 2 | 0 (Off): 9,572 (99.5%); 1 (On): 46 (0.5%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_4_amount` | 9,618 | -1–1 | -0.33461 / 0.13135 / 0.98451 | 32.24163% | 32.24163% | 0–0.03125 raw / 0.5–0.51562 norm: 3,196 (33.2%) |
| `modulation_4_power` | 9,618 | -10–10 | 0.01457 / 0.15599 / 0.29741 | 99.40736% | 99.40736% | 0–0.3125 raw / 0.5–0.51562 norm: 9,563 (99.4%) |
| `modulation_4_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_4_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 5

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 59.8% (5,751/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_5_bipolar` | 9,618 | 2 | 0 (Off): 9,092 (94.5%); 1 (On): 526 (5.5%) |
| `modulation_5_bypass` | 9,618 | 2 | 0 (Off): 9,558 (99.4%); 1 (On): 60 (0.6%) |
| `modulation_5_stereo` | 9,618 | 2 | 0 (Off): 9,561 (99.4%); 1 (On): 57 (0.6%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_5_amount` | 9,618 | -1–1 | -0.28918 / 0.03088 / 0.98057 | 40.0811% | 40.0811% | 0–0.03125 raw / 0.5–0.51562 norm: 3,936 (40.9%) |
| `modulation_5_power` | 9,618 | -10–10 | 0.01494 / 0.15614 / 0.29733 | 99.53213% | 99.53213% | 0–0.3125 raw / 0.5–0.51562 norm: 9,578 (99.6%) |
| `modulation_5_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_5_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 6

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 53.6% (5,156/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_6_bipolar` | 9,618 | 2 | 0 (Off): 9,163 (95.3%); 1 (On): 455 (4.7%) |
| `modulation_6_bypass` | 9,618 | 2 | 0 (Off): 9,570 (99.5%); 1 (On): 48 (0.5%) |
| `modulation_6_stereo` | 9,618 | 2 | 0 (Off): 9,565 (99.4%); 1 (On): 53 (0.6%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_6_amount` | 9,618 | -1–1 | -0.28395 / 0.02695 / 0.97789 | 46.309% | 46.309% | 0–0.03125 raw / 0.5–0.51562 norm: 4,532 (47.1%) |
| `modulation_6_power` | 9,618 | -10–10 | 0.01485 / 0.1561 / 0.29736 | 99.51133% | 99.51133% | 0–0.3125 raw / 0.5–0.51562 norm: 9,574 (99.5%) |
| `modulation_6_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_6_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 7

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 48.4% (4,659/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_7_bipolar` | 9,618 | 2 | 0 (Off): 9,222 (95.9%); 1 (On): 396 (4.1%) |
| `modulation_7_bypass` | 9,618 | 2 | 0 (Off): 9,576 (99.6%); 1 (On): 42 (0.4%) |
| `modulation_7_stereo` | 9,618 | 2 | 0 (Off): 9,592 (99.7%); 1 (On): 26 (0.3%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_7_amount` | 9,618 | -1–1 | -0.22877 / 0.02511 / 0.9749 | 51.38282% | 51.38282% | 0–0.03125 raw / 0.5–0.51562 norm: 4,980 (51.8%) |
| `modulation_7_power` | 9,618 | -10–10 | 0.01513 / 0.1563 / 0.29747 | 99.58411% | 99.58411% | 0–0.3125 raw / 0.5–0.51562 norm: 9,580 (99.6%) |
| `modulation_7_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_7_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 8

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 43.3% (4,162/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_8_bipolar` | 9,618 | 2 | 0 (Off): 9,257 (96.2%); 1 (On): 361 (3.8%) |
| `modulation_8_bypass` | 9,618 | 2 | 0 (Off): 9,592 (99.7%); 1 (On): 26 (0.3%) |
| `modulation_8_stereo` | 9,618 | 2 | 0 (Off): 9,586 (99.7%); 1 (On): 32 (0.3%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_8_amount` | 9,618 | -1–1 | -0.19502 / 0.02327 / 0.97191 | 56.18632% | 56.18632% | 0–0.03125 raw / 0.5–0.51562 norm: 5,459 (56.8%) |
| `modulation_8_power` | 9,618 | -7.12323–10 | 0.01516 / 0.15633 / 0.2975 | 99.6153% | 99.6153% | 0–0.3125 raw / 0.5–0.51562 norm: 9,580 (99.6%) |
| `modulation_8_ramp_down` | 287 | -10–-4.01221 | -9.99532 / -9.95322 / -9.91112 | — | 0% | -10–-9.90644 raw: 286 (99.7%) |
| `modulation_8_ramp_up` | 287 | -10–-2.343 | -9.99402 / -9.94018 / -9.88634 | — | 0% | -10–-9.88036 raw: 286 (99.7%) |

### Modulation slot 9

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 38.9% (3,742/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_9_bipolar` | 9,618 | 2 | 0 (Off): 9,274 (96.4%); 1 (On): 344 (3.6%) |
| `modulation_9_bypass` | 9,618 | 2 | 0 (Off): 9,597 (99.8%); 1 (On): 21 (0.2%) |
| `modulation_9_stereo` | 9,618 | 2 | 0 (Off): 9,581 (99.6%); 1 (On): 37 (0.4%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_9_amount` | 9,618 | -1–1 | -0.18335 / 0.02176 / 0.90677 | 60.86504% | 60.86504% | 0–0.03125 raw / 0.5–0.51562 norm: 5,908 (61.4%) |
| `modulation_9_power` | 9,618 | -9.11111–3.90476 | 0.0152 / 0.15609 / 0.29698 | 99.78166% | 99.78166% | 0–0.3125 raw / 0.5–0.51562 norm: 9,599 (99.8%) |
| `modulation_9_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_9_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 10

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 35.3% (3,395/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_10_bipolar` | 9,618 | 2 | 0 (Off): 9,307 (96.8%); 1 (On): 311 (3.2%) |
| `modulation_10_bypass` | 9,618 | 2 | 0 (Off): 9,590 (99.7%); 1 (On): 28 (0.3%) |
| `modulation_10_stereo` | 9,618 | 2 | 0 (Off): 9,588 (99.7%); 1 (On): 30 (0.3%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_10_amount` | 9,618 | -1–1 | -0.1069 / 0.02113 / 0.82451 | 64.45207% | 64.45207% | 0–0.03125 raw / 0.5–0.51562 norm: 6,236 (64.8%) |
| `modulation_10_power` | 9,618 | -10–10 | 0.0153 / 0.15618 / 0.29707 | 99.79206% | 99.79206% | 0–0.3125 raw / 0.5–0.51562 norm: 9,599 (99.8%) |
| `modulation_10_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_10_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 11

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 31.9% (3,069/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_11_bipolar` | 9,618 | 2 | 0 (Off): 9,332 (97.0%); 1 (On): 286 (3.0%) |
| `modulation_11_bypass` | 9,618 | 2 | 0 (Off): 9,606 (99.9%); 1 (On): 12 (0.1%) |
| `modulation_11_stereo` | 9,618 | 2 | 0 (Off): 9,584 (99.6%); 1 (On): 34 (0.4%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_11_amount` | 9,618 | -1–1 | -0.08833 / 0.02009 / 0.75015 | 68.22624% | 68.22624% | 0–0.03125 raw / 0.5–0.51562 norm: 6,594 (68.6%) |
| `modulation_11_power` | 9,618 | -8.0241–10 | 0.01539 / 0.15622 / 0.29705 | 99.84404% | 99.84404% | 0–0.3125 raw / 0.5–0.51562 norm: 9,603 (99.8%) |
| `modulation_11_ramp_down` | 287 | -10–4 | -9.98906 / -9.89062 / -9.79219 | — | 0% | -10–-9.78125 raw: 286 (99.7%) |
| `modulation_11_ramp_up` | 287 | -10–-3.85686 | -9.9952 / -9.95201 / -9.90881 | — | 0% | -10–-9.90401 raw: 286 (99.7%) |

### Modulation slot 12

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 28.7% (2,761/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_12_bipolar` | 9,618 | 2 | 0 (Off): 9,386 (97.6%); 1 (On): 232 (2.4%) |
| `modulation_12_bypass` | 9,618 | 2 | 0 (Off): 9,602 (99.8%); 1 (On): 16 (0.2%) |
| `modulation_12_stereo` | 9,618 | 2 | 0 (Off): 9,576 (99.6%); 1 (On): 42 (0.4%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_12_amount` | 9,618 | -1–1 | -0.04823 / 0.01949 / 0.67085 | 71.34539% | 71.34539% | 0–0.03125 raw / 0.5–0.51562 norm: 6,911 (71.9%) |
| `modulation_12_power` | 9,618 | -10–9.88332 | 0.01545 / 0.15622 / 0.29699 | 99.86484% | 99.86484% | 0–0.3125 raw / 0.5–0.51562 norm: 9,607 (99.9%) |
| `modulation_12_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_12_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 13

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 26.1% (2,510/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_13_bipolar` | 9,618 | 2 | 0 (Off): 9,415 (97.9%); 1 (On): 203 (2.1%) |
| `modulation_13_bypass` | 9,618 | 2 | 0 (Off): 9,606 (99.9%); 1 (On): 12 (0.1%) |
| `modulation_13_stereo` | 9,618 | 2 | 0 (Off): 9,603 (99.8%); 1 (On): 15 (0.2%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_13_amount` | 9,618 | -1–1 | -0.0799 / 0.01882 / 0.56268 | 73.79913% | 73.79913% | 0–0.03125 raw / 0.5–0.51562 norm: 7,121 (74.0%) |
| `modulation_13_power` | 9,618 | -10–10 | 0.01548 / 0.15628 / 0.29708 | 99.86484% | 99.86484% | 0–0.3125 raw / 0.5–0.51562 norm: 9,605 (99.9%) |
| `modulation_13_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_13_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 14

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 23.8% (2,290/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_14_bipolar` | 9,618 | 2 | 0 (Off): 9,406 (97.8%); 1 (On): 212 (2.2%) |
| `modulation_14_bypass` | 9,618 | 2 | 0 (Off): 9,606 (99.9%); 1 (On): 12 (0.1%) |
| `modulation_14_stereo` | 9,618 | 2 | 0 (Off): 9,587 (99.7%); 1 (On): 31 (0.3%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_14_amount` | 9,618 | -1–1 | 0.0002161 / 0.01861 / 0.52704 | 76.07611% | 76.07611% | 0–0.03125 raw / 0.5–0.51562 norm: 7,353 (76.5%) |
| `modulation_14_power` | 9,618 | -9.6053–4.55069 | 0.01535 / 0.15614 / 0.29692 | 99.86484% | 99.86484% | 0–0.3125 raw / 0.5–0.51562 norm: 9,606 (99.9%) |
| `modulation_14_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_14_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 15

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 21.5% (2,069/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_15_bipolar` | 9,618 | 2 | 0 (Off): 9,402 (97.8%); 1 (On): 216 (2.2%) |
| `modulation_15_bypass` | 9,618 | 2 | 0 (Off): 9,599 (99.8%); 1 (On): 19 (0.2%) |
| `modulation_15_stereo` | 9,618 | 2 | 0 (Off): 9,597 (99.8%); 1 (On): 21 (0.2%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_15_amount` | 9,618 | -1–1 | 0.0002603 / 0.01819 / 0.53744 | 78.23872% | 78.23872% | 0–0.03125 raw / 0.5–0.51562 norm: 7,544 (78.4%) |
| `modulation_15_power` | 9,618 | -10–9.64995 | 0.01545 / 0.15625 / 0.29705 | 99.85444% | 99.85444% | 0–0.3125 raw / 0.5–0.51562 norm: 9,605 (99.9%) |
| `modulation_15_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_15_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 16

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 19.6% (1,889/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_16_bipolar` | 9,618 | 2 | 0 (Off): 9,459 (98.3%); 1 (On): 159 (1.7%) |
| `modulation_16_bypass` | 9,618 | 2 | 0 (Off): 9,605 (99.9%); 1 (On): 13 (0.1%) |
| `modulation_16_stereo` | 9,618 | 2 | 0 (Off): 9,581 (99.6%); 1 (On): 37 (0.4%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_16_amount` | 9,618 | -1–1 | 0.0004758 / 0.01795 / 0.50812 | 80.1518% | 80.1518% | 0–0.03125 raw / 0.5–0.51562 norm: 7,740 (80.5%) |
| `modulation_16_power` | 9,618 | -10–10 | 0.01544 / 0.1562 / 0.29696 | 99.89603% | 99.89603% | 0–0.3125 raw / 0.5–0.51562 norm: 9,608 (99.9%) |
| `modulation_16_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_16_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 17

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 18.2% (1,754/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_17_bipolar` | 9,618 | 2 | 0 (Off): 9,444 (98.2%); 1 (On): 174 (1.8%) |
| `modulation_17_bypass` | 9,618 | 2 | 0 (Off): 9,606 (99.9%); 1 (On): 12 (0.1%) |
| `modulation_17_stereo` | 9,618 | 2 | 0 (Off): 9,589 (99.7%); 1 (On): 29 (0.3%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_17_amount` | 9,618 | -1–1 | 0.0005146 / 0.01767 / 0.45051 | 81.75296% | 81.75296% | 0–0.03125 raw / 0.5–0.51562 norm: 7,885 (82.0%) |
| `modulation_17_power` | 9,618 | -1.66223–10 | 0.01561 / 0.15635 / 0.29709 | 99.89603% | 99.89603% | 0–0.3125 raw / 0.5–0.51562 norm: 9,609 (99.9%) |
| `modulation_17_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_17_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 18

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 16.8% (1,615/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_18_bipolar` | 9,618 | 2 | 0 (Off): 9,446 (98.2%); 1 (On): 172 (1.8%) |
| `modulation_18_bypass` | 9,618 | 2 | 0 (Off): 9,608 (99.9%); 1 (On): 10 (0.1%) |
| `modulation_18_stereo` | 9,618 | 2 | 0 (Off): 9,603 (99.8%); 1 (On): 15 (0.2%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_18_amount` | 9,618 | -1–1 | 0.0006774 / 0.01754 / 0.41519 | 83.16698% | 83.16698% | 0–0.03125 raw / 0.5–0.51562 norm: 8,020 (83.4%) |
| `modulation_18_power` | 9,618 | -10–1.4 | 0.01551 / 0.15625 / 0.29699 | 99.90643% | 99.90643% | 0–0.3125 raw / 0.5–0.51562 norm: 9,609 (99.9%) |
| `modulation_18_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_18_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 19

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 15.2% (1,465/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_19_bipolar` | 9,618 | 2 | 0 (Off): 9,447 (98.2%); 1 (On): 171 (1.8%) |
| `modulation_19_bypass` | 9,618 | 2 | 0 (Off): 9,608 (99.9%); 1 (On): 10 (0.1%) |
| `modulation_19_stereo` | 9,618 | 2 | 0 (Off): 9,602 (99.8%); 1 (On): 16 (0.2%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_19_amount` | 9,618 | -1–1 | 0.0007012 / 0.01721 / 0.39956 | 84.88251% | 84.88251% | 0–0.03125 raw / 0.5–0.51562 norm: 8,194 (85.2%) |
| `modulation_19_power` | 9,618 | -10–10 | 0.01545 / 0.15622 / 0.29699 | 99.85444% | 99.85444% | 0–0.3125 raw / 0.5–0.51562 norm: 9,607 (99.9%) |
| `modulation_19_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_19_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 20

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 14.0% (1,348/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_20_bipolar` | 9,618 | 2 | 0 (Off): 9,470 (98.5%); 1 (On): 148 (1.5%) |
| `modulation_20_bypass` | 9,618 | 2 | 0 (Off): 9,607 (99.9%); 1 (On): 11 (0.1%) |
| `modulation_20_stereo` | 9,618 | 2 | 0 (Off): 9,605 (99.9%); 1 (On): 13 (0.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_20_amount` | 9,618 | -1–1 | 0.0007557 / 0.01704 / 0.30884 | 86.13017% | 86.13017% | 0–0.03125 raw / 0.5–0.51562 norm: 8,306 (86.4%) |
| `modulation_20_power` | 9,618 | -9.86843–10 | 0.01554 / 0.15623 / 0.29693 | 99.93762% | 99.93762% | 0–0.3125 raw / 0.5–0.51562 norm: 9,612 (99.9%) |
| `modulation_20_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_20_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 21

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 12.6% (1,214/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_21_bipolar` | 9,618 | 2 | 0 (Off): 9,487 (98.6%); 1 (On): 131 (1.4%) |
| `modulation_21_bypass` | 9,618 | 2 | 0 (Off): 9,609 (99.9%); 1 (On): 9 (0.1%) |
| `modulation_21_stereo` | 9,618 | 2 | 0 (Off): 9,607 (99.9%); 1 (On): 11 (0.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_21_amount` | 9,618 | -1–1 | 0.0008214 / 0.01692 / 0.28134 | 87.09711% | 87.09711% | 0–0.03125 raw / 0.5–0.51562 norm: 8,402 (87.4%) |
| `modulation_21_power` | 9,618 | -3.31313–10 | 0.01554 / 0.15623 / 0.29693 | 99.93762% | 99.93762% | 0–0.3125 raw / 0.5–0.51562 norm: 9,612 (99.9%) |
| `modulation_21_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_21_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 22

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 11.8% (1,134/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_22_bipolar` | 9,618 | 2 | 0 (Off): 9,481 (98.6%); 1 (On): 137 (1.4%) |
| `modulation_22_bypass` | 9,618 | 2 | 0 (Off): 9,609 (99.9%); 1 (On): 9 (0.1%) |
| `modulation_22_stereo` | 9,618 | 2 | 0 (Off): 9,607 (99.9%); 1 (On): 11 (0.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_22_amount` | 9,618 | -1–1 | 0.0007922 / 0.01667 / 0.26331 | 88.36556% | 88.36556% | 0–0.03125 raw / 0.5–0.51562 norm: 8,515 (88.5%) |
| `modulation_22_power` | 9,618 | -10–9.87674 | 0.01535 / 0.15614 / 0.29692 | 99.85444% | 99.85444% | 0–0.3125 raw / 0.5–0.51562 norm: 9,606 (99.9%) |
| `modulation_22_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_22_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 23

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 11.1% (1,072/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_23_bipolar` | 9,618 | 2 | 0 (Off): 9,427 (98.0%); 1 (On): 191 (2.0%) |
| `modulation_23_bypass` | 9,618 | 2 | 0 (Off): 9,612 (99.9%); 1 (On): 6 (0.1%) |
| `modulation_23_stereo` | 9,618 | 2 | 0 (Off): 9,605 (99.9%); 1 (On): 13 (0.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_23_amount` | 9,618 | -1–1 | 0.000939 / 0.0167 / 0.24442 | 89.10376% | 89.10376% | 0–0.03125 raw / 0.5–0.51562 norm: 8,581 (89.2%) |
| `modulation_23_power` | 9,618 | -10–10 | 0.01547 / 0.15618 / 0.2969 | 99.92722% | 99.92722% | 0–0.3125 raw / 0.5–0.51562 norm: 9,611 (99.9%) |
| `modulation_23_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_23_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 24

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 10.2% (983/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_24_bipolar` | 9,618 | 2 | 0 (Off): 9,512 (98.9%); 1 (On): 106 (1.1%) |
| `modulation_24_bypass` | 9,618 | 2 | 0 (Off): 9,610 (99.9%); 1 (On): 8 (0.1%) |
| `modulation_24_stereo` | 9,618 | 2 | 0 (Off): 9,607 (99.9%); 1 (On): 11 (0.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_24_amount` | 9,618 | -1–1 | 0.00112 / 0.01674 / 0.23984 | 89.89395% | 89.89395% | 0–0.03125 raw / 0.5–0.51562 norm: 8,656 (90.0%) |
| `modulation_24_power` | 9,618 | -10–3.42159 | 0.01556 / 0.15622 / 0.29687 | 99.96881% | 99.96881% | 0–0.3125 raw / 0.5–0.51562 norm: 9,615 (100.0%) |
| `modulation_24_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_24_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 25

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 9.5% (917/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_25_bipolar` | 9,618 | 2 | 0 (Off): 9,489 (98.7%); 1 (On): 129 (1.3%) |
| `modulation_25_bypass` | 9,618 | 2 | 0 (Off): 9,611 (99.9%); 1 (On): 7 (0.1%) |
| `modulation_25_stereo` | 9,618 | 2 | 0 (Off): 9,607 (99.9%); 1 (On): 11 (0.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_25_amount` | 9,618 | -1–1 | 0.00104 / 0.01655 / 0.20747 | 90.58016% | 90.58016% | 0–0.03125 raw / 0.5–0.51562 norm: 8,719 (90.7%) |
| `modulation_25_power` | 9,618 | -10–9.87674 | 0.01553 / 0.15622 / 0.2969 | 99.94801% | 99.94801% | 0–0.3125 raw / 0.5–0.51562 norm: 9,613 (99.9%) |
| `modulation_25_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_25_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 26

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 8.7% (841/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_26_bipolar` | 9,618 | 2 | 0 (Off): 9,528 (99.1%); 1 (On): 90 (0.9%) |
| `modulation_26_bypass` | 9,618 | 2 | 0 (Off): 9,610 (99.9%); 1 (On): 8 (0.1%) |
| `modulation_26_stereo` | 9,618 | 2 | 0 (Off): 9,606 (99.9%); 1 (On): 12 (0.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_26_amount` | 9,618 | -1–1 | 0.00117 / 0.01657 / 0.18162 | 91.11042% | 91.11042% | 0–0.03125 raw / 0.5–0.51562 norm: 8,782 (91.3%) |
| `modulation_26_power` | 9,618 | -2.31919–10 | 0.01556 / 0.15623 / 0.2969 | 99.95841% | 99.95841% | 0–0.3125 raw / 0.5–0.51562 norm: 9,614 (100.0%) |
| `modulation_26_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_26_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 27

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 8.1% (778/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_27_bipolar` | 9,618 | 2 | 0 (Off): 9,523 (99.0%); 1 (On): 95 (1.0%) |
| `modulation_27_bypass` | 9,618 | 2 | 0 (Off): 9,610 (99.9%); 1 (On): 8 (0.1%) |
| `modulation_27_stereo` | 9,618 | 2 | 0 (Off): 9,604 (99.9%); 1 (On): 14 (0.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_27_amount` | 9,618 | -1–1 | 0.00111 / 0.01643 / 0.14919 | 91.70306% | 91.70306% | 0–0.03125 raw / 0.5–0.51562 norm: 8,830 (91.8%) |
| `modulation_27_power` | 9,618 | -10–10 | 0.01557 / 0.15625 / 0.29693 | 99.94801% | 99.94801% | 0–0.3125 raw / 0.5–0.51562 norm: 9,613 (99.9%) |
| `modulation_27_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_27_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 28

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 7.6% (727/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_28_bipolar` | 9,618 | 2 | 0 (Off): 9,520 (99.0%); 1 (On): 98 (1.0%) |
| `modulation_28_bypass` | 9,618 | 2 | 0 (Off): 9,615 (100.0%); 1 (On): 3 (0.0%) |
| `modulation_28_stereo` | 9,618 | 2 | 0 (Off): 9,602 (99.8%); 1 (On): 16 (0.2%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_28_amount` | 9,618 | -1–1 | 0.00122 / 0.01639 / 0.12266 | 92.62841% | 92.62841% | 0–0.03125 raw / 0.5–0.51562 norm: 8,918 (92.7%) |
| `modulation_28_power` | 9,618 | -4.20513–10 | 0.01553 / 0.15622 / 0.2969 | 99.94801% | 99.94801% | 0–0.3125 raw / 0.5–0.51562 norm: 9,613 (99.9%) |
| `modulation_28_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_28_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 29

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 7.0% (672/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_29_bipolar` | 9,618 | 2 | 0 (Off): 9,485 (98.6%); 1 (On): 133 (1.4%) |
| `modulation_29_bypass` | 9,618 | 2 | 0 (Off): 9,611 (99.9%); 1 (On): 7 (0.1%) |
| `modulation_29_stereo` | 9,618 | 2 | 0 (Off): 9,605 (99.9%); 1 (On): 13 (0.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_29_amount` | 9,618 | -1–1 | 0.00124 / 0.01635 / 0.10454 | 92.97151% | 92.97151% | 0–0.03125 raw / 0.5–0.51562 norm: 8,948 (93.0%) |
| `modulation_29_power` | 9,618 | -4.62564–0 | 0.01553 / 0.15618 / 0.29684 | 99.96881% | 99.96881% | 0–0.3125 raw / 0.5–0.51562 norm: 9,615 (100.0%) |
| `modulation_29_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_29_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 30

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 6.5% (622/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_30_bipolar` | 9,618 | 2 | 0 (Off): 9,536 (99.1%); 1 (On): 82 (0.9%) |
| `modulation_30_bypass` | 9,618 | 2 | 0 (Off): 9,612 (99.9%); 1 (On): 6 (0.1%) |
| `modulation_30_stereo` | 9,618 | 2 | 0 (Off): 9,613 (99.9%); 1 (On): 5 (0.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_30_amount` | 9,618 | -1–1 | 0.00117 / 0.01621 / 0.03124 | 93.44978% | 93.44978% | 0–0.03125 raw / 0.5–0.51562 norm: 8,997 (93.5%) |
| `modulation_30_power` | 9,618 | 0–1.64697 | 0.01562 / 0.15625 / 0.29688 | 99.9896% | 99.9896% | 0–0.3125 raw / 0.5–0.51562 norm: 9,617 (100.0%) |
| `modulation_30_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_30_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 31

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 6.0% (577/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_31_bipolar` | 9,618 | 2 | 0 (Off): 9,553 (99.3%); 1 (On): 65 (0.7%) |
| `modulation_31_bypass` | 9,618 | 2 | 0 (Off): 9,608 (99.9%); 1 (On): 10 (0.1%) |
| `modulation_31_stereo` | 9,618 | 2 | 0 (Off): 9,609 (99.9%); 1 (On): 9 (0.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_31_amount` | 9,618 | -1–1 | 0.00119 / 0.01614 / 0.03108 | 94.05282% | 94.05282% | 0–0.03125 raw / 0.5–0.51562 norm: 9,051 (94.1%) |
| `modulation_31_power` | 9,618 | -2.48485–1.37167 | 0.01556 / 0.15622 / 0.29687 | 99.96881% | 99.96881% | 0–0.3125 raw / 0.5–0.51562 norm: 9,615 (100.0%) |
| `modulation_31_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_31_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 32

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 5.6% (540/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_32_bipolar` | 9,618 | 2 | 0 (Off): 9,536 (99.1%); 1 (On): 82 (0.9%) |
| `modulation_32_bypass` | 9,618 | 2 | 0 (Off): 9,614 (100.0%); 1 (On): 4 (0.0%) |
| `modulation_32_stereo` | 9,618 | 2 | 0 (Off): 9,609 (99.9%); 1 (On): 9 (0.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_32_amount` | 9,618 | -1–1 | 0.00129 / 0.0162 / 0.03112 | 94.23997% | 94.23997% | 0–0.03125 raw / 0.5–0.51562 norm: 9,067 (94.3%) |
| `modulation_32_power` | 9,618 | -3.42912–0 | 0.01559 / 0.15622 / 0.29684 | 99.9896% | 99.9896% | 0–0.3125 raw / 0.5–0.51562 norm: 9,617 (100.0%) |
| `modulation_32_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_32_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 33

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 5.1% (492/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_33_bipolar` | 9,618 | 2 | 0 (Off): 9,546 (99.3%); 1 (On): 72 (0.7%) |
| `modulation_33_bypass` | 9,618 | 2 | 0 (Off): 9,610 (99.9%); 1 (On): 8 (0.1%) |
| `modulation_33_stereo` | 9,618 | 2 | 0 (Off): 9,613 (99.9%); 1 (On): 5 (0.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_33_amount` | 9,618 | -1–1 | 0.00124 / 0.01609 / 0.03095 | 94.60387% | 94.60387% | 0–0.03125 raw / 0.5–0.51562 norm: 9,102 (94.6%) |
| `modulation_33_power` | 9,618 | 0–0.76014 | 0.01562 / 0.15625 / 0.29688 | 99.9896% | 99.9896% | 0–0.3125 raw / 0.5–0.51562 norm: 9,617 (100.0%) |
| `modulation_33_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_33_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 34

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 4.8% (457/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_34_bipolar` | 9,618 | 2 | 0 (Off): 9,555 (99.3%); 1 (On): 63 (0.7%) |
| `modulation_34_bypass` | 9,618 | 2 | 0 (Off): 9,614 (100.0%); 1 (On): 4 (0.0%) |
| `modulation_34_stereo` | 9,618 | 2 | 0 (Off): 9,614 (100.0%); 1 (On): 4 (0.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_34_amount` | 9,618 | -1–1 | 0.00129 / 0.0161 / 0.0309 | 94.89499% | 94.89499% | 0–0.03125 raw / 0.5–0.51562 norm: 9,133 (95.0%) |
| `modulation_34_power` | 9,618 | -4.80404–9.83434 | 0.01553 / 0.1562 / 0.29687 | 99.95841% | 99.95841% | 0–0.3125 raw / 0.5–0.51562 norm: 9,614 (100.0%) |
| `modulation_34_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_34_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 35

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 4.1% (398/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_35_bipolar` | 9,618 | 2 | 0 (Off): 9,553 (99.3%); 1 (On): 65 (0.7%) |
| `modulation_35_bypass` | 9,618 | 2 | 0 (Off): 9,615 (100.0%); 1 (On): 3 (0.0%) |
| `modulation_35_stereo` | 9,618 | 2 | 0 (Off): 9,611 (99.9%); 1 (On): 7 (0.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_35_amount` | 9,618 | -1–1 | 0.00132 / 0.01606 / 0.03079 | 95.40445% | 95.40445% | 0–0.03125 raw / 0.5–0.51562 norm: 9,179 (95.4%) |
| `modulation_35_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_35_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_35_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 36

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 3.9% (371/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_36_bipolar` | 9,618 | 2 | 0 (Off): 9,564 (99.4%); 1 (On): 54 (0.6%) |
| `modulation_36_bypass` | 9,618 | 2 | 0 (Off): 9,613 (99.9%); 1 (On): 5 (0.1%) |
| `modulation_36_stereo` | 9,618 | 2 | 0 (Off): 9,615 (100.0%); 1 (On): 3 (0.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_36_amount` | 9,618 | -1–1 | 0.00129 / 0.01595 / 0.03062 | 95.82034% | 95.82034% | 0–0.03125 raw / 0.5–0.51562 norm: 9,221 (95.9%) |
| `modulation_36_power` | 9,618 | 0–10 | 0.01562 / 0.15625 / 0.29688 | 99.9896% | 99.9896% | 0–0.3125 raw / 0.5–0.51562 norm: 9,617 (100.0%) |
| `modulation_36_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_36_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 37

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 3.6% (348/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_37_bipolar` | 9,618 | 2 | 0 (Off): 9,555 (99.3%); 1 (On): 63 (0.7%) |
| `modulation_37_bypass` | 9,618 | 2 | 0 (Off): 9,616 (100.0%); 1 (On): 2 (0.0%) |
| `modulation_37_stereo` | 9,618 | 2 | 0 (Off): 9,609 (99.9%); 1 (On): 9 (0.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_37_amount` | 9,618 | -1–1 | 0.00133 / 0.01596 / 0.03059 | 96.08027% | 96.08027% | 0–0.03125 raw / 0.5–0.51562 norm: 9,244 (96.1%) |
| `modulation_37_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_37_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_37_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 38

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 3.2% (306/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_38_bipolar` | 9,618 | 2 | 0 (Off): 9,572 (99.5%); 1 (On): 46 (0.5%) |
| `modulation_38_bypass` | 9,618 | 2 | 0 (Off): 9,616 (100.0%); 1 (On): 2 (0.0%) |
| `modulation_38_stereo` | 9,618 | 2 | 0 (Off): 9,611 (99.9%); 1 (On): 7 (0.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_38_amount` | 9,618 | -1–1 | 0.00141 / 0.016 / 0.0306 | 96.26742% | 96.26742% | 0–0.03125 raw / 0.5–0.51562 norm: 9,264 (96.3%) |
| `modulation_38_power` | 9,618 | -3.47879–4.18077 | 0.01556 / 0.15622 / 0.29687 | 99.96881% | 99.96881% | 0–0.3125 raw / 0.5–0.51562 norm: 9,615 (100.0%) |
| `modulation_38_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_38_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 39

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 3.1% (294/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_39_bipolar` | 9,618 | 2 | 0 (Off): 9,573 (99.5%); 1 (On): 45 (0.5%) |
| `modulation_39_bypass` | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |
| `modulation_39_stereo` | 9,618 | 2 | 0 (Off): 9,610 (99.9%); 1 (On): 8 (0.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_39_amount` | 9,618 | -1–1 | 0.00137 / 0.01591 / 0.03044 | 96.6937% | 96.6937% | 0–0.03125 raw / 0.5–0.51562 norm: 9,304 (96.7%) |
| `modulation_39_power` | 9,618 | 0–1.32525 | 0.01562 / 0.15625 / 0.29688 | 99.9896% | 99.9896% | 0–0.3125 raw / 0.5–0.51562 norm: 9,617 (100.0%) |
| `modulation_39_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_39_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 40

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 2.7% (259/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_40_bipolar` | 9,618 | 2 | 0 (Off): 9,568 (99.5%); 1 (On): 50 (0.5%) |
| `modulation_40_bypass` | 9,618 | 2 | 0 (Off): 9,615 (100.0%); 1 (On): 3 (0.0%) |
| `modulation_40_stereo` | 9,618 | 2 | 0 (Off): 9,615 (100.0%); 1 (On): 3 (0.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_40_amount` | 9,618 | -1–1 | 0.00142 / 0.0159 / 0.03038 | 97.07839% | 97.07839% | 0–0.03125 raw / 0.5–0.51562 norm: 9,338 (97.1%) |
| `modulation_40_power` | 9,618 | 0–2 | 0.01562 / 0.15625 / 0.29688 | 99.9896% | 99.9896% | 0–0.3125 raw / 0.5–0.51562 norm: 9,617 (100.0%) |
| `modulation_40_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_40_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 41

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 2.5% (242/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_41_bipolar` | 9,618 | 2 | 0 (Off): 9,563 (99.4%); 1 (On): 55 (0.6%) |
| `modulation_41_bypass` | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |
| `modulation_41_stereo` | 9,618 | 2 | 0 (Off): 9,613 (99.9%); 1 (On): 5 (0.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_41_amount` | 9,618 | -1–1 | 0.00137 / 0.01584 / 0.03031 | 97.14078% | 97.14078% | 0–0.03125 raw / 0.5–0.51562 norm: 9,347 (97.2%) |
| `modulation_41_power` | 9,618 | 0–2 | 0.01562 / 0.15625 / 0.29688 | 99.97921% | 99.97921% | 0–0.3125 raw / 0.5–0.51562 norm: 9,617 (100.0%) |
| `modulation_41_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_41_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 42

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 2.3% (221/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_42_bipolar` | 9,618 | 2 | 0 (Off): 9,573 (99.5%); 1 (On): 45 (0.5%) |
| `modulation_42_bypass` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_42_stereo` | 9,618 | 2 | 0 (Off): 9,612 (99.9%); 1 (On): 6 (0.1%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_42_amount` | 9,618 | -1–1 | 0.00142 / 0.01583 / 0.03025 | 97.48388% | 97.48388% | 0–0.03125 raw / 0.5–0.51562 norm: 9,383 (97.6%) |
| `modulation_42_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_42_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_42_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 43

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 2.0% (190/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_43_bipolar` | 9,618 | 2 | 0 (Off): 9,586 (99.7%); 1 (On): 32 (0.3%) |
| `modulation_43_bypass` | 9,618 | 2 | 0 (Off): 9,616 (100.0%); 1 (On): 2 (0.0%) |
| `modulation_43_stereo` | 9,618 | 2 | 0 (Off): 9,616 (100.0%); 1 (On): 2 (0.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_43_amount` | 9,618 | -1–1 | 0.00142 / 0.01581 / 0.03021 | 97.68143% | 97.68143% | 0–0.03125 raw / 0.5–0.51562 norm: 9,397 (97.7%) |
| `modulation_43_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_43_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_43_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 44

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 2.0% (188/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_44_bipolar` | 9,618 | 2 | 0 (Off): 9,583 (99.6%); 1 (On): 35 (0.4%) |
| `modulation_44_bypass` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_44_stereo` | 9,618 | 2 | 0 (Off): 9,616 (100.0%); 1 (On): 2 (0.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_44_amount` | 9,618 | -1–1 | 0.00145 / 0.01584 / 0.03023 | 97.69183% | 97.69183% | 0–0.03125 raw / 0.5–0.51562 norm: 9,396 (97.7%) |
| `modulation_44_power` | 9,618 | 0–10 | 0.01562 / 0.15625 / 0.29688 | 99.9896% | 99.9896% | 0–0.3125 raw / 0.5–0.51562 norm: 9,617 (100.0%) |
| `modulation_44_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_44_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 45

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 1.6% (157/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_45_bipolar` | 9,618 | 2 | 0 (Off): 9,581 (99.6%); 1 (On): 37 (0.4%) |
| `modulation_45_bypass` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_45_stereo` | 9,618 | 2 | 0 (Off): 9,615 (100.0%); 1 (On): 3 (0.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_45_amount` | 9,618 | -1–1 | 0.00146 / 0.01581 / 0.03016 | 97.98295% | 97.98295% | 0–0.03125 raw / 0.5–0.51562 norm: 9,424 (98.0%) |
| `modulation_45_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_45_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_45_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 46

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 1.5% (145/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_46_bipolar` | 9,618 | 2 | 0 (Off): 9,578 (99.6%); 1 (On): 40 (0.4%) |
| `modulation_46_bypass` | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |
| `modulation_46_stereo` | 9,618 | 2 | 0 (Off): 9,614 (100.0%); 1 (On): 4 (0.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_46_amount` | 9,618 | -0.99–1 | 0.00148 / 0.01581 / 0.03014 | 98.10771% | 98.10771% | 0–0.03125 raw / 0.5–0.51562 norm: 9,438 (98.1%) |
| `modulation_46_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_46_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_46_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 47

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 1.4% (135/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_47_bipolar` | 9,618 | 2 | 0 (Off): 9,596 (99.8%); 1 (On): 22 (0.2%) |
| `modulation_47_bypass` | 9,618 | 2 | 0 (Off): 9,616 (100.0%); 1 (On): 2 (0.0%) |
| `modulation_47_stereo` | 9,618 | 2 | 0 (Off): 9,614 (100.0%); 1 (On): 4 (0.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_47_amount` | 9,618 | -1–1 | 0.00151 / 0.01581 / 0.03011 | 98.30526% | 98.30526% | 0–0.03125 raw / 0.5–0.51562 norm: 9,456 (98.3%) |
| `modulation_47_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_47_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_47_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 48

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 1.4% (130/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_48_bipolar` | 9,618 | 2 | 0 (Off): 9,578 (99.6%); 1 (On): 40 (0.4%) |
| `modulation_48_bypass` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_48_stereo` | 9,618 | 2 | 0 (Off): 9,614 (100.0%); 1 (On): 4 (0.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_48_amount` | 9,618 | -1–1 | 0.00146 / 0.01575 / 0.03004 | 98.35725% | 98.35725% | 0–0.03125 raw / 0.5–0.51562 norm: 9,464 (98.4%) |
| `modulation_48_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_48_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_48_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 49

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 1.2% (114/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_49_bipolar` | 9,618 | 2 | 0 (Off): 9,589 (99.7%); 1 (On): 29 (0.3%) |
| `modulation_49_bypass` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_49_stereo` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_49_amount` | 9,618 | -1–1 | 0.0015 / 0.01575 / 0.03001 | 98.61718% | 98.61718% | 0–0.03125 raw / 0.5–0.51562 norm: 9,488 (98.6%) |
| `modulation_49_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_49_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_49_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 50

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 1.1% (102/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_50_bipolar` | 9,618 | 2 | 0 (Off): 9,596 (99.8%); 1 (On): 22 (0.2%) |
| `modulation_50_bypass` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_50_stereo` | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_50_amount` | 9,618 | -1–1 | 0.0015 / 0.01575 / 0.02999 | 98.68996% | 98.68996% | 0–0.03125 raw / 0.5–0.51562 norm: 9,493 (98.7%) |
| `modulation_50_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_50_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_50_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 51

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 0.9% (91/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_51_bipolar` | 9,618 | 2 | 0 (Off): 9,593 (99.7%); 1 (On): 25 (0.3%) |
| `modulation_51_bypass` | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |
| `modulation_51_stereo` | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_51_amount` | 9,618 | -1–1 | 0.00152 / 0.01575 / 0.02997 | 98.83552% | 98.83552% | 0–0.03125 raw / 0.5–0.51562 norm: 9,507 (98.8%) |
| `modulation_51_power` | 9,618 | -0.12582–0 | 0.01559 / 0.15622 / 0.29684 | 99.9896% | 99.9896% | 0–0.3125 raw / 0.5–0.51562 norm: 9,617 (100.0%) |
| `modulation_51_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_51_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 52

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 0.8% (77/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_52_bipolar` | 9,618 | 2 | 0 (Off): 9,595 (99.8%); 1 (On): 23 (0.2%) |
| `modulation_52_bypass` | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |
| `modulation_52_stereo` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_52_amount` | 9,618 | -1–1 | 0.00145 / 0.01565 / 0.02985 | 98.99147% | 98.99147% | 0–0.03125 raw / 0.5–0.51562 norm: 9,522 (99.0%) |
| `modulation_52_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_52_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_52_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 53

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 0.8% (74/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_53_bipolar` | 9,618 | 2 | 0 (Off): 9,595 (99.8%); 1 (On): 23 (0.2%) |
| `modulation_53_bypass` | 9,618 | 2 | 0 (Off): 9,616 (100.0%); 1 (On): 2 (0.0%) |
| `modulation_53_stereo` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_53_amount` | 9,618 | -1–1 | 0.00149 / 0.01569 / 0.02989 | 99.02267% | 99.02267% | 0–0.03125 raw / 0.5–0.51562 norm: 9,524 (99.0%) |
| `modulation_53_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_53_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_53_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 54

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 0.6% (59/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_54_bipolar` | 9,618 | 2 | 0 (Off): 9,605 (99.9%); 1 (On): 13 (0.1%) |
| `modulation_54_bypass` | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |
| `modulation_54_stereo` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_54_amount` | 9,618 | -1–1 | 0.00151 / 0.01568 / 0.02985 | 99.20981% | 99.20981% | 0–0.03125 raw / 0.5–0.51562 norm: 9,542 (99.2%) |
| `modulation_54_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_54_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_54_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 55

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 0.5% (49/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_55_bipolar` | 9,618 | 2 | 0 (Off): 9,613 (99.9%); 1 (On): 5 (0.1%) |
| `modulation_55_bypass` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_55_stereo` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_55_amount` | 9,618 | -1–1 | 0.00152 / 0.01568 / 0.02984 | 99.29299% | 99.29299% | 0–0.03125 raw / 0.5–0.51562 norm: 9,550 (99.3%) |
| `modulation_55_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_55_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_55_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 56

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 0.5% (44/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_56_bipolar` | 9,618 | 2 | 0 (Off): 9,610 (99.9%); 1 (On): 8 (0.1%) |
| `modulation_56_bypass` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_56_stereo` | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_56_amount` | 9,618 | -1–1 | 0.00154 / 0.01569 / 0.02984 | 99.35538% | 99.35538% | 0–0.03125 raw / 0.5–0.51562 norm: 9,556 (99.4%) |
| `modulation_56_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_56_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_56_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 57

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 0.5% (45/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_57_bipolar` | 9,618 | 2 | 0 (Off): 9,610 (99.9%); 1 (On): 8 (0.1%) |
| `modulation_57_bypass` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_57_stereo` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_57_amount` | 9,618 | -0.54317–1 | 0.00156 / 0.0157 / 0.02984 | 99.42816% | 99.42816% | 0–0.03125 raw / 0.5–0.51562 norm: 9,563 (99.4%) |
| `modulation_57_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_57_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_57_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 58

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 0.4% (37/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_58_bipolar` | 9,618 | 2 | 0 (Off): 9,613 (99.9%); 1 (On): 5 (0.1%) |
| `modulation_58_bypass` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_58_stereo` | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_58_amount` | 9,618 | -1–1 | 0.00155 / 0.01568 / 0.02982 | 99.48014% | 99.48014% | 0–0.03125 raw / 0.5–0.51562 norm: 9,568 (99.5%) |
| `modulation_58_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_58_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_58_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 59

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 0.3% (29/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_59_bipolar` | 9,618 | 2 | 0 (Off): 9,595 (99.8%); 1 (On): 23 (0.2%) |
| `modulation_59_bypass` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_59_stereo` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_59_amount` | 9,618 | -0.41674–0.91944 | 0.00154 / 0.01567 / 0.0298 | 99.49054% | 99.49054% | 0–0.03125 raw / 0.5–0.51562 norm: 9,570 (99.5%) |
| `modulation_59_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_59_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_59_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 60

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 0.3% (29/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_60_bipolar` | 9,618 | 2 | 0 (Off): 9,615 (100.0%); 1 (On): 3 (0.0%) |
| `modulation_60_bypass` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_60_stereo` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_60_amount` | 9,618 | -0.13459–1 | 0.00156 / 0.01568 / 0.0298 | 99.59451% | 99.59451% | 0–0.03125 raw / 0.5–0.51562 norm: 9,579 (99.6%) |
| `modulation_60_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_60_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_60_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 61

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 0.2% (19/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_61_bipolar` | 9,618 | 2 | 0 (Off): 9,612 (99.9%); 1 (On): 6 (0.1%) |
| `modulation_61_bypass` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_61_stereo` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_61_amount` | 9,618 | -0.4384–1 | 0.00154 / 0.01565 / 0.02976 | 99.6361% | 99.6361% | 0–0.03125 raw / 0.5–0.51562 norm: 9,583 (99.6%) |
| `modulation_61_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_61_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_61_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 62

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 0.2% (18/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_62_bipolar` | 9,618 | 2 | 0 (Off): 9,615 (100.0%); 1 (On): 3 (0.0%) |
| `modulation_62_bypass` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_62_stereo` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_62_amount` | 9,618 | -1–1 | 0.00155 / 0.01566 / 0.02976 | 99.68808% | 99.68808% | 0–0.03125 raw / 0.5–0.51562 norm: 9,588 (99.7%) |
| `modulation_62_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_62_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_62_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 63

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 0.2% (16/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_63_bipolar` | 9,618 | 2 | 0 (Off): 9,608 (99.9%); 1 (On): 10 (0.1%) |
| `modulation_63_bypass` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_63_stereo` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_63_amount` | 9,618 | -1–0.75 | 0.00153 / 0.01563 / 0.02973 | 99.70888% | 99.70888% | 0–0.03125 raw / 0.5–0.51562 norm: 9,590 (99.7%) |
| `modulation_63_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_63_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_63_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

### Modulation slot 64

7 scalar parameters: 3 categorical/enum and 4 continuous. routed 0.2% (15/9,618); changed 0.0% (0/9,618).

**Categorical / enum value frequencies**

| Parameter | Observed | Distinct | Modal values |
|---|---:|---:|---|
| `modulation_64_bipolar` | 9,618 | 2 | 0 (Off): 9,615 (100.0%); 1 (On): 3 (0.0%) |
| `modulation_64_bypass` | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |
| `modulation_64_stereo` | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |

**Continuous value distributions**

| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |
|---|---:|---|---|---:|---:|---|
| `modulation_64_amount` | 9,618 | -0.58414–1 | 0.00156 / 0.01566 / 0.02976 | 99.72967% | 99.72967% | 0–0.03125 raw / 0.5–0.51562 norm: 9,592 (99.7%) |
| `modulation_64_power` | 9,618 | 0–0 | 0.01562 / 0.15623 / 0.29684 | 100% | 100% | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_64_ramp_down` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |
| `modulation_64_ramp_up` | 287 | -10–-10 | -10 / -10 / -10 | — | 0% | -10–-10 raw: 287 (100.0%) |

**Modulation-matrix relationship analysis**

The corpus contains **93,308** connected routes, **86,212** live routes, **416** bypassed routes, **6,693** connected zero-amount routes, and **1,709** custom remaps. Source/destination family pairs are shown in the modulation matrix figure; each slot's scalar controls are listed above.

## Interpretation cautions

- A categorical mode is a storage-value mode, not a claim that the corresponding UI choice is perceptually dominant.
- A continuous dominant bin is a range, not an exact mode. This avoids pretending that arbitrary floating-point values have exact categorical semantics.
- The normalized histogram follows Vital's raw control scale. It is useful for comparing differently ranged controls, while the raw summary preserves the actual serialized values.
- Wavetable and sample payloads are not included. Wavetable editor component-type counts are structural occurrences only; they do not expose embedded waveform/audio data.
- These aggregates contain no preset paths, names, authors, raw wavetable/sample payloads, or per-file records.
