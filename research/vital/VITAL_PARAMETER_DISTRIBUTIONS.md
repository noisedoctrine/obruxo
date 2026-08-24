# Vital Parameter Value Distributions

This companion report uses the **file weighted** aggregate (9,618 parsed presets). It adds value-level frequencies and distributions to the prevalence census.

The row-oriented exports are the complete machine-readable detail: [`vital_usage_categorical_values.csv`](vital_usage_categorical_values.csv) contains one row per categorical value, and [`vital_usage_continuous_bins.csv`](vital_usage_continuous_bins.csv) contains all 64 histogram bins for each continuous parameter, with both file-weighted and exact-deduplicated counts.

Categorical values are Vital's raw numeric ordinals. Labels are included when the pinned atlas provides an option list. Their denominator is the observed value count under the existing policy that fills missing common scalar keys with the atlas default.

Continuous parameters include raw-value min/max/mean/stddev, histogram quantiles, default and zero prevalence, and dominant bins. Atlas-backed controls use 64 bins in normalized control position, preserving the parameter's scale metadata; version-introduced controls without atlas bounds use adaptive raw-value bins. 16 atlas-backed controls also use complete raw-value bins because their corpus observations exceed the pinned bounds; their partial normalized histograms remain in the JSON. For an `Exponential` parameter, the raw Vital value is already the logarithmic storage domain.

## Categorical and enum frequencies (351 parameters)

The table shows every parameter's most frequent values; the CSV retains every observed ordinal, including the tail.

| Parameter | Scale | Observed | Distinct | Modal values |
|---|---|---:|---:|---|
| `bypass` | Indexed | 9,618 | 1 | 0: 9,618 (100.0%) |
| `chorus_on` | Indexed | 9,618 | 2 | 1 (On): 4,835 (50.3%); 0 (Off): 4,783 (49.7%) |
| `chorus_sync` | Indexed | 9,618 | 4 | 1 (Tempo): 9,378 (97.5%); 0 (Seconds): 213 (2.2%); 2 (Tempo Dotted): 16 (0.2%); 3 (Tempo Triplets): 11 (0.1%) |
| `chorus_tempo` | Indexed | 9,618 | 11 | 4 (4/1): 7,470 (77.7%); 0 (Freeze): 848 (8.8%); 3 (8/1): 464 (4.8%); 2 (16/1): 223 (2.3%); 5 (2/1): 197 (2.0%); 6 (1/1): 146 (1.5%); 1 (32/1): 123 (1.3%); 7 (1/2): 55 (0.6%); … 3 more in CSV |
| `chorus_voices` | Indexed | 9,618 | 5 | 4: 7,196 (74.8%); 2: 996 (10.4%); 1: 992 (10.3%); 3: 433 (4.5%); 1.5: 1 (0.0%) |
| `compressor_enabled_bands` | Indexed | 9,618 | 4 | 0 (Multiband): 8,301 (86.3%); 3 (Single Band): 763 (7.9%); 2 (High Band): 343 (3.6%); 1 (Low Band): 211 (2.2%) |
| `compressor_on` | Indexed | 9,618 | 2 | 1 (On): 6,738 (70.1%); 0 (Off): 2,880 (29.9%) |
| `delay_aux_sync` | Indexed | 9,618 | 4 | 1 (Tempo): 8,646 (89.9%); 2 (Tempo Dotted): 651 (6.8%); 0 (Seconds): 217 (2.3%); 3 (Tempo Triplets): 104 (1.1%) |
| `delay_aux_tempo` | Indexed | 9,618 | 9 | 9 (2/1): 8,041 (83.6%); 8 (4/1): 895 (9.3%); 10 (1/1): 367 (3.8%); 7 (8/1): 119 (1.2%); 12 (1/4): 106 (1.1%); 11 (1/2): 70 (0.7%); 6 (16/1): 12 (0.1%); 4 (Freeze): 6 (0.1%); … 1 more in CSV |
| `delay_on` | Indexed | 9,618 | 2 | 0 (Off): 5,710 (59.4%); 1 (On): 3,908 (40.6%) |
| `delay_style` | Indexed | 9,618 | 4 | 0 (Mono): 6,391 (66.4%); 2 (Ping Pong): 2,027 (21.1%); 1 (Stereo): 838 (8.7%); 3 (Mid Ping Pong): 362 (3.8%) |
| `delay_sync` | Indexed | 9,618 | 4 | 1 (Tempo): 8,521 (88.6%); 2 (Tempo Dotted): 547 (5.7%); 0 (Seconds): 430 (4.5%); 3 (Tempo Triplets): 120 (1.2%) |
| `delay_tempo` | Indexed | 9,618 | 9 | 9 (2/1): 7,533 (78.3%); 8 (4/1): 1,069 (11.1%); 10 (1/1): 445 (4.6%); 12 (1/4): 239 (2.5%); 7 (8/1): 156 (1.6%); 11 (1/2): 122 (1.3%); 6 (16/1): 24 (0.2%); 4 (Freeze): 22 (0.2%); … 1 more in CSV |
| `distortion_filter_order` | Indexed | 9,618 | 3 | 0 (None): 7,299 (75.9%); 1 (Pre): 1,262 (13.1%); 2 (Post): 1,057 (11.0%) |
| `distortion_on` | Indexed | 9,618 | 2 | 1 (On): 6,253 (65.0%); 0 (Off): 3,365 (35.0%) |
| `distortion_type` | Indexed | 9,618 | 6 | 0 (Soft Clip): 6,340 (65.9%); 1 (Hard Clip): 1,418 (14.7%); 5 (Down Sample): 731 (7.6%); 3 (Sine Fold): 529 (5.5%); 2 (Linear Fold): 315 (3.3%); 4 (Bit Crush): 285 (3.0%) |
| `effect_chain_order` | Indexed | 9,618 | 2,730 | 0: 3,355 (34.9%); 1.512e+04: 127 (1.3%); 2.57e+05: 115 (1.2%); 7.56e+04: 113 (1.2%); 3.024e+04: 81 (0.8%); 1.814e+05: 72 (0.7%); 3024: 67 (0.7%); 6.048e+04: 61 (0.6%); … 2,722 more in CSV |
| `eq_band_mode` | Indexed | 9,618 | 2 | 0 (Shelf): 9,385 (97.6%); 1 (Notch): 233 (2.4%) |
| `eq_high_mode` | Indexed | 9,618 | 2 | 0 (Shelf): 8,213 (85.4%); 1 (Low Pass): 1,405 (14.6%) |
| `eq_low_mode` | Indexed | 9,618 | 2 | 0 (Shelf): 7,013 (72.9%); 1 (High Pass): 2,605 (27.1%) |
| `eq_on` | Indexed | 9,618 | 2 | 1 (On): 6,186 (64.3%); 0 (Off): 3,432 (35.7%) |
| `filter_1_filter_input` | Indexed | 9,618 | 2 | 0 (Off): 9,287 (96.6%); 1 (On): 331 (3.4%) |
| `filter_1_model` | Indexed | 9,618 | 8 | 0 (Analog): 6,971 (72.5%); 6 (Comb): 667 (6.9%); 3 (Digital): 558 (5.8%); 1 (Dirty): 500 (5.2%); 2 (Ladder): 478 (5.0%); 7 (Phaser): 185 (1.9%); 5 (Formant): 149 (1.5%); 4 (Diode): 110 (1.1%) |
| `filter_1_on` | Indexed | 9,618 | 2 | 1 (On): 7,118 (74.0%); 0 (Off): 2,500 (26.0%) |
| `filter_1_style` | Indexed | 9,618 | 6 | 0 (12dB): 6,278 (65.3%); 1 (24dB): 2,515 (26.1%); 2 (Notch Blend): 275 (2.9%); 4 (B/P/N): 262 (2.7%); 3 (Notch Spread): 241 (2.5%); 5: 47 (0.5%) |
| `filter_2_filter_input` | Indexed | 9,618 | 2 | 0 (Off): 8,131 (84.5%); 1 (On): 1,487 (15.5%) |
| `filter_2_model` | Indexed | 9,618 | 8 | 0 (Analog): 7,575 (78.8%); 6 (Comb): 592 (6.2%); 3 (Digital): 364 (3.8%); 1 (Dirty): 352 (3.7%); 2 (Ladder): 304 (3.2%); 7 (Phaser): 174 (1.8%); 5 (Formant): 166 (1.7%); 4 (Diode): 91 (0.9%) |
| `filter_2_on` | Indexed | 9,618 | 2 | 0 (Off): 5,324 (55.4%); 1 (On): 4,294 (44.6%) |
| `filter_2_style` | Indexed | 9,618 | 6 | 0 (12dB): 7,382 (76.8%); 1 (24dB): 1,405 (14.6%); 2 (Notch Blend): 283 (2.9%); 4 (B/P/N): 248 (2.6%); 3 (Notch Spread): 244 (2.5%); 5: 56 (0.6%) |
| `filter_fx_model` | Indexed | 9,618 | 8 | 0 (Analog): 7,929 (82.4%); 6 (Comb): 418 (4.3%); 3 (Digital): 373 (3.9%); 1 (Dirty): 276 (2.9%); 2 (Ladder): 221 (2.3%); 7 (Phaser): 202 (2.1%); 5 (Formant): 121 (1.3%); 4 (Diode): 78 (0.8%) |
| `filter_fx_on` | Indexed | 9,618 | 2 | 0 (Off): 6,201 (64.5%); 1 (On): 3,417 (35.5%) |
| `filter_fx_style` | Indexed | 9,618 | 6 | 0 (12dB): 7,736 (80.4%); 1 (24dB): 1,367 (14.2%); 2 (Notch Blend): 186 (1.9%); 4 (B/P/N): 174 (1.8%); 3 (Notch Spread): 120 (1.2%); 5: 35 (0.4%) |
| `flanger_on` | Indexed | 9,618 | 2 | 0 (Off): 8,470 (88.1%); 1 (On): 1,148 (11.9%) |
| `flanger_sync` | Indexed | 9,618 | 4 | 1 (Tempo): 9,467 (98.4%); 0 (Seconds): 113 (1.2%); 2 (Tempo Dotted): 22 (0.2%); 3 (Tempo Triplets): 16 (0.2%) |
| `flanger_tempo` | Indexed | 9,618 | 11 | 4 (4/1): 8,673 (90.2%); 0 (Freeze): 465 (4.8%); 3 (8/1): 86 (0.9%); 6 (1/1): 83 (0.9%); 5 (2/1): 80 (0.8%); 7 (1/2): 62 (0.6%); 2 (16/1): 49 (0.5%); 8 (1/4): 40 (0.4%); … 3 more in CSV |
| `legato` | Indexed | 9,618 | 2 | 0 (Off): 7,843 (81.5%); 1 (On): 1,775 (18.5%) |
| `lfo_1_keytrack_transpose` | Indexed | 9,618 | 25 | -12: 9,449 (98.2%); 0: 54 (0.6%); -60: 22 (0.2%); -24: 19 (0.2%); 12: 18 (0.2%); 24: 11 (0.1%); -36: 8 (0.1%); -48: 7 (0.1%); … 17 more in CSV |
| `lfo_1_smooth_mode` | Indexed | 9,618 | 2 | 1 (On): 9,556 (99.4%); 0 (Off): 62 (0.6%) |
| `lfo_1_sync` | Indexed | 9,618 | 5 | 1 (Tempo): 8,097 (84.2%); 0 (Seconds): 1,087 (11.3%); 4 (Keytrack): 160 (1.7%); 2 (Tempo Dotted): 158 (1.6%); 3 (Tempo Triplets): 116 (1.2%) |
| `lfo_1_sync_type` | Indexed | 9,618 | 7 | 0 (Trigger): 6,374 (66.3%); 2 (Envelope): 1,780 (18.5%); 1 (Sync): 1,324 (13.8%); 4 (Loop Point): 61 (0.6%); 3 (Sustain Envelope): 27 (0.3%); 5 (Loop Hold): 27 (0.3%); 6: 25 (0.3%) |
| `lfo_1_tempo` | Indexed | 9,618 | 13 | 7 (1/2): 4,716 (49.0%); 8 (1/4): 1,138 (11.8%); 6 (1/1): 922 (9.6%); 9 (1/8): 773 (8.0%); 5 (2/1): 635 (6.6%); 4 (4/1): 462 (4.8%); 10 (1/16): 433 (4.5%); 12 (1/64): 175 (1.8%); … 5 more in CSV |
| `lfo_2_keytrack_transpose` | Indexed | 9,618 | 24 | -12: 9,506 (98.8%); 0: 36 (0.4%); 12: 25 (0.3%); -24: 16 (0.2%); -60: 5 (0.1%); 36: 5 (0.1%); -36: 3 (0.0%); 7: 3 (0.0%); … 16 more in CSV |
| `lfo_2_smooth_mode` | Indexed | 9,618 | 2 | 1 (On): 9,568 (99.5%); 0 (Off): 50 (0.5%) |
| `lfo_2_sync` | Indexed | 9,618 | 5 | 1 (Tempo): 8,629 (89.7%); 0 (Seconds): 695 (7.2%); 2 (Tempo Dotted): 115 (1.2%); 4 (Keytrack): 100 (1.0%); 3 (Tempo Triplets): 79 (0.8%) |
| `lfo_2_sync_type` | Indexed | 9,618 | 7 | 0 (Trigger): 7,360 (76.5%); 1 (Sync): 1,131 (11.8%); 2 (Envelope): 1,029 (10.7%); 3 (Sustain Envelope): 33 (0.3%); 4 (Loop Point): 28 (0.3%); 6: 21 (0.2%); 5 (Loop Hold): 16 (0.2%) |
| `lfo_2_tempo` | Indexed | 9,618 | 14 | 7 (1/2): 6,373 (66.3%); 6 (1/1): 689 (7.2%); 8 (1/4): 638 (6.6%); 5 (2/1): 474 (4.9%); 4 (4/1): 408 (4.2%); 9 (1/8): 401 (4.2%); 10 (1/16): 244 (2.5%); 3 (8/1): 148 (1.5%); … 6 more in CSV |
| `lfo_3_keytrack_transpose` | Indexed | 9,618 | 18 | -12: 9,543 (99.2%); 12: 18 (0.2%); 0: 14 (0.1%); -24: 9 (0.1%); 24: 8 (0.1%); 19: 4 (0.0%); 36: 4 (0.0%); 7: 4 (0.0%); … 10 more in CSV |
| `lfo_3_smooth_mode` | Indexed | 9,618 | 2 | 1 (On): 9,564 (99.4%); 0 (Off): 54 (0.6%) |
| `lfo_3_sync` | Indexed | 9,618 | 5 | 1 (Tempo): 8,955 (93.1%); 0 (Seconds): 480 (5.0%); 2 (Tempo Dotted): 73 (0.8%); 4 (Keytrack): 68 (0.7%); 3 (Tempo Triplets): 42 (0.4%) |
| `lfo_3_sync_type` | Indexed | 9,618 | 7 | 0 (Trigger): 8,216 (85.4%); 1 (Sync): 748 (7.8%); 2 (Envelope): 598 (6.2%); 4 (Loop Point): 21 (0.2%); 3 (Sustain Envelope): 19 (0.2%); 6: 14 (0.1%); 5 (Loop Hold): 2 (0.0%) |
| `lfo_3_tempo` | Indexed | 9,618 | 13 | 7 (1/2): 7,646 (79.5%); 6 (1/1): 397 (4.1%); 8 (1/4): 378 (3.9%); 5 (2/1): 342 (3.6%); 4 (4/1): 272 (2.8%); 9 (1/8): 199 (2.1%); 10 (1/16): 132 (1.4%); 3 (8/1): 102 (1.1%); … 5 more in CSV |
| `lfo_4_keytrack_transpose` | Indexed | 9,618 | 12 | -12: 9,562 (99.4%); 0: 16 (0.2%); -5: 8 (0.1%); 12: 8 (0.1%); 24: 7 (0.1%); 7: 7 (0.1%); 19: 3 (0.0%); 11: 2 (0.0%); … 4 more in CSV |
| `lfo_4_smooth_mode` | Indexed | 9,618 | 2 | 1 (On): 9,567 (99.5%); 0 (Off): 51 (0.5%) |
| `lfo_4_sync` | Indexed | 9,618 | 5 | 1 (Tempo): 9,203 (95.7%); 0 (Seconds): 273 (2.8%); 2 (Tempo Dotted): 68 (0.7%); 4 (Keytrack): 52 (0.5%); 3 (Tempo Triplets): 22 (0.2%) |
| `lfo_4_sync_type` | Indexed | 9,618 | 6 | 0 (Trigger): 8,746 (90.9%); 1 (Sync): 515 (5.4%); 2 (Envelope): 324 (3.4%); 6: 16 (0.2%); 4 (Loop Point): 9 (0.1%); 3 (Sustain Envelope): 8 (0.1%) |
| `lfo_4_tempo` | Indexed | 9,618 | 13 | 7 (1/2): 8,421 (87.6%); 5 (2/1): 231 (2.4%); 6 (1/1): 229 (2.4%); 8 (1/4): 198 (2.1%); 4 (4/1): 195 (2.0%); 9 (1/8): 112 (1.2%); 3 (8/1): 85 (0.9%); 10 (1/16): 57 (0.6%); … 5 more in CSV |
| `lfo_5_keytrack_transpose` | Indexed | 9,618 | 12 | -12: 9,546 (99.3%); 0: 24 (0.2%); 12: 18 (0.2%); -24: 15 (0.2%); 24: 4 (0.0%); 7: 4 (0.0%); -5: 2 (0.0%); -26: 1 (0.0%); … 4 more in CSV |
| `lfo_5_smooth_mode` | Indexed | 9,618 | 2 | 1 (On): 9,576 (99.6%); 0 (Off): 42 (0.4%) |
| `lfo_5_sync` | Indexed | 9,618 | 5 | 1 (Tempo): 9,350 (97.2%); 0 (Seconds): 169 (1.8%); 4 (Keytrack): 56 (0.6%); 2 (Tempo Dotted): 40 (0.4%); 3 (Tempo Triplets): 3 (0.0%) |
| `lfo_5_sync_type` | Indexed | 9,618 | 6 | 0 (Trigger): 9,121 (94.8%); 1 (Sync): 308 (3.2%); 2 (Envelope): 178 (1.9%); 4 (Loop Point): 4 (0.0%); 6: 4 (0.0%); 3 (Sustain Envelope): 3 (0.0%) |
| `lfo_5_tempo` | Indexed | 9,618 | 13 | 7 (1/2): 8,842 (91.9%); 4 (4/1): 161 (1.7%); 6 (1/1): 159 (1.7%); 5 (2/1): 151 (1.6%); 8 (1/4): 82 (0.9%); 9 (1/8): 73 (0.8%); 3 (8/1): 62 (0.6%); 10 (1/16): 43 (0.4%); … 5 more in CSV |
| `lfo_6_keytrack_transpose` | Indexed | 9,618 | 8 | -12: 9,575 (99.6%); 12: 13 (0.1%); 0: 9 (0.1%); 24: 7 (0.1%); 7: 6 (0.1%); -24: 4 (0.0%); -60: 2 (0.0%); 36: 2 (0.0%) |
| `lfo_6_smooth_mode` | Indexed | 9,618 | 2 | 1 (On): 9,570 (99.5%); 0 (Off): 48 (0.5%) |
| `lfo_6_sync` | Indexed | 9,618 | 5 | 1 (Tempo): 9,460 (98.4%); 0 (Seconds): 100 (1.0%); 4 (Keytrack): 36 (0.4%); 2 (Tempo Dotted): 18 (0.2%); 3 (Tempo Triplets): 4 (0.0%) |
| `lfo_6_sync_type` | Indexed | 9,618 | 4 | 0 (Trigger): 9,311 (96.8%); 1 (Sync): 170 (1.8%); 2 (Envelope): 136 (1.4%); 6: 1 (0.0%) |
| `lfo_6_tempo` | Indexed | 9,618 | 13 | 7 (1/2): 9,167 (95.3%); 4 (4/1): 92 (1.0%); 6 (1/1): 85 (0.9%); 5 (2/1): 84 (0.9%); 8 (1/4): 64 (0.7%); 9 (1/8): 45 (0.5%); 3 (8/1): 30 (0.3%); 10 (1/16): 26 (0.3%); … 5 more in CSV |
| `lfo_7_keytrack_transpose` | Indexed | 9,618 | 5 | -12: 9,599 (99.8%); 0: 11 (0.1%); 12: 5 (0.1%); -24: 2 (0.0%); 24: 1 (0.0%) |
| `lfo_7_smooth_mode` | Indexed | 9,618 | 2 | 1 (On): 9,575 (99.6%); 0 (Off): 43 (0.4%) |
| `lfo_7_sync` | Indexed | 9,618 | 5 | 1 (Tempo): 9,535 (99.1%); 0 (Seconds): 46 (0.5%); 4 (Keytrack): 26 (0.3%); 2 (Tempo Dotted): 8 (0.1%); 3 (Tempo Triplets): 3 (0.0%) |
| `lfo_7_sync_type` | Indexed | 9,618 | 7 | 0 (Trigger): 9,446 (98.2%); 1 (Sync): 100 (1.0%); 2 (Envelope): 63 (0.7%); 6: 4 (0.0%); 3 (Sustain Envelope): 2 (0.0%); 5 (Loop Hold): 2 (0.0%); 4 (Loop Point): 1 (0.0%) |
| `lfo_7_tempo` | Indexed | 9,618 | 13 | 7 (1/2): 9,364 (97.4%); 5 (2/1): 59 (0.6%); 4 (4/1): 47 (0.5%); 6 (1/1): 45 (0.5%); 8 (1/4): 36 (0.4%); 9 (1/8): 23 (0.2%); 3 (8/1): 21 (0.2%); 10 (1/16): 10 (0.1%); … 5 more in CSV |
| `lfo_8_keytrack_transpose` | Indexed | 9,618 | 5 | -12: 9,593 (99.7%); 12: 15 (0.2%); 0: 8 (0.1%); 2: 1 (0.0%); 24: 1 (0.0%) |
| `lfo_8_smooth_mode` | Indexed | 9,618 | 2 | 1 (On): 9,617 (100.0%); 0 (Off): 1 (0.0%) |
| `lfo_8_sync` | Indexed | 9,618 | 5 | 1 (Tempo): 9,556 (99.4%); 4 (Keytrack): 25 (0.3%); 0 (Seconds): 24 (0.2%); 2 (Tempo Dotted): 8 (0.1%); 3 (Tempo Triplets): 5 (0.1%) |
| `lfo_8_sync_type` | Indexed | 9,618 | 6 | 0 (Trigger): 9,535 (99.1%); 1 (Sync): 44 (0.5%); 2 (Envelope): 35 (0.4%); 4 (Loop Point): 2 (0.0%); 3 (Sustain Envelope): 1 (0.0%); 5 (Loop Hold): 1 (0.0%) |
| `lfo_8_tempo` | Indexed | 9,618 | 13 | 7 (1/2): 9,466 (98.4%); 6 (1/1): 37 (0.4%); 5 (2/1): 30 (0.3%); 4 (4/1): 28 (0.3%); 8 (1/4): 21 (0.2%); 9 (1/8): 12 (0.1%); 10 (1/16): 10 (0.1%); 3 (8/1): 9 (0.1%); … 5 more in CSV |
| `modulation_10_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,307 (96.8%); 1 (On): 311 (3.2%) |
| `modulation_10_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,590 (99.7%); 1 (On): 28 (0.3%) |
| `modulation_10_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,588 (99.7%); 1 (On): 30 (0.3%) |
| `modulation_11_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,332 (97.0%); 1 (On): 286 (3.0%) |
| `modulation_11_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,606 (99.9%); 1 (On): 12 (0.1%) |
| `modulation_11_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,584 (99.6%); 1 (On): 34 (0.4%) |
| `modulation_12_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,386 (97.6%); 1 (On): 232 (2.4%) |
| `modulation_12_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,602 (99.8%); 1 (On): 16 (0.2%) |
| `modulation_12_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,576 (99.6%); 1 (On): 42 (0.4%) |
| `modulation_13_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,415 (97.9%); 1 (On): 203 (2.1%) |
| `modulation_13_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,606 (99.9%); 1 (On): 12 (0.1%) |
| `modulation_13_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,603 (99.8%); 1 (On): 15 (0.2%) |
| `modulation_14_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,406 (97.8%); 1 (On): 212 (2.2%) |
| `modulation_14_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,606 (99.9%); 1 (On): 12 (0.1%) |
| `modulation_14_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,587 (99.7%); 1 (On): 31 (0.3%) |
| `modulation_15_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,402 (97.8%); 1 (On): 216 (2.2%) |
| `modulation_15_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,599 (99.8%); 1 (On): 19 (0.2%) |
| `modulation_15_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,597 (99.8%); 1 (On): 21 (0.2%) |
| `modulation_16_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,459 (98.3%); 1 (On): 159 (1.7%) |
| `modulation_16_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,605 (99.9%); 1 (On): 13 (0.1%) |
| `modulation_16_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,581 (99.6%); 1 (On): 37 (0.4%) |
| `modulation_17_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,444 (98.2%); 1 (On): 174 (1.8%) |
| `modulation_17_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,606 (99.9%); 1 (On): 12 (0.1%) |
| `modulation_17_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,589 (99.7%); 1 (On): 29 (0.3%) |
| `modulation_18_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,446 (98.2%); 1 (On): 172 (1.8%) |
| `modulation_18_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,608 (99.9%); 1 (On): 10 (0.1%) |
| `modulation_18_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,603 (99.8%); 1 (On): 15 (0.2%) |
| `modulation_19_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,447 (98.2%); 1 (On): 171 (1.8%) |
| `modulation_19_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,608 (99.9%); 1 (On): 10 (0.1%) |
| `modulation_19_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,602 (99.8%); 1 (On): 16 (0.2%) |
| `modulation_1_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 8,945 (93.0%); 1 (On): 673 (7.0%) |
| `modulation_1_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,552 (99.3%); 1 (On): 66 (0.7%) |
| `modulation_1_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,556 (99.4%); 1 (On): 62 (0.6%) |
| `modulation_20_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,470 (98.5%); 1 (On): 148 (1.5%) |
| `modulation_20_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,607 (99.9%); 1 (On): 11 (0.1%) |
| `modulation_20_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,605 (99.9%); 1 (On): 13 (0.1%) |
| `modulation_21_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,487 (98.6%); 1 (On): 131 (1.4%) |
| `modulation_21_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,609 (99.9%); 1 (On): 9 (0.1%) |
| `modulation_21_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,607 (99.9%); 1 (On): 11 (0.1%) |
| `modulation_22_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,481 (98.6%); 1 (On): 137 (1.4%) |
| `modulation_22_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,609 (99.9%); 1 (On): 9 (0.1%) |
| `modulation_22_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,607 (99.9%); 1 (On): 11 (0.1%) |
| `modulation_23_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,427 (98.0%); 1 (On): 191 (2.0%) |
| `modulation_23_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,612 (99.9%); 1 (On): 6 (0.1%) |
| `modulation_23_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,605 (99.9%); 1 (On): 13 (0.1%) |
| `modulation_24_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,512 (98.9%); 1 (On): 106 (1.1%) |
| `modulation_24_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,610 (99.9%); 1 (On): 8 (0.1%) |
| `modulation_24_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,607 (99.9%); 1 (On): 11 (0.1%) |
| `modulation_25_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,489 (98.7%); 1 (On): 129 (1.3%) |
| `modulation_25_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,611 (99.9%); 1 (On): 7 (0.1%) |
| `modulation_25_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,607 (99.9%); 1 (On): 11 (0.1%) |
| `modulation_26_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,528 (99.1%); 1 (On): 90 (0.9%) |
| `modulation_26_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,610 (99.9%); 1 (On): 8 (0.1%) |
| `modulation_26_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,606 (99.9%); 1 (On): 12 (0.1%) |
| `modulation_27_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,523 (99.0%); 1 (On): 95 (1.0%) |
| `modulation_27_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,610 (99.9%); 1 (On): 8 (0.1%) |
| `modulation_27_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,604 (99.9%); 1 (On): 14 (0.1%) |
| `modulation_28_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,520 (99.0%); 1 (On): 98 (1.0%) |
| `modulation_28_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,615 (100.0%); 1 (On): 3 (0.0%) |
| `modulation_28_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,602 (99.8%); 1 (On): 16 (0.2%) |
| `modulation_29_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,485 (98.6%); 1 (On): 133 (1.4%) |
| `modulation_29_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,611 (99.9%); 1 (On): 7 (0.1%) |
| `modulation_29_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,605 (99.9%); 1 (On): 13 (0.1%) |
| `modulation_2_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,006 (93.6%); 1 (On): 612 (6.4%) |
| `modulation_2_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,578 (99.6%); 1 (On): 40 (0.4%) |
| `modulation_2_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,560 (99.4%); 1 (On): 58 (0.6%) |
| `modulation_30_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,536 (99.1%); 1 (On): 82 (0.9%) |
| `modulation_30_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,612 (99.9%); 1 (On): 6 (0.1%) |
| `modulation_30_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,613 (99.9%); 1 (On): 5 (0.1%) |
| `modulation_31_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,553 (99.3%); 1 (On): 65 (0.7%) |
| `modulation_31_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,608 (99.9%); 1 (On): 10 (0.1%) |
| `modulation_31_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,609 (99.9%); 1 (On): 9 (0.1%) |
| `modulation_32_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,536 (99.1%); 1 (On): 82 (0.9%) |
| `modulation_32_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,614 (100.0%); 1 (On): 4 (0.0%) |
| `modulation_32_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,609 (99.9%); 1 (On): 9 (0.1%) |
| `modulation_33_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,546 (99.3%); 1 (On): 72 (0.7%) |
| `modulation_33_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,610 (99.9%); 1 (On): 8 (0.1%) |
| `modulation_33_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,613 (99.9%); 1 (On): 5 (0.1%) |
| `modulation_34_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,555 (99.3%); 1 (On): 63 (0.7%) |
| `modulation_34_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,614 (100.0%); 1 (On): 4 (0.0%) |
| `modulation_34_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,614 (100.0%); 1 (On): 4 (0.0%) |
| `modulation_35_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,553 (99.3%); 1 (On): 65 (0.7%) |
| `modulation_35_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,615 (100.0%); 1 (On): 3 (0.0%) |
| `modulation_35_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,611 (99.9%); 1 (On): 7 (0.1%) |
| `modulation_36_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,564 (99.4%); 1 (On): 54 (0.6%) |
| `modulation_36_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,613 (99.9%); 1 (On): 5 (0.1%) |
| `modulation_36_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,615 (100.0%); 1 (On): 3 (0.0%) |
| `modulation_37_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,555 (99.3%); 1 (On): 63 (0.7%) |
| `modulation_37_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,616 (100.0%); 1 (On): 2 (0.0%) |
| `modulation_37_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,609 (99.9%); 1 (On): 9 (0.1%) |
| `modulation_38_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,572 (99.5%); 1 (On): 46 (0.5%) |
| `modulation_38_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,616 (100.0%); 1 (On): 2 (0.0%) |
| `modulation_38_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,611 (99.9%); 1 (On): 7 (0.1%) |
| `modulation_39_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,573 (99.5%); 1 (On): 45 (0.5%) |
| `modulation_39_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |
| `modulation_39_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,610 (99.9%); 1 (On): 8 (0.1%) |
| `modulation_3_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,086 (94.5%); 1 (On): 532 (5.5%) |
| `modulation_3_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,558 (99.4%); 1 (On): 60 (0.6%) |
| `modulation_3_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,572 (99.5%); 1 (On): 46 (0.5%) |
| `modulation_40_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,568 (99.5%); 1 (On): 50 (0.5%) |
| `modulation_40_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,615 (100.0%); 1 (On): 3 (0.0%) |
| `modulation_40_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,615 (100.0%); 1 (On): 3 (0.0%) |
| `modulation_41_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,563 (99.4%); 1 (On): 55 (0.6%) |
| `modulation_41_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |
| `modulation_41_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,613 (99.9%); 1 (On): 5 (0.1%) |
| `modulation_42_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,573 (99.5%); 1 (On): 45 (0.5%) |
| `modulation_42_bypass` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_42_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,612 (99.9%); 1 (On): 6 (0.1%) |
| `modulation_43_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,586 (99.7%); 1 (On): 32 (0.3%) |
| `modulation_43_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,616 (100.0%); 1 (On): 2 (0.0%) |
| `modulation_43_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,616 (100.0%); 1 (On): 2 (0.0%) |
| `modulation_44_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,583 (99.6%); 1 (On): 35 (0.4%) |
| `modulation_44_bypass` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_44_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,616 (100.0%); 1 (On): 2 (0.0%) |
| `modulation_45_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,581 (99.6%); 1 (On): 37 (0.4%) |
| `modulation_45_bypass` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_45_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,615 (100.0%); 1 (On): 3 (0.0%) |
| `modulation_46_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,578 (99.6%); 1 (On): 40 (0.4%) |
| `modulation_46_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |
| `modulation_46_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,614 (100.0%); 1 (On): 4 (0.0%) |
| `modulation_47_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,596 (99.8%); 1 (On): 22 (0.2%) |
| `modulation_47_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,616 (100.0%); 1 (On): 2 (0.0%) |
| `modulation_47_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,614 (100.0%); 1 (On): 4 (0.0%) |
| `modulation_48_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,578 (99.6%); 1 (On): 40 (0.4%) |
| `modulation_48_bypass` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_48_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,614 (100.0%); 1 (On): 4 (0.0%) |
| `modulation_49_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,589 (99.7%); 1 (On): 29 (0.3%) |
| `modulation_49_bypass` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_49_stereo` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_4_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,142 (95.1%); 1 (On): 476 (4.9%) |
| `modulation_4_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,573 (99.5%); 1 (On): 45 (0.5%) |
| `modulation_4_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,572 (99.5%); 1 (On): 46 (0.5%) |
| `modulation_50_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,596 (99.8%); 1 (On): 22 (0.2%) |
| `modulation_50_bypass` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_50_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |
| `modulation_51_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,593 (99.7%); 1 (On): 25 (0.3%) |
| `modulation_51_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |
| `modulation_51_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |
| `modulation_52_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,595 (99.8%); 1 (On): 23 (0.2%) |
| `modulation_52_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |
| `modulation_52_stereo` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_53_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,595 (99.8%); 1 (On): 23 (0.2%) |
| `modulation_53_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,616 (100.0%); 1 (On): 2 (0.0%) |
| `modulation_53_stereo` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_54_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,605 (99.9%); 1 (On): 13 (0.1%) |
| `modulation_54_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |
| `modulation_54_stereo` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_55_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,613 (99.9%); 1 (On): 5 (0.1%) |
| `modulation_55_bypass` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_55_stereo` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_56_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,610 (99.9%); 1 (On): 8 (0.1%) |
| `modulation_56_bypass` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_56_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |
| `modulation_57_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,610 (99.9%); 1 (On): 8 (0.1%) |
| `modulation_57_bypass` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_57_stereo` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_58_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,613 (99.9%); 1 (On): 5 (0.1%) |
| `modulation_58_bypass` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_58_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |
| `modulation_59_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,595 (99.8%); 1 (On): 23 (0.2%) |
| `modulation_59_bypass` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_59_stereo` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_5_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,092 (94.5%); 1 (On): 526 (5.5%) |
| `modulation_5_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,558 (99.4%); 1 (On): 60 (0.6%) |
| `modulation_5_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,561 (99.4%); 1 (On): 57 (0.6%) |
| `modulation_60_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,615 (100.0%); 1 (On): 3 (0.0%) |
| `modulation_60_bypass` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_60_stereo` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_61_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,612 (99.9%); 1 (On): 6 (0.1%) |
| `modulation_61_bypass` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_61_stereo` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_62_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,615 (100.0%); 1 (On): 3 (0.0%) |
| `modulation_62_bypass` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_62_stereo` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_63_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,608 (99.9%); 1 (On): 10 (0.1%) |
| `modulation_63_bypass` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_63_stereo` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_64_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,615 (100.0%); 1 (On): 3 (0.0%) |
| `modulation_64_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,617 (100.0%); 1 (On): 1 (0.0%) |
| `modulation_64_stereo` | Indexed | 9,618 | 1 | 0 (Off): 9,618 (100.0%) |
| `modulation_6_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,163 (95.3%); 1 (On): 455 (4.7%) |
| `modulation_6_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,570 (99.5%); 1 (On): 48 (0.5%) |
| `modulation_6_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,565 (99.4%); 1 (On): 53 (0.6%) |
| `modulation_7_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,222 (95.9%); 1 (On): 396 (4.1%) |
| `modulation_7_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,576 (99.6%); 1 (On): 42 (0.4%) |
| `modulation_7_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,592 (99.7%); 1 (On): 26 (0.3%) |
| `modulation_8_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,257 (96.2%); 1 (On): 361 (3.8%) |
| `modulation_8_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,592 (99.7%); 1 (On): 26 (0.3%) |
| `modulation_8_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,586 (99.7%); 1 (On): 32 (0.3%) |
| `modulation_9_bipolar` | Indexed | 9,618 | 2 | 0 (Off): 9,274 (96.4%); 1 (On): 344 (3.6%) |
| `modulation_9_bypass` | Indexed | 9,618 | 2 | 0 (Off): 9,597 (99.8%); 1 (On): 21 (0.2%) |
| `modulation_9_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,581 (99.6%); 1 (On): 37 (0.4%) |
| `mpe_enabled` | Indexed | 9,618 | 2 | 0 (Off): 9,574 (99.5%); 1 (On): 44 (0.5%) |
| `osc_1_destination` | Indexed | 9,618 | 5 | 0 (FILTER 1): 7,545 (78.4%); 2 (FILTER 1+2): 775 (8.1%); 3 (EFFECTS): 730 (7.6%); 1 (FILTER 2): 505 (5.3%); 4 (DIRECT OUT): 63 (0.7%) |
| `osc_1_distortion_type` | Indexed | 9,618 | 13 | 0 (None): 5,712 (59.4%); 7 (FM <- Osc): 1,289 (13.4%); 1 (Sync): 707 (7.4%); 2 (Formant): 395 (4.1%); 4 (Bend): 344 (3.6%); 5 (Squeeze): 226 (2.3%); 8 (FM <- Osc): 203 (2.1%); 3 (Quantize): 182 (1.9%); … 5 more in CSV |
| `osc_1_midi_track` | Indexed | 9,618 | 2 | 1 (On): 9,493 (98.7%); 0 (Off): 125 (1.3%) |
| `osc_1_on` | Indexed | 9,618 | 2 | 1 (On): 9,223 (95.9%); 0 (Off): 395 (4.1%) |
| `osc_1_smooth_interpolation` | Indexed | 9,618 | 2 | 0 (Off): 9,014 (93.7%); 1 (On): 604 (6.3%) |
| `osc_1_spectral_morph_type` | Indexed | 9,618 | 16 | 0 (None): 6,385 (66.4%); 2 (Formant Scale): 532 (5.5%); 3 (Harmonic Stretch): 401 (4.2%); 5 (Smear): 379 (3.9%); 1 (Vocode): 353 (3.7%); 7 (Low Pass): 343 (3.6%); 4 (Inharmonic Stretch): 297 (3.1%); 6 (Random Amplitudes): 222 (2.3%); … 8 more in CSV |
| `osc_1_spectral_unison` | Indexed | 9,618 | 3 | 1 (On): 9,598 (99.8%); 2: 11 (0.1%); 0 (Off): 9 (0.1%) |
| `osc_1_stack_style` | Indexed | 9,618 | 13 | 0 (Unison): 9,132 (94.9%); 1 (Center Drop 12): 85 (0.9%); 3 (Octave): 76 (0.8%); 10 (Odd Harmonics): 75 (0.8%); 9 (Harmonics): 46 (0.5%); 11: 41 (0.4%); 12: 32 (0.3%); 4 (2x Octave): 27 (0.3%); … 5 more in CSV |
| `osc_1_transpose` | Indexed | 9,618 | 85 | 0: 4,971 (51.7%); -12: 1,550 (16.1%); -24: 1,453 (15.1%); 12: 612 (6.4%); -36: 147 (1.5%); 24: 126 (1.3%); -48: 89 (0.9%); -20: 34 (0.4%); … 77 more in CSV |
| `osc_1_transpose_quantize` | Indexed | 9,618 | 117 | 0: 9,366 (97.4%); 1: 58 (0.6%); 4096: 16 (0.2%); 32: 8 (0.1%); 16: 6 (0.1%); 1161: 5 (0.1%); 128: 5 (0.1%); 1193: 3 (0.0%); … 109 more in CSV |
| `osc_1_unison_voices` | Indexed | 9,618 | 16 | 1: 4,606 (47.9%); 16: 864 (9.0%); 2: 748 (7.8%); 3: 608 (6.3%); 4: 538 (5.6%); 8: 435 (4.5%); 5: 405 (4.2%); 7: 367 (3.8%); … 8 more in CSV |
| `osc_1_view_2d` | Indexed | 9,618 | 3 | 1 (On): 8,796 (91.5%); 0 (Off): 705 (7.3%); 2: 117 (1.2%) |
| `osc_2_destination` | Indexed | 9,618 | 6 | 1 (FILTER 2): 4,702 (48.9%); 0 (FILTER 1): 2,211 (23.0%); 2 (FILTER 1+2): 1,546 (16.1%); 3 (EFFECTS): 1,045 (10.9%); 4 (DIRECT OUT): 113 (1.2%); 9 (EQ): 1 (0.0%) |
| `osc_2_distortion_type` | Indexed | 9,618 | 13 | 0 (None): 6,902 (71.8%); 1 (Sync): 591 (6.1%); 7 (FM <- Osc): 401 (4.2%); 8 (FM <- Osc): 346 (3.6%); 2 (Formant): 318 (3.3%); 4 (Bend): 256 (2.7%); 3 (Quantize): 173 (1.8%); 5 (Squeeze): 165 (1.7%); … 5 more in CSV |
| `osc_2_midi_track` | Indexed | 9,618 | 2 | 1 (On): 9,530 (99.1%); 0 (Off): 88 (0.9%) |
| `osc_2_on` | Indexed | 9,618 | 2 | 1 (On): 7,721 (80.3%); 0 (Off): 1,897 (19.7%) |
| `osc_2_smooth_interpolation` | Indexed | 9,618 | 2 | 0 (Off): 9,204 (95.7%); 1 (On): 414 (4.3%) |
| `osc_2_spectral_morph_type` | Indexed | 9,618 | 16 | 0 (None): 7,092 (73.7%); 5 (Smear): 381 (4.0%); 7 (Low Pass): 355 (3.7%); 2 (Formant Scale): 354 (3.7%); 3 (Harmonic Stretch): 318 (3.3%); 1 (Vocode): 271 (2.8%); 4 (Inharmonic Stretch): 199 (2.1%); 6 (Random Amplitudes): 119 (1.2%); … 8 more in CSV |
| `osc_2_spectral_unison` | Indexed | 9,618 | 3 | 1 (On): 9,600 (99.8%); 0 (Off): 10 (0.1%); 2: 8 (0.1%) |
| `osc_2_stack_style` | Indexed | 9,618 | 13 | 0 (Unison): 9,277 (96.5%); 1 (Center Drop 12): 69 (0.7%); 10 (Odd Harmonics): 61 (0.6%); 3 (Octave): 39 (0.4%); 4 (2x Octave): 33 (0.3%); 11: 30 (0.3%); 6 (2x Power Chord): 25 (0.3%); 9 (Harmonics): 24 (0.2%); … 5 more in CSV |
| `osc_2_transpose` | Indexed | 9,618 | 90 | 0: 4,744 (49.3%); -12: 1,582 (16.4%); -24: 990 (10.3%); 12: 967 (10.1%); 24: 185 (1.9%); 7: 153 (1.6%); -36: 119 (1.2%); -48: 67 (0.7%); … 82 more in CSV |
| `osc_2_transpose_quantize` | Indexed | 9,618 | 95 | 0: 9,431 (98.1%); 1: 34 (0.4%); 4096: 9 (0.1%); 129: 6 (0.1%); 128: 5 (0.1%); 16: 5 (0.1%); 8: 5 (0.1%); 1161: 4 (0.0%); … 87 more in CSV |
| `osc_2_unison_voices` | Indexed | 9,618 | 17 | 1: 5,558 (57.8%); 2: 698 (7.3%); 16: 589 (6.1%); 3: 530 (5.5%); 4: 510 (5.3%); 5: 348 (3.6%); 8: 337 (3.5%); 7: 293 (3.0%); … 9 more in CSV |
| `osc_2_view_2d` | Indexed | 9,618 | 3 | 1 (On): 9,111 (94.7%); 0 (Off): 451 (4.7%); 2: 56 (0.6%) |
| `osc_3_destination` | Indexed | 9,618 | 5 | 3 (EFFECTS): 6,054 (62.9%); 0 (FILTER 1): 1,848 (19.2%); 1 (FILTER 2): 946 (9.8%); 4 (DIRECT OUT): 420 (4.4%); 2 (FILTER 1+2): 350 (3.6%) |
| `osc_3_distortion_type` | Indexed | 9,618 | 13 | 0 (None): 7,959 (82.8%); 1 (Sync): 401 (4.2%); 7 (FM <- Osc): 225 (2.3%); 8 (FM <- Osc): 211 (2.2%); 2 (Formant): 198 (2.1%); 4 (Bend): 143 (1.5%); 5 (Squeeze): 98 (1.0%); 9 (FM <- Sample): 82 (0.9%); … 5 more in CSV |
| `osc_3_midi_track` | Indexed | 9,618 | 2 | 1 (On): 9,542 (99.2%); 0 (Off): 76 (0.8%) |
| `osc_3_on` | Indexed | 9,618 | 2 | 1 (On): 5,162 (53.7%); 0 (Off): 4,456 (46.3%) |
| `osc_3_smooth_interpolation` | Indexed | 9,618 | 2 | 0 (Off): 9,309 (96.8%); 1 (On): 309 (3.2%) |
| `osc_3_spectral_morph_type` | Indexed | 9,618 | 16 | 0 (None): 7,936 (82.5%); 7 (Low Pass): 317 (3.3%); 2 (Formant Scale): 199 (2.1%); 5 (Smear): 198 (2.1%); 3 (Harmonic Stretch): 153 (1.6%); 1 (Vocode): 150 (1.6%); 4 (Inharmonic Stretch): 106 (1.1%); 13: 101 (1.1%); … 8 more in CSV |
| `osc_3_spectral_unison` | Indexed | 9,618 | 3 | 1 (On): 9,602 (99.8%); 0 (Off): 11 (0.1%); 2: 5 (0.1%) |
| `osc_3_stack_style` | Indexed | 9,618 | 13 | 0 (Unison): 9,371 (97.4%); 1 (Center Drop 12): 41 (0.4%); 10 (Odd Harmonics): 41 (0.4%); 3 (Octave): 36 (0.4%); 6 (2x Power Chord): 27 (0.3%); 9 (Harmonics): 20 (0.2%); 4 (2x Octave): 16 (0.2%); 11: 15 (0.2%); … 5 more in CSV |
| `osc_3_transpose` | Indexed | 9,618 | 83 | 0: 6,110 (63.5%); -12: 1,117 (11.6%); -24: 864 (9.0%); 12: 564 (5.9%); 24: 150 (1.6%); -36: 88 (0.9%); 7: 86 (0.9%); -48: 73 (0.8%); … 75 more in CSV |
| `osc_3_transpose_quantize` | Indexed | 9,618 | 73 | 0: 9,472 (98.5%); 1: 27 (0.3%); 4096: 8 (0.1%); 1024: 6 (0.1%); 129: 4 (0.0%); 33: 4 (0.0%); 4: 4 (0.0%); 1193: 3 (0.0%); … 65 more in CSV |
| `osc_3_unison_voices` | Indexed | 9,618 | 16 | 1: 7,039 (73.2%); 16: 488 (5.1%); 2: 375 (3.9%); 3: 311 (3.2%); 8: 258 (2.7%); 4: 253 (2.6%); 5: 190 (2.0%); 6: 180 (1.9%); … 8 more in CSV |
| `osc_3_view_2d` | Indexed | 9,618 | 3 | 1 (On): 9,330 (97.0%); 0 (Off): 252 (2.6%); 2: 36 (0.4%) |
| `oversampling` | Indexed | 9,618 | 4 | 1 (2x): 9,231 (96.0%); 2 (4x): 207 (2.2%); 0 (1x): 128 (1.3%); 3 (8x): 52 (0.5%) |
| `phaser_on` | Indexed | 9,618 | 2 | 0 (Off): 8,160 (84.8%); 1 (On): 1,458 (15.2%) |
| `phaser_sync` | Indexed | 9,618 | 4 | 1 (Tempo): 9,467 (98.4%); 0 (Seconds): 108 (1.1%); 2 (Tempo Dotted): 27 (0.3%); 3 (Tempo Triplets): 16 (0.2%) |
| `phaser_tempo` | Indexed | 9,618 | 12 | 3 (8/1): 8,446 (87.8%); 0 (Freeze): 572 (5.9%); 4 (4/1): 157 (1.6%); 5 (2/1): 117 (1.2%); 2 (16/1): 88 (0.9%); 6 (1/1): 77 (0.8%); 1 (32/1): 54 (0.6%); 7 (1/2): 41 (0.4%); … 4 more in CSV |
| `pitch_bend_range` | Indexed | 9,618 | 21 | 2: 8,861 (92.1%); 12: 378 (3.9%); 0: 90 (0.9%); 1: 70 (0.7%); 7: 38 (0.4%); 4: 29 (0.3%); 5: 27 (0.3%); 24: 25 (0.3%); … 13 more in CSV |
| `polyphony` | Indexed | 9,618 | 29 | 8: 6,007 (62.5%); 1: 2,680 (27.9%); 12: 158 (1.6%); 16: 147 (1.5%); 2: 118 (1.2%); 32: 108 (1.1%); 4: 89 (0.9%); 6: 54 (0.6%); … 21 more in CSV |
| `portamento_force` | Indexed | 9,618 | 2 | 0 (Off): 8,546 (88.9%); 1 (On): 1,072 (11.1%) |
| `portamento_scale` | Indexed | 9,618 | 2 | 0 (Off): 9,438 (98.1%); 1 (On): 180 (1.9%) |
| `random_1_keytrack_transpose` | Indexed | 9,618 | 9 | -12: 9,604 (99.9%); 0: 4 (0.0%); -13: 3 (0.0%); -24: 2 (0.0%); -22: 1 (0.0%); -23: 1 (0.0%); 11: 1 (0.0%); 12: 1 (0.0%); … 1 more in CSV |
| `random_1_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,144 (95.1%); 1 (On): 474 (4.9%) |
| `random_1_style` | Indexed | 9,618 | 4 | 0 (Perlin): 8,850 (92.0%); 1 (Sample & Hold): 495 (5.1%); 2 (Sine Interpolate): 182 (1.9%); 3 (Lorenz Attractor): 91 (0.9%) |
| `random_1_sync` | Indexed | 9,618 | 5 | 1 (Tempo): 9,070 (94.3%); 0 (Seconds): 483 (5.0%); 2 (Tempo Dotted): 32 (0.3%); 3 (Tempo Triplets): 28 (0.3%); 4 (Keytrack): 5 (0.1%) |
| `random_1_sync_type` | Indexed | 9,618 | 2 | 0 (Off): 9,019 (93.8%); 1 (On): 599 (6.2%) |
| `random_1_tempo` | Indexed | 9,618 | 13 | 8 (1/4): 8,435 (87.7%); 9 (1/8): 222 (2.3%); 7 (1/2): 199 (2.1%); 6 (1/1): 155 (1.6%); 0 (Freeze): 151 (1.6%); 10 (1/16): 150 (1.6%); 5 (2/1): 95 (1.0%); 4 (4/1): 74 (0.8%); … 5 more in CSV |
| `random_2_keytrack_transpose` | Indexed | 9,618 | 2 | -12: 9,617 (100.0%); 0: 1 (0.0%) |
| `random_2_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,457 (98.3%); 1 (On): 161 (1.7%) |
| `random_2_style` | Indexed | 9,618 | 4 | 0 (Perlin): 9,264 (96.3%); 1 (Sample & Hold): 217 (2.3%); 2 (Sine Interpolate): 104 (1.1%); 3 (Lorenz Attractor): 33 (0.3%) |
| `random_2_sync` | Indexed | 9,618 | 4 | 1 (Tempo): 9,322 (96.9%); 0 (Seconds): 273 (2.8%); 2 (Tempo Dotted): 15 (0.2%); 3 (Tempo Triplets): 8 (0.1%) |
| `random_2_sync_type` | Indexed | 9,618 | 2 | 0 (Off): 9,398 (97.7%); 1 (On): 220 (2.3%) |
| `random_2_tempo` | Indexed | 9,618 | 13 | 8 (1/4): 9,133 (95.0%); 7 (1/2): 96 (1.0%); 6 (1/1): 86 (0.9%); 9 (1/8): 75 (0.8%); 0 (Freeze): 67 (0.7%); 10 (1/16): 44 (0.5%); 4 (4/1): 40 (0.4%); 5 (2/1): 32 (0.3%); … 5 more in CSV |
| `random_3_keytrack_transpose` | Indexed | 9,618 | 2 | -12: 9,617 (100.0%); 28: 1 (0.0%) |
| `random_3_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,560 (99.4%); 1 (On): 58 (0.6%) |
| `random_3_style` | Indexed | 9,618 | 4 | 0 (Perlin): 9,487 (98.6%); 1 (Sample & Hold): 85 (0.9%); 2 (Sine Interpolate): 29 (0.3%); 3 (Lorenz Attractor): 17 (0.2%) |
| `random_3_sync` | Indexed | 9,618 | 5 | 1 (Tempo): 9,505 (98.8%); 0 (Seconds): 97 (1.0%); 2 (Tempo Dotted): 10 (0.1%); 3 (Tempo Triplets): 5 (0.1%); 4 (Keytrack): 1 (0.0%) |
| `random_3_sync_type` | Indexed | 9,618 | 2 | 0 (Off): 9,532 (99.1%); 1 (On): 86 (0.9%) |
| `random_3_tempo` | Indexed | 9,618 | 12 | 8 (1/4): 9,410 (97.8%); 6 (1/1): 38 (0.4%); 7 (1/2): 35 (0.4%); 0 (Freeze): 31 (0.3%); 5 (2/1): 24 (0.2%); 10 (1/16): 22 (0.2%); 4 (4/1): 18 (0.2%); 3 (8/1): 13 (0.1%); … 4 more in CSV |
| `random_4_keytrack_transpose` | Indexed | 9,618 | 1 | -12: 9,618 (100.0%) |
| `random_4_stereo` | Indexed | 9,618 | 2 | 0 (Off): 9,589 (99.7%); 1 (On): 29 (0.3%) |
| `random_4_style` | Indexed | 9,618 | 4 | 0 (Perlin): 9,550 (99.3%); 1 (Sample & Hold): 54 (0.6%); 2 (Sine Interpolate): 12 (0.1%); 3 (Lorenz Attractor): 2 (0.0%) |
| `random_4_sync` | Indexed | 9,618 | 3 | 1 (Tempo): 9,556 (99.4%); 0 (Seconds): 52 (0.5%); 2 (Tempo Dotted): 10 (0.1%) |
| `random_4_sync_type` | Indexed | 9,618 | 2 | 0 (Off): 9,579 (99.6%); 1 (On): 39 (0.4%) |
| `random_4_tempo` | Indexed | 9,618 | 11 | 8 (1/4): 9,525 (99.0%); 7 (1/2): 20 (0.2%); 0 (Freeze): 16 (0.2%); 6 (1/1): 13 (0.1%); 9 (1/8): 11 (0.1%); 10 (1/16): 10 (0.1%); 5 (2/1): 10 (0.1%); 4 (4/1): 7 (0.1%); … 3 more in CSV |
| `reverb_on` | Indexed | 9,618 | 2 | 1 (On): 6,532 (67.9%); 0 (Off): 3,086 (32.1%) |
| `sample_bounce` | Indexed | 9,618 | 2 | 0 (Off): 9,312 (96.8%); 1 (On): 306 (3.2%) |
| `sample_destination` | Indexed | 9,618 | 5 | 3 (EFFECTS): 7,103 (73.9%); 0 (FILTER 1): 1,246 (13.0%); 1 (FILTER 2): 785 (8.2%); 2 (FILTER 1+2): 252 (2.6%); 4 (DIRECT OUT): 232 (2.4%) |
| `sample_keytrack` | Indexed | 9,618 | 2 | 0 (Off): 8,279 (86.1%); 1 (On): 1,339 (13.9%) |
| `sample_loop` | Indexed | 9,618 | 2 | 1 (On): 8,943 (93.0%); 0 (Off): 675 (7.0%) |
| `sample_on` | Indexed | 9,618 | 2 | 0 (Off): 6,213 (64.6%); 1 (On): 3,405 (35.4%) |
| `sample_random_phase` | Indexed | 9,618 | 2 | 0 (Off): 9,114 (94.8%); 1 (On): 504 (5.2%) |
| `sample_transpose` | Indexed | 9,618 | 87 | 0: 8,440 (87.8%); 12: 238 (2.5%); -12: 182 (1.9%); 24: 103 (1.1%); 48: 84 (0.9%); -24: 75 (0.8%); -48: 40 (0.4%); 7: 29 (0.3%); … 79 more in CSV |
| `sample_transpose_quantize` | Indexed | 9,618 | 28 | 0: 9,570 (99.5%); 1: 11 (0.1%); 4096: 6 (0.1%); 128: 4 (0.0%); 32: 3 (0.0%); 4: 2 (0.0%); 1024: 1 (0.0%); 1144: 1 (0.0%); … 20 more in CSV |
| `stereo_mode` | Indexed | 9,618 | 2 | 0 (SPREAD): 9,581 (99.6%); 1 (ROTATE): 37 (0.4%) |
| `view_spectrogram` | Indexed | 9,618 | 3 | 0 (Off): 8,921 (92.8%); 1 (On): 696 (7.2%); 2: 1 (0.0%) |
| `voice_override` | Indexed | 9,618 | 2 | 0 (Kill): 9,557 (99.4%); 1 (Steal): 61 (0.6%) |
| `voice_priority` | Indexed | 9,618 | 5 | 4 (Round Robin): 9,535 (99.1%); 0 (Newest): 44 (0.5%); 1 (Oldest): 17 (0.2%); 2 (Highest): 13 (0.1%); 3 (Lowest): 9 (0.1%) |
| `voice_transpose` | Indexed | 9,618 | 65 | 0: 9,164 (95.3%); -12: 101 (1.1%); 12: 65 (0.7%); -24: 50 (0.5%); -6: 13 (0.1%); 24: 13 (0.1%); -10: 12 (0.1%); -1: 11 (0.1%); … 57 more in CSV |

## Continuous parameter distributions (553 parameters)

Quantiles are reported in the distribution domain shown in the `Domain` column. The raw range and raw mean remain visible even when the histogram uses normalized position.

| Parameter | Scale | Observed | Raw range | Raw mean ± sd | Default | Zero | Domain | p05 / p50 / p95 | Dominant bins |
|---|---|---:|---|---|---:|---:|---|---|---|
| `beats_per_minute` | Linear | 9,618 | 0.16667–7 | 2.18633 ± 0.44295 | 18.31982% | 0% | raw | 1.4428 / 2.15163 / 2.91047 | 1.98177–2.08854 raw: 1,957 (20.3%)<br>2.19531–2.30208 raw: 1,298 (13.5%)<br>2.08854–2.19531 raw: 1,265 (13.2%) |
| `chorus_cutoff` | Linear | 9,618 | 8–136 | 69.29119 ± 22.52119 | 64.97193% | 0% | normalized | 56.16286 / 61.3522 / 123.64074 | 60–62 raw / 0.40625–0.42188 norm: 6,306 (65.6%)<br>134–136 raw / 0.98438–1 norm: 394 (4.1%)<br>88–90 raw / 0.625–0.64062 norm: 122 (1.3%) |
| `chorus_delay_1` | Exponential | 9,618 | -10–-5.63784 | -8.87731 ± 0.6563 | 76.88709% | 0% | raw | -9.57734 / -9.00969 / -7.48784 | -9.04578–-8.97762 raw: 7,450 (77.5%)<br>-10–-9.93184 raw: 376 (3.9%)<br>-5.70599–-5.63784 raw: 144 (1.5%) |
| `chorus_delay_2` | Exponential | 9,618 | -10–-5.47292 | -7.26332 ± 0.84009 | 75.54585% | 0% | raw | -9.52171 / -6.99919 / -6.6567 | -7.02911–-6.95837 raw: 7,287 (75.8%)<br>-10–-9.92926 raw: 366 (3.8%)<br>-5.68513–-5.6144 raw: 201 (2.1%) |
| `chorus_dry_wet` | Linear | 9,618 | 0–1 | 0.36255 ± 0.2114 | 48.5236% | 8.71283% | normalized | 0.0084 / 0.50119 / 0.51557 | 0.5–0.51562 raw / 0.5–0.51562 norm: 4,700 (48.9%)<br>0–0.01562 raw / 0–0.01562 norm: 894 (9.3%)<br>0.1875–0.20312 raw / 0.1875–0.20312 norm: 205 (2.1%) |
| `chorus_feedback` | Linear | 9,618 | -0.95–0.95584 | 0.32919 ± 0.19749 | 68.0287% | 9.9189% | raw | -0.0143 / 0.40103 / 0.47376 | 0.39004–0.41982 raw: 6,557 (68.2%)<br>-0.02686–0.00292 raw: 538 (5.6%)<br>0.00292–0.0327 raw: 418 (4.3%) |
| `chorus_frequency` | Exponential | 9,618 | -6–3 | -2.96869 ± 0.47193 | 97.26554% | 0% | normalized | -3.04127 / -2.97627 / -2.91126 | -3.04688–-2.90625 raw / 0.32812–0.34375 norm: 9,362 (97.3%)<br>1.875–2.01562 raw / 0.875–0.89062 norm: 37 (0.4%)<br>-3.46875–-3.32812 raw / 0.28125–0.29688 norm: 28 (0.3%) |
| `chorus_mod_depth` | Linear | 9,618 | 0–1 | 0.48815 ± 0.17823 | 68.17426% | 2.03785% | normalized | 0.13676 / 0.50714 / 0.86177 | 0.5–0.51562 raw / 0.5–0.51562 norm: 6,600 (68.6%)<br>0.98438–1 raw / 0.98438–1 norm: 378 (3.9%)<br>0–0.01562 raw / 0–0.01562 norm: 234 (2.4%) |
| `chorus_spread` | Linear | 9,618 | 0–1 | 0.79556 ± 0.32893 | 67.72718% | 3.54544% | normalized | 0.0844 / 0.98853 / 0.99885 | 0.98438–1 raw / 0.98438–1 norm: 6,551 (68.1%)<br>0–0.01562 raw / 0–0.01562 norm: 366 (3.8%)<br>0.28125–0.29688 raw / 0.28125–0.29688 norm: 105 (1.1%) |
| `compressor_attack` | Linear | 9,618 | 0–1 | 0.52992 ± 0.20678 | 53.98212% | 2.59929% | normalized | 0.14103 / 0.50921 / 0.98894 | 0.5–0.51562 raw / 0.5–0.51562 norm: 5,247 (54.6%)<br>0.98438–1 raw / 0.98438–1 norm: 681 (7.1%)<br>0–0.01562 raw / 0–0.01562 norm: 272 (2.8%) |
| `compressor_band_gain` | Linear | 9,618 | -30–30 | 11.78196 ± 4.46774 | 70.31607% | 0.14556% | normalized | 6.03057 / 11.73646 / 18.42309 | 11.25–12.1875 raw / 0.6875–0.70312 norm: 7,016 (72.9%)<br>13.125–14.0625 raw / 0.71875–0.73438 norm: 179 (1.9%)<br>15–15.9375 raw / 0.75–0.76562 norm: 159 (1.7%) |
| `compressor_band_lower_ratio` | Linear | 9,618 | -1.00811–1.01154 | 0.73815 ± 0.23614 | 81.51383% | 0.62383% | raw | 0.26348 / 0.80474 / 0.82196 | 0.79064–0.8222 raw: 7,930 (82.4%)<br>0.85375–0.88531 raw: 119 (1.2%)<br>0.8222–0.85375 raw: 99 (1.0%) |
| `compressor_band_lower_threshold` | Linear | 9,618 | -80–-1 | -36.89665 ± 7.89919 | 72.59305% | 0% | normalized | -50.18304 / -35.64308 / -27.68048 | -36.25–-35 raw / 0.54688–0.5625 norm: 7,086 (73.7%)<br>-32.5–-31.25 raw / 0.59375–0.60938 norm: 135 (1.4%)<br>-30–-28.75 raw / 0.625–0.64062 norm: 134 (1.4%) |
| `compressor_band_upper_ratio` | Linear | 9,618 | -0.57592–1.02341 | 0.80343 ± 0.15223 | 79.70472% | 0.31192% | raw | 0.42287 / 0.85865 / 0.87267 | 0.84849–0.87348 raw: 7,717 (80.2%)<br>0.64857–0.67356 raw: 94 (1.0%)<br>0.77352–0.79851 raw: 88 (0.9%) |
| `compressor_band_upper_threshold` | Linear | 9,618 | -79–-1 | -25.01971 ± 5.72673 | 69.91058% | 0% | normalized | -33.38292 / -24.37536 / -16.25932 | -25–-23.75 raw / 0.6875–0.70312 norm: 6,907 (71.8%)<br>-23.75–-22.5 raw / 0.70312–0.71875 norm: 170 (1.8%)<br>-21.25–-20 raw / 0.73438–0.75 norm: 167 (1.7%) |
| `compressor_high_gain` | Linear | 9,618 | -30–30 | 14.70765 ± 6.23351 | 69.4947% | 0.29112% | normalized | 3.36121 / 16.31965 / 19.82474 | 15.9375–16.875 raw / 0.76562–0.78125 norm: 6,782 (70.5%)<br>11.25–12.1875 raw / 0.6875–0.70312 norm: 171 (1.8%)<br>12.1875–13.125 raw / 0.70312–0.71875 norm: 145 (1.5%) |
| `compressor_high_lower_ratio` | Linear | 9,618 | -1.01698–1.0283 | 0.74071 ± 0.24138 | 81.79455% | 0.44708% | raw | 0.30967 / 0.79413 / 0.84044 | 0.77264–0.8046 raw: 5,398 (56.1%)<br>0.8046–0.83656 raw: 2,535 (26.4%)<br>0.83656–0.86851 raw: 207 (2.2%) |
| `compressor_high_lower_threshold` | Linear | 9,618 | -79–-1 | -37.04458 ± 8.21201 | 73.09212% | 0% | normalized | -53.97147 / -34.47253 / -31.23503 | -35–-33.75 raw / 0.5625–0.57812 norm: 7,113 (74.0%)<br>-37.5–-36.25 raw / 0.53125–0.54688 norm: 160 (1.7%)<br>-40–-38.75 raw / 0.5–0.51562 norm: 131 (1.4%) |
| `compressor_high_upper_ratio` | Linear | 9,618 | -0.58412–1.02264 | 0.9303 ± 0.18136 | 80.5053% | 0.28072% | raw | 0.47588 / 1.00728 / 1.0211 | 0.99754–1.02264 raw: 7,860 (81.7%)<br>0.8218–0.8469 raw: 91 (0.9%)<br>0.64606–0.67116 raw: 90 (0.9%) |
| `compressor_high_upper_threshold` | Linear | 9,618 | -79–0.3046 | -30.08217 ± 7.05565 | 69.8482% | 0.0104% | raw | -40.93187 / -30.00044 / -20.09142 | -30.67376–-29.43463 raw: 6,628 (68.9%)<br>-29.43463–-28.19549 raw: 390 (4.1%)<br>-28.19549–-26.95636 raw: 170 (1.8%) |
| `compressor_low_gain` | Linear | 9,618 | -30–30 | 14.17071 ± 7.86617 | 71.81327% | 0.19755% | normalized | 1.16611 / 16.32296 / 19.20798 | 15.9375–16.875 raw / 0.76562–0.78125 norm: 7,001 (72.8%)<br>-30–-29.0625 raw / 0–0.01562 norm: 145 (1.5%)<br>9.375–10.3125 raw / 0.65625–0.67188 norm: 132 (1.4%) |
| `compressor_low_lower_ratio` | Linear | 9,618 | -1.33529–1.55634 | 0.74195 ± 0.23882 | 85.51674% | 0.65502% | raw | 0.30157 / 0.80872 / 0.83235 | 0.78825–0.83343 raw: 8,276 (86.0%)<br>0.6527–0.69789 raw: 155 (1.6%)<br>0.87861–0.92379 raw: 94 (1.0%) |
| `compressor_low_lower_threshold` | Linear | 9,618 | -79–-1 | -37.05127 ± 8.60314 | 78.06197% | 0% | normalized | -54.80757 / -34.45507 / -32.65804 | -35–-33.75 raw / 0.5625–0.57812 norm: 7,564 (78.6%)<br>-80–-78.75 raw / 0–0.01562 norm: 159 (1.7%)<br>-38.75–-37.5 raw / 0.51562–0.53125 norm: 91 (0.9%) |
| `compressor_low_upper_ratio` | Linear | 9,618 | -0.02553–1.0283 | 0.85968 ± 0.13869 | 84.32106% | 0.24953% | raw | 0.53418 / 0.90361 / 0.91256 | 0.89657–0.91304 raw: 7,962 (82.8%)<br>0.88011–0.89657 raw: 186 (1.9%)<br>0.99537–1.01184 raw: 155 (1.6%) |
| `compressor_low_upper_threshold` | Linear | 9,618 | -79–0.25473 | -28.9024 ± 7.76052 | 76.75192% | 0.0104% | raw | -39.79739 / -27.64353 / -21.51998 | -28.22744–-26.98908 raw: 7,250 (75.4%)<br>-29.46579–-28.22744 raw: 276 (2.9%)<br>-25.75073–-24.51237 raw: 180 (1.9%) |
| `compressor_mix` | Linear | 9,618 | 0–1 | 0.85526 ± 0.28149 | 73.94469% | 2.59929% | normalized | 0.17143 / 0.98949 / 0.99895 | 0.98438–1 raw / 0.98438–1 norm: 7,147 (74.3%)<br>0–0.01562 raw / 0–0.01562 norm: 270 (2.8%)<br>0.5–0.51562 raw / 0.5–0.51562 norm: 116 (1.2%) |
| `compressor_release` | Linear | 9,618 | 0–1 | 0.50278 ± 0.19429 | 57.49636% | 2.06904% | normalized | 0.15223 / 0.50728 / 0.98616 | 0.5–0.51562 raw / 0.5–0.51562 norm: 5,579 (58.0%)<br>0.98438–1 raw / 0.98438–1 norm: 544 (5.7%)<br>0–0.01562 raw / 0–0.01562 norm: 222 (2.3%) |
| `delay_aux_frequency` | Exponential | 9,618 | -2–9 | 2.05851 ± 0.54888 | 97.10959% | 0.08318% | normalized | 1.96064 / 2.04019 / 2.11975 | 1.95312–2.125 raw / 0.35938–0.375 norm: 9,350 (97.2%)<br>8.82812–9 raw / 0.98438–1 norm: 16 (0.2%)<br>1.78125–1.95312 raw / 0.34375–0.35938 norm: 15 (0.2%) |
| `delay_dry_wet` | Linear | 9,618 | 0–1 | 0.2656 ± 0.14067 | 57.66272% | 9.96049% | normalized | 0.00736 / 0.33181 / 0.35847 | 0.32812–0.34375 raw / 0.32812–0.34375 norm: 5,602 (58.2%)<br>0–0.01562 raw / 0–0.01562 norm: 1,021 (10.6%)<br>0.09375–0.10938 raw / 0.09375–0.10938 norm: 186 (1.9%) |
| `delay_feedback` | Linear | 9,618 | -1–1 | 0.44649 ± 0.17552 | 68.36141% | 0.39509% | normalized | 0.11327 / 0.51209 / 0.60469 | 0.5–0.53125 raw / 0.75–0.76562 norm: 6,666 (69.3%)<br>0.375–0.40625 raw / 0.6875–0.70312 norm: 175 (1.8%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 173 (1.8%) |
| `delay_filter_cutoff` | Linear | 9,618 | 8–136 | 66.87648 ± 19.1809 | 67.48804% | 0% | normalized | 57.748 / 61.31522 / 105.61522 | 60–62 raw / 0.40625–0.42188 norm: 6,532 (67.9%)<br>134–136 raw / 0.98438–1 norm: 229 (2.4%)<br>80–82 raw / 0.5625–0.57812 norm: 134 (1.4%) |
| `delay_filter_spread` | Linear | 9,618 | 0–1 | 0.78558 ± 0.34783 | 69.33874% | 4.80349% | normalized | 0.0153 / 0.98876 / 0.99887 | 0.98438–1 raw / 0.98438–1 norm: 6,684 (69.5%)<br>0–0.01562 raw / 0–0.01562 norm: 491 (5.1%)<br>0.25–0.26562 raw / 0.25–0.26562 norm: 97 (1.0%) |
| `delay_frequency` | Exponential | 9,618 | -2–9 | 2.1228 ± 0.75665 | 94.88459% | 0.07278% | normalized | 1.96046 / 2.04187 / 2.12327 | 1.95312–2.125 raw / 0.35938–0.375 norm: 9,137 (95.0%)<br>8.82812–9 raw / 0.98438–1 norm: 33 (0.3%)<br>5.21875–5.39062 raw / 0.65625–0.67188 norm: 28 (0.3%) |
| `distortion_drive` | Linear | 9,618 | -30–30 | 4.0311 ± 11.26906 | 28.70659% | 28.70659% | normalized | -15.60298 / 1.71607 / 29.06473 | 0–0.9375 raw / 0.5–0.51562 norm: 2,948 (30.7%)<br>29.0625–30 raw / 0.98438–1 norm: 483 (5.0%)<br>4.6875–5.625 raw / 0.57812–0.59375 norm: 394 (4.1%) |
| `distortion_filter_blend` | Linear | 9,618 | 0–2 | 0.14704 ± 0.45675 | 87.40902% | 87.40902% | normalized | 0.00178 / 0.01783 / 1.41585 | 0–0.03125 raw / 0–0.01562 norm: 8,427 (87.6%)<br>1.96875–2 raw / 0.98438–1 norm: 375 (3.9%)<br>1–1.03125 raw / 0.5–0.51562 norm: 62 (0.6%) |
| `distortion_filter_cutoff` | Linear | 9,618 | 8–136 | 80.85988 ± 21.49082 | 69.0996% | 0% | normalized | 41.61852 / 81.0359 / 127.54186 | 80–82 raw / 0.5625–0.57812 norm: 6,714 (69.8%)<br>134–136 raw / 0.98438–1 norm: 367 (3.8%)<br>8–10 raw / 0–0.01562 norm: 222 (2.3%) |
| `distortion_filter_resonance` | Linear | 9,618 | 0–1 | 0.43418 ± 0.16579 | 71.93803% | 7.3404% | normalized | 0.00991 / 0.50563 / 0.51535 | 0.5–0.51562 raw / 0.5–0.51562 norm: 6,961 (72.4%)<br>0–0.01562 raw / 0–0.01562 norm: 758 (7.9%)<br>0.26562–0.28125 raw / 0.26562–0.28125 norm: 92 (1.0%) |
| `distortion_mix` | Linear | 9,618 | 0–1 | 0.82626 ± 0.32114 | 72.01081% | 5.16739% | normalized | 0.01423 / 0.98928 / 0.99893 | 0.98438–1 raw / 0.98438–1 norm: 7,009 (72.9%)<br>0–0.01562 raw / 0–0.01562 norm: 528 (5.5%)<br>0.5–0.51562 raw / 0.5–0.51562 norm: 102 (1.1%) |
| `env_1_attack` | Quartic | 9,618 | 0–2.37842 | 0.26548 ± 0.26912 | 46.11146% | 12.94448% | normalized | 0.40377 / 0.71801 / 0.90941 | 0–0.8409 raw / 0–0.01562 norm: 9,046 (94.1%)<br>0.8409–1 raw / 0.01562–0.03125 norm: 245 (2.5%)<br>1–1.10668 raw / 0.03125–0.04688 norm: 114 (1.2%) |
| `env_1_attack_power` | Linear | 9,618 | -20–20 | 0.23306 ± 1.68719 | 87.46101% | 87.46101% | normalized | 0.0084 / 0.32762 / 2.31503 | 0–0.625 raw / 0.5–0.51562 norm: 8,473 (88.1%)<br>2.5–3.125 raw / 0.5625–0.57812 norm: 119 (1.2%)<br>1.25–1.875 raw / 0.53125–0.54688 norm: 116 (1.2%) |
| `env_1_decay` | Quartic | 9,618 | 0–2.37842 | 0.94135 ± 0.24049 | 46.60012% | 1.99626% | normalized | 0.58183 / 0.92252 / 1.23055 | 0.8409–1 raw / 0.01562–0.03125 norm: 6,043 (62.8%)<br>0–0.8409 raw / 0–0.01562 norm: 2,098 (21.8%)<br>1–1.10668 raw / 0.03125–0.04688 norm: 644 (6.7%) |
| `env_1_decay_power` | Linear | 9,618 | -20–20 | -1.94001 ± 2.38203 | 79.49678% | 0.84217% | normalized | -3.62057 / -2.17764 / 0.77096 | -2.5–-1.875 raw / 0.4375–0.45312 norm: 7,826 (81.4%)<br>-3.125–-2.5 raw / 0.42188–0.4375 norm: 196 (2.0%)<br>0–0.625 raw / 0.5–0.51562 norm: 154 (1.6%) |
| `env_1_delay` | Quartic | 9,618 | 0–0.77379 | 0.00248 ± 0.02744 | 97.75421% | 97.75421% | normalized | 0.23648 / 0.42052 / 0.49372 | 0–0.5 raw / 0–0.01562 norm: 9,610 (99.9%)<br>0.5–0.5946 raw / 0.01562–0.03125 norm: 7 (0.1%)<br>0.74767–0.78254 raw / 0.07812–0.09375 norm: 1 (0.0%) |
| `env_1_hold` | Quartic | 9,618 | 0–1.41421 | 0.03354 ± 0.16307 | 93.85527% | 93.85527% | normalized | 0.2385 / 0.42412 / 0.49794 | 0–0.5 raw / 0–0.01562 norm: 9,288 (96.6%)<br>0.5–0.5946 raw / 0.01562–0.03125 norm: 50 (0.5%)<br>0.5946–0.65804 raw / 0.03125–0.04688 norm: 39 (0.4%) |
| `env_1_release` | Quartic | 9,618 | 0–2.37842 | 0.64511 ± 0.32462 | 39.23893% | 7.49636% | normalized | 0.42524 / 0.75619 / 1.22996 | 0–0.8409 raw / 0–0.01562 norm: 7,353 (76.5%)<br>0.8409–1 raw / 0.01562–0.03125 norm: 908 (9.4%)<br>1–1.10668 raw / 0.03125–0.04688 norm: 486 (5.1%) |
| `env_1_release_power` | Linear | 9,618 | -20–20 | -1.99263 ± 1.36352 | 88.86463% | 0.7174% | normalized | -2.4946 / -2.18223 / -1.43472 | -2.5–-1.875 raw / 0.4375–0.45312 norm: 8,659 (90.0%)<br>0–0.625 raw / 0.5–0.51562 norm: 126 (1.3%)<br>-3.125–-2.5 raw / 0.42188–0.4375 norm: 110 (1.1%) |
| `env_1_sustain` | Linear | 9,618 | 0–1 | 0.72816 ± 0.38573 | 55.67686% | 14.83676% | normalized | 0.00503 / 0.9861 / 0.99861 | 0.98438–1 raw / 0.98438–1 norm: 5,407 (56.2%)<br>0–0.01562 raw / 0–0.01562 norm: 1,494 (15.5%)<br>0.75–0.76562 raw / 0.75–0.76562 norm: 96 (1.0%) |
| `env_2_attack` | Quartic | 9,618 | 0–2.37842 | 0.20136 ± 0.22092 | 77.89561% | 7.39239% | normalized | 0.40127 / 0.71358 / 0.83778 | 0–0.8409 raw / 0–0.01562 norm: 9,273 (96.4%)<br>0.8409–1 raw / 0.01562–0.03125 norm: 102 (1.1%)<br>1–1.10668 raw / 0.03125–0.04688 norm: 63 (0.7%) |
| `env_2_attack_power` | Linear | 9,618 | -20–20 | 0.0321 ± 1.02093 | 94.65585% | 94.65585% | normalized | 0.01711 / 0.31356 / 0.61001 | 0–0.625 raw / 0.5–0.51562 norm: 9,124 (94.9%)<br>-1.875–-1.25 raw / 0.45312–0.46875 norm: 51 (0.5%)<br>1.875–2.5 raw / 0.54688–0.5625 norm: 46 (0.5%) |
| `env_2_decay` | Quartic | 9,618 | 0–2.37842 | 0.92295 ± 0.21429 | 63.9842% | 0.97733% | normalized | 0.5797 / 0.91341 / 1.07899 | 0.8409–1 raw / 0.01562–0.03125 norm: 6,833 (71.0%)<br>0–0.8409 raw / 0–0.01562 norm: 2,129 (22.1%)<br>1–1.10668 raw / 0.03125–0.04688 norm: 245 (2.5%) |
| `env_2_decay_power` | Linear | 9,618 | -20–20 | -2.16631 ± 1.83918 | 87.44022% | 0.64462% | normalized | -3.39369 / -2.19617 / -1.87808 | -2.5–-1.875 raw / 0.4375–0.45312 norm: 8,503 (88.4%)<br>-3.125–-2.5 raw / 0.42188–0.4375 norm: 146 (1.5%)<br>-3.75–-3.125 raw / 0.40625–0.42188 norm: 112 (1.2%) |
| `env_2_delay` | Quartic | 9,618 | 0–1.41421 | 0.00283 ± 0.04104 | 98.67956% | 98.67956% | normalized | 0.23651 / 0.42058 / 0.49378 | 0–0.5 raw / 0–0.01562 norm: 9,605 (99.9%)<br>0.5–0.5946 raw / 0.01562–0.03125 norm: 2 (0.0%)<br>0.5946–0.65804 raw / 0.03125–0.04688 norm: 2 (0.0%) |
| `env_2_hold` | Quartic | 9,618 | 0–1.41421 | 0.00972 ± 0.08193 | 97.7854% | 97.7854% | normalized | 0.237 / 0.42145 / 0.4948 | 0–0.5 raw / 0–0.01562 norm: 9,526 (99.0%)<br>0.5–0.5946 raw / 0.01562–0.03125 norm: 15 (0.2%)<br>0.5946–0.65804 raw / 0.03125–0.04688 norm: 15 (0.2%) |
| `env_2_release` | Quartic | 9,618 | 0–2.37842 | 0.58379 ± 0.22944 | 78.43627% | 4.01331% | normalized | 0.40721 / 0.72413 / 1.04545 | 0–0.8409 raw / 0–0.01562 norm: 8,744 (90.9%)<br>0.8409–1 raw / 0.01562–0.03125 norm: 319 (3.3%)<br>1–1.10668 raw / 0.03125–0.04688 norm: 188 (2.0%) |
| `env_2_release_power` | Linear | 9,618 | -20–19.18002 | -1.98755 ± 0.63888 | 97.73342% | 0.16635% | normalized | -2.47232 / -2.18565 / -1.89897 | -2.5–-1.875 raw / 0.4375–0.45312 norm: 9,435 (98.1%)<br>0–0.625 raw / 0.5–0.51562 norm: 31 (0.3%)<br>-3.125–-2.5 raw / 0.42188–0.4375 norm: 22 (0.2%) |
| `env_2_sustain` | Linear | 9,618 | 0–1 | 0.70281 ± 0.43073 | 64.95113% | 21.28301% | normalized | 0.00354 / 0.98798 / 0.9988 | 0.98438–1 raw / 0.98438–1 norm: 6,253 (65.0%)<br>0–0.01562 raw / 0–0.01562 norm: 2,120 (22.0%)<br>0.1875–0.20312 raw / 0.1875–0.20312 norm: 43 (0.4%) |
| `env_3_attack` | Quartic | 9,618 | 0–2.02273 | 0.17313 ± 0.15465 | 91.12082% | 3.04637% | normalized | 0.39941 / 0.71027 / 0.83389 | 0–0.8409 raw / 0–0.01562 norm: 9,447 (98.2%)<br>0.8409–1 raw / 0.01562–0.03125 norm: 45 (0.5%)<br>1.10668–1.18921 raw / 0.04688–0.0625 norm: 29 (0.3%) |
| `env_3_attack_power` | Linear | 9,618 | -19.04001–15.53999 | 0.01572 ± 0.66132 | 97.84779% | 97.84779% | normalized | 0.02601 / 0.31323 / 0.60045 | 0–0.625 raw / 0.5–0.51562 norm: 9,417 (97.9%)<br>1.25–1.875 raw / 0.53125–0.54688 norm: 31 (0.3%)<br>-1.875–-1.25 raw / 0.45312–0.46875 norm: 27 (0.3%) |
| `env_3_decay` | Quartic | 9,618 | 0–2.37842 | 0.96302 ± 0.14815 | 84.90331% | 0.34311% | normalized | 0.70343 / 0.92362 / 0.99631 | 0.8409–1 raw / 0.01562–0.03125 norm: 8,401 (87.3%)<br>0–0.8409 raw / 0–0.01562 norm: 982 (10.2%)<br>1–1.10668 raw / 0.03125–0.04688 norm: 83 (0.9%) |
| `env_3_decay_power` | Linear | 9,618 | -20–14.98 | -2.04747 ± 0.9764 | 95.62279% | 0.15596% | normalized | -2.483 / -2.18964 / -1.89628 | -2.5–-1.875 raw / 0.4375–0.45312 norm: 9,220 (95.9%)<br>-3.125–-2.5 raw / 0.42188–0.4375 norm: 42 (0.4%)<br>-3.75–-3.125 raw / 0.40625–0.42188 norm: 36 (0.4%) |
| `env_3_delay` | Quartic | 9,618 | 0–1.41421 | 0.00319 ± 0.04584 | 99.15783% | 99.15783% | normalized | 0.23656 / 0.42068 / 0.4939 | 0–0.5 raw / 0–0.01562 norm: 9,596 (99.8%)<br>0.5–0.5946 raw / 0.01562–0.03125 norm: 6 (0.1%)<br>0.65804–0.70711 raw / 0.04688–0.0625 norm: 2 (0.0%) |
| `env_3_hold` | Quartic | 9,618 | 0–1.41421 | 0.00402 ± 0.05611 | 99.20981% | 99.20981% | normalized | 0.23665 / 0.42083 / 0.49408 | 0–0.5 raw / 0–0.01562 norm: 9,582 (99.6%)<br>0.5–0.5946 raw / 0.01562–0.03125 norm: 9 (0.1%)<br>0.65804–0.70711 raw / 0.04688–0.0625 norm: 3 (0.0%) |
| `env_3_release` | Quartic | 9,618 | 0–2.37842 | 0.55966 ± 0.15194 | 91.15201% | 2.17301% | normalized | 0.4014 / 0.71381 / 0.83805 | 0–0.8409 raw / 0–0.01562 norm: 9,261 (96.3%)<br>0.8409–1 raw / 0.01562–0.03125 norm: 140 (1.5%)<br>1–1.10668 raw / 0.03125–0.04688 norm: 61 (0.6%) |
| `env_3_release_power` | Linear | 9,618 | -20–14.68004 | -2.00247 ± 0.49258 | 99.22021% | 0.0104% | normalized | -2.47044 / -2.18737 / -1.90429 | -2.5–-1.875 raw / 0.4375–0.45312 norm: 9,555 (99.3%)<br>-4.375–-3.75 raw / 0.39062–0.40625 norm: 7 (0.1%)<br>-3.125–-2.5 raw / 0.42188–0.4375 norm: 5 (0.1%) |
| `env_3_sustain` | Linear | 9,618 | 0–1 | 0.86837 ± 0.32468 | 84.9137% | 9.48222% | normalized | 0.00788 / 0.9908 / 0.99908 | 0.98438–1 raw / 0.98438–1 norm: 8,169 (84.9%)<br>0–0.01562 raw / 0–0.01562 norm: 954 (9.9%)<br>0.1875–0.20312 raw / 0.1875–0.20312 norm: 25 (0.3%) |
| `env_4_attack` | Quartic | 9,618 | 0–1.89539 | 0.16049 ± 0.10334 | 96.83926% | 0.92535% | normalized | 0.39836 / 0.7084 / 0.8317 | 0–0.8409 raw / 0–0.01562 norm: 9,547 (99.3%)<br>1.18921–1.25744 raw / 0.0625–0.07812 norm: 16 (0.2%)<br>0.8409–1 raw / 0.01562–0.03125 norm: 13 (0.1%) |
| `env_4_attack_power` | Linear | 9,618 | -20–8.04 | -0.00532 ± 0.39029 | 98.94989% | 98.94989% | normalized | 0.02763 / 0.31178 / 0.59592 | 0–0.625 raw / 0.5–0.51562 norm: 9,519 (99.0%)<br>-1.875–-1.25 raw / 0.45312–0.46875 norm: 20 (0.2%)<br>-3.125–-2.5 raw / 0.42188–0.4375 norm: 11 (0.1%) |
| `env_4_decay` | Quartic | 9,618 | 0–2.00304 | 0.98654 ± 0.08568 | 94.64546% | 0.10397% | normalized | 0.84372 / 0.92815 / 0.99433 | 0.8409–1 raw / 0.01562–0.03125 norm: 9,193 (95.6%)<br>0–0.8409 raw / 0–0.01562 norm: 357 (3.7%)<br>1–1.10668 raw / 0.03125–0.04688 norm: 24 (0.2%) |
| `env_4_decay_power` | Linear | 9,618 | -20–18.28 | -2.02233 ± 0.70663 | 98.43003% | 0.07278% | normalized | -2.47421 / -2.18862 / -1.90304 | -2.5–-1.875 raw / 0.4375–0.45312 norm: 9,471 (98.5%)<br>-3.125–-2.5 raw / 0.42188–0.4375 norm: 16 (0.2%)<br>-3.75–-3.125 raw / 0.40625–0.42188 norm: 13 (0.1%) |
| `env_4_delay` | Quartic | 9,618 | 0–0.8409 | 0.0009296 ± 0.02014 | 99.66729% | 99.66729% | normalized | 0.23646 / 0.42049 / 0.49368 | 0–0.5 raw / 0–0.01562 norm: 9,613 (99.9%)<br>0.5–0.5946 raw / 0.01562–0.03125 norm: 3 (0.0%)<br>0.65804–0.70711 raw / 0.04688–0.0625 norm: 1 (0.0%) |
| `env_4_hold` | Quartic | 9,618 | 0–1.05511 | 0.00118 ± 0.02579 | 99.69848% | 99.69848% | normalized | 0.23649 / 0.42055 / 0.49374 | 0–0.5 raw / 0–0.01562 norm: 9,608 (99.9%)<br>0.5–0.5946 raw / 0.01562–0.03125 norm: 7 (0.1%)<br>0.65804–0.70711 raw / 0.04688–0.0625 norm: 1 (0.0%) |
| `env_4_release` | Quartic | 9,618 | 0–2.37842 | 0.55178 ± 0.10697 | 96.54814% | 1.06051% | normalized | 0.39897 / 0.70948 / 0.83297 | 0–0.8409 raw / 0–0.01562 norm: 9,489 (98.7%)<br>0.8409–1 raw / 0.01562–0.03125 norm: 39 (0.4%)<br>1–1.10668 raw / 0.03125–0.04688 norm: 27 (0.3%) |
| `env_4_release_power` | Linear | 9,618 | -20–4.74531 | -2.00068 ± 0.25862 | 99.79206% | 0% | normalized | -2.46915 / -2.18737 / -1.90559 | -2.5–-1.875 raw / 0.4375–0.45312 norm: 9,599 (99.8%)<br>-0.625–0 raw / 0.48438–0.5 norm: 3 (0.0%)<br>0.625–1.25 raw / 0.51562–0.53125 norm: 3 (0.0%) |
| `env_4_sustain` | Linear | 9,618 | 0–1 | 0.95205 ± 0.2072 | 94.59347% | 3.6598% | normalized | 0.4952 / 0.99174 / 0.99917 | 0.98438–1 raw / 0.98438–1 norm: 9,099 (94.6%)<br>0–0.01562 raw / 0–0.01562 norm: 363 (3.8%)<br>0.15625–0.17188 raw / 0.15625–0.17188 norm: 9 (0.1%) |
| `env_5_attack` | Quartic | 9,618 | 0–1.57574 | 0.15329 ± 0.06011 | 98.68996% | 0.43668% | normalized | 0.39794 / 0.70764 / 0.83081 | 0–0.8409 raw / 0–0.01562 norm: 9,588 (99.7%)<br>0.8409–1 raw / 0.01562–0.03125 norm: 10 (0.1%)<br>1.10668–1.18921 raw / 0.04688–0.0625 norm: 9 (0.1%) |
| `env_5_attack_power` | Linear | 9,618 | -19.4–14.82 | 0.0034 ± 0.39419 | 99.59451% | 99.59451% | normalized | 0.03 / 0.31237 / 0.59474 | 0–0.625 raw / 0.5–0.51562 norm: 9,579 (99.6%)<br>-1.875–-1.25 raw / 0.45312–0.46875 norm: 7 (0.1%)<br>1.875–2.5 raw / 0.54688–0.5625 norm: 6 (0.1%) |
| `env_5_decay` | Quartic | 9,618 | 0–2.37842 | 0.99426 ± 0.06313 | 97.7854% | 0.13516% | normalized | 0.84814 / 0.92964 / 0.99404 | 0.8409–1 raw / 0.01562–0.03125 norm: 9,430 (98.0%)<br>0–0.8409 raw / 0–0.01562 norm: 152 (1.6%)<br>1–1.10668 raw / 0.03125–0.04688 norm: 13 (0.1%) |
| `env_5_decay_power` | Linear | 9,618 | -20–20 | -2.00785 ± 0.54845 | 99.23061% | 0.0104% | normalized | -2.47153 / -2.18819 / -1.90485 | -2.5–-1.875 raw / 0.4375–0.45312 norm: 9,546 (99.3%)<br>-3.75–-3.125 raw / 0.40625–0.42188 norm: 9 (0.1%)<br>-3.125–-2.5 raw / 0.42188–0.4375 norm: 9 (0.1%) |
| `env_5_delay` | Quartic | 9,618 | 0–0.98885 | 0.00114 ± 0.03159 | 99.84404% | 99.84404% | normalized | 0.2365 / 0.42057 / 0.49377 | 0–0.5 raw / 0–0.01562 norm: 9,606 (99.9%)<br>0.8409–0.86603 raw / 0.125–0.14062 norm: 4 (0.0%)<br>0.94941–0.96717 raw / 0.20312–0.21875 norm: 4 (0.0%) |
| `env_5_hold` | Quartic | 9,618 | 0–0.99437 | 0.00103 ± 0.02568 | 99.80245% | 99.80245% | normalized | 0.23648 / 0.42052 / 0.49372 | 0–0.5 raw / 0–0.01562 norm: 9,610 (99.9%)<br>0.5–0.5946 raw / 0.01562–0.03125 norm: 4 (0.0%)<br>0.78254–0.81329 raw / 0.09375–0.10938 norm: 2 (0.0%) |
| `env_5_release` | Quartic | 9,618 | 0–1.89601 | 0.54858 ± 0.06182 | 98.70035% | 0.42628% | normalized | 0.39811 / 0.70796 / 0.83118 | 0–0.8409 raw / 0–0.01562 norm: 9,571 (99.5%)<br>0.8409–1 raw / 0.01562–0.03125 norm: 17 (0.2%)<br>1.41422–1.45648 raw / 0.125–0.14062 norm: 8 (0.1%) |
| `env_5_release_power` | Linear | 9,618 | -16.28–1.42 | -2.00086 ± 0.1582 | 99.90643% | 0.0104% | normalized | -2.46892 / -2.18747 / -1.90601 | -2.5–-1.875 raw / 0.4375–0.45312 norm: 9,610 (99.9%)<br>0–0.625 raw / 0.5–0.51562 norm: 2 (0.0%)<br>-16.875–-16.25 raw / 0.07812–0.09375 norm: 1 (0.0%) |
| `env_5_sustain` | Linear | 9,618 | 0–1 | 0.98052 ± 0.13432 | 97.83739% | 1.4764% | normalized | 0.98483 / 0.99201 / 0.9992 | 0.98438–1 raw / 0.98438–1 norm: 9,410 (97.8%)<br>0–0.01562 raw / 0–0.01562 norm: 150 (1.6%)<br>0.625–0.64062 raw / 0.625–0.64062 norm: 10 (0.1%) |
| `env_6_attack` | Quartic | 9,618 | 0–1.25094 | 0.15107 ± 0.03854 | 99.30339% | 0.21834% | normalized | 0.39778 / 0.70737 / 0.83049 | 0–0.8409 raw / 0–0.01562 norm: 9,603 (99.8%)<br>0.8409–1 raw / 0.01562–0.03125 norm: 9 (0.1%)<br>1.18921–1.25744 raw / 0.0625–0.07812 norm: 5 (0.1%) |
| `env_6_attack_power` | Linear | 9,618 | -12.30001–13.86 | 0.0002807 ± 0.20985 | 99.84404% | 99.84404% | normalized | 0.03064 / 0.3123 / 0.59396 | 0–0.625 raw / 0.5–0.51562 norm: 9,603 (99.8%)<br>-1.875–-1.25 raw / 0.45312–0.46875 norm: 7 (0.1%)<br>1.875–2.5 raw / 0.54688–0.5625 norm: 2 (0.0%) |
| `env_6_decay` | Quartic | 9,618 | 0–1.4079 | 0.99776 ± 0.03217 | 99.23061% | 0.0104% | normalized | 0.85007 / 0.93018 / 0.99372 | 0.8409–1 raw / 0.01562–0.03125 norm: 9,555 (99.3%)<br>0–0.8409 raw / 0–0.01562 norm: 57 (0.6%)<br>1–1.10668 raw / 0.03125–0.04688 norm: 3 (0.0%) |
| `env_6_decay_power` | Linear | 9,618 | -11.91353–8.62 | -2.00198 ± 0.22396 | 99.74007% | 0% | normalized | -2.46959 / -2.1877 / -1.9058 | -2.5–-1.875 raw / 0.4375–0.45312 norm: 9,595 (99.8%)<br>-4.375–-3.75 raw / 0.39062–0.40625 norm: 6 (0.1%)<br>-5–-4.375 raw / 0.375–0.39062 norm: 4 (0.0%) |
| `env_6_delay` | Quartic | 9,618 | 0–0.57901 | 0.000218 ± 0.00923 | 99.91682% | 99.91682% | normalized | 0.23644 / 0.42045 / 0.49363 | 0–0.5 raw / 0–0.01562 norm: 9,617 (100.0%)<br>0.5–0.5946 raw / 0.01562–0.03125 norm: 1 (0.0%) |
| `env_6_hold` | Quartic | 9,618 | 0–1.22729 | 0.0002275 ± 0.01432 | 99.96881% | 99.96881% | normalized | 0.23644 / 0.42046 / 0.49364 | 0–0.5 raw / 0–0.01562 norm: 9,616 (100.0%)<br>0.5–0.5946 raw / 0.01562–0.03125 norm: 1 (0.0%)<br>1.22474–1.23316 raw / 0.5625–0.57812 norm: 1 (0.0%) |
| `env_6_release` | Quartic | 9,618 | 0–2.37842 | 0.54771 ± 0.03772 | 99.55292% | 0.17675% | normalized | 0.39778 / 0.70737 / 0.83049 | 0–0.8409 raw / 0–0.01562 norm: 9,603 (99.8%)<br>0.8409–1 raw / 0.01562–0.03125 norm: 3 (0.0%)<br>1–1.10668 raw / 0.03125–0.04688 norm: 3 (0.0%) |
| `env_6_release_power` | Linear | 9,618 | -2–0.64 | -1.99949 ± 0.03294 | 99.96881% | 0% | normalized | -2.46874 / -2.18743 / -1.90613 | -2.5–-1.875 raw / 0.4375–0.45312 norm: 9,615 (100.0%)<br>-1.875–-1.25 raw / 0.45312–0.46875 norm: 1 (0.0%)<br>-0.625–0 raw / 0.48438–0.5 norm: 1 (0.0%) |
| `env_6_sustain` | Linear | 9,618 | 0–1 | 0.99323 ± 0.08122 | 99.29299% | 0.60304% | normalized | 0.98505 / 0.99213 / 0.99921 | 0.98438–1 raw / 0.98438–1 norm: 9,550 (99.3%)<br>0–0.01562 raw / 0–0.01562 norm: 60 (0.6%)<br>0.01562–0.03125 raw / 0.01562–0.03125 norm: 1 (0.0%) |
| `eq_band_cutoff` | Linear | 9,618 | 8–136 | 76.42425 ± 17.44478 | 46.58973% | 0% | normalized | 44.71702 / 80.63489 / 103.02203 | 80–82 raw / 0.5625–0.57812 norm: 4,689 (48.8%)<br>76–78 raw / 0.53125–0.54688 norm: 246 (2.6%)<br>74–76 raw / 0.51562–0.53125 norm: 244 (2.5%) |
| `eq_band_gain` | Linear | 9,618 | -15–15 | -0.14475 ± 5.33218 | 49.89603% | 49.89603% | normalized | -11.00657 / 0.24053 / 8.61348 | 0–0.46875 raw / 0.5–0.51562 norm: 4,914 (51.1%)<br>-15–-14.53125 raw / 0–0.01562 norm: 316 (3.3%)<br>14.53125–15 raw / 0.98438–1 norm: 167 (1.7%) |
| `eq_band_resonance` | Quadratic | 9,618 | 0–1 | 0.40596 ± 0.15325 | 66.9162% | 5.12581% | normalized | 0.10173 / 0.43989 / 0.58692 | 0.43301–0.45069 raw / 0.1875–0.20312 norm: 6,486 (67.4%)<br>0–0.125 raw / 0–0.01562 norm: 726 (7.5%)<br>0.25–0.27951 raw / 0.0625–0.07812 norm: 209 (2.2%) |
| `eq_high_cutoff` | Linear | 9,618 | 8–136 | 101.64947 ± 15.60051 | 45.83073% | 0% | normalized | 75.57368 / 101.18302 / 129.21442 | 100–102 raw / 0.71875–0.73438 norm: 4,699 (48.9%)<br>134–136 raw / 0.98438–1 norm: 309 (3.2%)<br>102–104 raw / 0.73438–0.75 norm: 264 (2.7%) |
| `eq_high_gain` | Linear | 9,618 | -15–19.74391 | -0.2592 ± 5.42331 | 50.68621% | 50.68621% | raw | -14.21283 / 0.46249 / 8.6167 | 0.20046–0.74334 raw: 4,990 (51.9%)<br>-15–-14.45713 raw: 466 (4.8%)<br>-0.34241–0.20046 raw: 367 (3.8%) |
| `eq_high_resonance` | Quadratic | 9,618 | 0–1 | 0.29915 ± 0.1011 | 78.23872% | 5.10501% | normalized | 0.10519 / 0.31745 / 0.37144 | 0.30619–0.33072 raw / 0.09375–0.10938 norm: 7,619 (79.2%)<br>0–0.125 raw / 0–0.01562 norm: 679 (7.1%)<br>0.125–0.17678 raw / 0.01562–0.03125 norm: 162 (1.7%) |
| `eq_low_cutoff` | Linear | 9,618 | 8–136 | 41.10883 ± 12.43008 | 45.5812% | 0% | normalized | 18.97234 / 41.08741 / 62.18485 | 40–42 raw / 0.25–0.26562 norm: 4,656 (48.4%)<br>44–46 raw / 0.28125–0.29688 norm: 309 (3.2%)<br>38–40 raw / 0.23438–0.25 norm: 286 (3.0%) |
| `eq_low_gain` | Linear | 9,618 | -15–16.13977 | -1.56859 ± 6.07102 | 51.55958% | 51.55958% | raw | -14.74816 / -0.20485 / 6.93772 | -0.40323–0.08332 raw: 4,931 (51.3%)<br>-15–-14.51344 raw: 929 (9.7%)<br>0.08332–0.56988 raw: 307 (3.2%) |
| `eq_low_resonance` | Quadratic | 9,618 | 0–1 | 0.30364 ± 0.09295 | 80.25577% | 3.6702% | normalized | 0.12009 / 0.31751 / 0.35823 | 0.30619–0.33072 raw / 0.09375–0.10938 norm: 7,802 (81.1%)<br>0–0.125 raw / 0–0.01562 norm: 521 (5.4%)<br>0.25–0.27951 raw / 0.0625–0.07812 norm: 181 (1.9%) |
| `filter_1_blend` | Linear | 9,618 | 0–2 | 0.24974 ± 0.57081 | 78.49865% | 78.49865% | normalized | 0.00197 / 0.0197 / 1.9756 | 0–0.03125 raw / 0–0.01562 norm: 7,626 (79.3%)<br>1.96875–2 raw / 0.98438–1 norm: 617 (6.4%)<br>1–1.03125 raw / 0.5–0.51562 norm: 550 (5.7%) |
| `filter_1_blend_transpose` | Linear | 9,618 | 0–84 | 41.85632 ± 7.15207 | 94.52069% | 0.85257% | normalized | 42.02709 / 42.65113 / 43.27517 | 42–43.3125 raw / 0.5–0.51562 norm: 9,102 (94.6%)<br>82.6875–84 raw / 0.98438–1 norm: 104 (1.1%)<br>0–1.3125 raw / 0–0.01562 norm: 100 (1.0%) |
| `filter_1_cutoff` | Linear | 9,618 | 8–136 | 65.21122 ± 28.29684 | 27.71886% | 0% | normalized | 16.3425 / 61.22698 / 122.62308 | 60–62 raw / 0.40625–0.42188 norm: 2,824 (29.4%)<br>8–10 raw / 0–0.01562 norm: 379 (3.9%)<br>134–136 raw / 0.98438–1 norm: 255 (2.7%) |
| `filter_1_drive` | Linear | 9,618 | 0–20 | 2.38811 ± 5.17751 | 72.37471% | 72.37471% | normalized | 0.02124 / 0.21242 / 16.90264 | 0–0.3125 raw / 0–0.01562 norm: 7,074 (73.5%)<br>19.6875–20 raw / 0.98438–1 norm: 419 (4.4%)<br>2.5–2.8125 raw / 0.125–0.14062 norm: 81 (0.8%) |
| `filter_1_formant_resonance` | Linear | 9,618 | 0.3–1 | 0.84736 ± 0.04054 | 98.21169% | 0% | normalized | 0.84731 / 0.85232 / 0.85733 | 0.84688–0.85781 raw / 0.78125–0.79688 norm: 9,447 (98.2%)<br>0.98906–1 raw / 0.98438–1 norm: 43 (0.4%)<br>0.3–0.31094 raw / 0–0.01562 norm: 28 (0.3%) |
| `filter_1_formant_spread` | Linear | 9,618 | -1–1 | -0.00245 ± 0.07816 | 98.84591% | 98.84591% | normalized | 0.00137 / 0.01559 / 0.02982 | 0–0.03125 raw / 0.5–0.51562 norm: 9,507 (98.8%)<br>-1–-0.96875 raw / 0–0.01562 norm: 31 (0.3%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 9 (0.1%) |
| `filter_1_formant_transpose` | Linear | 9,618 | -12–12 | -0.06751 ± 1.34293 | 97.82699% | 97.82699% | normalized | 0.01374 / 0.1862 / 0.35867 | 0–0.375 raw / 0.5–0.51562 norm: 9,410 (97.8%)<br>-12–-11.625 raw / 0–0.01562 norm: 54 (0.6%)<br>11.625–12 raw / 0.98438–1 norm: 19 (0.2%) |
| `filter_1_formant_x` | Linear | 9,618 | 0–1 | 0.49927 ± 0.05357 | 96.89125% | 0.39509% | normalized | 0.50055 / 0.5078 / 0.51505 | 0.5–0.51562 raw / 0.5–0.51562 norm: 9,328 (97.0%)<br>0–0.01562 raw / 0–0.01562 norm: 39 (0.4%)<br>0.6875–0.70312 raw / 0.6875–0.70312 norm: 33 (0.3%) |
| `filter_1_formant_y` | Linear | 9,618 | 0–1 | 0.49966 ± 0.06033 | 97.50468% | 0.56145% | normalized | 0.5006 / 0.50781 / 0.51501 | 0.5–0.51562 raw / 0.5–0.51562 norm: 9,386 (97.6%)<br>0–0.01562 raw / 0–0.01562 norm: 57 (0.6%)<br>0.98438–1 raw / 0.98438–1 norm: 49 (0.5%) |
| `filter_1_keytrack` | Linear | 9,618 | -1–1 | 0.1422 ± 0.36311 | 80.08942% | 80.08942% | normalized | 0.00126 / 0.01881 / 0.98795 | 0–0.03125 raw / 0.5–0.51562 norm: 7,705 (80.1%)<br>0.96875–1 raw / 0.98438–1 norm: 1,250 (13.0%)<br>-1–-0.96875 raw / 0–0.01562 norm: 62 (0.6%) |
| `filter_1_mix` | Linear | 9,618 | 0–1 | 0.9603 ± 0.16558 | 91.14161% | 1.51799% | normalized | 0.70299 / 0.9915 / 0.99915 | 0.98438–1 raw / 0.98438–1 norm: 8,838 (91.9%)<br>0–0.01562 raw / 0–0.01562 norm: 167 (1.7%)<br>0.5–0.51562 raw / 0.5–0.51562 norm: 29 (0.3%) |
| `filter_1_resonance` | Linear | 9,618 | 0–1 | 0.31895 ± 0.26902 | 29.4136% | 25.88896% | normalized | 0.00287 / 0.36555 / 0.77753 | 0.5–0.51562 raw / 0.5–0.51562 norm: 2,897 (30.1%)<br>0–0.01562 raw / 0–0.01562 norm: 2,619 (27.2%)<br>0.98438–1 raw / 0.98438–1 norm: 233 (2.4%) |
| `filter_2_blend` | Linear | 9,618 | 0–2 | 0.27028 ± 0.59843 | 78.70659% | 78.70659% | normalized | 0.00197 / 0.01971 / 1.97852 | 0–0.03125 raw / 0–0.01562 norm: 7,623 (79.3%)<br>1.96875–2 raw / 0.98438–1 norm: 701 (7.3%)<br>1–1.03125 raw / 0.5–0.51562 norm: 552 (5.7%) |
| `filter_2_blend_transpose` | Linear | 9,618 | 0–84 | 41.61748 ± 6.79415 | 94.92618% | 0.99813% | normalized | 42.02355 / 42.64547 / 43.2674 | 42–43.3125 raw / 0.5–0.51562 norm: 9,133 (95.0%)<br>0–1.3125 raw / 0–0.01562 norm: 108 (1.1%)<br>82.6875–84 raw / 0.98438–1 norm: 75 (0.8%) |
| `filter_2_cutoff` | Linear | 9,618 | 8–136 | 64.59766 ± 22.16269 | 54.32522% | 0% | normalized | 30.22833 / 61.16829 / 111.72041 | 60–62 raw / 0.40625–0.42188 norm: 5,342 (55.5%)<br>8–10 raw / 0–0.01562 norm: 215 (2.2%)<br>134–136 raw / 0.98438–1 norm: 153 (1.6%) |
| `filter_2_drive` | Linear | 9,618 | 0–20 | 1.44166 ± 4.24224 | 83.72843% | 83.72843% | normalized | 0.01851 / 0.18512 / 12.28217 | 0–0.3125 raw / 0–0.01562 norm: 8,117 (84.4%)<br>19.6875–20 raw / 0.98438–1 norm: 257 (2.7%)<br>1.25–1.5625 raw / 0.0625–0.07812 norm: 57 (0.6%) |
| `filter_2_formant_resonance` | Linear | 9,618 | 0.3–1 | 0.84695 ± 0.04156 | 98.34685% | 0% | normalized | 0.84731 / 0.85232 / 0.85732 | 0.84688–0.85781 raw / 0.78125–0.79688 norm: 9,459 (98.3%)<br>0.98906–1 raw / 0.98438–1 norm: 34 (0.4%)<br>0.3–0.31094 raw / 0–0.01562 norm: 32 (0.3%) |
| `filter_2_formant_spread` | Linear | 9,618 | -1–1 | -0.00301 ± 0.06908 | 98.91869% | 98.91869% | normalized | 0.00135 / 0.01556 / 0.02978 | 0–0.03125 raw / 0.5–0.51562 norm: 9,514 (98.9%)<br>-1–-0.96875 raw / 0–0.01562 norm: 19 (0.2%)<br>-0.34375–-0.3125 raw / 0.32812–0.34375 norm: 5 (0.1%) |
| `filter_2_formant_transpose` | Linear | 9,618 | -12–12 | -0.04611 ± 1.25765 | 98.04533% | 98.04533% | normalized | 0.01463 / 0.18672 / 0.35882 | 0–0.375 raw / 0.5–0.51562 norm: 9,430 (98.0%)<br>-12–-11.625 raw / 0–0.01562 norm: 32 (0.3%)<br>-11.625–-11.25 raw / 0.01562–0.03125 norm: 27 (0.3%) |
| `filter_2_formant_x` | Linear | 9,618 | 0–1 | 0.49882 ± 0.05178 | 97.50468% | 0.40549% | normalized | 0.50057 / 0.50777 / 0.51498 | 0.5–0.51562 raw / 0.5–0.51562 norm: 9,386 (97.6%)<br>0–0.01562 raw / 0–0.01562 norm: 43 (0.4%)<br>0.98438–1 raw / 0.98438–1 norm: 28 (0.3%) |
| `filter_2_formant_y` | Linear | 9,618 | 0–1 | 0.499 ± 0.05464 | 97.89977% | 0.49906% | normalized | 0.50061 / 0.50779 / 0.51497 | 0.5–0.51562 raw / 0.5–0.51562 norm: 9,420 (97.9%)<br>0–0.01562 raw / 0–0.01562 norm: 51 (0.5%)<br>0.98438–1 raw / 0.98438–1 norm: 33 (0.3%) |
| `filter_2_keytrack` | Linear | 9,618 | -1–1 | 0.10375 ± 0.31783 | 86.31732% | 86.31732% | normalized | 0.00137 / 0.01765 / 0.98412 | 0–0.03125 raw / 0.5–0.51562 norm: 8,304 (86.3%)<br>0.96875–1 raw / 0.98438–1 norm: 948 (9.9%)<br>-1–-0.96875 raw / 0–0.01562 norm: 38 (0.4%) |
| `filter_2_mix` | Linear | 9,618 | 0–1 | 0.95882 ± 0.16977 | 91.77584% | 1.65315% | normalized | 0.68185 / 0.99153 / 0.99915 | 0.98438–1 raw / 0.98438–1 norm: 8,869 (92.2%)<br>0–0.01562 raw / 0–0.01562 norm: 175 (1.8%)<br>0.79688–0.8125 raw / 0.79688–0.8125 norm: 21 (0.2%) |
| `filter_2_resonance` | Linear | 9,618 | 0–1 | 0.39005 ± 0.23169 | 56.13433% | 16.95779% | normalized | 0.00439 / 0.50428 / 0.68237 | 0.5–0.51562 raw / 0.5–0.51562 norm: 5,454 (56.7%)<br>0–0.01562 raw / 0–0.01562 norm: 1,712 (17.8%)<br>0.98438–1 raw / 0.98438–1 norm: 139 (1.4%) |
| `filter_fx_blend` | Linear | 9,618 | 0–2 | 0.1655 ± 0.48829 | 86.68122% | 86.68122% | normalized | 0.0018 / 0.01796 / 1.79978 | 0–0.03125 raw / 0–0.01562 norm: 8,367 (87.0%)<br>1.96875–2 raw / 0.98438–1 norm: 438 (4.6%)<br>1–1.03125 raw / 0.5–0.51562 norm: 292 (3.0%) |
| `filter_fx_blend_transpose` | Linear | 9,618 | 0–84 | 42.05031 ± 6.00078 | 96.42337% | 0.57184% | normalized | 42.04453 / 42.65653 / 43.26854 | 42–43.3125 raw / 0.5–0.51562 norm: 9,281 (96.5%)<br>82.6875–84 raw / 0.98438–1 norm: 84 (0.9%)<br>0–1.3125 raw / 0–0.01562 norm: 66 (0.7%) |
| `filter_fx_cutoff` | Linear | 9,618 | 8–136 | 68.76685 ± 28.50588 | 56.26949% | 0% | normalized | 27.99211 / 61.21136 / 134.38305 | 60–62 raw / 0.40625–0.42188 norm: 5,474 (56.9%)<br>134–136 raw / 0.98438–1 norm: 596 (6.2%)<br>8–10 raw / 0–0.01562 norm: 255 (2.7%) |
| `filter_fx_drive` | Linear | 9,618 | 0–20 | 1.11175 ± 3.62112 | 85.24641% | 85.24641% | normalized | 0.0181 / 0.18104 / 8.48931 | 0–0.3125 raw / 0–0.01562 norm: 8,300 (86.3%)<br>19.6875–20 raw / 0.98438–1 norm: 167 (1.7%)<br>1.875–2.1875 raw / 0.09375–0.10938 norm: 50 (0.5%) |
| `filter_fx_formant_resonance` | Linear | 9,618 | 0.12309–1 | 0.84539 ± 0.05107 | 98.28447% | 0% | raw | 0.84957 / 0.85595 / 0.86234 | 0.84928–0.86298 raw: 9,288 (96.6%)<br>0.83558–0.84928 raw: 167 (1.7%)<br>0.28751–0.30121 raw: 53 (0.6%) |
| `filter_fx_formant_spread` | Linear | 9,618 | -1–1 | -0.00375 ± 0.07749 | 98.66916% | 98.66916% | normalized | 0.00128 / 0.01553 / 0.02978 | 0–0.03125 raw / 0.5–0.51562 norm: 9,490 (98.7%)<br>-1–-0.96875 raw / 0–0.01562 norm: 26 (0.3%)<br>-0.53125–-0.5 raw / 0.23438–0.25 norm: 13 (0.1%) |
| `filter_fx_formant_transpose` | Linear | 9,618 | -12–12 | -0.02267 ± 1.11781 | 98.47162% | 98.47162% | normalized | 0.01563 / 0.18699 / 0.35834 | 0–0.375 raw / 0.5–0.51562 norm: 9,471 (98.5%)<br>-12–-11.625 raw / 0–0.01562 norm: 37 (0.4%)<br>11.625–12 raw / 0.98438–1 norm: 19 (0.2%) |
| `filter_fx_formant_x` | Linear | 9,618 | 0–1 | 0.49877 ± 0.04729 | 98.02454% | 0.29112% | normalized | 0.50061 / 0.50778 / 0.51494 | 0.5–0.51562 raw / 0.5–0.51562 norm: 9,434 (98.1%)<br>0–0.01562 raw / 0–0.01562 norm: 32 (0.3%)<br>0.98438–1 raw / 0.98438–1 norm: 17 (0.2%) |
| `filter_fx_formant_y` | Linear | 9,618 | 0–1 | 0.49906 ± 0.04821 | 98.02454% | 0.39509% | normalized | 0.50061 / 0.50778 / 0.51495 | 0.5–0.51562 raw / 0.5–0.51562 norm: 9,434 (98.1%)<br>0–0.01562 raw / 0–0.01562 norm: 38 (0.4%)<br>0.98438–1 raw / 0.98438–1 norm: 26 (0.3%) |
| `filter_fx_keytrack` | Linear | 9,618 | -1–1 | 0.06515 ± 0.27186 | 90.15388% | 90.15388% | normalized | 0.00125 / 0.01685 / 0.97617 | 0–0.03125 raw / 0.5–0.51562 norm: 8,671 (90.2%)<br>0.96875–1 raw / 0.98438–1 norm: 632 (6.6%)<br>-1–-0.96875 raw / 0–0.01562 norm: 57 (0.6%) |
| `filter_fx_mix` | Linear | 9,618 | 0–1 | 0.92632 ± 0.23235 | 88.19921% | 3.6806% | normalized | 0.25962 / 0.99119 / 0.99912 | 0.98438–1 raw / 0.98438–1 norm: 8,528 (88.7%)<br>0–0.01562 raw / 0–0.01562 norm: 379 (3.9%)<br>0.5–0.51562 raw / 0.5–0.51562 norm: 32 (0.3%) |
| `filter_fx_resonance` | Linear | 9,618 | 0–1 | 0.37277 ± 0.22286 | 57.88106% | 17.81036% | normalized | 0.00408 / 0.50381 / 0.56484 | 0.5–0.51562 raw / 0.5–0.51562 norm: 5,610 (58.3%)<br>0–0.01562 raw / 0–0.01562 norm: 1,843 (19.2%)<br>0.01562–0.03125 raw / 0.01562–0.03125 norm: 85 (0.9%) |
| `flanger_center` | Linear | 9,618 | 8–136 | 63.80473 ± 11.63766 | 87.22188% | 0% | normalized | 51.775 / 64.98172 / 68.83235 | 64–66 raw / 0.4375–0.45312 norm: 8,425 (87.6%)<br>8–10 raw / 0–0.01562 norm: 91 (0.9%)<br>134–136 raw / 0.98438–1 norm: 68 (0.7%) |
| `flanger_depth` | — | 2 | 0.0003139–0.037 | 0.01865 ± 0.01834 | —% | 0% | raw | 0.0003426 / 0.0006005 / 0.0008584 | 0.0003139–0.000887 raw: 1 (50.0%)<br>0.03642–0.037 raw: 1 (50.0%) |
| `flanger_dry_wet` | Linear | 9,618 | -0.00278–0.74048 | 0.44698 ± 0.13796 | 84.43543% | 3.3271% | raw | 0.05822 / 0.50118 / 0.50756 | 0.4966–0.50821 raw: 7,872 (81.8%)<br>-0.00278–0.00884 raw: 359 (3.7%)<br>0.48498–0.4966 raw: 269 (2.8%) |
| `flanger_feedback` | Linear | 9,618 | -1–1 | 0.47459 ± 0.17881 | 86.48368% | 1.12289% | normalized | 0.22905 / 0.51495 / 0.53117 | 0.5–0.53125 raw / 0.75–0.76562 norm: 8,338 (86.7%)<br>0–0.03125 raw / 0.5–0.51562 norm: 108 (1.1%)<br>0.375–0.40625 raw / 0.6875–0.70312 norm: 54 (0.6%) |
| `flanger_frequency` | Exponential | 9,618 | -5–2 | 1.9417 ± 0.54189 | 98.59638% | 0.05199% | normalized | 1.89464 / 1.94454 / 1.99444 | 1.89062–2 raw / 0.98438–1 norm: 9,485 (98.6%)<br>-3.90625–-3.79688 raw / 0.15625–0.17188 norm: 18 (0.2%)<br>0.25–0.35938 raw / 0.75–0.76562 norm: 11 (0.1%) |
| `flanger_mod_depth` | Linear | 9,618 | 0–1 | 0.49747 ± 0.1188 | 88.31358% | 1.65315% | normalized | 0.35198 / 0.50775 / 0.56655 | 0.5–0.51562 raw / 0.5–0.51562 norm: 8,506 (88.4%)<br>0–0.01562 raw / 0–0.01562 norm: 180 (1.9%)<br>0.98438–1 raw / 0.98438–1 norm: 177 (1.8%) |
| `flanger_phase_offset` | Linear | 9,618 | 0–1 | 0.33511 ± 0.10127 | 90.04991% | 2.23539% | normalized | 0.28258 / 0.33586 / 0.34366 | 0.32812–0.34375 raw / 0.32812–0.34375 norm: 8,673 (90.2%)<br>0–0.01562 raw / 0–0.01562 norm: 248 (2.6%)<br>0.98438–1 raw / 0.98438–1 norm: 64 (0.7%) |
| `lfo_1_delay_time` | Linear | 9,618 | 0–4 | 0.01036 ± 0.14244 | 98.33645% | 98.33645% | normalized | 0.00316 / 0.03159 / 0.06002 | 0–0.0625 raw / 0–0.01562 norm: 9,513 (98.9%)<br>0.0625–0.125 raw / 0.01562–0.03125 norm: 11 (0.1%)<br>0.125–0.1875 raw / 0.03125–0.04688 norm: 11 (0.1%) |
| `lfo_1_fade_time` | Linear | 9,618 | 0–8 | 0.00503 ± 0.15248 | 99.75047% | 99.75047% | normalized | 0.00626 / 0.06264 / 0.11902 | 0–0.125 raw / 0–0.01562 norm: 9,595 (99.8%)<br>0.125–0.25 raw / 0.01562–0.03125 norm: 3 (0.0%)<br>0.375–0.5 raw / 0.04688–0.0625 norm: 3 (0.0%) |
| `lfo_1_frequency` | Exponential | 9,618 | -7–9 | 0.95387 ± 1.23175 | 86.73321% | 0.13516% | normalized | -0.70509 / 1.12287 / 1.67341 | 1–1.25 raw / 0.5–0.51562 norm: 8,378 (87.1%)<br>1.5–1.75 raw / 0.53125–0.54688 norm: 55 (0.6%)<br>-3.5–-3.25 raw / 0.21875–0.23438 norm: 53 (0.6%) |
| `lfo_1_keytrack_tune` | Linear | 9,618 | -1–0.26 | -0.0001943 ± 0.01593 | 99.84404% | 99.84404% | normalized | 0.00154 / 0.01562 / 0.0297 | 0–0.03125 raw / 0.5–0.51562 norm: 9,606 (99.9%)<br>-0.03125–0 raw / 0.48438–0.5 norm: 4 (0.0%)<br>-1–-0.96875 raw / 0–0.01562 norm: 2 (0.0%) |
| `lfo_1_phase` | Linear | 9,618 | 0–1 | 0.00997 ± 0.07822 | 96.71449% | 96.71449% | normalized | 0.0007997 / 0.008 / 0.01519 | 0–0.01562 raw / 0–0.01562 norm: 9,395 (97.7%)<br>0.25–0.26562 raw / 0.25–0.26562 norm: 28 (0.3%)<br>0.48438–0.5 raw / 0.48438–0.5 norm: 22 (0.2%) |
| `lfo_1_smooth_time` | Exponential | 9,618 | -10–4 | -7.82039 ± 1.70501 | 69.70264% | 0.03119% | normalized | -9.94793 / -7.51329 / -6.63464 | -7.59375–-7.375 raw / 0.17188–0.1875 norm: 6,738 (70.1%)<br>-10–-9.78125 raw / 0–0.01562 norm: 2,020 (21.0%)<br>3.78125–4 raw / 0.98438–1 norm: 51 (0.5%) |
| `lfo_1_stereo` | Linear | 9,618 | -0.5–0.5 | 0.00676 ± 0.06114 | 96.64171% | 96.64171% | normalized | 0.0006987 / 0.00797 / 0.01524 | 0–0.01562 raw / 0.5–0.51562 norm: 9,299 (96.7%)<br>0.48438–0.5 raw / 0.98438–1 norm: 99 (1.0%)<br>0.04688–0.0625 raw / 0.54688–0.5625 norm: 18 (0.2%) |
| `lfo_2_delay_time` | Linear | 9,618 | 0–4 | 0.00537 ± 0.09403 | 99.14743% | 99.14743% | normalized | 0.00314 / 0.03144 / 0.05974 | 0–0.0625 raw / 0–0.01562 norm: 9,559 (99.4%)<br>0.125–0.1875 raw / 0.03125–0.04688 norm: 8 (0.1%)<br>0.375–0.4375 raw / 0.09375–0.10938 norm: 6 (0.1%) |
| `lfo_2_fade_time` | Linear | 9,618 | 0–4.08 | 0.00312 ± 0.08388 | 99.79206% | 99.79206% | normalized | 0.00626 / 0.06262 / 0.11897 | 0–0.125 raw / 0–0.01562 norm: 9,599 (99.8%)<br>0.875–1 raw / 0.10938–0.125 norm: 10 (0.1%)<br>2–2.125 raw / 0.25–0.26562 norm: 3 (0.0%) |
| `lfo_2_frequency` | Exponential | 9,618 | -7–9 | 0.93049 ± 0.93404 | 92.13974% | 0.09357% | normalized | 1.00059 / 1.12242 / 1.24426 | 1–1.25 raw / 0.5–0.51562 norm: 8,880 (92.3%)<br>-3.75–-3.5 raw / 0.20312–0.21875 norm: 40 (0.4%)<br>0.25–0.5 raw / 0.45312–0.46875 norm: 33 (0.3%) |
| `lfo_2_keytrack_tune` | Linear | 9,618 | -0.03631–1 | 0.0002802 ± 0.0162 | 99.94801% | 99.94801% | normalized | 0.00155 / 0.01562 / 0.02969 | 0–0.03125 raw / 0.5–0.51562 norm: 9,612 (99.9%)<br>-0.03125–0 raw / 0.48438–0.5 norm: 2 (0.0%)<br>-0.0625–-0.03125 raw / 0.46875–0.48438 norm: 1 (0.0%) |
| `lfo_2_phase` | Linear | 9,618 | 0–1 | 0.00504 ± 0.05449 | 98.34685% | 98.34685% | normalized | 0.0007911 / 0.00791 / 0.01503 | 0–0.01562 raw / 0–0.01562 norm: 9,497 (98.7%)<br>0.25–0.26562 raw / 0.25–0.26562 norm: 15 (0.2%)<br>0.5–0.51562 raw / 0.5–0.51562 norm: 14 (0.1%) |
| `lfo_2_smooth_time` | Exponential | 9,618 | -10–4 | -7.73682 ± 1.34366 | 80.66126% | 0.02079% | normalized | -9.92372 / -7.50148 / -7.37975 | -7.59375–-7.375 raw / 0.17188–0.1875 norm: 7,777 (80.9%)<br>-10–-9.78125 raw / 0–0.01562 norm: 1,379 (14.3%)<br>-9.34375–-9.125 raw / 0.04688–0.0625 norm: 22 (0.2%) |
| `lfo_2_stereo` | Linear | 9,618 | -0.5–0.5 | 0.00369 ± 0.04857 | 98.05573% | 98.05573% | normalized | 0.0007287 / 0.0079 / 0.01507 | 0–0.01562 raw / 0.5–0.51562 norm: 9,431 (98.1%)<br>0.48438–0.5 raw / 0.98438–1 norm: 58 (0.6%)<br>-0.5–-0.48438 raw / 0–0.01562 norm: 10 (0.1%) |
| `lfo_3_delay_time` | Linear | 9,618 | 0–4 | 0.00512 ± 0.1208 | 99.55292% | 99.55292% | normalized | 0.00314 / 0.03135 / 0.05957 | 0–0.0625 raw / 0–0.01562 norm: 9,585 (99.7%)<br>3.9375–4 raw / 0.98438–1 norm: 7 (0.1%)<br>0.0625–0.125 raw / 0.01562–0.03125 norm: 3 (0.0%) |
| `lfo_3_fade_time` | Linear | 9,618 | 0–4.98899 | 0.00228 ± 0.08745 | 99.87523% | 99.87523% | normalized | 0.00626 / 0.06257 / 0.11887 | 0–0.125 raw / 0–0.01562 norm: 9,607 (99.9%)<br>0.125–0.25 raw / 0.01562–0.03125 norm: 2 (0.0%)<br>0.375–0.5 raw / 0.04688–0.0625 norm: 1 (0.0%) |
| `lfo_3_frequency` | Exponential | 9,618 | -7–9 | 0.9047 ± 0.82442 | 94.72863% | 0.02079% | normalized | 1.00326 / 1.122 / 1.24073 | 1–1.25 raw / 0.5–0.51562 norm: 9,112 (94.7%)<br>-3.25–-3 raw / 0.23438–0.25 norm: 50 (0.5%)<br>-3.75–-3.5 raw / 0.20312–0.21875 norm: 45 (0.5%) |
| `lfo_3_keytrack_tune` | Linear | 9,618 | 0–0.98294 | 0.0002532 ± 0.01485 | 99.95841% | 99.95841% | normalized | 0.00156 / 0.01563 / 0.02969 | 0–0.03125 raw / 0.5–0.51562 norm: 9,615 (100.0%)<br>0.46875–0.5 raw / 0.73438–0.75 norm: 1 (0.0%)<br>0.9375–0.96875 raw / 0.96875–0.98438 norm: 1 (0.0%) |
| `lfo_3_phase` | Linear | 9,618 | 0–0.95735 | 0.00236 ± 0.03543 | 99.2722% | 99.2722% | normalized | 0.0007857 / 0.00786 / 0.01493 | 0–0.01562 raw / 0–0.01562 norm: 9,563 (99.4%)<br>0.5–0.51562 raw / 0.5–0.51562 norm: 14 (0.1%)<br>0.48438–0.5 raw / 0.48438–0.5 norm: 8 (0.1%) |
| `lfo_3_smooth_time` | Exponential | 9,618 | -10–4 | -7.69694 ± 1.05909 | 86.57725% | 0.02079% | normalized | -9.89471 / -7.49681 / -7.38331 | -7.59375–-7.375 raw / 0.17188–0.1875 norm: 8,341 (86.7%)<br>-10–-9.78125 raw / 0–0.01562 norm: 999 (10.4%)<br>-8.25–-8.03125 raw / 0.125–0.14062 norm: 19 (0.2%) |
| `lfo_3_stereo` | Linear | 9,618 | -0.5–0.5 | 0.00347 ± 0.0457 | 98.55479% | 98.55479% | normalized | 0.0007564 / 0.00789 / 0.01502 | 0–0.01562 raw / 0.5–0.51562 norm: 9,479 (98.6%)<br>0.48438–0.5 raw / 0.98438–1 norm: 55 (0.6%)<br>-0.5–-0.48438 raw / 0–0.01562 norm: 10 (0.1%) |
| `lfo_4_delay_time` | Linear | 9,618 | 0–4 | 0.00308 ± 0.07279 | 99.55292% | 99.55292% | normalized | 0.00314 / 0.03136 / 0.05959 | 0–0.0625 raw / 0–0.01562 norm: 9,583 (99.6%)<br>0.0625–0.125 raw / 0.01562–0.03125 norm: 7 (0.1%)<br>0.125–0.1875 raw / 0.03125–0.04688 norm: 4 (0.0%) |
| `lfo_4_fade_time` | Linear | 9,618 | 0–8 | 0.00127 ± 0.0893 | 99.95841% | 99.95841% | normalized | 0.00625 / 0.06251 / 0.11877 | 0–0.125 raw / 0–0.01562 norm: 9,615 (100.0%)<br>0.625–0.75 raw / 0.07812–0.09375 norm: 1 (0.0%)<br>3.375–3.5 raw / 0.42188–0.4375 norm: 1 (0.0%) |
| `lfo_4_frequency` | Exponential | 9,618 | -7–9 | 0.91776 ± 0.67205 | 96.56893% | 0.02079% | normalized | 1.00621 / 1.1227 / 1.23918 | 1–1.25 raw / 0.5–0.51562 norm: 9,288 (96.6%)<br>-3.75–-3.5 raw / 0.20312–0.21875 norm: 42 (0.4%)<br>-4–-3.75 raw / 0.1875–0.20312 norm: 33 (0.3%) |
| `lfo_4_keytrack_tune` | Linear | 9,618 | -0.02651–1 | 0.000195 ± 0.01445 | 99.89603% | 99.89603% | normalized | 0.00154 / 0.01562 / 0.02969 | 0–0.03125 raw / 0.5–0.51562 norm: 9,608 (99.9%)<br>-0.03125–0 raw / 0.48438–0.5 norm: 7 (0.1%)<br>0.96875–1 raw / 0.98438–1 norm: 2 (0.0%) |
| `lfo_4_phase` | Linear | 9,618 | 0–1 | 0.00232 ± 0.0378 | 99.24101% | 99.24101% | normalized | 0.0007854 / 0.00785 / 0.01492 | 0–0.01562 raw / 0–0.01562 norm: 9,566 (99.5%)<br>0.01562–0.03125 raw / 0.01562–0.03125 norm: 7 (0.1%)<br>0.75–0.76562 raw / 0.75–0.76562 norm: 7 (0.1%) |
| `lfo_4_smooth_time` | Exponential | 9,618 | -10–4 | -7.6476 ± 0.95774 | 90.30984% | 0% | normalized | -9.85994 / -7.49366 / -7.38476 | -7.59375–-7.375 raw / 0.17188–0.1875 norm: 8,693 (90.4%)<br>-10–-9.78125 raw / 0–0.01562 norm: 751 (7.8%)<br>-7.8125–-7.59375 raw / 0.15625–0.17188 norm: 24 (0.2%) |
| `lfo_4_stereo` | Linear | 9,618 | -0.5–0.5 | 0.00216 ± 0.03328 | 99.23061% | 99.23061% | normalized | 0.0007692 / 0.00785 / 0.01494 | 0–0.01562 raw / 0.5–0.51562 norm: 9,544 (99.2%)<br>0.48438–0.5 raw / 0.98438–1 norm: 32 (0.3%)<br>-0.14062–-0.125 raw / 0.35938–0.375 norm: 5 (0.1%) |
| `lfo_5_delay_time` | Linear | 9,618 | 0–3.5 | 0.0014 ± 0.06248 | 99.91682% | 99.91682% | normalized | 0.00313 / 0.03127 / 0.05941 | 0–0.0625 raw / 0–0.01562 norm: 9,611 (99.9%)<br>3.3125–3.375 raw / 0.82812–0.84375 norm: 2 (0.0%)<br>0.25–0.3125 raw / 0.0625–0.07812 norm: 1 (0.0%) |
| `lfo_5_fade_time` | Linear | 9,618 | 0–3.26849 | 0.0004911 ± 0.03367 | 99.88563% | 99.88563% | normalized | 0.00626 / 0.06256 / 0.11886 | 0–0.125 raw / 0–0.01562 norm: 9,608 (99.9%)<br>0.125–0.25 raw / 0.01562–0.03125 norm: 9 (0.1%)<br>3.25–3.375 raw / 0.40625–0.42188 norm: 1 (0.0%) |
| `lfo_5_frequency` | Exponential | 9,618 | -7–9 | 0.95796 ± 0.54206 | 97.82699% | 0.0104% | normalized | 1.00889 / 1.12378 / 1.23867 | 1–1.25 raw / 0.5–0.51562 norm: 9,417 (97.9%)<br>-3–-2.75 raw / 0.25–0.26562 norm: 18 (0.2%)<br>-3.5–-3.25 raw / 0.21875–0.23438 norm: 17 (0.2%) |
| `lfo_5_keytrack_tune` | Linear | 9,618 | -0.01044–0 | -1.085e-06 ± 0.0001064 | 99.9896% | 99.9896% | normalized | 0.00156 / 0.01562 / 0.02968 | 0–0.03125 raw / 0.5–0.51562 norm: 9,617 (100.0%)<br>-0.03125–0 raw / 0.48438–0.5 norm: 1 (0.0%) |
| `lfo_5_phase` | Linear | 9,618 | 0–0.87411 | 0.0006773 ± 0.01901 | 99.76087% | 99.76087% | normalized | 0.0007826 / 0.00783 / 0.01487 | 0–0.01562 raw / 0–0.01562 norm: 9,600 (99.8%)<br>0.01562–0.03125 raw / 0.01562–0.03125 norm: 4 (0.0%)<br>0.5–0.51562 raw / 0.5–0.51562 norm: 3 (0.0%) |
| `lfo_5_smooth_time` | Exponential | 9,618 | -10–4 | -7.62678 ± 0.75087 | 92.81555% | 0% | normalized | -9.81927 / -7.49113 / -7.38516 | -7.59375–-7.375 raw / 0.17188–0.1875 norm: 8,933 (92.9%)<br>-10–-9.78125 raw / 0–0.01562 norm: 582 (6.1%)<br>-6.71875–-6.5 raw / 0.23438–0.25 norm: 18 (0.2%) |
| `lfo_5_stereo` | Linear | 9,618 | -0.5–0.5 | 0.0005024 ± 0.02009 | 99.6361% | 99.6361% | normalized | 0.0007661 / 0.00782 / 0.01488 | 0–0.01562 raw / 0.5–0.51562 norm: 9,583 (99.6%)<br>0.48438–0.5 raw / 0.98438–1 norm: 10 (0.1%)<br>0.04688–0.0625 raw / 0.54688–0.5625 norm: 3 (0.0%) |
| `lfo_6_delay_time` | Linear | 9,618 | 0–4 | 0.0011 ± 0.05351 | 99.91682% | 99.91682% | normalized | 0.00313 / 0.03127 / 0.05941 | 0–0.0625 raw / 0–0.01562 norm: 9,611 (99.9%)<br>2.1875–2.25 raw / 0.54688–0.5625 norm: 2 (0.0%)<br>0.0625–0.125 raw / 0.01562–0.03125 norm: 1 (0.0%) |
| `lfo_6_fade_time` | Linear | 9,618 | 0–2.80814 | 0.0008759 ± 0.04959 | 99.96881% | 99.96881% | normalized | 0.00625 / 0.06251 / 0.11877 | 0–0.125 raw / 0–0.01562 norm: 9,615 (100.0%)<br>2.75–2.875 raw / 0.34375–0.35938 norm: 3 (0.0%) |
| `lfo_6_frequency` | Exponential | 9,618 | -5.695–5.83389 | 0.98451 ± 0.35142 | 98.73155% | 0.0104% | normalized | 1.01063 / 1.12455 / 1.23847 | 1–1.25 raw / 0.5–0.51562 norm: 9,497 (98.7%)<br>0–0.25 raw / 0.4375–0.45312 norm: 11 (0.1%)<br>-3.75–-3.5 raw / 0.20312–0.21875 norm: 10 (0.1%) |
| `lfo_6_keytrack_tune` | Linear | 9,618 | 0–0.04604 | 1.382e-05 ± 0.0007837 | 99.96881% | 99.96881% | normalized | 0.00156 / 0.01563 / 0.02969 | 0–0.03125 raw / 0.5–0.51562 norm: 9,615 (100.0%)<br>0.03125–0.0625 raw / 0.51562–0.53125 norm: 3 (0.0%) |
| `lfo_6_phase` | Linear | 9,618 | 0–0.61549 | 0.0001591 ± 0.00818 | 99.91682% | 99.91682% | normalized | 0.0007816 / 0.00782 / 0.01485 | 0–0.01562 raw / 0–0.01562 norm: 9,613 (99.9%)<br>0.25–0.26562 raw / 0.25–0.26562 norm: 2 (0.0%)<br>0.01562–0.03125 raw / 0.01562–0.03125 norm: 1 (0.0%) |
| `lfo_6_smooth_time` | Exponential | 9,618 | -10–4 | -7.60407 ± 0.57277 | 95.06134% | 0% | normalized | -7.59296 / -7.48949 / -7.38602 | -7.59375–-7.375 raw / 0.17188–0.1875 norm: 9,149 (95.1%)<br>-10–-9.78125 raw / 0–0.01562 norm: 422 (4.4%)<br>-9.125–-8.90625 raw / 0.0625–0.07812 norm: 8 (0.1%) |
| `lfo_6_stereo` | Linear | 9,618 | -0.5–0.5 | 0.0003594 ± 0.01565 | 99.83365% | 99.83365% | normalized | 0.0007776 / 0.00782 / 0.01486 | 0–0.01562 raw / 0.5–0.51562 norm: 9,602 (99.8%)<br>0.48438–0.5 raw / 0.98438–1 norm: 6 (0.1%)<br>-0.5–-0.48438 raw / 0–0.01562 norm: 2 (0.0%) |
| `lfo_7_delay_time` | Linear | 9,618 | 0–4 | 0.0007629 ± 0.04467 | 99.94801% | 99.94801% | normalized | 0.00313 / 0.03126 / 0.0594 | 0–0.0625 raw / 0–0.01562 norm: 9,613 (99.9%)<br>0.6875–0.75 raw / 0.17188–0.1875 norm: 2 (0.0%)<br>0.5625–0.625 raw / 0.14062–0.15625 norm: 1 (0.0%) |
| `lfo_7_fade_time` | Linear | 9,618 | 0–4.92575 | 0.00117 ± 0.07186 | 99.95841% | 99.95841% | normalized | 0.00625 / 0.06252 / 0.11879 | 0–0.125 raw / 0–0.01562 norm: 9,614 (100.0%)<br>4.875–5 raw / 0.60938–0.625 norm: 2 (0.0%)<br>0.375–0.5 raw / 0.04688–0.0625 norm: 1 (0.0%) |
| `lfo_7_frequency` | Exponential | 9,618 | -5.66682–6.1275 | 0.98459 ± 0.26672 | 99.37617% | 0% | normalized | 1.01137 / 1.12449 / 1.23761 | 1–1.25 raw / 0.5–0.51562 norm: 9,564 (99.4%)<br>-3.5–-3.25 raw / 0.21875–0.23438 norm: 11 (0.1%)<br>-3.75–-3.5 raw / 0.20312–0.21875 norm: 5 (0.1%) |
| `lfo_7_keytrack_tune` | Linear | 9,618 | 0–0.02457 | 2.555e-06 ± 0.0002505 | 99.9896% | 99.9896% | normalized | 0.00156 / 0.01562 / 0.02968 | 0–0.03125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `lfo_7_phase` | Linear | 9,618 | 0–0.50179 | 8.485e-05 ± 0.00573 | 99.94801% | 99.94801% | normalized | 0.0007816 / 0.00782 / 0.01485 | 0–0.01562 raw / 0–0.01562 norm: 9,613 (99.9%)<br>0.01562–0.03125 raw / 0.01562–0.03125 norm: 3 (0.0%)<br>0.25–0.26562 raw / 0.25–0.26562 norm: 1 (0.0%) |
| `lfo_7_smooth_time` | Exponential | 9,618 | -10–4 | -7.5773 ± 0.51284 | 96.40258% | 0% | normalized | -7.59022 / -7.48814 / -7.38606 | -7.59375–-7.375 raw / 0.17188–0.1875 norm: 9,274 (96.4%)<br>-10–-9.78125 raw / 0–0.01562 norm: 317 (3.3%)<br>-9.78125–-9.5625 raw / 0.01562–0.03125 norm: 5 (0.1%) |
| `lfo_7_stereo` | Linear | 9,618 | -0.08791–0.5 | 0.0001983 ± 0.01017 | 99.93762% | 99.93762% | normalized | 0.0007799 / 0.00781 / 0.01485 | 0–0.01562 raw / 0.5–0.51562 norm: 9,613 (99.9%)<br>0.48438–0.5 raw / 0.98438–1 norm: 4 (0.0%)<br>-0.09375–-0.07812 raw / 0.40625–0.42188 norm: 1 (0.0%) |
| `lfo_8_delay_time` | Linear | 9,618 | 0–4 | 0.0007942 ± 0.0486 | 99.95841% | 99.95841% | normalized | 0.00313 / 0.03126 / 0.05939 | 0–0.0625 raw / 0–0.01562 norm: 9,615 (100.0%)<br>1.5–1.5625 raw / 0.375–0.39062 norm: 1 (0.0%)<br>2.0625–2.125 raw / 0.51562–0.53125 norm: 1 (0.0%) |
| `lfo_8_fade_time` | Linear | 9,618 | 0–2.30681 | 0.0002398 ± 0.02352 | 99.9896% | 99.9896% | normalized | 0.00625 / 0.0625 / 0.11875 | 0–0.125 raw / 0–0.01562 norm: 9,617 (100.0%)<br>2.25–2.375 raw / 0.28125–0.29688 norm: 1 (0.0%) |
| `lfo_8_frequency` | Exponential | 9,618 | -7–6.5733 | 0.98319 ± 0.33603 | 99.48014% | 0.0104% | normalized | 1.01147 / 1.12454 / 1.23762 | 1–1.25 raw / 0.5–0.51562 norm: 9,568 (99.5%)<br>-7–-6.75 raw / 0–0.01562 norm: 5 (0.1%)<br>-5.5–-5.25 raw / 0.09375–0.10938 norm: 5 (0.1%) |
| `lfo_8_keytrack_tune` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.00156 / 0.01562 / 0.02968 | 0–0.03125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `lfo_8_phase` | Linear | 9,618 | 0–0.625 | 0.000141 ± 0.00826 | 99.95841% | 99.95841% | normalized | 0.0007815 / 0.00781 / 0.01485 | 0–0.01562 raw / 0–0.01562 norm: 9,614 (100.0%)<br>0.01562–0.03125 raw / 0.01562–0.03125 norm: 1 (0.0%)<br>0.25–0.26562 raw / 0.25–0.26562 norm: 1 (0.0%) |
| `lfo_8_smooth_time` | Exponential | 9,618 | -10–1.32804 | -7.55826 ± 0.41323 | 97.35912% | 0% | normalized | -7.58822 / -7.48717 / -7.38611 | -7.59375–-7.375 raw / 0.17188–0.1875 norm: 9,368 (97.4%)<br>-10–-9.78125 raw / 0–0.01562 norm: 228 (2.4%)<br>-9.5625–-9.34375 raw / 0.03125–0.04688 norm: 5 (0.1%) |
| `lfo_8_stereo` | Linear | 9,618 | 0–0.5 | 6.371e-05 ± 0.00521 | 99.96881% | 99.96881% | normalized | 0.0007813 / 0.00781 / 0.01485 | 0–0.01562 raw / 0.5–0.51562 norm: 9,616 (100.0%)<br>0.09375–0.10938 raw / 0.59375–0.60938 norm: 1 (0.0%)<br>0.48438–0.5 raw / 0.98438–1 norm: 1 (0.0%) |
| `macro_control_1` | Linear | 9,618 | 0–1 | 0.18022 ± 0.32654 | 67.78956% | 67.78956% | normalized | 0.00113 / 0.01127 / 0.9911 | 0–0.01562 raw / 0–0.01562 norm: 6,669 (69.3%)<br>0.98438–1 raw / 0.98438–1 norm: 846 (8.8%)<br>0.5–0.51562 raw / 0.5–0.51562 norm: 120 (1.2%) |
| `macro_control_2` | Linear | 9,618 | 0–1 | 0.12513 ± 0.28118 | 76.0969% | 76.0969% | normalized | 0.00101 / 0.01008 / 0.98674 | 0–0.01562 raw / 0–0.01562 norm: 7,450 (77.5%)<br>0.98438–1 raw / 0.98438–1 norm: 568 (5.9%)<br>0.5–0.51562 raw / 0.5–0.51562 norm: 87 (0.9%) |
| `macro_control_3` | Linear | 9,618 | 0–1 | 0.09717 ± 0.25357 | 81.96091% | 81.96091% | normalized | 0.0009442 / 0.00944 / 0.84014 | 0–0.01562 raw / 0–0.01562 norm: 7,957 (82.7%)<br>0.98438–1 raw / 0.98438–1 norm: 426 (4.4%)<br>0.5–0.51562 raw / 0.5–0.51562 norm: 63 (0.7%) |
| `macro_control_4` | Linear | 9,618 | 0–1 | 0.0761 ± 0.22665 | 85.24641% | 85.24641% | normalized | 0.0009066 / 0.00907 / 0.69325 | 0–0.01562 raw / 0–0.01562 norm: 8,287 (86.2%)<br>0.98438–1 raw / 0.98438–1 norm: 332 (3.5%)<br>0.5–0.51562 raw / 0.5–0.51562 norm: 51 (0.5%) |
| `mod_wheel` | Linear | 9,618 | 0–1 | 0.04158 ± 0.18096 | 92.83635% | 92.83635% | normalized | 0.0008394 / 0.00839 / 0.26602 | 0–0.01562 raw / 0–0.01562 norm: 8,951 (93.1%)<br>0.98438–1 raw / 0.98438–1 norm: 247 (2.6%)<br>0.01562–0.03125 raw / 0.01562–0.03125 norm: 41 (0.4%) |
| `modulation_10_amount` | Linear | 9,618 | -1–1 | 0.10147 ± 0.29759 | 64.45207% | 64.45207% | normalized | -0.1069 / 0.02113 / 0.82451 | 0–0.03125 raw / 0.5–0.51562 norm: 6,236 (64.8%)<br>0.96875–1 raw / 0.98438–1 norm: 409 (4.3%)<br>0.5–0.53125 raw / 0.75–0.76562 norm: 236 (2.5%) |
| `modulation_10_power` | Linear | 9,618 | -10–10 | -0.00163 ± 0.26002 | 99.79206% | 99.79206% | normalized | 0.0153 / 0.15618 / 0.29707 | 0–0.3125 raw / 0.5–0.51562 norm: 9,599 (99.8%)<br>-10–-9.6875 raw / 0–0.01562 norm: 2 (0.0%)<br>-2.8125–-2.5 raw / 0.35938–0.375 norm: 2 (0.0%) |
| `modulation_10_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_10_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_11_amount` | Linear | 9,618 | -1–1 | 0.08804 ± 0.28433 | 68.22624% | 68.22624% | normalized | -0.08833 / 0.02009 / 0.75015 | 0–0.03125 raw / 0.5–0.51562 norm: 6,594 (68.6%)<br>0.96875–1 raw / 0.98438–1 norm: 377 (3.9%)<br>0.5–0.53125 raw / 0.75–0.76562 norm: 180 (1.9%) |
| `modulation_11_power` | Linear | 9,618 | -8.0241–10 | 0.00283 ± 0.26533 | 99.84404% | 99.84404% | normalized | 0.01539 / 0.15622 / 0.29705 | 0–0.3125 raw / 0.5–0.51562 norm: 9,603 (99.8%)<br>9.6875–10 raw / 0.98438–1 norm: 5 (0.1%)<br>-4.0625–-3.75 raw / 0.29688–0.3125 norm: 2 (0.0%) |
| `modulation_11_ramp_down` | — | 287 | -10–4 | -9.95122 ± 0.82495 | —% | 0% | raw | -9.98906 / -9.89062 / -9.79219 | -10–-9.78125 raw: 286 (99.7%)<br>3.78125–4 raw: 1 (0.3%) |
| `modulation_11_ramp_up` | — | 287 | -10–-3.85686 | -9.9786 ± 0.36199 | —% | 0% | raw | -9.9952 / -9.95201 / -9.90881 | -10–-9.90401 raw: 286 (99.7%)<br>-3.95284–-3.85686 raw: 1 (0.3%) |
| `modulation_12_amount` | Linear | 9,618 | -1–1 | 0.07719 ± 0.26776 | 71.34539% | 71.34539% | normalized | -0.04823 / 0.01949 / 0.67085 | 0–0.03125 raw / 0.5–0.51562 norm: 6,911 (71.9%)<br>0.96875–1 raw / 0.98438–1 norm: 324 (3.4%)<br>0.5–0.53125 raw / 0.75–0.76562 norm: 154 (1.6%) |
| `modulation_12_power` | Linear | 9,618 | -10–9.88332 | 0.0001228 ± 0.18184 | 99.86484% | 99.86484% | normalized | 0.01545 / 0.15622 / 0.29699 | 0–0.3125 raw / 0.5–0.51562 norm: 9,607 (99.9%)<br>3.4375–3.75 raw / 0.67188–0.6875 norm: 2 (0.0%)<br>-10–-9.6875 raw / 0–0.01562 norm: 1 (0.0%) |
| `modulation_12_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_12_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_13_amount` | Linear | 9,618 | -1–1 | 0.06204 ± 0.25738 | 73.79913% | 73.79913% | normalized | -0.0799 / 0.01882 / 0.56268 | 0–0.03125 raw / 0.5–0.51562 norm: 7,121 (74.0%)<br>0.96875–1 raw / 0.98438–1 norm: 269 (2.8%)<br>0.5–0.53125 raw / 0.75–0.76562 norm: 163 (1.7%) |
| `modulation_13_power` | Linear | 9,618 | -10–10 | -0.000374 ± 0.23002 | 99.86484% | 99.86484% | normalized | 0.01548 / 0.15628 / 0.29708 | 0–0.3125 raw / 0.5–0.51562 norm: 9,605 (99.9%)<br>-10–-9.6875 raw / 0–0.01562 norm: 3 (0.0%)<br>3.125–3.4375 raw / 0.65625–0.67188 norm: 2 (0.0%) |
| `modulation_13_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_13_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_14_amount` | Linear | 9,618 | -1–1 | 0.06153 ± 0.23951 | 76.07611% | 76.07611% | normalized | 0.0002161 / 0.01861 / 0.52704 | 0–0.03125 raw / 0.5–0.51562 norm: 7,353 (76.5%)<br>0.96875–1 raw / 0.98438–1 norm: 242 (2.5%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 154 (1.6%) |
| `modulation_14_power` | Linear | 9,618 | -9.6053–4.55069 | -0.00534 ± 0.23285 | 99.86484% | 99.86484% | normalized | 0.01535 / 0.15614 / 0.29692 | 0–0.3125 raw / 0.5–0.51562 norm: 9,606 (99.9%)<br>-9.6875–-9.375 raw / 0.01562–0.03125 norm: 5 (0.1%)<br>-4.6875–-4.375 raw / 0.26562–0.28125 norm: 1 (0.0%) |
| `modulation_14_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_14_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_15_amount` | Linear | 9,618 | -1–1 | 0.05908 ± 0.2343 | 78.23872% | 78.23872% | normalized | 0.0002603 / 0.01819 / 0.53744 | 0–0.03125 raw / 0.5–0.51562 norm: 7,544 (78.4%)<br>0.96875–1 raw / 0.98438–1 norm: 241 (2.5%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 121 (1.3%) |
| `modulation_15_power` | Linear | 9,618 | -10–9.64995 | -0.0002418 ± 0.2002 | 99.85444% | 99.85444% | normalized | 0.01545 / 0.15625 / 0.29705 | 0–0.3125 raw / 0.5–0.51562 norm: 9,605 (99.9%)<br>-10–-9.6875 raw / 0–0.01562 norm: 2 (0.0%)<br>-0.9375–-0.625 raw / 0.45312–0.46875 norm: 2 (0.0%) |
| `modulation_15_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_15_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_16_amount` | Linear | 9,618 | -1–1 | 0.05198 ± 0.22089 | 80.1518% | 80.1518% | normalized | 0.0004758 / 0.01795 / 0.50812 | 0–0.03125 raw / 0.5–0.51562 norm: 7,740 (80.5%)<br>0.96875–1 raw / 0.98438–1 norm: 214 (2.2%)<br>0.5–0.53125 raw / 0.75–0.76562 norm: 116 (1.2%) |
| `modulation_16_power` | Linear | 9,618 | -10–10 | -0.00138 ± 0.16519 | 99.89603% | 99.89603% | normalized | 0.01544 / 0.1562 / 0.29696 | 0–0.3125 raw / 0.5–0.51562 norm: 9,608 (99.9%)<br>-3.4375–-3.125 raw / 0.32812–0.34375 norm: 2 (0.0%)<br>-10–-9.6875 raw / 0–0.01562 norm: 1 (0.0%) |
| `modulation_16_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_16_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_17_amount` | Linear | 9,618 | -1–1 | 0.04537 ± 0.20726 | 81.75296% | 81.75296% | normalized | 0.0005146 / 0.01767 / 0.45051 | 0–0.03125 raw / 0.5–0.51562 norm: 7,885 (82.0%)<br>0.96875–1 raw / 0.98438–1 norm: 196 (2.0%)<br>0.1875–0.21875 raw / 0.59375–0.60938 norm: 92 (1.0%) |
| `modulation_17_power` | Linear | 9,618 | -1.66223–10 | 0.00245 ± 0.14635 | 99.89603% | 99.89603% | normalized | 0.01561 / 0.15635 / 0.29709 | 0–0.3125 raw / 0.5–0.51562 norm: 9,609 (99.9%)<br>0.3125–0.625 raw / 0.51562–0.53125 norm: 3 (0.0%)<br>0.625–0.9375 raw / 0.53125–0.54688 norm: 2 (0.0%) |
| `modulation_17_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_17_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_18_amount` | Linear | 9,618 | -1–1 | 0.04145 ± 0.20293 | 83.16698% | 83.16698% | normalized | 0.0006774 / 0.01754 / 0.41519 | 0–0.03125 raw / 0.5–0.51562 norm: 8,020 (83.4%)<br>0.96875–1 raw / 0.98438–1 norm: 178 (1.9%)<br>0.1875–0.21875 raw / 0.59375–0.60938 norm: 95 (1.0%) |
| `modulation_18_power` | Linear | 9,618 | -10–1.4 | -0.00155 ± 0.13697 | 99.90643% | 99.90643% | normalized | 0.01551 / 0.15625 / 0.29699 | 0–0.3125 raw / 0.5–0.51562 norm: 9,609 (99.9%)<br>1.25–1.5625 raw / 0.5625–0.57812 norm: 3 (0.0%)<br>0.9375–1.25 raw / 0.54688–0.5625 norm: 2 (0.0%) |
| `modulation_18_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_18_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_19_amount` | Linear | 9,618 | -1–1 | 0.03876 ± 0.19692 | 84.88251% | 84.88251% | normalized | 0.0007012 / 0.01721 / 0.39956 | 0–0.03125 raw / 0.5–0.51562 norm: 8,194 (85.2%)<br>0.96875–1 raw / 0.98438–1 norm: 169 (1.8%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 87 (0.9%) |
| `modulation_19_power` | Linear | 9,618 | -10–10 | -0.00226 ± 0.24217 | 99.85444% | 99.85444% | normalized | 0.01545 / 0.15622 / 0.29699 | 0–0.3125 raw / 0.5–0.51562 norm: 9,607 (99.9%)<br>-10–-9.6875 raw / 0–0.01562 norm: 4 (0.0%)<br>-2.8125–-2.5 raw / 0.35938–0.375 norm: 2 (0.0%) |
| `modulation_19_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_19_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_1_amount` | Linear | 9,618 | -1–1 | 0.37116 ± 0.44187 | 13.25639% | 13.25639% | normalized | -0.30374 / 0.34168 / 0.99093 | 0.96875–1 raw / 0.98438–1 norm: 1,660 (17.3%)<br>0–0.03125 raw / 0.5–0.51562 norm: 1,395 (14.5%)<br>0.5–0.53125 raw / 0.75–0.76562 norm: 492 (5.1%) |
| `modulation_1_power` | Linear | 9,618 | -10–10 | 0.00327 ± 0.37833 | 99.05386% | 99.05386% | normalized | 0.01455 / 0.15636 / 0.29818 | 0–0.3125 raw / 0.5–0.51562 norm: 9,536 (99.1%)<br>2.5–2.8125 raw / 0.625–0.64062 norm: 9 (0.1%)<br>0.625–0.9375 raw / 0.53125–0.54688 norm: 7 (0.1%) |
| `modulation_1_ramp_down` | — | 287 | -10–-1.98779 | -9.97208 ± 0.47212 | —% | 0% | raw | -9.99374 / -9.9374 / -9.88107 | -10–-9.87481 raw: 286 (99.7%)<br>-2.11298–-1.98779 raw: 1 (0.3%) |
| `modulation_1_ramp_up` | — | 287 | -10–4 | -9.95122 ± 0.82495 | —% | 0% | raw | -9.98906 / -9.89062 / -9.79219 | -10–-9.78125 raw: 286 (99.7%)<br>3.78125–4 raw: 1 (0.3%) |
| `modulation_20_amount` | Linear | 9,618 | -1–1 | 0.03013 ± 0.18039 | 86.13017% | 86.13017% | normalized | 0.0007557 / 0.01704 / 0.30884 | 0–0.03125 raw / 0.5–0.51562 norm: 8,306 (86.4%)<br>0.96875–1 raw / 0.98438–1 norm: 126 (1.3%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 77 (0.8%) |
| `modulation_20_power` | Linear | 9,618 | -9.86843–10 | -0.0002489 ± 0.18316 | 99.93762% | 99.93762% | normalized | 0.01554 / 0.15623 / 0.29693 | 0–0.3125 raw / 0.5–0.51562 norm: 9,612 (99.9%)<br>-10–-9.6875 raw / 0–0.01562 norm: 2 (0.0%)<br>-0.3125–0 raw / 0.48438–0.5 norm: 1 (0.0%) |
| `modulation_20_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_20_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_21_amount` | Linear | 9,618 | -1–1 | 0.02861 ± 0.16692 | 87.09711% | 87.09711% | normalized | 0.0008214 / 0.01692 / 0.28134 | 0–0.03125 raw / 0.5–0.51562 norm: 8,402 (87.4%)<br>0.96875–1 raw / 0.98438–1 norm: 114 (1.2%)<br>0.1875–0.21875 raw / 0.59375–0.60938 norm: 80 (0.8%) |
| `modulation_21_power` | Linear | 9,618 | -3.31313–10 | 0.00101 ± 0.1126 | 99.93762% | 99.93762% | normalized | 0.01554 / 0.15623 / 0.29693 | 0–0.3125 raw / 0.5–0.51562 norm: 9,612 (99.9%)<br>-3.4375–-3.125 raw / 0.32812–0.34375 norm: 1 (0.0%)<br>-0.9375–-0.625 raw / 0.45312–0.46875 norm: 1 (0.0%) |
| `modulation_21_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_21_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_22_amount` | Linear | 9,618 | -1–1 | 0.02328 ± 0.16937 | 88.36556% | 88.36556% | normalized | 0.0007922 / 0.01667 / 0.26331 | 0–0.03125 raw / 0.5–0.51562 norm: 8,515 (88.5%)<br>0.96875–1 raw / 0.98438–1 norm: 98 (1.0%)<br>0.5–0.53125 raw / 0.75–0.76562 norm: 63 (0.7%) |
| `modulation_22_power` | Linear | 9,618 | -10–9.87674 | -0.00412 ± 0.23637 | 99.85444% | 99.85444% | normalized | 0.01535 / 0.15614 / 0.29692 | 0–0.3125 raw / 0.5–0.51562 norm: 9,606 (99.9%)<br>-10–-9.6875 raw / 0–0.01562 norm: 3 (0.0%)<br>-4.6875–-4.375 raw / 0.26562–0.28125 norm: 2 (0.0%) |
| `modulation_22_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_22_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_23_amount` | Linear | 9,618 | -1–1 | 0.02332 ± 0.15198 | 89.10376% | 89.10376% | normalized | 0.000939 / 0.0167 / 0.24442 | 0–0.03125 raw / 0.5–0.51562 norm: 8,581 (89.2%)<br>0.96875–1 raw / 0.98438–1 norm: 76 (0.8%)<br>0.09375–0.125 raw / 0.54688–0.5625 norm: 64 (0.7%) |
| `modulation_23_power` | Linear | 9,618 | -10–10 | -0.000906 ± 0.16013 | 99.92722% | 99.92722% | normalized | 0.01547 / 0.15618 / 0.2969 | 0–0.3125 raw / 0.5–0.51562 norm: 9,611 (99.9%)<br>-10–-9.6875 raw / 0–0.01562 norm: 1 (0.0%)<br>-5–-4.6875 raw / 0.25–0.26562 norm: 1 (0.0%) |
| `modulation_23_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_23_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_24_amount` | Linear | 9,618 | -1–1 | 0.02438 ± 0.14688 | 89.89395% | 89.89395% | normalized | 0.00112 / 0.01674 / 0.23984 | 0–0.03125 raw / 0.5–0.51562 norm: 8,656 (90.0%)<br>0.96875–1 raw / 0.98438–1 norm: 80 (0.8%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 66 (0.7%) |
| `modulation_24_power` | Linear | 9,618 | -10–3.42159 | -0.0009463 ± 0.11079 | 99.96881% | 99.96881% | normalized | 0.01556 / 0.15622 / 0.29687 | 0–0.3125 raw / 0.5–0.51562 norm: 9,615 (100.0%)<br>-10–-9.6875 raw / 0–0.01562 norm: 1 (0.0%)<br>-2.8125–-2.5 raw / 0.35938–0.375 norm: 1 (0.0%) |
| `modulation_24_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_24_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_25_amount` | Linear | 9,618 | -1–1 | 0.02264 ± 0.15241 | 90.58016% | 90.58016% | normalized | 0.00104 / 0.01655 / 0.20747 | 0–0.03125 raw / 0.5–0.51562 norm: 8,719 (90.7%)<br>0.96875–1 raw / 0.98438–1 norm: 89 (0.9%)<br>0.1875–0.21875 raw / 0.59375–0.60938 norm: 55 (0.6%) |
| `modulation_25_power` | Linear | 9,618 | -10–9.87674 | 0.0009111 ± 0.1701 | 99.94801% | 99.94801% | normalized | 0.01553 / 0.15622 / 0.2969 | 0–0.3125 raw / 0.5–0.51562 norm: 9,613 (99.9%)<br>-0.3125–0 raw / 0.48438–0.5 norm: 2 (0.0%)<br>-10–-9.6875 raw / 0–0.01562 norm: 1 (0.0%) |
| `modulation_25_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_25_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_26_amount` | Linear | 9,618 | -1–1 | 0.02098 ± 0.14106 | 91.11042% | 91.11042% | normalized | 0.00117 / 0.01657 / 0.18162 | 0–0.03125 raw / 0.5–0.51562 norm: 8,782 (91.3%)<br>0.96875–1 raw / 0.98438–1 norm: 81 (0.8%)<br>0.09375–0.125 raw / 0.54688–0.5625 norm: 58 (0.6%) |
| `modulation_26_power` | Linear | 9,618 | -2.31919–10 | 0.00181 ± 0.14614 | 99.95841% | 99.95841% | normalized | 0.01556 / 0.15623 / 0.2969 | 0–0.3125 raw / 0.5–0.51562 norm: 9,614 (100.0%)<br>9.6875–10 raw / 0.98438–1 norm: 2 (0.0%)<br>-2.5–-2.1875 raw / 0.375–0.39062 norm: 1 (0.0%) |
| `modulation_26_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_26_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_27_amount` | Linear | 9,618 | -1–1 | 0.01557 ± 0.12817 | 91.70306% | 91.70306% | normalized | 0.00111 / 0.01643 / 0.14919 | 0–0.03125 raw / 0.5–0.51562 norm: 8,830 (91.8%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 58 (0.6%)<br>0.15625–0.1875 raw / 0.57812–0.59375 norm: 55 (0.6%) |
| `modulation_27_power` | Linear | 9,618 | -10–10 | -0.0001391 ± 0.18739 | 99.94801% | 99.94801% | normalized | 0.01557 / 0.15625 / 0.29693 | 0–0.3125 raw / 0.5–0.51562 norm: 9,613 (99.9%)<br>-10–-9.6875 raw / 0–0.01562 norm: 2 (0.0%)<br>3.75–4.0625 raw / 0.6875–0.70312 norm: 1 (0.0%) |
| `modulation_27_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_27_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_28_amount` | Linear | 9,618 | -1–1 | 0.01513 ± 0.11954 | 92.62841% | 92.62841% | normalized | 0.00122 / 0.01639 / 0.12266 | 0–0.03125 raw / 0.5–0.51562 norm: 8,918 (92.7%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 47 (0.5%)<br>0.15625–0.1875 raw / 0.57812–0.59375 norm: 45 (0.5%) |
| `modulation_28_power` | Linear | 9,618 | -4.20513–10 | 0.00159 ± 0.15049 | 99.94801% | 99.94801% | normalized | 0.01553 / 0.15622 / 0.2969 | 0–0.3125 raw / 0.5–0.51562 norm: 9,613 (99.9%)<br>9.6875–10 raw / 0.98438–1 norm: 2 (0.0%)<br>-4.375–-4.0625 raw / 0.28125–0.29688 norm: 1 (0.0%) |
| `modulation_28_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_28_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_29_amount` | Linear | 9,618 | -1–1 | 0.01607 ± 0.12084 | 92.97151% | 92.97151% | normalized | 0.00124 / 0.01635 / 0.10454 | 0–0.03125 raw / 0.5–0.51562 norm: 8,948 (93.0%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 53 (0.6%)<br>0.96875–1 raw / 0.98438–1 norm: 48 (0.5%) |
| `modulation_29_power` | Linear | 9,618 | -4.62564–0 | -0.00109 ± 0.06372 | 99.96881% | 99.96881% | normalized | 0.01553 / 0.15618 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,615 (100.0%)<br>-4.6875–-4.375 raw / 0.26562–0.28125 norm: 1 (0.0%)<br>-3.75–-3.4375 raw / 0.3125–0.32812 norm: 1 (0.0%) |
| `modulation_29_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_29_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_2_amount` | Linear | 9,618 | -1–1 | 0.30027 ± 0.44153 | 17.9975% | 17.9975% | normalized | -0.40269 / 0.25994 / 0.98853 | 0–0.03125 raw / 0.5–0.51562 norm: 1,852 (19.3%)<br>0.96875–1 raw / 0.98438–1 norm: 1,313 (13.7%)<br>0.5–0.53125 raw / 0.75–0.76562 norm: 464 (4.8%) |
| `modulation_2_power` | Linear | 9,618 | -9.73686–10 | -0.0001854 ± 0.3526 | 99.24101% | 99.24101% | normalized | 0.0143 / 0.15594 / 0.29758 | 0–0.3125 raw / 0.5–0.51562 norm: 9,548 (99.3%)<br>-1.5625–-1.25 raw / 0.42188–0.4375 norm: 6 (0.1%)<br>-1.25–-0.9375 raw / 0.4375–0.45312 norm: 6 (0.1%) |
| `modulation_2_ramp_down` | — | 287 | -10–-3.49008 | -9.97732 ± 0.3836 | —% | 0% | raw | -9.99491 / -9.94914 / -9.90337 | -10–-9.89828 raw: 286 (99.7%)<br>-3.5918–-3.49008 raw: 1 (0.3%) |
| `modulation_2_ramp_up` | — | 287 | -10–0.4112 | -9.96372 ± 0.61348 | —% | 0% | raw | -9.99187 / -9.91866 / -9.84546 | -10–-9.83733 raw: 286 (99.7%)<br>0.24852–0.4112 raw: 1 (0.3%) |
| `modulation_30_amount` | Linear | 9,618 | -1–1 | 0.01119 ± 0.11238 | 93.44978% | 93.44978% | normalized | 0.00117 / 0.01621 / 0.03124 | 0–0.03125 raw / 0.5–0.51562 norm: 8,997 (93.5%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 47 (0.5%)<br>0.1875–0.21875 raw / 0.59375–0.60938 norm: 41 (0.4%) |
| `modulation_30_power` | Linear | 9,618 | 0–1.64697 | 0.0001712 ± 0.01679 | 99.9896% | 99.9896% | normalized | 0.01562 / 0.15625 / 0.29688 | 0–0.3125 raw / 0.5–0.51562 norm: 9,617 (100.0%)<br>1.5625–1.875 raw / 0.57812–0.59375 norm: 1 (0.0%) |
| `modulation_30_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_30_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_31_amount` | Linear | 9,618 | -1–1 | 0.01106 ± 0.11487 | 94.05282% | 94.05282% | normalized | 0.00119 / 0.01614 / 0.03108 | 0–0.03125 raw / 0.5–0.51562 norm: 9,051 (94.1%)<br>0.96875–1 raw / 0.98438–1 norm: 46 (0.5%)<br>0.1875–0.21875 raw / 0.59375–0.60938 norm: 33 (0.3%) |
| `modulation_31_power` | Linear | 9,618 | -2.48485–1.37167 | -0.0002606 ± 0.03224 | 99.96881% | 99.96881% | normalized | 0.01556 / 0.15622 / 0.29687 | 0–0.3125 raw / 0.5–0.51562 norm: 9,615 (100.0%)<br>-2.5–-2.1875 raw / 0.375–0.39062 norm: 1 (0.0%)<br>-1.5625–-1.25 raw / 0.42188–0.4375 norm: 1 (0.0%) |
| `modulation_31_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_31_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_32_amount` | Linear | 9,618 | -1–1 | 0.01195 ± 0.11135 | 94.23997% | 94.23997% | normalized | 0.00129 / 0.0162 / 0.03112 | 0–0.03125 raw / 0.5–0.51562 norm: 9,067 (94.3%)<br>0.15625–0.1875 raw / 0.57812–0.59375 norm: 46 (0.5%)<br>0.09375–0.125 raw / 0.54688–0.5625 norm: 41 (0.4%) |
| `modulation_32_power` | Linear | 9,618 | -3.42912–0 | -0.0003565 ± 0.03496 | 99.9896% | 99.9896% | normalized | 0.01559 / 0.15622 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,617 (100.0%)<br>-3.4375–-3.125 raw / 0.32812–0.34375 norm: 1 (0.0%) |
| `modulation_32_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_32_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_33_amount` | Linear | 9,618 | -1–1 | 0.0104 ± 0.11026 | 94.60387% | 94.60387% | normalized | 0.00124 / 0.01609 / 0.03095 | 0–0.03125 raw / 0.5–0.51562 norm: 9,102 (94.6%)<br>0.96875–1 raw / 0.98438–1 norm: 46 (0.5%)<br>0.09375–0.125 raw / 0.54688–0.5625 norm: 31 (0.3%) |
| `modulation_33_power` | Linear | 9,618 | 0–0.76014 | 7.903e-05 ± 0.00775 | 99.9896% | 99.9896% | normalized | 0.01562 / 0.15625 / 0.29688 | 0–0.3125 raw / 0.5–0.51562 norm: 9,617 (100.0%)<br>0.625–0.9375 raw / 0.53125–0.54688 norm: 1 (0.0%) |
| `modulation_33_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_33_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_34_amount` | Linear | 9,618 | -1–1 | 0.01004 ± 0.10355 | 94.89499% | 94.89499% | normalized | 0.00129 / 0.0161 / 0.0309 | 0–0.03125 raw / 0.5–0.51562 norm: 9,133 (95.0%)<br>0.96875–1 raw / 0.98438–1 norm: 39 (0.4%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 35 (0.4%) |
| `modulation_34_power` | Linear | 9,618 | -4.80404–9.83434 | 0.0003914 ± 0.11235 | 99.95841% | 99.95841% | normalized | 0.01553 / 0.1562 / 0.29687 | 0–0.3125 raw / 0.5–0.51562 norm: 9,614 (100.0%)<br>-5–-4.6875 raw / 0.25–0.26562 norm: 1 (0.0%)<br>-1.5625–-1.25 raw / 0.42188–0.4375 norm: 1 (0.0%) |
| `modulation_34_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_34_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_35_amount` | Linear | 9,618 | -1–1 | 0.00983 ± 0.09439 | 95.40445% | 95.40445% | normalized | 0.00132 / 0.01606 / 0.03079 | 0–0.03125 raw / 0.5–0.51562 norm: 9,179 (95.4%)<br>0.125–0.15625 raw / 0.5625–0.57812 norm: 36 (0.4%)<br>0.09375–0.125 raw / 0.54688–0.5625 norm: 34 (0.4%) |
| `modulation_35_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_35_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_35_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_36_amount` | Linear | 9,618 | -1–1 | 0.00753 ± 0.09109 | 95.82034% | 95.82034% | normalized | 0.00129 / 0.01595 / 0.03062 | 0–0.03125 raw / 0.5–0.51562 norm: 9,221 (95.9%)<br>0.96875–1 raw / 0.98438–1 norm: 31 (0.3%)<br>0.125–0.15625 raw / 0.5625–0.57812 norm: 30 (0.3%) |
| `modulation_36_power` | Linear | 9,618 | 0–10 | 0.00104 ± 0.10196 | 99.9896% | 99.9896% | normalized | 0.01562 / 0.15625 / 0.29688 | 0–0.3125 raw / 0.5–0.51562 norm: 9,617 (100.0%)<br>9.6875–10 raw / 0.98438–1 norm: 1 (0.0%) |
| `modulation_36_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_36_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_37_amount` | Linear | 9,618 | -1–1 | 0.00841 ± 0.09328 | 96.08027% | 96.08027% | normalized | 0.00133 / 0.01596 / 0.03059 | 0–0.03125 raw / 0.5–0.51562 norm: 9,244 (96.1%)<br>0.96875–1 raw / 0.98438–1 norm: 35 (0.4%)<br>0.15625–0.1875 raw / 0.57812–0.59375 norm: 31 (0.3%) |
| `modulation_37_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_37_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_37_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_38_amount` | Linear | 9,618 | -1–1 | 0.00833 ± 0.08099 | 96.26742% | 96.26742% | normalized | 0.00141 / 0.016 / 0.0306 | 0–0.03125 raw / 0.5–0.51562 norm: 9,264 (96.3%)<br>0.15625–0.1875 raw / 0.57812–0.59375 norm: 32 (0.3%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 28 (0.3%) |
| `modulation_38_power` | Linear | 9,618 | -3.47879–4.18077 | -0.0001675 ± 0.06026 | 99.96881% | 99.96881% | normalized | 0.01556 / 0.15622 / 0.29687 | 0–0.3125 raw / 0.5–0.51562 norm: 9,615 (100.0%)<br>-3.75–-3.4375 raw / 0.3125–0.32812 norm: 1 (0.0%)<br>-2.5–-2.1875 raw / 0.375–0.39062 norm: 1 (0.0%) |
| `modulation_38_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_38_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_39_amount` | Linear | 9,618 | -1–1 | 0.00687 ± 0.09288 | 96.6937% | 96.6937% | normalized | 0.00137 / 0.01591 / 0.03044 | 0–0.03125 raw / 0.5–0.51562 norm: 9,304 (96.7%)<br>0.96875–1 raw / 0.98438–1 norm: 38 (0.4%)<br>0.21875–0.25 raw / 0.60938–0.625 norm: 25 (0.3%) |
| `modulation_39_power` | Linear | 9,618 | 0–1.32525 | 0.0001378 ± 0.01351 | 99.9896% | 99.9896% | normalized | 0.01562 / 0.15625 / 0.29688 | 0–0.3125 raw / 0.5–0.51562 norm: 9,617 (100.0%)<br>1.25–1.5625 raw / 0.5625–0.57812 norm: 1 (0.0%) |
| `modulation_39_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_39_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_3_amount` | Linear | 9,618 | -1–1 | 0.25747 ± 0.42401 | 25.20274% | 25.20274% | normalized | -0.38862 / 0.18857 / 0.98641 | 0–0.03125 raw / 0.5–0.51562 norm: 2,518 (26.2%)<br>0.96875–1 raw / 0.98438–1 norm: 1,108 (11.5%)<br>0.5–0.53125 raw / 0.75–0.76562 norm: 423 (4.4%) |
| `modulation_3_power` | Linear | 9,618 | -10–10 | 0.0005602 ± 0.38434 | 99.30339% | 99.30339% | normalized | 0.01455 / 0.15612 / 0.29769 | 0–0.3125 raw / 0.5–0.51562 norm: 9,553 (99.3%)<br>-1.25–-0.9375 raw / 0.4375–0.45312 norm: 6 (0.1%)<br>9.6875–10 raw / 0.98438–1 norm: 6 (0.1%) |
| `modulation_3_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_3_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_40_amount` | Linear | 9,618 | -1–1 | 0.00538 ± 0.08185 | 97.07839% | 97.07839% | normalized | 0.00142 / 0.0159 / 0.03038 | 0–0.03125 raw / 0.5–0.51562 norm: 9,338 (97.1%)<br>0.125–0.15625 raw / 0.5625–0.57812 norm: 28 (0.3%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 24 (0.2%) |
| `modulation_40_power` | Linear | 9,618 | 0–2 | 0.0002079 ± 0.02039 | 99.9896% | 99.9896% | normalized | 0.01562 / 0.15625 / 0.29688 | 0–0.3125 raw / 0.5–0.51562 norm: 9,617 (100.0%)<br>1.875–2.1875 raw / 0.59375–0.60938 norm: 1 (0.0%) |
| `modulation_40_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_40_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_41_amount` | Linear | 9,618 | -1–1 | 0.00415 ± 0.06664 | 97.14078% | 97.14078% | normalized | 0.00137 / 0.01584 / 0.03031 | 0–0.03125 raw / 0.5–0.51562 norm: 9,347 (97.2%)<br>0.125–0.15625 raw / 0.5625–0.57812 norm: 20 (0.2%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 17 (0.2%) |
| `modulation_41_power` | Linear | 9,618 | 0–2 | 0.0002216 ± 0.02044 | 99.97921% | 99.97921% | normalized | 0.01562 / 0.15625 / 0.29688 | 0–0.3125 raw / 0.5–0.51562 norm: 9,617 (100.0%)<br>1.875–2.1875 raw / 0.59375–0.60938 norm: 1 (0.0%) |
| `modulation_41_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_41_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_42_amount` | Linear | 9,618 | -1–1 | 0.00453 ± 0.06592 | 97.48388% | 97.48388% | normalized | 0.00142 / 0.01583 / 0.03025 | 0–0.03125 raw / 0.5–0.51562 norm: 9,383 (97.6%)<br>0.125–0.15625 raw / 0.5625–0.57812 norm: 19 (0.2%)<br>0.15625–0.1875 raw / 0.57812–0.59375 norm: 16 (0.2%) |
| `modulation_42_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_42_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_42_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_43_amount` | Linear | 9,618 | -1–1 | 0.00339 ± 0.05848 | 97.68143% | 97.68143% | normalized | 0.00142 / 0.01581 / 0.03021 | 0–0.03125 raw / 0.5–0.51562 norm: 9,397 (97.7%)<br>0.125–0.15625 raw / 0.5625–0.57812 norm: 28 (0.3%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 18 (0.2%) |
| `modulation_43_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_43_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_43_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_44_amount` | Linear | 9,618 | -1–1 | 0.00411 ± 0.06657 | 97.69183% | 97.69183% | normalized | 0.00145 / 0.01584 / 0.03023 | 0–0.03125 raw / 0.5–0.51562 norm: 9,396 (97.7%)<br>0.21875–0.25 raw / 0.60938–0.625 norm: 25 (0.3%)<br>0.125–0.15625 raw / 0.5625–0.57812 norm: 20 (0.2%) |
| `modulation_44_power` | Linear | 9,618 | 0–10 | 0.00104 ± 0.10196 | 99.9896% | 99.9896% | normalized | 0.01562 / 0.15625 / 0.29688 | 0–0.3125 raw / 0.5–0.51562 norm: 9,617 (100.0%)<br>9.6875–10 raw / 0.98438–1 norm: 1 (0.0%) |
| `modulation_44_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_44_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_45_amount` | Linear | 9,618 | -1–1 | 0.00343 ± 0.06196 | 97.98295% | 97.98295% | normalized | 0.00146 / 0.01581 / 0.03016 | 0–0.03125 raw / 0.5–0.51562 norm: 9,424 (98.0%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 30 (0.3%)<br>0.125–0.15625 raw / 0.5625–0.57812 norm: 15 (0.2%) |
| `modulation_45_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_45_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_45_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_46_amount` | Linear | 9,618 | -0.99–1 | 0.00446 ± 0.05617 | 98.10771% | 98.10771% | normalized | 0.00148 / 0.01581 / 0.03014 | 0–0.03125 raw / 0.5–0.51562 norm: 9,438 (98.1%)<br>0.125–0.15625 raw / 0.5625–0.57812 norm: 18 (0.2%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 16 (0.2%) |
| `modulation_46_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_46_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_46_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_47_amount` | Linear | 9,618 | -1–1 | 0.00336 ± 0.05711 | 98.30526% | 98.30526% | normalized | 0.00151 / 0.01581 / 0.03011 | 0–0.03125 raw / 0.5–0.51562 norm: 9,456 (98.3%)<br>0.125–0.15625 raw / 0.5625–0.57812 norm: 24 (0.2%)<br>0.15625–0.1875 raw / 0.57812–0.59375 norm: 21 (0.2%) |
| `modulation_47_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_47_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_47_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_48_amount` | Linear | 9,618 | -1–1 | 0.00317 ± 0.05354 | 98.35725% | 98.35725% | normalized | 0.00146 / 0.01575 / 0.03004 | 0–0.03125 raw / 0.5–0.51562 norm: 9,464 (98.4%)<br>0.1875–0.21875 raw / 0.59375–0.60938 norm: 17 (0.2%)<br>-0.0625–-0.03125 raw / 0.46875–0.48438 norm: 14 (0.1%) |
| `modulation_48_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_48_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_48_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_49_amount` | Linear | 9,618 | -1–1 | 0.00264 ± 0.05082 | 98.61718% | 98.61718% | normalized | 0.0015 / 0.01575 / 0.03001 | 0–0.03125 raw / 0.5–0.51562 norm: 9,488 (98.6%)<br>0.15625–0.1875 raw / 0.57812–0.59375 norm: 14 (0.1%)<br>0.1875–0.21875 raw / 0.59375–0.60938 norm: 14 (0.1%) |
| `modulation_49_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_49_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_49_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_4_amount` | Linear | 9,618 | -1–1 | 0.23117 ± 0.4103 | 32.24163% | 32.24163% | normalized | -0.33461 / 0.13135 / 0.98451 | 0–0.03125 raw / 0.5–0.51562 norm: 3,196 (33.2%)<br>0.96875–1 raw / 0.98438–1 norm: 972 (10.1%)<br>0.5–0.53125 raw / 0.75–0.76562 norm: 385 (4.0%) |
| `modulation_4_power` | Linear | 9,618 | -10–10 | -0.00295 ± 0.36096 | 99.40736% | 99.40736% | normalized | 0.01457 / 0.15599 / 0.29741 | 0–0.3125 raw / 0.5–0.51562 norm: 9,563 (99.4%)<br>-1.5625–-1.25 raw / 0.42188–0.4375 norm: 8 (0.1%)<br>-2.1875–-1.875 raw / 0.39062–0.40625 norm: 5 (0.1%) |
| `modulation_4_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_4_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_50_amount` | Linear | 9,618 | -1–1 | 0.00264 ± 0.04501 | 98.68996% | 98.68996% | normalized | 0.0015 / 0.01575 / 0.02999 | 0–0.03125 raw / 0.5–0.51562 norm: 9,493 (98.7%)<br>0.125–0.15625 raw / 0.5625–0.57812 norm: 16 (0.2%)<br>0.15625–0.1875 raw / 0.57812–0.59375 norm: 10 (0.1%) |
| `modulation_50_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_50_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_50_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_51_amount` | Linear | 9,618 | -1–1 | 0.00189 ± 0.03669 | 98.83552% | 98.83552% | normalized | 0.00152 / 0.01575 / 0.02997 | 0–0.03125 raw / 0.5–0.51562 norm: 9,507 (98.8%)<br>0.1875–0.21875 raw / 0.59375–0.60938 norm: 14 (0.1%)<br>0.125–0.15625 raw / 0.5625–0.57812 norm: 13 (0.1%) |
| `modulation_51_power` | Linear | 9,618 | -0.12582–0 | -1.308e-05 ± 0.00128 | 99.9896% | 99.9896% | normalized | 0.01559 / 0.15622 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,617 (100.0%)<br>-0.3125–0 raw / 0.48438–0.5 norm: 1 (0.0%) |
| `modulation_51_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_51_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_52_amount` | Linear | 9,618 | -1–1 | 0.0003474 ± 0.03193 | 98.99147% | 98.99147% | normalized | 0.00145 / 0.01565 / 0.02985 | 0–0.03125 raw / 0.5–0.51562 norm: 9,522 (99.0%)<br>0.125–0.15625 raw / 0.5625–0.57812 norm: 15 (0.2%)<br>-0.25–-0.21875 raw / 0.375–0.39062 norm: 9 (0.1%) |
| `modulation_52_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_52_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_52_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_53_amount` | Linear | 9,618 | -1–1 | 0.000927 ± 0.03841 | 99.02267% | 99.02267% | normalized | 0.00149 / 0.01569 / 0.02989 | 0–0.03125 raw / 0.5–0.51562 norm: 9,524 (99.0%)<br>0.125–0.15625 raw / 0.5625–0.57812 norm: 20 (0.2%)<br>0.5–0.53125 raw / 0.75–0.76562 norm: 9 (0.1%) |
| `modulation_53_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_53_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_53_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_54_amount` | Linear | 9,618 | -1–1 | 0.00113 ± 0.04262 | 99.20981% | 99.20981% | normalized | 0.00151 / 0.01568 / 0.02985 | 0–0.03125 raw / 0.5–0.51562 norm: 9,542 (99.2%)<br>0.15625–0.1875 raw / 0.57812–0.59375 norm: 13 (0.1%)<br>-1–-0.96875 raw / 0–0.01562 norm: 5 (0.1%) |
| `modulation_54_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_54_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_54_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_55_amount` | Linear | 9,618 | -1–1 | 0.00117 ± 0.03495 | 99.29299% | 99.29299% | normalized | 0.00152 / 0.01568 / 0.02984 | 0–0.03125 raw / 0.5–0.51562 norm: 9,550 (99.3%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 9 (0.1%)<br>0.375–0.40625 raw / 0.6875–0.70312 norm: 6 (0.1%) |
| `modulation_55_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_55_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_55_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_56_amount` | Linear | 9,618 | -1–1 | 0.00139 ± 0.0367 | 99.35538% | 99.35538% | normalized | 0.00154 / 0.01569 / 0.02984 | 0–0.03125 raw / 0.5–0.51562 norm: 9,556 (99.4%)<br>0.15625–0.1875 raw / 0.57812–0.59375 norm: 9 (0.1%)<br>0.40625–0.4375 raw / 0.70312–0.71875 norm: 8 (0.1%) |
| `modulation_56_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_56_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_56_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_57_amount` | Linear | 9,618 | -0.54317–1 | 0.00152 ± 0.03009 | 99.42816% | 99.42816% | normalized | 0.00156 / 0.0157 / 0.02984 | 0–0.03125 raw / 0.5–0.51562 norm: 9,563 (99.4%)<br>0.1875–0.21875 raw / 0.59375–0.60938 norm: 14 (0.1%)<br>0.09375–0.125 raw / 0.54688–0.5625 norm: 6 (0.1%) |
| `modulation_57_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_57_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_57_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_58_amount` | Linear | 9,618 | -1–1 | 0.00147 ± 0.03165 | 99.48014% | 99.48014% | normalized | 0.00155 / 0.01568 / 0.02982 | 0–0.03125 raw / 0.5–0.51562 norm: 9,568 (99.5%)<br>0.375–0.40625 raw / 0.6875–0.70312 norm: 6 (0.1%)<br>0.4375–0.46875 raw / 0.71875–0.73438 norm: 6 (0.1%) |
| `modulation_58_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_58_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_58_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_59_amount` | Linear | 9,618 | -0.41674–0.91944 | 0.0009006 ± 0.02177 | 99.49054% | 99.49054% | normalized | 0.00154 / 0.01567 / 0.0298 | 0–0.03125 raw / 0.5–0.51562 norm: 9,570 (99.5%)<br>0.1875–0.21875 raw / 0.59375–0.60938 norm: 11 (0.1%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 6 (0.1%) |
| `modulation_59_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_59_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_59_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_5_amount` | Linear | 9,618 | -1–1 | 0.18669 ± 0.37962 | 40.0811% | 40.0811% | normalized | -0.28918 / 0.03088 / 0.98057 | 0–0.03125 raw / 0.5–0.51562 norm: 3,936 (40.9%)<br>0.96875–1 raw / 0.98438–1 norm: 775 (8.1%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 314 (3.3%) |
| `modulation_5_power` | Linear | 9,618 | -10–10 | 0.0013 ± 0.30035 | 99.53213% | 99.53213% | normalized | 0.01494 / 0.15614 / 0.29733 | 0–0.3125 raw / 0.5–0.51562 norm: 9,578 (99.6%)<br>-1.875–-1.5625 raw / 0.40625–0.42188 norm: 6 (0.1%)<br>-4.0625–-3.75 raw / 0.29688–0.3125 norm: 4 (0.0%) |
| `modulation_5_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_5_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_60_amount` | Linear | 9,618 | -0.13459–1 | 0.00166 ± 0.03324 | 99.59451% | 99.59451% | normalized | 0.00156 / 0.01568 / 0.0298 | 0–0.03125 raw / 0.5–0.51562 norm: 9,579 (99.6%)<br>0.96875–1 raw / 0.98438–1 norm: 7 (0.1%)<br>0.1875–0.21875 raw / 0.59375–0.60938 norm: 4 (0.0%) |
| `modulation_60_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_60_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_60_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_61_amount` | Linear | 9,618 | -0.4384–1 | 0.0008866 ± 0.02918 | 99.6361% | 99.6361% | normalized | 0.00154 / 0.01565 / 0.02976 | 0–0.03125 raw / 0.5–0.51562 norm: 9,583 (99.6%)<br>-0.25–-0.21875 raw / 0.375–0.39062 norm: 7 (0.1%)<br>0.5–0.53125 raw / 0.75–0.76562 norm: 5 (0.1%) |
| `modulation_61_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_61_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_61_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_62_amount` | Linear | 9,618 | -1–1 | 0.0008593 ± 0.03201 | 99.68808% | 99.68808% | normalized | 0.00155 / 0.01566 / 0.02976 | 0–0.03125 raw / 0.5–0.51562 norm: 9,588 (99.7%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 7 (0.1%)<br>0.96875–1 raw / 0.98438–1 norm: 6 (0.1%) |
| `modulation_62_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_62_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_62_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_63_amount` | Linear | 9,618 | -1–0.75 | -5.774e-05 ± 0.02065 | 99.70888% | 99.70888% | normalized | 0.00153 / 0.01563 / 0.02973 | 0–0.03125 raw / 0.5–0.51562 norm: 9,590 (99.7%)<br>0.15625–0.1875 raw / 0.57812–0.59375 norm: 8 (0.1%)<br>-0.125–-0.09375 raw / 0.4375–0.45312 norm: 5 (0.1%) |
| `modulation_63_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_63_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_63_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_64_amount` | Linear | 9,618 | -0.58414–1 | 0.00105 ± 0.02605 | 99.72967% | 99.72967% | normalized | 0.00156 / 0.01566 / 0.02976 | 0–0.03125 raw / 0.5–0.51562 norm: 9,592 (99.7%)<br>0.4375–0.46875 raw / 0.71875–0.73438 norm: 7 (0.1%)<br>0.75–0.78125 raw / 0.875–0.89062 norm: 5 (0.1%) |
| `modulation_64_power` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.01562 / 0.15623 / 0.29684 | 0–0.3125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `modulation_64_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_64_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_6_amount` | Linear | 9,618 | -1–1 | 0.15855 ± 0.36345 | 46.309% | 46.309% | normalized | -0.28395 / 0.02695 / 0.97789 | 0–0.03125 raw / 0.5–0.51562 norm: 4,532 (47.1%)<br>0.96875–1 raw / 0.98438–1 norm: 681 (7.1%)<br>0.5–0.53125 raw / 0.75–0.76562 norm: 286 (3.0%) |
| `modulation_6_power` | Linear | 9,618 | -10–10 | 0.00227 ± 0.35084 | 99.51133% | 99.51133% | normalized | 0.01485 / 0.1561 / 0.29736 | 0–0.3125 raw / 0.5–0.51562 norm: 9,574 (99.5%)<br>-0.625–-0.3125 raw / 0.46875–0.48438 norm: 7 (0.1%)<br>-1.25–-0.9375 raw / 0.4375–0.45312 norm: 6 (0.1%) |
| `modulation_6_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_6_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_7_amount` | Linear | 9,618 | -1–1 | 0.14348 ± 0.34935 | 51.38282% | 51.38282% | normalized | -0.22877 / 0.02511 / 0.9749 | 0–0.03125 raw / 0.5–0.51562 norm: 4,980 (51.8%)<br>0.96875–1 raw / 0.98438–1 norm: 600 (6.2%)<br>0.25–0.28125 raw / 0.625–0.64062 norm: 269 (2.8%) |
| `modulation_7_power` | Linear | 9,618 | -10–10 | 0.00925 ± 0.41205 | 99.58411% | 99.58411% | normalized | 0.01513 / 0.1563 / 0.29747 | 0–0.3125 raw / 0.5–0.51562 norm: 9,580 (99.6%)<br>9.6875–10 raw / 0.98438–1 norm: 13 (0.1%)<br>-1.875–-1.5625 raw / 0.40625–0.42188 norm: 3 (0.0%) |
| `modulation_7_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_7_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_8_amount` | Linear | 9,618 | -1–1 | 0.12918 ± 0.33049 | 56.18632% | 56.18632% | normalized | -0.19502 / 0.02327 / 0.97191 | 0–0.03125 raw / 0.5–0.51562 norm: 5,459 (56.8%)<br>0.96875–1 raw / 0.98438–1 norm: 536 (5.6%)<br>0.125–0.15625 raw / 0.5625–0.57812 norm: 237 (2.5%) |
| `modulation_8_power` | Linear | 9,618 | -7.12323–10 | 0.01009 ± 0.37386 | 99.6153% | 99.6153% | normalized | 0.01516 / 0.15633 / 0.2975 | 0–0.3125 raw / 0.5–0.51562 norm: 9,580 (99.6%)<br>9.6875–10 raw / 0.98438–1 norm: 11 (0.1%)<br>-1.5625–-1.25 raw / 0.42188–0.4375 norm: 4 (0.0%) |
| `modulation_8_ramp_down` | — | 287 | -10–-4.01221 | -9.97914 ± 0.35283 | —% | 0% | raw | -9.99532 / -9.95322 / -9.91112 | -10–-9.90644 raw: 286 (99.7%)<br>-4.10577–-4.01221 raw: 1 (0.3%) |
| `modulation_8_ramp_up` | — | 287 | -10–-2.343 | -9.97332 ± 0.45119 | —% | 0% | raw | -9.99402 / -9.94018 / -9.88634 | -10–-9.88036 raw: 286 (99.7%)<br>-2.46264–-2.343 raw: 1 (0.3%) |
| `modulation_9_amount` | Linear | 9,618 | -1–1 | 0.10997 ± 0.31416 | 60.86504% | 60.86504% | normalized | -0.18335 / 0.02176 / 0.90677 | 0–0.03125 raw / 0.5–0.51562 norm: 5,908 (61.4%)<br>0.96875–1 raw / 0.98438–1 norm: 466 (4.8%)<br>0.5–0.53125 raw / 0.75–0.76562 norm: 215 (2.2%) |
| `modulation_9_power` | Linear | 9,618 | -9.11111–3.90476 | -0.0034 ± 0.14055 | 99.78166% | 99.78166% | normalized | 0.0152 / 0.15609 / 0.29698 | 0–0.3125 raw / 0.5–0.51562 norm: 9,599 (99.8%)<br>-2.5–-2.1875 raw / 0.375–0.39062 norm: 4 (0.0%)<br>-2.8125–-2.5 raw / 0.35938–0.375 norm: 3 (0.0%) |
| `modulation_9_ramp_down` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `modulation_9_ramp_up` | — | 287 | -10–-10 | -10 ± 0 | —% | 0% | raw | -10 / -10 / -10 | -10–-10 raw: 287 (100.0%) |
| `osc_1_detune_power` | Linear | 9,618 | -5–5 | 1.46478 ± 1.0419 | 88.9582% | 0.3847% | normalized | 0.9313 / 1.48377 / 1.62457 | 1.40625–1.5625 raw / 0.64062–0.65625 norm: 8,575 (89.2%)<br>4.84375–5 raw / 0.98438–1 norm: 236 (2.5%)<br>-5–-4.84375 raw / 0–0.01562 norm: 115 (1.2%) |
| `osc_1_detune_range` | Linear | 9,618 | 0–48 | 2.20983 ± 2.85053 | 95.61239% | 0.76939% | normalized | 1.5229 / 1.87322 / 2.22354 | 1.5–2.25 raw / 0.03125–0.04688 norm: 9,265 (96.3%)<br>0–0.75 raw / 0–0.01562 norm: 149 (1.5%)<br>0.75–1.5 raw / 0.01562–0.03125 norm: 49 (0.5%) |
| `osc_1_distortion_amount` | Linear | 9,618 | 0–1 | 0.42937 ± 0.20681 | 59.84612% | 9.28467% | normalized | 0.00771 / 0.50556 / 0.69191 | 0.5–0.51562 raw / 0.5–0.51562 norm: 5,903 (61.4%)<br>0–0.01562 raw / 0–0.01562 norm: 974 (10.1%)<br>0.98438–1 raw / 0.98438–1 norm: 186 (1.9%) |
| `osc_1_distortion_phase` | Linear | 9,618 | 0–1 | 0.49984 ± 0.06546 | 95.06134% | 0.66542% | normalized | 0.50046 / 0.50782 / 0.51518 | 0.5–0.51562 raw / 0.5–0.51562 norm: 9,188 (95.5%)<br>0–0.01562 raw / 0–0.01562 norm: 67 (0.7%)<br>0.98438–1 raw / 0.98438–1 norm: 46 (0.5%) |
| `osc_1_distortion_spread` | Linear | 9,618 | -0.5–0.5 | 0.00203 ± 0.05443 | 95.55001% | 95.55001% | normalized | 0.0005282 / 0.00788 / 0.01524 | 0–0.01562 raw / 0.5–0.51562 norm: 9,195 (95.6%)<br>0.48438–0.5 raw / 0.98438–1 norm: 42 (0.4%)<br>-0.5–-0.48438 raw / 0–0.01562 norm: 26 (0.3%) |
| `osc_1_frame_spread` | Linear | 9,618 | -128–128 | 1.07835 ± 16.39972 | 94.72863% | 94.72863% | normalized | 0.1401 / 2.0357 / 3.9313 | 0–4 raw / 0.5–0.51562 norm: 9,132 (94.9%)<br>124–128 raw / 0.98438–1 norm: 59 (0.6%)<br>-128–-124 raw / 0–0.01562 norm: 34 (0.4%) |
| `osc_1_level` | Quadratic | 9,618 | 0–1 | 0.52758 ± 0.30964 | 44.61426% | 17.58162% | normalized | 0.06066 / 0.69787 / 0.9829 | 0.69597–0.70711 raw / 0.48438–0.5 norm: 4,361 (45.3%)<br>0–0.125 raw / 0–0.01562 norm: 2,042 (21.2%)<br>0.99216–1 raw / 0.98438–1 norm: 472 (4.9%) |
| `osc_1_pan` | Linear | 9,618 | -1–1 | -0.00702 ± 0.10835 | 95.49802% | 95.49802% | normalized | 0.0007176 / 0.01538 / 0.03003 | 0–0.03125 raw / 0.5–0.51562 norm: 9,226 (95.9%)<br>-1–-0.96875 raw / 0–0.01562 norm: 63 (0.7%)<br>-0.03125–0 raw / 0.48438–0.5 norm: 45 (0.5%) |
| `osc_1_phase` | Linear | 9,618 | 0–1 | 0.4378 ± 0.17887 | 81.76336% | 11.81119% | normalized | 0.00654 / 0.50651 / 0.51508 | 0.5–0.51562 raw / 0.5–0.51562 norm: 7,886 (82.0%)<br>0–0.01562 raw / 0–0.01562 norm: 1,148 (11.9%)<br>0.98438–1 raw / 0.98438–1 norm: 74 (0.8%) |
| `osc_1_random_phase` | Linear | 9,618 | 0–1 | 0.71048 ± 0.44621 | 69.25556% | 26.39842% | normalized | 0.00291 / 0.98875 / 0.99887 | 0.98438–1 raw / 0.98438–1 norm: 6,678 (69.4%)<br>0–0.01562 raw / 0–0.01562 norm: 2,582 (26.8%)<br>0.5–0.51562 raw / 0.5–0.51562 norm: 21 (0.2%) |
| `osc_1_spectral_morph_amount` | Linear | 9,618 | 0–1 | 0.45614 ± 0.20064 | 63.5891% | 6.96611% | normalized | 0.0099 / 0.50632 / 0.77261 | 0.5–0.51562 raw / 0.5–0.51562 norm: 6,264 (65.1%)<br>0–0.01562 raw / 0–0.01562 norm: 759 (7.9%)<br>0.98438–1 raw / 0.98438–1 norm: 313 (3.3%) |
| `osc_1_spectral_morph_phase` | — | 7,517 | 0–1 | 0.49679 ± 0.0424 | —% | 0.439% | raw | 0.50056 / 0.50772 / 0.51489 | 0.5–0.51562 raw: 7,375 (98.1%)<br>0–0.01562 raw: 35 (0.5%)<br>0.32812–0.34375 raw: 10 (0.1%) |
| `osc_1_spectral_morph_spread` | Linear | 9,618 | -0.5–0.5 | 0.00126 ± 0.05312 | 95.31088% | 95.31088% | normalized | 0.0004611 / 0.00783 / 0.0152 | 0–0.01562 raw / 0.5–0.51562 norm: 9,178 (95.4%)<br>0.48438–0.5 raw / 0.98438–1 norm: 36 (0.4%)<br>-0.5–-0.48438 raw / 0–0.01562 norm: 24 (0.2%) |
| `osc_1_stereo_spread` | Linear | 9,618 | 0–1 | 0.96571 ± 0.16169 | 94.57268% | 1.56997% | normalized | 0.84039 / 0.99175 / 0.99917 | 0.98438–1 raw / 0.98438–1 norm: 9,104 (94.7%)<br>0–0.01562 raw / 0–0.01562 norm: 172 (1.8%)<br>0.5–0.51562 raw / 0.5–0.51562 norm: 26 (0.3%) |
| `osc_1_tune` | Linear | 9,618 | -1–1 | -0.0004382 ± 0.1066 | 92.97151% | 92.97151% | normalized | 0.0005405 / 0.01555 / 0.03056 | 0–0.03125 raw / 0.5–0.51562 norm: 9,011 (93.7%)<br>-0.03125–0 raw / 0.48438–0.5 norm: 70 (0.7%)<br>0.09375–0.125 raw / 0.54688–0.5625 norm: 47 (0.5%) |
| `osc_1_unison_blend` | Linear | 9,618 | 0–1 | 0.79111 ± 0.0963 | 92.71158% | 0.53026% | normalized | 0.79706 / 0.80464 / 0.81221 | 0.79688–0.8125 raw / 0.79688–0.8125 norm: 8,927 (92.8%)<br>0.98438–1 raw / 0.98438–1 norm: 252 (2.6%)<br>0–0.01562 raw / 0–0.01562 norm: 60 (0.6%) |
| `osc_1_unison_detune` | Quadratic | 9,618 | 0–10 | 3.19799 ± 1.85348 | 40.96486% | 10.76107% | normalized | 0.67664 / 3.81937 / 4.61748 | 4.33013–4.50694 raw / 0.1875–0.20312 norm: 4,021 (41.8%)<br>0–1.25 raw / 0–0.01562 norm: 1,641 (17.1%)<br>1.25–1.76777 raw / 0.01562–0.03125 norm: 652 (6.8%) |
| `osc_1_wave_frame` | Linear | 9,618 | 0–256 | 47.86882 ± 72.46376 | 58.43211% | 58.43211% | normalized | 0.33515 / 3.35145 / 208.19167 | 0–4 raw / 0–0.01562 norm: 5,739 (59.7%)<br>252–256 raw / 0.98438–1 norm: 338 (3.5%)<br>100–104 raw / 0.39062–0.40625 norm: 134 (1.4%) |
| `osc_2_detune_power` | Linear | 9,618 | -5–5 | 1.4905 ± 0.82242 | 92.2749% | 0.20794% | normalized | 1.4085 / 1.48459 / 1.56067 | 1.40625–1.5625 raw / 0.64062–0.65625 norm: 8,887 (92.4%)<br>4.84375–5 raw / 0.98438–1 norm: 153 (1.6%)<br>-5–-4.84375 raw / 0–0.01562 norm: 50 (0.5%) |
| `osc_2_detune_range` | Linear | 9,618 | 0–48 | 2.13419 ± 2.41963 | 97.36952% | 0.42628% | normalized | 1.52778 / 1.8734 / 2.21902 | 1.5–2.25 raw / 0.03125–0.04688 norm: 9,391 (97.6%)<br>0–0.75 raw / 0–0.01562 norm: 92 (1.0%)<br>0.75–1.5 raw / 0.01562–0.03125 norm: 41 (0.4%) |
| `osc_2_distortion_amount` | Linear | 9,618 | 0–1 | 0.45977 ± 0.17813 | 71.03348% | 6.16552% | normalized | 0.01147 / 0.50667 / 0.67113 | 0.5–0.51562 raw / 0.5–0.51562 norm: 6,923 (72.0%)<br>0–0.01562 raw / 0–0.01562 norm: 655 (6.8%)<br>0.98438–1 raw / 0.98438–1 norm: 178 (1.9%) |
| `osc_2_distortion_phase` | Linear | 9,618 | 0–1 | 0.499 ± 0.05955 | 95.93471% | 0.56145% | normalized | 0.50049 / 0.5078 / 0.51511 | 0.5–0.51562 raw / 0.5–0.51562 norm: 9,246 (96.1%)<br>0–0.01562 raw / 0–0.01562 norm: 55 (0.6%)<br>0.51562–0.53125 raw / 0.51562–0.53125 norm: 44 (0.5%) |
| `osc_2_distortion_spread` | Linear | 9,618 | -0.5–0.5 | 0.00101 ± 0.04088 | 97.46309% | 97.46309% | normalized | 0.0006244 / 0.00783 / 0.01504 | 0–0.01562 raw / 0.5–0.51562 norm: 9,380 (97.5%)<br>0.48438–0.5 raw / 0.98438–1 norm: 28 (0.3%)<br>-0.03125–-0.01562 raw / 0.46875–0.48438 norm: 14 (0.1%) |
| `osc_2_frame_spread` | Linear | 9,618 | -128–128 | 0.51783 ± 11.71677 | 96.86005% | 96.86005% | normalized | 0.16198 / 2.01715 / 3.87232 | 0–4 raw / 0.5–0.51562 norm: 9,331 (97.0%)<br>124–128 raw / 0.98438–1 norm: 24 (0.2%)<br>-128–-124 raw / 0–0.01562 norm: 19 (0.2%) |
| `osc_2_level` | Quadratic | 9,618 | 0–1 | 0.44144 ± 0.31989 | 34.63298% | 23.49761% | normalized | 0.05203 / 0.56857 / 0.80872 | 0.69597–0.70711 raw / 0.48438–0.5 norm: 3,376 (35.1%)<br>0–0.125 raw / 0–0.01562 norm: 2,775 (28.9%)<br>0.99216–1 raw / 0.98438–1 norm: 280 (2.9%) |
| `osc_2_pan` | Linear | 9,618 | -1–1 | 0.00213 ± 0.11645 | 95.10293% | 95.10293% | normalized | 0.0009115 / 0.01564 / 0.03037 | 0–0.03125 raw / 0.5–0.51562 norm: 9,183 (95.5%)<br>0.96875–1 raw / 0.98438–1 norm: 48 (0.5%)<br>-1–-0.96875 raw / 0–0.01562 norm: 34 (0.4%) |
| `osc_2_phase` | Linear | 9,618 | 0–1 | 0.44478 ± 0.16764 | 83.59326% | 10.06446% | normalized | 0.00755 / 0.50668 / 0.51508 | 0.5–0.51562 raw / 0.5–0.51562 norm: 8,046 (83.7%)<br>0–0.01562 raw / 0–0.01562 norm: 995 (10.3%)<br>0.98438–1 raw / 0.98438–1 norm: 57 (0.6%) |
| `osc_2_random_phase` | Linear | 9,618 | 0–1 | 0.75554 ± 0.42309 | 73.97588% | 22.29154% | normalized | 0.00345 / 0.98946 / 0.99894 | 0.98438–1 raw / 0.98438–1 norm: 7,129 (74.1%)<br>0–0.01562 raw / 0–0.01562 norm: 2,176 (22.6%)<br>0.6875–0.70312 raw / 0.6875–0.70312 norm: 16 (0.2%) |
| `osc_2_spectral_morph_amount` | Linear | 9,618 | 0–1 | 0.46864 ± 0.18074 | 70.50322% | 5.44812% | normalized | 0.013 / 0.50686 / 0.73549 | 0.5–0.51562 raw / 0.5–0.51562 norm: 6,910 (71.8%)<br>0–0.01562 raw / 0–0.01562 norm: 578 (6.0%)<br>0.98438–1 raw / 0.98438–1 norm: 255 (2.7%) |
| `osc_2_spectral_morph_phase` | — | 7,517 | 0–1 | 0.49924 ± 0.02457 | —% | 0.05321% | raw | 0.48655 / 0.50412 / 0.51452 | 0.5–0.51562 raw: 5,078 (67.6%)<br>0.48438–0.5 raw: 2,374 (31.6%)<br>0.29688–0.3125 raw: 10 (0.1%) |
| `osc_2_spectral_morph_spread` | Linear | 9,618 | -0.5–0.5 | 0.0001333 ± 0.03609 | 97.23435% | 97.23435% | normalized | 0.0006057 / 0.00783 / 0.01505 | 0–0.01562 raw / 0.5–0.51562 norm: 9,361 (97.3%)<br>0.03125–0.04688 raw / 0.53125–0.54688 norm: 22 (0.2%)<br>-0.5–-0.48438 raw / 0–0.01562 norm: 14 (0.1%) |
| `osc_2_stereo_spread` | Linear | 9,618 | 0–1 | 0.97766 ± 0.13296 | 96.50655% | 1.15409% | normalized | 0.98462 / 0.99191 / 0.99919 | 0.98438–1 raw / 0.98438–1 norm: 9,284 (96.5%)<br>0–0.01562 raw / 0–0.01562 norm: 118 (1.2%)<br>0.5–0.51562 raw / 0.5–0.51562 norm: 20 (0.2%) |
| `osc_2_tune` | Linear | 9,618 | -1–1 | 0.00255 ± 0.09839 | 91.74465% | 91.74465% | normalized | 0.0004638 / 0.01569 / 0.03091 | 0–0.03125 raw / 0.5–0.51562 norm: 8,884 (92.4%)<br>-0.03125–0 raw / 0.48438–0.5 norm: 75 (0.8%)<br>0.03125–0.0625 raw / 0.51562–0.53125 norm: 66 (0.7%) |
| `osc_2_unison_blend` | Linear | 9,618 | 0–1 | 0.79393 ± 0.07976 | 95.36286% | 0.41589% | normalized | 0.79729 / 0.80466 / 0.81203 | 0.79688–0.8125 raw / 0.79688–0.8125 norm: 9,177 (95.4%)<br>0.98438–1 raw / 0.98438–1 norm: 166 (1.7%)<br>0–0.01562 raw / 0–0.01562 norm: 41 (0.4%) |
| `osc_2_unison_detune` | Quadratic | 9,618 | 0–10 | 3.47535 ± 1.71012 | 53.04637% | 8.56727% | normalized | 0.77374 / 4.35766 / 4.5056 | 4.33013–4.50694 raw / 0.1875–0.20312 norm: 5,157 (53.6%)<br>0–1.25 raw / 0–0.01562 norm: 1,255 (13.0%)<br>1.25–1.76777 raw / 0.01562–0.03125 norm: 474 (4.9%) |
| `osc_2_wave_frame` | Linear | 9,618 | 0–256 | 42.54396 ± 69.80864 | 63.16282% | 63.16282% | normalized | 0.31204 / 3.12038 / 200.55333 | 0–4 raw / 0–0.01562 norm: 6,164 (64.1%)<br>252–256 raw / 0.98438–1 norm: 292 (3.0%)<br>104–108 raw / 0.40625–0.42188 norm: 118 (1.2%) |
| `osc_3_detune_power` | Linear | 9,618 | -5–5 | 1.50288 ± 0.59186 | 95.43564% | 0.15596% | normalized | 1.41099 / 1.48453 / 1.55807 | 1.40625–1.5625 raw / 0.64062–0.65625 norm: 9,195 (95.6%)<br>4.84375–5 raw / 0.98438–1 norm: 87 (0.9%)<br>-5–-4.84375 raw / 0–0.01562 norm: 30 (0.3%) |
| `osc_3_detune_range` | Linear | 9,618 | 0–48 | 2.0269 ± 1.17022 | 98.50281% | 0.34311% | normalized | 1.53144 / 1.87342 / 2.2154 | 1.5–2.25 raw / 0.03125–0.04688 norm: 9,491 (98.7%)<br>0–0.75 raw / 0–0.01562 norm: 66 (0.7%)<br>0.75–1.5 raw / 0.01562–0.03125 norm: 17 (0.2%) |
| `osc_3_distortion_amount` | Linear | 9,618 | 0–1 | 0.47541 ± 0.13924 | 82.81347% | 3.52464% | normalized | 0.0992 / 0.50724 / 0.52799 | 0.5–0.51562 raw / 0.5–0.51562 norm: 8,021 (83.4%)<br>0–0.01562 raw / 0–0.01562 norm: 365 (3.8%)<br>0.98438–1 raw / 0.98438–1 norm: 106 (1.1%) |
| `osc_3_distortion_phase` | Linear | 9,618 | 0–1 | 0.49934 ± 0.04635 | 97.46309% | 0.3639% | normalized | 0.5006 / 0.5078 / 0.515 | 0.5–0.51562 raw / 0.5–0.51562 norm: 9,392 (97.7%)<br>0–0.01562 raw / 0–0.01562 norm: 38 (0.4%)<br>0.98438–1 raw / 0.98438–1 norm: 14 (0.1%) |
| `osc_3_distortion_spread` | Linear | 9,618 | -0.5–0.5 | 0.0004408 ± 0.0294 | 98.36764% | 98.36764% | normalized | 0.0006848 / 0.00783 / 0.01497 | 0–0.01562 raw / 0.5–0.51562 norm: 9,465 (98.4%)<br>0.04688–0.0625 raw / 0.54688–0.5625 norm: 16 (0.2%)<br>0.48438–0.5 raw / 0.98438–1 norm: 13 (0.1%) |
| `osc_3_frame_spread` | Linear | 9,618 | -128–128 | 0.32377 ± 8.45742 | 98.12851% | 98.12851% | normalized | 0.18206 / 2.01504 / 3.84801 | 0–4 raw / 0.5–0.51562 norm: 9,444 (98.2%)<br>4–8 raw / 0.51562–0.53125 norm: 12 (0.1%)<br>48–52 raw / 0.6875–0.70312 norm: 12 (0.1%) |
| `osc_3_level` | Quadratic | 9,618 | 0–1 | 0.52245 ± 0.29462 | 51.98586% | 15.71013% | normalized | 0.06315 / 0.69798 / 0.7897 | 0.69597–0.70711 raw / 0.48438–0.5 norm: 5,027 (52.3%)<br>0–0.125 raw / 0–0.01562 norm: 1,884 (19.6%)<br>0.99216–1 raw / 0.98438–1 norm: 262 (2.7%) |
| `osc_3_pan` | Linear | 9,618 | -1–1 | -0.00227 ± 0.08484 | 96.90164% | 96.90164% | normalized | 0.00105 / 0.01552 / 0.02999 | 0–0.03125 raw / 0.5–0.51562 norm: 9,346 (97.2%)<br>-0.03125–0 raw / 0.48438–0.5 norm: 30 (0.3%)<br>-1–-0.96875 raw / 0–0.01562 norm: 25 (0.3%) |
| `osc_3_phase` | Linear | 9,618 | 0–1 | 0.46366 ± 0.13738 | 89.17654% | 6.56062% | normalized | 0.01161 / 0.50711 / 0.51498 | 0.5–0.51562 raw / 0.5–0.51562 norm: 8,587 (89.3%)<br>0–0.01562 raw / 0–0.01562 norm: 647 (6.7%)<br>0.98438–1 raw / 0.98438–1 norm: 31 (0.3%) |
| `osc_3_random_phase` | Linear | 9,618 | 0–1 | 0.83379 ± 0.36731 | 82.38719% | 15.38781% | normalized | 0.00501 / 0.99052 / 0.99905 | 0.98438–1 raw / 0.98438–1 norm: 7,925 (82.4%)<br>0–0.01562 raw / 0–0.01562 norm: 1,499 (15.6%)<br>0.67188–0.6875 raw / 0.67188–0.6875 norm: 15 (0.2%) |
| `osc_3_spectral_morph_amount` | Linear | 9,618 | 0–1 | 0.4751 ± 0.14664 | 80.60927% | 3.36868% | normalized | 0.09313 / 0.50711 / 0.56496 | 0.5–0.51562 raw / 0.5–0.51562 norm: 7,810 (81.2%)<br>0–0.01562 raw / 0–0.01562 norm: 356 (3.7%)<br>0.98438–1 raw / 0.98438–1 norm: 153 (1.6%) |
| `osc_3_spectral_morph_phase` | — | 7,517 | 0–1 | 0.4991 ± 0.0228 | —% | 0.07982% | raw | 0.50046 / 0.50767 / 0.51487 | 0.5–0.51562 raw: 7,336 (97.6%)<br>0.48438–0.5 raw: 94 (1.3%)<br>0.40625–0.42188 raw: 22 (0.3%) |
| `osc_3_spectral_morph_spread` | Linear | 9,618 | -0.5–0.5 | 0.000837 ± 0.03168 | 98.1701% | 98.1701% | normalized | 0.0006531 / 0.00781 / 0.01497 | 0–0.01562 raw / 0.5–0.51562 norm: 9,446 (98.2%)<br>-0.04688–-0.03125 raw / 0.45312–0.46875 norm: 14 (0.1%)<br>0.48438–0.5 raw / 0.98438–1 norm: 13 (0.1%) |
| `osc_3_stereo_spread` | Linear | 9,618 | 0–1 | 0.98781 ± 0.09771 | 98.03493% | 0.64462% | normalized | 0.98486 / 0.99203 / 0.9992 | 0.98438–1 raw / 0.98438–1 norm: 9,429 (98.0%)<br>0–0.01562 raw / 0–0.01562 norm: 65 (0.7%)<br>0.6875–0.70312 raw / 0.6875–0.70312 norm: 9 (0.1%) |
| `osc_3_tune` | Linear | 9,618 | -1–1 | -0.0014 ± 0.07386 | 95.86193% | 95.86193% | normalized | 0.0008608 / 0.01548 / 0.0301 | 0–0.03125 raw / 0.5–0.51562 norm: 9,252 (96.2%)<br>-0.03125–0 raw / 0.48438–0.5 norm: 52 (0.5%)<br>-0.125–-0.09375 raw / 0.4375–0.45312 norm: 33 (0.3%) |
| `osc_3_unison_blend` | Linear | 9,618 | 0–1 | 0.7975 ± 0.06279 | 96.97442% | 0.24953% | normalized | 0.79747 / 0.80472 / 0.81197 | 0.79688–0.8125 raw / 0.79688–0.8125 norm: 9,331 (97.0%)<br>0.98438–1 raw / 0.98438–1 norm: 130 (1.4%)<br>0–0.01562 raw / 0–0.01562 norm: 32 (0.3%) |
| `osc_3_unison_detune` | Quadratic | 9,618 | 0–10 | 3.80992 ± 1.4508 | 69.82741% | 6.22791% | normalized | 0.95373 / 4.38868 / 4.5012 | 4.33013–4.50694 raw / 0.1875–0.20312 norm: 6,760 (70.3%)<br>0–1.25 raw / 0–0.01562 norm: 826 (8.6%)<br>1.76777–2.16506 raw / 0.03125–0.04688 norm: 297 (3.1%) |
| `osc_3_wave_frame` | Linear | 9,618 | 0–256 | 27.97407 ± 59.04193 | 74.28779% | 74.28779% | normalized | 0.26618 / 2.66178 / 160.63 | 0–4 raw / 0–0.01562 norm: 7,226 (75.1%)<br>252–256 raw / 0.98438–1 norm: 189 (2.0%)<br>116–120 raw / 0.45312–0.46875 norm: 74 (0.8%) |
| `phaser_blend` | Linear | 9,618 | 0–2 | 0.9915 ± 0.21534 | 92.95072% | 2.15221% | normalized | 1.00034 / 1.01547 / 1.0306 | 1–1.03125 raw / 0.5–0.51562 norm: 8,940 (93.0%)<br>0–0.03125 raw / 0–0.01562 norm: 212 (2.2%)<br>1.96875–2 raw / 0.98438–1 norm: 142 (1.5%) |
| `phaser_center` | Linear | 9,618 | 8–136 | 78.7018 ± 11.82782 | 85.89104% | 0% | normalized | 60.76944 / 80.95543 / 81.99811 | 80–82 raw / 0.5625–0.57812 norm: 8,301 (86.3%)<br>8–10 raw / 0–0.01562 norm: 74 (0.8%)<br>70–72 raw / 0.48438–0.5 norm: 54 (0.6%) |
| `phaser_dry_wet` | Linear | 9,618 | 0–1 | 0.87309 ± 0.29256 | 81.89852% | 4.16927% | normalized | 0.05691 / 0.99048 / 0.99905 | 0.98438–1 raw / 0.98438–1 norm: 7,893 (82.1%)<br>0–0.01562 raw / 0–0.01562 norm: 440 (4.6%)<br>0.5–0.51562 raw / 0.5–0.51562 norm: 54 (0.6%) |
| `phaser_feedback` | Linear | 9,618 | 0–1 | 0.48995 ± 0.11873 | 83.70763% | 2.1938% | normalized | 0.2712 / 0.50765 / 0.60999 | 0.5–0.51562 raw / 0.5–0.51562 norm: 8,071 (83.9%)<br>0–0.01562 raw / 0–0.01562 norm: 227 (2.4%)<br>0.59375–0.60938 raw / 0.59375–0.60938 norm: 56 (0.6%) |
| `phaser_frequency` | Exponential | 9,618 | -5–2 | -2.97904 ± 0.35058 | 98.47162% | 0% | normalized | -3.02644 / -2.97646 / -2.92649 | -3.03125–-2.92188 raw / 0.28125–0.29688 norm: 9,472 (98.5%)<br>1.89062–2 raw / 0.98438–1 norm: 30 (0.3%)<br>-5–-4.89062 raw / 0–0.01562 norm: 16 (0.2%) |
| `phaser_mod_depth` | Linear | 9,618 | 0–48 | 23.3013 ± 6.43025 | 86.33812% | 3.07756% | normalized | 9.0925 / 24.35845 / 24.74884 | 24–24.75 raw / 0.5–0.51562 norm: 8,314 (86.4%)<br>0–0.75 raw / 0–0.01562 norm: 320 (3.3%)<br>47.25–48 raw / 0.98438–1 norm: 152 (1.6%) |
| `phaser_phase_offset` | Linear | 9,618 | 0–1 | 0.33067 ± 0.11358 | 87.8665% | 3.41027% | normalized | 0.10501 / 0.33574 / 0.34373 | 0.32812–0.34375 raw / 0.32812–0.34375 norm: 8,458 (87.9%)<br>0–0.01562 raw / 0–0.01562 norm: 388 (4.0%)<br>0.98438–1 raw / 0.98438–1 norm: 89 (0.9%) |
| `pitch_wheel` | Linear | 9,618 | -1–1 | -0.00149 ± 0.0434 | 99.02267% | 99.02267% | normalized | 0.00145 / 0.01559 / 0.02973 | 0–0.03125 raw / 0.5–0.51562 norm: 9,564 (99.4%)<br>-1–-0.96875 raw / 0–0.01562 norm: 12 (0.1%)<br>-0.0625–-0.03125 raw / 0.46875–0.48438 norm: 4 (0.0%) |
| `portamento_slope` | Linear | 9,618 | -8–8 | 0.07249 ± 0.99799 | 90.47619% | 90.47619% | normalized | 0.00217 / 0.12606 / 0.24995 | 0–0.25 raw / 0.5–0.51562 norm: 8,733 (90.8%)<br>7.75–8 raw / 0.98438–1 norm: 51 (0.5%)<br>-0.75–-0.5 raw / 0.45312–0.46875 norm: 41 (0.4%) |
| `portamento_time` | Exponential | 9,618 | -10–4 | -8.36975 ± 2.8502 | 70.77355% | 0.0104% | normalized | -9.98469 / -9.84691 / -2.49671 | -10–-9.78125 raw / 0–0.01562 norm: 6,871 (71.4%)<br>-3.4375–-3.21875 raw / 0.46875–0.48438 norm: 149 (1.5%)<br>-4.3125–-4.09375 raw / 0.40625–0.42188 norm: 129 (1.3%) |
| `random_1_frequency` | Exponential | 9,618 | -7–9 | 0.98074 ± 0.72926 | 95.06134% | 0.02079% | normalized | 1.00401 / 1.12225 / 1.2405 | 1–1.25 raw / 0.5–0.51562 norm: 9,150 (95.1%)<br>8.75–9 raw / 0.98438–1 norm: 33 (0.3%)<br>-1.5–-1.25 raw / 0.34375–0.35938 norm: 31 (0.3%) |
| `random_1_keytrack_tune` | Linear | 9,618 | -1–0.56223 | -5.791e-05 ± 0.01282 | 99.93762% | 99.93762% | normalized | 0.00155 / 0.01562 / 0.02969 | 0–0.03125 raw / 0.5–0.51562 norm: 9,612 (99.9%)<br>-1–-0.96875 raw / 0–0.01562 norm: 1 (0.0%)<br>-0.28125–-0.25 raw / 0.35938–0.375 norm: 1 (0.0%) |
| `random_2_frequency` | Exponential | 9,618 | -7–9 | 0.97027 ± 0.50775 | 97.25515% | 0% | normalized | 1.00748 / 1.12312 / 1.23875 | 1–1.25 raw / 0.5–0.51562 norm: 9,356 (97.3%)<br>-2.5–-2.25 raw / 0.28125–0.29688 norm: 25 (0.3%)<br>-2–-1.75 raw / 0.3125–0.32812 norm: 20 (0.2%) |
| `random_2_keytrack_tune` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.00156 / 0.01562 / 0.02968 | 0–0.03125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `random_3_frequency` | Exponential | 9,618 | -4.52356–9 | 0.99091 ± 0.32459 | 98.97068% | 0.02079% | normalized | 1.01073 / 1.12436 / 1.23798 | 1–1.25 raw / 0.5–0.51562 norm: 9,522 (99.0%)<br>0–0.25 raw / 0.4375–0.45312 norm: 16 (0.2%)<br>-3–-2.75 raw / 0.25–0.26562 norm: 6 (0.1%) |
| `random_3_keytrack_tune` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.00156 / 0.01562 / 0.02968 | 0–0.03125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `random_4_frequency` | Exponential | 9,618 | -3.64–9 | 0.99778 ± 0.2054 | 99.48014% | 0% | normalized | 1.01167 / 1.12474 / 1.2378 | 1–1.25 raw / 0.5–0.51562 norm: 9,569 (99.5%)<br>-1–-0.75 raw / 0.375–0.39062 norm: 7 (0.1%)<br>2.5–2.75 raw / 0.59375–0.60938 norm: 6 (0.1%) |
| `random_4_keytrack_tune` | Linear | 9,618 | 0–0 | 0 ± 0 | 100% | 100% | normalized | 0.00156 / 0.01562 / 0.02968 | 0–0.03125 raw / 0.5–0.51562 norm: 9,618 (100.0%) |
| `reverb_chorus_amount` | Quadratic | 9,618 | 0–1 | 0.24577 ± 0.12777 | 80.29736% | 3.46226% | normalized | 0.12812 / 0.23488 / 0.48607 | 0.21651–0.25 raw / 0.04688–0.0625 norm: 7,813 (81.2%)<br>0–0.125 raw / 0–0.01562 norm: 476 (4.9%)<br>0.30619–0.33072 raw / 0.09375–0.10938 norm: 130 (1.4%) |
| `reverb_chorus_frequency` | Exponential | 9,618 | -8–3 | -2.38729 ± 1.37395 | 83.20857% | 0.03119% | normalized | -6.15208 / -2.08032 / -1.98771 | -2.15625–-1.98438 raw / 0.53125–0.54688 norm: 8,032 (83.5%)<br>-8–-7.82812 raw / 0–0.01562 norm: 170 (1.8%)<br>-6.28125–-6.10938 raw / 0.15625–0.17188 norm: 65 (0.7%) |
| `reverb_decay_time` | Exponential | 9,618 | -6–6 | 0.21033 ± 1.70404 | 46.51695% | 46.51695% | normalized | -3.17414 / 0.12792 / 2.82004 | 0–0.1875 raw / 0.5–0.51562 norm: 4,571 (47.5%)<br>0.9375–1.125 raw / 0.57812–0.59375 norm: 268 (2.8%)<br>1.5–1.6875 raw / 0.625–0.64062 norm: 249 (2.6%) |
| `reverb_delay` | Linear | 9,618 | 0–0.3 | 0.01054 ± 0.03547 | 83.34373% | 83.34373% | normalized | 0.0002791 / 0.00279 / 0.06819 | 0–0.00469 raw / 0–0.01562 norm: 8,077 (84.0%)<br>0.02344–0.02813 raw / 0.07812–0.09375 norm: 124 (1.3%)<br>0.01875–0.02344 raw / 0.0625–0.07812 norm: 107 (1.1%) |
| `reverb_dry_wet` | Linear | 9,618 | 0–1 | 0.27116 ± 0.18278 | 37.64816% | 10.13724% | normalized | 0.00722 / 0.25802 / 0.62161 | 0.25–0.26562 raw / 0.25–0.26562 norm: 3,736 (38.8%)<br>0–0.01562 raw / 0–0.01562 norm: 1,040 (10.8%)<br>0.1875–0.20312 raw / 0.1875–0.20312 norm: 192 (2.0%) |
| `reverb_high_shelf_cutoff` | Linear | 9,618 | 0–128 | 94.67837 ± 12.90685 | 66.10522% | 0.17675% | normalized | 87.3663 / 91.28644 / 126.13598 | 90–92 raw / 0.70312–0.71875 norm: 6,490 (67.5%)<br>126–128 raw / 0.98438–1 norm: 517 (5.4%)<br>100–102 raw / 0.78125–0.79688 norm: 162 (1.7%) |
| `reverb_high_shelf_gain` | Linear | 9,618 | -6–0 | -1.5592 ± 1.51642 | 66.04284% | 6.46704% | normalized | -5.93011 / -0.99264 / -0.07193 | -1.03125–-0.9375 raw / 0.82812–0.84375 norm: 6,405 (66.6%)<br>-6–-5.90625 raw / 0–0.01562 norm: 645 (6.7%)<br>-0.09375–0 raw / 0.98438–1 norm: 628 (6.5%) |
| `reverb_low_shelf_cutoff` | Linear | 9,618 | 0–128 | 13.89161 ± 20.88844 | 61.64483% | 61.64483% | normalized | 0.16149 / 1.61495 / 55.35375 | 0–2 raw / 0–0.01562 norm: 5,955 (61.9%)<br>40–42 raw / 0.3125–0.32812 norm: 180 (1.9%)<br>38–40 raw / 0.29688–0.3125 norm: 165 (1.7%) |
| `reverb_low_shelf_gain` | Linear | 9,618 | -6–0 | -1.78714 ± 2.52194 | 61.91516% | 61.91516% | normalized | -5.977 / -0.07559 / -0.00757 | -0.09375–0 raw / 0.98438–1 norm: 5,965 (62.0%)<br>-6–-5.90625 raw / 0–0.01562 norm: 1,960 (20.4%)<br>-4.6875–-4.59375 raw / 0.21875–0.23438 norm: 54 (0.6%) |
| `reverb_pre_high_cutoff` | Linear | 9,618 | 0–128 | 108.16014 ± 12.30263 | 84.16511% | 0.32231% | normalized | 91.57925 / 110.96168 / 116.05349 | 110–112 raw / 0.85938–0.875 norm: 8,141 (84.6%)<br>126–128 raw / 0.98438–1 norm: 329 (3.4%)<br>106–108 raw / 0.82812–0.84375 norm: 62 (0.6%) |
| `reverb_pre_low_cutoff` | Linear | 9,618 | 0–128 | 12.02342 ± 23.90945 | 73.9031% | 73.9031% | normalized | 0.1346 / 1.34598 / 65.49683 | 0–2 raw / 0–0.01562 norm: 7,145 (74.3%)<br>50–52 raw / 0.39062–0.40625 norm: 95 (1.0%)<br>32–34 raw / 0.25–0.26562 norm: 90 (0.9%) |
| `reverb_size` | Linear | 9,618 | 0–1 | 0.50984 ± 0.18276 | 60.10605% | 2.00665% | normalized | 0.17116 / 0.50819 / 0.88529 | 0.5–0.51562 raw / 0.5–0.51562 norm: 5,832 (60.6%)<br>0.98438–1 raw / 0.98438–1 norm: 385 (4.0%)<br>0–0.01562 raw / 0–0.01562 norm: 219 (2.3%) |
| `sample_level` | Quadratic | 9,618 | 0–1 | 0.47735 ± 0.31903 | 55.42732% | 18.94365% | normalized | 0.05421 / 0.69793 / 0.70694 | 0.69597–0.70711 raw / 0.48438–0.5 norm: 5,346 (55.6%)<br>0–0.125 raw / 0–0.01562 norm: 2,557 (26.6%)<br>0.125–0.17678 raw / 0.01562–0.03125 norm: 279 (2.9%) |
| `sample_pan` | Linear | 9,618 | -1–1 | -0.00151 ± 0.05353 | 98.66916% | 98.66916% | normalized | 0.00137 / 0.01557 / 0.02978 | 0–0.03125 raw / 0.5–0.51562 norm: 9,518 (99.0%)<br>-1–-0.96875 raw / 0–0.01562 norm: 12 (0.1%)<br>-0.03125–0 raw / 0.48438–0.5 norm: 11 (0.1%) |
| `sample_tune` | Linear | 9,618 | -1–1 | 0.00116 ± 0.05721 | 98.74194% | 98.74194% | normalized | 0.00143 / 0.01565 / 0.02987 | 0–0.03125 raw / 0.5–0.51562 norm: 9,510 (98.9%)<br>0.96875–1 raw / 0.98438–1 norm: 10 (0.1%)<br>-0.03125–0 raw / 0.48438–0.5 norm: 9 (0.1%) |
| `stereo_routing` | Linear | 9,618 | 0–1 | 0.95832 ± 0.16576 | 91.38074% | 1.50759% | normalized | 0.65505 / 0.99147 / 0.99915 | 0.98438–1 raw / 0.98438–1 norm: 8,812 (91.6%)<br>0–0.01562 raw / 0–0.01562 norm: 160 (1.7%)<br>0.5–0.51562 raw / 0.5–0.51562 norm: 49 (0.5%) |
| `velocity_track` | Linear | 9,618 | -1–1 | 0.03604 ± 0.17462 | 93.51216% | 93.51216% | normalized | 0.00144 / 0.01645 / 0.25211 | 0–0.03125 raw / 0.5–0.51562 norm: 9,006 (93.6%)<br>0.96875–1 raw / 0.98438–1 norm: 200 (2.1%)<br>0.4375–0.46875 raw / 0.71875–0.73438 norm: 30 (0.3%) |
| `voice_amplitude` | Linear | 9,618 | 1–1 | 1 ± 0 | 100% | 0% | normalized | 0.98516 / 0.99219 / 0.99922 | 0.98438–1 raw / 0.98438–1 norm: 9,618 (100.0%) |
| `voice_tune` | Linear | 9,618 | -1–1 | 0.0002578 ± 0.064 | 98.79393% | 98.79393% | normalized | 0.00139 / 0.01562 / 0.02985 | 0–0.03125 raw / 0.5–0.51562 norm: 9,505 (98.8%)<br>0.96875–1 raw / 0.98438–1 norm: 14 (0.1%)<br>-0.34375–-0.3125 raw / 0.32812–0.34375 norm: 6 (0.1%) |
| `volume` | SquareRoot | 9,618 | 0–7399 | 5466 ± 709.25655 | 55.48971% | 0.40549% | normalized | 4396 / 5563 / 6526 | 5465–5665 raw / 0.85938–0.875 norm: 5,606 (58.3%)<br>5869–6077 raw / 0.89062–0.90625 norm: 395 (4.1%)<br>6077–6288 raw / 0.90625–0.92188 norm: 378 (3.9%) |

## Interpretation cautions

- A categorical mode is a storage-value mode, not a claim that the corresponding UI choice is perceptually dominant.
- A continuous dominant bin is a range, not an exact mode. This avoids pretending that arbitrary floating-point values have exact categorical semantics.
- The normalized histogram follows Vital's raw control scale. It is useful for comparing differently ranged controls, while the raw summary preserves the actual serialized values.
- These aggregates contain no preset paths, names, authors, raw wavetable/sample payloads, or per-file records.
