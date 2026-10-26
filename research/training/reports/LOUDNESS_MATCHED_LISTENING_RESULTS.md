# Correct the loudness-confounded listening condition

The listener submitted all **twelve original responses**, then reported **five comparisons** with severe loudness differences that prevented useful comparison. The [validated feedback record](data/original_level_listener_feedback.json) preserves the response-file identity and this limitation. Affected trial IDs were not supplied. Keep the full original packet/responses; do not remove five guesses or report a tone winner from the unmatched pass.

The replacement uses integrated BS.1770/EBU R128 loudness matching with one constant gain per complete clip. Its -23-LUFS target may be lowered by shared headroom attenuation; the checked true-peak ceiling is -1 dBTP. No compression, limiting, trimming, channel balancing or dynamic envelope processing is applied. Original native recordings, model presets, raw reconstruction gates and the 192-family confirmation reservation remain unchanged.

| Matching sensitivity in this panel | Largest difference |
|---|---:|
| Original reconstruction versus reference integrated loudness | 9.745 LU |
| Relative gain from LUFS versus whole-clip RMS matching | 1.974 dB |
| Relative gain from LUFS versus sample-peak matching | 10.388 dB |

These are maxima across deliberately selected diagnostic cases, not population estimates. LUFS controls weighted, gated overall energy; it does not make envelopes, crest factors or perceived loudness universally identical. Original/processed LUFS, RMS, peaks and gains are retained separately. Absolute level remains a reconstruction dimension.

All **36 exported WAVs** pass the [independent-meter check](data/loudness_matched_listening_meter.json), using pyloudnorm 0.2.0/DeMan separately from FFmpeg 7.1's meter. The largest within-comparison spread is **0.00528 LU**; the largest disagreement between meters is **0.00738 LU**, below the frozen 0.1/0.2-LU limits. Every waveform is an exact float32 round trip of its original multiplied by its recorded constant gain. Frame counts, tails, stereo and original hashes are preserved. Processing metadata is bound into the anonymous assignment identity.

New packet ID: `39f30583088381eed55167ae14458f43adbd9e90177e81b7e76958dec95d4079`. It has fresh private assignments/order, a separate draft key and an explicit integrated-LUFS banner. Original-level tests now require a level-accuracy purpose. Instant replay and answer pills remain; waveform-hash URL parameters prevent reuse of old cached clips. Old responses remain readable. No new human tone judgments have been received for this condition.

The [initial preparation record](data/loudness_matched_listening_initial_check.json) preserves a metadata-path failure after the audio checks passed. A separate read-only completion corrected that path and verified the existing packet without regenerating audio, changing tolerances or overwriting the failure. The [final report](data/loudness_matched_listening_summary.json) records **22.39 aggregate execution seconds / 0.14 GiB peak RAM**, with 49.41 MiB retained before the summary/preview, within ten minutes / 4 GiB / 128 MiB. Fifteen focused Python and five page-behavior tests pass, alongside Ruff/diff checks. [Review dependencies](data/listening_meter_packages.json) are pinned and installed in a separate workspace folder.

There are **zero native renders, inverse inference calls or optimizer updates**. No model is promoted. The completed fit's four-of-six result and failed held-input gates remain unchanged. [Usage and condition definitions](../LISTENING.md).
