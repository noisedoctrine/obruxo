# The training recordings need correction

Completed 2026-10-22, with no new audio or training. The new basic model predicts the same patches much less accurately when their input recordings use the wider sound-training performances. The sound-training and existing development recordings also contain a perfect association between note pattern and filter enable. Correct these data issues before another long sound fit; keep the 14 outputs.

The [frozen card](INPUT_NOTE_DIAGNOSIS_CARD.md), [plan](data/input_note_diagnosis_plan.json) and [results](data/input_note_diagnosis_summary.json) retain the audit. Plan ID: `ac83224b85e56aef1fce1a34a0d43c5cd05410156a416d28d97223a84a04260c`. Both parents were independently audited under the same corrected renderer and frozen frontend. Every selected feature file and source request/audio identity was checked. The model weights stay unchanged.

## Same patches, different recordings

All 64 source patches are present in both the new initialization replay and the earlier sound-training feature bank. Pass both recordings through the same new model:

| Input recordings | Source-active control error | Unison accuracy | Filter-enable accuracy |
|---|---:|---:|---:|
| Initialization | **4.45%** | **100%** | **100%** |
| Wider sound-training performances | **29.59%** | **42.19%** | **85.94%** |

Initialization uses only MIDI notes 60, 62, 64 and 67. The paired sound-training inputs contain 45 distinct notes spanning MIDI 37–84; **60/64** cases include notes outside the initialization range. Settings error rises 6.65 times. Pitch, duration, note pattern and velocity change together, so this does not isolate pitch as the sole cause. These are parameter diagnostics, not perceptual scores or proof of the whole earlier quality gap.

## A pattern shortcut in the data

The source list alternates filter-off/on, while `index % 4` assigns its input pattern. Consequently:

| Source filter | Held note | Staccato | Melody | Chord |
|---|---:|---:|---:|---:|
| Off: 64-state training panel | 16 | 0 | 16 | 0 |
| On: 64-state training panel | 0 | 16 | 0 | 16 |
| Off: 12-state development panel | 3 | 0 | 3 | 0 |
| On: 12-state development panel | 0 | 3 | 0 | 3 |

A rule using only the note-pattern type predicts the filter setting correctly on **64/64 training cases and 12/12 development cases**, without measuring the filter's sound. The original 32-state subset has the same issue. This does not prove that the network uses the shortcut, but it prevents these panels from establishing that filter inference generalizes across how a patch is played.

## Consequence for the plan

The corrected-render initialization still proves that the small model can learn its original recordings. It does not yet prove that the encoder can infer a stable patch from varied notes. Fix input diversity and category-independent performance assignment before the next costly gradient run:

- Give each training patch multiple input performances, covering a wider pitch range and an informative held note.
- Balance note-pattern types independently of oscillator/filter categories; test that construction directly.
- Let sound training infer from both recorded views of each patch and test cross-note reconstruction, instead of always inferring from its first view.
- Retain the old development panel as a regression comparison and add counterbalanced performances. Fresh confirmation still needs untouched families/performances later.
- Fit category scales after the input distribution and starting weights are fixed. The prepared mathematical check adds no native rendering.

Do not expand heads or relax the sound-quality thresholds. Preserve all historical failures; this diagnosis does not turn an unsuccessful model into a qualified one.

Cost: **43.97 seconds**, **0.62 GiB** peak process-tree RAM, zero native requests and zero optimizer updates, within five-minute / 4-GiB / 64-MiB limits. Two focused tests and Ruff pass; the reported errors were independently recomputed from the saved case rows. No test source was used.
