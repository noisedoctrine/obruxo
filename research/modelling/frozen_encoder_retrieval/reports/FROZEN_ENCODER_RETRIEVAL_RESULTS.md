# Frozen audio encoder retrieval benchmark

Issue: [#152](https://github.com/noisedoctrine/obruxo/issues/152). This report covers frozen pretrained representations on original Vital source presets and existing MIDI performances. No synthesis controls are predicted, tuned, or optimized.

## Execution status

The frozen validation cohort contains 131 verified original presets, 12 compatible unique MIDI excerpts, 3144 retrieval episodes, and 1572 planned original-preset renders.
Render profile: `vital-1.6.4-vst3-sha256-a622a2c99b4066cd7945a4ab9bbdd698e7632a30702f6f0a7ccbf26a56b576e1-dawdreamer-0.8.3-full-1.5.5-state-verified-seed-bits-v2-expression-midi-v1-curve-input-v1-corpus-readback-v1`. Selected renderer capabilities: `{"channel_pressure": true, "control_changes": [1, 64, 74], "max_channels": 16, "note_off_velocity": true, "notes": true, "pitch_bend": true, "polyphony": true, "tempo_changes": false}`.
Source pairing and MIDI compatibility audit: {"eligible_preset_records": 131, "midi_descriptive_meta_ignored:track_name": 16, "midi_excluded:IndexError:list index out of range": 1, "midi_excluded:ValueError:no_whole_note_fits_excerpt": 1, "midi_excluded:midi.capability.control_change": 1, "midi_excluded:midi.capability.opaque": 3, "midi_excluded:midi.capability.tempo_changes": 2, "midi_excluded:midi.note.unmatched_off": 2, "midi_excluded:midi.note.unmatched_on": 2, "midi_pair_candidates": 20, "midi_parsed": 12, "pair_excluded:pair.missing_midi": 111}. These counts reflect the selected existing split and direct PresetShare preset/MIDI pairing rules; source WAV quality is not an eligibility condition.
The 12 compatible existing MIDI excerpts are cross-rendered on all 131 admitted source presets. These are new Vital renders, not claims of original PresetShare preset/MIDI capture pairs.
The existing source audit counted 9,620 files and 1,153 statically eligible states (131 validation, 904 train, 118 test). This experiment uses validation only; static eligibility is not native render eligibility.
Vital render stage: 1554 QA accepted of 1572 planned requests; this run rendered 0 and reused 1560 cached WAVs in 58.57 seconds.
QA policy `finite_silent_reject_clipping_tail_warn_v2` rejects malformed, silent, or failed renders and records clipping/tail thresholds as warnings. Accepted warning counts: {"amplitude_above_unit_range": 362, "tail_rms_above_warning": 342}; excluded renders: {"render_exception:RuntimeError": 12, "render_qa_silence": 6}.
The comparison uses 3096 identical episodes across the available arms; held-out reserved sources used: **no**.
The gallery retained 129 of 131 states with complete renders across the frozen gallery-MIDI pool; 2 incomplete states were removed once for all arms.

### Render and feature coverage

| Condition | Planned episodes | Render QA qualified | Excluded | Feature qualified by arm | Common across arms |
|---|---:|---:|---:|---|---:|
| cross_midi_common_gallery | 1572 | 1548 | 24 | clap_music:official_clip: 1548, dac_44khz:masked_mean: 1548, dac_44khz:temporal_pyramid_8: 1548 | 1548 |
| cross_midi_mixed_gallery | 1572 | 1548 | 24 | clap_music:official_clip: 1548, dac_44khz:masked_mean: 1548, dac_44khz:temporal_pyramid_8: 1548 | 1548 |
No known multi-state lineage/alias neighbors were present in the render-qualified gallery, so that diagnostic has zero negatives. The corpus snapshot also lacks instrument-category labels; the spectral hard-negative result is only a coarse audio-similarity proxy, not a same-category or human-verified perceptual-neighbor result.
Measured real Vital and frozen encoder execution completed; see results report for exact coverage, cache use, model availability, and limitations.
Controlled perturbation diagnostic: 16 real Vital renders, 16 QA accepted, 2 pretrained encoders; uncertainty is not estimated for this small diagnostic set.

Runtime preflight: Python `3.12.13`; missing encoder runtime packages: `easydict`, `muq`, `x_clip`. Checkpoint root supplied: **yes**. Vital plugin environment variable configured: **no**.
The existing [corpus audit report](../../../training/reports/CORPUS_INPUT_RESULTS.md) records prior static eligibility findings. Those audit counts are not native render eligibility. The separate [PresetShare audit](../../basic_pitch/README.md) has its own pairing rules and counts.

The default command selects the corpus-qualified full-preset expression renderer identity, including the established curve-input, ignored-empty-map, and native scalar readback policies. It supports note/polyphony, pitch-bend, channel-pressure, CC 1/64/74, release-velocity and multiple channels; tempo changes remain unsupported and are counted as exclusions. Unsupported events are rejected rather than silently removed. Musical spans are complete-note excerpts of 2–8 seconds plus a fixed 2-second release tail; model input is mono and 4–10 seconds, downmixed by channel mean and polyphase-resampled. No shared gain normalization is applied; EnCodec retains its own checkpoint-configured RMS normalization when enabled.

Render and feature stages are independent, resumable commands. Vital remains a single-worker in-process renderer; an asynchronous manifest reader feeds a bounded queue of request batches. Render keys include exact preset/MIDI/excerpt/renderer/policy identities. Feature keys include audio hash, checkpoint digest/revision, loader revision, sample rate, preprocessing, extraction tap, pooling, and FP32 precision. Weights are never downloaded by the run commands. Full paths, identifiers, WAVs, caches, checkpoints, and per-episode results belong in ignored local outputs.

No reconstruction head, source-label warm start, shared-model training, direct patch optimization, or audio-loss training is part of this experiment. The recommendation below identifies candidates for the later small 14-output follow-up; retrieval results do not establish reconstruction quality.

## Frozen-control scope

The linked [vital_control_inventory.jsonl](./vital_control_inventory.jsonl) contains one record for every exact control/instance from the existing 979-record Vital control inventory. Main retrieval arms pass original source preset inputs unchanged; every control group is fixed with 0 enabled instances and 0 output values. No controls are omitted from the inventory.

### LAION-CLAP Music HTSAT-base — official_clip

Checkpoint: `lukewys/laion_clap/music_audioset_epoch_15_esc_90.14.pt`; requested sample rate: 48000 Hz; representation: official audio clip embedding; output size: 512-D. Status: **executed**.

| Measure / Vital control family | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Checkpoint provenance / license | SHA-256 `fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd`; source `b3708341862f581175dba5c356a4ebf74a9b6651`; loader `laion_clap==1.1.7`; runtime `{"laion_clap": "1.1.7", "numpy": "1.26.4", "scipy": "1.17.1", "torch": "2.12.1+xpu", "transformers": "4.50.0"}`; CC0-1.0 | — | exact local lock; frozen inference |
| Cross-MIDI common-gallery Recall@1 | 0.7074 [0.6731, 0.7403] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | 0.8889 [0.8663, 0.9096] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | 0.7872 [0.7608, 0.8126] | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | 0.7074 [0.6731, 0.7403] | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | 0.6479 [0.6098, 0.6848] | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | 0.4787 [0.4367, 0.5200] (1548 episodes; 198144 negatives) | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | 0.6848 [0.6499, 0.7203] (1548 episodes; 7740 negatives) | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run (0 episodes; 0 negatives) | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | [129] / 1548 | — | fixed candidate sets across encoders |
| Features created / cache reused | 1554 / 0 | 1554 | resumable extraction pass; cache identity checked |
| Extraction pass time / peak RSS / accelerator memory | 149.881 s / 3835752448 / 851259904 bytes | — | local inference and cache validation; no training |
| Global settings ❌ 0 / 610 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Oscillators and voices ❌ 0 / 88 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Filters ❌ 0 / 53 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Envelopes ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| LFOs and modulation ❌ 0 / 170 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Effects ❌ 0 / 46 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Stratum: duration_gap_band = 0-1s | Recall@1 0.6534; Recall@5 0.8527; MRR 0.7427 | 903 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: duration_gap_band = 1-3s | Recall@1 0.7829; Recall@5 0.9395; MRR 0.8495 | 645 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: pitch_register_gap_band = 0-5 semitones | Recall@1 0.9070; Recall@5 0.9767; MRR 0.9371 | 387 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: pitch_register_gap_band = 13+ semitones | Recall@1 0.5714; Recall@5 0.8250; MRR 0.6839 | 903 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: pitch_register_gap_band = 6-12 semitones | Recall@1 0.8837; Recall@5 0.9806; MRR 0.9241 | 258 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_articulation = held | Recall@1 0.6076; Recall@5 0.8421; MRR 0.7107 | 1032 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_articulation = phrased | Recall@1 0.9070; Recall@5 0.9826; MRR 0.9403 | 516 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_duration_band = 2-4s | Recall@1 0.9380; Recall@5 0.9845; MRR 0.9584 | 129 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_duration_band = 6-8s | Recall@1 0.6864; Recall@5 0.8802; MRR 0.7717 | 1419 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_expression = plain | Recall@1 0.7074; Recall@5 0.8889; MRR 0.7872 | 1548 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_note_density_band = dense | Recall@1 0.7390; Recall@5 0.9199; MRR 0.8151 | 387 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_note_density_band = moderate | Recall@1 0.6645; Recall@5 0.8560; MRR 0.7513 | 903 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_note_density_band = sparse | Recall@1 0.8101; Recall@5 0.9574; MRR 0.8710 | 258 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_polyphony = monophonic | Recall@1 0.7829; Recall@5 0.9380; MRR 0.8478 | 516 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_polyphony = polyphonic | Recall@1 0.6696; Recall@5 0.8643; MRR 0.7569 | 1032 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: source_preset_lineage_group_size = single-state | Recall@1 0.7074; Recall@5 0.8889; MRR 0.7872 | 1548 episodes | frozen encoder; family bootstrap not stratified |
### MuQ-MuLan — official_clip

Checkpoint: `OpenMuQ/MuQ-MuLan-large/pytorch_model.bin`; requested sample rate: 24000 Hz; representation: official audio embedding; output size: 512-D. Status: **unavailable / not run**.

| Measure / Vital control family | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Extraction pass time / peak RSS / accelerator memory | not run s / not measured / not measured bytes | — | local inference and cache validation; no training |
| Global settings ❌ 0 / 610 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Oscillators and voices ❌ 0 / 88 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Filters ❌ 0 / 53 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Envelopes ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| LFOs and modulation ❌ 0 / 170 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Effects ❌ 0 / 46 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |

Unavailable: no verified local checkpoint lock entry; missing Python packages: muq, easydict, x_clip.

### CIR single-timbre AST — official_clip

Checkpoint: `Google Drive: ICASSP26_inst-direct-with-data-aug_final.pt`; requested sample rate: 16000 Hz; representation: mean of final CLS and distillation tokens; output size: 768-D. Status: **unavailable / not run**.

| Measure / Vital control family | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Extraction pass time / peak RSS / accelerator memory | not run s / not measured / not measured bytes | — | local inference and cache validation; no training |
| Global settings ❌ 0 / 610 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Oscillators and voices ❌ 0 / 88 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Filters ❌ 0 / 53 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Envelopes ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| LFOs and modulation ❌ 0 / 170 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Effects ❌ 0 / 46 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |

Unavailable: no verified local checkpoint lock entry.

### BEATs iter3+ AS2M — masked_mean

Checkpoint: `lpepino/beats_ckpts/BEATs_iter3_plus_AS2M.pt`; requested sample rate: 16000 Hz; representation: pretrained iter3+ final pre-predictor features; output size: 768-D. Status: **unavailable / not run**.

| Measure / Vital control family | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Extraction pass time / peak RSS / accelerator memory | not run s / not measured / not measured bytes | — | local inference and cache validation; no training |
| Global settings ❌ 0 / 610 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Oscillators and voices ❌ 0 / 88 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Filters ❌ 0 / 53 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Envelopes ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| LFOs and modulation ❌ 0 / 170 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Effects ❌ 0 / 46 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |

Unavailable: no verified local checkpoint lock entry.

### BEATs iter3+ AS2M — temporal_pyramid_8

Checkpoint: `lpepino/beats_ckpts/BEATs_iter3_plus_AS2M.pt`; requested sample rate: 16000 Hz; representation: pretrained iter3+ final pre-predictor features; output size: 12288-D. Status: **unavailable / not run**.

| Measure / Vital control family | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Extraction pass time / peak RSS / accelerator memory | not run s / not measured / not measured bytes | — | local inference and cache validation; no training |
| Global settings ❌ 0 / 610 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Oscillators and voices ❌ 0 / 88 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Filters ❌ 0 / 53 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Envelopes ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| LFOs and modulation ❌ 0 / 170 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Effects ❌ 0 / 46 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |

Unavailable: no verified local checkpoint lock entry.

### Meta EnCodec 24 kHz — masked_mean

Checkpoint: `facebook/encodec_24khz`; requested sample rate: 24000 Hz; representation: continuous encoder output before RVQ; output size: 128-D. Status: **unavailable / not run**.

| Measure / Vital control family | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Extraction pass time / peak RSS / accelerator memory | not run s / not measured / not measured bytes | — | local inference and cache validation; no training |
| Global settings ❌ 0 / 610 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Oscillators and voices ❌ 0 / 88 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Filters ❌ 0 / 53 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Envelopes ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| LFOs and modulation ❌ 0 / 170 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Effects ❌ 0 / 46 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |

Unavailable: no verified local checkpoint lock entry.

### Meta EnCodec 24 kHz — temporal_pyramid_8

Checkpoint: `facebook/encodec_24khz`; requested sample rate: 24000 Hz; representation: continuous encoder output before RVQ; output size: 2048-D. Status: **unavailable / not run**.

| Measure / Vital control family | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Extraction pass time / peak RSS / accelerator memory | not run s / not measured / not measured bytes | — | local inference and cache validation; no training |
| Global settings ❌ 0 / 610 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Oscillators and voices ❌ 0 / 88 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Filters ❌ 0 / 53 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Envelopes ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| LFOs and modulation ❌ 0 / 170 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Effects ❌ 0 / 46 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |

Unavailable: no verified local checkpoint lock entry.

### Descript Audio Codec 44.1 kHz — masked_mean

Checkpoint: `descript/dac_44khz`; requested sample rate: 44100 Hz; representation: continuous encoder output before RVQ; output size: 1024-D. Status: **executed**.

| Measure / Vital control family | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Checkpoint provenance / license | SHA-256 `128180a5e4d41ceabfeb952b687fb9da7cf468c2291acf5be9cadb5ab091cc46`; source `c1bc521685adf9cfe247bc39a5ca58917eda1ac4`; loader `transformers==4.50.0`; runtime `{"numpy": "1.26.4", "scipy": "1.17.1", "torch": "2.12.1+xpu", "transformers": "4.50.0"}`; MIT | — | exact local lock; frozen inference |
| Cross-MIDI common-gallery Recall@1 | 0.3140 [0.2733, 0.3559] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | 0.5349 [0.4884, 0.5807] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | 0.4210 [0.3817, 0.4605] | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | 0.3140 [0.2733, 0.3559] | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | 0.3482 [0.3075, 0.3895] | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | 0.2894 [0.2487, 0.3301] (1548 episodes; 198144 negatives) | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | 0.4729 [0.4283, 0.5155] (1548 episodes; 7740 negatives) | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run (0 episodes; 0 negatives) | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | [129] / 1548 | — | fixed candidate sets across encoders |
| Features created / cache reused | 432 / 1122 | 1554 | resumable extraction pass; cache identity checked |
| Extraction pass time / peak RSS / accelerator memory | 407.344 s / 2888515584 / 990163968 bytes | — | local inference and cache validation; no training |
| Resumable pass scope | 432 new and 1122 cache-reused features | 1554 | elapsed time is this pass, not cumulative across earlier partial runs |
| Global settings ❌ 0 / 610 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Oscillators and voices ❌ 0 / 88 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Filters ❌ 0 / 53 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Envelopes ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| LFOs and modulation ❌ 0 / 170 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Effects ❌ 0 / 46 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Stratum: duration_gap_band = 0-1s | Recall@1 0.2924; Recall@5 0.5172; MRR 0.4002 | 903 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: duration_gap_band = 1-3s | Recall@1 0.3442; Recall@5 0.5597; MRR 0.4501 | 645 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: pitch_register_gap_band = 0-5 semitones | Recall@1 0.3230; Recall@5 0.5659; MRR 0.4371 | 387 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: pitch_register_gap_band = 13+ semitones | Recall@1 0.2824; Recall@5 0.4795; MRR 0.3842 | 903 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: pitch_register_gap_band = 6-12 semitones | Recall@1 0.4109; Recall@5 0.6822; MRR 0.5257 | 258 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_articulation = held | Recall@1 0.2984; Recall@5 0.5078; MRR 0.4034 | 1032 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_articulation = phrased | Recall@1 0.3450; Recall@5 0.5891; MRR 0.4562 | 516 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_duration_band = 2-4s | Recall@1 0.2403; Recall@5 0.4884; MRR 0.3569 | 129 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_duration_band = 6-8s | Recall@1 0.3206; Recall@5 0.5391; MRR 0.4268 | 1419 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_expression = plain | Recall@1 0.3140; Recall@5 0.5349; MRR 0.4210 | 1548 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_note_density_band = dense | Recall@1 0.4341; Recall@5 0.6460; MRR 0.5408 | 387 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_note_density_band = moderate | Recall@1 0.2680; Recall@5 0.4862; MRR 0.3743 | 903 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_note_density_band = sparse | Recall@1 0.2946; Recall@5 0.5388; MRR 0.4045 | 258 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_polyphony = monophonic | Recall@1 0.2403; Recall@5 0.5039; MRR 0.3619 | 516 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_polyphony = polyphonic | Recall@1 0.3508; Recall@5 0.5504; MRR 0.4506 | 1032 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: source_preset_lineage_group_size = single-state | Recall@1 0.3140; Recall@5 0.5349; MRR 0.4210 | 1548 episodes | frozen encoder; family bootstrap not stratified |
### Descript Audio Codec 44.1 kHz — temporal_pyramid_8

Checkpoint: `descript/dac_44khz`; requested sample rate: 44100 Hz; representation: continuous encoder output before RVQ; output size: 16384-D. Status: **executed**.

| Measure / Vital control family | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Checkpoint provenance / license | SHA-256 `128180a5e4d41ceabfeb952b687fb9da7cf468c2291acf5be9cadb5ab091cc46`; source `c1bc521685adf9cfe247bc39a5ca58917eda1ac4`; loader `transformers==4.50.0`; runtime `{"numpy": "1.26.4", "scipy": "1.17.1", "torch": "2.12.1+xpu", "transformers": "4.50.0"}`; MIT | — | exact local lock; frozen inference |
| Cross-MIDI common-gallery Recall@1 | 0.3663 [0.3211, 0.4109] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | 0.6021 [0.5562, 0.6486] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | 0.4744 [0.4323, 0.5156] | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | 0.3663 [0.3211, 0.4109] | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | 0.3882 [0.3411, 0.4283] | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | 0.2694 [0.2235, 0.3133] (1548 episodes; 198144 negatives) | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | 0.4554 [0.4057, 0.5032] (1548 episodes; 7740 negatives) | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run (0 episodes; 0 negatives) | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | [129] / 1548 | — | fixed candidate sets across encoders |
| Features created / cache reused | 432 / 1122 | 1554 | resumable extraction pass; cache identity checked |
| Extraction pass time / peak RSS / accelerator memory | 407.344 s / 2888515584 / 990163968 bytes | — | local inference and cache validation; no training |
| Resumable pass scope | 432 new and 1122 cache-reused features | 1554 | elapsed time is this pass, not cumulative across earlier partial runs |
| Global settings ❌ 0 / 610 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Oscillators and voices ❌ 0 / 88 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Filters ❌ 0 / 53 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Envelopes ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| LFOs and modulation ❌ 0 / 170 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Effects ❌ 0 / 46 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Stratum: duration_gap_band = 0-1s | Recall@1 0.3444; Recall@5 0.5703; MRR 0.4508 | 903 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: duration_gap_band = 1-3s | Recall@1 0.3969; Recall@5 0.6465; MRR 0.5074 | 645 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: pitch_register_gap_band = 0-5 semitones | Recall@1 0.3721; Recall@5 0.6227; MRR 0.4852 | 387 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: pitch_register_gap_band = 13+ semitones | Recall@1 0.3245; Recall@5 0.5670; MRR 0.4359 | 903 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: pitch_register_gap_band = 6-12 semitones | Recall@1 0.5039; Recall@5 0.6938; MRR 0.5927 | 258 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_articulation = held | Recall@1 0.3527; Recall@5 0.5843; MRR 0.4594 | 1032 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_articulation = phrased | Recall@1 0.3934; Recall@5 0.6376; MRR 0.5043 | 516 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_duration_band = 2-4s | Recall@1 0.2868; Recall@5 0.5814; MRR 0.4104 | 129 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_duration_band = 6-8s | Recall@1 0.3735; Recall@5 0.6039; MRR 0.4802 | 1419 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_expression = plain | Recall@1 0.3663; Recall@5 0.6021; MRR 0.4744 | 1548 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_note_density_band = dense | Recall@1 0.5039; Recall@5 0.7442; MRR 0.6084 | 387 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_note_density_band = moderate | Recall@1 0.3068; Recall@5 0.5526; MRR 0.4203 | 903 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_note_density_band = sparse | Recall@1 0.3682; Recall@5 0.5620; MRR 0.4625 | 258 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_polyphony = monophonic | Recall@1 0.2965; Recall@5 0.5368; MRR 0.4108 | 516 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: query_polyphony = polyphonic | Recall@1 0.4012; Recall@5 0.6347; MRR 0.5062 | 1032 episodes | frozen encoder; family bootstrap not stratified |
| Stratum: source_preset_lineage_group_size = single-state | Recall@1 0.3663; Recall@5 0.6021; MRR 0.4744 | 1548 episodes | frozen encoder; family bootstrap not stratified |
## Comparative findings

| Executed arm | Common-gallery R@1 (family 95% CI) | MRR (family 95% CI) | Mixed-gallery R@1 | Acoustic hard-negative top-5 win |
|---|---:|---:|---:|---:|
| clap_music:official_clip | 0.7074 [0.6731, 0.7403] | 0.7872 [0.7608, 0.8126] | 0.6479 | 0.6848 (1548 queries) |
| dac_44khz:temporal_pyramid_8 | 0.3663 [0.3211, 0.4109] | 0.4744 [0.4323, 0.5156] | 0.3882 | 0.4554 (1548 queries) |
| dac_44khz:masked_mean | 0.3140 [0.2733, 0.3559] | 0.4210 [0.3817, 0.4605] | 0.3482 | 0.4729 (1548 queries) |
`clap_music:official_clip` led on common-gallery Recall@1 (0.7074) with paired family intervals above zero; recommend it as the lead for a later small 14-output/reconstruction follow-up. Keep `dac_44khz:temporal_pyramid_8` as a secondary comparator only if a second arm is useful. paired ΔR@1 versus `dac_44khz:temporal_pyramid_8`: +0.3411 [+0.2939, +0.3889]; paired ΔR@1 versus `dac_44khz:masked_mean`: +0.3934 [+0.3482, +0.4360].
These findings cover only the checkpointed arms, this selected validation cohort, and the stated retrieval metrics. They do not establish reconstruction quality or rank unavailable encoders.
Descriptive CLAP strata point to wider register gaps, held notes, and longer clips as harder cases: CLAP R@1 was 0.9070 at 0–5 semitones of register gap and 0.5714 at 13+ semitones; 0.6076 on held versus 0.9070 on phrased queries; 0.9380 on 2–4s versus 0.6864 on 6–8s queries. These are not separately powered experiments.
Within DAC, the temporal pyramid exceeded masked mean pooling by 0.0523 Recall@1 (paired family 95% interval [0.0265, 0.0801]).
The selected MIDI set contains no supported expressive-event queries, so main-benchmark sensitivity to sustain, pitch bend, channel pressure, and CC modulation remains unmeasured.

## Controlled preset perturbations

This secondary diagnostic uses one original validation preset, three single-control perturbations, and 4 real MIDI excerpts selected to cover low monophonic, high polyphonic, and longest-note monophonic performances, plus a nonconstant-velocity excerpt where available. Each query variant is matched against the four-version gallery rendered on another MIDI. The four-candidate Recall@5 is necessarily 1.0 when all features exist and is shown only for consistency; this set is too small for confidence intervals or general encoder recommendations.
MIDI excerpt properties: polyphony 1, median pitch 48, clip 7.76s, median note 0.34s, velocity 100–100, phrased; polyphony 2, median pitch 74, clip 8.00s, median note 0.25s, velocity 100–100, held; polyphony 1, median pitch 77, clip 7.97s, median note 0.19s, velocity 100–100, held; polyphony 4, median pitch 55, clip 8.00s, median note 0.50s, velocity 45–108, held. The original source carried 5 inherited Vital schema diagnostics ({"vital.parameter.unknown": 5}); the three variants introduced 0 new diagnostics. Velocity ranges are recorded, but response to velocity was not separately isolated. The diagnostic measures representation similarity only and has no human audibility threshold. The perturbed route source is `env_2`; this does not test slow-LFO cycle or phase observability.
The linked [vital_control_inventory_controlled.jsonl](./vital_control_inventory_controlled.jsonl) records every exact Vital control/instance for each version (979 rows per version). No model predicts or optimizes these controls; each perturbation is manually specified, with every other source value held fixed.

### Original source preset

| Measure / Vital control family | Result or coverage | Output count | Update method / training signal |
|---|---:|---:|---|
| Applied Vital field | source preset unchanged | 0 | manual perturbation; no training, labels, gradients, or loss |
| LAION-CLAP Music HTSAT-base — official_clip cross-MIDI variant retrieval | R@1 0.2500; R@5 1.0000; MRR 0.5417 (4 queries) | 512-D | frozen backbone; cosine retrieval |
| LAION-CLAP Music HTSAT-base — official_clip same-MIDI control | R@1 1.0000; MRR 1.0000 | 512-D | same performance; controlled versions are hard negatives |
| LAION-CLAP Music HTSAT-base — official_clip exact-version cross-MIDI cosine | 0.7548 | 512-D | pooled frozen feature; cosine similarity |
| LAION-CLAP Music HTSAT-base — official_clip extraction time / peak RSS / accelerator memory | 3.689 s / 4342087680 / 851259904 bytes | — | local inference only; no training |
| Descript Audio Codec 44.1 kHz — masked_mean cross-MIDI variant retrieval | R@1 0.5000; R@5 1.0000; MRR 0.7500 (4 queries) | 1024-D | frozen backbone; cosine retrieval |
| Descript Audio Codec 44.1 kHz — masked_mean same-MIDI control | R@1 1.0000; MRR 1.0000 | 1024-D | same performance; controlled versions are hard negatives |
| Descript Audio Codec 44.1 kHz — masked_mean exact-version cross-MIDI cosine | 0.8734 | 1024-D | pooled frozen feature; cosine similarity |
| Descript Audio Codec 44.1 kHz — masked_mean extraction time / peak RSS / accelerator memory | 18.429 s / 5699436544 / 989552128 bytes | — | local inference only; no training |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 cross-MIDI variant retrieval | R@1 0.0000; R@5 1.0000; MRR 0.4167 (4 queries) | 16384-D | frozen backbone; cosine retrieval |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 same-MIDI control | R@1 1.0000; MRR 1.0000 | 16384-D | same performance; controlled versions are hard negatives |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 exact-version cross-MIDI cosine | 0.9399 | 16384-D | pooled frozen feature; cosine similarity |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 extraction time / peak RSS / accelerator memory | 18.429 s / 5699436544 / 989552128 bytes | — | local inference only; no training |
| Global settings ❌ 0 / 610 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Oscillators and voices ❌ 0 / 88 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Filters ❌ 0 / 53 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Envelopes ❌ 0 / 6 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| LFOs and modulation ❌ 0 / 170 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Effects ❌ 0 / 46 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |

### Filter cutoff perturbation

| Measure / Vital control family | Result or coverage | Output count | Update method / training signal |
|---|---:|---:|---|
| Applied Vital field | filter_1_cutoff: 8 → 33.6 raw (0.000 → 0.200 normalized) | 0 | manual perturbation; no training, labels, gradients, or loss |
| LAION-CLAP Music HTSAT-base — official_clip cross-MIDI variant retrieval | R@1 0.5000; R@5 1.0000; MRR 0.7083 (4 queries) | 512-D | frozen backbone; cosine retrieval |
| LAION-CLAP Music HTSAT-base — official_clip same-MIDI control | R@1 1.0000; MRR 1.0000 | 512-D | same performance; controlled versions are hard negatives |
| LAION-CLAP Music HTSAT-base — official_clip exact-version cross-MIDI cosine | 0.7845 | 512-D | pooled frozen feature; cosine similarity |
| LAION-CLAP Music HTSAT-base — official_clip extraction time / peak RSS / accelerator memory | 3.689 s / 4342087680 / 851259904 bytes | — | local inference only; no training |
| Descript Audio Codec 44.1 kHz — masked_mean cross-MIDI variant retrieval | R@1 0.5000; R@5 1.0000; MRR 0.6250 (4 queries) | 1024-D | frozen backbone; cosine retrieval |
| Descript Audio Codec 44.1 kHz — masked_mean same-MIDI control | R@1 1.0000; MRR 1.0000 | 1024-D | same performance; controlled versions are hard negatives |
| Descript Audio Codec 44.1 kHz — masked_mean exact-version cross-MIDI cosine | 0.8175 | 1024-D | pooled frozen feature; cosine similarity |
| Descript Audio Codec 44.1 kHz — masked_mean extraction time / peak RSS / accelerator memory | 18.429 s / 5699436544 / 989552128 bytes | — | local inference only; no training |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 cross-MIDI variant retrieval | R@1 0.7500; R@5 1.0000; MRR 0.8125 (4 queries) | 16384-D | frozen backbone; cosine retrieval |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 same-MIDI control | R@1 1.0000; MRR 1.0000 | 16384-D | same performance; controlled versions are hard negatives |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 exact-version cross-MIDI cosine | 0.9411 | 16384-D | pooled frozen feature; cosine similarity |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 extraction time / peak RSS / accelerator memory | 18.429 s / 5699436544 / 989552128 bytes | — | local inference only; no training |
| Global settings ❌ 0 / 610 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Oscillators and voices ❌ 0 / 88 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Filters ✅ 1 / 53 | fixed source except the listed field | 0 | manual perturbation; no model output; no supervision or loss |
| Envelopes ❌ 0 / 6 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| LFOs and modulation ❌ 0 / 170 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Effects ❌ 0 / 46 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |

### Amplitude envelope attack perturbation

| Measure / Vital control family | Result or coverage | Output count | Update method / training signal |
|---|---:|---:|---|
| Applied Vital field | env_1_attack: 0 → 0.475684 raw (0.000 → 0.200 normalized) | 0 | manual perturbation; no training, labels, gradients, or loss |
| LAION-CLAP Music HTSAT-base — official_clip cross-MIDI variant retrieval | R@1 0.7500; R@5 1.0000; MRR 0.8333 (4 queries) | 512-D | frozen backbone; cosine retrieval |
| LAION-CLAP Music HTSAT-base — official_clip same-MIDI control | R@1 1.0000; MRR 1.0000 | 512-D | same performance; controlled versions are hard negatives |
| LAION-CLAP Music HTSAT-base — official_clip exact-version cross-MIDI cosine | 0.7472 | 512-D | pooled frozen feature; cosine similarity |
| LAION-CLAP Music HTSAT-base — official_clip extraction time / peak RSS / accelerator memory | 3.689 s / 4342087680 / 851259904 bytes | — | local inference only; no training |
| Descript Audio Codec 44.1 kHz — masked_mean cross-MIDI variant retrieval | R@1 0.7500; R@5 1.0000; MRR 0.8750 (4 queries) | 1024-D | frozen backbone; cosine retrieval |
| Descript Audio Codec 44.1 kHz — masked_mean same-MIDI control | R@1 1.0000; MRR 1.0000 | 1024-D | same performance; controlled versions are hard negatives |
| Descript Audio Codec 44.1 kHz — masked_mean exact-version cross-MIDI cosine | 0.8851 | 1024-D | pooled frozen feature; cosine similarity |
| Descript Audio Codec 44.1 kHz — masked_mean extraction time / peak RSS / accelerator memory | 18.429 s / 5699436544 / 989552128 bytes | — | local inference only; no training |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 cross-MIDI variant retrieval | R@1 0.7500; R@5 1.0000; MRR 0.8125 (4 queries) | 16384-D | frozen backbone; cosine retrieval |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 same-MIDI control | R@1 1.0000; MRR 1.0000 | 16384-D | same performance; controlled versions are hard negatives |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 exact-version cross-MIDI cosine | 0.9430 | 16384-D | pooled frozen feature; cosine similarity |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 extraction time / peak RSS / accelerator memory | 18.429 s / 5699436544 / 989552128 bytes | — | local inference only; no training |
| Global settings ✅ 1 / 610 | fixed source except the listed field | 0 | manual perturbation; no model output; no supervision or loss |
| Oscillators and voices ❌ 0 / 88 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Filters ❌ 0 / 53 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Envelopes ❌ 0 / 6 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| LFOs and modulation ❌ 0 / 170 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Effects ❌ 0 / 46 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |

### Live modulation amount perturbation

| Measure / Vital control family | Result or coverage | Output count | Update method / training signal |
|---|---:|---:|---|
| Applied Vital field | modulation_1_amount: 1 → 0.5 raw (1.000 → 0.750 normalized); env_2 → osc_1_level | 0 | manual perturbation; no training, labels, gradients, or loss |
| LAION-CLAP Music HTSAT-base — official_clip cross-MIDI variant retrieval | R@1 0.5000; R@5 1.0000; MRR 0.7083 (4 queries) | 512-D | frozen backbone; cosine retrieval |
| LAION-CLAP Music HTSAT-base — official_clip same-MIDI control | R@1 1.0000; MRR 1.0000 | 512-D | same performance; controlled versions are hard negatives |
| LAION-CLAP Music HTSAT-base — official_clip exact-version cross-MIDI cosine | 0.7759 | 512-D | pooled frozen feature; cosine similarity |
| LAION-CLAP Music HTSAT-base — official_clip extraction time / peak RSS / accelerator memory | 3.689 s / 4342087680 / 851259904 bytes | — | local inference only; no training |
| Descript Audio Codec 44.1 kHz — masked_mean cross-MIDI variant retrieval | R@1 0.0000; R@5 1.0000; MRR 0.3333 (4 queries) | 1024-D | frozen backbone; cosine retrieval |
| Descript Audio Codec 44.1 kHz — masked_mean same-MIDI control | R@1 1.0000; MRR 1.0000 | 1024-D | same performance; controlled versions are hard negatives |
| Descript Audio Codec 44.1 kHz — masked_mean exact-version cross-MIDI cosine | 0.8573 | 1024-D | pooled frozen feature; cosine similarity |
| Descript Audio Codec 44.1 kHz — masked_mean extraction time / peak RSS / accelerator memory | 18.429 s / 5699436544 / 989552128 bytes | — | local inference only; no training |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 cross-MIDI variant retrieval | R@1 0.0000; R@5 1.0000; MRR 0.3750 (4 queries) | 16384-D | frozen backbone; cosine retrieval |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 same-MIDI control | R@1 1.0000; MRR 1.0000 | 16384-D | same performance; controlled versions are hard negatives |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 exact-version cross-MIDI cosine | 0.9379 | 16384-D | pooled frozen feature; cosine similarity |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 extraction time / peak RSS / accelerator memory | 18.429 s / 5699436544 / 989552128 bytes | — | local inference only; no training |
| Global settings ✅ 1 / 610 | fixed source except the listed field | 0 | manual perturbation; no model output; no supervision or loss |
| Oscillators and voices ❌ 0 / 88 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Filters ❌ 0 / 53 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Envelopes ❌ 0 / 6 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| LFOs and modulation ❌ 0 / 170 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Effects ❌ 0 / 46 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |

## Baselines

| Condition | MIDI-only R@1 / R@5 / MRR | Family bootstrap 95% interval for R@1 | Random R@1 / R@5 / MRR |
|---|---:|---:|---:|
| Common gallery MIDI | 0.0078 / 0.0388 / 0.0422 | [0.0078, 0.0078] | 0.0078 / 0.0388 / 0.0422 |
| Mixed gallery MIDI | 0.0078 / 0.0388 / 0.0421 | [0.0071, 0.0084] | 0.0078 / 0.0388 / 0.0422 |

## Evaluation protocol

The primary condition uses one common, existing gallery MIDI per episode and a query render of the same target preset on another existing MIDI. Query/gallery MIDI IDs differ; selection rotates deterministically. Every admitted validation preset is paired with all unique compatible same-split MIDI excerpts, making multiple cross-renders per preset. Gallery candidates are all admitted source states. The mixed-gallery condition uses a deterministic candidate-specific MIDI assignment and is reported separately. A same-MIDI different-patch negative check compares each cross-MIDI positive against other patches rendered on the query MIDI. Known frozen lineage groups are reported as family-level retrieval and family-cluster bootstrap units.

Metrics use cosine similarity over FP32 cached vectors; candidate sets and query episodes are paired across available arms. Acoustic hard negatives are selected independently of the tested encoders using cosine similarity over normalized 64-bin log-frequency power spectra of the target query and same-MIDI candidate renders; each encoder’s cross-MIDI positive is then compared against the five nearest such negatives on that query MIDI. The report includes exact-state and family-level Recall@1, Recall@5, MRR, family-cluster 95% bootstrap intervals, MIDI-only metadata and random baselines, and strata for polyphony, articulation, expression, source-lineage group size, pitch gap, duration, and note density. Sequence outputs include a valid-frame mean and an 8-bin ordered mean/std pyramid. No pooling or similarity setting is selected on reserved test labels.

This benchmark uses new Vital renders only; source WAVs are not used as model inputs or mixed with renders.
