# Frozen audio encoder retrieval benchmark

Issue: [#152](https://github.com/noisedoctrine/obruxo/issues/152). This report covers frozen pretrained representations on original Vital source presets and existing MIDI performances. No synthesis controls are predicted, tuned, or optimized.

## Execution status

No prepared real dataset was supplied to this report command. No raw corpus MIDI or Vital preset was parsed, a benchmark data manifest was not prepared, and Vital renders, encoder extraction, and retrieval evaluation were not executed.
Available local inputs: complete audited snapshot at `research/training/outputs/full-curve-corpus-qualified-20261018/snapshot` and its PresetShare source directory. The snapshot census records 9620 discovered files and 131 statically eligible validation states (904 train, 118 reserved). These are pre-existing audit counts; renderer pairing and MIDI compatibility were not evaluated here.
Local preparation/render runtime is missing: `yaml`, `mido`, `numpy`, `scipy`, `dawdreamer`, `torch`, `transformers`.
Empirical execution is unfinished: prepare stopped before reading source files because PyYAML is missing. No Vital renders, encoder feature caches, or realistic retrieval scores were produced. Issue #152 remains open pending real-corpus results and the controlled patch-perturbation diagnostic.
Controlled patch-perturbation diagnostic: not run; no controlled score is claimed.

Runtime preflight: Python `3.13.12`; missing encoder runtime packages: `easydict`, `einops`, `huggingface_hub`, `laion_clap`, `librosa`, `muq`, `numpy`, `scipy`, `torch`, `torchaudio`, `transformers`, `x_clip`. Checkpoint root supplied: **no**. Vital plugin environment variable configured: **no**.
The existing [corpus audit report](../../../training/reports/CORPUS_INPUT_RESULTS.md) records prior static eligibility findings. Those audit counts are not native render eligibility. The separate [PresetShare audit](../../basic_pitch/README.md) has its own pairing rules and counts.

The default command selects the corpus-qualified full-preset expression renderer identity, including the established curve-input, ignored-empty-map, and native scalar readback policies. It supports note/polyphony, pitch-bend, channel-pressure, CC 1/64/74, release-velocity and multiple channels; tempo changes remain unsupported and are counted as exclusions. Unsupported events are rejected rather than silently removed. Musical spans are complete-note excerpts of 2–8 seconds plus a fixed 2-second release tail; model input is mono and 4–10 seconds, downmixed by channel mean and polyphase-resampled. No shared gain normalization is applied; EnCodec retains its own checkpoint-configured RMS normalization when enabled.

Render and feature stages are independent, resumable commands. Vital remains a single-worker in-process renderer; an asynchronous manifest reader feeds a bounded queue of request batches. Render keys include exact preset/MIDI/excerpt/renderer/policy identities. Feature keys include audio hash, checkpoint digest/revision, loader revision, sample rate, preprocessing, extraction tap, pooling, and FP32 precision. Weights are never downloaded by the run commands. Full paths, identifiers, WAVs, caches, checkpoints, and per-episode results belong in ignored local outputs.

No reconstruction head, source-label warm start, shared-model training, direct patch optimization, or audio-loss training is part of this experiment. No candidate can be recommended for reconstruction until the retrieval comparison has actually run; if results remain unavailable, the decision is inconclusive.

## Frozen-control scope

The linked [vital_control_inventory.jsonl](./vital_control_inventory.jsonl) contains one record for every exact control/instance from the existing 979-record Vital control inventory. All arms use original source preset inputs unchanged; each group is fixed with 0 enabled instances and 0 output values. No controls are omitted from the inventory; the renderer passes the original preset document through unchanged.

### LAION-CLAP Music HTSAT-base — official_clip

Checkpoint: `lukewys/laion_clap/music_audioset_epoch_15_esc_90.14.pt`; requested sample rate: 48000 Hz; representation: official audio clip embedding; output size: 512-D. Status: **unavailable / not run**.

| Measure / Vital control family | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Extraction time / peak memory | not run s / not measured bytes | — | local inference only; no training |
| Global settings ❌ 0 / 610 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Oscillators and voices ❌ 0 / 88 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Filters ❌ 0 / 53 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Envelopes ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| LFOs and modulation ❌ 0 / 170 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Effects ❌ 0 / 46 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |

Unavailable: checkpoint root not configured; missing Python packages: numpy, scipy, torch, laion_clap.

### MuQ-MuLan — official_clip

Checkpoint: `OpenMuQ/MuQ-MuLan-large/pytorch_model.bin`; requested sample rate: 24000 Hz; representation: official audio embedding; output size: 512-D. Status: **unavailable / not run**.

| Measure / Vital control family | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Extraction time / peak memory | not run s / not measured bytes | — | local inference only; no training |
| Global settings ❌ 0 / 610 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Oscillators and voices ❌ 0 / 88 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Filters ❌ 0 / 53 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Envelopes ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| LFOs and modulation ❌ 0 / 170 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Effects ❌ 0 / 46 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |

Unavailable: checkpoint root not configured; missing Python packages: numpy, scipy, torch, muq, torchaudio, librosa, transformers, huggingface_hub, einops, easydict, x_clip.

### CIR single-timbre AST — official_clip

Checkpoint: `Google Drive: ICASSP26_inst-direct-with-data-aug_final.pt`; requested sample rate: 16000 Hz; representation: mean of final CLS and distillation tokens; output size: 768-D. Status: **unavailable / not run**.

| Measure / Vital control family | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Extraction time / peak memory | not run s / not measured bytes | — | local inference only; no training |
| Global settings ❌ 0 / 610 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Oscillators and voices ❌ 0 / 88 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Filters ❌ 0 / 53 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Envelopes ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| LFOs and modulation ❌ 0 / 170 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Effects ❌ 0 / 46 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |

Unavailable: checkpoint root not configured; missing Python packages: numpy, scipy, torch, transformers, librosa.

### BEATs iter3+ AS2M — masked_mean

Checkpoint: `lpepino/beats_ckpts/BEATs_iter3_plus_AS2M.pt`; requested sample rate: 16000 Hz; representation: pretrained iter3+ final pre-predictor features; output size: 768-D. Status: **unavailable / not run**.

| Measure / Vital control family | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Extraction time / peak memory | not run s / not measured bytes | — | local inference only; no training |
| Global settings ❌ 0 / 610 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Oscillators and voices ❌ 0 / 88 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Filters ❌ 0 / 53 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Envelopes ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| LFOs and modulation ❌ 0 / 170 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Effects ❌ 0 / 46 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |

Unavailable: checkpoint root not configured; missing Python packages: numpy, scipy, torch, torchaudio.

### BEATs iter3+ AS2M — temporal_pyramid_8

Checkpoint: `lpepino/beats_ckpts/BEATs_iter3_plus_AS2M.pt`; requested sample rate: 16000 Hz; representation: pretrained iter3+ final pre-predictor features; output size: 12288-D. Status: **unavailable / not run**.

| Measure / Vital control family | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Extraction time / peak memory | not run s / not measured bytes | — | local inference only; no training |
| Global settings ❌ 0 / 610 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Oscillators and voices ❌ 0 / 88 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Filters ❌ 0 / 53 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Envelopes ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| LFOs and modulation ❌ 0 / 170 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Effects ❌ 0 / 46 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |

Unavailable: checkpoint root not configured; missing Python packages: numpy, scipy, torch, torchaudio.

### Meta EnCodec 24 kHz — masked_mean

Checkpoint: `facebook/encodec_24khz`; requested sample rate: 24000 Hz; representation: continuous encoder output before RVQ; output size: 128-D. Status: **unavailable / not run**.

| Measure / Vital control family | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Extraction time / peak memory | not run s / not measured bytes | — | local inference only; no training |
| Global settings ❌ 0 / 610 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Oscillators and voices ❌ 0 / 88 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Filters ❌ 0 / 53 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Envelopes ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| LFOs and modulation ❌ 0 / 170 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Effects ❌ 0 / 46 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |

Unavailable: checkpoint root not configured; missing Python packages: numpy, scipy, torch, transformers.

### Meta EnCodec 24 kHz — temporal_pyramid_8

Checkpoint: `facebook/encodec_24khz`; requested sample rate: 24000 Hz; representation: continuous encoder output before RVQ; output size: 2048-D. Status: **unavailable / not run**.

| Measure / Vital control family | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Extraction time / peak memory | not run s / not measured bytes | — | local inference only; no training |
| Global settings ❌ 0 / 610 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Oscillators and voices ❌ 0 / 88 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Filters ❌ 0 / 53 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Envelopes ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| LFOs and modulation ❌ 0 / 170 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Effects ❌ 0 / 46 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |

Unavailable: checkpoint root not configured; missing Python packages: numpy, scipy, torch, transformers.

### Descript Audio Codec 44.1 kHz — masked_mean

Checkpoint: `descript/dac_44khz`; requested sample rate: 44100 Hz; representation: continuous encoder output before RVQ; output size: 1024-D. Status: **unavailable / not run**.

| Measure / Vital control family | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Extraction time / peak memory | not run s / not measured bytes | — | local inference only; no training |
| Global settings ❌ 0 / 610 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Oscillators and voices ❌ 0 / 88 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Filters ❌ 0 / 53 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Envelopes ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| LFOs and modulation ❌ 0 / 170 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Effects ❌ 0 / 46 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |

Unavailable: checkpoint root not configured; missing Python packages: numpy, scipy, torch, transformers.

### Descript Audio Codec 44.1 kHz — temporal_pyramid_8

Checkpoint: `descript/dac_44khz`; requested sample rate: 44100 Hz; representation: continuous encoder output before RVQ; output size: 16384-D. Status: **unavailable / not run**.

| Measure / Vital control family | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Extraction time / peak memory | not run s / not measured bytes | — | local inference only; no training |
| Global settings ❌ 0 / 610 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Oscillators and voices ❌ 0 / 88 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Filters ❌ 0 / 53 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Envelopes ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| LFOs and modulation ❌ 0 / 170 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Effects ❌ 0 / 46 | fixed source input | 0 | unchanged original preset; no loss/gradient |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source input | 0 | unchanged original preset; no loss/gradient |

Unavailable: checkpoint root not configured; missing Python packages: numpy, scipy, torch, transformers.

## Baselines

| Condition | MIDI-only R@1 / R@5 / MRR | Family bootstrap 95% interval for R@1 | Random R@1 / R@5 / MRR |
|---|---:|---:|---:|
| Common gallery MIDI | not run | not run | not run |
| Mixed gallery MIDI | not run | not run | not run |

## Evaluation protocol

The primary condition uses one common, existing gallery MIDI per episode and a query render of the same target preset on another existing MIDI. Query/gallery MIDI IDs differ; selection rotates deterministically. Every admitted validation preset is paired with all unique compatible same-split MIDI excerpts, making multiple cross-renders per preset. Gallery candidates are all admitted source states. The mixed-gallery condition uses a deterministic candidate-specific MIDI assignment and is reported separately. A same-MIDI different-patch negative check compares each cross-MIDI positive against other patches rendered on the query MIDI. Known frozen lineage groups are reported as family-level retrieval and family-cluster bootstrap units.

Metrics use cosine similarity over FP32 cached vectors; candidate sets and query episodes are paired across available arms. The report includes Recall@1/5, MRR, family-cluster 95% bootstrap intervals, MIDI-only metadata and random baselines, and strata for polyphony, articulation, expression, pitch gap, duration, and note density. Sequence outputs include a valid-frame mean and an 8-bin ordered mean/std pyramid. No pooling or similarity setting is selected on reserved test labels.

This benchmark uses new Vital renders only; source WAVs are not used as model inputs or mixed with renders.
