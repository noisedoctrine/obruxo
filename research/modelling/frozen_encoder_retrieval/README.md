# Frozen encoder retrieval benchmark

This package implements the real Vital/MIDI retrieval shootout from [issue #152](https://github.com/noisedoctrine/obruxo/issues/152). It performs no Vital-control prediction, patch optimization, backbone fine-tuning, or reconstruction training. The main query population is the complete PresetShare-paired, renderer-compatible `validation` membership drawn from an existing frozen corpus snapshot; `train` is an explicit alternative for development only. Test and independent-confirmation reserves are rejected by the CLI.

## Source and split contract

Prepare from a complete artifact produced by the repository's corpus audit and its matching original data root. Use the existing identity and lineage assignments in `splits.json` and `members.json`; do not make a new split. Source preset bytes are rehashed and their stored-state identity is rechecked. Only eligible rows in the chosen `train` or `validation` split are opened. Excluded/ineligible rows and reserved source paths are not resolved or read.

The input adapter pairs one direct-child `.vital` with one direct-child `.mid`. Source WAVs are not enumerated, opened, validated, or used as eligibility criteria; they are never benchmark audio. Ambiguous preset/MIDI directories are excluded, and the paired MIDI must pass the selected renderer's compatibility checks. Each admitted preset is then cross-rendered on every unique compatible MIDI excerpt from the selected existing split. These are explicitly new Vital renders; the benchmark does not relabel them as original captured WAV/MIDI pairs. Exact normalized MIDI-event duplicates collapse to one excerpt identity. Source file hashes and normalized excerpt IDs are retained only in ignored local manifests.

Every compatible source MIDI uses its leading, complete-note excerpt up to eight musical seconds, with a minimum of two seconds, plus the renderer's fixed two-second release tail. The model waveform is therefore 4–10 seconds. Tempo changes, channels, and supported expressive events are kept as authored. The selected renderer's declared capabilities determine eligibility; unsupported events fail with a recorded exclusion reason. No velocity, channel, tempo, or event is silently rewritten.

The main validation candidate set includes all statically eligible, byte-verified original source states that also pass the exact PresetShare source pairing and selected renderer MIDI checks. For each target state, every unique compatible same-split MIDI excerpt becomes a query. A deterministic rotating MIDI from the frozen gallery pool (up to the configured pool size) is rendered on every candidate state; query and common-gallery MIDI IDs must differ. The mixed-gallery condition assigns candidate-specific gallery MIDIs deterministically. Same-MIDI diagnostics compare the query against the full other-preset gallery and known lineage neighbors. A separate acoustic hard-negative check selects up to five same-MIDI candidates by an independent 64-bin log-frequency spectral descriptor, then checks whether a target's cross-MIDI positive outranks those negatives. These auxiliary query-MIDI renders support diagnostics and do not gate a primary episode, which requires only its target query and gallery requests. Per-condition render QA and encoder feature coverage are reported. The primary matrix size is preflighted as presets × unique MIDI excerpts; `--max-render-requests` fails before plan creation when the full matrix exceeds its budget and never truncates presets or MIDI files.

Known source lineage/alias groups remain intact. Exact state identity deduplicates display-name aliases. Family retrieval and family-cluster bootstrap are reported alongside exact-state retrieval. The spectral hard-negative set is a coarse audio-similarity proxy, not a human-verified duplicate-timbre or source-instrument label.

## Prepare, render, cache, evaluate

Use a complete existing snapshot path and the exact data root it audits. For the main run, use the checked-in full-corpus expression renderer profile. It preserves the corpus-qualified curve import, ignored-empty-map policy, and native float32 readbacks, alongside polyphony, multiple channels, pitch bend, channel pressure, release velocity, and CC 1/64/74. Tempo changes remain unsupported. `OBRUXO_VITAL_PLUGIN` or `--plugin-path` must identify the verified Vital 1.6.4 VST3 binary. The renderer still enforces one in-process worker.

```powershell
python research/modelling/frozen_encoder_retrieval/run.py doctor

python research/modelling/frozen_encoder_retrieval/run.py prepare `
  --snapshot research/training/outputs/<complete-source-snapshot> `
  --data-root datasets/presetshare/raw/presetshare_files `
  --output research/modelling/frozen_encoder_retrieval/outputs/issue152-validation `
  --max-render-requests 50000

python research/modelling/frozen_encoder_retrieval/run.py render `
  --dataset research/modelling/frozen_encoder_retrieval/outputs/issue152-validation `
  --batch-size 32 --queue-depth 2

python research/modelling/frozen_encoder_retrieval/run.py embed `
  --dataset research/modelling/frozen_encoder_retrieval/outputs/issue152-validation `
  --model clap_music --checkpoint-root research/modelling/frozen_encoder_retrieval/local_checkpoints `
  --lock research/modelling/frozen_encoder_retrieval/local_checkpoints/checkpoint-lock.json --device cpu

python research/modelling/frozen_encoder_retrieval/run.py evaluate `
  --dataset research/modelling/frozen_encoder_retrieval/outputs/issue152-validation

python research/modelling/frozen_encoder_retrieval/run.py report `
  --dataset research/modelling/frozen_encoder_retrieval/outputs/issue152-validation
```

The controlled follow-up uses the same original validation preset corpus and three or four real MIDI excerpts with varied polyphony, register, note duration, and available velocity ranges. It perturbs one filter, envelope, and live modulation field in separate variants, pre-renders those variants, and caches the same encoder features:

```powershell
python research/modelling/frozen_encoder_retrieval/run_controlled_perturbation.py `
  --dataset research/modelling/frozen_encoder_retrieval/outputs/issue152-validation `
  --renderer-config research/data_generation/configs/renderer_full_corpus.yaml `
  --plugin-path 'C:\Program Files\Common Files\VST3\Vital.vst3' `
  --checkpoint-root research/modelling/frozen_encoder_retrieval/local_checkpoints `
  --lock research/modelling/frozen_encoder_retrieval/local_checkpoints/checkpoint-lock.json `
  --output research/modelling/frozen_encoder_retrieval/outputs/issue152-controlled `
  --models clap_music dac_44khz --device xpu

python research/modelling/frozen_encoder_retrieval/run.py report `
  --dataset research/modelling/frozen_encoder_retrieval/outputs/issue152-validation `
  --checkpoint-root research/modelling/frozen_encoder_retrieval/local_checkpoints `
  --lock research/modelling/frozen_encoder_retrieval/local_checkpoints/checkpoint-lock.json `
  --controlled-summary research/modelling/frozen_encoder_retrieval/outputs/issue152-controlled/controlled_summary.private.json
```

The opt-in end-to-end integration test uses the real snapshot, renderer, and plugin on a small two-preset/two-MIDI validation slice, then runs the real feature-cache and evaluation stages with a deterministic test-only adapter. It verifies stage wiring and cache/evaluation contracts, and its generated scores are never research results. Set `OBRUXO_RUN_REAL_INTEGRATION=1`, `OBRUXO_ISSUE152_SNAPSHOT`, `OBRUXO_ISSUE152_DATA_ROOT`, `OBRUXO_ISSUE152_RENDERER_CONFIG`, and `OBRUXO_VITAL_PLUGIN`, then run the package test discovery command. It is skipped in the ordinary unit-test run. If rendering and model evaluation use separate Python environments, make evaluation import the same pinned NumPy/SciPy environment as feature extraction; a mismatched native NumPy runtime can crash during matrix evaluation. Real pretrained checkpoints are exercised by the separate full-corpus experiment reported below.

The plan, renderer, and feature extraction are separate commands. The render command reads a bounded queue of durable manifest batches and calls the existing `run_batch` with exactly one worker in the consumer thread. This can run independently of a later trainer; a future training producer can use the same durable request manifest and bounded queue without calling Vital synchronously in each batch. Each successful WAV and feature cache is reused on resume. Plan hashes, original preset/MIDI hashes, render provenance, checkpoint revision/digest, and cache identity are checked before reuse. A changed plan, source, checkpoint, loader/runtime, preprocessing, tap, pooling rule, or precision fails closed.

Private manifests, WAVs, checkpoint files, feature arrays, and per-episode scores belong in the ignored `outputs/` and `local_checkpoints/` directories, never in Git. The report generator publishes only aggregate counts and metrics.

## Frozen candidates and adapters

Checkpoint acquisition is a separate, manual step. The runner sets Hugging Face offline mode and only loads local files; it never downloads or substitutes weights. Fill a local lock from verified downloads and compatible source checkouts using [checkpoint-lock.example.json](checkpoint-lock.example.json). Every selected checkpoint must have its exact HF/source revision, SHA-256, loader revision, relevant runtime package versions, and `license_status: approved`. MuQ-MuLan and CIR also require separately pinned local architecture/config assets; a cold cache fails during preflight instead of reaching the network. The exact BEATs mirror SHA from the issue comment is enforced. MuQ-MuLan is non-commercial; CIR remains unavailable until its checkpoint terms are verified.

| Arm | Exact checkpoint and feature | Input rate | Output |
|---|---|---:|---:|
| LAION-CLAP Music | `lukewys/laion_clap/music_audioset_epoch_15_esc_90.14.pt`; `HTSAT-base`, `enable_fusion=False`; official clip embedding | 48 kHz | 512-D |
| MuQ-MuLan | `OpenMuQ/MuQ-MuLan-large/pytorch_model.bin`; audio branch official latent | 24 kHz | 512-D |
| CIR single-timbre AST | Authors' `ICASSP26_inst-direct-with-data-aug_final.pt`, not the Triplet checkpoint; mean of final CLS and distillation tokens | 16 kHz | 768-D |
| BEATs iter3+ AS2M | `lpepino/beats_ckpts/BEATs_iter3_plus_AS2M.pt`; pretrained, not AudioSet cpt1/cpt2; pre-predictor sequence | 16 kHz | 768-D per frame |
| Meta EnCodec | `facebook/encodec_24khz`; continuous mono encoder output before RVQ | 24 kHz | 128-D per frame |
| Descript DAC | `descript/dac_44khz`; continuous mono encoder output before RVQ | 44.1 kHz | 1024-D per frame |

Clip adapters use the official clip latent. Sequence adapters are frozen/eval, FP32, and return both a valid-frame temporal mean and an eight-bin ordered mean/std representation. BEATs flattened patches are reshaped by its 128-bin mel-frequency patch count, then frequency-averaged to a time sequence; partial final time patches are excluded. No padded frame enters a pool. Cosine scoring normalizes each cached output at evaluation time. Stereo is averaged to mono and resampling uses SciPy polyphase resampling. No shared gain normalization is applied; EnCodec's own checkpoint-configured RMS normalization is retained when enabled.

## What is executed in this checkout

See [the result report](reports/FROZEN_ENCODER_RETRIEVAL_RESULTS.md) for the executed real-corpus comparison, render/feature coverage, measured results, controlled diagnostic, exact provenance, and remaining unavailable arms. An unavailable encoder is an explicit report row, never a substitute result.
