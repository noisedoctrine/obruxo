# Frozen encoder retrieval benchmark

This package implements the real Vital/MIDI retrieval shootout from [issue #152](https://github.com/noisedoctrine/obruxo/issues/152). It performs no Vital-control prediction, patch optimization, backbone fine-tuning, or reconstruction training. The main query population is the complete PresetShare-paired, renderer-compatible `validation` membership drawn from an existing frozen corpus snapshot; `train` is an explicit alternative for development only. Test and independent-confirmation reserves are rejected by the CLI.

## Source and split contract

Prepare from a complete artifact produced by the repository's corpus audit and its matching original data root. Use the existing identity and lineage assignments in `splits.json` and `members.json`; do not make a new split. Source preset bytes are rehashed and their stored-state identity is rechecked. Only eligible rows in the chosen `train` or `validation` split are opened. Excluded/ineligible rows and reserved source paths are not resolved or read.

The current input adapter reuses the existing PresetShare evaluator's direct-child pairing contract: one `.mid`, at most one `.wav`, and one `.vital` source preset. Ambiguous directories are excluded, the paired MIDI must pass the selected renderer's compatibility checks, and source WAVs are never used as benchmark audio. Each admitted preset is then cross-rendered on every unique compatible MIDI excerpt from the selected existing split. These are explicitly new Vital renders; the benchmark does not relabel them as original captured WAV/MIDI pairs. Exact normalized MIDI-event duplicates collapse to one excerpt identity. Source file hashes and normalized excerpt IDs are retained only in ignored local manifests.

Every compatible source MIDI uses its leading, complete-note excerpt up to eight musical seconds, with a minimum of two seconds, plus the renderer's fixed two-second release tail. The model waveform is therefore 4–10 seconds. Tempo changes, channels, and supported expressive events are kept as authored. The selected renderer's declared capabilities determine eligibility; unsupported events fail with a recorded exclusion reason. No velocity, channel, tempo, or event is silently rewritten.

The main validation candidate set includes all statically eligible, byte-verified original source states that also pass the exact PresetShare source pairing and selected renderer MIDI checks. For each target state, every unique compatible same-split MIDI excerpt becomes a query. A deterministic rotating MIDI from the frozen gallery pool (up to the configured pool size) is rendered on every candidate state; query and common-gallery MIDI IDs must differ. The mixed-gallery condition assigns candidate-specific gallery MIDIs deterministically. A same-MIDI different-patch check compares each cross-MIDI positive against other patches rendered on the query MIDI. This requires the full candidate × MIDI render matrix; its actual size is reported before rendering, with no hard-coded candidate cap.

Known source lineage/alias groups remain intact. Exact state identity deduplicates display-name aliases. Family retrieval and family-cluster bootstrap are reported alongside exact-state retrieval. The pipeline does not infer fuzzy near-duplicate timbres or source instrument categories; those remain a limitation for hard-negative interpretation.

## Prepare, render, cache, evaluate

Use a complete existing snapshot path and the exact data root it audits. For the main run, use the checked-in full-preset expression renderer profile, which reuses the existing full Vital state contract and supports polyphony, multiple channels, pitch bend, channel pressure, release velocity, and CC 1/64/74. Tempo changes remain unsupported. `OBRUXO_VITAL_PLUGIN` or `--plugin-path` must identify the verified Vital 1.6.4 VST3 binary. The renderer still enforces one in-process worker.

```powershell
python research/modelling/frozen_encoder_retrieval/run.py doctor

python research/modelling/frozen_encoder_retrieval/run.py prepare `
  --snapshot research/training/outputs/<complete-source-snapshot> `
  --data-root datasets/presetshare/raw/presetshare_files/data `
  --output research/modelling/frozen_encoder_retrieval/outputs/issue152-validation

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

The plan, renderer, and feature extraction are separate commands. The render command reads a bounded queue of durable manifest batches and calls the existing `run_batch` with exactly one worker in the consumer thread. This can run independently of a later trainer; a future training producer can use the same durable request manifest and bounded queue without calling Vital synchronously in each batch. Each successful WAV and feature cache is reused on resume. Plan hashes, original preset/MIDI hashes, render provenance, checkpoint revision/digest, and cache identity are checked before reuse. A changed plan, source, checkpoint, loader/runtime, preprocessing, tap, pooling rule, or precision fails closed.

Private manifests, WAVs, checkpoint files, feature arrays, and per-episode scores belong in the ignored `outputs/` and `local_checkpoints/` directories, never in Git. The report generator publishes only aggregate counts and metrics.

## Frozen candidates and adapters

Checkpoint acquisition is a separate, manual step. The runner sets Hugging Face offline mode and only loads local files; it never downloads or substitutes weights. Fill a local lock from verified downloads and compatible source checkouts using [checkpoint-lock.example.json](checkpoint-lock.example.json). Every selected checkpoint must have its exact HF/source revision, SHA-256, loader revision, relevant runtime package versions, and `license_status: approved`. The exact BEATs mirror SHA from the issue comment is enforced. MuQ-MuLan is non-commercial; CIR remains unavailable until its checkpoint terms are verified.

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

See [the result report](reports/FROZEN_ENCODER_RETRIEVAL_RESULTS.md). The implementation preflight lists local data paths and checks Python dependencies; it does not parse or render raw files, load weights, or claim retrieval scores. The exact missing environment/data dependencies are recorded there. An unavailable encoder is an explicit report row, never a substitute result.
