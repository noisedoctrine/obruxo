# Frozen encoder retrieval benchmark

This package implements the real Vital/MIDI retrieval shootout from [issue #152](https://github.com/noisedoctrine/obruxo/issues/152). It performs no Vital-control prediction, patch optimization, backbone fine-tuning, or reconstruction training. The original cohort uses the PresetShare-paired, renderer-compatible `validation` membership from an existing frozen snapshot. The expanded cohort keeps that same validation preset gallery but draws held-out MIDI performances from paired `train` source directories; an exact-MIDI-ID exclusion manifest prevents overlap with the original validation performances. Its 50 MIDI IDs are selected deterministically to cover polyphony, articulation, pitch register, density, velocity range, and expression. Test and independent-confirmation reserves are rejected by the CLI.

## Results and presentations

**Start here:** [Frozen encoder visual summary](reports/FROZEN_ENCODER_VISUAL_SUMMARY.md) — five charts of the key quality, speed and memory tradeoffs.

The [research results report](reports/FROZEN_ENCODER_RETRIEVAL_RESULTS.md) is the authoritative source for metrics, methodology, and limitations. These presentations explain the results for collaborators:

- [Frozen encoder retrieval (PDF)](reports/presentations/frozen_encoder_retrieval.pdf) · [editable PowerPoint](reports/presentations/frozen_encoder_retrieval.pptx) — identification of Vital presets across different MIDI performances; based on the [report at commit 2aa787a](https://github.com/noisedoctrine/obruxo/blob/2aa787aea562db5d334a54c485b70704e5943a9e/research/modelling/frozen_encoder_retrieval/reports/FROZEN_ENCODER_RETRIEVAL_RESULTS.md).
- [Preset versus MIDI sensitivity (PDF)](reports/presentations/preset_midi_sensitivity.pdf) · [editable PowerPoint](reports/presentations/preset_midi_sensitivity.pptx) — comparison of preset-induced and performance-induced embedding changes, and their interaction; based on [PR #158](https://github.com/noisedoctrine/obruxo/pull/158).

Both presentations focus on the shared 129-preset × 50-performance cohort. They summarize frozen-representation measurements, not patch reconstruction results.

## Source and split contract

Prepare from a complete artifact produced by the repository's corpus audit and its matching original data root. Use the existing identity and lineage assignments in `splits.json` and `members.json`; do not make a new split. Source preset bytes are rehashed and their stored-state identity is rechecked. Only eligible rows in the chosen `train` or `validation` split are opened. Excluded/ineligible rows and reserved source paths are not resolved or read.

The input adapter pairs one direct-child `.vital` with one direct-child `.mid`. Source WAVs are not enumerated, opened, validated, or used as eligibility criteria; they are never benchmark audio. Ambiguous preset/MIDI directories are excluded, and the paired MIDI must pass the selected renderer's compatibility checks. Each admitted preset is then cross-rendered on every unique compatible MIDI excerpt from the selected existing split. These are explicitly new Vital renders; the benchmark does not relabel them as original captured WAV/MIDI pairs. Exact normalized MIDI-event duplicates collapse to one excerpt identity. Source file hashes and normalized excerpt IDs are retained only in ignored local manifests.

Every compatible source MIDI uses its leading, complete-note excerpt up to eight musical seconds, with a minimum of two seconds, plus the renderer's fixed two-second release tail. The model waveform is therefore 4–10 seconds. Tempo changes, channels, and supported expressive events are kept as authored. The selected renderer's declared capabilities determine eligibility; unsupported events fail with a recorded exclusion reason. No velocity, channel, tempo, or event is silently rewritten.

The validation candidate set includes all statically eligible, byte-verified original states that pass exact PresetShare pairing and renderer compatibility. The original cohort pairs them with compatible validation MIDI. The expanded cohort pairs the same candidate states with a deterministic 50-ID sample of compatible train-source MIDI, excluding all exact MIDI IDs in the original cohort. For each target state, every selected MIDI becomes a query. A deterministic rotating MIDI from the frozen gallery pool (up to the configured pool size) is rendered on every candidate state; query and common-gallery MIDI IDs must differ. The mixed-gallery condition assigns candidate-specific gallery MIDIs deterministically. Same-MIDI diagnostics compare the query against the full other-preset gallery and known lineage neighbors. A separate acoustic hard-negative check selects up to five same-MIDI candidates by an independent 64-bin log-frequency spectral descriptor, then checks whether a target's cross-MIDI positive outranks those negatives. These auxiliary query-MIDI renders support diagnostics and do not gate a primary episode, which requires only its target query and gallery requests. Per-condition render QA and encoder feature coverage are reported. The primary matrix size is preflighted as presets × unique MIDI excerpts; `--max-render-requests` fails before plan creation when the full matrix exceeds its budget and never truncates presets or MIDI files.

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

# Expanded held-out performance cohort, using the same validation preset gallery:
python research/modelling/frozen_encoder_retrieval/run.py prepare `
  --snapshot research/training/outputs/<complete-source-snapshot> `
  --data-root datasets/presetshare/raw/presetshare_files `
  --split validation --performance-source-split train --max-performances 50 `
  --exclude-performance-manifest research/modelling/frozen_encoder_retrieval/outputs/issue152-validation-real-cross-v2b/performances.private.jsonl `
  --output research/modelling/frozen_encoder_retrieval/outputs/issue152-validation-train-midi-v3-50 `
  --max-render-requests 10000

python research/modelling/frozen_encoder_retrieval/run.py render `
  --dataset research/modelling/frozen_encoder_retrieval/outputs/issue152-validation `
  --batch-size 32 --queue-depth 2

python research/modelling/frozen_encoder_retrieval/run.py embed `
  --dataset research/modelling/frozen_encoder_retrieval/outputs/issue152-validation `
  --model clap_music --checkpoint-root research/modelling/frozen_encoder_retrieval/local_checkpoints `
  --lock research/modelling/frozen_encoder_retrieval/local_checkpoints/checkpoint-lock.json --device cpu

python research/modelling/frozen_encoder_retrieval/run.py evaluate `
  --dataset research/modelling/frozen_encoder_retrieval/outputs/issue152-validation

# Add a newly extracted family to an existing evaluation without recomputing the other arms:
python research/modelling/frozen_encoder_retrieval/run.py embed `
  --dataset research/modelling/frozen_encoder_retrieval/outputs/issue152-validation-train-midi-v3-50 `
  --model basic_pitch_trunk --checkpoint-root research/modelling/frozen_encoder_retrieval/local_checkpoints `
  --lock research/modelling/frozen_encoder_retrieval/local_checkpoints/checkpoint-lock.json --device xpu

python research/modelling/frozen_encoder_retrieval/run.py evaluate `
  --dataset research/modelling/frozen_encoder_retrieval/outputs/issue152-validation-train-midi-v3-50 `
  --models basic_pitch_trunk

python research/modelling/frozen_encoder_retrieval/run.py report `
  --dataset research/modelling/frozen_encoder_retrieval/outputs/issue152-validation
```

The controlled follow-up uses one validation preset and the complete 50-performance held-out cohort, covering varied polyphony, register, note duration, articulation, and velocity. It perturbs one filter, envelope, and routed LFO frequency in separate variants. Every variant is rendered with the source Vital seed and a distinct independent seed; same-MIDI retrieval uses the independent-seed gallery so a query cannot match itself. Patch distances are compared with same-preset seed-repeat variance, with uncertainty bootstrapped over all 50 MIDI identities:

```powershell
python research/modelling/frozen_encoder_retrieval/run_controlled_perturbation.py `
  --dataset research/modelling/frozen_encoder_retrieval/outputs/issue152-validation-train-midi-v3-50 `
  --renderer-config research/data_generation/configs/renderer_full_corpus.yaml `
  --plugin-path 'C:\Program Files\Common Files\VST3\Vital.vst3' `
  --checkpoint-root research/modelling/frozen_encoder_retrieval/local_checkpoints `
  --lock research/modelling/frozen_encoder_retrieval/local_checkpoints/checkpoint-lock.json `
  --output research/modelling/frozen_encoder_retrieval/outputs/issue152-controlled-midi-v3-50 `
  --models clap_music muq_mulan dac_44khz encodec_24khz beats_iter3_plus cir_ast --device xpu

python research/modelling/frozen_encoder_retrieval/run.py report `
  --dataset research/modelling/frozen_encoder_retrieval/outputs/issue152-validation-train-midi-v3-50 `
  --comparison-dataset research/modelling/frozen_encoder_retrieval/outputs/issue152-validation-real-cross-v2b `
  --checkpoint-root research/modelling/frozen_encoder_retrieval/local_checkpoints `
  --lock research/modelling/frozen_encoder_retrieval/local_checkpoints/checkpoint-lock.json `
  --controlled-summary research/modelling/frozen_encoder_retrieval/outputs/issue152-controlled-midi-v3-50/controlled_summary.private.json
```

The opt-in end-to-end integration test uses the real snapshot, renderer, and plugin on a small two-preset/two-MIDI validation slice, then runs the real feature-cache and evaluation stages with a deterministic test-only adapter. It verifies stage wiring and cache/evaluation contracts, and its generated scores are never research results. Set `OBRUXO_RUN_REAL_INTEGRATION=1`, `OBRUXO_ISSUE152_SNAPSHOT`, `OBRUXO_ISSUE152_DATA_ROOT`, `OBRUXO_ISSUE152_RENDERER_CONFIG`, and `OBRUXO_VITAL_PLUGIN`, then run the package test discovery command. It is skipped in the ordinary unit-test run. If rendering and model evaluation use separate Python environments, make evaluation import the same pinned NumPy/SciPy environment as feature extraction; a mismatched native NumPy runtime can crash during matrix evaluation. Real pretrained checkpoints are exercised by the separate full-corpus experiment reported below.

The plan, renderer, and feature extraction are separate commands. The render command reads a bounded queue of durable manifest batches and calls the existing `run_batch` with exactly one worker in the consumer thread. This can run independently of a later trainer; a future training producer can use the same durable request manifest and bounded queue without calling Vital synchronously in each batch. Each successful WAV and feature cache is reused on resume. Plan hashes, original preset/MIDI hashes, render provenance, checkpoint revision/digest, and cache identity are checked before reuse. A changed plan, source, checkpoint, loader/runtime, preprocessing, tap, pooling rule, or precision fails closed.

Private manifests, WAVs, checkpoint files, feature arrays, and per-episode scores belong in the ignored `outputs/` and `local_checkpoints/` directories, never in Git. The report generator publishes only aggregate counts and metrics.

## Issue #155 follow-up

The follow-up reuses the expanded dataset's accepted renders and feature caches. It scores three fixed common-gallery MIDI offsets, recomputes clean-query and warning-free-gallery ranks, then re-embeds only 100 gain-reduced amplitude-warning WAVs across the evaluated encoder families. The separate encoder-blind preset screen renders short and long held/phrased examples and writes private before/after/repeat listening WAVs. It stops at the shortlist; the final controlled diagnostic requires the user's choice. Run the shortlist script with the Vital-rendering Python environment that has DawDreamer installed; the pinned benchmark environment is sufficient for cached-feature evaluation.

```powershell
python research/modelling/frozen_encoder_retrieval/run_issue155_validation.py `
  --dataset research/modelling/frozen_encoder_retrieval/outputs/issue152-validation-train-midi-v3-50 `
  --checkpoint-root research/modelling/frozen_encoder_retrieval/local_checkpoints `
  --lock research/modelling/frozen_encoder_retrieval/local_checkpoints/checkpoint-lock.json `
  --output research/modelling/frozen_encoder_retrieval/outputs/issue155-validation `
  --device xpu --gallery-offsets 1 7 19 --sample-count 100

python research/modelling/frozen_encoder_retrieval/run_controlled_candidate_shortlist.py `
  --dataset research/modelling/frozen_encoder_retrieval/outputs/issue152-validation-train-midi-v3-50 `
  --renderer-config research/data_generation/configs/renderer_full_corpus.yaml `
  --plugin-path 'C:\Program Files\Common Files\VST3\Vital.vst3' `
  --output research/modelling/frozen_encoder_retrieval/outputs/issue155-controlled-candidates

python research/modelling/frozen_encoder_retrieval/run.py report `
  --dataset research/modelling/frozen_encoder_retrieval/outputs/issue152-validation-train-midi-v3-50 `
  --comparison-dataset research/modelling/frozen_encoder_retrieval/outputs/issue152-validation-real-cross-v2b `
  --checkpoint-root research/modelling/frozen_encoder_retrieval/local_checkpoints `
  --lock research/modelling/frozen_encoder_retrieval/local_checkpoints/checkpoint-lock.json `
  --controlled-summary research/modelling/frozen_encoder_retrieval/outputs/issue152-controlled-midi-v3-50/controlled_summary.private.json `
  --issue155-results research/modelling/frozen_encoder_retrieval/outputs/issue155-validation/issue155_validation_summary.private.json `
  --candidate-shortlist research/modelling/frozen_encoder_retrieval/outputs/issue155-controlled-candidates/controlled_candidate_shortlist.private.json `
  --output research/modelling/frozen_encoder_retrieval/reports/FROZEN_ENCODER_RETRIEVAL_RESULTS.md
```

The report step uses Matplotlib to write the embedded PNG summaries under `reports/figures/`; it reads cached evaluations and shortlist measurements and does not run encoder inference or rendering.

The private shortlist, WAVs and adjusted feature caches remain under ignored `outputs/`. Read `controlled_candidate_shortlist.private.md` and listen to its linked examples before choosing candidates; do not start `run_controlled_perturbation.py` until that selection is approved.

## Issue #157 cached sensitivity analysis

The preset-versus-MIDI sensitivity analysis uses only the accepted frozen-embedding cache. It verifies cache identity, render receipts, archive hashes, checkpoint and loader provenance, feature shape and numeric validity before computing the exhaustive anchored comparison and crossed-bootstrap intervals. It fails closed on incomplete or mismatched caches; it does not render audio or run encoder inference. Run it from the repository root in the pinned benchmark environment:

```powershell
python research/modelling/frozen_encoder_retrieval/run_issue157_sensitivity.py `
  --dataset research/modelling/frozen_encoder_retrieval/outputs/issue152-validation-train-midi-v3-50 `
  --checkpoint-root research/modelling/frozen_encoder_retrieval/local_checkpoints `
  --device xpu --bootstrap-replicates 1000 --bootstrap-combinations 32768
```

The command updates [the research report](reports/FROZEN_ENCODER_RETRIEVAL_RESULTS.md), its machine-readable result JSON, and four figures. The analysis covers seven encoder families and eleven cached representation variants on the shared 129-preset × 50-performance matrix.

## Issue #161 three-MIDI audible-substitute retrieval

This catalog-selection test preselects 12 MIDI triplets from MIDI event records, then uses MIDI 1 embeddings to retrieve among 128 different source families rendered on MIDI 2. It scores the selected alternatives on the same third MIDI using existing WAVs. The oracle and fixed random alternative are evaluation references only. The runner verifies the existing 11-arm cache and selected WAV receipt/hash identities and does not render, extract features, load an encoder, or use reserved sources. Raw triplet IDs, case scores, WAV paths, and acoustic matrices stay in ignored `outputs/`.

```powershell
& research/modelling/frozen_encoder_retrieval/local_checkpoints/bench-env/Scripts/python.exe -B `
  research/modelling/frozen_encoder_retrieval/run_issue161_substitution.py
```

The [runner](run_issue161_substitution.py) pins seed 161, freezes the MIDI-event-only triplet manifest before reading retrieval metrics, checks exact sample alignment and accepted render receipts, and updates the [research report](reports/FROZEN_ENCODER_RETRIEVAL_RESULTS.md) plus its sanitized [aggregate result](reports/issue161_substitution_results.json). Add `--prepare-only` to freeze the sample and verify selected WAVs without scoring. Focused tests use the same benchmark interpreter:

For an interrupted run that already completed the full cache/WAV audit and saved all retrieval choices, `--resume-scoring --device xpu` revalidates source manifests and the selected MIDI-3 receipts, validates all saved 11-arm choices against the 128-family pool, and resumes only the acoustic scoring and aggregation. `--device cpu` is available for diagnostics.

```powershell
& research/modelling/frozen_encoder_retrieval/local_checkpoints/bench-env/Scripts/python.exe -B -m pytest -q `
  research/modelling/frozen_encoder_retrieval/tests/test_issue161_substitution.py
```

## Frozen candidates and adapters

Checkpoint acquisition is separate from benchmark execution. The runner sets Hugging Face offline mode and only loads local files; it never downloads or substitutes weights. Fill a local lock from exact downloads and compatible source checkouts using [checkpoint-lock.example.json](checkpoint-lock.example.json). Every selected checkpoint records its HF/source revision, SHA-256, loader revision, runtime package versions, and terms status. Use `license_status: approved` for terms cleared for this use or `research-use-unverified` when locally authorized research proceeds while terms remain unresolved; only `explicitly-prohibited` stops extraction. MuQ-MuLan is used only for non-commercial research. This benchmark keeps all checkpoints local and does not redistribute weights; that does not clear future deployment or distribution. CIR and BEATs are evaluated with unresolved checkpoint-term notes retained in the report. MuQ-MuLan and CIR also require separately pinned local architecture/config assets; a cold cache fails during preflight instead of reaching the network. The exact BEATs mirror SHA from the issue comment is enforced.

| Arm | Exact checkpoint and feature | Input rate | Output |
|---|---|---:|---:|
| LAION-CLAP Music | `lukewys/laion_clap/music_audioset_epoch_15_esc_90.14.pt`; `HTSAT-base`, `enable_fusion=False`; official clip embedding | 48 kHz | 512-D |
| Spotify Basic Pitch contour trunk | Pinned ICASSP-2022 checkpoint; post-BN/ReLU `contour_conv1` map, pooled from three contour bins to 88 semitones | 22.05 kHz | 704-D per frame |
| MuQ-MuLan | `OpenMuQ/MuQ-MuLan-large/pytorch_model.bin`; audio branch official latent | 24 kHz | 512-D |
| CIR single-timbre AST | Authors' `ICASSP26_inst-direct-with-data-aug_final.pt`, not the Triplet checkpoint; mean of final CLS and distillation tokens | 16 kHz | 768-D |
| BEATs iter3+ AS2M | `lpepino/beats_ckpts/BEATs_iter3_plus_AS2M.pt`; pretrained, not AudioSet cpt1/cpt2; pre-predictor sequence | 16 kHz | 768-D per frame |
| Meta EnCodec | `facebook/encodec_24khz`; continuous mono encoder output before RVQ | 24 kHz | 128-D per frame |
| Descript DAC | `descript/dac_44khz`; continuous mono encoder output before RVQ | 44.1 kHz | 1024-D per frame |

Clip adapters use the official clip latent. Sequence adapters are frozen/eval, FP32, and return both a valid-frame temporal mean and an eight-bin ordered mean/std representation. Basic Pitch input is windowed using its fixed two-second context and overlap; its contour-branch hidden map is averaged across the three sub-semitone contour bins before flattening channels × semitones into each frame. Basic Pitch's pretrained objective is note transcription, so this representation is an exploratory performance-feature baseline, not a timbre-trained embedding. BEATs flattened patches are reshaped by its 128-bin mel-frequency patch count, then frequency-averaged to a time sequence; partial final time patches are excluded. No padded frame enters a pool. Cosine scoring normalizes each cached output at evaluation time. Stereo is averaged to mono and resampling uses SciPy polyphase resampling. The original #152 input path applies no shared gain normalization; issue #164 adds a separate active-RMS cache. EnCodec's checkpoint-configured internal RMS normalization is retained in both conditions when enabled.

## What is executed in this checkout

See [the result report](reports/FROZEN_ENCODER_RETRIEVAL_RESULTS.md) for the executed real-corpus comparison, render/feature coverage, measured results, controlled diagnostic, exact provenance, and remaining unavailable arms. An unavailable encoder is an explicit report row, never a substitute result.

## Issue #164 padding-invariant active-RMS comparison

The first command revalidates the original cache, accepted WAV receipts, issue #155/#157/#161 inputs, and all locked checkpoint hashes. It derives one cohort-wide target from the maximum measured peak-to-active-RMS ratio so no normalized source sample exceeds 0.95. Input normalization uses stereo mean-square 20 ms windows at every sample offset, zero extension, and a relative −40 dB gate; it runs before mono downmix and encoder resampling. `features_active_rms/` holds the separate normalized cache. The original `features/` cache and source WAVs remain untouched.

```powershell
$base = 'research/modelling/frozen_encoder_retrieval'
$dataset = "$base/outputs/issue152-validation-train-midi-v3-50"
$py = "$base/local_checkpoints/bench-env/Scripts/python.exe"
& $py -B "$base/run_issue164_normalization.py" --stage preflight
$target = (& $py -B -c "import json; print(json.load(open('$base/outputs/issue164-active-rms/preflight.private.json'))['normalization']['target_active_rms'])")

foreach ($model in @('cir_ast','beats_iter3_plus','clap_music','dac_44khz','encodec_24khz','muq_mulan')) {
  & $py -B "$base/run.py" embed --dataset $dataset --model $model --checkpoint-root "$base/local_checkpoints" --lock "$base/local_checkpoints/checkpoint-lock.json" --device xpu --active-rms-target $target
}
$oldPythonpath = $env:PYTHONPATH
$env:PYTHONPATH = "$env:USERPROFILE/miniforge3/envs/obruxo-data-generation/Lib/site-packages"
& $py -B "$base/run.py" embed --dataset $dataset --model basic_pitch_trunk --checkpoint-root "$base/local_checkpoints" --lock "$base/local_checkpoints/checkpoint-lock.json" --device xpu --active-rms-target $target
if ($null -eq $oldPythonpath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue } else { $env:PYTHONPATH = $oldPythonpath }

& $py -B "$base/run_issue164_normalization.py" --stage evaluate
& $py -B "$base/run_issue164_normalization.py" --stage analyze --device xpu
```

To open a small live progress window while the background run is active, use `& $py -B "$base/show_issue164_progress.py"`. Play/Pause and Stop control the display only; extraction continues in the background.

Use the exact target printed by preflight for all seven embedding commands. `evaluate` reuses the original evaluation's acoustic hard-negative selections. `analyze` pairs issue #152 retrieval, issue #155's three existing gallery-MIDI assignments, and issue #161's fixed triplets against the preserved baselines; it recomputes only the normalized-side rankings, active-RMS-matched acoustic matrices and small ±6 dB input-gain probe. Active-RMS matching is an evaluation-time candidate-gain operation: it leaves both inputs and cached embeddings unchanged. The output is the sanitized [issue #164 aggregate](reports/issue164_active_rms_results.json), detailed result section and concise figures in the [research report](reports/FROZEN_ENCODER_RETRIEVAL_RESULTS.md) and [visual summary](FROZEN_ENCODER_VISUAL_SUMMARY.md), with all request-level provenance and matrices stored under ignored `outputs/issue164-active-rms/`.

## Issue #167 Vital JEPA V7

Place the [pinned V7 checkpoint](https://huggingface.co/duckking032/vital-jepa-v7/tree/e5ebdfb02dd107c15125b12c1bb4af03a7e31592/checkpoints/jepa_v7) at `local_checkpoints/vital_jepa_v7/selected.pt` under this directory. The [checkpoint lock](issue167-checkpoint-lock.json) records the expected file digest and runtime. The adapter uses a leading six-second stereo window, mid/side log-mel features and a 384-dimensional residual mean of online encoder tokens. Its framing and pooling are local choices based on the model configuration, checkpoint and published V5 pooling description.

From the repository root in PowerShell, use the native renderer Python for `render` and the frozen benchmark Python for analysis and efficiency. `--processes` controls the number of isolated native processes. Each process uses the pinned single-worker plugin profile.

```powershell
$base = 'research/modelling/frozen_encoder_retrieval'
$native = Join-Path $env:USERPROFILE 'miniforge3/envs/obruxo-data-generation/python.exe'
$bench = "$base/local_checkpoints/bench-env/Scripts/python.exe"
$env:PYTHONPATH = "$base;research/data_generation;research/training"
& $native -B -u "$base/run_issue167_v7.py" render --processes 4
& $bench -B -u "$base/run_issue167_analysis.py"
```

The run reads the accepted Issue #152 cohort, Issue #161 triplets, Issue #164 acoustic matrix contract and PR #160 resource sample. It writes the recovered audio and new evidence under `outputs/issue167-v7/` and uses `status.json` for progress. Open `show_issue167_progress.py` with the native Python for a monitor. The analysis CLI accepts `--wait-for-render` when a render worker is already running.

## Issue #169 V7 representation sweep

From the repository root, use the same frozen benchmark Python and retained Issue #167 run:

```powershell
& $bench -B -u "$base/run_issue169_tokens.py" --device xpu
& $bench -B -u "$base/run_issue169_retrieval.py"
```

The token worker verifies the WAV hashes and reuses completed token shards under `outputs/issue169-v7-retrieval/`. The retrieval worker reads those shards and saves derived vectors and query ranks in the same output directory. The [retrieval report](reports/VITAL_JEPA_V7_REPRESENTATION_SWEEP.md) includes the plot, head input assumptions, and a link to the aggregate results. Review the retrieval results before selecting candidates for substitute scoring and sensitivity analysis.

After selecting candidates, run the acoustic substitute and preset/MIDI sensitivity stage from the retained Issue #167 caches:

```powershell
& $bench -B -u "$base/run_issue169_stage2.py"
```

This stage reuses the 6,450 cached embeddings, 12 triplets, WAV receipts, and active-RMS acoustic matrices; it does not render audio or run encoder inference. Its [report](reports/VITAL_JEPA_V7_STAGE2.md) and machine-readable results are saved under `reports/`.
