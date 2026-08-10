# Comparative performance-transcription benchmark

This workspace compares exactly the frozen Basic Pitch baseline, Timbre-Trap base, the three required YourMT3+ variants, and MuScriptor small/medium/large. It owns no model integration or training. It consumes the landed #24 cost contract and #25 manifest, ground truth, metrics, aggregation, and sanitized-report seams.

## Reproducibility and availability

`config/models.yaml` records public source/checkpoint identity, hashes where the public artifact identity is available, licenses, native sample rates, output representation, and stock inference settings before any PresetShare result is inspected. Candidate source trees and weights are acquired manually from their official public locations into task-owned storage; they are never vendored, auto-downloaded, uploaded, or written into this repository. MuScriptor weights are gated, so a new login or terms acceptance is a blocker and is recorded as unavailable.

Only the existing user-managed `py312` environment may be used for local execution. The YAML files are reproducibility metadata only; they are not applied by the implementation. Missing dependencies, source checkouts, gated credentials, checkpoints, or runtimes produce explicit unavailable/dependency-unavailable results. There is no CPU/XPU, precision, decoder, model, or checkpoint fallback.

## Fixed semantics

- Basic Pitch quality consumes the existing #25 result and cost consumes the existing #24 result when the identities match.
- Timbre-Trap remains frame-only: native-frequency peak picking and the fixed `0.5` threshold happen before deterministic nearest-frame/nearest-50-cent mapping to #25's MIDI 21–108 grid. No note-event decoder is invented.
- YourMT3+ uses its official executable source and deterministic stock arguments; emitted events are normalized losslessly.
- MuScriptor uses `transcribe_to_midi`, `prelude_forcing=True`, greedy deterministic decoding, batch size 1, and no serialization velocity.
- Accuracy calls #25's note/frame metrics and 10,000-replicate seed-0 preset-cluster bootstrap. Executable candidates expose both success-only and failure-penalized views; unavailable candidates receive no fabricated F1.
- Full-precision cost follows #24's end-to-end clip boundary. Alternative models use only `pytorch_cpu` and `pytorch_xpu`; OpenVINO is not added. Quantization is only CPU dynamic qint8 ordinary `torch.nn.Linear`, with no calibration, search, XPU, backward, or batch sweep.

## Commands

```text
python research/modelling/performance_transcription_benchmark/run.py verify-model --config .../config/models.yaml --model-id <id> --source-root <pinned-checkout> --checkpoint <checkpoint>
python research/modelling/performance_transcription_benchmark/run.py evaluate --config .../config/models.yaml --model-id <id> --source-root <pinned-checkout> --checkpoint <checkpoint> --manifest <private-#25-manifest> --output <ignored-output>
python research/modelling/performance_transcription_benchmark/run.py benchmark --config .../config/models.yaml --model-id <id> --source-root <pinned-checkout> --checkpoint <checkpoint> --smoke-manifest <private-#24-manifest> --output <ignored-output>
python research/modelling/performance_transcription_benchmark/run.py report --input <ignored-run-root> --json research/modelling/performance_transcription_benchmark/reports/model_comparison.json --markdown research/modelling/performance_transcription_benchmark/reports/model_comparison.md
```

Add `--quantized` only to `evaluate` or `benchmark` for the fixed CPU dynamic-INT8-Linear experiment. Commands refuse overwrite without `--force` and return 0 for completed or genuinely unavailable routes, 2 for invalid identity/arguments, and 3 for a requested executable route that fails after safe partial persistence.

Private manifests, pair results, temporary MIDI, predictions, and runtime state stay under ignored `outputs/`. Public reports contain aggregate information only and reject local paths, source filenames, pair/preset/request identifiers, private hashes, and row predictions. Source WAV/MIDI/presets and existing/derived audio are read-only. #26 never independently renders missing audio: it reuses the exact #25 eligible population and its recorded existing/derived audio provenance. Derived-render eligibility is the deliberate, default-off `--allow-derived-render` capability implemented by #25 under the current #22 Rule 6 allowance; this workspace does not rerender.

CI runs only Ruff and synthetic/public tests. Full candidate and PresetShare runs remain bounded, foreground, local research validation. Later OBRUXO projector/fusion/training work is a separate follow-up.
