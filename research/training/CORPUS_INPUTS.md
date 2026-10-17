# Original corpus sources and explicit label preparation

The expanded static audit finds a usable candidate population under the qualified 1.5.5 schema and factory catalog. Static acceptance is not native audio acceptance. The [results](reports/CORPUS_INPUT_RESULTS.md) retain the source-loading failure found during native probing.

`CorpusInputCodec` prepares model labels while keeping the original preset as the required training-audio source. It validates the original document first. It canonicalizes five display fields (`preset_style`, `macro1` through `macro4`), replaces unregistered assets only in disabled generator slots, and omits a mapping only when its connection is empty. Every operation has a path/reason ledger. It never changes the 775 scalar controls, active assets, active connection mappings, LFO/custom curves or seeds. Unknown active assets and invalid versions/ranges still fail.

The encoder/output domain restores all registered generator destinations, currently `0–14`. A source can be silent at its own output while serving as an FM modulator; isolated silence does not make that source or its stored destination disposable. The parent authored generator continues sampling its earlier bounded direct-output curriculum. This new codec owns its decoder instance, so restoring the full domain leaves earlier codec instances, identities and decoder banks unchanged.

The model family and head count stay the same, with four destination heads expanded from five to fifteen choices. The existing curve decoder remains frozen. Its reconstruction error on original corpus curves is measured separately from audio quality; label encoding does not establish that a trained model can reconstruct these sounds.

## Reproduce the checks

Use the dependency environment in [README](README.md), the qualified private factory/LFO banks and a new output directory for every command. Numerical-library thread limits such as `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` and `OPENBLAS_NUM_THREADS=2` keep local CPU work bounded.

```text
python research/training/reports/scripts/audit_corpus_profile.py --corpus CORPUS --asset-bank FACTORY_BANK.json --output AUDIT
python research/training/reports/scripts/audit_corpus_inputs.py --snapshot AUDIT/snapshot --corpus CORPUS --asset-bank FACTORY_BANK.json --lfo-bank LFO_BANK.json --output LABELS
python research/training/reports/scripts/audit_corpus_aliases.py --snapshot AUDIT/snapshot --corpus CORPUS --asset-bank FACTORY_BANK.json --output ALIASES
python research/training/reports/scripts/probe_corpus_inputs.py --snapshot AUDIT/snapshot --corpus CORPUS --asset-bank FACTORY_BANK.json --lfo-bank LFO_BANK.json --output NATIVE
python research/training/reports/scripts/qualify_silent_modulators.py --asset-bank FACTORY_BANK.json --output FM_CHECK
```

The complete snapshot stores and verifies the selected renderer's performance bank. The older generic audit incorrectly wrote the legacy eight-performance bank for the sixteen-performance curve profile; that mismatch is fixed. Legacy snapshots keep their original bank. A substituted bank fails verification even if its artifact checksum has been updated.

The label audit checks only frozen training states, verifies their original byte hashes, writes private candidate labels and measures active LFO projection error. It does not fit a dictionary or render audio. The alias audit uses all statically eligible records only to check split isolation, before lossy curve encoding. It removes declared non-audio metadata and applies the existing native curve-input policy; it does not fix random phases, change assignments or infer acoustic/near-duplicate ancestry.

The native probe uses training reference performances, original versus prepared diagnostic copies, all four random phases fixed, and a fresh process per request. It records source failures and leaves them unadmitted. Its successful bounded label-copy panel is not a claim that the complete corpus has passed native loading/audio QA. Actual training must continue rendering the original source, not the label-prepared document.

`inspect_corpus_readback.py --snapshot AUDIT/snapshot --corpus CORPUS --state TRAINING_STATE_SHA --output DIAGNOSIS.json` captures loaded scalar values without changing tolerances or rendering audio. The current blocker is a strict-check mismatch caused by a precisely reproduced native numeric round trip. That requires a separately verified rendering-contract correction before broader source ingestion.

Raw presets, candidate labels, source paths, audio and weights remain local. Corpus streaming, remaining native compatibility, mixed authored/corpus coverage, decoder refit policy, useful model training and release remain open. Final-test audio stays reserved and GitHub Actions stays disabled.
