# Original corpus sources and explicit label preparation

The expanded static audit finds a usable candidate population under the qualified 1.5.5 schema and factory catalog. Static acceptance is not native audio acceptance. The [results](reports/CORPUS_INPUT_RESULTS.md) retain the source-loading failures and their separately identified native correction.

`CorpusInputCodec` prepares model labels while keeping the original preset as the required training-audio source. It validates the original document first. It canonicalizes five display fields (`preset_style`, `macro1` through `macro4`), replaces unregistered assets only in disabled generator slots, and omits a mapping only when its connection is empty. Every operation has a path/reason ledger. It never changes the 775 scalar controls, active assets, active connection mappings, LFO/custom curves or seeds. Unknown active assets and invalid versions/ranges still fail.

The encoder/output domain restores all registered generator destinations, currently `0–14`. A source can be silent at its own output while serving as an FM modulator; isolated silence does not make that source or its stored destination disposable. The parent authored generator continues sampling its earlier bounded direct-output curriculum. This new codec owns its decoder instance, so restoring the full domain leaves earlier codec instances, identities and decoder banks unchanged.

The model family and head count stay the same, with four destination heads expanded from five to fifteen choices. The existing curve decoder remains frozen. Its reconstruction error on original corpus curves is measured separately from audio quality; label encoding does not establish that a trained model can reconstruct these sounds.

## Exact native corpus readback

`CorpusNativeCodec` uses the separate renderer suffix `-corpus-readback-v1`. It keeps the original source scalars intact. The usual continuous-control comparison still applies; a value outside that tolerance passes only when its loaded value exactly equals the existing encoder's declared normalized float32 round trip. Discrete controls must match exactly. Missing, non-finite or genuinely changed values fail. The renderer records every value accepted through this additional rule.

Native Vital discards a mapping on an empty connection. This profile omits that mapping from its applied copy before curve import and records the path; the original request remains intact. Active mappings and all other state still receive the full scalar and six-family readback checks. Existing curve import, expressive MIDI and signed seed handling remain in force. Earlier renderer, codec and decoder-bank identities stay unchanged; start a new run under the new contract.

The qualification reads the earlier static snapshot under its original contract, verifies that the new schema differs only by the declared native policy, and retains the frozen membership. It does not silently reinterpret that snapshot as a new training contract. Complete original-source streaming and native/audio admission remain separate work.

## Reproduce the checks

Use the dependency environment in [README](README.md), the qualified private factory/LFO banks and a new output directory for every command. Numerical-library thread limits such as `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` and `OPENBLAS_NUM_THREADS=2` keep local CPU work bounded.

```text
python research/training/reports/scripts/audit_corpus_profile.py --corpus CORPUS --asset-bank FACTORY_BANK.json --output AUDIT
python research/training/reports/scripts/audit_corpus_inputs.py --snapshot AUDIT/snapshot --corpus CORPUS --asset-bank FACTORY_BANK.json --lfo-bank LFO_BANK.json --output LABELS
python research/training/reports/scripts/audit_corpus_aliases.py --snapshot AUDIT/snapshot --corpus CORPUS --asset-bank FACTORY_BANK.json --output ALIASES
python research/training/reports/scripts/probe_corpus_inputs.py --snapshot AUDIT/snapshot --corpus CORPUS --asset-bank FACTORY_BANK.json --lfo-bank LFO_BANK.json --output NATIVE
python research/training/reports/scripts/qualify_silent_modulators.py --asset-bank FACTORY_BANK.json --output FM_CHECK
python research/training/reports/scripts/qualify_corpus_native.py --snapshot AUDIT/snapshot --corpus CORPUS --asset-bank FACTORY_BANK.json --lfo-bank LFO_BANK.json --output NATIVE_CONTRACT
```

The complete snapshot stores and verifies the selected renderer's performance bank. The older generic audit incorrectly wrote the legacy eight-performance bank for the sixteen-performance curve profile; that mismatch is fixed. Legacy snapshots keep their original bank. A substituted bank fails verification even if its artifact checksum has been updated.

The label audit checks only frozen training states, verifies their original byte hashes, writes private candidate labels and measures active LFO projection error. It does not fit a dictionary or render audio. The alias audit uses all statically eligible records only to check split isolation, before lossy curve encoding. It removes declared non-audio metadata and applies the existing native curve-input policy; it does not fix random phases, change assignments or infer acoustic/near-duplicate ancestry.

The native probe uses training reference performances, original versus prepared diagnostic copies, all four random phases fixed, and a fresh process per request. It records source failures and leaves them unadmitted. Its successful bounded label-copy panel is not a claim that the complete corpus has passed native loading/audio QA. Actual training must continue rendering the original source, not the label-prepared document.

`inspect_corpus_readback.py --snapshot AUDIT/snapshot --corpus CORPUS --state TRAINING_STATE_SHA --output DIAGNOSIS.json` captures loaded scalar values without changing tolerances or rendering audio. This historical diagnosis motivated the separate native corpus profile. Its qualification retains both legacy failures as negative controls and checks the corrected original requests, label copies, fresh repeats and precise legacy references. It also checks singleton/decimal active LFOs, mappings and custom modes 13–15 across both profiles.

Raw presets, candidate labels, source paths, audio and weights remain local. Population-wide native/audio admission, mixed authored/corpus coverage, decoder refit policy, useful model training and release remain open. Final-test audio stays reserved and GitHub Actions stays disabled.

## Train from original sources

`CorpusPool` connects the qualified candidate labels to every statically eligible training member in the original snapshot. Its separate dataset identity includes the complete original split ledger, source contract, candidate digest, targets, relative paths and raw byte hashes. It never rebuilds or filters that ledger. Validation and test sources have no entries in this training-only pool and are never opened by its reader. The builder rejects missing/duplicate members, changed ancestry, codec mismatches and incomplete candidate coverage.

Every scheduled read verifies its relative path, rejects links and root escapes, checks the original byte hash/state identity and re-encodes the source to verify its target before rendering. The provider returns the original bytes. The existing render request canonicalizes JSON formatting; its complete original document, assets, display labels and phase values remain the audio source. Label-prepared or decoded presets are never substituted. Audio is generated during training and bounded feature replay supplies recovery.

```text
python research/training/run.py create-corpus-pool --snapshot AUDIT/snapshot --corpus CORPUS --candidates LABELS/candidates.json --output PRIVATE_POOL.json
python research/training/reports/scripts/run_corpus_source_integration.py --pool PRIVATE_POOL.json --output INTEGRATION
```

The existing `train --pool PRIVATE_POOL.json` command loads this dataset type. The absolute source root appears only in the private wrapper, outside the dataset identity, so `PilotPool.load(path, corpus_root=NEW_ROOT)` can relocate identical source files. This still checks bytes and labels on every read. No corpus paths or dataset payloads enter the exported model codec. Earlier authored pools retain their exact identities and source behavior.

The current sixteen-performance bank has a longest held reference note of 1.5 seconds. Six of the 904 training states need longer for their complete amplitude attack; the maximum requested delay-plus-attack is about 32 seconds. They remain in the dataset. Consequently `attack_compatible` refuses the complete current corpus, and the bounded source-integration check uses the existing `state_round_robin` schedule. The current frontend also rejects audio longer than sixteen seconds; longer observations require explicit bounded feature handling. This check does not establish adequate temporal observations or native/audio acceptance for the whole corpus. A separately declared longer-note performance path, authored/corpus mixing, decoder refitting and useful broad training remain required.
