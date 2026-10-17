# Corpus admission boundaries and label compatibility

The [input policy](../CORPUS_INPUTS.md) makes original corpus presets encodable without treating display text or unused asset content as required neural outputs. All source presets remain intact. The [compatibility receipt](data/corpus_compatibility.json) verifies current/historical sources and unchanged legacy codecs, frozen splits and LFO bank.

## Complete static audit

The [plan](data/corpus_static_plan.json) and [result](data/corpus_static_summary.json) inspect all 9,620 files: 9,618 parse, with one invalid JSON file and one invalid UTF-8 file excluded. There are 9,592 stored-state identities, matching the original frozen corpus. The expanded schema/catalog admits 1,155 files representing 1,153 distinct states: 904 training, 131 validation and 118 reserved test states. These are static counts, not native admission or model quality.

Overlapping rejection counts are 8,421 unresolved active assets, 3,367 schema failures and 3,037 unreviewed versions. Unresolved content is not automatically classified as imported or unusable. The first 64 training-source diagnostic found no additional asset matches from merely changing version headers or numeric spelling.

The initial complete scan exposed a snapshot bug: its contract named the sixteen-performance bank, but the generic writer saved the old eight-performance document. The corrected scan verifies the [saved bank](data/corpus_static_performances.json) against its [contract](data/corpus_static_contract.json), passes all four gates and preserves the same source identities, memberships and eligibility counts. Legacy bank behavior and tamper rejection pass regression checks.

## All training labels, with no source edits

The [label audit](data/corpus_labels_summary.json) encodes all 904 statically eligible training states. All 775 scalar values remain unchanged. Display text, unknown assets in disabled slots and ignored mappings on empty connections receive explicit label-only ledgers. Unknown active content is rejected.

Two failures in the first pass exposed real representation gaps. One preset uses destination `9` on an enabled, zero-output oscillator feeding FM; the old combined output domain stopped at `4`. The new codec restores the full registered `0–14` domain. The other stores a line mapping on an empty connection, which native Vital does not retain; only that ignored label mapping is omitted. Active mappings remain intact. Older authored codec instances keep their five-choice destination domain and exact identities.

The fixed decoder's active LFO projection MSE is 0.0043576 over 1,073 shapes/104,081 sampled values. This measures representation error, not sound reconstruction. No asset or curve dictionary is fitted. The [alias audit](data/corpus_aliases_summary.json), before lossy curve encoding, finds 1,151 distinct canonical programs, two alias groups and no cross-split alias groups. It changes no splits and is not an acoustic or general near-duplicate census.

## Native context and the remaining loading check

The [FM fixture panel](data/corpus_fm_summary.json) accepts all ten requests and passes all five gates. A source on a silent direct-output destination still changes the audible FM carrier; changing its pitch or routing it directly changes the sound. Disabling it removes its contribution. Changing disabled sample assets has exactly zero effect, while enabled sample assets change the FM sound. The fixed-phase repeat is exact. These checks justify preserving silent modulators and the off-switch guard used by label preparation. The panel takes 17.34 seconds and peaks below 453 MiB, within the 8 GiB limit.

The [real-source probe](data/corpus_native_summary.json) makes nineteen requests: six original/prepared/repeat triples pass with exact audio, and one original source fails the strict scalar check before audio QA. All random phases are fixed and each request has a fresh process. The failed source remains explicitly unadmitted; the bounded label-copy panel is not complete native corpus acceptance.

The [readback diagnosis](data/corpus_readback_diagnosis.json) reproduces `lfo_1_smooth_time` requested as `0.41464757919311523` and loaded as `0.41464805603027344`. The loaded value exactly matches the encoder's existing native-grid projection. Production tolerances have not changed. A precise rendering-contract correction, including ignored empty-connection state, remains necessary before full corpus streaming/training.

All 65 distinct selected local tests pass across the regression/final runs, including seven corpus tests. The final seven-test run also trains/exports/reloads the full-domain model with a fake renderer; this is not a trained corpus model. Ruff and whitespace checks pass. No real corpus model is trained or promoted. Raw payloads, features, audio and weights remain local; final-test audio stays reserved and Actions stays disabled.
