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

The [readback diagnosis](data/corpus_readback_diagnosis.json) reproduces `lfo_1_smooth_time` requested as `0.41464757919311523` and loaded as `0.41464805603027344`. The loaded value exactly matches the encoder's existing native-grid projection. Earlier rendering contracts retain their original tolerances and failures. The separate profile below supplies the precise correction before full corpus streaming/training.

All 65 distinct selected local tests pass across the regression/final runs, including seven corpus tests. The final seven-test run also trains/exports/reloads the full-domain model with a fake renderer; this is not a trained corpus model. Ruff and whitespace checks pass. No real corpus model is trained or promoted. Raw payloads, features, audio and weights remain local; final-test audio stays reserved and Actions stays disabled.

## Separate native corpus contract

The new [plan](data/corpus_readback_native_plan.json) and [result](data/corpus_readback_native_summary.json) qualify the renderer suffix `-corpus-readback-v1`. Continuous values outside the earlier tolerance are accepted only at the exact declared normalized float32 round trip; no scalar input is changed. Discrete values remain exact. Ignored empty-connection mappings are omitted only from the applied copy, with a path ledger. The request bytes, active mappings, musical phase controls, seeds and assets remain intact.

All ten gates pass: 26 successful requests and two expected legacy failures in 48.56 seconds, peaking at 497,639,424 bytes (475 MiB) against an 8 GiB limit. Each of eight comparison groups has identical float32 audio hashes across fresh-process repeats and precise legacy references. The two real training sources also match their label-prepared diagnostic copies exactly. Six authored groups exercise active singleton/decimal LFOs and mappings with custom modes 13, 14 and 15. Every successful new-profile request verifies 775 scalars and all six nested state families. All four random phases are fixed for these checks; actual training retains requested phase settings.

The first qualification fixture incorrectly projected every scalar in its legacy references. Actual original/label/repeat groups were exact, but the overbroad reference edits changed the sound. The corrected fixture changes only the diagnosed smoothing field, or only the ignored empty mapping. It retains exact audio requirements; no production adjustment or tolerance widening was needed for that fixture correction.

All 75 selected local regression tests pass, including ten new corpus-profile checks. Ruff and whitespace checks pass. The publisher verifies source hashes, the new codec identity, unchanged earlier corpus/curve/combined codec identities and the unchanged LFO bank. The source snapshot is verified under its original contract; its frozen splits and sixteen-performance bank remain intact. This bounded check resolves these two diagnosed loading issues. Population-wide native/audio admission, original-source streaming, mixed training, decoder refitting and model quality remain open. No final-test audio is rendered and no model is promoted.

## Original-source training stream

The private source pool contains all 904 eligible training states while retaining the original 9,592-state manifest. It verifies bytes, state identity and re-encoded targets before each render; the normal render request canonicalizes formatting while preserving the complete original document and phase controls. Model exports contain the codec and registered assets, without private corpus paths or source presets. The builder supports an explicit root relocation without changing dataset identity; every relocated source still has to pass the same checks.

The [training plan](data/corpus_source_training_plan.json) and [result](data/corpus_source_training_summary.json) pass all eleven gates. Eight distinct real presets supply eight audio-only feature examples, used in each of three updates. All successful renders verify 775 scalars and all six state families. There are nine attempts: eight accepted and one rejected for `audio.clipping`. The rejection remains recorded, with no source-control edits or dataset removal. This is a pipeline integration check, not a quality result or native admission of all 904 states.

The collection process stopped after six accepted examples. It was confirmed no longer running, then resumed from its acknowledged checkpoint; those six features were retained and only two more examples were rendered. A subsequent step-one recovery exactly matches model, optimizer, RNG, batch RNG, replay and cursor at step three, with zero recovery renders. The frozen frontend and export/reload checks pass. Peak process-tree memory, including dataset loading, is 1,633,271,808 bytes (1.53 GiB), within the 8 GiB budget.

The initial integration configuration exposed a genuine observation gap: six training states have a delay-plus-attack longer than any existing reference note. The longest reference is 1.5 seconds; the maximum requested attack span is 32.0003 seconds. The complete corpus therefore fails the existing attack-compatible sampler's preflight. The corrected source-integration check uses the existing state-round-robin schedule and retains every state. It does not claim that this schedule observes every patch adequately. A separately declared longer-note path is the next required capability, followed by authored/corpus mixing, broader source QA, decoder refitting and useful broad model training.

All 48 selected local tests pass, including four new source-pool tests for original documents/phases, root relocation, changed bytes/targets, membership/path/policy rejection, and exact training recovery/export. Ruff and whitespace checks pass. Public evidence contains metadata only; dataset paths, presets, features, audio and weights stay local. Final-test audio remains reserved and Actions remains disabled.
