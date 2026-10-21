# Archived diagnostic sources

`run_destination_check.py`, `run_combined_check.py` and `run_combined_integration.py` qualify the [combined curriculum](../../COMBINED_CURRICULUM.md). They check native routing domains, all registered assets, phase-controlled fresh-process repeats, audible family effects, bounded on-demand collection and exact optimizer/RNG recovery. `summarize_combined_exposure.py` reports actual retained supervision without a learning-quality claim.

`audit_capabilities.py --pool PATH --output NEW_DIRECTORY` produces the [combined recipe matrix](../../COMBINED_COVERAGE.md) for a frozen nested-state/expression pool. It counts all 775 scalars and six nested families on training states only, with no native renders or optimizer updates. It refuses narrow codecs and records runtime hashes. The counts do not qualify audibility or learning.

`run_expression_check.py` and `run_expression_integration.py` are current portable checks for [expressive MIDI](../../EXPRESSIVE_MIDI.md). They qualify note timing, controller effects, MPE isolation and fixed-phase repeats, then verify collection across all eight reference performances, original labels, audio-only inputs, updates and export reload. The integration is a pipeline check without model promotion.

`run_nested_check.py`, `confirm_seed_bit_alias.py` and `run_nested_training_check.py` are current portable checks for [custom curves and seed outputs](../../CUSTOM_CURVES_AND_SEEDS.md). They preserve exact seed bits under a separate native profile, check oscillator-slot effects and run original-audio training across thirteen families. The alias script directly compares both JSON encodings in the pinned native engine; signed JSON is confined to that diagnostic and is not accepted by the unsigned public preset API.

`run_routing_check.py` and `run_routing_training_check.py` are current portable checks for [ordered connections](../../ORDERED_ROUTING.md). They write new output directories and fixed plans before native work. The first checks the complete declared vocabulary and all slot positions; the second runs bounded original-audio training and five validation families. Neither uses the historical layout below.

`run_identity_screen.py` runs the current frozen paired-identity feasibility screen from the checkout. It writes a plan before rendering, shares a bounded feature bank across both arms and measures native reconstruction plus reserved-performance representations. See [PAIRED_IDENTITY.md](../../PAIRED_IDENTITY.md).

`qualify_alpha_imports.py` is the current portable factory-import check, run directly from the checkout. It compares three pinned original exports with the imported alpha assets using native audio. Its arguments and validation boundary are documented in [ALPHA.md](../../ALPHA.md); it does not use the historical directory layout below.

These are the exact scripts used for the existing-artifact analysis and train-only feature probe on 2026-10-15 (Malaysia time). They preserve the original source bytes, including the script hash in `../data/cached_head_probe_plan.json`.

They use the historical local layout: execute copies from `ARTIFACT_ROOT/tmp/`, with this code checkout at `ARTIFACT_ROOT/tmp/training-pipeline-pr/` and the ignored training outputs at `ARTIFACT_ROOT/research/training/outputs/`. They write their diagnostic JSON into the checkout. Reproduction requires the pinned local checkpoints, pools and completed evaluation artifacts, plus the Python dependencies documented in `../../TRAINER.md`. Do not overwrite the frozen results when reproducing them; use a separate checkout and artifact root.

The subsequent native intervention has a separate, portable runner, `../../release_probe.py`, which accepts `--artifact-root`, `--plan`, `--output` and `--resume`. See [the experiment report](../RELEASE_DIAGNOSTIC_RESULTS.md) for its exact invocation, frozen identities and interpretation.

## Sound-helper evidence

`summarize_sound_proxy.py --run /path/to/completed/screen --tests-passed N` checks the completed plan and source hashes, then publishes the sound-helper report, frozen plan, family summaries and a descriptive bootstrap over validation patches. Supply the actual completed local-test count.

`summarize_basic_spectral_helper.py --source COMPLETED_RUN --output NEW_JSON` publishes the basic temporal helper's frozen qualification and resource evidence, including failures. It recomputes the admission decision and checks plan identity, accepted qualification requests and exact repeats. The paired variant additionally reports complete training pairs and descriptive agreement under both native audio objectives.

`diagnose_paired_spectral_helper.py --source COMPLETED_PAIRED_RUN --output NEW_JSON` inspects all saved training pairs against a no-sound-change baseline. It verifies feature hashes and alignment, performs no renders or optimizer updates, and does not qualify an artifact. See [the paired result](../PAIRED_SPECTRAL_HELPER_RESULTS.md).

`check_spectral_time_coverage.py` freezes a 12-pair, 36-render diagnostic plan, then executes it only with `--execute`. Its required `--source`, `--diagnosis` and `--output` arguments and limits are described in [the card](../SPECTRAL_TIME_COVERAGE_CARD.md). `summarize_spectral_time_coverage.py --source COMPLETED_RUN --output NEW_JSON` verifies and publishes the completed evidence. Neither script performs training or promotes the helper.

`check_event_spectral_frames.py --source COMPLETED_PAIRED_RUN --prior-evidence TIME_COVERAGE_JSON --output NEW_DIRECTORY` freezes the sampling repair's native preflight; add `--execute` to run the unchanged plan once. `summarize_event_spectral_frames.py --source COMPLETED_PREFLIGHT --output NEW_JSON` verifies and publishes it without training tensors. See [the card](../EVENT_SPECTRAL_FRAMES_CARD.md). Full helper training exposes the separate opt-in flags `--paired-changes --event-frames`; the successful preflight does not start a full fit.

`run_event_response_fit.py --parent COMPLETED_PAIRED_RUN --repaired COMPLETED_EVENT_PREFLIGHT --output NEW_DIRECTORY` freezes a matched short fitting check; use `--execute` to run it and `--execute --resume` only within the original limits. `summarize_event_response_fit.py --source COMPLETED_FIT --parent COMPLETED_PAIRED_RUN --repaired COMPLETED_EVENT_PREFLIGHT --output NEW_JSON` re-scores both final checkpoints and verifies update accounting before publishing. See [the result](../EVENT_RESPONSE_FIT_RESULTS.md); these are diagnostic checkpoints, not qualified helpers.

`run_native_gradient_screen.py --pool EXISTING_BASIC_POOL --output NEW_DIRECTORY` freezes the separately costed direct native-gradient qualification. Add `--execute` to run the unchanged plan once; no automatic retry/resume or inverse training. See [the card](../NATIVE_GRADIENT_CARD.md).

`summarize_native_gradient_screen.py --source COMPLETED_RUN --output NEW_JSON` reconstructs every recorded native request, numerical gradient and final candidate, checks receipts and recomputes all gates before publishing the [result](../NATIVE_GRADIENT_RESULTS.md). It verifies saved scalar measurements; discarded audio is not re-scored.

`run_native_inverse_fit.py --pool BASIC_POOL --initial INCUMBENT_EXPORT --retrieval MATCHING_TRAINING_CHECKPOINT --qualification NATIVE_QUALIFICATION_JSON --output NEW_DIRECTORY` freezes the matched inverse-training card. Add `--execute` to train, or `--execute --resume` only for that same attempted run and original limits. See [the card](../NATIVE_INVERSE_FIT_CARD.md). It checkpoints both arms every update, logs every native probe loss and does not run validation unless the fit gate passes.

`verify_native_inverse_fit.py --source COMPLETED_RUN --pool BASIC_POOL --initial INCUMBENT_EXPORT --retrieval MATCHING_TRAINING_CHECKPOINT --qualification NATIVE_QUALIFICATION_JSON --output NEW_JSON` replays both complete update sequences from the logged native losses, verifies final models/optimizers/RNG and diagnostic exports, re-scores cached training targets and input features from exact source repeats, reconstructs hard audit decisions and recomputes gates. Candidate audit audio is discarded and is not independently re-scored.

`audit_native_inverse_prefix.py` takes the same paths to audit an interrupted run without new renders. It verifies the actual checkpoint/optimizer/RNG prefix and reconstructs the terminal rejected probe; see [the stopped-run result](../NATIVE_INVERSE_FIT_RESULTS.md). Reproduce from the pinned training source when auditing an older frozen contract.
