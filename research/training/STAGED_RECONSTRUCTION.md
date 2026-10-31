# Basic reconstruction experiment protocol

Revised 31 October 2026. The [training plan](../modelling/TRAINING_PLAN.md) sets priorities; this document defines the common experiment requirements. The [review](reports/TRAINING_REVIEW_20261031.md) preserves the reasoning and evidence. Existing cards remain authoritative for their frozen runs. The [previous detailed protocol](https://github.com/noisedoctrine/obruxo/blob/ced6b0b/research/training/STAGED_RECONSTRUCTION.md) remains available; this consolidation changes no recorded result, running deadline or split.

## Current contract and status

Stage A predicts nine continuous controls, three oscillator-unison scores and two filter-enable scores: **14 dense outputs, 11 control values at codec entry**. Oscillator 1 level/pan/detune, filter 1 cutoff/resonance and amplitude ADSR are the continuous controls. All other patch state comes from the pinned basic template. No LFO, route, map, custom-curve or seed heads exist in this model.

| Stage | Meaning | Current result |
|---|---|---|
| A0 | Legal differentiable/control interfaces and bounded resources | Passed for the declared basic implementation. |
| A1 | Proposed sound updates improve real native audio | Corrected native estimator/production-choice path passed bounded qualification. Learned helpers failed and remain refused. |
| A2 | One shared audio-only model learns reliable basic reconstruction | Unfinished. Direct fits pass individually; shared fits fail important input/preservation gates. Balanced-data fit is running. |
| A3 | Independent quality, replication, listening and usable bundle | Unfinished and conditional on A2. Some inference/runtime components already pass. |

Pipeline connectivity is not A1, A1 is not A2, and a three-patch fit is not A3. No implementation checklist can substitute for the next stage's evidence.

## Before a new run

Freeze a concise experiment card and machine-readable identities before collecting/scoring new outcomes. Include:

- The question and changed factor, expected decision for each outcome, matched comparator and reason to use new native work rather than saved evidence.
- Source/ancestry groups, data split, input and playback performance roles, phase/renderer/frontend/codec identity, feature/replay policy and every evaluation denominator.
- Exact model, trainable/frozen counts, dense outputs, optimizer/update schedule, loss terms/weights, production decoding and any approximate derivative.
- Planned native queries by purpose, charged optimizer attempts, acceptance/rollback behavior, resource caps, checkpoints, stopping/recovery rules and the original deadline.
- Fit versus generalization versus release criteria, earlier-sound preservation, treatment of invalid outputs, and which outcome permits further work.

Use the qualified local renderer, fixed oscillator 1–3/sampler phases and 32,768-frame preroll for controlled correctness comparisons. Keep other source state and source lineage pinned. Same-runtime exactness needs controlled phases and retained tensors; random-phase variation is a separate training policy. Never loosen waveform tolerance to conceal uncontrolled phases.

Native feedback is a measured numerical estimate. Production/export uses deterministic hard argmax; training uses the qualified production-choice forward path with its declared backward approximation. Loss-only gradient connectivity, category calibration and inward-boundary checks remain necessary when those components change. Do not assume Gumbel-softmax supplies derivatives through the plugin itself.

## The next decision

Complete and audit the [balanced fit](reports/BALANCED_BASIC_FIT_CARD.md) unchanged. It reuses the audited sound-only control, trains on identical note/duration coverage for all three patches, and retains 104 old plus 36 grid evaluation cells. The additional eighteen reciprocal cells are training-only. Its 48-update/six-hour/8,000-request limits and full per-input/six-group/earlier-32 gates are not revised here.

Before another long fit, isolate process/plugin/state/synthesis/QA costs on the actual basic path, then qualify and measure an efficient worker policy. A twelve-render profile confirmed sample equality but did not establish a reuse speedup or pure DSP cost. Do not translate the old `render_seconds` timer into synthesis-only speed.

If balanced learning fails, the next learning diagnostic is an inexpensive shared supervised fit on that same small balanced bank, with real-render endpoints. It asks whether the inputs/features/model can learn a valid mapping. It is not final parameter-matching training and does not authorize another unchanged large initialization. If that baseline succeeds and native learning fails, test feedback/exposure or a qualified optimization-and-distillation route before adding architecture. Keep these as alternative decisions, not a factorial sweep.

Input LUFS/peak normalization remains a later independent comparison; it does not enter the running fit. The plan specifies preserving the gain and original-level loss/export semantics before that experiment is implemented.

## Data and evaluation

Generate further training audio on demand from immutable patch/performance recipes, with bounded queue/replay. Cached identical features are allowed when actual audio/frontend hashes agree. No full pre-rendered corpus is required. Pairing and ancestry must survive failures and resume; do not quietly drop difficult sources.

Cross pitch/duration/pattern with patch identity/category. Inspect the construction directly for shortcuts, including perfect pattern/filter associations. Use event coverage that retains attack, sustain and release. The current basic template ignores velocity, so identical low/high-velocity audio cannot establish velocity isolation.

Evaluate one inferred patch under its input and changed performances. Source MIDI is for synthesis/evaluation, never an inverse-model input. Compare fixed template, train-only retrieval, matched settings-trained model and compatible earlier sound model on the same cases. Score legal hard exports through real Vital; include invalid/silent/clipped cases in the declared failure denominator.

Keep combined spectral/envelope/stereo loss and MRSTFT separately, plus level, median/tail, per-input/per-family errors and earlier-sound preservation. Retain original amplitude/timing/channels in the primary raw loss. Loudness-matched tone metrics/listening are separate views. Neither correct category labels nor low probe pitch accuracy establish audible success.

Freeze practical improvement, preservation and tail thresholds before scoring. The working development/confirmation margin remains 10%; existing cards retain their stricter group-specific rules. New diagnostic criteria cannot relabel old failures or bypass full A2/A3 admission. Independent confirmation needs entire new families and reserved performances, second-seed evidence and actual blinded judgments. Existing reused panels are development data. The 192 reserved roots and final test remain unavailable for tuning.

## Recovery, evidence and review

Checkpoint complete model/optimizer/RNG/sampler state at declared update boundaries. Record requests, outcomes, accepted/refused steps, timeouts and all recovery charges. Resume only with matching contracts and the original deadline. A known identical retry may be used only within the frozen allowance; preserve the failed attempt. A missing sandbox process listing is not evidence an escalated process died.

Verify fixed-phase source/export equality, same-runtime state recovery and retained native endpoint scores. Replay from stored training scalar measurements must be described separately from independent waveform rescoring. Preserve all failed results and immutable scientific evidence; live heartbeat files are operational status, not final evidence.

Use a single reusable experiment path where possible. Review one substantive outcome/capability per PR, consolidate routine status records, keep reviewable PRs ready and run appropriate local checks with Actions disabled. Approval to merge is separate from permission to implement or run an already-authorized bounded experiment. No automatic merges or follow-up training launches.
