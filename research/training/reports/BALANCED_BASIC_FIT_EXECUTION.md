# Balanced three-sound fit: execution started

The [declared balanced-data comparison](BALANCED_BASIC_FIT_CARD.md) is implemented and started locally at **11:54 on 31 October 2026, Kuala Lumpur time**. Native completion, endpoint quality and final verification are pending. This is one new sound-trained candidate; the preceding audited sound-only model is reused as the control.

The same original untrained bridge, 14 outputs, trainable mask, optimizer, sound objective, rehearsal and acceptance policy remain in force. Every patch receives the same note/duration coverage under the explicit 48-update order. The implementation preserves all **104 old evaluation cells**, adds **36 grid evaluation cells**, and uses **18 additional training-only reciprocal cells**. The cap bank contains **74 distinct input recordings**, compared with the original 56.

**16 focused tests pass**, covering patch-independent sampling, the exact old panel, reciprocal playback coverage, data-contract checks, native acceptance and exact candidate recovery. Ruff **0.16.2** and diff checks pass. The [real-initializer software check](data/balanced_basic_fit_software_check.json) preserves all 104 old starting predictions, matches all 158 logical cold-export comparisons, and exactly recovers model/Adam/trace/RNG. It takes **1.53 seconds / 0.67 GiB**, adding zero native requests or experimental updates.

The [execution declaration](data/balanced_basic_fit_declaration.json) records plan **63e8b02e659bb5b8c78c8f1e2810b1eae23df9a1778fc652316efcc9c0e3d69c**, implementation `8935d1e`, source identities, data/control links and full schedule. Its estimated evidence upper bound is **1.24 GiB**, below the 2-GiB cap. The six-hour / 4-GiB job allows at most 8,000 requests and checkpoints every update. It independently replays its first two native updates and automatically runs the final offline audit and waveform rescoring.

The [first-two-update replay](data/balanced_basic_fit_prefix_verification.json) now passes exactly for model, Adam and RNG with zero added renders. Both proposed updates are accepted. This qualifies continuation of the frozen run; endpoint sound quality remains unmeasured.

The [qualified data bank](BALANCED_BASIC_DATA_RESULTS.md) remains unchanged. The 192 reserved roots are unscored, Actions stays disabled and Colab remains deferred. This run does not advance A2 until the complete frozen reconstruction and preservation gates pass; A3, second seed, human listening and release remain incomplete.
