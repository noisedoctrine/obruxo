# Native controls now settle before the first note

The stopped inverse run exposed a renderer startup transient. The preset settings read back correctly, but immediate MIDI started while their audio-rate values were settling. In the diagnostic, the four-voice patch peaked at **1.03198** at sample 63; after a silent lead the same requested sound peaked at **0.24395**. Its level response then matched the schema's quadratic scale. The clipping penalty investigated during diagnosis is **not enabled in training**.

[Startup waveform/state evidence](data/native_settling_summary.json) · [Failed shorter lead](data/native_preroll_short_summary.json) · [Qualified settled profile](data/native_preroll_settled_summary.json) · [Frozen cards](NATIVE_PREROLL_CARD.md)

## Qualified profile

New sound-training runs use **32,768 silent frames (0.743 seconds)** before MIDI. Every note/controller is shifted by exactly that count; only the added lead is removed. The first note attack and complete requested tail remain. The lead is synthesized offline; it is not a real-time wait.

| Check | Result |
|---|---|
| Historical zero-lead request | Exact previous audio hash; legacy behavior preserved |
| Seven settled outputs, including four voices/maximum level/fast attack | All pass unchanged strict QA and retain requested frame counts |
| Source and patch repeats | Sample-exact |
| Quadratic level-response relative waveform error | **1.90275e-7**, below the frozen 1e-5 tolerance |
| 32,768 versus 40,960 silent frames | **Zero waveform difference** |

The earlier 8,192-frame attempt removed clipping but failed its original accuracy gates: level error 0.00029038 and settling error 0.00240226. Its largest difference remained at sample 63 and disappeared later in the observation, confirming a decaying startup effect. The longer profile passes the **same** gates. Both results are retained; no threshold was relaxed.

The successful eight-request check took **14.37 seconds**, with **791,842,816 bytes (0.74 GiB)** peak process-tree RAM. All saved native waveforms were independently re-hashed and re-scored, every request was reconstructed, and frozen source was checked against commit `d8945cc`. No model-quality claim follows from this renderer check.

## Contract and recovery

`RenderRequest.preroll_frames` defaults to zero for historical requests. Nonzero values participate in request identity and appear in provenance. Simple notes use a bounded sub-sample guard so host floor/nearest conversion preserves integer note-on, duration and note-off samples; expressive MIDI moves every event without changing channels or release velocities.

The native-gradient and inverse-training runners now default to the qualified settled profile. Qualification must match its profile and renderer/QA/transport source hashes. Cached targets declare their profile, and old cold targets are rejected even when their recipe and file hashes match. Diagnostic exports retain the profile. There are still **14 model outputs**, with no new prediction heads.

The exception-progress bug is also fixed: the outer runner records each committed checkpoint before a later render failure can interrupt fitting. A regression test covers the original incorrect zero-update summary. The stopped run remains unchanged and its 34-update replay audit remains historical evidence, not corrected training.

Renderer, expressive transport, full-readback and recovery checks passed locally; the exact note clock was rechecked after its guard change. Nine targeted profile/cache/training checks also pass. GitHub Actions stays disabled.

## Next training gate

Recheck the numerical-gradient method on newly rendered settled audio under its [new card](SETTLED_NATIVE_GRADIENT_CARD.md). Historical qualification applies to the old observation profile. Do not reuse the old source cache or partially sound-trained models. After that gate passes, freeze a fresh matched inverse trial from the same historical starting model, with new optimizers and fresh target renders. Basic reconstruction, fresh confirmation, listening and release remain unfinished.
