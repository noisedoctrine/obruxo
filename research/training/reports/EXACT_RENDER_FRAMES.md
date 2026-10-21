# Exact native render length: regression fixed

Verified locally on 2026-10-20. All 16 duration-related failures from the [A1 helper run](BASIC_SPECTRAL_HELPER_RESULTS.md) now render the exact requested number of samples and pass the training stream's audio checks.

For the failing 150-BPM/two-beat performance with a two-second tail, the request requires 123,480 stereo frames. Converting that count to seconds and back yields `123479.99999999999` in floating-point arithmetic. The old native call returned 123,479 frames, which the strict stream correctly rejected. A separate 154,350-frame control already returned the right length.

The renderer now asks for one guard sample beyond the requested end and copies exactly the requested prefix. It still rejects genuinely short or malformed output; it does not pad missing audio. The frame policy is recorded in renderer provenance, so an in-progress collector cannot silently switch policies during recovery. Preset, MIDI, tail and request identities remain unchanged.

[Native evidence](data/exact_render_frames_summary.json) covers all 16 failed requests, an additional repeat of the first repaired case and two renders of the previously correct control: **19 post-fix requests**, all accepted, in 28.49 seconds. Both phase-fixed repeat pairs were sample-exact. The control audio hash matches its pre-fix hash. Including the two reproductions before the fix, this check used 21 native requests and zero optimizer updates.

All **33 focused renderer/stream tests pass**, including the floating-point boundary, exact sample-prefix preservation, and rejection of short or malformed buffers. Ruff and whitespace checks pass. The old experiment's failures and unqualified helper remain unchanged; this pipeline repair does not establish useful model gradients. The next learning comparison must still target that separate failure.
