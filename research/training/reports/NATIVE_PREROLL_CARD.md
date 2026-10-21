# Fixed pre-roll renderer qualification

Declared before execution on 2026-10-21. The startup diagnostic shows the same patch peaking at 1.032 without a lead and 0.244 after control settling. The proposed clipping-penalty continuation is not enabled. Correct the rendering observation before collecting new training targets.

The new request field `preroll_frames` defaults to zero for historical request identity/audio. Nonzero values are included in the request hash. Shift all MIDI by that exact sample count, render the additional lead and trim only that lead. Returned frames and original note/tail duration stay unchanged. Use whole audio buffers; qualify **8,192 frames** as the new training profile, comparing it with 16,384 frames for settling stability. Guard simple note API time conversion below half a sample; the expressive transport keeps every controller, channel and release velocity.

The frozen native check contains eight requests: the historical cold rejected patch; source/repeat and rejected-patch/repeat with 8,192-frame pre-roll; a level-reduced version; the same rejected patch with 16,384 frames; and a four-voice maximum-level/fast-attack fixture. The cold audio must retain its historical hash. All pre-rolled outputs must pass strict QA with the original frame count. Source and patch repeats must be sample-exact. Relative waveform error must be at most 1e-5 both for the documented quadratic level response and for 8,192 versus 16,384 frames. No normalization, limiting or true-attack cropping is permitted.

Caps: eight native requests, ten minutes, 8 GiB process-tree RAM and 64 MiB output. Keep unmodified native waveforms, receipts and source hashes. This qualifies a renderer profile, not model reconstruction quality. Old target caches and partially sound-trained models cannot be relabelled as using the corrected profile. The numerical-gradient path must be checked on the corrected profile before a fresh, separately costed inverse run.

Final plan ID: `e66bbb1a7ed699297441d19305328bd2c9ed8a33eede6c8fd25037a05ae2c198`. The earlier unexecuted draft was replaced before any native request to include the guarded simple-note clock.

Fifty-six focused local checks passed across rendering, expressive transport, full readback and inverse recovery, including the exception-progress regression. After tightening the simple note clock, all 22 renderer unit checks pass again. Commit the profile source before native execution.
