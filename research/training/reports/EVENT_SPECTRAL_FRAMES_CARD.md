# Verify transition-aware training samples before another fit

Declared 2026-10-20 before native execution. The [time-coverage diagnosis](SPECTRAL_TIME_COVERAGE_RESULTS.md) found short control responses missing from sampled training measurements. This change preserves the complete valid L/R/side envelope and changes each spectral resolution from 16 uniformly sampled frames to 32 selected frames. It does not change the helper architecture, 14-output inverse head, spectral measurements or native qualification gates.

## Fixed sampling policy

For every unique note start and end, select the nearest valid spectral frame at offsets **0, 0.025 and 0.1 seconds**. Break equal-distance ties toward the earlier frame and deduplicate anchors. Fill the remaining slots uniformly without replacement to reach 32 sorted frames at each FFT resolution. Permit at most eight unique start/end times, giving at most 24 anchors and at least eight broad-coverage frames. The current held/repeated/melodic/chord training generator fits this bound; unsupported performances are rejected instead of silently dropping transitions.

Keep every valid envelope frame. Batch recordings of different lengths with explicit false time masks; padded values contribute neither loss nor gradient. Original/changed pairs use identical MIDI, lengths and RNG seeds, so they retain exactly the same times. Selection never consults source control labels or observed sound-change magnitudes. It deliberately emphasizes transition times and is not an unbiased estimate of full-recording spectral loss. Qualification continues to use complete recordings.

The implementation is opt-in through `--paired-changes --event-frames`. Plans and exported helper metadata record the policy; resume rejects changed plans. Existing uniform-mode artifacts retain their original metadata and loader support. Replay still obeys both 512-record and 512-MiB ceilings, evicting records when either bound is reached. Historical experiment reproduction requires its frozen source revision; no old measurements or checkpoints are rewritten.

## Native preflight

Use the same 12 training pairs selected in the previous diagnostic: three each for attack, decay, release and detune. Keep their original targets, performances, phases and pair IDs. Render original, changed and original repeat: **36 requests**, with no substitutions or retries. Require all audio hashes to match the historical training receipts and all 12 repeats to be exact.

For each spectral resolution, require a sampled mean absolute log1p response greater than `1e-7` whenever the full-recording mean exceeds `1e-7`. Preserve every valid envelope value exactly and verify pair time alignment. This gate checks that the known response signal remains visible; it is not a model-quality score or proof that all future responses are captured.

Save the 24 new sampled feature records with checksums, retaining no waveform archive. Measure serialized record and replay bytes. Project a 1,024-record archive using the largest observed new record, the parent run's retained non-record files and a 10-MiB margin; require it below 2 GiB. This is a sizing estimate, not a guaranteed maximum for future recordings. The actual collection and replay caps remain enforced during training.

Limits: **36 native attempts, zero learning updates during the preflight, 15 minutes, 8 GiB process-tree RAM, 64 MiB new diagnostic files**, one native worker and two CPU threads. Pin source, policy, recipes and parent evidence in a plan before execution. Any failed gate stops advancement and remains reported. No third full fit starts as part of this check.

Nineteen local tests pass before collection, including five new checks for exact event/envelope preservation, explicit performance limits, masked variable-length batches, aligned collection recovery and exact optimizer/minibatch recovery. Ruff passes.

Prepare with `check_event_spectral_frames.py --source COMPLETED_PAIRED_RUN --prior-evidence SPECTRAL_TIME_COVERAGE_JSON --output NEW_DIRECTORY`; execute the unchanged plan with the same arguments plus `--execute`. A successful preflight permits a separately declared fitting test, not inverse training or helper promotion. All work stays local.

Frozen plan ID: `5e39ceda09145ab31fdd3fa0fe5db6ca9ba920947f5e939c8fdbcacad428626f`. The selected IDs remain attack 8/244/496, decay 12/264/497, release 16/268/501 and detune 21/256/489. All 1,024 original training recipes pass the new event-count limit before execution.
