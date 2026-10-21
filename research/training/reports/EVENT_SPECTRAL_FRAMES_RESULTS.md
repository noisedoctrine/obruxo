# Training-frame coverage repair passes its native check

The frozen local preflight completed on 2026-10-20. **The revised samples retain every tested spectral response across the 12 diagnostic pairs and preserve all valid envelope values exactly.** This repairs the demonstrated data-coverage gap; it does not establish improved model learning or qualify either existing helper.

[Frozen card](EVENT_SPECTRAL_FRAMES_CARD.md) · [Verified evidence](data/event_spectral_frames_summary.json) · [Previous missed-frame diagnosis](SPECTRAL_TIME_COVERAGE_RESULTS.md)

## What changed and what passed

Training can now retain 32 spectral frames per resolution, including note-start/end anchors at offsets 0/25/100 ms and at least eight uniformly selected frames. It retains the entire valid L/R/side envelope. Selection uses MIDI timing, recording length and seeded randomness; it does not use control labels or observed response peaks. Paired recordings select the same times. Variable envelope lengths are padded with false masks during batching.

All 36 native requests passed and matched the historical training audio hashes. All 12 repeats were exact. All 24 records preserve their complete valid envelopes, and every pair has identical sample times. Each of the 36 full spectral responses exceeds the declared signal floor and remains visible in the revised samples. All 1,024 original training recipes fit the explicit limit of eight unique note-event times.

Examples below report mean absolute log1p sound change, **not model error**. Nonzero values show that supervision retains a change. Larger values are not automatically better, because spectral transition sampling deliberately changes the weighting over time.

| Previously missed measurement | Old sampled change | Revised sampled change | Full-recording change |
|---|---:|---:|---:|
| Attack pair 244, FFT-2048 | 0 | 0.001693 | 0.002250 |
| Attack pair 496, envelope | 0 | 0.000652 | 0.000652 |
| Release pair 16, FFT-2048 | 0 | 0.004364 | 0.001432 |
| Release pair 268, envelope | 0 | 0.002625 | 0.002625 |
| Release pair 268, FFT-2048 | 0.0000000295 | 0.001551 | 0.000423 |
| Release pair 501, FFT-2048 | 0 | 0.001478 | 0.000308 |

Complete envelopes remove their previous sampling bias. The selected spectra emphasize transitions and are not unbiased full-recording estimates. The unchanged qualification pipeline still evaluates full recordings. This is a check on the known training failures, not independent model validation or proof that arbitrary future performances retain every important change.

## Local resources and compatibility

The preflight took **68.39 seconds** with **861,667,328 bytes (0.80 GiB)** peak process-tree RAM and zero sampling errors. Its 24 feature records, plan, receipts and reports occupy **33,719,438 bytes (32.16 MiB)**. No waveforms were archived and there were zero learning updates during the preflight.

The largest record is 1,394,937 serialized bytes / 1,390,050 replay tensor bytes. The existing 512-MiB replay limit therefore holds at least 386 records of this observed maximum size; its independent 512-record ceiling also remains enforced. Using the largest observed record for all 1,024 training requests, the parent's retained non-record overhead and a 10-MiB margin projects **1,979,904,596 bytes (1.84 GiB)**, below the 2-GiB archive cap. This sizing estimate does not replace runtime enforcement for future recordings.

Enable the implementation with `run_basic_spectral_helper.py --paired-changes --event-frames ...` only under a new declared training card. The default uniform policy remains available for historical comparisons. New plans/artifact metadata identify the sampling policy, and changed-plan resume is refused. Previously completed runs remain untouched; reproduce them from their frozen revisions rather than resuming them against changed source.

The helper architecture stays at 1,413,033 parameters. The inverse prediction head stays at **14 dense outputs**, reducing to 11 values/selections at codec export. No new controls, LFOs, routes or assets were added.

## Verification and next step

Twenty focused local tests pass: the prior 14 helper/pair/time-coverage checks and six event-frame checks. New checks cover exact frame/envelope preservation, explicit scope limits, padded-loss/gradient masking, collection recovery, pair alignment, exact learning-state recovery, policy compatibility and refusal of unqualified artifacts. Ruff and whitespace checks pass. Publication verifies native attempt/outcome counts, pair membership, audio hashes, preserved responses and all retained feature hashes. The code and card were committed before rendering as `92158d0`.

Next, freeze a short real-data fitting check to determine whether the helper can learn the now-visible attack/release changes before paying for another full 12,000-update attempt. A successful data preflight does not guarantee a successful helper. Both existing artifacts remain unqualified, inverse spectral training remains pending, final-test states stay reserved and Colab stays deferred.
