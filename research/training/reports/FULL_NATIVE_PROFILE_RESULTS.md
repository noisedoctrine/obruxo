# Full-preset native profile

Completed 2026-10-16 on the local Windows CPU. The separate full-profile renderer passed **36/36 native attempts** across twelve independently authored fixtures. All three renders of each case had identical float32 audio, verified state and provenance, including fresh and reused worker processes. All six qualification gates passed.

| Check | Result |
|---|---|
| Supplied scalar controls checked per render | 775 |
| Ordered modulation slots checked per render | 64 |
| Native ramp defaults checked per render | 128 |
| Nested structures checked | LFOs, custom warps, random seeds, modulation routes, wavetables, sample |
| Fixed-phase exact comparisons | 12/12 cases, three renders each |
| Worker lifetime | Six reused processes, four fresh-plugin jobs each |
| Total qualification time | 97.56 seconds |
| Peak process/application memory | 392.90 MiB, below the 8 GiB gate |
| Memory sampling errors | 0 |
| Local regression checks | 135 passed after integrating current main; final 17-check profile rerun also passed |
| Static checks | Ruff and branch whitespace checks passed |

The [frozen plan](data/full_native_profile_plan.json) pins the complete case order, request identities, source hashes, contract identity, phase controls, timeout, attempt and memory limits. The [summary](data/full_native_profile_summary.json) records every case and gate. The CLI preserves per-render readback and QA in the local run directory. The final source hashes match those used for qualification.

State equality here compares the native verification records and requested/applied JSON hashes. It does not compare the entire saved VST3 byte stream.

The fixtures exercise the third oscillator and expanded ordinals, LFO/custom curves, remapped routing and a connection in slot 64 to slot 1's amount, mono/stereo sampler playback, voice/portamento settings, two effect orders, explicit seeds and a Shepard component. All oscillator phase-randomization amounts and sampler random phase are zero for correctness. Uncontrolled-phase behavior is a separate question; training pools are not rewritten.

Wavetable version headers migrate explicitly to `1.6.4`. Sample readback has the reviewed four-zero prefix, unchanged length, and at most one PCM16 unit of aligned rounding. That comparison is deliberately not described as a byte-exact sample round trip. Other asset drift is rejected. Per-request sample rounding measurements remain in QA; renderer identity describes the fixed verification policy and stays constant across patches.

The shared validator also exposed a Windows checkout defect: Git converted the hash-pinned template to CRLF. An explicit LF rule preserves the approved bytes and hash. The original `research/vital/full_contract.py` import remains compatible with the shared implementation.

This completes the broad native rendering boundary. It does not complete broad learning or reconstruction-quality gates. These are authored correctness fixtures using the reviewed template and explicit test derivatives; they do not certify every factory asset or every legal combination. Broad learned targets, registered asset/curve banks, training population, learning curves, Linux/Colab qualification and final release checks remain open. The final-test reserves were untouched, no WAV archive was created, and no paid compute was used.

Reproduce from a full repository checkout with the pinned data-generation environment:

```powershell
python research/training/run.py full-check --output research/training/outputs/my-full-profile-check
python -m pytest research/training/tests/test_full_render.py research/training/tests/test_render_worker.py research/training/tests/test_stream.py research/training/tests/test_conditioning.py research/vital/test_full_contract.py research/data_generation/tests/unit/test_render.py research/data_generation/tests/unit/test_juce_codec.py -q
```
