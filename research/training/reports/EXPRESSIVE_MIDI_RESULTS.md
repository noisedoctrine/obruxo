# Expressive MIDI rendering and training ingestion

Completed locally on 2026-10-17. Pitch bend, channel pressure, modulation wheel, sustain pedal, MPE slide and release velocity now reach Vital through a separately identified all-event MIDI profile. The performance bank contains eight reference and eight transfer performances. Native behavior and training ingestion pass. This is a capability increment; no reconstruction-quality experiment or model promotion is claimed.

## Native correctness

| Check | Result |
|---|---|
| Note transport | Eleven fixtures: the original eight phrases and sample-boundary fixtures at 44.1, 48 and 96 kHz |
| Expression effects | Eleven fixtures, including global and per-note bend/pressure/slide, release velocity and pedal |
| MPE isolation | Three two-note fixtures; the other note's measured amplitude changes by at most `1.50e-7` relative |
| Expanded performance bank | All eight new reference/transfer phrases render and repeat exactly |
| Repeatability | All 33 fixtures repeat exactly with oscillator and sampler random phase fixed |
| Total | 93/93 attempts accepted; all eight native gates pass |
| Resources | 86.41 seconds; 485.16 MiB peak process memory; no memory-sampling errors |

The initial qualification accepted all 91 attempts but correctly failed its old/new note-audio equality gate for one chord. Diagnosis located the difference at note-off: the earlier note API converts `61740 / 44100` back to integer sample `61739`. The new transport retains the intended half-even boundary, sample `61740`. Moving only that diagnostic file's note-offs one sample earlier reproduces the old audio exactly, as does repeating the old API. The final plan adds both checks. It retains `notes_audio_exact: false` for this chord and requires the specific verified float-floor explanation; it does not introduce a general numerical tolerance. The other ten note fixtures match the previous API exactly.

The [initial failed summary](data/expression_initial_native_summary.json) and [four-render diagnosis](data/expression_noteoff_diagnosis.json) remain visible. The final [native plan](data/expression_native_plan.json) and [summary](data/expression_native_summary.json) record the changed qualification explicitly. This is an intentional timing correction under a new renderer identity; earlier artifacts keep their transport.

MIDI events retain integer half-even sample offsets through an exact quarter-sample SMF clock. Vital processes controller response at its native control rate, so measured changes can begin slightly before the nominal sample inside that processing block. Every measured response is within the declared 128-sample host-buffer window. This is not sample-exact controller response. Tempo automation, polyphonic key pressure and CCs beyond 1/64/74 remain rejected.

## Training integration

The new codec inherits the three custom curves, exact seed bits, eight LFOs, ordered connections and mapping outputs. Four additional expression sources occupy unused connection slots. Existing connections and positional amount references stay intact. The declared source curriculum enables MPE, uses a two-semitone bend range and lowers stored master volume to `2500` (-30 dB). Audio is not normalized after rendering.

| Measurement | Result |
|---|---|
| Frozen authored prefix | 32 states: 25 train, two validation, five test; held-out audio untouched |
| Native feature collection | 16/16 accepted, zero rejections/errors; nine distinct training states |
| Performance coverage | All eight reference performances, twice each; transfer performances excluded from training |
| Model inputs | Audio-derived timbre and frozen Basic Pitch only |
| Original supervision | Every retained LFO/map/custom-curve label and state/performance recipe verified |
| Model | 5,730,849 trainable parameters; 7,187 continuous and 898 categorical outputs |
| Updates | Three CPU updates, batch four, four threads; diagnostic weights only |
| Retained replay | 1,084,160 bytes: 618,560 features + 49,664 LFO + 397,312 map + 18,624 custom-curve labels |
| Training segment | 28.01 seconds, including 3.28 seconds feature extraction and 0.94 seconds updates; checkpoint work 16.35 seconds |
| Peak process-tree memory | 1,013.92 MiB, below 8 GiB; no sampling errors |
| Recovery/export | Saved model reload matches; frozen frontend unchanged |
| Gates | All nine integration gates pass; strict tiny-overfit remains false |

The optional `performance_round_robin` sampler covers the entire reference bank before the next rotated set of shuffled states. It visits every state/performance pair once per epoch. Saved performance identities are checked during recovery. Earlier sampler defaults and contract identities are unchanged. A previously exported nested-state model also loads unchanged under this runtime.

The [integration plan](data/expression_integration_plan.json), [summary](data/expression_integration_summary.json) and [evidence manifest](data/expression_evidence_manifest.json) retain source hashes, resource measurements, bank identities and local artifact hashes. Plans pin raw runtime bytes; the manifest also records CRLF-to-LF normalized source hashes for checkout comparison. Runtime sources were unchanged during both completed runs. The environment was Windows 11, Python 3.12.13, DawDreamer 0.8.3 and Torch 2.12.1+xpu, executing on CPU. Native checks and regression tests overlapped; these timings are not isolated throughput measurements. Only metadata is committed. No waveform archive, third-party asset payloads or weights are included.

## Remaining work

The full training suite plus MIDI/renderer checks passed: 344 tests in 925.40 seconds. The additional one-sample boundary regression, added after that run collected its tests, passed separately. Ruff, whitespace checks and 67 staged-source comparisons against the completed plans pass. Tests cover legacy compatibility, legal/unsupported MIDI, original labels, sampler coverage, audio-only inputs and exact training recovery.

Expressive audio ingestion does not establish learned invariance, useful held-out transfer or broad combined control coverage. The next task is a declared capability matrix and combined sampler/voice/effect/connection curriculum, followed by measured scaling and quality gates. One-point curve inputs, arbitrary graph/layout coverage, robustness and release qualification remain open. Free local CPU stays qualified; Linux/Colab, final-test audio and GitHub Actions remain untouched. Reproduction commands are in [EXPRESSIVE_MIDI.md](../EXPRESSIVE_MIDI.md).
