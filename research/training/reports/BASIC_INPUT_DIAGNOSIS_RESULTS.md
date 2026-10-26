# Saved failures and a faster blind listening page

The [saved-prediction diagnosis](data/basic_input_control_diagnosis.json) counts each of the twelve training patches once per input recording. Both models select the source unison count correctly on all twelve original recordings. On held notes, the parameter control selects 6/12 correctly and the sound candidate 5/12. Both also have large continuous-control errors there. This does not isolate unison as the sole cause of the reconstruction gap.

| Candidate input | Level error | Pan error | Attack error | Release error |
|---|---:|---:|---:|---:|
| Original | 5.16% | 3.45% | 5.74% | 9.84% |
| Held note | 35.67% | 26.08% | 36.85% | 32.43% |
| Alternate | 30.21% | 19.49% | 37.32% | 26.31% |

Errors are mean fractions of the allowed control range, using source-active controls. They help describe the failed predictions; exact source-parameter recovery is not the acceptance objective, and these numbers do not establish an audible cause or select a new architecture. The [independently verified sound result](TRAINABLE_BASIC_FIT_RESULTS.md) remains the quality decision.

The [frozen diagnostic selection](data/basic_input_diagnosis_plan.json) prepares twelve anonymous source/candidate/control comparisons across eight independent patches, covering median and large-gap examples in all six groups. Full retained native waveforms are exported without loudness normalization, with exact float32 sample round trips and checked file/receipt hashes. Preparation takes **7.86 seconds / 0.39 GiB**, retaining 98.58 MiB within five minutes / 4 GiB / 128 MiB. There are zero new renders or optimizer updates. All [36 served WAVs](data/basic_input_listening_preview_verification.json) match their packet hashes.

At the listener's request, every play button now pauses other clips, seeks to zero and starts immediately; audio is preloaded. All response options are visible radio pills with keyboard support. Browser drafts preserve progress across reloads. The [live page update](data/basic_input_listening_page_refresh.json) preserves the packet identity, audio, anonymous assignments and six selections already made by the actual listener; the previous page/key are archived privately. No judgments are invented or decoded from incomplete responses.

Four diagnosis tests, six packet/refresh tests and four page-behavior tests pass, alongside Ruff/diff checks. The live browser confirms zero dropdowns, all 96 response pills, restored selections and playback restarting near zero with one active clip. The listener has begun the comparisons; a complete response file is still required. The answer key and audio remain local. This deliberately selected diagnostic packet cannot qualify a release or substitute for fresh confirmation.
