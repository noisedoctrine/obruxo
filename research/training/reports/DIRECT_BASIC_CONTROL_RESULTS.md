# Three individual sounds fit successfully within the existing controls

All three predeclared direct-control cases pass. The solver adjusts the same nine continuous controls and two category selections, starting from the qualified untrained audio model's prediction. It uses measured native sound loss on both held-note and original performances, without source-control supervision or network training. This establishes individual fitting on three already-used training patches; it does not establish a working shared audio-to-patch model.

| Held note | Initial joint sound error | Direct final | Matched settings-trained control | Reduction from initialization |
|---|---:|---:|---:|---:|
| MIDI 36 | 0.87550 | **0.12216** | 0.26912 | **86.05%** |
| MIDI 61 | 0.49147 | **0.20087** | 0.46862 | **59.13%** |
| MIDI 83 | 0.77838 | **0.46301** | 0.72033 | **40.52%** |

Every case improves each performance's combined sound error by at least 10%, improves MRSTFT on both performances, lowers joint error by at least 25%, and beats its matched settings-trained control. All 96 scheduled updates complete: 73 continuous moves and four category changes are accepted. No case reaches the eight-refusal stop. The final categories happen to match the source categories; exact parameter identity was not an objective.

Plan `642b1d087a02752b9ec38d6038a328da8bf3f7dd77349bd0833f0ac5d23f5a1a` uses **2,598 accepted native requests**, with zero invalid trials, timeouts or recoveries. Native execution takes **1 h 44 min / 1.00 GiB peak RAM**, retaining **36.88 MiB before the summary**, within four hours / 4 GiB / 2 GiB and the 4,608-request cap. The [completion record](data/direct_basic_control_service_completion.json) confirms normal exit and release of the scoped awake request. The 14-output / 11-codec interface remains unchanged.

The [independent publication audit](data/direct_basic_control_verification.json) replays all controls, category decisions and Adam states exactly from the recorded 2,598 scalar measurements. All 24 retained initial/final/repeat observations are rescored; fixed-phase repeats are exact. The first audit takes 45.64 seconds; the terminal publication replay takes **30.00 seconds / 0.69 GiB**, adding zero renders or optimizer applications.

The first verifier also hashed its supervisor's still-pending operational status. After verification returned, that one file became completed. [Publication provenance](data/direct_basic_control_publication_plan.json) reconstructs the exact original Windows bytes and proves this was the only changed record: all scientific evidence and all 48 pinned source files are unchanged. A separate terminal snapshot retains both status history and the original verification, then repeats the full independent audit. Its complete evidence manifest verifies after publication; no failure or receipt is overwritten.

The [frozen protocol](DIRECT_BASIC_CONTROL_CARD.md), seventeen passing software tests and [prepared execution](DIRECT_BASIC_CONTROL_EXECUTION.md) remain available. The result supports retaining the small control vocabulary and sound objective. Next establish that one shared audio-to-patch model can learn these three cases; preserve full six-group/earlier-32 admission requirements before fresh confirmation. The 192 reserved roots, second seed, clean human tone listening and model release remain unfinished. No model is promoted.
