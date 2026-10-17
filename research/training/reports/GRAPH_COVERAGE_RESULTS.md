# Complete connection coverage and native responses

The [connection curriculum](../GRAPH_COVERAGE.md) closes the earlier 326-of-431 destination gap without removing patch components. The [training audit](data/graph_coverage.json) finds all 431 destination names, all 32 source names and all 64 ordered slots under declared active supervision. It retains 769 varied scalar controls and six declared constants.

## Frozen population and evaluation

The full pool contains 1,024 states: 820 training, 108 validation and 96 reserved test descendants. These remain only 205, 27 and 24 independent families, respectively, exactly matching the original frozen ancestry. Each family keeps its original curve-input state and adds three destination variants. No family moves between splits. Independent seeded destination permutations avoid alphabetical held-out subsets.

Related evaluation cases are averaged within their frozen family before checkpoint selection and bootstrap resampling. Missing case scores make that family incomplete. Tests cover unequal family sizes, failed views, mixed/missing ancestry, legacy statistics and actual evaluator propagation. The static audit takes 1,215.52 seconds, including full pool reconstruction and validation; it renders no audio.

## Native sound checks

All 1,506 native requests across five panels pass audio QA, scalar/nested readback, exact connected repeats, source integrity and the 8 GiB memory bound. Oscillator and sampler random phases are fixed, with a fresh process for every request. Every one of the 431 destinations changes the sound in at least one explicit context. These are contextual responses, not a claim that every control matters in every patch or is recoverable from every recording.

| Panel | Requests | Targets checked | Targets responding | Seconds |
|---|---:|---:|---:|---:|
| [Initial complete sweep](data/graph_base_summary.json) | 1,293 | 431 | 369 | 2,664.01 |
| [Envelope/effect prerequisites](data/graph_contexts1_summary.json) | 51 | 17 | 17 | 95.25 |
| [EQ, filter and LFO modes](data/graph_contexts2_summary.json) | 54 | 18 | 15 | 103.46 |
| [Wider unison and reverb](data/graph_contexts3_summary.json) | 66 | 22 | 16 | 117.63 |
| [Connected macros, custom warp, filters and multiple notes](data/graph_contexts4_summary.json) | 42 | 14 | 14 | 78.87 |

The initial zeros remain in the published evidence. They exposed insufficient contexts: disabled distortion filtering, full-sustain envelopes, flat EQ shelves, LFO smoothing bypassing fade behavior, insufficient unison voices, inactive filter/warp modes, disconnected macros and a single note unable to exercise polyphony/glide/bend behavior. Explicit alternative settings and MIDI are stored in each panel's plan. No zero was used to declare a fundamental destination unsupported. The final [compatibility receipt](data/graph_compatibility.json) records the union: 431 observed responses and no unresolved destinations.

Rendering remains offline. The 44.40-minute initial sweep includes three native requests per destination, process startup, readback and repeat checks. Summed panel times are 50.99 minutes; some panels ran concurrently. Peak native panel memory is approximately 481 MiB. These diagnostic timings are not a projected training epoch.

The initial checker, complete data audit and training run are pinned to the source snapshot recorded in the compatibility receipt. The final checker adds explicit MIDI, macro exits, probe direction after overrides and a guard against overriding random-phase controls. The receipt verifies earlier fingerprints against their committed source and confirms that only this diagnostic helper differs. Training sources, renderer import, parent codec identities and the train-only LFO bank remain unchanged.

## On-demand training and recovery

The bounded [training plan](data/graph_training_plan.json) and [result](data/graph_training_summary.json) pass all fourteen gates. Seventeen acknowledged attempts produce sixteen accepted training states, one QA rejection and no errors. The retained examples include three original states and thirteen connection variants; all 64 slots and 32 source names survive in every consumed connection variant. All three singleton curve families are actively supervised: six LFOs, four mappings and fourteen custom curves. Every retained example participates in each of three updates.

Recovery from update one reproduces model, optimizer, all saved RNGs, replay and sampler cursor exactly, without rendering new audio. Source/applied hashes and import ledgers match, the frontend remains frozen, and exported weights reload with the same codec. Inputs remain audio-derived features only. Training preserves original requested phases; exact recovery uses retained features rather than regenerating waveforms.

The CPU training segment takes 67.74 seconds, excluding initial pool verification; peak training/recovery memory is 1.12 GiB. The existing 5,725,689-parameter architecture and 7,187 continuous/898 categorical outputs are unchanged. Three updates establish pipeline integration, not useful reconstruction or strict tiny-overfit. No model is promoted. All 93 distinct selected local checks pass, including eight graph tests; the final eight-test graph run takes 127.05 seconds. Ruff and whitespace checks also pass.

Raw presets, asset payloads, features, audio and weights remain local. The final-test audio reserve remains untouched and GitHub Actions remains disabled. Broader corpus/layout admission, decoder refit policy, source robustness, measured data scaling, successful combined learning, Linux/Colab qualification and release quality remain open.
