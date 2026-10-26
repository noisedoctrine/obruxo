# Proposed component check: fit three known sounds by adjusting controls directly

Before another inverse-model change, test whether the existing sound objective and a bounded native optimizer can reconstruct three individual training patches. Adjust the same nine continuous controls and two legal category choices directly. Do not train a network, add prediction heads or supervise source parameter values. This separates a component-fitting question from the shared audio-to-patch mapping; neither outcome proves a unique cause of the network failure.

**Preparation only until implemented and qualified.** Complete local software tests, exact interruption/replay checks, source/runtime pins and a concrete immutable plan before launch. Incorporate the listener's actual diagnostic feedback when deciding to execute. This card does not start native work or admit a model.

Select the existing held-note inputs nearest the low, middle and high registers: training indices **2, 8 and 11**, at MIDI **36, 61 and 83**. They are already used training/development sounds, with source categories (one voice/filter on), (four voices/filter off) and (four voices/filter on). For each, optimize one patch against both its held-note and original performances with equal weight. Known MIDI drives only the diagnostic renderer. Start from the qualified, untrained trainable-input initializer's prediction, not the rejected fitted model or source controls.

Use at most 32 direct continuous updates per patch. Measure fresh bounded central differences at the current controls with the qualified 0.025 normalized probe size. Skip inactive controls. Use Adam at 0.03 with its default beta/epsilon settings, and cap every proposed continuous move at 0.025. Measure up to three trial scales (1, 1/2, 1/4); commit only an actual joint sound-loss decrease greater than 1e-6. Preserve Adam advance and exact control rollback for refused moves. Keep inactive stored values stable. These are direct-control optimizer applications, not inverse-model training updates.

At iterations 0, 8, 16 and 24, measure all six legal unison/filter combinations at the current continuous values. Prefer the current category on ties; change only on a measured joint improvement. This diagnostic avoids assuming that a category derivative or exact source category is required to find a useful patch. Keep the inverse model's production interface unchanged at 14 dense values / 11 codec selections.

Use one isolated native worker, CPU float32 and two Torch threads. Preserve the qualified Vital engine/profile, all three oscillator and sampler random phases fixed, 32,768-frame preroll, source bytes, performances and objective. Keep sources on demand or reuse only the exact verified retained source recordings. Never change a patch to hide a rejected render. A bad source/gradient render fails the check; an invalid trial remains charged and recorded under a predeclared rejection rule.

| Maximum native allocation | Requests |
|---|---:|
| 3 patches × 32 updates × 18 continuous probes × 2 performances | 3,456 |
| 3 patches × 32 updates × 3 trial scales × 2 performances | 576 |
| 3 patches × 4 category rounds × 6 choices × 2 performances | 144 |
| Initial/final patches, both performances, exact repeats for each of 3 patches | 24 |
| Planned maximum | **4,200** |
| Bounded recovery reserve | 408 |
| Hard cap | **4,608** |

Declare **four hours / 4 GiB process-tree RAM / 2 GiB evidence / 512 MiB resident data**, at most two native executions and 33 charged direct-update attempts per patch including recovery. Keep the original start/deadline across recovery. Checkpoint direct controls, categories, Adam, measurement/acceptance cursors and all receipts after every completed update; preserve incomplete requests without inventing outcomes. Retain initial/final comparison audio and exact-repeat evidence. No automatic budget extension or additional epochs.

Freeze these diagnostic success rules before execution: every patch is valid, its joint combined sound loss is at least 25% below initialization and no worse than its matched parameter-control prediction, each performance's combined loss improves at least 10% from initialization, and MRSTFT improves on both performances. Independently rescore retained initial/final audio, reproduce the direct update trajectory and verify final fixed-phase repeats. Record all misses. If a patch reaches eight consecutive refused continuous updates, stop that patch and report its actual status; any accepted category or continuous change resets this count. Do not silently add steps.

Passing shows only that this direct solver fits these three known cases. It does not train the inverse model, pass A2/A3, validate fresh families or qualify a release. A failure does not establish that the synthesis vocabulary or perceptual training is impossible. Use the actual result to choose the next mechanism; do not sweep optimizer settings automatically. The 192 fresh validation roots and all final-test reservations stay untouched.
