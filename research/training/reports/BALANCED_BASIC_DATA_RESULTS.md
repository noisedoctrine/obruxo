# Identical note coverage is ready for the three-sound fit

The [source-data protocol](BALANCED_BASIC_DATA_CARD.md) passes. Every one of the same three existing training patches now has raw audio and frozen-frontend features at **MIDI 36, 60 and 84**, with **0.4- and 1.2-second** notes: eighteen sources in total. Source controls and assets remain unchanged within each patch. No inverse-model training or reconstruction-quality admission occurs.

All **24 native requests** succeed, including **six exact fixed-phase repeat pairs**. The independent offline check reconstructs all eighteen feature records exactly, verifies the MIDI/request identities and retained samples, and confirms all sources belong to the training split. Neither the original reserved bank nor the 24 confirmation-performance identities overlaps this grid. No confirmation audio or test state is scored.

The [frozen plan](data/balanced_basic_data_plan.json) has ID **ceb2cfb40ac58cb823685645ad5ddf06ece1e0ca5f1533800f4063d2d10708cb**. Collection takes **38.30 seconds / 0.86 GiB peak RAM**, retaining **50.09 MiB before summary**. The [offline audit](data/balanced_basic_data_verification.json) takes **2.21 seconds / 0.57 GiB**, adds zero renders, and preserves the initializer. The [summary](data/balanced_basic_data_summary.json) records zero experimental updates and no recovery. All resource limits pass.

Six focused tests cover identical cross-patch coverage, exact note durations/frame counts, reserved-identity rejection and incomplete/corrupted labels. Ruff 0.16.2 and diff checks pass. The original 14-output codec and fixed oscillator/sampler phases remain in force; Actions stays disabled and Colab stays deferred.

The [next-fit protocol](BALANCED_BASIC_FIT_CARD.md) fixes a 48-update candidate against the audited sound-only control, with the same optimizer, sound objective and rehearsal. Its per-patch sampling counts are explicit, and the old reconstruction/preservation gates remain unchanged. Implement and freeze that fit's complete machine-readable plan before execution. Source qualification alone does not establish that balanced data improves reconstruction.

The recipe remains usable for source rendering inside the training loop. This small retained bank is qualification evidence and an optional exact cache; it does not require pre-rendering the eventual corpus. A2/A3, fresh confirmation, second seed, clean listening and release remain incomplete.
