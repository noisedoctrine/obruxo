# Varied recordings for the basic model

The [input audit](reports/INPUT_NOTE_DIAGNOSIS_RESULTS.md) found two problems: initialization covered only four nearby notes, and the later sound-training panel linked note pattern perfectly to filter enable. This change prepares more useful training observations. It does not establish reconstruction quality or qualify a new model.

`train --prediction-scope basic --sampling-policy paired_views --input-performances varied` gives each training patch two observations:

- A held note lasting two seconds, so slow envelope settings can be heard.
- A staccato passage, melody or chord. These alternatives are shuffled and balanced separately within each source control category, with counts differing by at most one. Subsequent sampler epochs rotate their assignment.

Generated notes span MIDI 36–84, with varied tempo and velocity. A patch's state ID, seed, epoch and view determine its performance reproducibly. The observations have their own contract linked to the unchanged patch codec. They cannot masquerade as recordings from the old frozen performance bank. Training accepts only training states and reference observations; existing held-out performances stay reserved.

Rendering keeps the qualified 32,768-frame lead and fixed phases for all three oscillators and the sampler. Audio is rendered on demand, features are cached within the declared limit, and checkpoint recovery validates the actual performance and request identities. For 404 states, both views require 808 feature records, about 29.8 MiB. The explicit basic scope permits up to 2,048 records while retaining the existing byte limit. Broader codecs retain their 512-record limit.

`--initial-model path/to/model.pt` can start from compatible inference weights. The loader checks the pool, codec, model shape, frozen frontend and render profile before creating the output directory. It records the file and tensor hashes, starts a fresh optimizer and preserves the requested learning seed. A resumed checkpoint carries that identity; do not supply the initial file again. Omitting both new options preserves historical training contracts.

The model still predicts **14 dense outputs**: nine continuous values, three unison scores and two filter-enable scores. No additional prediction head or training loss is introduced. The new input policy uses the existing paired batch sampler; the identity-loss weight remains zero unless explicitly requested.

Local checks cover deterministic generation, complete pairs, category balance, pitch/duration bounds, split rejection, CLI wiring, compatible initialization and exact model/optimizer/random-state recovery. The [native-render check](reports/VARIED_INPUT_NATIVE_RESULTS.md) passes all 76 requests, 36 exact repeat pairs and four historical hash comparisons. The [completed fit](reports/VARIED_BOOTSTRAP_RESULTS.md) improves wider-note error by 45–47% but worsens original-input error about threefold, failing admission. Next reuse both saved input distributions together before qualifying new weights. Sound training must then use both input views and retain the earlier development panel alongside counterbalanced performances. Parameter fitting alone cannot complete A2.
