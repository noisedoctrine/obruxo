# Full-preset native rendering

The [expression profile](EXPRESSIVE_MIDI.md) extends the seed-bit profile with an all-event MIDI transport and explicit controller capabilities. It has a separate identity and native timing/isolation qualification. Earlier profiles retain their note-only transport and reject unsupported expression.

The [custom-curve and seed model](CUSTOM_CURVES_AND_SEEDS.md) adds a separate `full-1.5.5-state-verified-seed-bits-v2` profile. It recognizes Vital's exact signed 32-bit seed readback as the same unsigned bit pattern, with a distinct renderer/contract identity and dedicated native checks. The v1 profile documented below keeps its original strict nested comparison and artifact identities.

`full-check` qualifies the rendering boundary needed before expanding the model beyond the authored alpha vocabulary. The separate profile accepts the complete **1.5.5 inventory: 775 scalars and all six nested structures**, rendered by the reviewed Vital 1.6.4 binary. It uses the existing strict [full preset contract](../vital/FULL_CONTRACT_1_5_5.md). It does not change the pilot/alpha codec, pools, checkpoints or default renderer identity.

The [completed local qualification](reports/FULL_NATIVE_PROFILE_RESULTS.md) passed 36/36 native attempts and exact fixed-phase comparisons for all twelve cases, in 97.56 seconds at 392.90 MiB peak process memory.

```powershell
python research/training/run.py full-check --output research/training/outputs/full-native-profile
```

Use the pinned data-generation Python environment and the reviewed local Vital plugin. `--plugin-path` selects its location; its binary hash must still match. `--timeout` defaults to 30 seconds per attempt. The output directory must be new. There is no retry or recovery allowance that can hide a failed attempt.

The fixed run makes 36 renders: twelve independent authored cases once in fresh processes, then twice in bounded reused processes, with the second pass reversed. Each render still constructs a fresh plugin instance. Each process handles at most four jobs. Cases cover all six nested structures, the third oscillator, expanded ordinals, LFO and custom curves, route remapping, a connection in slot 64 to slot 1's amount, mono/stereo sampler playback, voice/portamento settings, two effect orders, seed edits and a Shepard component. These are correctness fixtures, not a training population or held-out quality set.

All three oscillator phase-randomization controls and sampler random phase are zero for these diagnostics. Qualification requires identical float32 audio, verified native state and provenance across fresh/reused renders. Actual training requests retain their own requested phase settings. No final-test reserve is read and no audio archive is written.

Every accepted full-profile render checks all 775 requested scalars, all 64 ordered modulation slots, LFOs, custom warps, seeds, wavetables and sample metadata/payloads. Native-only ramp fields must all retain their reviewed default. Wavetable header migration to `1.6.4` is explicit. Sample readback must have exactly the reviewed four-zero prefix, unchanged length, and at most one PCM16 unit of rounding after alignment; this is not a byte-exact sample round trip. Other asset changes fail closed.

The output pins source hashes, contract identity, exact requests, attempt limits and execution order. Qualification also requires measured application/process memory below 8 GiB and verified worker recycling. Timing is recorded as resource evidence; this run is not a renderer speed comparison.

For an explicitly authored full-profile request:

```python
from obruxo_data.hashing import canonical_json
from obruxo_data.render.base import FULL_PRESET_RENDERER_ID, RenderRequest
from obruxo_training.stream import NativeRendererFactory, render_isolated

request = RenderRequest(preset_json=canonical_json(full_document), performance=performance,
                        renderer_id=FULL_PRESET_RENDERER_ID)
audio, metadata = render_isolated(request, renderer_factory=NativeRendererFactory(renderer_id=FULL_PRESET_RENDERER_ID))
```

The caller must use the same explicit identity for request and factory. Raw full-profile JSON is validated before request canonicalization, so duplicate keys, unknown controls and version changes are rejected. The legacy factory default remains unchanged.

This repository checkout supplies the full contract resources under `research/vital/contract_1_5_5`; a standalone installation must include those resources before claiming portability. Linux/Colab native execution remains unqualified. The factory template's pinned bytes now have an explicit LF checkout rule, fixing Windows checkout failures without changing the approved content or hash.

Remaining coverage work includes the broad learned codec and heads, a declared canonical factory asset bank, train-only curve assets, broad training recipes, performance variation and measured learning. Passing these authored rendering checks does not establish that a model predicts those controls or that every factory asset has been qualified.
