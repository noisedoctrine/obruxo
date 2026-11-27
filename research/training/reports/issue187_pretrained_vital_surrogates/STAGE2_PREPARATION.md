# Issue 187 — Stage 2 preparation

**No model-quality result is claimed:** this change prepares the frozen MIDI path and the two Stage 2 adapter arms; it does not train them or select an audio trunk. The CPU checks verify exact equivalence to SpanSynth-Edit's released MIDI rows and encoder output. The required XPU forward/backward qualification is still pending because the active Issue 187 Stage 2 worker is using the XPU.

## Implemented architecture

The frozen MIDI path converts OBRUXO tick-domain `Performance` notes to absolute 48 kHz samples through the repository's `TempoMap`. It retains pitch, velocity, polyphony, note onset and note-off duration; channel 10 is encoded as drums and melodic programs are an explicit input rather than an assumed label. MIDI spans are represented on the released 25 Hz grid (1,920 samples/frame), with a 128-event frame cap and a 20.48 second maximum. Rows carry event kind, pitch, Fine40 program category, normalized velocity, sub-frame onset/offset, and log-scaled remaining note duration. Overflow and unsupported program values fail explicitly.

The extracted frozen SpanSynth-Edit MIDI encoder has **56,160 parameters**. It embeds event kind (16D) and pitch (32D), projects the 55D event row through a 256D SiLU MLP to 128D, then combines one all-event sum and five learned gated sums. RMSNorm and a 5-to-768 count projection produce one 768D vector per 25 Hz frame. Cached embeddings are float32 and clipped to valid MIDI duration. Cache identity includes score content, duration, preprocessing version, and checkpoint revision; writes are atomic.

Both trainable variants receive the same 33D expanded Issue 187 patch representation (13 continuous values plus 20 categorical features), 768D frozen MIDI vectors, and explicit `[seconds, sin(2πt), cos(2πt)]` timing features. Patch, MIDI and timing projections map to width 64, are summed and LayerNorm-fused, and pass through two residual blocks. MIDI features are sampled at actual target-frame times: codec-latent times for direct prediction and native 25 Hz token times for Stable Audio 3 conditioning.

| Adapter arm | Shared output interface | Output shape |
|---|---|---|
| Direct decoder input | Linear projection from 64D adapter state to codec latent channels | `[B, latent_channels, T_codec]` |
| Pretrained generator | Linear projection from 64D adapter state to the configured conditioning width plus valid-frame mask | `[B, T_25Hz, cond_dim]` and `[B, T_25Hz]` |

The direct connector's output width is set by the selected codec. The generator connector takes its width from the selected generator's conditioning config. The XPU smoke harness uses the already downloaded SAME-S/Stable Audio 3 Small Music pair (256 latent channels, 768 conditioning dimensions); this is a compatibility fixture, **not a Stage 1 selection**. It calls PR #188's existing Stable Audio velocity connector instead of copying that integration.

## Adapter parameter counts

Counts use the actual expanded patch width (33), direct latent width 256, and available Stable Audio 3 Small Music conditioning width 768. Each backbone has exactly **81,920 matrix projection weights** across its two blocks, excluding bias and normalization parameters.

| Arm | Residual MLP | Transformer | Difference |
|---|---:|---:|---:|
| Direct decoder input | **151,488** | **151,936** | 448 (0.30%) |
| Generator conditioning | **184,768** | **185,216** | 448 (0.24%) |

The MLP uses two pre-LayerNorm `64 → 320 → 64` GELU residual blocks. The Transformer uses two bidirectional pre-RMSNorm blocks, four 16D heads, Q/K RMSNorm, RoPE, scaled dot-product attention and a `64 → 128` SwiGLU. Its additional normalization and projection biases account for the 448-parameter total-count difference; the matrix weights remain exactly matched.

## Checkpoint provenance and verification

The full public checkpoint was downloaded from [`mimbres/spansynth-edit`](https://huggingface.co/mimbres/spansynth-edit) at immutable revision [`9a3eff857e821c91f21a05bda8048b7541f4d9e4`](https://huggingface.co/mimbres/spansynth-edit/tree/9a3eff857e821c91f21a05bda8048b7541f4d9e4/v3-sq-v8-127750step), then only `midi_encoder.*` was extracted. The encoder implementation was compared with upstream SpanSynth source commit [`f593bf21e48cc4aa836002a3834cfae42e8d4254`](https://github.com/mimbres/spansynth-edit/tree/f593bf21e48cc4aa836002a3834cfae42e8d4254); upstream and weights are Apache-2.0. The full checkpoint is 1,923,354,344 bytes; only the 56,160-parameter encoder is needed at runtime. SHA-256 values and exact extraction counts are in [STAGE2_CHECKPOINT_PROVENANCE.json](STAGE2_CHECKPOINT_PROVENANCE.json).

The CPU equivalence check used a three-note overlapping score and reported zero maximum absolute difference for all MIDI row tensors and the `[1, 512, 768]` encoder output. See [STAGE2_CPU_PARITY.json](STAGE2_CPU_PARITY.json). The focused regression tests cover tempo changes, overlapping notes, timing/velocity/release rows, valid-frame masks, cache reuse and keys, temporal alignment, both connector shapes, nonzero patch/adapter gradients, exact parameter counts, and sub-1% total-count matching.

| XPU check | Status |
|---|---|
| Extracted MIDI encoder load, frozen placement, upstream-vs-XPU and CPU-vs-XPU numerical comparison | **Not run** |
| Both backbones through the existing SAME-S decoder with finite nonzero adapter and patch gradients | **Not run** |
| Both backbones through the existing Stable Audio 3 generator objective with frozen weights | **Not run** |
| XPU forward/backward runtime and peak allocated memory | **Not measured** |

Auto-review rejected the XPU qualification command because the active Issue 187 worker was advancing through Stage 2 latent caching on the same device (latest status observed: 1,524/2,174 at `2026-11-27T20:29:19Z`). It explicitly required waiting for that worker to become idle or complete and prohibited an indirect retry. No XPU command was launched. The reproducible qualification command is `qualify_issue187_stage2_xpu.py`; run it after the active experiment releases the device. Do not treat the CPU parity result as XPU qualification.

## Remaining integration

- Run the XPU qualification above when the current worker no longer uses the device; record actual timings, peak allocated memory, and numerical differences.
- Once Stage 1 selects a trunk, do only the final codec/generator interface and latent-grid integration smoke check for that selection. The connectors are dimension-configurable, so the trunk choice does not change the adapter backbones.
- Formal training, training curves, and architecture selection remain out of scope for this preparation.
