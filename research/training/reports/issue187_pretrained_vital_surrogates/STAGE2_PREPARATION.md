# Issue 187 — Stage 2 preparation

**No model-quality result is claimed:** this change prepares the frozen MIDI path and the two Stage 2 adapter arms; it does not train them or select an audio trunk. CPU parity against the released SpanSynth-Edit implementation is exact. Encoder and direct-decoder XPU checks passed; Stable Audio generator-conditioning backward still fails with nonfinite gradients.

## Implemented architecture

The frozen MIDI path converts OBRUXO tick-domain `Performance` notes to absolute 48 kHz samples through the repository's `TempoMap`. It retains pitch, velocity, polyphony, note onset and note-off duration; channel 10 is encoded as drums and melodic programs are an explicit input rather than an assumed label. MIDI spans are represented on the released 25 Hz grid (1,920 samples/frame), with a 128-event frame cap and a 20.48 second maximum. Rows carry event kind, pitch, Fine40 program category, normalized velocity, sub-frame onset/offset, and log-scaled remaining note duration. Overflow and unsupported program values fail explicitly.

The extracted frozen SpanSynth-Edit MIDI encoder has **56,160 parameters**. It embeds event kind (16D) and pitch (32D), projects the 55D event row through a 256D SiLU MLP to 128D, then combines one all-event sum and five learned gated sums. RMSNorm and a 5-to-768 count projection produce one 768D vector per 25 Hz frame. Cached embeddings are float32 and clipped to valid MIDI duration. Cache identity includes score content, duration, preprocessing version, and checkpoint revision; writes are atomic.

Both trainable variants receive the same 33D expanded Issue 187 patch representation (13 continuous values plus 20 categorical features), 768D frozen MIDI vectors, a 768D causal note-history summary with a two-second half-life, and explicit `[seconds, sin(2πt), cos(2πt)]` timing features. Patch, MIDI, note-history and timing projections map to width 64, are summed and LayerNorm-fused, and pass through two residual blocks. MIDI and history features are interval-averaged onto each target frame so short events are retained during 25 Hz to codec-rate downsampling; frame masks mark any destination interval overlapping valid source time. Transformer self-attention excludes padded keys.

| Adapter arm | Shared output interface | Output shape |
|---|---|---|
| Direct decoder input | Linear projection from 64D adapter state to codec latent channels | `[B, latent_channels, T_codec]` |
| Pretrained generator | Linear projection from 64D adapter state to the configured conditioning width plus valid-frame mask | `[B, T_25Hz, cond_dim]` and `[B, T_25Hz]` |

The direct connector's output width is set by the selected codec. The generator connector takes its width from the selected generator's conditioning config. The XPU smoke harness uses the already downloaded SAME-S/Stable Audio 3 Small Music pair (256 latent channels, 768 conditioning dimensions); this is a compatibility fixture, **not a Stage 1 selection**. It calls PR #188's existing Stable Audio velocity connector instead of copying that integration.

## Adapter parameter counts

Counts use the actual expanded patch width (33), direct latent width 256, and available Stable Audio 3 Small Music conditioning width 768. Each backbone has exactly **81,920 matrix projection weights** across its two blocks, excluding bias and normalization parameters.

| Arm | Residual MLP | Transformer | Difference |
|---|---:|---:|---:|
| Direct decoder input | **200,704** | **201,152** | 448 (0.22%) |
| Generator conditioning | **233,984** | **234,432** | 448 (0.19%) |

The MLP uses two pre-LayerNorm `64 → 320 → 64` GELU residual blocks. The Transformer uses two bidirectional pre-RMSNorm blocks, four 16D heads, Q/K RMSNorm, RoPE, validity-masked scaled dot-product attention and a `64 → 128` SwiGLU. Both variants share the same 768→64 history projection. Its additional normalization and projection biases account for the 448-parameter total-count difference; the matrix weights remain exactly matched.

## Checkpoint provenance and verification

The full public checkpoint was downloaded from [`mimbres/spansynth-edit`](https://huggingface.co/mimbres/spansynth-edit) at immutable revision [`9a3eff857e821c91f21a05bda8048b7541f4d9e4`](https://huggingface.co/mimbres/spansynth-edit/tree/9a3eff857e821c91f21a05bda8048b7541f4d9e4/v3-sq-v8-127750step), then only `midi_encoder.*` was extracted. The encoder implementation was compared with upstream SpanSynth source commit [`f593bf21e48cc4aa836002a3834cfae42e8d4254`](https://github.com/mimbres/spansynth-edit/tree/f593bf21e48cc4aa836002a3834cfae42e8d4254); upstream and weights are Apache-2.0. The full checkpoint is 1,923,354,344 bytes; only the 56,160-parameter encoder is needed at runtime. SHA-256 values and exact extraction counts are in [STAGE2_CHECKPOINT_PROVENANCE.json](STAGE2_CHECKPOINT_PROVENANCE.json).

The CPU equivalence check used a three-note overlapping score and reported zero maximum absolute difference for all MIDI row tensors and the `[1, 512, 768]` encoder output. See [STAGE2_CPU_PARITY.json](STAGE2_CPU_PARITY.json). The focused regression tests cover tempo changes, overlapping notes, timing/velocity/release rows, valid-frame masks, cache reuse and keys, short-event interval aggregation, shared causal release history, Transformer padding invariance, both connector shapes, nonzero patch/adapter gradients, exact parameter counts, and sub-1% total-count matching.

| XPU check | Status |
|---|---|
| Extracted MIDI encoder load, frozen placement, upstream-vs-XPU and CPU-vs-XPU numerical comparison | **Passed**; upstream XPU max difference `4.47e-8`, CPU/XPU max difference `5.96e-7` |
| Both backbones through the existing SAME-S decoder, including audio decode | **Passed**; MLP 2.00 s / 536,314,368 bytes peak, Transformer 0.28 s / 537,488,896 bytes peak |
| Both backbones through the Stable Audio 3 generator, full two-step sampler and decoder | **Blocked**; FP16 XPU backward produced nonfinite velocity/audio gradients |
| Generator-conditioning adapter and patch-control gradients | **Blocked by generator backward NaNs**; the harness checks every adapter parameter for a finite gradient |
| Generator-path runtime and peak memory | **Not measured to completion**; failure is recorded in the smoke JSON |

The first XPU attempt was deferred while the active Issue 187 worker occupied the device. After that worker exited, encoder parity passed (1.27 s, 13,623,296 bytes peak) and both direct-decoder arms completed forward/backward with finite nonzero patch gradients. Full Stable Audio sampling revealed nonfinite XPU FP16 backward gradients, so generator-path training readiness is **not verified**. Resolve this before training the generator-conditioning arm; do not silently substitute precision or another generator. The reproducible command is `qualify_issue187_stage2_xpu.py`. The harness records completed phases when a later phase fails.

## Remaining integration

- Resolve and re-run Stable Audio generator backward qualification on XPU; the current FP16 path produced nonfinite gradients.
- The Stage 1 runner still instantiates its prior adapter. Formal training was deliberately not integrated in this preparation task; before resuming it, wire both Stage 2 backbones to the reusable latent cache and verify the selected codec/generator interfaces.
- Once Stage 1 selects a trunk, do the final codec/generator interface and latent-grid integration smoke check for that selection. The connectors are dimension-configurable, so the trunk choice does not change the adapter backbones.
- Variable-length batching through Stable Audio conditioning remains untested. Single-clip conditioning is trimmed to valid MIDI extent because the upstream DiT does not use cross-attention masks.
- Formal training, training curves, and architecture selection remain out of scope for this preparation.
