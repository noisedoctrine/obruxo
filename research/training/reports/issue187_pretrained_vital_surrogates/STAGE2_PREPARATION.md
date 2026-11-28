# Issue 187 — Stage 2 preparation

The Medium integration run is now underway. The measurements below describe the earlier compatibility fixtures. The [Stage 1 rescore and decision](STAGE1_INTERIM.md) establish which checkpoint the actual runner qualifies.

## Experiment runner and scheduling

| Work | Resource | Dependency |
|---|---|---|
| Recover missing #172 audio | CPU, three original worker histories at a time | Frozen render ledger and identical native readback path |
| Rescore codec outputs | XPU codec inference, then overlapping CPU waveform metrics and locked CIR inference | Receipt-verified #185 originals and original codec revisions |
| Qualify Medium adapters | XPU, one variant at a time | Medium selected for qualification, original receipts, pinned MIDI checkpoint |
| Run 500-update pilots, then extend viable arms | XPU training and validation, independent CIR scoring | Published Stage 1 decision, passing selected-model qualification, verified training data |
| Optimize 192 patches and render in Vital | XPU optimization and native CPU rendering | Stage 2 evidence for one arm and a comparable #173 baseline |

The original-worker replay recovered all 1,020 missing training clips exactly. The runner compares the frozen input receipt manifests before reusing legacy pairs, checks the selected codec revision, and verifies the shape, dtype and finiteness of each cached latent. Missing source audio returns a blocked status. [Receipt recovery evidence](STAGE2_AUDIO_RECOVERY_PROBE.json)

The Medium path uses the generator's embedded 256-channel codec on the 4,096-sample grid, with 768D SpanSynth-derived conditioning tokens at 25 Hz. The frozen generator uses its shipped `rf_denoiser` velocity objective. Validation uses eight pingpong steps with the checkpoint's distribution shift and independently seeded initial and per-step noise keyed to each clip. Pretrained generator weights are FP16, codec weights and adapter parameters are FP32. The runner reuses the same frozen codec in both arms and qualifies both adapter variants at a disposable optimizer update before the scheduled training checkpoints.

The #173 run retains legacy per-case MR-STFT, but the listening WAVs are later, gain-processed previews. None of the ten preview reference/baseline pairs match both original native waveform receipts. The full-panel primary MEL comparison would need receipt-compatible baseline audio if a Stage 2 survivor advances. No baseline was repeated to prepare this runner.

The local handoff has separate logs in `tmp/issue187-rescore`, `tmp/issue187-selected-smoke` and `tmp/issue187-formal`. The training queue waits for publication of the full Stage 1 comparison and the selected-model qualification, runs the 500-update pilots, then resumes viable configurations to a maximum of 5,500 updates each, validating every 500 updates. Stable Audio 3 Medium runs before SAME-L; SAME-L resumes from its saved checkpoints afterward. The locked CIR scorer follows validation manifests independently. The native inversion panel requires an explicit Stage 2 evidence-based decision.

**No Stage 2 model-quality result is claimed:** this change prepares the frozen MIDI path and the two adapter arms; it does not train them or select an audio trunk. CPU parity against the released SpanSynth-Edit implementation is exact. The XPU checks below distinguish the passing native flow-matching objective from the still-failing Transformer sampler backward.

## Implemented architecture

The frozen MIDI path converts OBRUXO tick-domain `Performance` notes to absolute 48 kHz samples through the repository's `TempoMap`. It retains pitch, velocity, polyphony, note onset and note-off duration; channel 10 is encoded as drums and melodic programs are an explicit input rather than an assumed label. MIDI spans are represented on the released 25 Hz grid (1,920 samples/frame), with a 128-event frame cap and a 20.48 second maximum. Rows carry event kind, pitch, Fine40 program category, normalized velocity, sub-frame onset/offset, and log-scaled remaining note duration. Overflow and unsupported program values fail explicitly.

The extracted frozen SpanSynth-Edit MIDI encoder has **56,160 parameters**. It embeds event kind (16D) and pitch (32D), projects the 55D event row through a 256D SiLU MLP to 128D, then combines one all-event sum and five learned gated sums. RMSNorm and a 5-to-768 count projection produce one 768D vector per 25 Hz frame. Cached embeddings are float32 and clipped to valid MIDI duration. Cache identity includes score content, duration, preprocessing version, and checkpoint revision; writes are atomic.

Both trainable variants receive the same 33D expanded Issue 187 patch representation (13 continuous values plus 20 categorical features), 768D frozen MIDI vectors, a 768D causal note-history summary with a two-second half-life, and explicit `[seconds, sin(2πt), cos(2πt)]` timing features. Patch, MIDI, note-history and timing projections map to width 64, are summed and LayerNorm-fused, and pass through two residual blocks. MIDI and history features are interval-averaged onto each target frame so short events are retained during 25 Hz to codec-rate downsampling; frame masks mark any destination interval overlapping valid source time. Transformer self-attention excludes padded keys.

| Adapter arm | Shared output interface | Output shape |
|---|---|---|
| Direct decoder input | Linear projection from 64D adapter state to codec latent channels | `[B, latent_channels, T_codec]` |
| Pretrained generator | Linear projection from 64D adapter state to the configured conditioning width plus valid-frame mask | `[B, T_25Hz, cond_dim]` and `[B, T_25Hz]` |

The direct connector's output width is set by the selected codec. The generator connector takes its width from the selected generator's conditioning config. The XPU smoke harness uses the already downloaded SAME-S/Stable Audio 3 Small Music pair (256 latent channels, 768 conditioning dimensions); this is a compatibility fixture, **not a Stage 1 selection**. It calls PR #188's existing Stable Audio velocity connector instead of copying that integration.

### Stage 2 configuration names

SAME-L names the embedded autoencoder's decoder; Stable Audio 3 Medium names the frozen DiT together with that same decoder. Configuration names follow data flow from the trainable adapter to its frozen downstream model.

| Configuration | Trainable adapter | Frozen downstream model |
|---|---|---|
| Residual MLP + SAME-L | Residual MLP | SAME-L decoder |
| Transformer + SAME-L | Transformer | SAME-L decoder |
| Residual MLP + Stable Audio 3 Medium | Residual MLP | Stable Audio 3 Medium DiT and SAME-L decoder |
| Transformer + Stable Audio 3 Medium | Transformer | Stable Audio 3 Medium DiT and SAME-L decoder |

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
| Extracted MIDI encoder load, frozen placement, upstream-vs-XPU and CPU-vs-XPU numerical comparison | **Passed**; upstream XPU max difference `5.96e-8`, CPU/XPU max difference `5.96e-7` |
| Both backbones through the existing SAME-S decoder, including audio decode | **Passed**; MLP 2.00 s / 536,314,368 bytes peak, Transformer 0.28 s / 537,488,896 bytes peak |
| Both backbones through the native Stable Audio rectified-flow training objective | **Passed**; MLP 2.92 s / 1,595,971,200 bytes peak, Transformer 0.31 s / 1,600,805,376 bytes peak; both had finite nonzero adapter/patch gradients and frozen generator weights |
| Residual MLP through a two-step Stable Audio sampler and codec decoder | **Forward passed; backward failed**; 1.39 s / 1,702,774,784 bytes peak. Sampler velocities, decoded audio, and decoder latent gradient were finite; generator-conditioning gradient was NaN. Latent magnitude grew `3.54 → 86.2` and velocity reached `371.5`. |
| Transformer through the same sampler and decoder | **Forward passed; backward failed**; 0.52 s / 1,700,711,424 bytes peak. Sampler velocities, decoded audio, and decoder latent gradient were finite; generator-conditioning gradient was NaN. Latent magnitude grew `3.54 → 85.1` and velocity reached `406`. |

The direct and generator smoke checks reused the SAME-S codec and Stable Audio 3 Small Music checkpoint already present for Stage 1; this is a compatibility fixture, not a trunk selection. The rerun used upstream-default SDPA with no test-only backend override and a synthetic `[1, 512, 768]` MIDI-token fixture (the extracted encoder has its own real-checkpoint XPU parity check). Generator training uses the PR #188 velocity connector and FP32 flow latents/noise while the frozen generator remains FP16. Both one-step flow-matching objectives pass. For sampler loss, each denoising output, latent update, and decoded waveform is checked before backward; the first nonfinite values occur in gradients flowing into generator conditioning. The reproducible harness records each backbone and objective separately so one failure does not hide the other arm.

## Historical handoff from preparation

- The legacy Stage 1 runner now fails closed before decoder-latent caching or training because it still instantiates the superseded residual MLP and custom MIDI path. Wire both prepared adapters to the cache and selected codec/generator before resuming Stage 2.
- Resolve the generator-conditioning backward NaNs seen in both sampler arms before using differentiable sampler loss. The native one-step flow-matching training objective passes for both backbones; the sampler forward and codec decode are finite.
- Once Stage 1 selects a trunk, do the final codec/generator interface and latent-grid integration smoke check for that selection. The connectors are dimension-configurable, so the trunk choice does not change the adapter backbones.
- Variable-length batching through Stable Audio conditioning remains untested. Single-clip conditioning is trimmed to valid MIDI extent because the upstream DiT does not use cross-attention masks.
- Stage 1 stopped at 1,536/2,174 cached latents on a strict render-receipt mismatch. The existing cache is preserved; the original waveform needed to regenerate the missing latent is absent from available pair artifacts. Diagnosis and the no-bypass recovery constraint are recorded in [STAGE1_INTERIM.md](STAGE1_INTERIM.md).
- Formal training, training curves, and architecture selection remain out of scope for this preparation.
