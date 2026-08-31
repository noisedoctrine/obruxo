# Minimal audio-to-Vital training

The question is small: **can we generate a sound, predict a few synth controls
from it, and turn those predictions into a playable Vital preset?** This checks
the complete path before building the full model architecture.

We start with one oscillator and vary only its octave and level. Vital generates
fresh example audio during training. Two copies of BasicPitch hear that audio:
one stays frozen to provide performance information, while the other can learn
features useful for timbre. A tiny transformer combines their time-varying
features, then a linear output layer produces four numbers. Ordinary deterministic
code turns those numbers into two control
values and inserts them into a fixed, valid preset. It never asks the model to
write JSON, choose effects, or invent a wavetable.

```text
                         frozen BasicPitch ----\
Fresh Vital audio ------>                        tiny transformer -> dense head -> .vital
                         trainable BasicPitch -/                                    |
                                                     compare sound <--- render in Vital
```

The training objective compares the sounds' frequency content on a scale that
gives more detail to perceptually relevant frequency differences. Matching the
original control values is not required: the point is to recover a similar sound.
The networks and loss run in PyTorch on XPU by default. Vital itself renders on
CPU. Requesting XPU fails clearly if it is unavailable; there is no silent CPU fallback.

Export now enforces a strict [preset contract](PRESET_CONTRACT.md): a fixed
1.5.5-labelled subset, verified in Vital 1.6.4, with only octave and level variable.
Every output file is checked before publication. The linked report explains the
version choice, corpus evidence, loader defaults and rejection rules.

## Why training needs one extra component

Vital can make sound, but it cannot tell the network how to change its predictions
to reduce the error. We first train a second, small network to approximate the
frequency representation Vital would produce from the two controls. This is the
**spectral surrogate**. Once trained, it stays fixed and provides the feedback
needed to train the audio-to-controls model.

That approximation can be wrong. We therefore report two separate results:
the error used during training, and the error measured by actually playing the
decoded presets in Vital. Only the second measures the exported preset's sound.
The surrogate is a temporary training aid; inference only needs the first network
and the deterministic decoder.

## What the check establishes

The automated checks cover valid preset export, repeatable decoding, spectral
feedback reaching both kinds of prediction, and loading a checkpoint to decode
audio again. Training also requires nonzero gradients in the trainable BasicPitch
branch, transformer and output layer, and confirms that the frozen branch's weights **and
running statistics** are unchanged. A native smoke run exercises the installed
Vital plugin and its preset validator.

A completed smoke run proves that the components connect and execute. It does
not establish useful reconstruction quality, generalization, or support for
arbitrary recordings. The short run intentionally trains each network for only
two steps. The held-out check uses six sounds, two per octave category, and keeps
them out of the training batches. The two-branch structure establishes separate
paths for performance and timbre information; it does not prove that the learned
features have disentangled those properties.

### Tiny-transformer smoke result — 31 August 2026

The transformer ran on **Intel Arc 140T XPU**, with both BasicPitch branches
retained, using two surrogate steps and two model steps at batch size two.
All **54 unit tests** passed, including attention gradient flow, sensitivity to
frame order, independent conditioning output, and checkpoint export/reload.
The unchanged renderer also passed six native integration tests in the prior run.

| Check | Observed result |
| --- | --- |
| Transformer | One encoder layer, width 32, two heads, feedforward width 64 |
| Conditioning | 53 tokens, 32 values each; no pooling before attention |
| Audio-to-preset parameters (excludes surrogate) | 72,192 total; 55,410 trainable; 38,496 in transformer fusion |
| Transformer-fusion gradient norm | 0.00318 and 0.00348 |
| Trainable BasicPitch gradient norm | 0.000251 and 0.000930 |
| Frozen BasicPitch | Weights and running statistics unchanged |
| PyTorch peak XPU allocation | 48.0 MiB during model training |
| Model optimization steps | 1.36 seconds first step; 0.046 seconds second step |
| Real Vital renders | 31; octave and level response checks passed |
| Saved-checkpoint inference on XPU | Exported octave and level reproduced within float32 tolerance |
| Expected surrogate loss, before → after | 0.51335 → 0.51290 |
| Real Vital spectral loss, before → after | 0.28683 → 0.28272 |

This establishes that the small transformer is practical for the present input
size and that spectral feedback reaches it through the current prediction head.
Two steps are not a throughput benchmark or evidence of useful reconstruction
quality. Timings exclude Vital rendering and audio transfer, include the loss,
backward pass, optimizer and gradient checks, and synchronize XPU at both ends.
Memory is PyTorch tensor allocation, not total device/driver memory. The earlier
MLP run below is historical context, not a controlled architecture comparison.
Local artifacts are in
[`simple-training-transformer-xpu`](../../data_generation/outputs/simple-training-transformer-xpu/).

### Building toward diffusion or flow matching

The reusable part is now explicit: `model.encode_audio(audio)` returns
`[batch, 53, 32]` conditioning tokens without calling the dense prediction head.
Changing that head does not change the encoder's output; a test checks this.
The current head reads the summary token and directly predicts the preset.

A next experiment could replace that head with a small conditional network that
receives a noisy preset vector, a timestep and these audio tokens, then learns
the denoising or flow target. Generated training controls, audio conditioning,
Vital rendering and spectral evaluation are already available for that test.
Sampling and the representation sent to the final preset decoder would need an
explicit contract: today's prediction is **logits**, while sampled training
controls are **one-hot categories plus a bounded level**. They are not interchangeable.

This run does **not** implement a noise schedule, denoising/flow objective,
timestep conditioning, categorical generative model or iterative sampler. Those
remain separate tests; passing the transformer check does not validate them.

### Earlier MLP smoke result — 31 August 2026

The corrected run completed on **Intel Arc 140T XPU**, PyTorch `2.12.1+xpu`,
using two surrogate steps and two model steps, each with a batch of two.
The regression suite passed **52 unit tests and 6 native integration tests**.

| Check | Observed result |
| --- | --- |
| Real Vital renders | 31, including control-response checks |
| Three octave choices | Peaks at 130.0, 261.7 and 523.3 Hz |
| Level response | Loud/quiet RMS ratio 7.19 |
| Trainable BasicPitch gradient norm | 0.00402 and 0.00544 across the two steps |
| Dense-head gradient norm | 0.00634 and 0.00815 |
| Frozen BasicPitch | Weights and running statistics unchanged |
| Checkpoint reload on XPU | Same exported octave and level |
| Held-out surrogate approximation error | 0.50796 |
| Expected surrogate loss, before → after | 0.50967 → 0.50885 |
| Real Vital spectral loss, before → after | **0.23281 → 0.23339; no improvement** |

This is a successful wiring/gradient check and an unsuccessful reconstruction
improvement check. Two surrogate steps leave a large approximation error, so this
run is not a quality baseline. This earlier run used the pooled-feature MLP,
before the transformer was added. Its verified local artifacts are in
[`simple-training-xpu`](../../data_generation/outputs/simple-training-xpu/), including
[`report.json`](../../data_generation/outputs/simple-training-xpu/report.json).
They are generated local files and are intentionally not committed.

## Run

Use the [data-generation environment](../../data_generation/README.md), including
the pinned Vita validator and DawDreamer renderer, plus PyTorch with XPU support.
The local environment uses PyTorch `2.12.1+xpu`. The committed native
[BasicPitch checkpoint](../basic_pitch/artifacts/basic_pitch_icassp_2022.pt) initializes
both branches; no model download is needed. Use the
already-reviewed local Vital VST3; plugin fingerprint checks remain enabled.
The native verification used the released Windows wheels for Vita `0.1.0` and
DawDreamer `0.8.3`; compiling Vita from the source pin requires native build tools.

From the repository root, with that environment active:

```powershell
# Short wiring check; deliberately too short to claim reconstruction quality.
python research/modelling/simple_training/pipeline.py train --proxy-steps 2 --steps 2 --batch-size 2 --output research/data_generation/outputs/simple-training-smoke

# Longer experiment (100 surrogate + 100 encoder steps, batch size 4).
python research/modelling/simple_training/pipeline.py train --output research/data_generation/outputs/simple-training

python research/modelling/simple_training/pipeline.py predict --checkpoint research/data_generation/outputs/simple-training/checkpoint.pt --audio research/data_generation/outputs/simple-training/target.wav --output research/data_generation/outputs/decoded.vital

python -m pytest research/modelling/simple_training
```

The commands now train the transformer model with the pinned 1.5.5 output contract.
Its version-4 checkpoint contract deliberately rejects earlier MLP/transformer
checkpoints with different output-template contracts instead of relabelling them silently.
Pass `--device cpu` explicitly for a CPU run, or optionally set `--plugin-path`.
Output directories/files must be new. Generated
artifacts stay in the existing ignored data-generation outputs directory.

For this local verification only, the two native wheels were installed under
`research/data_generation/outputs/smoke-deps`, leaving the user's Python environment
unchanged. To reuse that installation in a new PowerShell session, set
`$env:PYTHONPATH = (Resolve-Path research/data_generation/outputs/smoke-deps).Path`
before running the commands. A normally configured data-generation environment
does not need that extra path.

Each run saves `checkpoint.pt`, `report.json`, `target.vital`, `predicted.vital`,
`target.wav`, and `predicted.wav`. The report separates expected surrogate loss,
held-out surrogate approximation error, and **real Vital spectral loss** before
and after training. It records render count, renderer identity, seed, dense
predictions, and finite nonzero training gradient norms. A completed smoke run
means the pipeline executes, exports valid presets, and evaluates real audio;
inspect `real_vital_loss_improved` separately. Unit tests use an explicitly labelled
test renderer and do not replace the native smoke run.
The report also records the actual device, branch and transformer gradient norms, branch-state
checks, pretrained checkpoint fingerprint, and measured oscillator pitch/level
responses before training. The short native run makes 31 renders in total.

## Scope and limitations

Inference needs no plugin and accepts mono/stereo PCM or float WAV, resampling
to 22,050 Hz. It expects exactly 0.6 seconds and the training performance: one C4
starting at time zero, 0.5-second gate, same velocity. It does not transcribe notes,
trim silence, infer arbitrary performances, or reconstruct general presets. This
fixed-performance assumption is essential: octave transposition and played MIDI
pitch would otherwise be ambiguous. Random seeds reproduce parameter sampling,
not bit-identical Vital audio. This isolated experiment does not replace or decide
the full architecture in `MODEL_ARCHITECTURE.md`.

## Implementation details

### Dense prediction and preset mapping

The preset family starts with the committed init template: oscillator 1 enabled,
one unison voice, direct output, no random phase, filters, effects, sampler or
modulation routes. All controls except the following two remain fixed.

| Control | Prediction | Mapping into Vital |
| --- | --- | --- |
| `osc_1_transpose` | Three logits (unnormalized scores) | Highest score selects `-12`, `0`, or `12` semitones |
| `osc_1_level` | One logit | `0.2 + 0.6 * sigmoid(logit)`, in raw Vital units |

The four dense values appear in that order. `PresetCodec.decode()` preserves the
template's other fields/assets, rejects nonfinite predictions, and resolves tied
category scores to the first index. Saving uses the existing schema validator.
The checkpoint includes the exact template and output contract; inference refuses
mismatches. Identical dense inputs produce identical JSON. Batched and single-item
network evaluation can differ slightly through floating-point arithmetic.

The exported preset and wavetable version headers are now `1.5.5`, chosen from
corpus frequency and tested in the reviewed 1.6.4 plugin. The source init's
`99999.9.9` development header is never exported. This replaces the earlier
1.6.4-labelled experiment template without changing its two variable controls.
See [the output contract](PRESET_CONTRACT.md) for the exact subset and measured
backward-compatibility behavior. Historical smoke reports above retain their
original version/contract context and are not evidence for every preset version.

### BasicPitch branches and XPU

Both branches load the existing pretrained `BasicPitchICASSP2022` state dict.
The performance copy uses `requires_grad_(False)`, evaluation mode and `no_grad()`;
the timbre copy is trainable, including its neural frontend normalization. The
fixed CQT kernels stay buffers. We use the complete existing network instead of
adding a new trunk API for this experiment.

BasicPitch expects 43,844 samples, so the 13,230-sample clip is zero-padded for
the branches. We retain the first 52 frames, excluding the padding-only part,
and concatenate note, onset and contour values into 440 features per frame per
branch. Aligned branch features are concatenated and projected `880 -> 32`.
A separate `1 -> 32` projection turns log-energy into a leading summary token,
retaining level information removed by BasicPitch's normalization.

Learned positional embeddings distinguish the 53 tokens. One
`TransformerEncoderLayer` uses two attention heads, width 32, feedforward width
64, and zero dropout. It returns all conditioning tokens; a `32 -> 4` linear head
reads the leading token for the current direct-prediction objective. There is no
transcription decoder, MIDI input to the network, or claim of performance-invariant
timbre inference. The performance **features** are detached, but attention and
projection weights that consume them remain trainable.

PyTorch moves both branches, the surrogate, spectral loss buffers and batches to
the requested device. The frozen branch remains in evaluation mode even when the
outer model trains. Its state is checked before/after training; the trainable
branch must change. Checkpoints are saved on CPU for portability and contain
both branches, so inference does not need the original BasicPitch checkpoint.

### Dynamic data and the two training stages

Every step samples new controls and renders a C4 at velocity 100, with a
0.5-second gate and 0.1-second tail. There is no downloaded preset bank or saved
training corpus. The first stage fits the surrogate to those rendered spectra;
the second freezes it and trains the audio-to-controls network on fresh renders.
The six validation sounds use a separate random generator and are rendered once,
so the before/after comparison uses the same references.

### Spectral objective and gradient path

The loss is mean absolute distance between log-mel magnitudes at FFT sizes 512
and 2048, with 32 mel bands and 16 pooled time bins per resolution. Magnitude is
normalized by FFT size and compressed with `log1p(1000 * magnitude)`. Both
resolutions have equal weight. Audio is not loudness-normalized, so oscillator
level remains observable. This is a small perceptual spectral objective, not a
calibrated listening-quality metric or the A-weighted auraloss implementation.
See the [auraloss reference](https://github.com/csteinmetz1/auraloss) for broader
loss designs.

The surrogate predicts these **spectral features, not a waveform**. During model
training, it evaluates each of the three legal octave categories at the predicted
level. Softmax converts the category scores to probabilities; the objective is
the probability-weighted sum of the three spectral errors. Feedback can therefore
change both the octave scores and the continuous level. The surrogate never
receives an invalid mixture of Vital categories. No control-label loss, finite
differences, or invented backward pass through Vital is used.

At export, the decoder selects one category. This differs from the weighted
training objective, which is another reason to measure decoded audio separately.
The tiny surrogate may be inaccurate or poorly trained; even a falling training
loss can leave real Vital reconstruction unchanged or worse.
