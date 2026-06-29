# NERETRAK: Gemini Framework and Literature Review

This document outlines the independent framework design and literature review for **NERETRAK (NerellaTrak)**. It provides a formal blueprint for a fully local, open-source machine learning pipeline that recovers Vital synthesizer presets from raw audio.

---

## 1. System Architecture Overview

NERETRAK uses a factorized architecture that separates **musical content** (pitch, timing, dynamics, polyphony) from **synthesis identity** (the underlying preset parameters, modulations, and routing). 

```mermaid
flowchart TD
    subgraph Data Generation (Offline)
        A[Random Parameter Sampler] -->|Preset Vector| B[Vita Headless Synth]
        C[MIDI Clip Generator] -->|MIDI Sequence| B
        B -->|Audio Waveform| D[Log-Mel & STFT Features]
    end

    subgraph Representation Learning
        D -->|Audio Stream| E[Audio Encoder]
        C -->|MIDI Stream| F[MIDI Encoder]
        E & F -->|Disentangled Latents| G[Timbre Space z_timbre]
        E & F -->|Disentangled Latents| H[Content Space z_music]
    end

    subgraph Preset Synthesis
        G -->|z_timbre| I[Hierarchical Preset Decoder]
        I -->|Parameter Vectors| J[Vita Export Pipeline]
        J -->|JSON Serialization| K[.vital Preset File]
    end
```

The system operates in two core modes:
1. **Synthetic Training Loop**: Paired data is generated locally using `bindings.cpp` from the headless Vita wrapper. The encoder learns to map the audio to a latent space where notes do not affect preset identity.
2. **Inference Loop**: Raw target audio is recorded or provided by the user. An auxiliary transcription model extracts the notes, and the core network projects the audio into the timbre latent space, decodes the Vital parameters, and outputs a valid `.vital` file.

---

## 2. Topic 1: `.vital` File Structure and ML Representation

### 2.1 File Format Analysis
Vital presets are saved in a plain-text JSON structure. The main container is:

```json
{
  "synth_version": "1.5.5",
  "preset_name": "Init",
  "author": "Anonymous",
  "comments": "",
  "settings": {
    "osc_1_on": 1.0,
    "osc_1_level": 0.8,
    "filter_1_cutoff": 60.0,
    "modulations": [
      {
        "source": "lfo_1",
        "destination": "filter_1_cutoff",
        "amount": 0.4,
        "bipolar": false
      }
    ],
    "lfos": [
      {
        "name": "lfo_1",
        "num_points": 4,
        "points": [0.0, 0.0, 0.5, 1.0, 1.0, 0.0]
      }
    ],
    "wavetables": []
  }
}
```

The hierarchical components within the `"settings"` object must be mapped to ML-friendly representations:

1. **Scalar Parameters (~775 controls)**: Flat key-value pairs representing continuous knobs (e.g., levels, envelope ADSR, filter cutoffs) and discrete options (e.g., oscillator waveforms, routing switches, filter types).
2. **Modulation Matrix**: A list of objects connecting a **source** (e.g., LFO, Envelope, Mod Wheel) to a **destination** (e.g., Oscillator Pitch, Reverb Size).
3. **LFO Shapes**: Line generator coordinates mapping time to output values.
4. **Wavetables and Samples**: Binary waveform frames and audio samples, which are encoded in Base64 strings.

### 2.2 Machine Learning I/O Representation
To predict these parameters without forcing a neural network to output raw JSON text, we map the preset to a structured vector-and-graph representation:

| Component | VST/JSON Data | Model Output Representation |
| :--- | :--- | :--- |
| **Continuous Knobs** | Floats in physical range (e.g., 20Hz–20kHz) | Normalized sigmoid outputs in $[0, 1]$, mapped back to physical ranges using scaling metadata. |
| **Discrete Switches / Enums** | Integers representing options (e.g., Filter Mode) | One-hot encoded probability distributions (Softmax). |
| **Modulation Routing** | List of source-destination-amount paths | Adjacency matrix representation or a sparse sequence model (e.g., Transformer sequence of routing tokens). |
| **LFO Curves** | Coordinate arrays of line segments | Fixed-length resampled vector (e.g., 32 points) or spline control points predicted via continuous regression. |
| **Wavetables** | Base64-encoded float32 frames | Lookup index into a closed vocabulary of standard wavetables, or a low-dimensional latent vector (from a pre-trained autoencoder). |

> [!NOTE]
> Scaling parameters are defined in Vital's `synth_parameters.cpp` with different scale types (e.g., `kLinear`, `kExponential`, `kIndexed`). The model should output normalized values in $[0,1]$, which are denormalized prior to JSON creation based on their specific scale registry rules.

---

## 3. Topic 2: Best FOSS Audio-to-MIDI Option

To factor out the musical content from the audio signal, an auxiliary transcription model is required during inference to feed the note context to the model.

### 3.1 Candidate Comparison

| Candidate | License | Polyphony | Pitch Bends | Portability / Performance |
| :--- | :--- | :--- | :--- | :--- |
| **Spotify Basic Pitch** | Apache 2.0 | Yes | Yes (High Precision) | Lightweight, fast CPU/GPU inference, simple Python API. |
| **Magenta Onsets & Frames** | Apache 2.0 | Yes | No | Heavy, highly optimized for acoustic piano only. |
| **Google MT3** | Apache 2.0 | Yes | No | Extremely large transformer, high latency, difficult setup. |
| **Omnizart** | GPLv3 | Yes | No | Heavy dependencies, complex installation, slow. |

### 3.2 Framework Selection: Spotify Basic Pitch
**Basic Pitch** is selected as the optimal FOSS tool. Its lightweight architecture is designed for multi-pitch tracking and pitch bend detection, making it suitable for synthesizer performances (which frequently contain portamento, vibrato, and pitch modulation).

```python
from basic_pitch.inference import run_inference
from basic_pitch import ICASSP_2022_MODEL_PATH

# Extract note events and pitch bends from user audio
model_output, midi_data, note_events = run_inference(
    audio_path,
    model_path=ICASSP_2022_MODEL_PATH
)
```

### 3.3 Integration Strategy
* **Training Phase**: The transcription model is **not** used. The synthetic dataset generator creates audio by feeding a known, clean MIDI sequence into the Vita synth. This ground-truth MIDI is fed directly to the network.
* **Inference Phase**: When a user inputs an audio clip, Basic Pitch generates a note-event representation. This midi representation is converted to a piano-roll matrix (pitch $\times$ time) and fed alongside the spectrogram into the model to isolate musical structure.

---

## 4. Topic 3: Audio to Spectrogram: Prebuilt Layers

Instead of extracting spectrograms offline, the transformation should occur on-the-fly as the first layer of the PyTorch neural network. This allows GPU acceleration and end-to-end gradient updates.

### 4.1 Implementation with `nnAudio`
We propose using **nnAudio** as the spectrogram front-end. It implements 1D convolutional layers configured to perform Fourier transforms, which makes it faster than CPU-based libraries like Librosa.

```python
import torch
import torch.nn as nn
from nnAudio.features.mel import MelSpectrogram

class AudioSpectrogramFrontEnd(nn.Module):
    def __init__(self, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
        super().__init__()
        # PyTorch layer that converts raw waveform directly on GPU
        self.mel_spec = MelSpectrogram(
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            trainable_mel=False,  # Start frozen, can be ablated to True
            trainable_STFT=False
        )
        
    def forward(self, x):
        # x shape: [batch_size, num_samples]
        # output shape: [batch_size, n_mels, time_steps]
        return self.mel_spec(x)
```

### 4.2 Proposed Feature Stack
To capture the wide array of timbres that Vital can generate, the front-end will extract three complementary features:

1. **Log-Mel Spectrogram (128 bins)**: Captures global spectral envelopes and coarse timbral characteristics.
2. **Linear STFT Magnitude**: Essential for high-frequency details, such as white noise oscillators, filter resonance peaks, and bitcrushing effects, which are compressed in Mel-scale representations.
3. **Spectral Envelope & Energy Curves**: Used to regression-map envelope parameters (Attack, Decay, Sustain, Release) by monitoring signal rise and decay rates.

---

## 5. Topic 4: Negative Sampling and Metric Learning (Triplet & JEPA)

Synthesizers suffer from a **many-to-one mapping problem**: entirely different parameter combinations can produce perceptually identical sounds (e.g., setting an oscillator level to zero makes its other parameters irrelevant; phase offsets can change wavetable shapes without changing the sound). Naive Mean Squared Error (MSE) regression on parameters fails because it forces the model to average these paths, resulting in muddy presets.

To solve this, we propose a representation learning stage using a joint embedding space.

### 5.1 Triplet and InfoNCE Contrastive Learning
We define a contrastive metric learning objective where the audio encoder is forced to project the audio into a latent space ($z_{timbre}$) that represents synthesizer identity independent of note pitches.

We formulate the sampling pairs as follows:

```
Anchor (A):        audio_render(Preset_1, MIDI_sequence_X)
Positive (P):      audio_render(Preset_1, MIDI_sequence_Y)  <-- Same preset, different notes
Negative (N):      audio_render(Preset_2, MIDI_sequence_X)  <-- Different preset, same notes
Hard Negative (H): audio_render(Preset_1_perturbed, MIDI_sequence_X) <-- Param jitter
```

The contrastive loss (InfoNCE) minimizes the distance between $Anchor$ and $Positive$ while maximizing the distance to $Negative$ and $Hard\ Negative$ samples:

$$\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\text{sim}(z_A, z_P) / \tau)}{\exp(\text{sim}(z_A, z_P) / \tau) + \sum_{i} \exp(\text{sim}(z_A, z_N^{(i)}) / \tau)}$$

This forces the encoder to extract the timbre and routing structures while ignoring the musical content.

### 5.2 MIDI-Conditioned JEPA (Joint-Embedding Predictive Architecture)
We propose a custom **MIDI-conditioned Audio-JEPA** to learn invariant representations. Instead of predicting the raw audio waveform, the predictor works in the embedding space:

```
                  +-----------------+
                  |  Audio Encoder  |
                  +--------+--------+
                           |
                     Audio A (Preset 1, MIDI X)
                           |
                           v
                     z_timbre (A) ----+
                                      |
                                      v
+------------+       +----------------+
| MIDI Y     | ----> | Predictor Head | ----> Predicted z_timbre (B)
+------------+       +----------------+            |
                                                   | (Verify)
                                                   v
                                          True z_timbre (B) from
                                          Audio B (Preset 1, MIDI Y)
```

1. **Context Stream**: Audio rendered from **Preset 1** with **MIDI X** is encoded into a latent state.
2. **Target Stream**: Audio rendered from the **same Preset 1** but with **MIDI Y** is encoded using a target network (with stop-gradients or exponential moving average weights).
3. **Predictor**: Given the latent representation of the context stream and the target **MIDI Y**, the predictor must forecast the latent representation of the target stream.

This prevents collapse and encourages the model to extract the invariant synthesis properties needed to render the target notes.

---

## 6. Implementation Roadmap

```
Phase 1: Dataset Generation
├── Run Vita headless to extract the default initialization JSON schema.
├── Sample 100k random presets using musically-biased parameter distributions.
└── Render each preset using 3 distinct MIDI phrases (monophonic, chords, sweeps).

Phase 2: Representation Learning
├── Pre-train the Audio Encoder (nnAudio + AST backbone) using the Contrastive/JEPA loss.
└── Verify that representations of identical presets cluster together despite note differences.

Phase 3: Hierarchical Prediction
├── Train the continuous scalar heads (levels, decay times, filters).
├── Train the discrete categorization heads (oscillator shapes, FX switches).
└── Train the Modulation Graph head (source-to-destination paths).

Phase 4: Optimization and Validation
├── Render predicted presets back through Vita.
└── Calculate Mel-Spectral and CQT distance metrics between predicted and target audio.
```

---

## 7. Open Challenges & Mitigations

* **Wavetable Space**: Regressing arbitrary custom wavetable tables is extremely difficult.
  * *Mitigation*: Restrict Phase 1 to standard factory wavetables. For custom tables, train a separate autoencoder to map the Base64 wavetable structures to a 16-dimensional latent space, then train the model to predict those latents.
* **Modulation Destination Permutations**: In Vital, there are multiple redundant modulation slots.
  * *Mitigation*: Sort the predicted modulation assignments alphabetically by source and destination during serialization to eliminate slot-ordering permutations before calculation of loss.
* **Differentiability**: The synthesizer engine is non-differentiable, preventing end-to-end optimization of parameters directly from audio loss.
  * *Mitigation*: Train a feedforward neural network proxy (Neural Synth Proxy) that maps parameters to expected spectrogram features, and run gradients through this proxy during training.
