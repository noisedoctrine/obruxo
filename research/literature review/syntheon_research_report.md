# Syntheon Research Report
## Repository: https://github.com/gudgud96/syntheon


## Executive Summary

Syntheon is a deep learning-based parameter inference system for music synthesizers. Given an audio sample, it infers the best parameter preset for a given synthesizer that can recreate the audio sample. The project aims to simplify the sound design process by automating preset generation.

**Status:**
- **Vital synthesizer**: Fully supported and functional
- **Model**: WTSv2 (Diff-WTS) adapted from DDSP-PyTorch


## Design Choices

### Modular Architecture: Converter/Inferencer Pattern

**Rationale:** The project separates concerns into two distinct components for each synthesizer:

1. **Converter**: Handles bidirectional conversion between plugin preset formats and Python dictionaries
   - Design choice: Abstract base class (`SynthConverter`) with synthesizer-specific implementations
   - Benefit: Enables support for diverse preset formats
   - Implementation detail: Vital uses Base64 encoding for wavetables

2. **Inferencer**: Handles the deep learning inference pipeline
   - Design choice: Abstract base class (`Inferencer`) with synthesizer-specific model architectures
   - Benefit: Allows different model architectures per synthesizer while maintaining consistent API
   - Workflow: `load_model()` → `inference()` → `convert_to_preset()`

**Why this separation:** Different synthesizers have vastly different parameter spaces and preset formats. By separating format handling (converter) from inference logic (inferencer), the codebase remains modular and extensible.

### Model Architecture: WTSv2 (Diff-WTS)

**Choice:** Adapted from DDSP-PyTorch (Differentiable Digital Signal Processing)

**Rationale:**
- DDSP provides differentiable audio synthesis components
- Enables end-to-end training with spectral loss functions
- Proven effectiveness in audio parameter inference tasks

**Key architectural decisions:**
- **Multiscale STFT loss**: Uses 6 scales [4096, 2048, 1024, 512, 256, 128] with 75% overlap
  - Why: Captures both coarse and fine spectral details
  - Design choice: Hann windowing for smooth frequency response
- **Hidden size: 256**: Balance between model capacity and computational efficiency
- **CREPE for pitch extraction**: Large variant model
  - Why: State-of-the-art pitch detection accuracy
  - Trade-off: Computationally expensive but necessary for accurate parameter inference
- **Fixed audio constraints**: 16kHz, 4 seconds duration, 160 sample block size
  - Why: Simplifies preprocessing and model architecture
  - Limitation: Inflexible for variable-length inputs

### Wavetable Synthesis Design

**Design choice: Differentiable wavetable synthesis with attention mechanism**

**Implementation details:**
- Linear interpolation for smooth waveform generation
- Phase accumulation via frequency integration
- Attention mechanism for mixing multiple wavetables
- Tanh activation for range normalization [-1, 1]

**Why linear interpolation:** Simpler than higher-order interpolation, sufficient for audio synthesis, differentiable for backpropagation

**Performance trade-off:** Batch wavetable synthesis currently uses loop (marked as TODO for einsum parallelization)

### ADSR Envelope Design

**Design choice: Differentiable ADSR with power function shaping**

**Key features:**
- Power function support for convex/concave envelope shapes (pow > 0 for convex, pow < 0 for concave)
- Custom autograd function (`DiffRoundFunc`) for differentiable rounding
- Soft minimum clamping with temperature parameter

**Why differentiable rounding:** Enables gradient flow through discrete temporal quantization, important for training

**Reference:** Influenced by DiffSynth's envelope implementation

### Wavetable Inference Algorithm

**Algorithm choice: Pitch-based wavelet extraction**

**Steps:**
1. Find continuous pitch segments (threshold: 10 steps = 0.1 sec)
2. Calculate period from pitch
3. Locate local minimum within 2-period window
4. Extract wavelet from audio
5. Upsample to 512 samples
6. Normalize magnitude to [-1, 1]

**Rationale:** Assumes stable pitch regions contain representative waveform cycles

**Known limitation:** Prone to silence detection issues, edge cases not fully handled (marked as TODO)


## Technical Decisions and Findings

### Audio Processing Pipeline

**Decision: A-weighted loudness extraction**
- Why: Perceptually relevant loudness measurement
- Implementation: STFT-based with configurable frame rate
- Trade-off: Computationally more expensive than RMS but more accurate

**Decision: Nyquist filtering for harmonics**
- Implementation: `remove_above_nyquist()` function zeros out harmonics above sampling_rate/2
- Why: Prevents aliasing artifacts in synthesis
- Design choice: Soft masking rather than hard cutoff for differentiability

### Reverb Module Design

**Design choice: Learnable reverb impulse response**

**Implementation:**
- Noise-based impulse generation
- Exponential decay envelope (softplus activation)
- Sigmoid for wet/dry mix control
- FFT-based convolution for efficiency

**Why learnable reverb:** Captures room characteristics from training data, improves audio quality

**Design detail:** Direct path preservation (impulse[:, 0] = 1) ensures dry signal always present

### Parameter Scaling for Vital

**Discovery: Vital uses non-linear parameter scaling**

**Finding:**
- Attack/decay: Quartic scaling (4th root)
- Sustain: Linear scaling
- References: Vital's source code (synth_parameters.cpp, value_bridge.h)

**Implementation:** VitalConverter applies these transformations during `parseToPluginFile()`

**Why important:** Without correct scaling, inferred parameters won't map to audible changes in the synthesizer

### Cross-Synthesizer Design Patterns (from Dexed Implementation)

**Note: Dexed is incomplete in Syntheon, but its converter reveals transferable design concepts**

**Concept 1: Parameter Space Compression**
- Dexed uses bit-packing to store multiple parameters in single bytes
- Example: RC|LC combines rate curve and level curve into one byte
- **Transferable insight:** When parameter space is limited (e.g., MIDI standards, preset file size constraints), consider efficient encoding schemes
- **Relevance to Vital:** Vital's JSON format doesn't have this constraint, but understanding compression helps if you need to work with other formats

**Concept 2: Binary Format Handling**
- Dexed uses struct-based binary parsing with checksum validation
- **Transferable insight:** Binary formats require careful byte-level handling and validation
- **Relevance to Vital:** Vital uses JSON/Base64, but binary parsing skills are valuable for other synthesizers

**Concept 3: Multi-Voice Architecture**
- Dexed supports multiple voices in single preset file
- **Transferable insight:** Some synthesizers organize presets as collections of voices/patches
- **Relevance to Vital:** Vital uses single-voice presets, but multi-voice concept could inform preset management systems

**Concept 4: Parameter Grouping**
- Dexed groups parameters into oscillator-level and global-level
- **Transferable insight:** Hierarchical parameter organization improves converter maintainability
- **Relevance to Vital:** Vital already has hierarchical structure (oscillators, envelopes, filters), this reinforces that pattern



## Limitations and Known Issues

### Wavetable Inference Limitations

**Issue 1: Silence detection**
- Problem: Algorithm prone to extracting silent wavelets
- Current workaround: Fallback to first pitch if no continuous segment found
- Status: Marked as TODO, needs robust solution

**Issue 2: Edge cases**
- Problem: Algorithm assumes stable pitch regions
- Fails on: Plucked sounds, pitch bends, vibrato
- Status: Not fully handled

### Performance Limitations

**Issue 1: Batch wavetable synthesis**
- Problem: Uses Python loop instead of vectorized operations
- Impact: Slow batch processing
- Proposed solution: Parallelize with einsum (marked as TODO)

**Issue 2: Fixed audio constraints**
- Problem: Fixed 16kHz, 4-second duration
- Impact: Cannot handle variable-length or higher-quality audio
- Status: Design choice, not planned to change

### Model Limitations

**Issue 1: Limited test coverage**
- Problem: Only Vital tested with 6 audio files
- Impact: Unknown generalization performance
- Status: Basic validation only

### Architecture Limitations

**Issue 1: Single synthesizer at a time**
- Problem: Cannot infer parameters for multiple synthesizers simultaneously
- Impact: Requires separate inference runs
- Status: Design choice

**Issue 2: No training pipeline documentation**
- Problem: No clear way to train new models or retrain existing ones
- Impact: Difficult to improve model performance
- Status: Documentation gap


## Experimental Evidence

### Test Results (Vital)

**Test setup:** 6 audio files with loss thresholds [0.11, 0.06, 0.37, 0.42, 0.18, 0.15]

**Findings:**
- Loss values vary significantly across test cases (0.06 to 0.42)
- Suggests model performance depends on input audio characteristics
- All tests pass current thresholds, indicating baseline functionality

**Limitation:** No information about what types of sounds work better/worse

### Configuration Experiments

**Learning rate schedule:** 0.001 → 0.0001 over 400k steps
- Rationale: Gradual decay for stable convergence
- Evidence of effectiveness: Model included in package (implies successful training)

**Multiscale STFT choice:** [4096, 2048, 1024, 512, 256, 128]
- Rationale: Captures multiple time-frequency resolutions
- Trade-off: Higher scales = better frequency resolution, lower scales = better time resolution

**Model dimensions:**
- n_harmonic: 100 (number of harmonic partials)
- n_bands: 65 (number of frequency bands)
- n_wavetables: 10 (number of wavetables in model)
- n_mfcc: 30 (MFCC features)

**Why these values:** Not documented, likely empirically determined


## Comparison with Related Work

### DDSP-PyTorch Relationship

**Adapted components:**
- Core signal processing functions (multiscale_fft, loudness extraction, etc.)
- MLP and GRU builders
- Spectral operations

**Differences:**
- Syntheon focuses on wavetable synthesis (DDSP focuses on additive synthesis)
- Syntheon includes ADSR envelope generation
- Syntheon has synthesizer-specific converters

### DiffSynth Relationship

**Adapted components:**
- ADSR envelope implementation
- Power function shaping
- Differentiable rounding

**Differences:**
- Syntheon uses wavetable synthesis (DiffSynth uses different approach)
- Syntheon targets specific synthesizers (DiffSynth is more general)


## Research Gaps and Future Directions

### Identified TODOs from Code

1. **Replicating state-of-the-art approaches**
   - Suggests current model may not be SOTA
   - Opportunity: Compare with latest research

2. **Improving current model performance**
   - Indicates room for improvement
   - Areas: Loss functions, architecture, training data

3. **Incorporating new synthesizers**
   - Extensibility is priority
   - Challenge: Each requires custom converter and model

4. **Code refactoring**
   - Suggests technical debt
   - Areas: Batch processing, edge case handling

### Unanswered Questions

1. **Training data:** What dataset was used? How large?
2. **Loss functions:** What specific loss functions are used? How are they weighted?
3. **Ablation studies:** What components are essential? What can be removed?
4. **Generalization:** How well does it work on unseen sounds?
5. **Comparison:** How does it compare to manual sound design?

### Potential Research Directions

1. **Robust wavetable inference:** Address silence detection and edge cases
2. **Variable-length audio:** Remove fixed duration constraint
3. **Multi-synthesizer models:** Single model for multiple synthesizers
4. **Real-time inference:** Optimize for live performance
5. **User studies:** Evaluate usefulness for sound designers


## Key References

- **DDSP-PyTorch**: https://github.com/acids-ircam/ddsp_pytorch (core signal processing adapted from here)
- **DiffSynth**: https://github.com/hyakuchiki/diffsynth (ADSR envelope influenced by here)
- **Vital Synth**: https://vital.audio/ (supported synthesizer)
- **Presentation**: https://www.youtube.com/watch?v=nZ560W6bA3o&t=1s (ADC22 talk - no transcript available, slides link broken)
- **Author's related work**: https://github.com/gudgud96/diff-wave-synth (PyTorch implementation of Differentiable Wavetable Synthesis, likely related to WTSv2 model)
