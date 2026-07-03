# Modulation Discovery with DDSP - Research Notes

## Paper Details
**"Modulation Discovery with Differentiable Digital Signal Processing"** (WASPAA 2025)
- Authors: Christopher Mitcheltree, Hao Hao Tan, Joshua D. Reiss
- arXiv: https://arxiv.org/abs/2510.06204
- HTML: https://arxiv.org/html/2510.06204v1

**Author link**: Hao Hao Tan (same guy who wrote Syntheon)
- GitHub: https://github.com/gudgud96
- Publications: https://gudgud96.github.io/publications/
- Email: helloharry66@gmail.com


## Why This Matters for Our Vital Project

This is actually pretty important. Hao Hao Tan (the Syntheon author) co-authored this paper on discovering modulation signals from audio. Here's why it's relevant:

**Vital is specifically mentioned in the paper** - they cite Vital as an example of a soft synth that emphasizes modulation with drawable XY grids and advanced routing. They also mention that ~98% of Serum presets use modulation, so Vital is probably similar.

**The big gap in Syntheon**: Syntheon only infers static parameters. It doesn't handle time-varying modulation signals (LFOs, envelopes, automation). But Vital sounds often rely heavily on modulation - this paper fills that exact gap.

**Same technical foundation**: They use DDSP (Differentiable Digital Signal Processing), just like Syntheon's WTSv2 model adapted from DDSP-PyTorch.

**Similar architecture**: They built a "Mod Synth" with:
- Wavetable oscillator (controllable wavetable position)
- Resonant filter (controllable coefficients)
- Envelope

This is basically Vital's core architecture.


## What They Did

### The Core Idea
Instead of just predicting static synth parameters, they discover the actual modulation signals (LFOs, envelopes, automation curves) that make a sound evolve over time. They use:
- Modulation extraction (neural network)
- DDSP synthesis (differentiable synth)
- Self-supervised sound matching

### Three Modulation Parameterizations They Tried

1. **Framewise (default DDSP)**: Frame-by-frame at control rate f_s/100. High-dimensional, not very interpretable.

2. **Low-pass filtered**: Post-process with LPF at f_c < 20 Hz. Makes smooth LFO-like signals. This is probably what we want for Vital's LFOs.

3. **2D Bézier curves**: Inspired by drawable XY modulation grids (like Vital has). Piecewise Bézier curves with control points. Advantages:
   - Compact, low-dimensional representation
   - Differentiable (closed-form)
   - Interpretable - curve stays within convex hull of control points

The Bézier curve approach is particularly interesting because it matches Vital's XY modulation grids exactly.

### LFO-net
They use something called "LFO-net" (from prior work) to extract modulation signals. Key points:
- High inductive bias for modulation extraction
- Trainable with limited data
- MLP adapter converts to higher dimensions when needed (e.g., for filter coefficients)
- Information bottleneck + adapter gives interpretability at cost of some accuracy

### Their "Mod Synth" Architecture
Three differentiable modules in series:
1. Wavetable oscillator (PyTorch grid_sample)
2. Resonant filter (time-varying biquad filter)
3. Envelope

Training is self-supervised: compare input/output audio with perceptual loss.


## Techniques We Could Borrow

### For Vital LFOs
- **LFO-net**: Study this for Vital LFO parameter inference
- **Low-pass filtering**: Use this for envelope smoothing
- **Bézier curves**: Perfect match for Vital's XY modulation grids

### General Approach
- **Information bottleneck + adapter**: Gives interpretability at cost of accuracy. Trade-off worth considering.
- **Self-supervised sound matching**: Can work on any evolving audio, flexible for discovering modulation shapes.


## Comparison: Syntheon vs This Paper

| Aspect | Syntheon | This Paper |
|--------|----------|------------|
| Focus | Static parameters | Time-varying modulation |
| Modulation | Not addressed | Core focus |
| Parameterization | Static values | Framewise, LPF, Bézier |
| Interpretability | Limited (black box) | High (constrained) |
| DDSP | Yes (WTSv2) | Yes (direct) |
| Vital | Direct target | Cited, similar arch |
| LFOs | Not mentioned | Core component |


## Practical Next Steps

### Short-term
- Check Hao Hao Tan's GitHub for the code (paper says they released it)
- Study LFO-net architecture for Vital LFO inference
- Implement LPF parameterization for envelope smoothing
- Investigate directly predicting Bezier curves for Vital's XY grids




## Resources
- Paper says they released code and audio samples
- They also provide trained DDSP synths as a VST plugin
- Check https://github.com/gudgud96 for implementation
