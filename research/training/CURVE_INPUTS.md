# One-point and decimal curve inputs

The separate `curve-input-v1` renderer profile accepts original one-point LFOs, connection mappings and custom oscillator curves. It keeps the source preset in the recipe and records both requested and applied preset hashes. Before loading Vital, it rounds curve coordinates and powers to finite float32 values and expands a one-point line to a constant two-endpoint line at its stored y value. Every changed curve is listed in native QA. Other fields pass through the existing full-state loader and readback checks.

This import rule is explicit because the pinned Vital VST worker exits when rendering an active one-point LFO. The same one-point shape can be loaded and saved first, so successful readback alone is insufficient. Native one-point mappings and custom curves do render, and their audio exactly matches the constant expansion in the qualified fixtures. The one-point LFO's constant interpretation is a declared import rule; no unmodified native LFO waveform is available for comparison after the worker exits.

Ordinary multi-point curves retain their geometry and smoothing, with native float32 rounding of coordinates and powers. The source x position and power remain recorded for a singleton, whose constant interpretation uses its stored y. Invalid counts, lengths, ordering, coordinates and non-finite or float32-overflowing values fail. The full schema accepts 1–100 points; this panel qualifies singleton/constant and three-point precision fixtures, alongside the existing 97-point model output path. It is not a complete census of every corpus layout or curve combination.

Earlier renderer identities and decoder banks retain their behavior and identities. The new profile extends the existing expression/seed profile and keeps its sixteen-channel MIDI boundary, four-second training tail, exact seed-bit policy and native asset readback rules. The shared decoder implementation and historical `lfo-fit` policy are unchanged; this increment uses the previously qualified train-only bank. Broader corpus admission and a separately declared refitting policy remain open.

## Training sources

`CurveInputCodec` adds a separately identified source curriculum over the combined codec. It keeps original one-point shapes in rendered source recipes. Encoding and original-curve supervision use the same declared applied curves as the renderer, so audio and labels agree. Model inputs remain audio-derived timbre and frozen Basic Pitch features only. Model outputs retain the existing structured representation, including 97-point curve reconstructions; textual source point count is not a prediction target.

Four source families cover one-point LFOs, mappings, custom curves and their combination, assigned independently of the four synthesis scenes. The selected LFO and mapping get an enabled connection to a live generator's pan, with amount 0.05. The selected custom curve gets an enabled oscillator and custom mode 13/14/15 at amount 0.5; slot and mode cycle separately. These are declared authored source variants. Other source curves, registered assets, ordered slot identities, seed values and frozen root ancestry are retained. Source creation fails if a selected singleton is excluded by its supervision mask. Native audio QA remains required.

This activation policy corrects a measured gap in the initial transport test: it stored singleton curves while leaving the selected LFOs and custom curves inactive. The corrected integration requires every singleton family to be active and every retained example to be sampled in each update. The initial results and audit remain in the [completed report](reports/CURVE_INPUT_RESULTS.md).

## Run locally

```powershell
python research/training/run.py curve-pool --asset-bank path/to/factory-bank.json --lfo-bank path/to/lfo-bank.json --count 32 --output research/training/outputs/curve-pool
python research/training/reports/scripts/run_curve_input_discovery.py --output research/training/outputs/curve-discovery
python research/training/reports/scripts/run_curve_input_check.py --output research/training/outputs/curve-native
python research/training/reports/scripts/run_curve_precision_check.py --output research/training/outputs/curve-precision
python research/training/reports/scripts/run_curve_input_integration.py --pool research/training/outputs/curve-pool/pool.json --native research/training/outputs/curve-native --precision research/training/outputs/curve-precision --output research/training/outputs/curve-training
```

Use `FULL_CURVE_RENDERER_ID` when calling the renderer directly. The input policy and its source fingerprint are included in the full contract and codec descriptor; different policies require a new run. Qualification fixes all four generator random phases and uses fresh workers. Actual training preserves its requested phases and can use bounded worker reuse; exact optimization recovery reuses retained features.

The completed check uses free local CPU. Only metadata is committed. Linux/Colab, broader source/layout coverage, complete active graph/range coverage, useful reconstruction quality and release gates remain open. Final-test audio stays reserved and GitHub Actions stays disabled.
