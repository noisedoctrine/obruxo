# Runtime parity fixtures

These fixtures are generated entirely from the committed Vita init contract and synthetic MIDI events. They contain no Vital factory preset, preset bank, plugin binary, or downloaded corpus data.

Regenerate deliberately from the repository root:

```powershell
conda run --no-capture-output -n py312 python research/data_generation/tests/fixtures/generate_fixtures.py --force
```

Preset fixtures cover canonical init, oscillator 1 only, oscillator 1 through filter 1, one LFO route with its paired slot scalars, one enabled reverb, and one omitted-scalar document used only to test explicit runtime default filling. Performance fixtures cover a held note with an explicit release tail, a same-tick note-off/note-on boundary, an overlapping chord, and a tempo change that the first renderer must reject.

`reference_results.json` records aggregate render QA, request IDs, hashes, and measured repeatability for the installed reference Vital build. The matrix covers every pairing of the five component presets and three supported performances. Tolerances are calibrated above the observed maxima (1% relative RMS, 0.10 absolute peak, and 0.35 `log1p` spectral-magnitude RMSE), while lossless audio is intentionally not committed.
