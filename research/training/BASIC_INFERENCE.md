# Export a basic prediction from audio

The `infer` command accepts the original pilot export and both direct basic-model formats, including the trainable audio-feature branch. Use an exported `model.pt` or `diagnostic-spectral.pt`; training `checkpoint.pt` files also contain optimizer/replay state and are not inference artifacts.

```powershell
python research/training/run.py infer --model research/training/outputs/basic-direct-native-fit-20261024/diagnostic-spectral.pt --audio input.wav --output prediction.vital
```

Input is a 44.1 kHz WAV. Mono is copied to both channels; stereo retains the pan and unison cues. The model receives audio only. The embedded frontend, codec, template and assets produce one legal `.vital` preset. Existing output files are preserved.

Direct basic models keep **nine continuous controls and five categorical scores**, reduced to the same **11 codec selections**. The saved correction and copied input layers are restored in full. Profiled models retain their fixed oscillator/sampler random phases. Direct models use two Torch CPU threads, matching their native-fit audits; original pilot artifacts keep the command's existing four-thread setting. `--device` remains available.

Diagnostic exports retain `qualified_for_release: false`, which is also included in the command's JSON result. Reconstruction admission, fresh confirmation, listening and fresh-environment bundle qualification remain separate training-plan requirements.

Implementation: [prediction loader](obruxo_training/prediction.py). Verification covers the actual CLI, nonzero saved corrections, both basic formats, legacy behavior, fixed phases, changed weights/contracts and existing-file preservation.
