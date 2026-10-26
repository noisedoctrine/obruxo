# A portable local basic-model snapshot

Package an exported basic model together with its project runtime and schema data:

```powershell
python research/training/run.py bundle-basic --model research/training/outputs/basic-direct-native-fit-20261024/diagnostic-spectral.pt --output local-basic-model
python -I local-basic-model/run.py --verify-only
python -I local-basic-model/run.py --audio input.wav --output prediction.vital --report inference.json
```

The directory includes the unchanged model file, embedded frontend weights, codec/default assets, required Python packages from the project, static Vital schema and reviewed init preset. It can be moved outside the checkout. The launcher verifies file hashes and runtime versions before loading the model, and rejects project imports from outside the snapshot. Existing presets and reports are preserved.

Python and third-party runtime packages are separate. `bundle.json` records Python and dependency versions, and `requirements.lock` records the installed packages. Install that runtime once; the launcher uses local files. The supported snapshot scope is the 14-output basic codec, including legacy pilot, direct correction and trainable-input exports. Diagnostic qualification status is preserved; snapshot release qualification stays false.

The builder copies the available project and Basic Pitch readmes into its notice inventory. Complete release notices, a clean dependency-install test and model-quality/listening qualification remain required. An isolated-source test with shared installed packages must not be described as a clean dependency installation.

The implementation leaves the running sound fit's frozen numerical files unchanged. [Audio-to-preset usage](BASIC_INFERENCE.md) also works directly from the source checkout.
