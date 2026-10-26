# A portable local basic-model snapshot

Package an exported basic model together with its project runtime and schema data:

```powershell
python research/training/run.py bundle-basic --model research/training/outputs/basic-direct-native-fit-20261024/diagnostic-spectral.pt --output research/training/outputs/local-basic-model
python -I research/training/outputs/local-basic-model/run.py --verify-only
python -I research/training/outputs/local-basic-model/run.py --audio input.wav --output prediction.vital --report inference.json
```

The directory includes the unchanged model file, embedded frontend weights, codec/default assets, required Python packages from the project, static Vital schema and reviewed init preset. It can be moved outside the checkout. The launcher verifies file hashes and runtime versions before loading the model, and rejects project imports from outside the snapshot. Existing presets and reports are preserved.

Python and third-party runtime packages are separate. `bundle.json` records Python and direct dependency versions, and its `requirements.lock` records those six packages. The [Windows x64/Python 3.12.13 lock](runtime/windows-cp312-xpu.lock) additionally pins all 36 distributions and wheel hashes used by the [fresh-install qualification](reports/CLEAN_BASIC_RUNTIME_RESULTS.md). Other platforms need their own lock and check. The supported snapshot scope is the 14-output basic codec, including legacy pilot, direct correction and trainable-input exports. Diagnostic qualification status is preserved; snapshot release qualification stays false.

To repeat the qualified offline installation with that wheelhouse and Python version:

```powershell
python -I -m venv runtime-env
runtime-env/Scripts/python.exe -I -m pip --isolated install --no-index --find-links runtime-wheels --only-binary=:all: --require-hashes -r research/training/runtime/windows-cp312-xpu.lock
runtime-env/Scripts/python.exe -I -m pip check
runtime-env/Scripts/python.exe -I research/training/reports/scripts/check_local_runtime.py --bundle local-basic-model --audio input.wav --expected-preset-sha256 SAVED_PRESET_SHA256 --output runtime-check
```

`runtime-wheels` must contain the locked wheels. The tested environment occupies 4.70 GiB; reserve 8 GiB for environment and wheelhouse together. The checker requires a fresh environment without shared packages, compares exact preset bytes and records dependency-module/native-library origins. Its output directory must be new. A Windows venv uses its own oneMKL libraries even when its base interpreter comes from Conda. The XPU distribution is retained for reproducibility; inference runs on CPU.

The builder copies the available project and Basic Pitch readmes into its notice inventory. Complete release notices and model-quality/listening qualification remain required. The original isolated-source check shares installed packages; the separate fresh-install result above closes dependency isolation only for its recorded platform/runtime.

The implementation leaves the running sound fit's frozen numerical files unchanged. [Audio-to-preset usage](BASIC_INFERENCE.md) also works directly from the source checkout.
