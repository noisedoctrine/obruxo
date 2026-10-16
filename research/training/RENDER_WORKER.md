# Reusable isolated rendering

`RenderWorker` keeps one isolated Python process and its imported libraries alive across sequential jobs. Every job still constructs a fresh renderer, engine and Vital instance, loads the full patch, reads back the native state, hashes the plugin and checks the actual audio. Shared memory holds one bounded stereo float32 clip; audio is discarded after use.

The process restarts after at most 32 jobs by default, or immediately after a crash, Python error, timeout or cancellation. The next call can recover with a new process. A rejected audio buffer is reported and never delivered. A worker accepts only one active call at a time and must be closed, normally with a context manager.

```python
with RenderWorker(renderer_factory=NativeRendererFactory(plugin_path), max_jobs=32) as worker:
    audio, metadata = worker.render(request, timeout=30)
```

The online queue and trainer accept `StreamConfig(worker_jobs=32)`. The CLI exposes `--worker-jobs 32` on `benchmark` and `train`. Allowed lifetimes are 1–128 jobs. The existing one-job default retains its exact saved contract, so old checkpoints and feature caches still load. Reuse has a distinct stream contract and cannot silently replace the execution policy of a resumed run. The byte/clip queue limits and explicit acknowledgements still apply.

## Correctness uses fixed phase

Pipeline reproducibility checks disable oscillator phase randomization in diagnostic copies of patches. They keep the patch's fixed phase knobs and all other synthesis settings. The original pool is unchanged. Codec equality is checked before applying this diagnostic control to the rendered source/decoded pair, so it cannot conceal a codec error. The pilot and alpha codec checks record this policy and use a new check identity.

The worker check alternates five training families, compares fresh processes with reuse across the entire 32-job lifetime, then checks the restart. Its correctness gate requires identical float32 audio, native state readback and renderer provenance. Speed and memory are separate gates. Timing uses adjacent pairs with alternating execution order, including cold starts and process recycling.

Vital's [oscillator reset](https://github.com/mtytel/vital/blob/main/src/synthesis/producers/synth_oscillator.cpp) draws random phases; its [random generator](https://github.com/mtytel/vital/blob/main/src/synthesis/framework/utils.h) seeds new generators using a process-wide incrementing counter. This source behavior is consistent with the observed differences when reusing processes with oscillator randomness enabled. Random-phase variation therefore belongs in a separate sampling check, rather than the fixed-input pipeline correctness gate.

Actual training respects the requested patch's phase settings. A reuse contract records the process-scoped random-number policy. With random phases enabled, regenerated unconsumed audio need not be bit-identical after interruption; retained features, model, optimizer and training RNG still recover from the checkpoint. Cross-platform/native-version qualification remains separate.
