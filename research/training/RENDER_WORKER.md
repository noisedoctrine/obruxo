# Reusable isolated rendering

`RenderWorker` keeps one isolated Python process and renderer alive across sequential jobs. The qualified basic profile also keeps Vital loaded: it loads the complete patch once, then updates the eleven controls using the full loader's float32 normalization. Every job reads back native state, hashes the plugin and checks the audio. Shared memory holds one bounded stereo float32 clip; audio is discarded after use.

The process restarts after at most 128 jobs by default, or immediately after a crash, Python error, timeout or cancellation. The next call can recover with a new process. A rejected audio buffer is reported and never delivered. A worker accepts only one active call at a time and must be closed, normally with a context manager.

```python
with RenderWorker(renderer_factory=NativeRendererFactory(plugin_path)) as worker:
    audio, metadata = worker.render(request, timeout=30)
```

The online queue, trainer CLI and `NativeSession` collections/fits now default to 128 jobs per worker. The CLI exposes `--worker-jobs` on `benchmark` and `train`; allowed lifetimes are 1–128 jobs. Fresh correctness comparisons use this same worker implementation with `max_jobs=1`. There is no duplicate one-shot worker path. The v3 contract records that qualified basic patches reuse the plugin while other patches get fresh instances.

The basic profile requires the exact staged control bounds, legacy training renderer, 44.1kHz, 128-sample buffers, 32,768-frame preroll, fixed oscillator/sampler phases and unchanged init assets/routes/effects. It validates the fixed structure once, checks the changing controls each request, clears MIDI and uses the host render reset. Changed structure or unsupported requests take full validation and loading in the same renderer. This preserves broad preset support without a duplicate implementation.

Stream contracts and native receipts record the new renderer lifecycle. Historical runs must use their recorded source revision; the new implementation rejects silent continuation under a changed policy. The byte/clip queue limits and explicit acknowledgements still apply. `NativeSession.close()` releases its worker; a finalizer also handles object/process teardown.

## Correctness uses fixed phase

Pipeline reproducibility checks disable oscillator phase randomization in diagnostic copies of patches. They keep the patch's fixed phase knobs and all other synthesis settings. The original pool is unchanged. Codec equality is checked before applying this diagnostic control to the rendered source/decoded pair, so it cannot conceal a codec error. The pilot and alpha codec checks record this policy and use a new check identity.

The current worker check alternates basic unison/filter combinations, held notes and chords, including A–B–A ordering. It fixes oscillator and sampler phases and uses 32,768 frames of preroll. It compares fresh processes with reuse across the entire 128-job lifetime, then checks the restart. Its correctness gate requires bounded float32 audio error, identical native state readback and renderer provenance. A complete saved update separately checks loss, gradient and weight-update stability. The numeric limits are explicit in the recorded plan; audio hashes and tiny trace differences are reported rather than called bit-identical. Timing uses adjacent pairs with alternating execution order, including cold starts and process recycling. See the [current measurements](reports/RENDER_WORKER_RESULTS.md).

Vital's [oscillator reset](https://github.com/mtytel/vital/blob/main/src/synthesis/producers/synth_oscillator.cpp) draws random phases; its [random generator](https://github.com/mtytel/vital/blob/main/src/synthesis/framework/utils.h) seeds new generators using a process-wide incrementing counter. This source behavior is consistent with the observed differences when reusing processes with oscillator randomness enabled. Random-phase variation therefore belongs in a separate sampling check, rather than the fixed-input pipeline correctness gate.

Actual training respects the requested patch's phase settings. A reuse contract records the process-scoped random-number policy. With random phases enabled, regenerated unconsumed audio need not be bit-identical after interruption; retained features, model, optimizer and training RNG still recover from the checkpoint. Cross-platform/native-version qualification remains separate.

## Laptop execution

Windows standby was observed during local checks. A long scheduling pause can expire the wall-clock render deadline; that failed attempt remains in the log and may advance the recipe cursor. Exact recovery comparisons require the same retained features and accepted recipe sequence. Long unattended local runs need a continuously awake host, or checkpointed recovery with the resulting render failures counted explicitly. The pipeline does not change global power settings.
