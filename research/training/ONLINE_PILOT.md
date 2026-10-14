# Authored pilot and online rendering

The first online data pipeline is implemented: a compact authored population, a strict pilot codec, a deterministic recipe sampler, process-isolated rendering, audio QA, bounded prefetch and an acknowledged resume cursor. Audio is synthesized as the consumer requests it and remains in memory. No audio archive is required.

The [first trainer](TRAINER.md) consumes this stream and supplies the inverse model, optimizer and full training checkpoint. Use its [current results](reports/FIRST_TRAINER_RESULTS.md); the earlier [native measurements](reports/ONLINE_PILOT_RESULTS.md) predate a discovered patch-loading bug.

## Commands

Run from the repository root. `pilot` and `expand-pilot` use the small [contracts environment](requirements-contracts.txt). `benchmark` and `codec-check` require the existing [native rendering environment](../data_generation/README.md#setup), including the hash-approved Vital VST3, DawDreamer 0.8.3 and pinned Vita runtime. An optional `--plugin-path` locates that same reviewed plugin; it does not bypass fingerprint validation.

```sh
python research/training/run.py pilot --output research/training/outputs/my-pilot --count 256 --seed 0
python research/training/run.py benchmark --pool research/training/outputs/my-pilot/pool.json --output research/training/outputs/my-benchmark --count 100 --timeout 30
python research/training/run.py codec-check --pool research/training/outputs/my-pilot/pool.json --output research/training/outputs/my-codec-check --timeout 30
python research/training/run.py benchmark --pool research/training/outputs/my-pilot/pool.json --output research/training/outputs/my-resumed-benchmark --count 4 --timeout 30 --resume research/training/outputs/my-benchmark/cursor.json
```

Output directories must be new. A benchmark returns exit code 2 if any requested example fails or is rejected; its report retains the full denominator. A codec check returns 2 unless all six categorical combinations are covered, all renders pass QA, stored controls match at float32 precision, and all reconstruction comparisons pass the existing renderer tolerances. Structural/command errors return 1.

For future held-out evaluation, select `--split validation --role transfer`. The CLI defaults to train/reference. Transfer performances cannot enter the training sampler. No validation or test audio was consumed in the measured training-stream run or codec qualification.

## Authored population

The default pool contains 256 independent draws from the declared control domains. Each draw starts from the reviewed init template, retains its known assets and receives a stored-state identity. The generator performs static eligibility checks and freezes train/validation/test assignments before rendering. A shared init template is a synthesis vocabulary, not a common example ancestor; later mutations of an actual example must inherit its lineage.

`pool.json` stores target vectors, state identities, codec/contract descriptions and the split manifest. It does not duplicate 256 embedded asset payloads. The current seed-0 v2 pool contains 205 train, 21 validation and 30 test states. Loading verifies its content hash, codec contract, reconstructed state identities and split assignments. V2 authors a native-compatible `1.0.8` preset version; it does not inherit Vita's synthetic `99999.9.9` marker. This changes source identities and therefore the frozen split assignments. Generate a new pool instead of editing an old manifest.

These are simple, related synthetic sounds. This split can test the learning pipeline, but cannot establish broad timbre generalization or downloaded-corpus readiness. The separate corpus eligibility report remains unchanged.

`expand-pilot --pool PARENT --count N --output NEW_DIRECTORY` increases the total independent population while verifying the original targets, ordered prefix and all frozen assignments. The [first expansion](reports/POOL_EXPANSION_READINESS.md) grows from 256 to 512 states, with 404 training states. It preserves the original 21 validation and 30 test states, and reserves the new held-out draws too. Use the original pool for fixed validation comparisons; enlarging the evaluation pool would change its sampling schedule. Expanded training begins as a fresh run because recipe/sampler identities change.

## Pilot codec

`authored_osc_filter_env_v2` supports one init oscillator, its level/pan/unison controls, optional filter 1 and envelope 1. The oscillator is routed to filter 1 as in the reviewed init state. Other oscillators, the sampler, effects and modulation routes remain fixed at their reviewed template state. This is deliberately a small initial vocabulary; the final model still needs the broader capability matrix.

| Continuous control | Raw stored range |
|---|---:|
| Oscillator 1 level | 0.15–0.35 |
| Oscillator 1 pan | −0.7–0.7 |
| Unison detune | 0.5–2.5 |
| Filter 1 cutoff | 50–120 |
| Filter 1 resonance | 0.1–0.6 |
| Envelope 1 attack | 0.08–0.6 |
| Envelope 1 decay | 0.25–1.0 |
| Envelope 1 sustain | 0.25–0.9 |
| Envelope 1 release | 0.2–0.65 |

Continuous targets linearly map these **stored units** to [0, 1]; the table does not express seconds, hertz or display values. Two categorical heads represent unison voices `{1, 2, 4}` and filter enable `{0, 1}`. Their targets are zero-based indices into those lists.

`PilotCodec.continuous_mask(target)` excludes detune supervision with one voice and cutoff/resonance supervision with the filter disabled. Decoding fixes inactive controls to template values. Encoding rejects altered inactive controls, unsupported categories/ranges, extra routes/effects and changes to other fixed state. It never silently projects a broader preset into the pilot. Only the three previously declared top-level display fields are ignored.

The decoder returns complete valid preset JSON, preserving the fixed ordered modulation/LFO structures. `codec-check` selects one training state for each categorical combination and renders the source twice plus the encode/decode reconstruction. It reports source-repeat noise separately from reconstruction differences.

## Stream and recovery

`PilotSampler` shuffles each epoch's state/performance pairs deterministically. An epoch covers every state in its selected split with each of the four performances in its selected role. Recipe sequence IDs address that schedule directly; restarting from an acknowledged position recreates the same recipe identities.

`OnlineRenderStream` uses one producer and one fresh native process per request. Process startup is included in measured job latency. This first implementation favors fault isolation; later worker reuse must preserve native state isolation and be qualified separately.

- Reserve expected float32 stereo bytes **before** dispatching a render. Both the clip and byte limits must have capacity.
- Use bounded shared memory for audio and a small pipe message for metadata. Discard rejected audio before delivery.
- Independently check the returned shape, dtype, sample rate, request provenance and actual audio. Reject nonfinite samples, silence, clipping, excessive tails and native validation errors.
- Apply a per-attempt deadline covering process startup and rendering. Kill and reap failed, timed-out or cancelled workers. Retry at most once when explicitly configured; QA rejection is not retried to select a favorable render.
- Deliver every final success, rejection or error in sequence. The consumer explicitly acknowledges each handled outcome. Prefetch never advances the persisted consumption position by itself.

The default stream uses **4 clips / 32 MiB** of prefetch, within the frozen contract's maximum **32 clips / 256 MiB**. These limits include reserved in-flight output and queued PCM, not total process memory. Native buffers, a temporary shared-memory copy, metadata, consumer-held arrays, features and model tensors require additional memory. Delivered audio is read-only. Keep consumer batches/replay bounded separately.

Use the stream as a context manager so early termination cancels the worker and releases its resources:

```python
from pathlib import Path
from obruxo_training.pilot import PilotPool, PilotSampler
from obruxo_training.stream import OnlineRenderStream

pool = PilotPool.load(Path("research/training/outputs/my-pilot/pool.json"))
sampler = PilotSampler(pool)
with OnlineRenderStream(sampler, count=100) as stream:
    for outcome in stream:
        if outcome.audio is not None:
            target = pool.codec.encode(outcome.recipe.render_request().preset_json.encode())
            # Pass outcome.audio, target and the codec's active-control mask to the consumer.
        # Persist the outcome and finish handling it before acknowledgement.
        stream.acknowledge(outcome.recipe.sequence_id)
    cursor = stream.state_dict()
```

Add the two research workspaces to Python's import path when using this API directly, as described in the [contracts README](README.md). At most `prefetch_clips` delivered outcomes may remain unacknowledged; acknowledge completed microbatches before requesting more.

The cursor binds the sampler and stream configuration to the next unacknowledged sequence. Resuming regenerates unacknowledged/queued requests; Vital may produce different audio bytes. This is recipe recovery, not exact audio replay. A trainer must checkpoint model, optimizer, RNG and this cursor together at an optimizer-step boundary. Persisting the stream cursor alone is insufficient to resume optimization safely.

## Benchmark output and interpretation

`benchmark` couples the renderer to a small CPU log-spectrum consumer, writing `events.jsonl`, `cursor.json`, `summary.json` and `status.json`. Events distinguish render outcomes, consumed accepted buffers and handled failures. They include actual audio hashes and renderer provenance. No PCM, WAV or feature archive is written.

Reported latency includes native process startup and cleanup. Prefetch high-water marks measure reserved queue PCM; RSS measures the sampled sum of parent and child resident sets, which can double-count shared pages. RSS is available when `psutil` is installed; otherwise the report records it as unavailable. Measurements cover one machine and this limited patch vocabulary.

This consumer performs no optimizer steps. The [trainer report](reports/FIRST_TRAINER_RESULTS.md) measures the actual frozen frontend and inverse-model consumer. Linux/Colab Vital qualification and the broad factory/schema work remain separate gates.
