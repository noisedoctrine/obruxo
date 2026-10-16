# Optional candidate ranking from one recording

The [completed alpha check](reports/AUDIO_RANKING_RESULTS.md) passed the initial optional-path gates: 10.64% lower mean sound error, 100% ranked coverage, 15.99s model-warm p95 and 159/160 accepted native attempts. These are bounded development results; broader coverage and release gates remain open.

This implements the bounded candidate direction in Stage E of the [training plan](../modelling/TRAINING_PLAN.md). The model still accepts one audio clip. It can propose up to four legal patches and compare their rendered sounds using notes estimated internally from that same recording. The user does not supply MIDI.

The existing frozen Basic Pitch pass now exposes its dense posteriors alongside the unchanged timbre/pitch features. Existing frontend IDs, feature values, model heads and ordinary inference bundles remain compatible. The same pass serves patch prediction and stock note decoding; no transcription model is trained or fine-tuned.

The candidate bundle embeds the original inverse model, frozen frontend, codec/factory assets and at most 32 distinct training-only retrieval states. Export validates the source checkpoint and split membership. Inference needs no source pool, wavetable installation or separate training files. Model/bank hashes and frontend contracts are checked on load.

## Selection and limits

The proposer reuses the four-choice policy from direct-render training: model, training retrieval, one categorical alternative and one bounded continuous alternative, deduplicated in that order. It receives audio features and model outputs, never source settings or known notes.

Basic Pitch's stock decoder supplies note pitches and times. Posterior amplitude maps linearly to a MIDI velocity as an initial heuristic; it is not claimed to recover true performance velocity. Times are quantized at 960 ticks/second under tempo 120. Pitch bends are not represented. At most 32 notes and 16 seconds of source audio are supported by this first implementation.

Every candidate is rendered under the same estimated performance. Ranking uses native multiresolution spectral distance, zero padding to the longer recording and preserving gain. The additional synth tail may make a candidate longer than the input. At most four native attempts occur; ties keep proposal order.

No detected notes, excessive/invalid notes or rejected renders produce an explicit unranked ordinary-model fallback. They never silently produce the init patch or count as successful ranking. The report records the reason, candidate scores, actual render QA, estimated notes and elapsed time.

## Qualification and commands

The shipped path and evaluator-only best-of-four are measured separately. The native check freezes a balanced validation panel, source hashes, ranking identity and gates before rendering. It invokes the public audio-only ranker without known notes, then uses evaluator notes to measure its chosen patch, the ordinary prediction, training retrieval and diagnostic best-of-four under reference and reserved performances. Source-repeat audio measures variability. Oracle choices never enter deployed selection.

Initial feasibility gates require at least 5% lower mean reference/transfer error than ordinary prediction, at least 80% ranked coverage, at most 1% native failures, a measured 95th-percentile warm ranking time of 20 seconds or less, and sampled peak process-tree memory no greater than 8 GiB. This small development comparison is not a final release test.

```sh
python research/training/run.py candidate-check --pool /path/to/pool.json \
  --checkpoint /path/to/checkpoint.pt --output research/training/outputs/audio-ranking-check

python research/training/run.py infer-candidates --model /path/to/qualified/candidate-model.pt \
  --audio recording.wav --output prediction.vital
```

An optional `--plugin-path` selects the pinned native Vital binary. The check uses ten validation patches by default, at most 160 planned native attempts and 16 recovery attempts. `--resume` reuses hash-verified complete cases and counts all unfinished-case attempts against the fixed cap. No test states or training updates are used.

A matching passed check marks its bundle qualified. `candidate-export --pool ... --checkpoint ... --output ...` can create a diagnostic unqualified bundle without rendering; ordinary `infer-candidates` refuses it unless `--allow-unqualified` is explicitly supplied. The ordinary `infer` command stays the default top-1 path. Both WAV interfaces accept stereo or duplicated mono input with the existing integer PCM scaling.

The report defaults to `prediction.ranking.json`; existing prediction/report files are refused before rendering. Audio and posterior arrays are never archived. Local CPU is the current execution route; Linux/Colab native rendering remains unqualified.
