# Broader authored synthesis training

The alpha path makes synthesis coverage trainable before more narrow pilot tuning. It uses the existing on-demand renderer, frozen audio frontend, bounded feature replay, recovery, evaluation and offline export. The pilot codec and its checkpoints remain supported.

## Supported vocabulary

| Component | Alpha v1 support |
|---|---|
| Oscillators | Two, with independent level/pan; oscillator 1 has 1/2/4 unison voices; oscillator 2 has fine tuning and 0/−12/+12 semitone choices |
| Wavetables | Reviewed init plus hash-pinned Basic Shapes, Jaw Harp and Stabbed, independently selected per oscillator; frame scanning for external tables |
| Filter and envelope | Filter 1 cutoff/resonance and envelope 1 ADSR |
| Modulation | One ordered route: triggered, free-running LFO 1 → filter 1 cutoff; triangle/ramp-up/ramp-down shapes, frequency and amount |
| Effect | Chorus on/off, mix, modulation depth and rate; fixed effect order |
| Targets | 19 continuous controls and 9 categorical heads, derived from the saved codec |

Control bounds use Vital's stored units, not displayed seconds or hertz. Every non-target control remains fixed to the reviewed authored template.

The pool cycles through five authoring levels: wavetable, layered, filtered, modulated and chorus. Each level adds the named component to the earlier ones. Training shuffles this mixed population; this is a deterministic coverage curriculum, not an adaptive difficulty schedule. Each state receives a frozen ancestry/split assignment. Reference and transfer performances retain their existing roles.

Disabled oscillator, filter, route and chorus controls use fixed defaults and masked supervision. Conditional categorical heads are masked too. The single-frame init wavetable has no frame target. Decoding canonicalizes predictions; encoding rejects unsupported assets, routes, shapes, fixed settings and inactive-state deviations. It does not project arbitrary corpus presets into the smaller vocabulary.

Third oscillators, arbitrary modulation graphs, arbitrary LFO geometry, effect chains, samples and full modern-corpus decoding remain outside this codec. The separate 1.5.5 full-contract research is not the alpha decoder. The model still uses supervised parameter loss and one predicted candidate; paired identity losses, audio-based optimization and candidate refinement are subsequent independent directions.

## Factory import boundary

[`manifests/alpha_factory_v1.json`](manifests/alpha_factory_v1.json) commits relative paths, provenance declarations and exact source hashes. Supply the matching installed factory directory. No factory payload or audio corpus is committed.

The three known hashes have explicit import conversions. Basic Shapes changes its 0.8.1 table header to 1.0.8. The two 0.3.9 audio-source tables also rename `post_normalize` to `full_normalize`, materialize `interpolation_style=1` and `normalize_mult=true`, and fill a missing author string. Their original source bytes stay embedded and hash-checked in the local pool and inference bundle. Imported content hashes, the conversion policy and the full codec descriptor are also pinned. A different file receives no guessed migration.

The direct native import check compares the originals with the converted assets at frames 0, 127.5 and 255. The original reference goes directly through Vital's legacy importer; the candidate uses the unchanged production validation path. The training renderer has no validation bypass. Other legacy tables without an explicit version require separate qualification.

## Run

Use the pinned training/data-generation environment. Outputs must be new directories; generated artifacts under `outputs/` are ignored by Git.

```sh
python research/training/run.py alpha-pool \
  --wavetable-manifest research/training/manifests/alpha_factory_v1.json \
  --wavetable-root /path/to/Vital/Factory/Wavetables \
  --output research/training/outputs/alpha --count 256 --seed 0

python research/training/run.py codec-check \
  --pool research/training/outputs/alpha/pool.json \
  --output research/training/outputs/alpha-check --timeout 30

python research/training/run.py train \
  --pool research/training/outputs/alpha/pool.json \
  --output research/training/outputs/alpha-smoke --mode online --steps 256 \
  --replay-capacity 64 --replay-mib 8 --batch-size 8 --learning-rate 0.001 \
  --sampling-policy state_round_robin --updates-per-render 4 --fresh-examples 64 \
  --render-budget 80 --checkpoint-every 128 --cpu-threads 4 --timeout 30

python research/training/run.py evaluate \
  --pool research/training/outputs/alpha/pool.json \
  --checkpoint research/training/outputs/alpha-smoke/checkpoint.pt \
  --output research/training/outputs/alpha-smoke-eval --count 2 --timeout 30

python research/training/run.py infer \
  --model research/training/outputs/alpha-smoke/model.pt \
  --audio /path/to/input.wav --output research/training/outputs/predicted-alpha.vital
```

The 64-example run verifies the training path; it is not evidence of reconstruction quality. `codec-check` plans 14 interventions/56 renders for this bank, retains every failure, checks native repetition and codec round trips, and requires each intervention's spectral distance to exceed 0.01 and five times measured repeat noise. These checks cover feature activation, not the entire combinatorial parameter domain or every MIDI performance.

To repeat factory conversion qualification:

```sh
python research/training/reports/scripts/qualify_alpha_imports.py \
  --wavetable-root /path/to/Vital/Factory/Wavetables \
  --plugin-path /path/to/Vital.vst3 --output research/training/outputs/alpha-import-check
```

## Compute and next experiments

Audio is rendered during ingestion and discarded after feature extraction. This smoke run caps retained features at 64 examples/8 MiB, uses one render worker and four CPU training threads, and needs no GPU. The model and bundle carry their codec and assets; inference does not require the source factory directory. Resume and feature seeding retain their existing strict same-pool contracts. Pilot weights/replay cannot be resumed as alpha weights/replay.

Free Colab remains an optional optimizer host once a compatible feature prefix is available. Native on-demand rendering there still requires Linux plugin qualification; the tested Windows binary is not a Linux backend. Do not assume a GPU removes native rendering costs. Measure coupled throughput before increasing render workers, model size or the fresh-example budget. GitHub Actions remains disabled; checks run locally.

The next screens should hold this broader vocabulary and a development panel fixed:

1. **Synthesis identity:** learn a shared patch representation across different performances of the same training state; test unseen performances and report family-specific reconstruction. Merely adding two unrelated supervised examples is not an identity objective.
2. **Audio-based objective:** screen a small render-supervised surrogate or render/distill loop that supplies audio similarity during learning. Measure surrogate agreement against real Vital before relying on its gradient.
3. **Ambiguity and refinement:** predict several valid candidates, render them under an audio-derived performance estimate and refine within a fixed inference budget. Never use oracle MIDI in production inference.

Use one bounded feasibility screen per direction before seed replication or long learning curves. Compare compute, native QA and reconstruction on the same planned denominator; reserve final test states. Combine viable directions before detailed head tuning. The previous release diagnostic remains useful evidence, but its proposed release-head comparison is deferred.
