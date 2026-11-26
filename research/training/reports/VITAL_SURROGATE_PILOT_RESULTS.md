# Vital surrogate backbone pilot

Surrogate gradients improved real Vital reconstruction on unseen patches and MIDI. **Use the residual MLP as the starting backbone for the expanded codec in milestone 2.** Its expanded reconstructions were close to the Transformer's, and it was the only expanded arm with a paired 95% interval entirely above zero for the gain over matched forward-only search. [Issue #172](https://github.com/noisedoctrine/obruxo/issues/172)

Stereo MR-STFT measures how far the rendered patch is from the real Vital target. Lower is better. Each panel contains **64 held-out patch roots and 256 cases**, with two seen and two new MIDI performances per root. The table shows the **128 new-MIDI cases**, drawn from **16 distinct performances**. Compare backbones within each panel.

| Backbone | Basic MR-STFT ↓ | Expanded MR-STFT ↓ | Expanded gradient gain over forward-only ↑ |
|---|---:|---:|---:|
| Shared starting patch | 4.1994 | 5.4048 | — |
| Transformer | 1.8431 | 1.6261 | +0.0527 [−0.0297, +0.1460] |
| Conformer | **1.7751** | 1.7714 | +0.0248 [−0.0517, +0.0942] |
| Conditioned residual MLP | 1.8223 | **1.6202** | **+0.0791 [+0.0127, +0.1502]** |

## Reconstruction and forward prediction

| Backbone | Basic seen / new improved | Expanded seen / new improved | Basic / expanded new-MIDI forward error ↓ |
|---|---:|---:|---:|
| Transformer | 117/128 · 104/128 | 112/128 · 107/128 | 0.5776 · 0.5858 |
| Conformer | 121/128 · 102/128 | 111/128 · 108/128 | 0.5662 · 0.5641 |
| Conditioned residual MLP | 119/128 · 102/128 | 115/128 · 104/128 | **0.5639 · 0.5614** |

An improvement is a decrease in native MR-STFT beyond measured repeat-render noise, with a numerical floor of 0.00001. Forward error is the held-out stereo spectral and envelope loss. The forward-only search uses the same surrogate, initial categories and continuous bounds as gradient search. It makes random continuous proposals and both methods submit one final candidate to real Vital.

- **Basic.** All three arms benefited from gradients on new MIDI. The Conformer had the lowest mean native error. Its edge over the MLP was **0.0472 MR-STFT**, with a paired 95% interval of **[−0.0017, +0.1031]**.
- **Expanded.** The MLP led the Transformer by **0.0059 [−0.0788, +0.0868]** and the Conformer by **0.1511 [+0.0384, +0.2715]**. Its mean gain over forward-only search was larger on wet patches (**+0.1932**, 16 roots and 32 new-MIDI cases) than on dry patches (**+0.0410**, 48 roots and 96 cases). Treat the wet/dry difference as exploratory.
- **Basic sounds after expansion.** The change in mean new-MIDI improvement was **−0.0126** for the Transformer, **−0.0163** for the Conformer, and **+0.0280** for the MLP. Each paired interval included zero.

Paired intervals resample patch roots 10,000 times with seed 172 and keep both new-MIDI cases from each sampled root together. They describe patch-to-patch variability on this MIDI bank. This pilot used one learning seed, evaluator-known MIDI and a bounded categorical search. Confirm the gradient gain on fresh roots and MIDI with the Transformer as the matched comparator. Transcribed-MIDI inversion and downstream training are milestone 2 work. [Per-case scores](data/issue172_surrogate_cases.csv.gz) · [Aggregate metrics and paired intervals](data/issue172_surrogate_summary.json) · [Issue #173](https://github.com/noisedoctrine/obruxo/issues/173)

## Basic Vital control scope

These are inputs to the surrogate. Direct inversion optimizes active continuous inputs and searches legal categorical candidates. The backbone predicts stereo spectra and envelopes.

| Control family | Changeable / total | Surrogate input slots |
|---|---:|---:|
| Oscillator pitch, level, pan, enable and unison ✅ | 4 / 21 | 6 |
| Main filters ✅ | 3 / 32 | 3 |
| Envelopes ✅ | 4 / 54 | 4 |
| Oscillator waveform assets ❌ | 0 / 3 | 0 |
| Oscillator phase, warp, frame and routing controls ❌ | 0 / 69 | 0 |
| Reverb ❌ | 0 / 13 | 0 |
| LFOs and random sources ❌ | 0 / 136 | 0 |
| Modulation routing and maps ❌ | 0 / 512 | 0 |
| Sampler and sample assets ❌ | 0 / 12 | 0 |
| Chorus, compression, delay, distortion, EQ, FX filter, flanger and phaser ❌ | 0 / 99 | 0 |
| Global, voice and performance controls ❌ | 0 / 28 | 0 |
| **Total** | **11 / 979** | **13** |

[Exact per-control inventory](data/issue172_surrogate_control_scope.jsonl.gz)

## Expanded Vital control scope

The expansion adds two waveform selections, three filter responses, oscillator 2 enable/level/octave/fine tune, and reverb enable/mix/decay. Each switch uses one binary input slot. Reverb mix and decay are active only when reverb is on. The 128 wet roots belong to the 256 added roots, for 25% wet across the combined set of 512 training roots.

| Control family | Changeable / total | Surrogate input slots |
|---|---:|---:|
| Oscillator pitch, level, pan, enable and unison ✅ | 8 / 21 | 12 |
| Main filters ✅ | 4 / 32 | 6 |
| Envelopes ✅ | 4 / 54 | 4 |
| Oscillator waveform assets ✅ | 2 / 3 | 8 |
| Oscillator phase, warp, frame and routing controls ❌ | 0 / 69 | 0 |
| Reverb ✅ | 3 / 13 | 3 |
| LFOs and random sources ❌ | 0 / 136 | 0 |
| Modulation routing and maps ❌ | 0 / 512 | 0 |
| Sampler and sample assets ❌ | 0 / 12 | 0 |
| Chorus, compression, delay, distortion, EQ, FX filter, flanger and phaser ❌ | 0 / 99 | 0 |
| Global, voice and performance controls ❌ | 0 / 28 | 0 |
| **Total** | **21 / 979** | **33** |

[Exact per-control inventory](data/issue172_surrogate_control_scope.jsonl.gz)

## Execution and reproducibility

Each root has four MIDI performances and one original/perturbed pair per performance. Training used **1,024/1,024 basic pairs and 1,022/1,024 new expanded pairs**. Three renders in two expanded pairs failed audio silence QA. Expanded training reused the retained basic pairs and widened each fitted model's input layer. Each arm received 4,000 matched updates per stage. The evaluator used all six basic categorical combinations and shortlisted six of 24 prescreened expanded combinations for 16 optimization steps. All **768 planned evaluation cases** scored, including the basic preservation panel. The separate confirmation and test reservations stayed held out.

The full run took **3 hours 12 minutes** and peaked at **4.38 GiB** of process-tree RAM on the local Windows CPU. These are sequential timings from this run. The forward and backward columns cover the gradient search on a new-MIDI case.

| Backbone | Basic fit | Expanded fit | Expanded forward per case | Expanded backward per case |
|---|---:|---:|---:|---:|
| Transformer | 15m 51s | 8m 16s | 0.380 s | 0.297 s |
| Conformer | 17m 07s | 17m 31s | 0.516 s | 0.417 s |
| Conditioned residual MLP | 14m 36s | 17m 28s | 0.431 s | 0.283 s |

Model sizes were 786,273–789,921 parameters. Native preflight checked that the three filter modes produced ordered spectral centroids and that the maximum reverb decay cleared the two-second tail check. The published manifest captures the source fingerprints and the exact authored patch/MIDI assignments. [Frozen plan](data/issue172_surrogate_plan.json.gz) · [Aggregate metrics](data/issue172_surrogate_summary.json)

From the repository root, configure the Windows PyTorch and pinned Vita/DawDreamer import paths described in [AGENTS.md](../../../AGENTS.md) and the [local native runtime setup](../../modelling/simple_training/README.md#run). Use fresh output directories. The published source fingerprints hash the original checkout bytes, including line endings.

```powershell
$surrogatePy = Join-Path $env:USERPROFILE 'miniforge3/envs/py312/python.exe'
$surrogateVita = (Resolve-Path 'research/data_generation/outputs/smoke-deps').Path
$surrogateNative = Join-Path $env:USERPROFILE 'miniforge3/envs/obruxo-data-generation/Lib/site-packages'
$env:PYTHONPATH = "$surrogateVita;$surrogateNative"
$surrogatePool = 'research/training/outputs/issue172-reproduction-pool'
& $surrogatePy -B research/training/run.py pilot --count 4096 --seed 0 --output $surrogatePool
& $surrogatePy -B -u research/training/reports/scripts/run_surrogate_pilot.py `
  --pool "$surrogatePool/pool.json" `
  --reservation research/training/reports/data/basic_confirmation_reservation.json `
  --output research/training/outputs/issue172-reproduction
```

The runner saves a status file, native receipts, checkpoints, and per-case scores. Run [the progress monitor](scripts/show_surrogate_pilot.py) with the output directory to view the active stage. Use `--smoke --steps 2 --inversion-steps 2` for a reduced end-to-end check and `--probe` for the full cohort manifest and native capability checks without fitting.
