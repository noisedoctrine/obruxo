# Paired-seed replication of 205 versus 404 training states

Completed 2026-10-14. The replication outcome is mixed: expanded training slightly worsened the primary score for seed 1 (+0.0026, 9/21 wins) and improved it for seed 2 (-0.0405, 11/21 wins); both paired state-bootstrap intervals include zero. All four runs passed the tiny-overfit gate, yet they do not show consistent held-out gains. No replication candidate is promoted.

The immutable pre-run plan is [state_coverage_replication_plan.json](data/state_coverage_replication_plan.json), SHA-256 `f178aa867177f245996438219d2365c1088565b526fd7c4deafcfbeee4bf5b2d`. The experiment adds paired learning seeds 1 and 2 to the seed-0 discovery experiment. For each seed, both population arms used sampler seed 0, the same frozen population-specific replay contents and order as their verified source checkpoint, a fresh model and optimizer, and exactly 16,000 updates. No new training render was requested. The original 21 validation cases and fixed 205-record retrieval bank were reused. The initialization-weight hashes were reconstructed by replaying the frozen initialization RNG sequence; they were not captured from a separate pre-update checkpoint.

This reuses the same 21 validation states already used for discovery and candidate selection. The intervals are descriptive state bootstraps, unadjusted for repeated validation or checkpoint selection, and do not establish uncertainty across learning seeds. The final test split remains reserved.

## Paired validation results

The primary score averages reference and transfer MR-STFT within each state; lower is better. Each contrast is expanded minus the 205-state arm, so negative values favor expanded training. Seed 0 is discovery evidence; only seeds 1 and 2 are the prespecified replication. State bootstrap intervals resample the 21 cases within a seed and do not represent uncertainty across learning seeds.

| Learning seed | Mean paired delta | Paired state bootstrap 95% interval | Expanded wins / 21 |
|---:|---:|---:|---:|
| 0 (discovery) | -0.0782 | [-0.2398, 0.0676] | 12/21 |
| 1 (replication) | 0.0026 | [-0.1776, 0.1585] | 9/21 |
| 2 (replication) | -0.0405 | [-0.2164, 0.1439] | 11/21 |

Across the two replication seeds, the mean of the paired deltas is **-0.0190**, with per-seed values from **-0.0405** to **0.0026**. Two seeds provide a small repeatability check, not strong evidence of statistical significance. The [comparison plot](images/state_coverage_replication.png) shows the per-seed paired means and descriptive 95% state-bootstrap intervals; the machine-readable summary also records the full per-state delta ranges.

| Candidate | Primary | Reference | Transfer | Active MAE | Attack | Decay | Sustain | Release | Osc level | Unison acc. | Filter acc. | RMS ref. | RMS transfer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| incumbent | 0.9870 | 0.9307 | 1.0432 | 0.1938 | 0.1682 | 0.2315 | 0.2041 | 0.3325 | 0.1839 | 0.8095 | 1.0000 | 0.2867 | 0.2632 |
| one seed0 16000 | 0.9906 | 0.9162 | 1.0649 | 0.2004 | 0.1724 | 0.2376 | 0.2399 | 0.3113 | 0.1568 | 0.8095 | 1.0000 | 0.2507 | 0.2596 |
| expanded seed0 16000 | 0.9124 | 0.8304 | 0.9943 | 0.1884 | 0.1369 | 0.2374 | 0.1977 | 0.3321 | 0.1522 | 0.5714 | 1.0000 | 0.2142 | 0.2146 |
| one seed1 16000 | 1.0073 | 0.9385 | 1.0762 | 0.1884 | 0.1129 | 0.2603 | 0.1806 | 0.3348 | 0.1736 | 0.6190 | 1.0000 | 0.2761 | 0.2743 |
| expanded seed1 16000 | 1.0099 | 0.9124 | 1.1074 | 0.1925 | 0.1456 | 0.2796 | 0.2408 | 0.3153 | 0.1141 | 0.5238 | 1.0000 | 0.2701 | 0.2256 |
| one seed2 16000 | 1.1670 | 1.1059 | 1.2280 | 0.2263 | 0.1795 | 0.2967 | 0.2628 | 0.3933 | 0.2014 | 0.6667 | 1.0000 | 0.4152 | 0.3471 |
| expanded seed2 16000 | 1.1264 | 1.0460 | 1.2069 | 0.1984 | 0.1619 | 0.2931 | 0.2093 | 0.3479 | 0.1712 | 0.7143 | 0.9524 | 0.3823 | 0.3234 |

Active-control and envelope/level errors are normalized codec units computed from the emitted `.vital` patch against the fixed target state. Unison and filter columns are categorical accuracy. RMS columns are audio-relative errors from the native evaluation. These diagnostics describe the same 21 cases and do not replace the audio metric.

## Training and QA

| Population | Learning seed | Steps | Replay records | Replay MAE start → end | Replay loss start → end | Final categorical accuracy | Tiny-overfit gate | Segment wall time | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| one | 1 | 16,000 | 205 | 0.2455 → 0.0456 | 1.8819 → 0.0043 | 1.0000 | pass | 3.42 min | 595 MiB |
| expanded | 1 | 16,000 | 404 | 0.2475 → 0.0291 | 1.8800 → 0.0017 | 1.0000 | pass | 2.78 min | 629 MiB |
| one | 2 | 16,000 | 205 | 0.2457 → 0.0163 | 1.8708 → 0.0005 | 1.0000 | pass | 3.02 min | 595 MiB |
| expanded | 2 | 16,000 | 404 | 0.2476 → 0.0347 | 1.8728 → 0.0023 | 1.0000 | pass | 2.87 min | 629 MiB |

The four replication runs completed **64,000 optimizer updates** in 12.08 recorded wall minutes (summing each 4K/8K/16K segment); peak sampled training process-tree memory was 629 MiB. Replay stayed below 32 MiB per run, used no training WAV archive, and all four summaries report zero new training renders. Native evaluation took 18.68 minutes and attempted 462/462 renders: 462 accepted, 0 rejected, and 0 errors. Total recorded training plus evaluation time was 30.77 minutes, excluding setup, tests, and review. The 629 MiB peak is training-only. Every candidate scored 21/21 states in both roles; all 147/147 prediction presets across the compared candidates passed pilot-codec and Vital-schema validation. Final-test states were not evaluated.

The combined suite passed **217 tests** with **15 environment-dependent skips**, including the restored `test_training.py` and simple-training regressions; Ruff passed. The frozen plan and run-time source hashes were verified against the pre-integration experiment. The current PR integrates the refreshed `origin/main` renderer adapter and adds native LFO/modulation-route readback; those two renderer source files therefore differ from the execution-time hashes pinned by the plan. The historical plan and metrics remain unchanged and are evidence for the exact runtime hashes it records, not a claim that the integrated renderer produced fresh native results. The retained [seed-0 selected inference bundle](../outputs/state-coverage-validation-20261013/selected-model.pt) remains current; SHA-256 `5517db59dec68f0daca30789d08e75f41183e5577ba5a2082a4dc9488da15d4f`.

The training replay MAE is separate from validation patch MAE. Passing or missing the tiny-overfit gate is a learning-loop diagnostic only. Seed 1's expanded arm has lower unison accuracy than its 205-state arm (0.5238 vs 0.6190); seed 2's expanded arm misses one filter-on classification (0.9524 vs 1.0000). These specific categorical changes, plus the opposite signs in the primary paired deltas, reinforce that tiny-overfit success did not yield consistent held-out gains. No replication candidate is promoted; the seed-0 selected bundle remains the current candidate. The evaluator's automatically selected model is retained only as a comparison artifact.

## Reproduction

Use the pinned Python environment and run from the repository root in PowerShell. The code below sets the local import paths explicitly and issues the fixed 4K/8K/16K resumes for both populations and seeds. It creates a fresh timestamped suffix; every output path must be new and must not already exist. Do not point it at the recorded run directories. A final train exit code 2 is acceptable only when its summary says `complete: true` and records the tiny-overfit gate result.

```powershell
$python = 'C:\Users\angert\miniforge3\envs\py312\python.exe'
$env:PYTHONPATH = 'research/training;research/data_generation;research/data_generation/outputs/smoke-deps'
$suffix = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-repro'
$common = @('--mode','overfit','--steps','16000','--replay-mib','32','--batch-size','32','--learning-rate','0.001','--seed','0','--checkpoint-every','4000','--render-budget','480','--cpu-threads','4','--sampling-policy','state_round_robin','--updates-per-render','1','--timeout','30')
$arms = @(@{ name='one'; pool='research/training/outputs/authored-pilot-v2-20261005/pool.json'; source='research/training/outputs/cached-curve-one-step16000-20261011/checkpoint.pt'; capacity='205' }, @{ name='expanded'; pool='research/training/outputs/authored-expanded-20261013/pool.json'; source='research/training/outputs/state-coverage-expanded-16000-20261013/checkpoint.pt'; capacity='404' })
foreach ($seed in @(1,2)) { foreach ($arm in $arms) {
  $stem = "research/training/outputs/state-coverage-replication-$($arm.name)-seed$seed"
  $out4 = "$stem-4000-$suffix"; $out8 = "$stem-8000-$suffix"; $out16 = "$stem-16000-$suffix"
  foreach ($path in @($out4,$out8,$out16)) { if (Test-Path $path) { throw "Output already exists: $path" } }
  $base = @('--pool',$arm.pool,'--replay-capacity',$arm.capacity,'--learning-seed-override',"$seed") + $common
  & $python research/training/run.py train @base '--output' $out4 '--seed-replay' $arm.source '--stop-after' '4000'; if ($LASTEXITCODE -ne 0) { throw '4K segment failed' }
  & $python research/training/run.py train @base '--output' $out8 '--resume' "$out4/checkpoint.pt" '--stop-after' '8000'; if ($LASTEXITCODE -ne 0) { throw '8K segment failed' }
  & $python research/training/run.py train @base '--output' $out16 '--resume' "$out8/checkpoint.pt"; if ($LASTEXITCODE -notin @(0,2)) { throw '16K segment failed' }
  $status = Get-Content "$out16/status.json" | ConvertFrom-Json; if (-not $status.complete -or $status.step -ne 16000) { throw '16K checkpoint incomplete' }
} }
```

The frozen native comparison uses candidate order from the plan and the same 21 cases. Use `--resume` only with the same arguments and output directory after an interruption; the evaluator verifies committed case hashes and will reuse them.

```powershell
$checks = @(
  'incumbent=research/training/outputs/two-view-train-20261010/checkpoint.pt',
  'one_seed0_16000=research/training/outputs/cached-curve-one-step16000-20261011/checkpoint.pt',
  'expanded_seed0_16000=research/training/outputs/state-coverage-expanded-16000-20261013/checkpoint.pt',
  "one_seed1_16000=research/training/outputs/state-coverage-replication-one-seed1-16000-$suffix/checkpoint.pt",
  "expanded_seed1_16000=research/training/outputs/state-coverage-replication-expanded-seed1-16000-$suffix/checkpoint.pt",
  "one_seed2_16000=research/training/outputs/state-coverage-replication-one-seed2-16000-$suffix/checkpoint.pt",
  "expanded_seed2_16000=research/training/outputs/state-coverage-replication-expanded-seed2-16000-$suffix/checkpoint.pt"
)
$bindings = @('expanded_seed0_16000=research/training/outputs/authored-expanded-20261013/pool.json','expanded_seed1_16000=research/training/outputs/authored-expanded-20261013/pool.json','expanded_seed2_16000=research/training/outputs/authored-expanded-20261013/pool.json')
$evalOut = "research/training/outputs/state-coverage-replication-validation-$suffix"
if (Test-Path $evalOut) { throw "Output already exists: $evalOut" }
& $python research/training/run.py compare --pool research/training/outputs/authored-pilot-v2-20261005/pool.json --checkpoints $checks --checkpoint-pools $bindings --retrieval-checkpoint research/training/outputs/full-pilot-step4000-20261009/checkpoint.pt --output $evalOut --count 21 --timeout 30
```

The complete numerical record is [state_coverage_replication_summary.json](data/state_coverage_replication_summary.json), including checkpoint identities, per-seed paired state bootstrap, parameter and RMS diagnostics, and QA counts.


A bounded next research question is whether expanded training yields a repeatable primary-score gain on a newly frozen validation set across matched learning seeds, while leaving the 64 test states untouched.