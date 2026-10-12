# Quick research: issue #38

Implements the opt-in workflow in [issue #38](https://github.com/noisedoctrine/obruxo/issues/38): two audits and two controlled experiments. Nothing runs on import or `--help`. Research outputs remain ignored locally unless they are reviewed and promoted explicitly; the tiny smoke suite still uses only a fake renderer and mocked source/process interfaces.

The reviewed results from the first bounded run are in [`reports/RESULTS.md`](reports/RESULTS.md). Only sanitized aggregates and plots are promoted; raw presets, paths, audio, caches, and source checkouts remain local.

| Task | Command | Actual work when explicitly invoked |
|---|---|---|
| A: stored-state identity | `identity` | Reads corpus JSON, groups conservative identities, simulates file-level split crossings; no audio |
| Fixture preparation | `prepare-fixtures` | Selects eight distinct compatible groups using explicit metadata; no native rendering |
| B: measurement variability | `measurement` | At most 56 independent Vital renders and diagnostic comparisons |
| C: controlled Basic Pitch | `encoder` | Reuses B baselines, adds 16 renders, extracts frozen posteriors and stock events |
| D: modulation source audit | `source-audit`, then `source-review` | Collects source context; a reviewer supplies semantic findings; citation ranges are verified |

The experiment sequence is **A → prepare fixtures → B → C**. D is independent. There is deliberately no `run-all` command.

## 1. Setup on this Windows checkout

Open PowerShell in the OBRUXO repository root. These commands set variables and check dependencies; they do not start any research task.

```powershell
$quickPython = Join-Path $env:USERPROFILE 'miniforge3\envs\py312\python.exe'
$quickRunner = 'research/experiments/quick_research/run.py'
$quickOutput = 'research/experiments/quick_research/outputs/issue38-v1'

# If using another installation, activate the research environment and instead use:
# $quickPython = (Get-Command python).Source

& $quickPython $quickRunner --help
& $quickPython -c "import importlib.util; print({n: importlib.util.find_spec(n) is not None for n in ['numpy','scipy','pytest','yaml','torch','mir_eval','matplotlib','vita','dawdreamer']})"
```

The existing `py312` environment has the numerical, plotting, and Basic Pitch dependencies. At implementation time **Vita and DawDreamer were not directly importable there**. A reviewed dependency directory may be available from an earlier setup; the API probe below is authoritative, because a directory skeleton is not an installed runtime. If the probe fails, follow the pinned setup before scheduling B/C:

```powershell
$quickNativeDeps = Join-Path $env:LOCALAPPDATA 'Temp\obruxo-issue20-deps'
if (!(Test-Path (Join-Path $quickNativeDeps 'vita')) -or !(Test-Path (Join-Path $quickNativeDeps 'dawdreamer'))) {
    throw 'Reviewed native dependencies are missing; follow research/data_generation/README.md setup first.'
}
$env:PYTHONPATH = (@($quickNativeDeps, $env:PYTHONPATH) | Where-Object { $_ }) -join [IO.Path]::PathSeparator
& $quickPython -c "import vita, dawdreamer; assert hasattr(vita, 'Synth'); assert hasattr(dawdreamer, 'RenderEngine'); import importlib.metadata as m; print({n:m.version(n) for n in ['vita','dawdreamer']})"
if ($LASTEXITCODE -ne 0) {
    throw 'The native dependency directories exist but do not contain importable Vita/DawDreamer runtimes; a directory skeleton is not sufficient.'
}
```

This temporary location is a local convenience, not a portable dependency installation or a bundled plugin. The authoritative setup/version pins are in [data generation](../../data_generation/README.md) and [Basic Pitch](../../modelling/basic_pitch/README.md). B/C require the reviewed installed Vital VST3, DawDreamer 0.8.3, the pinned Vita/schema runtime, and the local Basic Pitch checkpoint. The native plugin's reviewed hash is enforced by the existing renderer. Do not install another unreviewed plugin build to bypass a mismatch.

A and D use the standard library plus Git for D. Fixture preparation needs the existing static authoring dependencies. B/C check for missing native dependencies before opening a plugin. No command installs packages, downloads model weights, fetches source, or logs into a service.

## 2. Run A: identity audit

**This command scans the corpus. Run it only when ready.**

```powershell
& $quickPython $quickRunner identity `
    --corpus 'datasets/presetshare/raw/presetshare_files/data' `
    --output "$quickOutput/identity" `
    --seed 0 --budget-seconds 1800
```

Read `identity/DATASET_IDENTITY_REPORT.md` and `identity/status.json`. Continue only after state `complete`.

Outputs include `identity_summary.json`, `identity_manifest.json`, private `members.jsonl`, and `exclusions.csv`. The three identity levels are raw bytes, strictly parsed/key-ordered JSON, and that representation without top-level `author`, `comments`, and `preset_name`. Decimal values are parsed losslessly; some equivalent numeric spellings intentionally remain distinct. Nested state, array order, synth version, and unknown fields are retained. Duplicate keys, nonfinite values, malformed roots, and symlinks are excluded explicitly. No presets are rewritten.

The file-split simulation hashes `seed:corpus-relative-path` into an expected 80/10/10 partition. It does not describe an existing training split, guarantee exact quota sizes, or infer acoustic equivalence. The summary includes bounded introduced-group examples with changed field pointers, not source payloads.

## 3. Freeze fixtures before measuring outcomes

```powershell
& $quickPython $quickRunner prepare-fixtures `
    --identity "$quickOutput/identity/identity_manifest.json" `
    --metadata 'datasets/presetshare/raw/presetshare_vital_metadata.csv' `
    --output "$quickOutput/fixtures" `
    --budget-seconds 1800
```

Selection requires two distinct identity groups for each exact metadata `type`: `Bass`, `Pad`, `Pluck`, `Lead`. Relative `preset_file` paths are resolved against the metadata CSV's directory, matching the corpus analysis convention. Candidates are ordered by group ID; only static compatibility is considered before freezing.

Inspect `fixtures/FIXTURE_PREPARATION.md`, `fixtures/static_exclusions.csv`, and `fixtures/fixtures.json`. State must be `complete` with eight fixtures. If it is `insufficient_fixtures`, **stop**: do not silently relax categories, remove unsupported fields, or proceed on a smaller population. Investigate compatibility or revise the study explicitly.

The current pinned schema is narrower than many newer presets. Unknown/missing controls are exclusions, not permission to strip or fill state. Static compatibility does not establish runtime compatibility. Later native/QA failures remain explicit outcomes; this implementation does not replace fixtures after seeing those outcomes.

## 4. Run B: parameter signal versus variability

**This command performs native rendering.**

```powershell
& $quickPython $quickRunner measurement `
    --fixtures "$quickOutput/fixtures/fixtures.json" `
    --output "$quickOutput/measurement" `
    --plugin 'C:\Program Files\Common Files\VST3\Vital.vst3' `
    --budget-seconds 1800
```

`--renderer-config` defaults to the existing `research/data_generation/configs/renderer.yaml`. The CLI preserves its plugin-hash checks, one-worker contract, and QA settings. Source hashes are rechecked before rendering.

The common performance is held C4, velocity 100, 120 BPM, two seconds held plus two seconds tail, stereo 44.1-kHz float32. Each fixture gets three baseline invocations and two independent invocations for each available perturbation: first enabled oscillator level and first enabled filter cutoff, each changed by 5% of its legal raw range. The direction is positive unless that exceeds the maximum. Absent filters are documented by the planned render count; no component is enabled for the probe. Phase/randomness settings are retained.

Read `measurement/MEASUREMENT_SIGNAL_REPORT.md`, then `measurement_ranges.csv`, `comparisons.csv`, `measurement_summary.json`, and `render_manifest.json`. Waveform RMSE, multiresolution STFT distance, and RMS-envelope MAE operate on unnormalized, identically scheduled audio. The exact numerical contract is in `quickresearch/metrics.py` and the saved summary.

Every repeat is a fresh `renderer.render()` call with a distinct repeat index and output path. Repeats share the same request ID but are not deduplicated. Resuming reuses an already completed repeat only after checking identity, file hash, decoded audio hash, shape, and sample rate. A corrupted successful cache entry is rerendered; an already recorded native error remains a failed outcome for that run.

The report compares changed audio against baseline repeat 0 and all three unchanged pairwise distances. Shared-reference dependence is explicit. Missing/invalid/clipped/silent data cannot establish observed separation. Overlap does not prove that a parameter is unidentifiable, and separation does not establish perceptual importance.

## 5. Run C: controlled frozen Basic Pitch analysis

Run after B completed its planned attempts. B may have `complete_with_failures`; those failures remain unavailable cases in C. An incomplete/latest-interrupted B invocation cannot supply a stale completed bank.

```powershell
& $quickPython $quickRunner encoder `
    --measurement "$quickOutput/measurement" `
    --output "$quickOutput/encoder" `
    --plugin 'C:\Program Files\Common Files\VST3\Vital.vst3' `
    --device xpu --budget-seconds 1800
```

The default checkpoint is `research/modelling/basic_pitch/artifacts/basic_pitch_icassp_2022.pt`; override with `--checkpoint` only for the corresponding compatible pinned model artifact. `--device xpu` fails if unavailable. `--device auto` can use CPU when XPU is unavailable and records that choice; forward failures never silently switch backends.

C reuses baseline repeat 0 from B and adds held C3 and four staccato C4 notes (starts 0/480/960/1440 ticks, duration 240 ticks, velocity 100, end 1920 ticks, two-second tail). MIDI is evaluator-only. The native Basic Pitch model is frozen, in inference/evaluation mode, and FP32. Existing frontend preparation, overlap unwrapping, stock event decoding, reference labels, and `mir_eval`-based scoring are reused.

Inspect `encoder/CONTROLLED_BASIC_PITCH_REPORT.md`, `case_results.json`, `posterior_diagnostics.csv`, and `encoder_summary.json`. Per-fixture figures overlay known MIDI and decoded events on note/onset posteriors. Diagnostic means use MIDI-held intervals; the acoustic release remains visible after note-off. Full note/onset/contour arrays, MIDI, and decoded events are retained locally.

Feature cache identity includes decoded audio hash, checkpoint hash, implementation hash, backend/precision, Torch version, decoder settings, and preparation contract. Cached array bytes and shapes are checked. These are fixture-level observations, not category rankings or evidence that a future conditioned inverse model will perform better.

## 6. Run D independently: source evidence, then explicit review

D operates on a local, immutable checkout. The audit command itself never clones, fetches, checks out, or edits source. If the pinned Vital source is not already present, create the checkout explicitly from GitHub first:

```powershell
$quickVitalSource = Join-Path $quickOutput 'vendor/vital-source'
New-Item -ItemType Directory -Force (Split-Path -Parent $quickVitalSource) | Out-Null
gh repo clone mtytel/vital $quickVitalSource -- --filter=blob:none --no-checkout
git -C $quickVitalSource checkout --detach '636ca0ef517a4db087a6a08a6a8a5e704e21f836'
```

Then collect the evidence packet. If using a separately maintained checkout, replace `$quickVitalSource` with its path; do not point it at OBRUXO itself.

```powershell
& $quickPython $quickRunner source-audit `
    --source $quickVitalSource `
    --revision '636ca0ef517a4db087a6a08a6a8a5e704e21f836' `
    --output "$quickOutput/source" `
    --max-hits 20 --budget-seconds 1800
```

This creates `SOURCE_AUDIT_WORKSHEET.md`, `source_packet.json`, `evidence.json`, and `review_answers.json`. Successful state is **`evidence_collected_review_required`**, not "audit proved". Each search has a bounded number of contexts; the packet is not exhaustive. The source checkout path is private; GitHub permalinks omit remote credentials.

Next, an engineer/agent must read the code and follow the actual call/data flow for the five questions in `review_answers.json`: loading/index mapping, graph order and accumulation, per-slot dynamic state, dependencies/cycles and route-amount modulation, and numerical versus semantic invariance. Supply an explanation, a status (`unresolved`, `supported`, `exception`, or `version_limited`), and citations using repository-relative paths with verified one-based `start`/`end` lines. A citation object has the shape:

```json
{"path": "src/REPLACE_WITH_VERIFIED_FILE.cpp", "start": 1, "end": 1}
```

Those are placeholders, not source evidence. Leave a question unresolved when source does not establish it. Record source/build coverage in `version_notes`; `build_match` stays `unverified` unless explicit evidence supports `verified` or `different`. The pinned source must not automatically be equated with installed Vital 1.6.4 or its newer ramp behavior.

When the worksheet contains the actual review:

```powershell
& $quickPython $quickRunner source-review `
    --packet "$quickOutput/source/source_packet.json" `
    --answers "$quickOutput/source/review_answers.json" `
    --output "$quickOutput/source-review" `
    --budget-seconds 1800
```

This checks commit consistency and citation existence/ranges, then writes `MODULATION_SLOT_SEMANTICS_AUDIT.md` and `review_provenance.json`. It does not mechanically prove the reviewer-authored interpretations. `review_partial` preserves unanswered questions. No permutation-render pipeline exists here; empirical follow-up requires a specific remaining uncertainty, separately scoped.

## Budgets, statuses, resume, and failures

- `--budget-seconds` accepts 1–1800. Each invocation has its own budget; resumed work records another invocation rather than pretending all work fit the first budget.
- A supervisor launches one hidden worker with native thread counts fixed to one. Its execution deadline includes worker startup, rehearsal, renders, inference, analysis, and report generation. Termination/report cleanup can add a small overhead after the deadline; no further scientific work is allowed then.
- The worker reserves report time and makes conservative projections from rehearsal costs before starting the rest of an empirical matrix. If it will not fit, state is `projected_budget_exceeded`; there is no silent reduced-population claim.
- The supervisor terminates its worker process tree on timeout or interruption. Native rendering/inference happen only inside that worker. Partial caches survive.
- Source audits are budgeted too, but source search completion is not semantic review completion.
- Treat **`status.json` as authoritative**, especially after a timeout. `INCOMPLETE.md` warns that older reports may remain. Downstream stages reject stale completed manifests when the upstream's latest invocation is incomplete.

Read status without executing anything:

```powershell
& $quickPython $quickRunner status --output "$quickOutput/measurement"
Get-Content "$quickOutput/measurement/worker.log" -Tail 30
```

To resume, repeat the original command with **the same scientific arguments plus `--resume`**. The budget may change, up to the 1800-second cap, because every invocation records its own execution allowance. Do not change code, other inputs, checkpoint, or renderer configuration under an existing run. These are fingerprinted; changes require a new output directory. A saved render error is not automatically retried; use a new run after fixing its cause so the prior failure remains visible. Never edit a frozen fixture manifest to bypass a mismatch.

Exit code `0` means the command produced its declared outcome, which can include `complete_with_failures`, `review_partial`, or `evidence_collected_review_required`; inspect the state and coverage. Exit code `2` means budget/coverage/fixture insufficiency. Exit code `1` means a configuration or execution failure. A `run.lock` prevents concurrent writers. Only remove a stale lock after confirming the recorded process and worker are no longer running; no command removes another process's lock automatically.

## Tiny smoke validation only

The smoke file uses four tiny synthetic JSON documents, short analytic NumPy waveforms, a fake renderer, and mocked Git/process execution. It does not read the real corpus, load a native plugin, load weights, perform model inference, or collect actual source-audit evidence.

```powershell
# Use a fresh directory: pytest may clear an existing --basetemp directory.
$quickSmoke = Join-Path (Get-Location) ('tmp/quick-research-smoke-' + [guid]::NewGuid().ToString('N'))
& $quickPython -m pytest research/experiments/quick_research/tests/test_tiny_smokes.py `
    -q -p no:cacheprovider --basetemp $quickSmoke
```

Outputs from smokes and research are ignored. Do not publish raw source paths, presets, embedded assets, generated audio, or the private group-membership inventory. Sanitized reports/figures can be reviewed and promoted later; this runner never commits or publishes them.
