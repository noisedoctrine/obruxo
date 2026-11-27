# Windows startup

Work from the repo root in PowerShell. Pick the installed Python for the task and invoke it with `&`. Bare `python` resolves to Miniforge base on this machine.

```powershell
$obruxoFrozen = 'research/modelling/frozen_encoder_retrieval'
$obruxoConda = Join-Path $env:USERPROFILE 'miniforge3/envs'
$obruxoBench = "$obruxoFrozen/local_checkpoints/bench-env/Scripts/python.exe"
$obruxoData = Join-Path $obruxoConda 'obruxo-data-generation/python.exe'
$obruxoTorch = Join-Path $obruxoConda 'py312/python.exe'
$obruxoNativeSite = Join-Path $obruxoConda 'obruxo-data-generation/Lib/site-packages'
```

| Task | Installed environment |
|---|---|
| Frozen cache analysis, CPU probes, frozen retrieval tests, or six of the seven encoder loaders | Project venv **`bench-env`** (`$obruxoBench`), built on Conda **`py312`**. Python 3.12.13, NumPy 1.26.4, SciPy 1.17.1, PyTorch 2.12.1+xpu. |
| Basic Pitch frozen loader | **`bench-env`** with `obruxo-data-generation` site-packages (`$obruxoNativeSite`) prepended to `PYTHONPATH` for that worker. Its lock calls for NumPy 2.4.6 and SciPy 1.18.0. The efficiency runner accepts `--pythonpath "basic_pitch_trunk=$obruxoNativeSite"`. |
| Preset/MIDI contracts and Vital rendering | Conda **`obruxo-data-generation`** (`$obruxoData`). It has the pinned [native packages](research/data_generation/environment.yml), including DawDreamer 0.8.3. |
| PyTorch and the native trainer | Conda **`py312`** (`$obruxoTorch`) contains PyTorch. For tasks that also need the native packages, prepend `$obruxoNativeSite` for that invocation and check the [trainer's environment and renderer profile](research/training/TRAINER.md#environment-and-commands). |

- Check `Test-Path` and verify the actual imports with a short command such as `& $obruxoBench -B -c 'import sys,numpy,scipy; print(sys.executable,numpy.__version__,scipy.__version__)'`. For encoder loading, check the per-model `runtime_versions` in `$obruxoFrozen/local_checkpoints/checkpoint-lock.json` before changing packages. Scope `PYTHONPATH` to the command that needs it and use `;` between entries on Windows.
- Start with the accepted feature manifests and WAVs in `research/modelling/frozen_encoder_retrieval/outputs/issue152-validation-train-midi-v3-50` for frozen comparisons. Check the selected inputs against their receipts and reuse existing results where applicable. A fresh worktree lacks the ignored environment, checkpoints and outputs. Point the [runners](research/modelling/frozen_encoder_retrieval/README.md) at the primary checkout's existing assets. For tasks that require actual rendering, use the run's pinned renderer profile and verify the local Vital VST3 path, normally `C:\Program Files\Common Files\VST3\Vital.vst3`.

| Startup failure | Recovery |
|---|---|
| `ModuleNotFoundError` or a package version mismatch | Check the interpreter path, import path and task lock first. The frozen CLI adds its sibling source directories automatically. Ad hoc scripts may need `research/data_generation`, `research/training`, or `research/modelling/basic_pitch` on `PYTHONPATH`. |
| `WinError 5` loading `c10_xpu.dll` | The Windows sandbox can block importing this PyTorch build even for CPU work. Retry the exact authorized Python command with escalation for native DLL access. Check `torch.xpu.is_available()` only for XPU runs. |
| `gh` cannot reach GitHub or read credentials, or Git cannot create `.git/index.lock` with access denied | Retry the specific authorized command with escalation for network, the Windows keyring, or writes to Git metadata. A sandboxed `gh auth status` failure is insufficient to diagnose expired credentials. |

## Pretrained audio on Intel XPU

- Use `py312` with pinned source checkouts and any extra dependencies in a scratch `pip --target --no-deps` directory. For Stable Audio, reuse the [patches and smoke probe](research/training/reports/issue187_pretrained_vital_surrogates/XPU_SETUP.md). Load only the codec for reconstruction.
- Pass `device="xpu"` explicitly. Some upstream loaders check only CUDA and disable FP16 on Intel GPUs. Cast during transfer to avoid a full FP32 copy on the GPU. Use the existing PyTorch SDPA fallback if compiled FlexAttention needs an unavailable compiler. Verify tensor placement, dtype, and decoder input gradients when needed.
- For gated Hugging Face models, verify that the weight file downloads. If Windows blocks cache symlinks, pass `local_dir` to `snapshot_download`. Reuse auxiliary weights and tokenizers when their content hashes match.

## Long-lived task progress

- For work that will run for several minutes or longer, do not keep the conversation busy with repeated model polling. Run the worker independently with durable logs and a small machine-readable status file, then open a lightweight Python `tkinter` monitor for the user.
- Keep the monitor focused on the current phase, per-stage completed/total counts, overall progress, failures when available, the last status update, and a few recent log events. Refresh from status/log files every few seconds; avoid heavy rescans.
- Include simple Play/Pause and Stop controls for the monitor, and label clearly whether they affect only monitoring or also the worker. Prefer safe, cooperative worker controls if pausing or stopping computation is requested.
- Once the monitor is visible and useful, stop polling continuously; let the user request progress updates when wanted.

# Research findings and experiment reporting

Every empirical training, model-comparison or diagnostic report, and its PR summary, must answer: **what did we learn, how strong is the evidence, and what should we do next?** Write for a technically competent reader who understands the project but has not participated in the experiment.

When a report exists, include a direct, clickable link to it near the top of the PR description. Link to the version on the PR's branch or commit, not a potentially outdated version on the default branch. If there are multiple reports, link each relevant report.

1. **Explain the research question first.** State what we want to learn, why it matters, and how the experiment answers it. Present the main measured finding and its practical implication early; if no empirical result exists, say so.
2. **Use plain, precise language.** Avoid unnecessary jargon, invented terminology, and overly clever phrasing. Define technical terms when first introduced. Explain what changes, what stays fixed, and what is measured, using a concrete example when it improves understanding.
3. **Give results context.** Put a compact cross-arm results table near the top, with meaningful matched baselines, metric definitions, units and direction (higher/lower is better). Report sample sizes, denominators and independent input diversity: e.g., 1,548 queries from 129 presets × 12 distinct MIDI performances are not 1,548 independent performances.
4. **Separate observations from conclusions.** Distinguish numerical advantages from statistically supported findings, exploratory comparisons from independent confirmation, and proxy metrics from actual downstream quality. Explain what the uncertainty covers and what the experiment does and does not establish.
5. **Use tables and figures purposefully.** Avoid redundant per-arm tables, excessive precision, or visualizations that do not aid interpretation. Share tables when results or control scope can be compared directly. Keep the first screen scannable.
6. **Keep technical details accessible but unobtrusive.** Explain the methodology needed to understand the findings in the main report. Move detailed statistical procedures, exhaustive inventories, render QA, cache reuse, installation, implementation details and provenance to an execution/reproducibility section or appendix. Link machine-readable evidence. Report engineering metrics prominently only when they are research decision criteria; engineering success is not evidence of model quality.
7. **End with a clear, evidence-limited decision.** State missing conditions, limitations and untested capabilities. Do not equate the best tested model with the best possible model. Explain what the findings mean for the project and whether further experimentation is justified.

## Vital-control scope reporting

Every training or model-comparison report must make the changeable Vital-control scope clear, but it must not displace the scientific results.

- For experiments that train, predict or edit Vital controls, use one shared control-scope table when the compared versions have identical coverage. When coverage differs, show the differences in a compact comparison or separate tables if needed for clarity. Keep the primary research-results table separate and first.
- Group scope rows by meaningful control family or capability. Include fixed and disabled groups as named groups; do not hide them in a broad “other” row.
- Put the ✅/❌ status immediately after the family name, followed by the enabled/total instance count. Use ✅ when at least one instance is changeable and ❌ when none are; the count shows partial coverage (for example, `1 / 3`).
- Keep dense output counts separate from control counts. For example, one unison control represented by three category scores counts as one enabled control and three outputs.
- Include output representation/size and the update method or training signal where they clarify what the model does.
- Maintain a linked, machine-readable per-control inventory (CSV, JSONL, SQLite or equivalent) for actual control-changing/prediction experiments, one record per exact Vital field/instance including fixed and disabled controls. Record experiment/version, status, output count, update method, supervision/loss path, ranges/choices, fixed value/source, masks and relevant notes. Scope tables must reconcile with this inventory.
- For frozen *representation-only* benchmarks with **no predicted or edited controls**, state that once, give a brief shared frozen-control summary, and link any applicable canonical inventory. Do not repeat the same zero-output, 979-control accounting inside every encoder's research-results table.
- Distinguish shared-model training from direct patch optimization, source-label warm starts from audio-loss training, fixed template values from omitted fields, and trainable heads from frozen heads.
