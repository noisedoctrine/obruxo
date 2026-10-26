# Local basic snapshots run outside the source worktree

The `bundle-basic` command packages a validated basic model, embedded frontend weights, codec/default assets, project runtime source and static schema. The snapshot launcher verifies their hashes and recorded runtime versions before loading, rejects external project imports and preserves existing output files. The model interface stays at 14 dense values / 11 codec selections.

**22 local tests pass**, covering all three supported basic model formats in isolated subprocesses, exact command output, changed model/code/schema files, dependency mismatch, existing destinations and path boundaries. The original pilot and both correction formats retain their existing behavior.

The [real-model check](data/local_basic_bundle_summary.json) creates two snapshots and runs six cases in a newly created venv from an empty working directory outside the source worktree. All project-module origins point inside their snapshot, and all six predicted presets match the exact hashes previously retained by native audits. The environment uses `python -I` and [shares the installed dependency packages](data/local_basic_bundle_environment.json). This is source isolation, **not a clean dependency installation**.

The check takes **28.76 seconds / 0.90 GiB**, retains 10.94 MiB and adds **zero native renders or optimizer updates**. Implementation `1cb2562` precedes execution. All 42 source files pinned by the running sound fit remain unchanged.

The snapshots keep release qualification false and preserve the source models' status. The available project/Basic Pitch readmes are copied into the notice inventory; complete release notices and a clean dependency-install check remain open, alongside reconstruction admission, fresh confirmation and listening. No model is promoted.

[Commands and runtime requirements](../LOCAL_BUNDLE.md). The local sound fit continues under its frozen plan.
