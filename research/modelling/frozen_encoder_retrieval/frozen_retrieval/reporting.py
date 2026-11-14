from __future__ import annotations

from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys

from .core import MODEL_REGISTRY
from .encoders import available_candidates, load_checkpoint_lock


GROUPS = (
    "Global settings", "Oscillators and voices", "Filters", "Envelopes", "LFOs and modulation",
    "Effects", "Wavetables, curves and samples",
)
REPO_ROOT = Path(__file__).resolve().parents[4]


def _control_group(row: dict) -> str:
    path = (row.get("data_path") or "").lower()
    key = (row.get("control_key") or "").lower()
    family = (row.get("family") or "").lower()
    text = " ".join((path, key, family))
    if path.startswith("settings.") and not any(token in text for token in ("filter", "envelope", "lfo", "osc", "unison", "reverb", "chorus", "delay", "phaser", "distortion")):
        return "Global settings"
    if any(token in text for token in ("oscillator", "osc_", "unison", "voice", "wavetable")):
        return "Oscillators and voices" if "wavetable" not in text else "Wavetables, curves and samples"
    if "filter" in text:
        return "Filters"
    if any(token in text for token in ("envelope", "env_")):
        return "Envelopes"
    if any(token in text for token in ("lfo", "modulation", "mod_")):
        return "LFOs and modulation"
    if any(token in text for token in ("reverb", "chorus", "delay", "phaser", "distortion", "compressor", "effect")):
        return "Effects"
    if any(token in text for token in ("sample", "curve", "asset", "wavetable")):
        return "Wavetables, curves and samples"
    raise ValueError(f"unclassified Vital control: {row.get('data_path')}/{row.get('control_key')}")


def build_control_inventory(source: Path, destination: Path) -> dict:
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["source_experiment_id"] = row["experiment_id"]
        row["experiment_id"] = "frozen_encoder_retrieval_issue152"
        row["changeable"] = False
        row["head_outputs"] = 0
        row["output_type"] = "fixed_input_only"
        row["update_method"] = "unchanged original Vital preset input; no model update"
        row["training_signal"] = "none; frozen representation retrieval only"
        row["gradient_path"] = "none"
        row["fixed_value"] = None
        row["fixed_value_source"] = ("source preset field passes through unchanged per original asset" if row.get("present_in_template")
                                      else "source preset field state passes through unchanged; no template value inferred")
        row["fixed_reason"] = "this benchmark predicts no Vital controls"
        row["control_family_group"] = _control_group(row)
        row["output_count"] = 0
        row["supervision_path"] = "none"
        row["source_value_materialized"] = False
        rows.append(row)
    if not rows or len({(row["control_key"], row.get("instance"), row.get("data_path")) for row in rows}) != len(rows):
        raise ValueError("source Vital inventory is empty or contains duplicate control instances")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for row in sorted(rows, key=lambda item: (item["control_family_group"], item["data_path"], str(item.get("instance")))):
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
    totals = Counter(row["control_family_group"] for row in rows)
    return {"total_controls": len(rows), "enabled_controls": 0,
            "controls_by_family": {group: totals[group] for group in GROUPS if totals[group]},
            "inventory_path": str(destination)}


def build_controlled_inventory(source: Path, destination: Path, controlled_fields: dict) -> dict:
    base_rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    versions = {"original": None, **controlled_fields}
    rows = []
    for version, change in versions.items():
        field = None if change is None else change.get("field")
        matching = [row for row in base_rows if row.get("vital_key") == field] if field else []
        if field and len(matching) != 1:
            raise ValueError(f"controlled field {field!r} must match one exact inventory record; found {len(matching)}")
        for base in base_rows:
            row = dict(base)
            # The perturbation report is per-control, but its source is one real
            # validation preset. Do not publish a stable identifier for that asset.
            row.pop("data_sha256", None)
            changed = field is not None and row.get("vital_key") == field
            row.update({"experiment_id": "issue152_controlled_perturbation",
                "experiment_version": version,
                "preset_source_provenance": "one original validation asset; source identity omitted from published inventory",
                "changeable": changed,
                "control_status": "manually_perturbed" if changed else "fixed_original_input",
                "head_outputs": 0,
                "output_type": "manual_input_perturbation_no_prediction" if changed else "fixed_input_only",
                "update_method": ("one-control normalized-coordinate perturbation; no optimization" if changed
                                  else "unchanged value from the selected original Vital preset"),
                "training_signal": "none; controlled acoustic sensitivity diagnostic",
                "gradient_path": "none", "fixed_value": None,
                "fixed_value_source": None if changed else "selected original validation preset; identity omitted from published inventory",
                "fixed_reason": None if changed else "not varied in this experiment version",
                "output_count": 0,
                "supervision_path": "manual parameter change; no labels or loss" if changed else "none",
                "source_value_materialized": False,
                "controlled_before_raw": change.get("before_raw") if changed else None,
                "controlled_after_raw": change.get("after_raw") if changed else None,
                "controlled_before_normalized": change.get("before_normalized") if changed else None,
                "controlled_after_normalized": change.get("after_normalized") if changed else None,
                "notes": ("One exact Vital field was changed for this sensitivity arm; all other source fields were preserved."
                          if changed else "No optimizer, trainable head, supervision, or audio loss."),
                "control_family_group": _control_group(row)})
            rows.append(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for row in sorted(rows, key=lambda item: (item["experiment_version"], item["control_family_group"],
                                                   item["data_path"], str(item.get("instance")))):
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
    totals = Counter(row["control_family_group"] for row in base_rows)
    versions_summary = {}
    for version in versions:
        selected = [row for row in rows if row["experiment_version"] == version]
        enabled = Counter(row["control_family_group"] for row in selected if row["changeable"])
        versions_summary[version] = {"total_controls": len(selected), "enabled_controls": sum(enabled.values()),
                                     "controls_by_family": {group: {"enabled": enabled[group], "total": totals[group]}
                                                             for group in GROUPS}}
    return {"records": len(rows), "records_per_version": len(base_rows),
            "versions": versions_summary, "inventory_path": str(destination)}


def _inventory_summary(path: Path) -> tuple[int, dict[str, int]]:
    totals = Counter()
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["experiment_id"] != "frozen_encoder_retrieval_issue152" or row["changeable"] or row["output_count"] != 0:
                raise ValueError("control inventory does not match this frozen retrieval experiment")
            count += 1
            totals[row["control_family_group"]] += 1
    return count, {group: totals[group] for group in GROUPS if totals[group]}


def _metric(result: dict | None, key: str) -> str:
    if not result or result.get(key) is None:
        return "not run"
    return f"{result[key]:.4f}"


def _interval(result: dict | None, key: str) -> str:
    interval = result.get("confidence_intervals", {}).get(key) if result else None
    return f" [{interval['lower_95']:.4f}, {interval['upper_95']:.4f}]" if interval else ""


def _local_input_status() -> dict:
    snapshot = REPO_ROOT / "research/training/outputs/full-curve-corpus-qualified-20261018/snapshot"
    source_root = REPO_ROOT / "datasets/presetshare/raw/presetshare_files/data"
    summary_path, status_path = snapshot / "summary.json", snapshot / "status.json"
    result = {"snapshot_complete": False, "source_root_present": source_root.is_dir(),
              "snapshot_relative_path": snapshot.relative_to(REPO_ROOT).as_posix()}
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("complete") is True and status.get("stage") == "complete":
            result["snapshot_complete"] = True
            result["snapshot_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    runtime_modules = ("yaml", "mido", "numpy", "scipy", "dawdreamer", "torch", "transformers")
    def module_available(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            return False

    result["missing_pipeline_modules"] = [name for name in runtime_modules if not module_available(name)]
    return result


def generate_report(destination: Path, inventory: Path, *, dataset_dir: Path | None = None,
                    checkpoint_root: Path | None = None, lock_path: Path | None = None,
                    execution_note: str | None = None, controlled_summary_path: Path | None = None) -> dict:
    inventory_count, groups = _inventory_summary(inventory)
    controlled = json.loads(controlled_summary_path.read_text(encoding="utf-8")) if controlled_summary_path else None
    controlled_inventory_path = destination.with_name("vital_control_inventory_controlled.jsonl") if controlled else None
    controlled_inventory = (build_controlled_inventory(inventory, controlled_inventory_path,
        controlled["controlled_fields"]) if controlled else None)
    arms = {}
    evaluation = None
    local_dataset = None
    if dataset_dir and dataset_dir.is_dir():
        local_dataset = dataset_dir
        evaluation_path = dataset_dir / "evaluation.private.json"
        if evaluation_path.is_file():
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        for model_dir in (dataset_dir / "features").glob("*") if (dataset_dir / "features").is_dir() else []:
            run_path = model_dir / "extraction.latest.json"
            if run_path.is_file():
                run = json.loads(run_path.read_text(encoding="utf-8"))
                for variant in run.get("dimensions_by_variant", {}):
                    arms[f"{model_dir.name}:{variant}"] = run
    availability = {row["id"]: row for row in available_candidates(checkpoint_root, lock_path)}
    locked_models = {}
    if lock_path and lock_path.is_file():
        try:
            locked_models = load_checkpoint_lock(lock_path).get("models", {})
        except (OSError, ValueError, json.JSONDecodeError):
            locked_models = {}
    lines = ["# Frozen audio encoder retrieval benchmark", "",
             "Issue: [#152](https://github.com/noisedoctrine/obruxo/issues/152). This report covers frozen pretrained representations on original Vital source presets and existing MIDI performances. No synthesis controls are predicted, tuned, or optimized.", "",
             "## Execution status", ""]
    if local_dataset:
        dataset_summary_path = local_dataset / "dataset_summary.private.json"
        if dataset_summary_path.is_file():
            dataset = json.loads(dataset_summary_path.read_text(encoding="utf-8"))
            lines.extend([f"The frozen {dataset.get('source_split')} cohort contains {dataset.get('presets_verified', 0)} verified original presets, {dataset.get('unique_midi_excerpts', 0)} compatible unique MIDI excerpts, {dataset.get('episodes', 0)} retrieval episodes, and {dataset.get('render_requests', 0)} planned original-preset renders.",
                          f"Render profile: `{dataset.get('renderer_id')}`. Selected renderer capabilities: `{json.dumps(dataset.get('renderer_capabilities', {}), sort_keys=True)}`."])
            lines.append(f"Source pairing and MIDI compatibility audit: {json.dumps(dataset.get('exclusions', {}), sort_keys=True)}. These counts reflect the selected existing split and direct PresetShare preset/MIDI pairing rules; source WAV quality is not an eligibility condition.")
            lines.append(f"The {dataset.get('unique_midi_excerpts', 0)} compatible existing MIDI excerpts are cross-rendered on all {dataset.get('presets_verified', 0)} admitted source presets. These are new Vital renders, not claims of original PresetShare preset/MIDI capture pairs.")
            audit = _local_input_status()
            if audit["snapshot_complete"]:
                census = audit["snapshot_summary"]
                split_counts = census.get("eligible_states_by_split", {})
                lines.append(f"The existing source audit counted {census.get('discovered_files', 0):,} files and {census.get('static_eligible_states', 0):,} statically eligible states ({split_counts.get('validation', 0):,} validation, {split_counts.get('train', 0):,} train, {split_counts.get('test', 0):,} test). This experiment uses validation only; static eligibility is not native render eligibility.")
        render_path = local_dataset / "render_run.latest.json"
        if render_path.is_file():
            render = json.loads(render_path.read_text(encoding="utf-8"))
            lines.append(f"Vital render stage: {render.get('accepted', 0)} QA accepted of {render.get('requested', 0)} planned requests; this run rendered {render.get('runs_rendered_this_run', 0)} and reused {render.get('cache_reused_this_run', 0)} cached WAVs in {render.get('wall_seconds_this_run', 0)} seconds.")
            warnings = render.get("qa_warning_counts", {})
            lines.append(f"QA policy `{render.get('qa_policy_id', 'unknown')}` rejects malformed, silent, or failed renders and records clipping/tail thresholds as warnings. Accepted warning counts: {json.dumps(warnings, sort_keys=True)}; excluded renders: {json.dumps(render.get('excluded', {}), sort_keys=True)}.")
        if evaluation:
            lines.extend([f"The comparison uses {evaluation.get('common_episode_count_across_available_arms', 0)} identical episodes across the available arms; held-out reserved sources used: **no**."])
            lines.append(f"The gallery retained {evaluation.get('gallery_candidate_states_render_qualified', 0)} of {evaluation.get('gallery_candidate_states_original', 0)} states with complete renders across the frozen gallery-MIDI pool; {evaluation.get('gallery_candidate_states_excluded', 0)} incomplete states were removed once for all arms.")
            coverage = evaluation.get("coverage_by_condition", {})
            lines.extend(["", "### Render and feature coverage", "",
                          "| Condition | Planned episodes | Render QA qualified | Excluded | Feature qualified by arm | Common across arms |",
                          "|---|---:|---:|---:|---|---:|"])
            for condition, record in sorted(coverage.items()):
                feature_counts = ", ".join(f"{arm}: {count}" for arm, count in sorted(record.get("feature_qualified_episodes_by_arm", {}).items())) or "none"
                lines.append(f"| {condition} | {record.get('planned_episodes', 0)} | {record.get('qa_qualified_episodes', 0)} | {record.get('excluded_episodes', 0)} | {feature_counts} | {record.get('common_feature_qualified_episodes_across_arms', 0)} |")
            first_primary = next((row.get("cross_midi_common_gallery", {}) for row in evaluation.get("model_results", {}).values()), {})
            if first_primary and first_primary.get("known_near_neighbor_negative_win_episodes", 0) == 0:
                lines.append("No known multi-state lineage/alias neighbors were present in the render-qualified gallery, so that diagnostic has zero negatives. The corpus snapshot also lacks instrument-category labels; the spectral hard-negative result is only a coarse audio-similarity proxy, not a same-category or human-verified perceptual-neighbor result.")
        if not render_path.is_file():
            lines.append("Vital pre-render stage: **not run**.")
        if not arms:
            lines.append("Encoder feature extraction: **not run** for this dataset.")
        if not evaluation:
            lines.append("Retrieval scores: **not run** for this dataset.")
            if not execution_note:
                lines.append("Empirical execution remains unfinished: the full real-corpus render matrix, encoder feature caches, and retrieval comparison are incomplete; no quantitative results are claimed.")
    else:
        lines.extend(["No prepared real dataset was supplied to this report command. No raw corpus MIDI or Vital preset was parsed, a benchmark data manifest was not prepared, and Vital renders, encoder extraction, and retrieval evaluation were not executed."])
        if not execution_note:
            lines.append("Empirical execution remains unfinished: the full real-corpus render matrix, encoder feature caches, and retrieval comparison are incomplete; no quantitative results are claimed.")
        local_inputs = _local_input_status()
        if local_inputs["snapshot_complete"] and local_inputs["source_root_present"]:
            audit = local_inputs["snapshot_summary"]
            counts = audit.get("eligible_states_by_split", {})
            lines.append(f"Available local inputs: complete audited snapshot at `{local_inputs['snapshot_relative_path']}` and its PresetShare source directory. The snapshot census records {audit.get('discovered_files', 0)} discovered files and {counts.get('validation', 0)} statically eligible validation states ({counts.get('train', 0)} train, {counts.get('test', 0)} reserved). These are pre-existing audit counts; renderer pairing and MIDI compatibility were not evaluated here.")
        if local_inputs["missing_pipeline_modules"]:
            lines.append(f"Local preparation/render runtime is missing: {', '.join(f'`{name}`' for name in local_inputs['missing_pipeline_modules'])}.")
    if execution_note:
        lines.append(execution_note)
    if controlled:
        lines.append(f"Controlled perturbation diagnostic: {controlled.get('render_requests', 0)} real Vital renders, {controlled.get('accepted_renders', 0)} QA accepted, {len(controlled.get('models', {}))} pretrained encoders; uncertainty is not estimated for this small diagnostic set.")
    else:
        lines.append("Controlled patch-perturbation diagnostic: not run; no controlled score is claimed.")
    missing_runtime = sorted({package for candidate in availability.values() for package, value in candidate.get("dependencies", {}).items()
                              if value.get("version") == "missing"})
    lines.extend(["", f"Runtime preflight: Python `{sys.version.split()[0]}`; missing encoder runtime packages: {', '.join(f'`{name}`' for name in missing_runtime) or 'none'}. Checkpoint root supplied: **{'yes' if checkpoint_root else 'no'}**. Vital plugin environment variable configured: **{'yes' if os.environ.get('OBRUXO_VITAL_PLUGIN') else 'no'}**.",
                  "The existing [corpus audit report](../../../training/reports/CORPUS_INPUT_RESULTS.md) records prior static eligibility findings. Those audit counts are not native render eligibility. The separate [PresetShare audit](../../basic_pitch/README.md) has its own pairing rules and counts.", "",
                  "The default command selects the corpus-qualified full-preset expression renderer identity, including the established curve-input, ignored-empty-map, and native scalar readback policies. It supports note/polyphony, pitch-bend, channel-pressure, CC 1/64/74, release-velocity and multiple channels; tempo changes remain unsupported and are counted as exclusions. Unsupported events are rejected rather than silently removed. Musical spans are complete-note excerpts of 2–8 seconds plus a fixed 2-second release tail; model input is mono and 4–10 seconds, downmixed by channel mean and polyphase-resampled. No shared gain normalization is applied; EnCodec retains its own checkpoint-configured RMS normalization when enabled.", "",
                  "Render and feature stages are independent, resumable commands. Vital remains a single-worker in-process renderer; an asynchronous manifest reader feeds a bounded queue of request batches. Render keys include exact preset/MIDI/excerpt/renderer/policy identities. Feature keys include audio hash, checkpoint digest/revision, loader revision, sample rate, preprocessing, extraction tap, pooling, and FP32 precision. Weights are never downloaded by the run commands. Full paths, identifiers, WAVs, caches, checkpoints, and per-episode results belong in ignored local outputs.", "",
                  "No reconstruction head, source-label warm start, shared-model training, direct patch optimization, or audio-loss training is part of this experiment. The recommendation below identifies candidates for the later small 14-output follow-up; retrieval results do not establish reconstruction quality.", "",
                  "## Frozen-control scope", "",
                  f"The linked [{inventory.name}](./{inventory.name}) contains one record for every exact control/instance from the existing 979-record Vital control inventory. Main retrieval arms pass original source preset inputs unchanged; every control group is fixed with 0 enabled instances and 0 output values. No controls are omitted from the inventory.", ""])

    model_results = (evaluation or {}).get("model_results", {})
    for model_name, spec in MODEL_REGISTRY.items():
        variants = ("official_clip",) if spec["kind"] == "clip" else ("masked_mean", "temporal_pyramid_8")
        for variant in variants:
            arm = f"{model_name}:{variant}"
            run = arms.get(arm)
            outcomes = model_results.get(arm, {})
            primary = outcomes.get("cross_midi_common_gallery")
            mixed = outcomes.get("cross_midi_mixed_gallery")
            display = f"{spec['name']} — {variant}"
            dimension = spec["dimension"] if variant == "official_clip" or spec["kind"] == "clip" else (spec["dimension"] if variant == "masked_mean" else 16 * spec["dimension"])
            status = "executed" if primary and primary.get("episodes") else "unavailable / not run"
            unavailable = availability.get(model_name, {}).get("unavailable_reasons", [])
            lines.extend([f"### {display}", "",
                          f"Checkpoint: `{spec['checkpoint']}`; requested sample rate: {spec['sample_rate'] or 'model-defined'} Hz; representation: {spec['tap']}; output size: {dimension}-D. Status: **{status}**.", "",
                          "| Measure / Vital control family | Result or coverage | Output count | Update / supervision path |",
                          "|---|---:|---:|---|"])
            if run:
                license_text = locked_models.get(model_name, {}).get("license", spec["license"])
                runtime_versions = locked_models.get(model_name, {}).get("runtime_versions", {})
                lines.append(f"| Checkpoint provenance / license | SHA-256 `{run.get('checkpoint_sha256', 'unknown')}`; source `{run.get('source_revision', 'unknown')}`; loader `{run.get('loader_revision', 'unknown')}`; runtime `{json.dumps(runtime_versions, sort_keys=True)}`; {license_text} | — | exact local lock; frozen inference |")
            metrics = (("Cross-MIDI common-gallery Recall@1", primary, "recall_at_1"),
                       ("Cross-MIDI common-gallery Recall@5", primary, "recall_at_5"),
                       ("Cross-MIDI common-gallery MRR", primary, "mrr"),
                       ("Cross-MIDI family-level Recall@1", primary, "family_recall_at_1"),
                       ("Cross-MIDI mixed-gallery Recall@1", mixed, "recall_at_1"),
                       ("Same-MIDI other-patch negative win rate", primary, "same_midi_negative_win_rate"),
                       ("Acoustic hard-negative top-5 win rate", primary, "acoustic_hard_negative_win_rate"),
                       ("Known source-lineage near-neighbor win rate", primary, "known_near_neighbor_negative_win_rate"))
            for label, record, key in metrics:
                value = _metric(record, key)
                interval_key = key.removesuffix("_rate")
                value += _interval(record, interval_key)
                if record and key.endswith("_rate"):
                    episodes = record.get(f"{interval_key}_episodes", 0)
                    candidates = record.get(f"{interval_key}_candidates", 0)
                    value += f" ({episodes} episodes; {candidates} negatives)"
                lines.append(f"| {label} | {value} | — | frozen backbone; cosine retrieval |")
            lines.append(f"| Gallery candidates / episodes | {primary.get('gallery_size', []) if primary else 'not run'} / {primary.get('episodes', 'not run') if primary else 'not run'} | — | fixed candidate sets across encoders |")
            if run:
                lines.append(f"| Features created / cache reused | {run.get('features_created', 0)} / {run.get('features_reused', 0)} | {run.get('feature_count', 0)} | resumable extraction pass; cache identity checked |")
            lines.append(f"| Extraction pass time / peak RSS / accelerator memory | {run.get('elapsed_seconds', 'not run') if run else 'not run'} s / {run.get('peak_process_rss_observed_bytes', 'not measured') if run else 'not measured'} / {run.get('peak_accelerator_memory_bytes', 'not measured') if run else 'not measured'} bytes | — | local inference and cache validation; no training |")
            if run and run.get("features_reused", 0):
                lines.append(f"| Resumable pass scope | {run.get('features_created', 0)} new and {run.get('features_reused', 0)} cache-reused features | {run.get('feature_count', 0)} | elapsed time is this pass, not cumulative across earlier partial runs |")
            for group in GROUPS:
                count = groups.get(group, 0)
                lines.append(f"| {group} ❌ 0 / {count} | fixed source input | 0 | unchanged original preset; no loss/gradient |")
            if primary and primary.get("strata"):
                for axis, buckets in sorted(primary["strata"].items()):
                    for label, metrics_by_stratum in sorted(buckets.items()):
                        lines.append(f"| Stratum: {axis} = {label} | Recall@1 {_metric(metrics_by_stratum, 'recall_at_1')}; Recall@5 {_metric(metrics_by_stratum, 'recall_at_5')}; MRR {_metric(metrics_by_stratum, 'mrr')} | {metrics_by_stratum.get('episodes', 0)} episodes | frozen encoder; family bootstrap not stratified |")
            if not run:
                lines.extend(["", f"Checkpoint/license: {spec['license']}." if not unavailable else f"Unavailable: {'; '.join(unavailable)}.", ""])
    observed_arms = []
    for arm, outcomes in model_results.items():
        primary = outcomes.get("cross_midi_common_gallery", {})
        if primary.get("episodes", 0):
            observed_arms.append((arm, primary, outcomes.get("cross_midi_mixed_gallery", {})))
    if observed_arms:
        ranked_arms = sorted(observed_arms,
            key=lambda item: (-item[1].get("recall_at_1", 0.0), -item[1].get("mrr", 0.0), item[0]))
        lines.extend(["## Comparative findings", "",
                      "| Executed arm | Common-gallery R@1 (family 95% CI) | MRR (family 95% CI) | Mixed-gallery R@1 | Acoustic hard-negative top-5 win |",
                      "|---|---:|---:|---:|---:|"])
        for arm, primary, mixed in ranked_arms:
            lines.append(f"| {arm} | {_metric(primary, 'recall_at_1')}{_interval(primary, 'recall_at_1')} | "
                         f"{_metric(primary, 'mrr')}{_interval(primary, 'mrr')} | "
                         f"{_metric(mixed, 'recall_at_1')} | {_metric(primary, 'acoustic_hard_negative_win_rate')} "
                         f"({primary.get('acoustic_hard_negative_win_episodes', 0)} queries) |")
        best_arm, best_primary, _ = ranked_arms[0]
        if len(ranked_arms) == 1:
            lines.append(f"Only `{best_arm}` completed this real benchmark; there is no measured encoder comparison to select a winner.")
        else:
            comparison_lines = []
            unresolved = False
            for other_arm, _, _ in ranked_arms[1:]:
                differences = (evaluation or {}).get("paired_family_bootstrap_differences", {})
                key, direction = (f"{best_arm} - {other_arm}", 1.0)
                if key not in differences:
                    key, direction = (f"{other_arm} - {best_arm}", -1.0)
                interval = differences.get(key, {}).get("cross_midi_common_gallery", {}).get("recall_at_1", {})
                if interval:
                    lower, upper = direction * interval["lower_95"], direction * interval["upper_95"]
                    if direction < 0:
                        lower, upper = upper, lower
                    comparison_lines.append(f"paired ΔR@1 versus `{other_arm}`: {direction * interval['difference']:+.4f} "
                        f"[{lower:+.4f}, {upper:+.4f}]")
                    unresolved |= lower <= 0 <= upper
            if unresolved or not comparison_lines:
                lines.append(f"`{best_arm}` had the highest observed common-gallery Recall@1 ({_metric(best_primary, 'recall_at_1')}) "
                    "and is the provisional lead for a later small 14-output/reconstruction follow-up; retain the next-ranked executed arm as a comparator because the paired family interval does not resolve a gap. "
                    + "; ".join(comparison_lines) + ".")
            else:
                lines.append(f"`{best_arm}` led on common-gallery Recall@1 ({_metric(best_primary, 'recall_at_1')}) with paired family intervals above zero; recommend it as the lead for a later small 14-output/reconstruction follow-up. Keep `{ranked_arms[1][0]}` as a secondary comparator only if a second arm is useful. "
                    + "; ".join(comparison_lines) + ".")
        lines.append("These findings cover only the checkpointed arms, this selected validation cohort, and the stated retrieval metrics. They do not establish reconstruction quality or rank unavailable encoders.")
        clap_primary = model_results.get("clap_music:official_clip", {}).get("cross_midi_common_gallery", {})
        strata = clap_primary.get("strata", {})
        pitch = strata.get("pitch_register_gap_band", {})
        articulation = strata.get("query_articulation", {})
        durations = strata.get("query_duration_band", {})
        notes = []
        if pitch.get("0-5 semitones") and pitch.get("13+ semitones"):
            notes.append(f"CLAP R@1 was {_metric(pitch['0-5 semitones'], 'recall_at_1')} at 0–5 semitones of register gap and {_metric(pitch['13+ semitones'], 'recall_at_1')} at 13+ semitones")
        if articulation.get("held") and articulation.get("phrased"):
            notes.append(f"{_metric(articulation['held'], 'recall_at_1')} on held versus {_metric(articulation['phrased'], 'recall_at_1')} on phrased queries")
        if durations.get("2-4s") and durations.get("6-8s"):
            notes.append(f"{_metric(durations['2-4s'], 'recall_at_1')} on 2–4s versus {_metric(durations['6-8s'], 'recall_at_1')} on 6–8s queries")
        if notes:
            lines.append("Descriptive CLAP strata point to wider register gaps, held notes, and longer clips as harder cases: " + "; ".join(notes) + ". These are not separately powered experiments.")
        dac_pooling = (evaluation or {}).get("paired_family_bootstrap_differences", {}).get(
            "dac_44khz:masked_mean - dac_44khz:temporal_pyramid_8", {}).get("cross_midi_common_gallery", {}).get("recall_at_1")
        if dac_pooling and dac_pooling["upper_95"] < 0:
            lines.append(f"Within DAC, the temporal pyramid exceeded masked mean pooling by {-dac_pooling['difference']:.4f} Recall@1 "
                         f"(paired family 95% interval [{-dac_pooling['upper_95']:.4f}, {-dac_pooling['lower_95']:.4f}]).")
        expression_buckets = strata.get("query_expression", {})
        if expression_buckets and set(expression_buckets) == {"plain"}:
            lines.append("The selected MIDI set contains no supported expressive-event queries, so main-benchmark sensitivity to sustain, pitch bend, channel pressure, and CC modulation remains unmeasured.")
        lines.append("")
    if controlled and controlled_inventory_path is not None and controlled_inventory is not None:
        midi_description = "; ".join(
            f"polyphony {row['max_polyphony']}, median pitch {row['pitch_median']}, clip {row['duration_seconds']:.2f}s, "
            f"median note {row['note_duration_median_seconds']:.2f}s, velocity {row['velocity_min']}–{row['velocity_max']}, {row['articulation']}"
            for row in controlled.get("midi_strata", []))
        validation = controlled.get("inherited_source_validation_diagnostics", {})
        modulation_source = controlled.get("controlled_fields", {}).get("modulation_amount_shift", {}).get("route_source", "unknown")
        modulation_observability = ("The perturbed route uses a slow LFO; these excerpts do not establish cycle/phase observability."
                                    if str(modulation_source).lower().startswith("lfo") else
                                    f"The perturbed route source is `{modulation_source}`; this does not test slow-LFO cycle or phase observability.")
        lines.extend(["## Controlled preset perturbations", "",
            f"This secondary diagnostic uses one original validation preset, three single-control perturbations, and {len(controlled.get('midi_strata', []))} real MIDI excerpts selected to cover low monophonic, high polyphonic, and longest-note monophonic performances, plus a nonconstant-velocity excerpt where available. Each query variant is matched against the four-version gallery rendered on another MIDI. The four-candidate Recall@5 is necessarily 1.0 when all features exist and is shown only for consistency; this set is too small for confidence intervals or general encoder recommendations.",
            f"MIDI excerpt properties: {midi_description or 'not recorded'}. The original source carried {validation.get('count', 'unreported')} inherited Vital schema diagnostics ({json.dumps(validation.get('codes', {}), sort_keys=True)}); the three variants introduced {validation.get('new_diagnostics_after_perturbation', 'unreported')} new diagnostics. Velocity ranges are recorded, but response to velocity was not separately isolated. The diagnostic measures representation similarity only and has no human audibility threshold. {modulation_observability}",
            f"The linked [{controlled_inventory_path.name}](./{controlled_inventory_path.name}) records every exact Vital control/instance for each version ({controlled_inventory['records_per_version']} rows per version). No model predicts or optimizes these controls; each perturbation is manually specified, with every other source value held fixed.", ""])
        labels = {"original": "Original source preset", "filter_cutoff_shift": "Filter cutoff perturbation",
                  "amp_attack_shift": "Amplitude envelope attack perturbation",
                  "modulation_amount_shift": "Live modulation amount perturbation"}
        for version in controlled["variants"]:
            version_summary = controlled_inventory["versions"][version]
            lines.extend([f"### {labels.get(version, version)}", "",
                          "| Measure / Vital control family | Result or coverage | Output count | Update method / training signal |",
                          "|---|---:|---:|---|"])
            change = controlled.get("controlled_fields", {}).get(version)
            if change:
                details = (f"{change['field']}: {change['before_raw']:g} → {change['after_raw']:g} raw "
                           f"({change['before_normalized']:.3f} → {change['after_normalized']:.3f} normalized)")
                if "route_source" in change:
                    details += f"; {change['route_source']} → {change['route_destination']}"
            else:
                details = "source preset unchanged"
            lines.append(f"| Applied Vital field | {details} | 0 | manual perturbation; no training, labels, gradients, or loss |")
            for model_name, model_result in controlled.get("models", {}).items():
                for tap, pool in model_result.get("pool_results", {}).items():
                    score = pool.get("per_target_variant", {}).get(version, {})
                    cross = score.get("cross_midi_retrieval", {})
                    same = score.get("same_midi_control", {})
                    dimension = (MODEL_REGISTRY[model_name]["dimension"] if tap in {"official_clip", "masked_mean"}
                                 else 16 * MODEL_REGISTRY[model_name]["dimension"])
                    prefix = f"{MODEL_REGISTRY[model_name]['name']} — {tap}"
                    lines.append(f"| {prefix} cross-MIDI variant retrieval | R@1 {_metric(cross, 'recall_at_1')}; R@5 {_metric(cross, 'recall_at_5')}; MRR {_metric(cross, 'mrr')} ({cross.get('queries', 0)} queries) | {dimension}-D | frozen backbone; cosine retrieval |")
                    lines.append(f"| {prefix} same-MIDI control | R@1 {_metric(same, 'recall_at_1')}; MRR {_metric(same, 'mrr')} | {dimension}-D | same performance; controlled versions are hard negatives |")
                    lines.append(f"| {prefix} exact-version cross-MIDI cosine | {_metric(score, 'same_variant_different_midi_mean_cosine')} | {dimension}-D | pooled frozen feature; cosine similarity |")
                    lines.append(f"| {prefix} extraction time / peak RSS / accelerator memory | {model_result.get('elapsed_seconds', 'not measured')} s / {model_result.get('peak_process_rss_observed_bytes', 'not measured')} / {model_result.get('peak_accelerator_memory_bytes', 'not measured')} bytes | — | local inference only; no training |")
            for group in GROUPS:
                counts = version_summary["controls_by_family"][group]
                enabled, total = counts["enabled"], counts["total"]
                status = "✅" if enabled else "❌"
                method = "manual perturbation; no model output" if enabled else "fixed original preset input"
                lines.append(f"| {group} {status} {enabled} / {total} | fixed source except the listed field | 0 | {method}; no supervision or loss |")
            lines.append("")
    baselines = (evaluation or {}).get("midi_only_and_random_baselines", {})
    lines.extend(["## Baselines", "", "| Condition | MIDI-only R@1 / R@5 / MRR | Family bootstrap 95% interval for R@1 | Random R@1 / R@5 / MRR |",
                  "|---|---:|---:|---:|"])
    for condition, label in (("cross_midi_common_gallery", "Common gallery MIDI"), ("cross_midi_mixed_gallery", "Mixed gallery MIDI")):
        baseline = baselines.get(condition)
        random_result = baselines.get("random", {}).get(condition) if baselines else None
        midi_text = "not run" if not baseline else " / ".join(_metric(baseline, key) for key in ("recall_at_1", "recall_at_5", "mrr"))
        random_text = "not run" if not random_result else " / ".join(_metric(random_result, key) for key in ("recall_at_1", "recall_at_5", "mrr"))
        lines.append(f"| {label} | {midi_text} | {_interval(baseline, 'recall_at_1').strip() or 'not run'} | {random_text} |")
    lines.extend(["", "## Evaluation protocol", "",
                  "The primary condition uses one common, existing gallery MIDI per episode and a query render of the same target preset on another existing MIDI. Query/gallery MIDI IDs differ; selection rotates deterministically. Every admitted validation preset is paired with all unique compatible same-split MIDI excerpts, making multiple cross-renders per preset. Gallery candidates are all admitted source states. The mixed-gallery condition uses a deterministic candidate-specific MIDI assignment and is reported separately. A same-MIDI different-patch negative check compares each cross-MIDI positive against other patches rendered on the query MIDI. Known frozen lineage groups are reported as family-level retrieval and family-cluster bootstrap units.", "",
                  "Metrics use cosine similarity over FP32 cached vectors; candidate sets and query episodes are paired across available arms. Acoustic hard negatives are selected independently of the tested encoders using cosine similarity over normalized 64-bin log-frequency power spectra of the target query and same-MIDI candidate renders; each encoder’s cross-MIDI positive is then compared against the five nearest such negatives on that query MIDI. The report includes exact-state and family-level Recall@1, Recall@5, MRR, family-cluster 95% bootstrap intervals, MIDI-only metadata and random baselines, and strata for polyphony, articulation, expression, source-lineage group size, pitch gap, duration, and note density. Sequence outputs include a valid-frame mean and an 8-bin ordered mean/std pyramid. No pooling or similarity setting is selected on reserved test labels.", "",
                  "This benchmark uses new Vital renders only; source WAVs are not used as model inputs or mixed with renders.", ""])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return {"models": len(MODEL_REGISTRY), "arms": sum(1 if spec["kind"] == "clip" else 2 for spec in MODEL_REGISTRY.values()),
            "control_records": inventory_count, "path": str(destination)}
