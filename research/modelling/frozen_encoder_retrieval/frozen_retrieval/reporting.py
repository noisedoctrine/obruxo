from __future__ import annotations

from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys

from .core import MODEL_REGISTRY
from .encoders import available_candidates


GROUPS = (
    "Global settings", "Oscillators and voices", "Filters", "Envelopes", "LFOs and modulation",
    "Effects", "Wavetables, curves and samples", "Other fixed controls",
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
    return "Other fixed controls"


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
                    execution_note: str | None = None) -> dict:
    inventory_count, groups = _inventory_summary(inventory)
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
    lines = ["# Frozen audio encoder retrieval benchmark", "",
             "Issue: [#152](https://github.com/noisedoctrine/obruxo/issues/152). This report covers frozen pretrained representations on original Vital source presets and existing MIDI performances. No synthesis controls are predicted, tuned, or optimized.", "",
             "## Execution status", ""]
    if local_dataset:
        dataset_summary_path = local_dataset / "dataset_summary.private.json"
        if dataset_summary_path.is_file():
            dataset = json.loads(dataset_summary_path.read_text(encoding="utf-8"))
            lines.extend([f"The frozen {dataset.get('source_split')} cohort contains {dataset.get('presets_verified', 0)} verified original presets, {dataset.get('unique_midi_excerpts', 0)} compatible unique MIDI excerpts, {dataset.get('episodes', 0)} retrieval episodes, and {dataset.get('render_requests', 0)} planned original-preset renders.",
                          f"Render profile: `{dataset.get('renderer_id')}`. Selected renderer capabilities: `{json.dumps(dataset.get('renderer_capabilities', {}), sort_keys=True)}`."])
        render_path = local_dataset / "render_run.latest.json"
        if render_path.is_file():
            render = json.loads(render_path.read_text(encoding="utf-8"))
            lines.append(f"Vital render stage: {render.get('accepted', 0)} QA accepted of {render.get('requested', 0)} planned requests; this run rendered {render.get('runs_rendered_this_run', 0)} and reused {render.get('cache_reused_this_run', 0)} cached WAVs in {render.get('wall_seconds_this_run', 0)} seconds.")
        if evaluation:
            lines.extend([f"The comparison uses {evaluation.get('common_episode_count_across_available_arms', 0)} identical episodes across the available arms; held-out reserved sources used: **no**."])
            coverage = evaluation.get("coverage_by_condition", {})
            lines.extend(["", "### Render and feature coverage", "",
                          "| Condition | Planned episodes | Render QA qualified | Excluded | Feature qualified by arm | Common across arms |",
                          "|---|---:|---:|---:|---|---:|"])
            for condition, record in sorted(coverage.items()):
                feature_counts = ", ".join(f"{arm}: {count}" for arm, count in sorted(record.get("feature_qualified_episodes_by_arm", {}).items())) or "none"
                lines.append(f"| {condition} | {record.get('planned_episodes', 0)} | {record.get('qa_qualified_episodes', 0)} | {record.get('excluded_episodes', 0)} | {feature_counts} | {record.get('common_feature_qualified_episodes_across_arms', 0)} |")
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
    lines.append("Controlled patch-perturbation diagnostic: not run; no controlled score is claimed.")
    missing_runtime = sorted({package for candidate in availability.values() for package, value in candidate.get("dependencies", {}).items()
                              if value.get("version") == "missing"})
    lines.extend(["", f"Runtime preflight: Python `{sys.version.split()[0]}`; missing encoder runtime packages: {', '.join(f'`{name}`' for name in missing_runtime) or 'none'}. Checkpoint root supplied: **{'yes' if checkpoint_root else 'no'}**. Vital plugin environment variable configured: **{'yes' if os.environ.get('OBRUXO_VITAL_PLUGIN') else 'no'}**.",
                  "The existing [corpus audit report](../../../training/reports/CORPUS_INPUT_RESULTS.md) records prior static eligibility findings. Those audit counts are not native render eligibility. The separate [PresetShare audit](../../basic_pitch/README.md) has its own pairing rules and counts.", "",
                  "The default command selects the corpus-qualified full-preset expression renderer identity, including the established curve-input, ignored-empty-map, and native scalar readback policies. It supports note/polyphony, pitch-bend, channel-pressure, CC 1/64/74, release-velocity and multiple channels; tempo changes remain unsupported and are counted as exclusions. Unsupported events are rejected rather than silently removed. Musical spans are complete-note excerpts of 2–8 seconds plus a fixed 2-second release tail; model input is mono and 4–10 seconds, downmixed by channel mean and polyphase-resampled. No shared gain normalization is applied; EnCodec retains its own checkpoint-configured RMS normalization when enabled.", "",
                  "Render and feature stages are independent, resumable commands. Vital remains a single-worker in-process renderer; an asynchronous manifest reader feeds a bounded queue of request batches. Render keys include exact preset/MIDI/excerpt/renderer/policy identities. Feature keys include audio hash, checkpoint digest/revision, loader revision, sample rate, preprocessing, extraction tap, pooling, and FP32 precision. Weights are never downloaded by the run commands. Full paths, identifiers, WAVs, caches, checkpoints, and per-episode results belong in ignored local outputs.", "",
                  "No reconstruction head, source-label warm start, shared-model training, direct patch optimization, or audio-loss training is part of this experiment. No candidate can be recommended for reconstruction until the retrieval comparison has actually run; if results remain unavailable, the decision is inconclusive.", "",
                  "## Frozen-control scope", "",
                  f"The linked [{inventory.name}](./{inventory.name}) contains one record for every exact control/instance from the existing 979-record Vital control inventory. All arms use original source preset inputs unchanged; each group is fixed with 0 enabled instances and 0 output values. No controls are omitted from the inventory; the renderer passes the original preset document through unchanged.", ""])

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
            metrics = (("Cross-MIDI common-gallery Recall@1", primary, "recall_at_1"),
                       ("Cross-MIDI common-gallery Recall@5", primary, "recall_at_5"),
                       ("Cross-MIDI common-gallery MRR", primary, "mrr"),
                       ("Cross-MIDI mixed-gallery Recall@1", mixed, "recall_at_1"),
                       ("Same-MIDI other-patch negative win rate", primary, "same_midi_negative_win_rate"),
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
            lines.append(f"| Extraction time / peak memory | {run.get('elapsed_seconds', 'not run') if run else 'not run'} s / {run.get('peak_process_rss_observed_bytes', 'not measured') if run else 'not measured'} bytes | — | local inference only; no training |")
            for group in GROUPS:
                count = groups.get(group, 0)
                if count:
                    lines.append(f"| {group} ❌ 0 / {count} | fixed source input | 0 | unchanged original preset; no loss/gradient |")
            if primary and primary.get("strata"):
                for axis, buckets in sorted(primary["strata"].items()):
                    for label, metrics_by_stratum in sorted(buckets.items()):
                        lines.append(f"| Stratum: {axis} = {label} | Recall@1 {_metric(metrics_by_stratum, 'recall_at_1')}; Recall@5 {_metric(metrics_by_stratum, 'recall_at_5')}; MRR {_metric(metrics_by_stratum, 'mrr')} | {metrics_by_stratum.get('episodes', 0)} episodes | frozen encoder; family bootstrap not stratified |")
            lines.extend(["", f"Checkpoint/license: {spec['license']}." if not unavailable else f"Unavailable: {'; '.join(unavailable)}.", ""])
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
                  "Metrics use cosine similarity over FP32 cached vectors; candidate sets and query episodes are paired across available arms. The report includes Recall@1/5, MRR, family-cluster 95% bootstrap intervals, MIDI-only metadata and random baselines, and strata for polyphony, articulation, expression, pitch gap, duration, and note density. Sequence outputs include a valid-frame mean and an 8-bin ordered mean/std pyramid. No pooling or similarity setting is selected on reserved test labels.", "",
                  "This benchmark uses new Vital renders only; source WAVs are not used as model inputs or mixed with renders.", ""])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return {"models": len(MODEL_REGISTRY), "arms": sum(1 if spec["kind"] == "clip" else 2 for spec in MODEL_REGISTRY.values()),
            "control_records": inventory_count, "path": str(destination)}
