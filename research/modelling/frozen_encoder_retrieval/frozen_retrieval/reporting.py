from __future__ import annotations

from collections import Counter, defaultdict
import importlib.util
import json
import math
import os
from pathlib import Path
import random
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
    if path.startswith("settings.") and not any(token in text for token in ("filter", "envelope", "env_", "lfo", "osc", "unison", "reverb", "chorus", "delay", "phaser", "distortion")):
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


def _controlled_midi_profile(rows: list[dict]) -> str:
    if not rows:
        return "not recorded"
    articulations = Counter(row.get("articulation", "unknown") for row in rows)
    variable_velocity = sum(row.get("velocity_min") != row.get("velocity_max") for row in rows)
    return (f"{len(rows)} MIDI IDs; polyphony {min(row['max_polyphony'] for row in rows)}–"
            f"{max(row['max_polyphony'] for row in rows)}, median pitch {min(row['pitch_median'] for row in rows)}–"
            f"{max(row['pitch_median'] for row in rows)}, clip length "
            f"{min(row['duration_seconds'] for row in rows):.2f}–{max(row['duration_seconds'] for row in rows):.2f}s, "
            f"median note duration {min(row['note_duration_median_seconds'] for row in rows):.2f}–"
            f"{max(row['note_duration_median_seconds'] for row in rows):.2f}s, variable velocity "
            f"{variable_velocity}/{len(rows)}, articulation {articulations['held']} held / "
            f"{articulations['phrased']} phrased")


def _percent(result: dict | None, key: str) -> str:
    if not result or result.get(key) is None:
        return "not run"
    return f"{100 * result[key]:.1f}%"


def _interval(result: dict | None, key: str) -> str:
    interval = result.get("confidence_intervals", {}).get(key) if result else None
    return f" [{interval['lower_95']:.4f}, {interval['upper_95']:.4f}]" if interval else ""


def _paired_delta(evaluation: dict, left: str, right: str) -> dict:
    pairs = evaluation.get("paired_cluster_bootstrap_differences", {})
    pair, direction = f"{left} - {right}", 1
    if pair not in pairs:
        pair, direction = f"{right} - {left}", -1
    interval = pairs.get(pair, {}).get("cross_midi_common_gallery", {}).get(
        "performance_and_family", {}).get("recall_at_1", {})
    if not interval:
        return {}
    lower, upper = direction * interval["lower_95"], direction * interval["upper_95"]
    if direction < 0:
        lower, upper = upper, lower
    return {"difference": direction * interval["difference"], "lower_95": lower, "upper_95": upper,
            "performance_clusters": interval.get("performance_clusters", 0),
            "family_clusters": interval.get("family_clusters", 0)}


def _performance_profile(dataset_dir: Path, split: str | None) -> dict:
    path = dataset_dir / "performances.private.jsonl"
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    stats = [row["stats"] for row in rows if (row.get("performance_source_split") or row.get("split")) == split]
    if not stats:
        return {}
    express = lambda key: sum(bool(row.get(key)) for row in stats)
    return {"performances": len(stats),
            "mono": sum(row.get("max_polyphony", 0) == 1 for row in stats),
            "polyphonic": sum(row.get("max_polyphony", 0) > 1 for row in stats),
            "articulation_counts": dict(Counter(row.get("articulation", "unknown") for row in stats)),
            "pitch_median_min": min(row["pitch_median"] for row in stats if row.get("pitch_median") is not None),
            "pitch_median_max": max(row["pitch_median"] for row in stats if row.get("pitch_median") is not None),
            "variable_velocity": sum(row.get("velocity_min", 0) < row.get("velocity_max", 0) for row in stats),
            "pitch_bend": express("pitch_bend_events"), "sustain": express("sustain_events"),
            "other_expression": sum(bool(row.get("expression_events") or row.get("channel_pressure_events")
                                         or any(int(controller) != 64 and count
                                                for controller, count in row.get("control_change_counts", {}).items()))
                                    for row in stats)}


def _summary_cohort_lines(label: str, dataset: dict, evaluation: dict | None) -> list[str]:
    performances = dataset.get("unique_midi_excerpts", 0)
    performance_split = dataset.get("performance_source_split") or dataset.get("source_split")
    condition = (evaluation or {}).get("coverage_by_condition", {}).get("cross_midi_common_gallery", {})
    episodes = condition.get("common_feature_qualified_episodes_across_arms", 0)
    evaluated_performances = next((outcomes.get("cross_midi_common_gallery", {}).get(
        "performance_cluster_bootstrap", {}).get("recall_at_1", {}).get("performance_clusters", 0)
        for outcomes in (evaluation or {}).get("model_results", {}).values()
        if outcomes.get("cross_midi_common_gallery", {}).get("episodes", 0)), 0)
    performance_count = (f"{performances} selected MIDI IDs; {evaluated_performances} with evaluated queries"
                         if evaluation else f"{performances} selected MIDI performance IDs; evaluation not run")
    lines = [f"### {label}: {performance_count}; {episodes} common-gallery episodes",
             f"Validation preset gallery: {dataset.get('presets_verified', 0)} verified states; performances sourced from `{performance_split}`. Reserved test and confirmation sources used: **no**.", "",
             "| Frozen representation | R@1 [crossed 95% CI] | R@5 | MRR | Same-MIDI hard-negative win | Acoustic hard-negative win |",
             "|---|---:|---:|---:|---:|---:|"]
    profile = dataset.get("performance_profile", {})
    if profile:
        articulation = ", ".join(f"{count} {kind}" for kind, count in sorted(profile["articulation_counts"].items()))
        lines[2:2] = [f"Selected-excerpt diversity: {profile['mono']} monophonic / {profile['polyphonic']} polyphonic; "
            f"{articulation}; median pitch {profile['pitch_median_min']}–{profile['pitch_median_max']}; "
            f"variable velocity in {profile['variable_velocity']}/{profile['performances']}; expressive events in "
            f"{profile['sustain']} sustain, {profile['pitch_bend']} pitch bend, and "
            f"{profile['other_expression']} other CC/pressure excerpts.", ""]
    if dataset.get("performance_selection_policy"):
        lines[2:2] = [f"Performance selection: deterministic coverage-balanced sample of {performances} from "
            f"{dataset.get('performance_pool_discovered', 'unknown')} compatible IDs; "
            f"{dataset.get('performance_sampled_out', 0)} sampled out and "
            f"{dataset.get('performance_identity_overlap_excluded', 0)} exact-ID overlaps excluded.", ""]
    lines = [line for index, line in enumerate(lines) if line or index == 0 or lines[index - 1] != ""]
    results = (evaluation or {}).get("model_results", {})
    ranked = []
    for arm, outcomes in results.items():
        primary = outcomes.get("cross_midi_common_gallery", {})
        if primary.get("episodes", 0):
            ranked.append((arm, outcomes, primary))
    ranked.sort(key=lambda item: (-item[2].get("recall_at_1", 0.0), -item[2].get("mrr", 0.0), item[0]))
    for arm, outcomes, primary in ranked:
        model_name, variant = arm.split(":", 1)
        spec = MODEL_REGISTRY[model_name]
        crossed_ci = primary.get("two_way_cluster_bootstrap", {}).get("recall_at_1", {})
        interval = lambda value: (f"[{100 * value['lower_95']:.1f}%, {100 * value['upper_95']:.1f}%]"
                                  if value and "lower_95" in value and "upper_95" in value else "not estimated")
        def negative_cell(metric: str, count_key: str) -> str:
            rate_key = f"{metric}_rate"
            return f"{_percent(primary, rate_key)} (n={primary.get(count_key, 0):,})"

        same = negative_cell("same_midi_negative_win", "same_midi_negative_win_episodes")
        acoustic = negative_cell("acoustic_hard_negative_win", "acoustic_hard_negative_win_episodes")
        dimension = spec["dimension"] if variant in {"official_clip", "masked_mean"} or spec["kind"] == "clip" else 16 * spec["dimension"]
        lines.append(f"| {spec['name']} — {variant} ({dimension}-D) | {_percent(primary, 'recall_at_1')} {interval(crossed_ci)} | {_percent(primary, 'recall_at_5')} | {_percent(primary, 'mrr')} | {same} | {acoustic} |")
    if not ranked:
        lines.append("| No arm completed | not run | not run | not run | not run | not run |")
    random_baseline = (evaluation or {}).get("midi_only_and_random_baselines", {}).get("random", {}).get(
        "cross_midi_common_gallery", {})
    if random_baseline:
        lines.append(f"| Random chance | {_percent(random_baseline, 'recall_at_1')} (fixed) | {_percent(random_baseline, 'recall_at_5')} | {_percent(random_baseline, 'mrr')} | — | — |")
    family_leaders = []
    seen_families = set()
    for arm, _, _ in ranked:
        family = arm.split(":", 1)[0]
        if family not in seen_families:
            seen_families.add(family)
            family_leaders.append(arm)
    reference = family_leaders[0] if family_leaders else None
    if len(family_leaders) > 1:
        lines.extend(["", f"Paired crossed-cluster Recall@1 differences from the leading family representation (`{reference}`; best pooling per family):", "",
                      "| Comparator | ΔR@1 (percentage points) | Crossed MIDI × preset-family 95% CI |",
                      "|---|---:|---:|"])
        for arm in family_leaders[1:]:
            delta = _paired_delta(evaluation, reference, arm)
            if delta:
                span = "includes zero" if delta["lower_95"] <= 0 <= delta["upper_95"] else ""
                lines.append(f"| `{arm}` | {100 * delta['difference']:+.1f} | [{100 * delta['lower_95']:+.1f}, {100 * delta['upper_95']:+.1f}] {span} |")
    basic_pitch = [(arm, primary) for arm, _, primary in ranked if arm.startswith("basic_pitch_trunk:")]
    if basic_pitch:
        best_bp_arm, best_bp = max(basic_pitch, key=lambda item: item[1].get("recall_at_1", 0.0))
        bp_delta = _paired_delta(evaluation, reference, best_bp_arm) if reference else {}
        bp_stability = (evaluation.get("ranking_stability_by_performance", {})
            .get("cross_midi_common_gallery", {}).get(best_bp_arm, {}))
        summary = (f"Basic Pitch's strongest pooling arm was `{best_bp_arm}` at {_percent(best_bp, 'recall_at_1')} R@1 "
                   f"({best_bp.get('episodes', 0):,} queries; 50 MIDI performances); it placed first on "
                   f"{bp_stability.get('first_place_performances', 0)}/{bp_stability.get('performance_clusters', 0)} "
                   "individual performances.")
        if bp_delta:
            summary += (f" The paired CIR-minus-Basic-Pitch gap was {100 * bp_delta['difference']:+.1f} pp "
                        f"[{100 * bp_delta['lower_95']:+.1f}, {100 * bp_delta['upper_95']:+.1f}] across "
                        "preset-family and MIDI-performance clusters.")
        summary += " These retrieval results do not support prioritizing this transcription-trained trunk for a later timbre-reconstruction screen."
        lines.extend(["", summary])
    lines.extend(["", "The table gives the headline retrieval measures on the shared candidate gallery. R@1 uncertainty is crossed over preset family and MIDI performance (2,000 bootstrap replicates); the heading gives the episode and independent-performance denominators. Hard-negative rates are descriptive proportions with query denominators. Query-audio quality and correct-match gallery quality are broken out below. For the full candidate gallery, the report gives the count and fraction of warned renders rather than labeling all candidates with one bucket. These are descriptive associations: warning status can covary with preset acoustics.", ""])
    lineage = [f"`{arm}` {_percent(primary, 'known_near_neighbor_negative_win_rate')} (n={primary.get('known_near_neighbor_negative_win_episodes', 0):,})"
        for arm, _, primary in ranked if primary.get("known_near_neighbor_negative_win_episodes", 0)]
    if lineage:
        lines.extend(["**Known source-lineage near-neighbor win rate:** " + "; ".join(lineage) + ". This diagnostic is limited to the identified source-lineage negatives.", ""])
    if evaluation:
        model_results = evaluation.get("model_results", {})
        quality_leaders = []
        for quality, label in (("clean", "clean"), ("warning", "warning")):
            candidates = [(arm, outcomes.get("cross_midi_common_gallery", {}).get("strata", {}).get(
                "query_render_quality", {}).get(quality, {})) for arm, outcomes in model_results.items()]
            candidates = [(arm, row) for arm, row in candidates if row.get("episodes", 0)]
            if candidates:
                arm, row = max(candidates, key=lambda item: (item[1].get("recall_at_1", 0), item[1].get("mrr", 0)))
                ci = row.get("performance_cluster_bootstrap", {})
                quality_leaders.append(f"{label}: `{arm}` {_percent(row, 'recall_at_1')} {interval(ci)} "
                    f"(q={row.get('episodes', 0):,}; {ci.get('performance_clusters', 0)} MIDI IDs)")
        if quality_leaders:
            lines.extend(["**Query-render QA strata:** " + "; ".join(quality_leaders) +
                ". These are per-stratum MIDI-cluster intervals; accepted warning status is associated with the rendered preset and does not imply a causal quality effect.", ""])
        quality_rows = []
        for arm, outcomes in sorted(model_results.items()):
            primary = outcomes.get("cross_midi_common_gallery", {})
            if not primary.get("episodes"):
                continue
            strata = primary.get("strata", {})
            def quality_text(axis: str) -> str:
                groups = strata.get(axis, {})
                return " / ".join(f"{_percent(groups[label], 'recall_at_1')} (n={groups[label].get('episodes', 0):,})"
                                   for label in ("clean", "warning") if label in groups) or "not split"
            gallery_quality = primary.get("gallery_render_quality", {})
            if not gallery_quality.get("episodes"):
                continue
            warned_fraction = gallery_quality.get("mean_warning_candidate_fraction")
            warned_text = (f"{100 * warned_fraction:.1f}% of candidates on average "
                           f"({gallery_quality.get('episodes_with_any_warning', 0):,}/{gallery_quality.get('episodes', 0):,} galleries contain a warning)"
                           if warned_fraction is not None else "not recorded")
            quality_rows.append((arm, quality_text("query_render_quality"),
                                 quality_text("correct_match_gallery_render_quality"), warned_text))
        if quality_rows:
            lines.extend(["**Render-quality diagnostics** (R@1 clean / warning; these buckets are descriptive):", "",
                "| Encoder arm | Query WAV | Correct-match gallery WAV | Warned candidates across all 129-candidate galleries |",
                "|---|---:|---:|---|"])
            lines.extend(f"| `{arm}` | {query} | {target} | {warned} |" for arm, query, target, warned in quality_rows)
            lines.append("")
        performance_stability = evaluation.get("ranking_stability_by_performance", {})
        condition_labels = (("cross_midi_common_gallery", "Common gallery"),
                            ("cross_midi_mixed_gallery", "Mixed gallery"))
        selected_by_condition = {}
        for condition, label in condition_labels:
            selected = {}
            for arm, outcomes in model_results.items():
                result = outcomes.get(condition, {})
                if not result.get("episodes"):
                    continue
                family = arm.split(":", 1)[0]
                prior = selected.get(family)
                if prior is None or (result.get("recall_at_1", 0), result.get("mrr", 0)) > (
                        prior[1].get("recall_at_1", 0), prior[1].get("mrr", 0)):
                    selected[family] = (arm, result)
            selected_by_condition[condition] = selected
        families = sorted(set().union(*(set(rows) for rows in selected_by_condition.values())))
        if families:
            performance_clusters = max((row.get("performance_clusters", 0)
                for rows in performance_stability.values() for row in rows.values()), default=0)
            lines.extend([f"**Per-performance ranking stability** (best observed pooling within each family and condition; bootstrap resamples {performance_clusters} MIDI IDs):", "",
                "| Encoder family | Common-gallery arm; R@1; first-place performances; bootstrap top-rank probability | Mixed-gallery arm; R@1; first-place performances; bootstrap top-rank probability |",
                "|---|---|---|"])
            for family in families:
                spec = MODEL_REGISTRY[family]
                cells = []
                for condition, _ in condition_labels:
                    arm, result = selected_by_condition[condition].get(family, (None, None))
                    stability = performance_stability.get(condition, {}).get(arm, {}) if arm else {}
                    if not result:
                        cells.append("not run")
                        continue
                    cells.append(f"`{arm}`; {_percent(result, 'recall_at_1')}; "
                        f"{stability.get('first_place_performances', 0)}/{stability.get('performance_clusters', 0)}; "
                        f"{stability.get('bootstrap_top_rank_probability', 0):.1%}")
                lines.append(f"| {spec['name']} | {cells[0]} | {cells[1]} |")
            lines.extend(["", "Pooling is selected descriptively within each evaluated condition; the primary common-gallery table above lists every arm. Top-rank probabilities compare all scored arms within the condition.", ""])
    return lines


def _issue155_followup_lines(results: dict, shortlist: dict | None) -> list[str]:
    def score_cell(summary: dict) -> str:
        metrics = summary.get("metrics", {})
        recall = metrics.get("recall_at_1", {})
        ci = recall.get("crossed_95_ci", {})
        r1 = recall.get("mean")
        r1_text = "not run" if r1 is None else f"{100 * r1:.1f}%"
        if "lower_95" in ci and "upper_95" in ci:
            r1_text += f" [{100 * ci['lower_95']:.1f}%, {100 * ci['upper_95']:.1f}%]"
        r5, mrr = metrics.get("recall_at_5", {}).get("mean"), metrics.get("mrr", {}).get("mean")
        return f"{r1_text} / {100 * r5:.1f}% / {100 * mrr:.1f}%" if r5 is not None and mrr is not None else r1_text

    def paired_cell(summary: dict | None) -> str:
        if not summary:
            return "not estimated"
        return (f"{100 * summary['difference']:+.1f} pp "
                f"[{100 * summary['lower_95']:+.1f}, {100 * summary['upper_95']:+.1f}]")

    assignment = results.get("gallery_assignment_experiment", {})
    offsets = assignment.get("offsets", [])
    arms = sorted(assignment.get("aggregate", {}))
    lines = ["## Issue #155 diagnostic results", "", "### Comparison-MIDI assignments", "",
        "The original four-MIDI result is preserved as the baseline. We also reassigned the common comparison MIDI three times (deterministic offsets 1, 7 and 19) using all 50 existing excerpts. Each search still ranks the same 129 preset candidates, and the query and gallery play different MIDI excerpts. The 6,450 query inputs repeat across assignments. R@1, R@5 and MRR averages combine each query's three assignments; rank-distribution metrics below use all three attempts and keep them together in the crossed bootstrap.", ""]
    lines.extend(["**Example:** Keep `A1` as the input. One round compares it with `A2`, `B2`, `C2`, ...; another round uses `A3`, `B3`, `C3`, ... instead. The labels are illustrative; the actual MIDI choices use the fixed offsets. We are changing what the comparison presets play, not adding new inputs or increasing the number of candidate presets.", ""])
    evaluated_followup_families = {arm.split(":", 1)[0] for arm in assignment.get("aggregate", {})}
    omitted_followup_families = sorted(set(MODEL_REGISTRY) - evaluated_followup_families)
    if omitted_followup_families:
        labels = ", ".join(MODEL_REGISTRY[name]["name"] for name in omitted_followup_families)
        lines.extend([f"The main 50-performance comparison above includes {labels}, but this previously completed three-offset, rank-failure, and gain-normalization follow-up does not. No Basic Pitch result is claimed for these supplementary analyses.", ""])
    if offsets and arms:
        headers = [f"Offset {offset} R@1 [crossed 95% CI] / R@5 / MRR" for offset in offsets]
        lines.extend(["| Encoder arm | " + " | ".join(headers) + " | Three-assignment query average |",
                      "|---|" + "---:|" * (len(headers) + 1)])
        for arm in arms:
            cells = [score_cell(assignment.get("per_assignment", {}).get(str(offset), {}).get(arm, {}))
                     for offset in offsets]
            cells.append(score_cell(assignment.get("aggregate", {}).get(arm, {})))
            lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
        midi_counts = assignment.get("gallery_midi_count_by_offset", {})
        midi_count = min(midi_counts.values()) if midi_counts else "n/e"
        lines.extend(["", f"Each assignment round used {midi_count} gallery MIDI excerpts and held query/preset pairs fixed; the gallery candidate count ranged {assignment.get('gallery_candidate_count', {}).get('min', 'n/e')}–{assignment.get('gallery_candidate_count', {}).get('max', 'n/e')} after render and feature checks. The averaged three-round R@1 change from the preserved four-MIDI baseline was:", ""])
        baseline_pairs = assignment.get("paired_differences", {}).get("assignment_average_minus_four_midi_baseline", {})
        lines.extend(["| Arm | Paired ΔR@1, three-round average minus four-MIDI baseline (95% crossed CI) |",
                      "|---|---:|"])
        for arm in arms:
            lines.append(f"| `{arm}` | {paired_cell(baseline_pairs.get(arm, {}).get('recall_at_1'))} |")
        cir_beats = assignment.get("paired_differences", {}).get("cir_minus_beats_temporal", {})
        lines.append("")
        lines.append("Paired CIR minus BEATs temporal-pyramid Recall@1 by assignment (95% crossed performance × preset-family intervals): " +
                     "; ".join(f"offset {offset}: {paired_cell(cir_beats.get(str(offset), {}).get('recall_at_1'))}"
                                for offset in offsets) +
                     f"; query-averaged: {paired_cell(cir_beats.get('aggregate', {}).get('recall_at_1'))}.")
        top_arms = assignment.get("per_assignment_winner", {})
        lines.append("Observed top arm by round: " + "; ".join(f"offset {offset}: `{top_arms.get(str(offset), 'n/e')}`"
            for offset in offsets) + ". Assignment repeats are correlated, not additional performances.")
        stability = assignment.get("ranking_stability_by_performance", {})
        best_by_family = assignment.get("best_pooling_arm_by_family", {})
        if stability and best_by_family:
            lines.extend(["", "Performance-level rank stability (50 MIDI identities; the aggregate-best pooling arm is held fixed across rounds; 2,000 MIDI-cluster bootstrap replicates):", "",
                "| Encoder family | Arm | " + " | ".join(f"Offset {offset}: R@1; first-place MIDIs; top-rank probability" for offset in offsets) + " | Query-averaged: R@1; first-place MIDIs; top-rank probability |",
                "|---|---|" + "---:|" * (len(offsets) + 1)])
            for family, arm in sorted(best_by_family.items()):
                cells = []
                for offset in offsets:
                    item = stability.get(str(offset), {}).get(arm, {})
                    cells.append("not available" if not item else
                        f"{100 * item.get('mean_performance_metric', 0):.1f}%; {item.get('first_place_performances', 0)}/{item.get('performance_clusters', 0)}; {item.get('bootstrap_top_rank_probability', 0):.1%}")
                aggregate_item = stability.get("aggregate", {}).get(family, {})
                cells.append("not available" if not aggregate_item else
                    f"{100 * aggregate_item.get('mean_performance_metric', 0):.1f}%; {aggregate_item.get('first_place_performances', 0)}/{aggregate_item.get('performance_clusters', 0)}; {aggregate_item.get('bootstrap_top_rank_probability', 0):.1%}")
                lines.append(f"| `{family}` | `{arm}` | " + " | ".join(cells) + " |")
        rank_failure = assignment.get("rank_failure_analysis", {})
        failure_arms = rank_failure.get("arms", {})
        if failure_arms:
            def interval_cell(value: float, ci: dict, *, percent: bool = False) -> str:
                scale, unit = (100, "%") if percent else (1, "")
                return f"{scale * value:.1f}{unit} [{scale * ci.get('lower_95', 0):.1f}{unit}, {scale * ci.get('upper_95', 0):.1f}{unit}]"

            offsets_text = ", ".join(str(offset) for offset in rank_failure.get("assignment_offsets", []))
            exemplar = next(iter(failure_arms.values()))
            lines.extend(["", "### How retrieval fails", "",
                f"Rank distributions and near/far shares use each of the three assignment attempts for {exemplar.get('query_pairs', 0):,} paired preset × MIDI queries ({exemplar.get('individual_retrieval_attempts', 0):,} attempts per encoder). Rank is the average position among tied scores; near misses have expected rank >1 and ≤5, and far failures have expected rank >5. Crossed bootstrap resamples {exemplar.get('performance_clusters', 0)} MIDI performances and {exemplar.get('preset_family_clusters', 0)} preset-family clusters, keeping a query's three assignment attempts together and using the same resamples for every encoder.", "",
                "A correct preset at rank 1 is a success; rank 4 is a near miss; rank 20 is a far failure. These categories apply to each individual search, not an average of several searches.", "",
                "| Encoder arm | Median expected rank per attempt [95% crossed CI] | 90th-percentile expected rank per attempt [95% crossed CI] | Near-miss attempts (expected rank >1 to 5) | Far-failure attempts (expected rank >5) |",
                "|---|---:|---:|---:|---:|"])
            for arm, summary in sorted(failure_arms.items()):
                rank = summary["rank_distribution"]
                lines.append(f"| `{arm}` | {interval_cell(rank['median_expected_rank'], rank['median_95_ci'])} | "
                    f"{interval_cell(rank['p90_expected_rank'], rank['p90_95_ci'])} | "
                    f"{interval_cell(rank['near_miss_fraction'], rank['near_miss_95_ci'], percent=True)} | "
                    f"{interval_cell(rank['far_failure_fraction'], rank['far_failure_95_ci'], percent=True)} |")
            lines.extend(["", "Per-preset accuracy averages Recall@1 across three assignment attempts within each preset × MIDI pair, then across the 50 unique MIDI inputs for each preset, and summarizes the 129 preset accuracies. The difficult-preset threshold is below 50% correct across MIDI inputs; intervals resample MIDI and preset-family clusters.", "",
                "| Encoder arm | Median Per-preset R@1 [95% crossed CI] | 10th-percentile Per-preset R@1 [95% crossed CI] | Presets below 50% R@1 (count / total; share CI) |",
                "|---|---:|---:|---:|"])
            for arm, summary in sorted(failure_arms.items()):
                preset = summary["per_preset_accuracy"]
                lines.append(f"| `{arm}` | {interval_cell(preset['median_recall_at_1'], preset['median_95_ci'], percent=True)} | "
                    f"{interval_cell(preset['p10_recall_at_1'], preset['p10_95_ci'], percent=True)} | "
                    f"{preset['presets_below_50_percent']}/{summary['preset_count']} "
                    f"({interval_cell(preset['below_50_percent_fraction'], preset['below_50_percent_fraction_95_ci'], percent=True)}) |")
        quality_by_arm = assignment.get("render_quality_by_arm", {})
        if quality_by_arm:
            def bucket_cell(groups: dict) -> str:
                cells = []
                for label in ("clean", "warning"):
                    item = groups.get(label)
                    if not item:
                        continue
                    ci = item.get("crossed_95_ci", {})
                    interval = (f" [{100 * ci['lower_95']:.1f}%, {100 * ci['upper_95']:.1f}%]"
                                if "lower_95" in ci and "upper_95" in ci else "")
                    cells.append(f"{label} {100 * item['recall_at_1']:.1f}%{interval} (n={item['queries']:,} unique pairs; m={item.get('assignment_observations', 0):,} assignment observations)")
                return " / ".join(cells) or "not split"
            quality_overlap = next(iter(quality_by_arm.values())).get("correct_match_gallery_quality_overlap", {})
            lines.extend(["", f"For query-WAV quality, clean and warning labels are mutually exclusive. Correct-match gallery-WAV groups overlap: a unique preset × query-MIDI pair is counted in each label seen across the three assignments. The groups overlap for {quality_overlap.get('clean_warning_overlap_queries', 0):,} pairs, and their union contains {quality_overlap.get('queries_in_either_group', 0):,} pairs. Here n is unique pairs and m is assignment observations.",
                "Render-quality results for the expanded comparison-MIDI assignments (each encoder uses the same queries and candidates; assignment rounds are averaged within query before confidence intervals):", "",
                "| Encoder arm | Query WAV R@1: clean / warning | Correct-match gallery WAV R@1: clean / warning (overlap allowed) | Mean warned candidates / fraction of 129 (fraction CI) |",
                "|---|---|---|---:|"])
            for arm, summary in sorted(quality_by_arm.items()):
                warning = summary.get("gallery_warning_fraction", {})
                warning_ci = warning.get("crossed_95_ci", {})
                warning_interval = (f" [{warning_ci['lower_95']:.3f}, {warning_ci['upper_95']:.3f}]"
                                    if "lower_95" in warning_ci and "upper_95" in warning_ci else "")
                lines.append(f"| `{arm}` | {bucket_cell(summary.get('query_render_quality', {}))} | "
                    f"{bucket_cell(summary.get('correct_match_gallery_render_quality', {}))} | "
                    f"{summary.get('mean_warning_candidates', 0):.1f} / {summary.get('mean_warning_fraction', 0):.3f}{warning_interval} |")
            lines.append("A long tail or high float peak is not an automatic failure. These descriptive strata do not establish that a warning caused a ranking change.")
    clean_query = results.get("clean_query_only", {})
    if clean_query.get("arms"):
        lines.extend(["", "### Audio quality checks", "",
            f"**A — Original audio:** the expanded retrieval comparison above uses {len(MODEL_REGISTRY)} encoder families and {results.get('input_dataset', {}).get('accepted_cached_clips', 'n/e'):,} accepted cached clips, with the original four-MIDI results retained.",
            "**B — Clean queries:** we re-summarized the existing cached-feature episode scores for queries whose WAVs had no QA warning. No model inference was repeated for this slice.", "",
            "| Encoder arm | Clean-query R@1 [crossed 95% CI] / R@5 / MRR | Queries |",
            "|---|---:|---:|"])
        for arm, summary in sorted(clean_query["arms"].items()):
            lines.append(f"| `{arm}` | {score_cell(summary)} | {summary.get('queries', 0):,} |")
        clean_query_pair = clean_query.get("paired_cir_minus_beats_temporal", {}).get("recall_at_1")
        if clean_query_pair:
            lines.append("Paired clean-query CIR minus BEATs temporal-pyramid ΔR@1: " + paired_cell(clean_query_pair) + ".")
        strict = results.get("strict_clean_gallery", {})
        if strict.get("status") == "completed":
            lines.extend(["", f"For a stricter check, we recomputed ranks after removing every warned candidate render and retained only clean query/correct-match pairs. This leaves {strict.get('episodes', 0):,} distinct queries ({strict.get('assignment_observations', 0):,} observations across the correlated assignment rounds); each query uses the same clean candidate set for all encoders, with {strict.get('clean_gallery_candidate_count', {}).get('min', 'n/e')}–{strict.get('clean_gallery_candidate_count', {}).get('max', 'n/e')} candidates.", "",
            "**Why an apparently higher clean-only score can be misleading:** If `B2` has a warning and gets removed, `A2` has one fewer wrong preset to beat. Nothing about `A1`, `A2` or their embeddings improved. Searching among 19–87 candidates instead of 129 is simply a different, easier-sized task.", "",
            "| Encoder arm | Clean-gallery R@1 [crossed 95% CI] / R@5 / MRR | Clean queries |",
                "|---|---:|---:|"])
            for arm, summary in sorted(strict.get("arms", {}).items()):
                lines.append(f"| `{arm}` | {score_cell(summary)} | {summary.get('queries', 0):,} |")
            strict_pair = strict.get("paired_cir_minus_beats_temporal", {}).get("recall_at_1")
            if strict_pair:
                lines.append("Paired clean-gallery CIR minus BEATs temporal-pyramid ΔR@1: " + paired_cell(strict_pair) + ".")
        else:
            lines.append(f"A fully warning-free gallery result was not estimable: {strict.get('reason', 'no qualifying episodes')}; no clean-gallery ranking claim is made.")
        lines.append("Clean/warning status is descriptive and may track preset acoustics; neither slice establishes a causal render-quality effect.")
    gain = results.get("gain_sensitivity", {})
    if gain.get("arms"):
        sample = gain.get("sample", {})
        extraction = gain.get("encoder_extraction", {})
        new_families = [name for name, record in extraction.items() if record.get("newly_extracted_features", 0)]
        reused_families = [name for name, record in extraction.items() if record.get("reused_features", 0)]
        if new_families and reused_families:
            extraction_note = (f"New adjusted features were extracted only for {', '.join(MODEL_REGISTRY[name]['name'] for name in new_families)}; "
                               f"existing adjusted-feature caches were reused for the other {len(reused_families)} families.")
        elif new_families:
            extraction_note = "Adjusted features were extracted only for " + ", ".join(
                MODEL_REGISTRY[name]["name"] for name in new_families) + "."
        else:
            extraction_note = "Existing adjusted-feature caches were reused; no encoder inference was repeated."
        lines.extend(["", f"**C — Gain-reduced warning sample:** a deterministic, stratified sample of {sample.get('size', 0)} WAVs was drawn from {results.get('gain_warning_sample_selection', {}).get('accepted_amplitude_warning_wavs', 0):,} accepted amplitude-warning renders. Copies were reduced linearly to a 0.95 peak. {extraction_note} Original cached features supplied the unchanged queries/candidates. This is a sensitivity sample, not a corpus-wide accuracy estimate.", "",
            "**Example:** First reduce the volume of `A1` alone, leaving the comparison recordings unchanged, and check whether the top-ranked preset changes. Separately leave `A1` unchanged and reduce the volume of its correct comparison recording `A2`. In the table, *changed* counts a different top-ranked preset, *fixed* means a wrong answer became right, and *newly wrong* means a right answer became wrong.", "",
            "| Encoder arm | Representation cosine distance (95% crossed CI) | Query R@1 before→after; changed / fixed / newly wrong | Correct gallery candidate R@1 before→after; changed / fixed / newly wrong | Δ target margin ×100 (cosine points; query / gallery) |",
            "|---|---:|---:|---:|---:|"])
        for arm, summary in sorted(gain["arms"].items()):
            distance = summary.get("adjusted_clip_embedding_cosine_distance", {})
            ci = distance.get("crossed_95_ci", {})
            distance_text = f"{distance.get('mean', 0):.4f} [{ci.get('lower_95', 0):.4f}, {ci.get('upper_95', 0):.4f}]"
            role_text = []
            margins = []
            for role in ("query", "correct_gallery_candidate"):
                data = summary["roles"][role]
                before = data["before"]["metrics"]["recall_at_1"]["mean"]
                after = data["after"]["metrics"]["recall_at_1"]["mean"]
                recall_delta = data["paired_after_minus_before"]["recall_at_1"]
                role_text.append(f"{100 * before:.1f}%→{100 * after:.1f}% (Δ {100 * recall_delta['difference']:+.1f} pp [{100 * recall_delta['lower_95']:+.1f}, {100 * recall_delta['upper_95']:+.1f}]); {data['changed_predictions']} / {data['corrected_predictions']} / {data['newly_wrong_predictions']}")
                margin = data["paired_after_minus_before"]["target_margin"]
                margins.append(f"{100 * margin['difference']:+.2f} [{100 * margin['lower_95']:+.2f}, {100 * margin['upper_95']:+.2f}]")
            lines.append(f"| `{arm}` | {distance_text} | {role_text[0]} | {role_text[1]} | {margins[0]} / {margins[1]} |")
        lines.append(f"Mean applied gain was {sample.get('gain_db_mean', 0):.1f} dB; adjusted peaks were at most {sample.get('peak_after_max', 0):.3f}. No limiter, compression, clipping, or other DSP was used.")
    if shortlist:
        selected = shortlist.get("selected_count", len(shortlist.get("candidates", [])))
        valid_controls = shortlist.get("presets_with_valid_control_candidates", {})
        family_counts = shortlist.get("shortlisted_by_control_family", {})
        strength_counts = shortlist.get("screen_strength_counts", {})
        strength_text = ", ".join(f"{strength_counts[label]} {label}"
            for label in ("strong", "borderline", "weak or inconclusive") if label in strength_counts)
        family_summary = ", ".join(f"{count} {family.replace('_', ' ')}" for family, count in sorted(family_counts.items()))
        lines.extend(["", "### Controlled-preset shortlist: awaiting user selection", "",
            f"The encoder-blind scan checked {shortlist.get('scanned_render_qualified_presets', 'n/e')} render-qualified presets. Active control candidates were found in {valid_controls.get('filter', 'n/e')} filter, {valid_controls.get('envelope', 'n/e')} envelope, and {valid_controls.get('routed_lfo', 'n/e')} routed-LFO presets. It selected {selected} presets across four short/long held/phrased excerpts ({shortlist.get('new_vital_renders', 'n/e')} changed/repeat renders; {family_summary}).",
            "**Example:** Take `A1` and change only its filter cutoff to obtain `A′1`. Also render the unchanged preset a second time with a different random seed, `A1*`. Compare the audio difference between `A1` and `A′1` with the ordinary repeat difference between `A1` and `A1*`. If the control change is no larger than the repeat difference, the evidence for a meaningful acoustic change is weak.",
            f"Screen evidence labels: {strength_text}. Strong means the spectral change exceeded independent-seed repeat variation in every context; borderline means all but one context with median change/repeat ratio ≥3; weak or inconclusive means those criteria were not met. These labels do not establish audibility.",
            "The acoustic screen compares each changed source-seed render with an unchanged-preset independent-seed repeat. Presets were sampled by parameter-value quantiles before measuring effect, so encoder scores and unusually large effects did not determine inclusion. Listening WAVs and exact preset/control mappings are in the ignored local `outputs/issue155-controlled-candidates/controlled_candidate_shortlist.private.md`; the shortlist remains for review, and the definitive multi-encoder controlled diagnostic has not been run."])
        candidates = shortlist.get("candidates", [])
        if candidates:
            lines.extend(["", "| Candidate | Control family | Screen evidence | Mean changed-audio distance | Mean repeat distance | Change exceeds repeat |",
                "|---|---|---|---:|---:|---:|"])
            lines.extend(f"| {item['candidate_id']} | {item['control']['family']} | {item.get('screen_strength', 'unclassified')} | {item['mean_change_spectral_distance']:.5f} | {item['mean_repeat_spectral_distance']:.5f} | {item['performances_exceeding_repeat_variation']}/{len(item['midi_measurements'])} |"
                for item in candidates)
            lines.append("Distances use the independent 64-band spectral descriptor; these four-context screens have no human audibility threshold or between-preset uncertainty estimate.")
    return lines


def _issue155_plain_english(results: dict, shortlist: dict | None) -> list[str]:
    assignment = results.get("gallery_assignment_experiment", {})
    aggregate = assignment.get("aggregate", {})
    if not aggregate:
        return []

    def r1(arm: str) -> float | None:
        return aggregate.get(arm, {}).get("metrics", {}).get("recall_at_1", {}).get("mean")

    scored_arms = [arm for arm in aggregate if r1(arm) is not None]
    if not scored_arms:
        return []
    leader = max(scored_arms, key=lambda arm: r1(arm))
    beats = assignment.get("best_pooling_arm_by_family", {}).get("beats_iter3_plus")
    leader_score, beats_score = r1(leader), r1(beats) if beats else None
    leader_name = MODEL_REGISTRY[leader.split(":", 1)[0]]["name"]
    beats_name = MODEL_REGISTRY[beats.split(":", 1)[0]]["name"] if beats else None
    paired = assignment.get("paired_differences", {}).get("cir_minus_beats_temporal", {}).get("aggregate", {}).get("recall_at_1", {})
    paired_text = (f" The paired CIR–BEATs gap was {100 * paired['difference']:+.1f} percentage points "
                   f"[{100 * paired['lower_95']:+.1f}, {100 * paired['upper_95']:+.1f}].") if paired else ""
    assignment_summary = (f"In the separate test of three comparison-MIDI assignments, {leader_name} retrieved the correct preset at rank one "
        f"for {100 * leader_score:.1f}% of queries" + (f", followed by {beats_name} at {100 * beats_score:.1f}%." if beats_score is not None else ".") + paired_text)
    assignment_summary += " Removing a warned gallery candidate makes the search easier: the correct preset has one fewer wrong preset to beat, without changing any embeddings."
    stability = assignment.get("ranking_stability_by_performance", {}).get("aggregate", {}).get(leader.split(":", 1)[0], {})
    if stability:
        assignment_summary += (f" It ranked first for {stability.get('first_place_performances', 0)} of "
            f"{stability.get('performance_clusters', 0)} MIDI performances ({stability.get('bootstrap_top_rank_probability', 0):.1%} top-rank probability).")

    sensitivity = results.get("gain_sensitivity", {}).get("arms", {})
    largest_drop = None
    for arm, summary in sensitivity.items():
        role = summary.get("roles", {}).get("query", {})
        delta = role.get("paired_after_minus_before", {}).get("recall_at_1", {})
        if "difference" in delta and (largest_drop is None or delta["difference"] < largest_drop[1]["difference"]):
            largest_drop = (arm, delta)
    sensitivity_summary = ""
    sample_size = results.get("gain_sensitivity", {}).get("sample", {}).get("size", 0)
    if largest_drop and sample_size:
        arm, delta = largest_drop
        arm_name = MODEL_REGISTRY[arm.split(":", 1)[0]]["name"]
        pooling = "temporal pyramid" if arm.endswith(":temporal_pyramid_8") else "masked mean" if arm.endswith(":masked_mean") else "official clip"
        sensitivity_summary = (f" In the {sample_size}-WAV gain-normalization sample, the largest query-ranking drop was "
            f"{arm_name} {pooling} ({100 * delta['difference']:+.1f} points; paired 95% interval "
            f"[{100 * delta['lower_95']:+.1f}, {100 * delta['upper_95']:+.1f}]); this small, warning-enriched sample does not estimate corpus-wide accuracy.")

    failure_summary = ""
    failure_arms = results.get("gallery_assignment_experiment", {}).get("rank_failure_analysis", {}).get("arms", {})
    cir = failure_arms.get("cir_ast:official_clip")
    if cir and failure_arms:
        worst_far_arm, worst_far = max(failure_arms.items(), key=lambda item: item[1]["rank_distribution"]["far_failure_fraction"])
        worst_preset_arm, worst_preset = max(failure_arms.items(), key=lambda item: item[1]["per_preset_accuracy"]["below_50_percent_fraction"])
        rank = cir["rank_distribution"]
        cir_presets = cir["per_preset_accuracy"]
        worst_far_label = f"{MODEL_REGISTRY[worst_far_arm.split(':', 1)[0]]['name']} {worst_far_arm.split(':', 1)[1]}"
        worst_preset_label = f"{MODEL_REGISTRY[worst_preset_arm.split(':', 1)[0]]['name']} {worst_preset_arm.split(':', 1)[1]}"
        failure_summary = (f" Across individual attempts in the three comparison-MIDI rounds, CIR's median expected rank was "
            f"{rank['median_expected_rank']:.1f}, with {rank['far_failure_fraction']:.1%} of attempts at expected rank above 5; "
            f"{worst_far_label} had the largest far-failure share ({worst_far['rank_distribution']['far_failure_fraction']:.1%}). "
            f"Only {cir_presets['presets_below_50_percent']}/{cir['preset_count']} CIR presets fell below 50% R@1 across the 50 excerpts, "
            f"compared with {worst_preset['per_preset_accuracy']['presets_below_50_percent']}/{worst_preset['preset_count']} for {worst_preset_label}.")

    candidate_summary = ""
    if shortlist:
        candidate_count = shortlist.get("selected_count", len(shortlist.get("candidates", [])))
        strength_counts = shortlist.get("screen_strength_counts", {})
        strengths = ", ".join(f"{strength_counts[label]} {label}"
            for label in ("strong", "borderline", "weak or inconclusive") if label in strength_counts)
        weak_candidates = [item for item in shortlist.get("candidates", []) if item.get("screen_strength") == "weak or inconclusive"]
        weak_text = "; weak/inconclusive examples: " + ", ".join(
            f"{item['candidate_id']} ({item['performances_exceeding_repeat_variation']}/{len(item['midi_measurements'])} contexts)"
            for item in weak_candidates) if weak_candidates else ""
        candidate_summary = (f" The automated screen prepared {candidate_count} preset candidates with before/after/repeat listening examples; "
            f"its consistency screen labels {strengths}{weak_text}; the definitive controlled diagnostic is waiting for the user's review and selection.")
    return ["### Plain-English readout", "", assignment_summary + failure_summary + sensitivity_summary + candidate_summary, ""]


def _issue155_render_quality(episode_metrics_path: Path) -> dict:
    if not episode_metrics_path.is_file():
        return {}
    from .core import two_way_cluster_bootstrap

    rows_by_arm: dict[str, list[dict]] = defaultdict(list)
    with episode_metrics_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get("gallery_assignment_offset") is not None:
                    rows_by_arm[row["arm"]].append(row)
    if not rows_by_arm:
        return {}
    arm_summaries = {}
    reference_keys = None
    for arm, rows in sorted(rows_by_arm.items()):
        by_offset: dict[int, set[tuple[str, str]]] = defaultdict(set)
        for row in rows:
            if row.get("gallery_candidate_count") is None:
                raise ValueError(f"issue #155 render-quality metrics are incomplete for {arm}")
            by_offset[int(row["gallery_assignment_offset"])].add((row["target_state_id"], row["query_midi_id"]))
        if len({frozenset(keys) for keys in by_offset.values()}) != 1:
            raise ValueError(f"issue #155 gallery assignments are not paired for {arm}")
        query_keys = next(iter(by_offset.values()))
        if reference_keys is None:
            reference_keys = query_keys
        elif query_keys != reference_keys:
            raise ValueError("issue #155 render-quality rows do not share one query cohort across encoders")

        def summarize_axis(axis: str) -> dict:
            by_label: dict[str, dict[tuple[str, str], list[dict]]] = defaultdict(lambda: defaultdict(list))
            for row in rows:
                label = row.get(axis)
                if label in {"clean", "warning", "excluded_or_unknown"}:
                    by_label[label][(row["target_state_id"], row["query_midi_id"])].append(row)
            result = {}
            for label, queries in by_label.items():
                collapsed = []
                for (state_id, midi_id), items in sorted(queries.items()):
                    exemplar = items[0]
                    collapsed.append({"episode_id": f"{state_id}|{midi_id}",
                        "target_family_id": exemplar["target_family_id"], "query_midi_id": midi_id,
                        **{metric: sum(float(item[metric]) for item in items) / len(items)
                           for metric in ("recall_at_1", "recall_at_5", "mrr")}})
                enough_clusters = (len(collapsed) >= 50 and len({row["query_midi_id"] for row in collapsed}) >= 3 and
                                   len({row["target_family_id"] for row in collapsed}) >= 3)
                ci = two_way_cluster_bootstrap(collapsed, metric="recall_at_1") if enough_clusters else {}
                result[label] = {"queries": len(collapsed), "assignment_observations": sum(map(len, queries.values())),
                    "recall_at_1": sum(row["recall_at_1"] for row in collapsed) / len(collapsed),
                    "crossed_95_ci": ({key: ci[key] for key in ("lower_95", "upper_95", "performance_clusters", "family_clusters", "replicates")}
                        if ci else {})}
            return result

        warning_fraction_rows = []
        fractions_by_query: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            fractions_by_query[(row["target_state_id"], row["query_midi_id"])].append(row)
        for (state_id, midi_id), items in sorted(fractions_by_query.items()):
            exemplar = items[0]
            warning_fraction_rows.append({"episode_id": f"{state_id}|{midi_id}",
                "target_family_id": exemplar["target_family_id"], "query_midi_id": midi_id,
                "warning_fraction": sum(float(item["gallery_warning_candidate_fraction"]) for item in items) / len(items)})
        enough_warning_clusters = (len(warning_fraction_rows) >= 50 and
            len({row["query_midi_id"] for row in warning_fraction_rows}) >= 3 and
            len({row["target_family_id"] for row in warning_fraction_rows}) >= 3)
        warning_ci = two_way_cluster_bootstrap(warning_fraction_rows, metric="warning_fraction") if enough_warning_clusters else {}
        gallery_counts = [int(row["gallery_warning_candidate_count"]) for row in rows]
        gallery_sizes = [int(row["gallery_candidate_count"]) for row in rows]
        arm_summaries[arm] = {"assignment_observations": len(rows), "queries": len(query_keys),
            "query_midi_ids": len({midi for _, midi in query_keys}),
            "query_render_quality": summarize_axis("query_render_quality"),
            "correct_match_gallery_render_quality": summarize_axis("correct_match_gallery_render_quality"),
            "correct_match_gallery_quality_overlap": {},
            "mean_warning_candidates": sum(gallery_counts) / len(gallery_counts),
            "mean_warning_fraction": sum(float(row["gallery_warning_candidate_fraction"]) for row in rows) / len(rows),
            "gallery_warning_fraction": {"estimate": warning_ci.get("estimate"),
                "crossed_95_ci": ({key: warning_ci[key] for key in ("lower_95", "upper_95", "performance_clusters", "family_clusters", "replicates")}
                    if warning_ci else {})},
            "gallery_candidate_count": {"min": min(gallery_sizes), "max": max(gallery_sizes)}}
        correct_gallery_groups = defaultdict(set)
        correct_gallery_observations = Counter()
        for row in rows:
            label = row.get("correct_match_gallery_render_quality")
            if label in {"clean", "warning"}:
                key = (row["target_state_id"], row["query_midi_id"])
                correct_gallery_groups[label].add(key)
                correct_gallery_observations[label] += 1
        clean_targets, warning_targets = correct_gallery_groups["clean"], correct_gallery_groups["warning"]
        arm_summaries[arm]["correct_match_gallery_quality_overlap"] = {
            "clean_warning_overlap_queries": len(clean_targets & warning_targets),
            "queries_in_either_group": len(clean_targets | warning_targets),
            "assignment_observations_by_label": dict(correct_gallery_observations)}
        for label, bucket in arm_summaries[arm]["correct_match_gallery_render_quality"].items():
            bucket["assignment_observations"] = correct_gallery_observations[label]
    return arm_summaries


def _collapse_issue155_rank_rows(rows_by_arm: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], list[int]]:
    collapsed_by_arm, reference_keys = {}, None
    offsets = sorted({int(row["gallery_assignment_offset"]) for rows in rows_by_arm.values() for row in rows})
    if len(offsets) < 2:
        raise ValueError("issue #155 rank analysis requires at least two repeated assignment rounds")
    expected_offsets = set(offsets)
    for arm, rows in sorted(rows_by_arm.items()):
        grouped: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
        for row in rows:
            key = (row["target_state_id"], row["query_midi_id"])
            offset = int(row["gallery_assignment_offset"])
            if offset in grouped[key]:
                raise ValueError(f"duplicate issue #155 rank row for {arm}/{key}/{offset}")
            grouped[key][offset] = row
        if any(set(rounds) != expected_offsets for rounds in grouped.values()):
            raise ValueError(f"issue #155 rank rows do not have the same assignment offsets for {arm}")
        if reference_keys is None:
            reference_keys = set(grouped)
        elif set(grouped) != reference_keys:
            raise ValueError("issue #155 rank rows do not share one query cohort across encoders")
        collapsed = []
        for (state_id, midi_id), rounds in sorted(grouped.items()):
            items = [rounds[offset] for offset in offsets]
            if len({item["target_family_id"] for item in items}) != 1:
                raise ValueError(f"target preset family changed across issue #155 rounds for {arm}/{state_id}")
            collapsed.append({"target_state_id": state_id, "target_family_id": items[0]["target_family_id"],
                "query_midi_id": midi_id,
                **{metric: sum(float(item[metric]) for item in items) / len(items)
                   for metric in ("rank_expected", "recall_at_1", "recall_at_5")}})
        collapsed_by_arm[arm] = collapsed
    return collapsed_by_arm, offsets


def _issue155_rank_statistics(rows: list[dict], *, attempts: list[dict], replicates: int = 2000, seed: int = 155) -> dict:
    import numpy as np

    if not rows or not attempts:
        raise ValueError("issue #155 rank statistics require query rows and individual attempts")

    def quantile(values, probability: float) -> float:
        ordered = sorted(float(value) for value in values)
        position = (len(ordered) - 1) * probability
        low, high = math.floor(position), math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    def interval(draws: list[float]) -> dict:
        ordered = sorted(draws)
        return {"lower_95": ordered[int(0.025 * (len(ordered) - 1))],
                "upper_95": ordered[int(0.975 * (len(ordered) - 1))], "replicates": len(ordered)}

    ranks = np.asarray([row["rank_expected"] for row in attempts], dtype=np.float64)
    near = ((ranks > 1) & (ranks <= 5)).astype(np.float64)
    far = (ranks > 5).astype(np.float64)
    performances = sorted({row["query_midi_id"] for row in rows})
    families = sorted({row["target_family_id"] for row in rows})
    state_ids = sorted({row["target_state_id"] for row in rows})
    performance_index = {key: index for index, key in enumerate(performances)}
    family_index = {key: index for index, key in enumerate(families)}
    state_index = {key: index for index, key in enumerate(state_ids)}
    attempt_performance_index = np.asarray([performance_index[row["query_midi_id"]] for row in attempts], dtype=np.intp)
    attempt_family_index = np.asarray([family_index[row["target_family_id"]] for row in attempts], dtype=np.intp)
    rank_order = np.argsort(ranks, kind="stable")
    sorted_ranks = ranks[rank_order]
    preset_by_midi = np.full((len(state_ids), len(performances)), np.nan, dtype=np.float64)
    state_families = {}
    for row in rows:
        state = state_index[row["target_state_id"]]
        performance = performance_index[row["query_midi_id"]]
        if not np.isnan(preset_by_midi[state, performance]):
            raise ValueError("issue #155 rank analysis expected one collapsed value per preset and MIDI")
        preset_by_midi[state, performance] = float(row["recall_at_1"])
        state_families[row["target_state_id"]] = row["target_family_id"]
    if np.isnan(preset_by_midi).any():
        raise ValueError("issue #155 rank analysis has missing preset × performance cells")
    state_family_index = np.asarray([family_index[state_families[state]] for state in state_ids], dtype=np.intp)
    observed_preset_accuracy = preset_by_midi.mean(axis=1)
    observed_difficult = int(np.sum(observed_preset_accuracy < 0.5))
    rng = random.Random(seed)
    draws = {key: [] for key in ("rank_median", "rank_p90", "near_miss", "far_failure",
        "preset_median", "preset_p10", "difficult_fraction")}

    def weighted_quantile(sorted_values, cumulative_weights, total: int, probability: float) -> float:
        position = (total - 1) * probability
        low, high = math.floor(position), math.ceil(position)
        low_index = int(np.searchsorted(cumulative_weights, low + 1, side="left"))
        high_index = int(np.searchsorted(cumulative_weights, high + 1, side="left"))
        return float(sorted_values[low_index] + (sorted_values[high_index] - sorted_values[low_index]) * (position - low))

    for _ in range(replicates):
        performance_counts = np.bincount([rng.randrange(len(performances)) for _ in performances], minlength=len(performances))
        family_counts = np.bincount([rng.randrange(len(families)) for _ in families], minlength=len(families))
        attempt_weights = performance_counts[attempt_performance_index] * family_counts[attempt_family_index]
        total = int(attempt_weights.sum())
        if not total:
            continue
        cumulative = np.cumsum(attempt_weights[rank_order])
        draws["rank_median"].append(weighted_quantile(sorted_ranks, cumulative, total, 0.5))
        draws["rank_p90"].append(weighted_quantile(sorted_ranks, cumulative, total, 0.9))
        draws["near_miss"].append(float(np.dot(attempt_weights, near) / total))
        draws["far_failure"].append(float(np.dot(attempt_weights, far) / total))

        preset_accuracy = (preset_by_midi @ performance_counts) / max(1, int(performance_counts.sum()))
        preset_weights = family_counts[state_family_index]
        preset_total = int(preset_weights.sum())
        preset_order = np.argsort(preset_accuracy, kind="stable")
        preset_cumulative = np.cumsum(preset_weights[preset_order])
        sorted_preset_accuracy = preset_accuracy[preset_order]
        draws["preset_median"].append(weighted_quantile(sorted_preset_accuracy, preset_cumulative, preset_total, 0.5))
        draws["preset_p10"].append(weighted_quantile(sorted_preset_accuracy, preset_cumulative, preset_total, 0.1))
        draws["difficult_fraction"].append(float(np.dot(preset_weights, preset_accuracy < 0.5) / preset_total))

    return {"query_pairs": len(rows), "individual_retrieval_attempts": len(attempts),
        "assignment_rounds_per_query_pair": len(attempts) // len(rows),
        "preset_count": len(state_ids), "performance_clusters": len(performances),
        "preset_family_clusters": len(families), "bootstrap_seed": seed,
        "rank_expected_values": ranks.tolist(), "per_preset_recall_at_1_values": observed_preset_accuracy.tolist(),
        "rank_expected_tie_handling": "average rank position across tied candidate scores",
        "rank_distribution": {"median_expected_rank": quantile(ranks, 0.5),
            "median_95_ci": interval(draws["rank_median"]), "p90_expected_rank": quantile(ranks, 0.9),
            "p90_95_ci": interval(draws["rank_p90"]),
            "near_miss_fraction": float(near.mean()), "near_miss_95_ci": interval(draws["near_miss"]),
            "far_failure_fraction": float(far.mean()), "far_failure_95_ci": interval(draws["far_failure"])},
        "per_preset_accuracy": {"midi_performances_per_preset": len(performances),
            "median_recall_at_1": quantile(observed_preset_accuracy, 0.5),
            "median_95_ci": interval(draws["preset_median"]),
            "p10_recall_at_1": quantile(observed_preset_accuracy, 0.1),
            "p10_95_ci": interval(draws["preset_p10"]),
            "presets_below_50_percent": observed_difficult,
            "below_50_percent_fraction": observed_difficult / len(state_ids),
            "below_50_percent_fraction_95_ci": interval(draws["difficult_fraction"])}}


def _issue155_rank_failure_analysis(episode_metrics_path: Path, *, replicates: int = 2000) -> dict:
    if not episode_metrics_path.is_file():
        return {}
    rows_by_arm: dict[str, list[dict]] = defaultdict(list)
    with episode_metrics_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get("gallery_assignment_offset") is not None:
                    required = ("arm", "target_state_id", "target_family_id", "query_midi_id",
                                "gallery_assignment_offset", "rank_expected", "recall_at_1", "recall_at_5")
                    if any(key not in row for key in required):
                        raise ValueError("issue #155 rank metrics are incomplete")
                    rows_by_arm[row["arm"]].append(row)
    if not rows_by_arm:
        return {"status": "not estimable", "reason": "no repeated comparison-MIDI assignment rows", "arms": {}}
    collapsed, offsets = _collapse_issue155_rank_rows(rows_by_arm)
    for arm, rows in collapsed.items():
        preset_count = len({row["target_state_id"] for row in rows})
        performance_count = len({row["query_midi_id"] for row in rows})
        if len(rows) != preset_count * performance_count:
            return {"status": "not estimable", "reason": f"incomplete preset × MIDI grid for {arm}",
                    "assignment_offsets": offsets, "arms": {}}
    return {"assignment_offsets": offsets, "rank_metrics_use_individual_assignment_attempts": True,
        "cluster_bootstrap_keeps_assignment_rounds_together": True,
        "confidence_interval": "95% crossed bootstrap over MIDI performance and preset-family clusters",
        "shared_bootstrap_resamples_across_arms": True,
        "arms": {arm: _issue155_rank_statistics(rows, attempts=rows_by_arm[arm], replicates=replicates, seed=155)
                 for arm, rows in sorted(collapsed.items())}}


def _write_issue155_visualizations(output_dir: Path, evaluation: dict | None,
                                   validation_results: dict | None, shortlist: dict | None) -> dict:
    rank_arms = ((validation_results or {}).get("gallery_assignment_experiment", {})
        .get("rank_failure_analysis", {}).get("arms", {}))
    model_results = (evaluation or {}).get("model_results", {})
    gain = (validation_results or {}).get("gain_sensitivity", {})
    candidates = (shortlist or {}).get("candidates", [])
    if not model_results or not rank_arms or not gain.get("arms") or not candidates:
        return {}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    output_dir.mkdir(parents=True, exist_ok=True)
    family_labels = {"cir_ast": "CIR", "beats_iter3_plus": "BEATs", "clap_music": "CLAP",
        "dac_44khz": "DAC", "encodec_24khz": "EnCodec", "muq_mulan": "MuQ-MuLan"}
    variant_labels = {"official_clip": "official", "masked_mean": "mean", "temporal_pyramid_8": "temporal"}
    line_styles = {"official_clip": "-", "masked_mean": "--", "temporal_pyramid_8": ":"}
    family_ids = list(dict.fromkeys(arm.split(":", 1)[0] for arm in sorted(model_results)))
    palette = plt.get_cmap("tab10")
    colors = {family: palette(index % 10) for index, family in enumerate(family_ids)}

    def label(arm: str) -> str:
        family, variant = arm.split(":", 1)
        return f"{family_labels.get(family, family)} · {variant_labels.get(variant, variant)}"

    def save(name: str, figure) -> str:
        figure.savefig(output_dir / name, dpi=170, bbox_inches="tight", facecolor="white",
            metadata={"Title": name, "Software": f"Matplotlib {matplotlib.__version__}"})
        plt.close(figure)
        return f"figures/{name}"

    figures = {"matplotlib_version": matplotlib.__version__}
    heatmap_rows = []
    for arm, outcomes in model_results.items():
        primary = outcomes.get("cross_midi_common_gallery", {})
        if not primary.get("episodes", 0):
            continue
        heatmap_rows.append((arm, [100 * float(primary.get(key, 0)) for key in (
            "recall_at_1", "recall_at_5", "mrr", "same_midi_negative_win_rate", "acoustic_hard_negative_win_rate")]))
    heatmap_rows.sort(key=lambda item: (-item[1][0], item[0]))
    if heatmap_rows:
        with plt.rc_context({"font.size": 8, "axes.titlesize": 12, "axes.labelsize": 9}):
            figure, axis = plt.subplots(figsize=(10, max(4.8, .42 * len(heatmap_rows) + 1.5)))
            values = np.asarray([row[1] for row in heatmap_rows])
            image = axis.imshow(values, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
            axis.set_xticks(range(5), ("R@1", "R@5", "MRR", "Same-MIDI\nhard negative", "Acoustic\nhard negative"))
            axis.set_yticks(range(len(heatmap_rows)), [label(row[0]) for row in heatmap_rows])
            axis.set_title("Encoder × metric comparison")
            for row_index, row in enumerate(values):
                for column_index, value in enumerate(row):
                    axis.text(column_index, row_index, f"{value:.1f}%", ha="center", va="center",
                        color="white" if value >= 58 else "#17212b", fontsize=8)
            colorbar = figure.colorbar(image, ax=axis, pad=.02)
            colorbar.set_label("Correct-preset or hard-negative win rate")
            colorbar.ax.yaxis.set_major_formatter(PercentFormatter(100))
            figures["encoder_metric_heatmap"] = save("issue155_encoder_metric_heatmap.png", figure)

    available_rank_arms = [(arm, stats) for arm, stats in sorted(rank_arms.items())
        if stats.get("rank_expected_values")]
    if available_rank_arms:
        with plt.rc_context({"font.size": 8, "axes.titlesize": 12, "axes.labelsize": 9}):
            figure, axis = plt.subplots(figsize=(10, 6))
            for arm, stats in available_rank_arms:
                ranks = np.sort(np.asarray(stats["rank_expected_values"], dtype=np.float64))
                cumulative = np.arange(1, len(ranks) + 1, dtype=np.float64) / len(ranks)
                family, variant = arm.split(":", 1)
                axis.step(ranks, cumulative, where="post", color=colors.get(family),
                    linestyle=line_styles.get(variant, "-"), linewidth=1.5, label=label(arm))
            axis.axvline(5, color="#606060", linestyle="--", linewidth=1, label="Rank 5")
            axis.set_xscale("log")
            axis.set_xlim(1, 129)
            axis.set_xticks((1, 2, 5, 10, 20, 50, 100), ("1", "2", "5", "10", "20", "50", "100"))
            axis.set_ylim(0, 1)
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.set_xlabel("Expected correct-preset rank (log scale)")
            axis.set_ylabel("Share of individual attempts at or below this rank")
            axis.set_title("Correct-preset rank distribution across comparison-MIDI attempts")
            axis.grid(axis="y", alpha=.22)
            axis.legend(ncol=3, fontsize=7, loc="lower right")
            figures["rank_distribution"] = save("issue155_correct_preset_rank_cdf.png", figure)

    preset_arms = [(arm, stats) for arm, stats in sorted(rank_arms.items())
        if stats.get("per_preset_recall_at_1_values")]
    if preset_arms:
        with plt.rc_context({"font.size": 8, "axes.titlesize": 12, "axes.labelsize": 9}):
            figure, axis = plt.subplots(figsize=(10, 5.8))
            positions = np.arange(1, len(preset_arms) + 1)
            distributions = [stats["per_preset_recall_at_1_values"] for _, stats in preset_arms]
            boxes = axis.boxplot(distributions, positions=positions, patch_artist=True, showfliers=False,
                medianprops={"color": "#15202b", "linewidth": 1.5}, widths=.62)
            for box, (arm, _) in zip(boxes["boxes"], preset_arms, strict=True):
                box.set_facecolor(colors.get(arm.split(":", 1)[0]))
                box.set_alpha(.75)
            axis.axhline(.5, color="#7c2d12", linestyle="--", linewidth=1.2, label="50% R@1 threshold")
            axis.set_xticks(positions, [label(arm) for arm, _ in preset_arms], rotation=35, ha="right")
            axis.set_ylim(0, 1)
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.set_ylabel("Recall@1 per preset")
            axis.set_title("Accuracy distribution across 129 preset candidates")
            axis.grid(axis="y", alpha=.22)
            axis.legend(loc="upper right", fontsize=8)
            figures["per_preset_accuracy"] = save("issue155_per_preset_accuracy_boxplots.png", figure)

    normalization_arms = [(arm, result) for arm, result in sorted(gain.get("arms", {}).items())
        if result.get("roles")]
    if normalization_arms:
        role_specs = (("query", "Normalize query audio"),
            ("correct_gallery_candidate", "Normalize correct-match gallery audio"))
        with plt.rc_context({"font.size": 8, "axes.titlesize": 11, "axes.labelsize": 9}):
            figure, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
            legend_handles = []
            for arm, result in normalization_arms:
                family, variant = arm.split(":", 1)
                for role_index, (role, title) in enumerate(role_specs):
                    role_result = result["roles"].get(role)
                    if not role_result:
                        continue
                    point_values, lower_errors, upper_errors = [], [], []
                    for condition in ("before", "after"):
                        metric = role_result[condition]["metrics"]["recall_at_1"]
                        point = float(metric["mean"])
                        ci = metric.get("crossed_95_ci", {})
                        point_values.append(point)
                        lower_errors.append(max(0.0, point - float(ci.get("lower_95", point))))
                        upper_errors.append(max(0.0, float(ci.get("upper_95", point)) - point))
                    line, = axes[role_index].plot((0, 1), point_values, color=colors.get(family),
                        linestyle=line_styles.get(variant, "-"), marker="o", markersize=4, linewidth=1.2,
                        label=label(arm))
                    axes[role_index].errorbar((0, 1), point_values, yerr=(lower_errors, upper_errors),
                        fmt="none", ecolor=colors.get(family), capsize=2, linewidth=.8)
                    if role_index == 0:
                        legend_handles.append(line)
            for axis, (_, title) in zip(axes, role_specs, strict=True):
                axis.set_title(title)
                axis.set_xticks((0, 1), ("Before", "After"))
                axis.set_ylim(0, 1)
                axis.yaxis.set_major_formatter(PercentFormatter(1))
                axis.grid(axis="y", alpha=.22)
            axes[0].set_ylabel("Recall@1 in the targeted sample")
            figure.suptitle(f"Before and after amplitude normalization ({gain.get('sample', {}).get('size', 0)} warning WAVs)", y=1.02)
            figure.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(.5, -.04))
            figures["normalization_comparison"] = save("issue155_gain_normalization_before_after.png", figure)

    if candidates:
        with plt.rc_context({"font.size": 8, "axes.titlesize": 12, "axes.labelsize": 9}):
            ordered_candidates = sorted(candidates, key=lambda item: item["candidate_id"])
            figure, axis = plt.subplots(figsize=(9, 6))
            change_color, repeat_color = "#176b87", "#777777"
            for position, candidate in enumerate(ordered_candidates):
                measurements = candidate.get("midi_measurements", [])
                if not measurements:
                    continue
                for field, color, vertical_offset in (
                        ("change_spectral_distance", change_color, .13),
                        ("repeat_spectral_distance", repeat_color, -.13)):
                    values = np.asarray([float(row[field]) for row in measurements], dtype=np.float64)
                    mean = float(values.mean())
                    axis.errorbar(mean, position + vertical_offset,
                        xerr=[[mean - float(values.min())], [float(values.max()) - mean]],
                        fmt="o", color=color, ecolor=color, capsize=2, markersize=4,
                        label=("Control change" if position == 0 and field == "change_spectral_distance" else
                               "Independent-seed repeat" if position == 0 else None))
            axis.set_yticks(range(len(ordered_candidates)), [
                f"{item['candidate_id']} · {item.get('screen_strength', 'unclassified')}" for item in ordered_candidates])
            axis.invert_yaxis()
            axis.set_xlabel("64-band spectral distance (1 − cosine)")
            axis.set_title("Controlled-preset acoustic change versus render variation")
            axis.grid(axis="x", alpha=.22)
            axis.legend(loc="lower right", fontsize=8)
            figures["controlled_acoustic_differences"] = save("issue155_controlled_acoustic_differences.png", figure)

    return figures


def _issue155_visual_summary_lines(figures: dict | None) -> list[str]:
    if not figures:
        return []
    descriptions = (
        ("encoder_metric_heatmap", "Encoder × metric heatmap",
         "All evaluated arms on the 50-performance cohort. Same-MIDI negatives test against every other preset; acoustic negatives test against only five preselected sounds. Compare encoders within each metric, not the two hard-negative percentages against each other."),
        ("rank_distribution", "Correct-preset rank distribution",
         "CDF of individual attempts from the three MIDI assignments; repeated attempts remain grouped for the table's crossed uncertainty estimates."),
        ("per_preset_accuracy", "Per-preset accuracy distributions",
         "Each box summarizes preset Recall@1 after averaging assignment rounds per MIDI; the dashed line marks the 50% difficult-preset threshold."),
        ("normalization_comparison", "Before/after normalization comparison",
         "Recall@1 for the targeted 100-WAV amplitude-warning sample, split by query versus correct-match gallery role; this sample is not a corpus-wide estimate."),
        ("controlled_acoustic_differences", "Controlled-preset acoustic differences",
         "Mean change and independent-seed repeat distances with four-context min–max ranges; this is a spectral screen, not an audibility judgment."))
    lines = ["### Visual summaries", ""]
    for key, title, caption in descriptions:
        if key in figures:
            filename = Path(figures[key]).as_posix()
            lines.extend([f"**{title}**", "", f"![{title}]({filename})", "", caption, ""])
    return lines


def _retrieval_explainer_lines(candidate_count: int) -> list[str]:
    if candidate_count < 2:
        raise ValueError("retrieval explanation requires at least two candidate presets")
    other_count = candidate_count - 1
    return [
        "### How to read these tests",
        "",
        "Call a recording `A1` when **preset A plays MIDI 1**. `A2` is the same preset playing MIDI 2; `B1` is a **different preset playing MIDI 1**. Higher cosine similarity means the encoder considers two recordings more alike.",
        "",
        f"**Main retrieval (R@1, R@5, MRR).** Start with `A1`. Choose the correct preset from all {candidate_count} candidate presets playing **MIDI 2**: `A2` is the correct answer; `B2`, `C2`, etc. are wrong. R@1 is how often A2 ranks first; R@5 is how often it ranks in the top five; MRR rewards higher ranks (rank 1 scores 1, rank 2 scores 1/2, rank 5 scores 1/5). Ties are scored fractionally.",
        "",
        "**Common versus mixed comparison MIDI.** Starting from `A1`, the common-MIDI test compares `A2`, `B2`, `C2`, ...: all 129 choices play MIDI 2. The mixed-MIDI test might compare `A2`, `B3`, `C4`, ...: different presets can play different MIDI, but none plays the input's MIDI 1. Both are 129-choice searches. The hard-negative checks below are separate: they deliberately test wrong presets playing MIDI 1.",
        "",
        f"**Same-MIDI hard negatives.** This is a *separate* check: does `similarity(A1, A2) > similarity(A1, B1)` for **every one of the {other_count} other presets** B? A2 has different MIDI but the right preset; every B1 has the wrong preset but exactly the same MIDI as the input. One B1 beating or tying A2 counts as a failure.",
        "",
        f"**Acoustic top-five hard negatives.** Use the same A1-versus-A2 comparison, but test against **five** B1 candidates rather than all {other_count}. Choose those five in advance by looking for the wrong-preset recordings with the most similar **coarse frequency-energy profiles** to A1 (64 frequency bands). This selection uses the audio waveform, not the encoder being evaluated. \"Acoustic\" does **not** mean a person judged the sounds similar.",
        "",
        "**Known-related-preset negatives.** Choose B because the source data identifies it as a related version of A, rather than selecting it by audio similarity. No eligible related-preset pairs were present in this cohort, so this check has no score.",
        "",
        f"**Do not compare the two hard-negative percentages directly.** Beating five selected wrong presets is a different test from beating all {other_count}. Compare *encoders within the same column*, not the two columns as if they were equally difficult. Neither hard-negative score is the main {candidate_count}-preset retrieval accuracy.",
        "",
        "**How to interpret uncertainty.** The 6,450 queries reuse the same 129 presets and 50 MIDI performances. They are not 6,450 independent instruments or performances; the confidence intervals account for repeated preset families and MIDI identities.",
        "",
    ]


def _reader_summary(current_dataset: dict, current_eval: dict | None, comparison_dataset: dict | None,
                    comparison_eval: dict | None, controls: dict, inventory_name: str,
                    availability: dict[str, dict], validation_results: dict | None = None,
                    candidate_shortlist: dict | None = None, visuals: dict | None = None) -> list[str]:
    lines = ["## Results and recommendation", ""]
    model_results = (current_eval or {}).get("model_results", {})
    current_ranking = [arm for arm, outcomes in sorted(model_results.items(),
                       key=lambda item: (-item[1].get("cross_midi_common_gallery", {}).get("recall_at_1", 0.0), item[0]))
                       if outcomes.get("cross_midi_common_gallery", {}).get("episodes", 0)]
    older_results = (comparison_eval or {}).get("model_results", {})
    older_ranking = [arm for arm, outcomes in sorted(older_results.items(),
                     key=lambda item: (-item[1].get("cross_midi_common_gallery", {}).get("recall_at_1", 0.0), item[0]))
                     if outcomes.get("cross_midi_common_gallery", {}).get("episodes", 0)]
    if current_ranking:
        leader = current_ranking[0]
        leader_result = model_results[leader]["cross_midi_common_gallery"]
        model_name, variant = leader.split(":", 1)
        gallery = leader_result.get("gallery_size", [])
        gallery_size = gallery[0] if gallery else current_dataset.get("presets_verified", "unknown")
        performance_source = current_dataset.get("performance_source_split") or current_dataset.get("source_split")
        performance_detail = f"{current_dataset.get('unique_midi_excerpts', 0)} MIDI performances from `{performance_source}`"
        if performance_source != current_dataset.get("source_split"):
            performance_detail += ", distinct from the earlier validation-source MIDI cohort"
        finding = (f"**Primary expanded-cohort result:** {MODEL_REGISTRY[model_name]['name']} ({variant}) ranked the correct preset first in "
                   f"{100 * leader_result.get('recall_at_1', 0):.1f}% of cross-MIDI queries "
                   f"(n={leader_result.get('episodes', 0):,} queries, {gallery_size} candidate presets; {performance_detail}). "
                   f"This ranks frozen representations across {len(MODEL_REGISTRY)} model families; it is retrieval evidence, not a reconstruction result.")
        lines.extend([finding, "",
                      "Presentation: [Frozen encoder retrieval (PDF)](presentations/frozen_encoder_retrieval.pdf).", ""])
        lines.extend(_retrieval_explainer_lines(int(gallery_size)))
    lines.extend(_summary_cohort_lines("Expanded held-out performance cohort", current_dataset, current_eval))
    if validation_results:
        lines.extend(["The 81.4% headline is the primary common-gallery assignment. Issue #155 reuses those same queries and preset targets, rotates only the gallery-MIDI assignment in three deterministic rounds, and reports that paired average separately.", ""])
        lines.extend(_issue155_plain_english(validation_results, candidate_shortlist))
        lines.extend(_issue155_visual_summary_lines(visuals))
        lines.extend(_issue155_followup_lines(validation_results, candidate_shortlist))
    if comparison_dataset and comparison_eval is not None:
        lines.extend(_summary_cohort_lines("Earlier 12-MIDI validation-source cohort", comparison_dataset, comparison_eval))
    measured_families = len({arm.split(":", 1)[0] for arm in current_ranking})
    if current_ranking:
        leader = current_ranking[0]
        leader_family = leader.split(":", 1)[0]
        runner_up = next((arm for arm in current_ranking if arm.split(":", 1)[0] != leader_family), None)
        recommendation = f"Observed expanded-cohort lead: `{leader}`."
        leader_stability = (current_eval or {}).get("ranking_stability_by_performance", {}).get(
            "cross_midi_common_gallery", {}).get(leader, {})
        if leader_stability:
            recommendation += (f" It ranked first on {leader_stability['first_place_performances']}/"
                f"{leader_stability['performance_clusters']} individual MIDI performances and led in "
                f"{leader_stability['bootstrap_top_rank_probability']:.1%} of 2,000 performance-cluster resamples.")
        if older_ranking:
            recommendation += (f" Earlier 12-MIDI validation-source cohort leader: `{older_ranking[0]}`; "
                "the earlier and expanded cohorts use different MIDI pools, so this is contextual rather than a paired stability comparison.")
        if runner_up:
            paired = (current_eval or {}).get("paired_cluster_bootstrap_differences", {})
            pair, direction = f"{leader} - {runner_up}", 1.0
            if pair not in paired:
                pair, direction = f"{runner_up} - {leader}", -1.0
            interval = paired.get(pair, {}).get("cross_midi_common_gallery", {}).get(
                "performance_and_family", {}).get("recall_at_1", {})
            if interval:
                lower, upper = direction * interval.get("lower_95", 0), direction * interval.get("upper_95", 0)
                if direction < 0:
                    lower, upper = upper, lower
                difference = direction * interval.get("difference", 0)
                recommendation += f" Crossed-cluster ΔR@1 versus `{runner_up}` is {100 * difference:+.1f} percentage points [{100 * lower:+.1f}, {100 * upper:+.1f}]."
                if lower <= 0 <= upper:
                    recommendation += " The interval spans zero, so keep both as candidates for the later 14-output reconstruction screen."
                else:
                    recommendation += f" Use `{leader}` as the lead and `{runner_up}` as a comparator for the later 14-output reconstruction screen."
                if leader_family == "cir_ast":
                    recommendation += " Keep CLAP Music as an established reference in that screen."
        lines.extend([recommendation, ""])
        leader_metrics = model_results[leader]["cross_midi_common_gallery"]
        leader_strata = leader_metrics.get("strata", {})
        axis_names = {"pitch_register_gap_band": "query-to-gallery pitch/register gap",
                      "query_articulation": "performance articulation", "query_polyphony": "query polyphony"}
        contrasts = []
        min_episodes = max(20, int(leader_metrics.get("episodes", 0) * 0.01))
        for axis, label in axis_names.items():
            buckets = []
            for bucket_name, bucket in leader_strata.get(axis, {}).items():
                ci = bucket.get("performance_cluster_bootstrap", {})
                if bucket.get("episodes", 0) >= min_episodes and ci.get("performance_clusters", 0) >= 3:
                    buckets.append((bucket_name, bucket, ci))
            if len(buckets) > 1:
                low = min(buckets, key=lambda item: item[1]["recall_at_1"])
                high = max(buckets, key=lambda item: item[1]["recall_at_1"])
                def describe(item: tuple) -> str:
                    name, bucket, ci = item
                    return (f"`{name}` {_percent(bucket, 'recall_at_1')} "
                            f"[{100 * ci['lower_95']:.1f}%, {100 * ci['upper_95']:.1f}%], {bucket['episodes']} queries / "
                            f"{ci['performance_clusters']} MIDI IDs")
                contrasts.append(f"{label}: lowest {describe(low)}; highest {describe(high)}")
        if contrasts:
            lines.extend(["**Largest descriptive performance strata for the expanded-cohort lead:**", ""])
            lines.extend(f"- {contrast}." for contrast in contrasts)
            lines.extend(["These are exploratory subgroup contrasts; the intervals resample MIDI IDs and do not imply a causal effect.", ""])
    common_episodes = (current_eval or {}).get("coverage_by_condition", {}).get(
        "cross_midi_common_gallery", {}).get("common_feature_qualified_episodes_across_arms", 0)
    family_coverage = (f"All {len(MODEL_REGISTRY)} encoder families have expanded-cohort retrieval results on {common_episodes:,} identical common-gallery episodes per family."
                       if measured_families == len(MODEL_REGISTRY) else
                       f"{measured_families} of {len(MODEL_REGISTRY)} encoder families have expanded-cohort retrieval results on {common_episodes:,} identical common-gallery episodes; incomplete families are listed below.")
    lines.extend([family_coverage,
                  f"No Vital controls are predicted or trained in this representation-only benchmark (0 outputs; all source controls stay fixed). The linked [{inventory_name}](./{inventory_name}) is the canonical per-field inventory."])
    if current_dataset.get("performance_identity_overlap_excluded") is not None:
        lines.append(f"The expanded performance IDs came from the permitted `{current_dataset.get('performance_source_split')}` source split; {current_dataset.get('performance_identity_overlap_excluded', 0)} exact MIDI identities overlapping the original cohort were excluded. This is held out from the original MIDI IDs but remains within PresetShare source MIDI, not an external live-performance sample. The score is frozen retrieval on newly rendered audio, not reconstruction quality.")
    if current_eval:
        lines.append("The acoustic top-5 negatives are selected by an independent spectral descriptor, not human-verified perceptual labels. Query and correct-match gallery quality are reported separately; warned candidates are counted within each full gallery.")
        if not any(result.get("cross_midi_common_gallery", {}).get("known_near_neighbor_negative_win_episodes", 0)
                   for result in current_eval.get("model_results", {}).values()):
            lines.append("The expanded gallery contained no known multi-state lineage or alias neighbors, so the known source-lineage negative check had zero eligible negatives.")
    unscored = []
    measured = {arm.split(":", 1)[0] for arm in current_ranking}
    measured_earlier = {arm.split(":", 1)[0] for arm in older_ranking}
    for model_name, spec in MODEL_REGISTRY.items():
        if model_name not in measured:
            candidate = availability.get(model_name, {})
            reasons = "; ".join(candidate.get("unavailable_reasons", [])) or "no measured results"
            source_link = f" ([source]({spec['source']}))" if spec.get("source") else ""
            license_link = f" ([license status]({spec['license_source']}))" if spec.get("license_source") else ""
            if model_name in measured_earlier:
                reasons = "expanded-cohort results are incomplete; original-cohort results are reported above"
            unscored.append(f"- {spec['name']}{source_link}: {reasons}; {candidate.get('license', spec['license'])}{license_link}.")
    if unscored:
        lines.extend(["**Unscored candidates and blockers:**", "", *unscored, ""])
    lines.extend(["", "**Scope:** retrieval only. No current OBRUXO CNN baseline, shared-model training, source-label warm start, direct patch optimization, audio-loss training, or reconstruction head was run.", ""])
    return lines


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


def _execution_status_lines(dataset_info: dict, evaluation: dict, local_dataset: Path,
                            arms: dict, locked_models: dict) -> list[str]:
    source_split = dataset_info.get("performance_source_split") or dataset_info.get("source_split", "unknown")
    lines = ["## Execution status", "",
        f"The retrieval gallery contains {dataset_info.get('presets_verified', 0)} verified `{dataset_info.get('source_split', 'unknown')}` presets. The permitted `{source_split}` MIDI pool contains {dataset_info.get('unique_midi_excerpts', 0)} performances; reserved test sources were not used."]
    render_path = local_dataset / "render_run.latest.json"
    if render_path.is_file():
        render = json.loads(render_path.read_text(encoding="utf-8"))
        warning_counts = render.get("qa_warning_counts", {})
        warning_text = ", ".join(f"{count:,} {name.replace('_', ' ')}" for name, count in sorted(warning_counts.items())) or "none recorded"
        lines.append(f"Vital render QA accepted {render.get('accepted', 0):,} of {render.get('requested', 0):,} planned requests. Accepted warning flags: {warning_text}; warned renders remain in the main comparison and are analyzed separately above.")
    coverage = evaluation.get("coverage_by_condition", {}).get("cross_midi_common_gallery", {})
    episodes = coverage.get("common_feature_qualified_episodes_across_arms", 0)
    gallery = evaluation.get("gallery_candidate_states_render_qualified", 0)
    families = {arm.split(":", 1)[0] for arm, outcomes in evaluation.get("model_results", {}).items()
                if outcomes.get("cross_midi_common_gallery", {}).get("episodes", 0)}
    lines.append(f"The primary results use {episodes:,} paired queries per measured arm and {gallery} common preset candidates across {len(families)} encoder families. Retrieval results are frozen-feature cosine rankings; they do not measure reconstruction quality.")
    coverage_by_condition = evaluation.get("coverage_by_condition", {})
    if coverage_by_condition:
        lines.extend(["", "### Render and feature coverage", "",
            "| Condition | Planned | Render-QA accepted | Excluded | Common across measured encoders |",
            "|---|---:|---:|---:|---:|"])
        for condition, row in sorted(coverage_by_condition.items()):
            lines.append(f"| {condition} | {row.get('planned_episodes', 0):,} | {row.get('qa_qualified_episodes', 0):,} | {row.get('excluded_episodes', 0):,} | {row.get('common_feature_qualified_episodes_across_arms', 0):,} |")
    lines.extend(["", "### Encoder checkpoints and representations", "",
        "| Encoder family | Checkpoint and SHA-256 | Representation and output size | Terms / execution |",
        "|---|---|---|---|"])
    for model_name, spec in MODEL_REGISTRY.items():
        variants = ("official_clip",) if spec["kind"] == "clip" else ("masked_mean", "temporal_pyramid_8")
        run = next((arms.get(f"{model_name}:{variant}") for variant in variants if arms.get(f"{model_name}:{variant}")), None)
        measured = all(evaluation.get("model_results", {}).get(f"{model_name}:{variant}", {}).get(
            "cross_midi_common_gallery", {}).get("episodes", 0) for variant in variants)
        tap = spec["tap"]
        if spec["kind"] == "clip":
            representation = f"{tap}; {spec['dimension']:,}-D"
        else:
            representation = f"masked mean {spec['dimension']:,}-D; temporal pyramid {16 * spec['dimension']:,}-D"
        checkpoint = spec["checkpoint"]
        source_link = f" ([source]({spec['source']}))" if spec.get("source") else ""
        digest = f"; SHA-256 `{run.get('checkpoint_sha256', 'not recorded')}`" if run else ""
        lock_entry = locked_models.get(model_name, {})
        terms = lock_entry.get("license", spec["license"])
        if lock_entry.get("license_status") == "research-use-unverified":
            terms = "terms unresolved; evaluated for authorized local non-commercial research"
        status = "executed" if measured else "not run or incomplete; Recall@1 not run for every listed arm"
        lines.append(f"| {spec['name']} | `{checkpoint}`{source_link}{digest} | {representation} | {terms}; {status} |")
    lines.extend(["", "Checkpoint files remain local and are not included in the repository. The BEATs and CIR checkpoint terms remain unresolved for any use beyond this authorized research evaluation."])
    return lines


def generate_report(destination: Path, inventory: Path, *, dataset_dir: Path | None = None,
                    checkpoint_root: Path | None = None, lock_path: Path | None = None,
                    execution_note: str | None = None, controlled_summary_path: Path | None = None,
                    comparison_dataset_dir: Path | None = None,
                    issue155_results_path: Path | None = None,
                    candidate_shortlist_path: Path | None = None) -> dict:
    inventory_count, groups = _inventory_summary(inventory)
    controlled = json.loads(controlled_summary_path.read_text(encoding="utf-8")) if controlled_summary_path else None
    issue155_results = json.loads(issue155_results_path.read_text(encoding="utf-8")) if issue155_results_path else None
    if issue155_results is not None:
        quality_path = issue155_results_path.parent / "gallery_assignment_episode_metrics.private.jsonl"
        assignment_results = issue155_results.setdefault("gallery_assignment_experiment", {})
        assignment_results["render_quality_by_arm"] = _issue155_render_quality(quality_path)
        assignment_results["rank_failure_analysis"] = _issue155_rank_failure_analysis(quality_path)
    candidate_shortlist = json.loads(candidate_shortlist_path.read_text(encoding="utf-8")) if candidate_shortlist_path else None
    controlled_inventory_path = destination.with_name("vital_control_inventory_controlled.jsonl") if controlled else None
    controlled_inventory = (build_controlled_inventory(inventory, controlled_inventory_path,
        controlled["controlled_fields"]) if controlled else None)
    arms = {}
    evaluation = None
    local_dataset = None
    dataset_info = {}
    comparison_eval = comparison_info = None
    if dataset_dir and dataset_dir.is_dir():
        local_dataset = dataset_dir
        summary_path = dataset_dir / "dataset_summary.private.json"
        if summary_path.is_file():
            dataset_info = json.loads(summary_path.read_text(encoding="utf-8"))
            dataset_info["performance_profile"] = _performance_profile(
                dataset_dir, dataset_info.get("performance_source_split") or dataset_info.get("source_split"))
        evaluation_path = dataset_dir / "evaluation.private.json"
        if evaluation_path.is_file():
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        for model_dir in (dataset_dir / "features").glob("*") if (dataset_dir / "features").is_dir() else []:
            run_path = model_dir / "extraction.latest.json"
            if run_path.is_file():
                run = json.loads(run_path.read_text(encoding="utf-8"))
                for variant in run.get("dimensions_by_variant", {}):
                    arms[f"{model_dir.name}:{variant}"] = run
    if comparison_dataset_dir and comparison_dataset_dir.is_dir():
        summary_path = comparison_dataset_dir / "dataset_summary.private.json"
        evaluation_path = comparison_dataset_dir / "evaluation.private.json"
        if summary_path.is_file() and evaluation_path.is_file():
            comparison_info = json.loads(summary_path.read_text(encoding="utf-8"))
            comparison_eval = json.loads(evaluation_path.read_text(encoding="utf-8"))
            comparison_info["performance_profile"] = _performance_profile(
                comparison_dataset_dir, comparison_info.get("performance_source_split") or comparison_info.get("source_split"))
    availability = {row["id"]: row for row in available_candidates(checkpoint_root, lock_path)}
    locked_models = {}
    if lock_path and lock_path.is_file():
        try:
            locked_models = load_checkpoint_lock(lock_path).get("models", {})
        except (OSError, ValueError, json.JSONDecodeError):
            locked_models = {}
    lines = ["# Frozen audio encoder retrieval benchmark", "",
             "Issues: [#152](https://github.com/noisedoctrine/obruxo/issues/152) baseline and [#155](https://github.com/noisedoctrine/obruxo/issues/155) follow-up. This report compares frozen pretrained representations on Vital presets and existing MIDI performances. No synthesis controls are predicted, tuned, or optimized.", "",
             "## Execution status", ""]
    if local_dataset:
        dataset_summary_path = local_dataset / "dataset_summary.private.json"
        if dataset_summary_path.is_file():
            dataset = json.loads(dataset_summary_path.read_text(encoding="utf-8"))
            lines.extend([f"The frozen `{dataset.get('source_split')}` preset gallery contains {dataset.get('presets_verified', 0)} verified original presets and is paired with {dataset.get('unique_midi_excerpts', 0)} compatible MIDI excerpts from `{dataset.get('performance_source_split') or dataset.get('source_split')}`; the plan covers {dataset.get('episodes', 0)} retrieval episodes and {dataset.get('render_requests', 0)} preset/MIDI renders.",
                          f"Render profile: `{dataset.get('renderer_id')}`. Selected renderer capabilities: `{json.dumps(dataset.get('renderer_capabilities', {}), sort_keys=True)}`."])
            lines.append(f"Source pairing and MIDI compatibility audit: {json.dumps(dataset.get('exclusions', {}), sort_keys=True)}. These counts reflect the selected existing split and direct PresetShare preset/MIDI pairing rules; source WAV quality is not an eligibility condition.")
            performance_split = dataset.get("performance_source_split") or dataset.get("source_split")
            lines.append(f"The {dataset.get('unique_midi_excerpts', 0)} compatible MIDI excerpts from the permitted `{performance_split}` performance-source split are cross-rendered on all {dataset.get('presets_verified', 0)} admitted `{dataset.get('source_split')}` preset states. These are new Vital renders, not claims of original PresetShare preset/MIDI capture pairs.")
            audit = _local_input_status()
            if audit["snapshot_complete"]:
                census = audit["snapshot_summary"]
                split_counts = census.get("eligible_states_by_split", {})
                lines.append(f"The existing source audit counted {census.get('discovered_files', 0):,} files and {census.get('static_eligible_states', 0):,} statically eligible states ({split_counts.get('validation', 0):,} validation, {split_counts.get('train', 0):,} train, {split_counts.get('test', 0):,} test). The candidate gallery is validation-only; MIDI performance sources come from `{performance_split}`. Static eligibility is not native render eligibility.")
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
        lines.append(f"Controlled perturbation diagnostic: {controlled.get('render_requests', 0)} real Vital renders, {controlled.get('accepted_renders', 0)} QA accepted, {len(controlled.get('models', {}))} pretrained encoders; its MIDI-cluster intervals are exploratory.")
    else:
        lines.append("Controlled patch-perturbation diagnostic: not run; no controlled score is claimed.")
    missing_runtime = sorted({package for candidate in availability.values() for package, value in candidate.get("dependencies", {}).items()
                              if value.get("version") == "missing"})
    lines.extend(["", f"Runtime preflight: Python `{sys.version.split()[0]}`; missing encoder runtime packages: {', '.join(f'`{name}`' for name in missing_runtime) or 'none'}. Checkpoint root supplied: **{'yes' if checkpoint_root else 'no'}**. Vital plugin environment variable configured: **{'yes' if os.environ.get('OBRUXO_VITAL_PLUGIN') else 'no'}**.",
                  "The existing [corpus audit report](../../../training/reports/CORPUS_INPUT_RESULTS.md) records prior static eligibility findings. Those audit counts are not native render eligibility. The separate [PresetShare audit](../../basic_pitch/README.md) has its own pairing rules and counts.", "",
                  "The default command selects the corpus-qualified full-preset expression renderer identity, including the established curve-input, ignored-empty-map, and native scalar readback policies. It supports note/polyphony, pitch-bend, channel-pressure, CC 1/64/74, release-velocity and multiple channels; tempo changes remain unsupported and are counted as exclusions. Unsupported events are rejected rather than silently removed. Musical spans are complete-note excerpts of 2–8 seconds plus a fixed 2-second release tail; model input is mono and 4–10 seconds, downmixed by channel mean and polyphase-resampled. No shared gain normalization is applied; EnCodec retains its own checkpoint-configured RMS normalization when enabled.", "",
                  "Render and feature stages are independent, resumable commands. Vital remains a single-worker in-process renderer; an asynchronous manifest reader feeds a bounded queue of request batches. Render keys include exact preset/MIDI/excerpt/renderer/policy identities. Feature keys include audio hash, checkpoint digest/revision, loader revision, sample rate, preprocessing, extraction tap, pooling, and FP32 precision. Weights are never downloaded by the run commands. Full paths, identifiers, WAVs, caches, checkpoints, and per-episode results belong in ignored local outputs.", "",
                  "No reconstruction head, source-label warm start, shared-model training, direct patch optimization, or audio-loss training is part of this experiment. Retrieval results do not establish reconstruction quality.", ""])
    unverified_terms = [model_name for model_name, entry in locked_models.items()
                        if entry.get("license_status") == "research-use-unverified"]
    if unverified_terms:
        labels = ", ".join(MODEL_REGISTRY[name]["name"] for name in unverified_terms)
        lines.append(f"Checkpoint-term note: {labels} were evaluated under authorized local non-commercial research use; their terms remain unresolved. Checkpoint weights are kept local and are not redistributed. This records the research run only and does not establish future deployment or distribution rights.")

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
            source_link = f" ([upstream checkpoint]({spec['source']}))" if spec.get("source") else ""
            lines.extend([f"### {display}", "",
                          f"Checkpoint: `{spec['checkpoint']}`{source_link}; requested sample rate: {spec['sample_rate'] or 'model-defined'} Hz; representation: {spec['tap']}; output size: {dimension}-D. Status: **{status}**.", "",
                          "| Measure | Result or coverage | Output count | Update / supervision path |",
                          "|---|---:|---:|---|"])
            if run:
                license_text = locked_models.get(model_name, {}).get("license", spec["license"])
                runtime_versions = locked_models.get(model_name, {}).get("runtime_versions", {})
                artifact_hashes = locked_models.get(model_name, {}).get("artifact_sha256", {})
                artifact_text = ("; artifact file SHA-256 " + ", ".join(f"`{name}` `{digest}`" for name, digest in sorted(artifact_hashes.items()))) if artifact_hashes else ""
                lines.append(f"| Checkpoint provenance / license | SHA-256 `{run.get('checkpoint_sha256', 'unknown')}`{artifact_text}; source `{run.get('source_revision', 'unknown')}`; loader `{run.get('loader_revision', 'unknown')}`; runtime `{json.dumps(runtime_versions, sort_keys=True)}`; {license_text} | — | exact local lock; frozen inference |")
                local_configs = locked_models.get(model_name, {}).get("local_configs", {})
                if local_configs:
                    config_provenance = "; ".join(f"{role} `{asset.get('repository_id')}@{asset.get('revision')}` (bundle SHA-256 `{asset.get('sha256')}`)" for role, asset in sorted(local_configs.items()))
                    lines.append(f"| Offline architecture configs | {config_provenance} | — | local config-only initialization; no transitive pretrained weights loaded |")
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
            if primary and primary.get("strata"):
                for axis, buckets in sorted(primary["strata"].items()):
                    for label, metrics_by_stratum in sorted(buckets.items()):
                        interval = metrics_by_stratum.get("two_way_cluster_bootstrap", {})
                        interval_text = (f"; 2-way 95% CI [{interval['lower_95']:.4f}, {interval['upper_95']:.4f}]"
                                         if "lower_95" in interval and "upper_95" in interval else "")
                        lines.append(f"| Stratum: {axis} = {label} | Recall@1 {_metric(metrics_by_stratum, 'recall_at_1')}{interval_text}; Recall@5 {_metric(metrics_by_stratum, 'recall_at_5')}; MRR {_metric(metrics_by_stratum, 'mrr')} | {metrics_by_stratum.get('episodes', 0)} episodes | frozen encoder; crossed performance/family bootstrap |")
            if not run:
                license_text = locked_models.get(model_name, {}).get("license", spec["license"])
                status = (f"Unavailable: {'; '.join(unavailable)}. Checkpoint/license: {license_text}."
                          if unavailable else f"Checkpoint/license: {license_text}.")
                lines.extend(["", status, ""])
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
        midi_description = _controlled_midi_profile(controlled.get("midi_strata", []))
        validation = controlled.get("inherited_source_validation_diagnostics", {})
        lfo_control = controlled.get("controlled_fields", {}).get("lfo_frequency_shift", {})
        modulation_source = lfo_control.get("route_source", "unknown")
        audio_changes = controlled.get("audio_change_diagnostics", {})
        lfo_audio = audio_changes.get("perturbations", {}).get("lfo_frequency_shift", {})
        lfo_distance = lfo_audio.get("distance", {})
        lfo_exceeds = lfo_audio.get("fraction_exceeding_same_midi_repeat_distance", {})
        audio_sensitivity = (f"The {modulation_source}-routed LFO frequency perturbation had audio-descriptor distance "
            f"{lfo_distance.get('estimate', 'not measured')} [95% MIDI-cluster CI "
            f"{lfo_distance.get('lower_95', 'not measured')}, {lfo_distance.get('upper_95', 'not measured')}]; "
            f"it exceeded identical-preset seed-repeat distance on {lfo_exceeds.get('estimate', 'not measured')} "
            f"of excerpts. This tests a routed LFO rate change, not phase reconstruction or full temporal-control recovery.")
        controlled_lead_summary = ""
        current_results = (evaluation or {}).get("model_results", {})
        if current_results:
            leader_arm = max(current_results, key=lambda arm: current_results[arm].get(
                "cross_midi_common_gallery", {}).get("recall_at_1", 0))
            leader_model, leader_tap = leader_arm.split(":", 1)
            leader_pool = controlled.get("models", {}).get(leader_model, {}).get("pool_results", {}).get(leader_tap, {})
            sensitivity = []
            for version, label in (("filter_cutoff_shift", "filter cutoff"), ("amp_attack_shift", "amplitude attack"),
                                   ("lfo_frequency_shift", "routed LFO frequency")):
                score = leader_pool.get("per_target_variant", {}).get(version, {}).get("cross_midi_retrieval", {})
                ci = score.get("confidence_intervals_by_midi", {}).get("recall_at_1", {})
                if score:
                    sensitivity.append(f"{label} {_percent(score, 'recall_at_1')} "
                        f"[{100 * ci.get('lower_95', 0):.1f}%, {100 * ci.get('upper_95', 0):.1f}%]")
            if sensitivity:
                controlled_lead_summary = (f"The expanded-cohort lead `{leader_arm}` scored among the four controlled versions as "
                    f"{'; '.join(sensitivity)} (95% MIDI-cluster intervals, 50 performances; one preset).")
        controlled_intro = ["## Controlled preset perturbations", "",
            "**What is compared here?** For one preset, `A1` is the original recording, `A′1` changes only one Vital control while keeping MIDI 1 fixed, and `A1*` is the unchanged preset rendered with a different random seed. We ask whether the control change affects the sound more than an ordinary repeat does. This is a different question from choosing preset A among 129 presets.",
            f"This secondary diagnostic uses one original validation preset, three single-control perturbations, {len(controlled.get('midi_strata', []))} MIDI performances, and source/independently randomized Vital renders for each version. It reports cross-MIDI identity retrieval, same-MIDI retrieval against a different random seed, and perturbation distance relative to the identical-preset repeat distance. Every query uses a source-seeded render and its gallery uses independent-seed renders; exact query audio is excluded from every gallery. MIDI-cluster intervals summarize variation for this one preset and do not generalize across presets or controls.",
            ]
        if controlled_lead_summary:
            controlled_intro.append(controlled_lead_summary)
        controlled_intro.extend(["R@5 is a four-version ceiling and is not interpreted. The LFO arm changes one selected routed frequency field; no phase or modulation route is optimized.",
            f"MIDI excerpt properties: {midi_description or 'not recorded'}. The original source carried {validation.get('count', 'unreported')} inherited Vital schema diagnostics ({json.dumps(validation.get('codes', {}), sort_keys=True)}); perturbations introduced {validation.get('new_diagnostics_after_perturbation', 'unreported')} new diagnostics. Velocity ranges are recorded, but velocity response was not isolated. The independent 64-bin log-frequency audio descriptor has no human audibility threshold. {audio_sensitivity}",
            f"The linked [{controlled_inventory_path.name}](./{controlled_inventory_path.name}) records every exact Vital control/instance for each version ({controlled_inventory['records_per_version']} rows per version). No model predicts or optimizes these controls; each perturbation is manually specified, with every other source value held fixed.", ""])
        lines.extend(controlled_intro)
        labels = {"original": "Original source preset", "filter_cutoff_shift": "Filter cutoff perturbation",
                  "amp_attack_shift": "Amplitude envelope attack perturbation",
                  "lfo_frequency_shift": "Routed LFO frequency perturbation"}
        for version in controlled["variants"]:
            version_summary = controlled_inventory["versions"][version]
            lines.extend([f"### {labels.get(version, version)}", "",
                          "| Measure | Result or coverage | Output count | Update method / training signal |",
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
                    same = pool.get("same_midi_independent_seed_control", {})
                    repeat = pool.get("identical_preset_repeat_distance", {})
                    dimension = (MODEL_REGISTRY[model_name]["dimension"] if tap in {"official_clip", "masked_mean"}
                                 else 16 * MODEL_REGISTRY[model_name]["dimension"])
                    prefix = f"{MODEL_REGISTRY[model_name]['name']} — {tap}"
                    cross_ci = cross.get("confidence_intervals_by_midi", {}).get("recall_at_1", {})
                    same_ci = same.get("confidence_intervals_by_midi", {}).get("recall_at_1", {})
                    perturb = score.get("perturbation_distance", {})
                    exceeds = score.get("perturbation_exceeds_repeat_distance_fraction", {})
                    lines.append(f"| {prefix} cross-MIDI variant retrieval | R@1 {_metric(cross, 'recall_at_1')} [{cross_ci.get('lower_95', 'n/e')}, {cross_ci.get('upper_95', 'n/e')}]; R@5 {_metric(cross, 'recall_at_5')}; MRR {_metric(cross, 'mrr')} ({cross.get('queries', 0)} queries) | {dimension}-D | frozen backbone; cosine retrieval |")
                    lines.append(f"| {prefix} same-MIDI independent-seed control | R@1 {_metric(same, 'recall_at_1')} [{same_ci.get('lower_95', 'n/e')}, {same_ci.get('upper_95', 'n/e')}]; MRR {_metric(same, 'mrr')} | {dimension}-D | distinct source/independent seed renders; query excluded |")
                    if version == "original":
                        lines.append(f"| {prefix} identical-preset seed-repeat distance | {_metric(repeat, 'estimate')} [{repeat.get('lower_95', 'n/e')}, {repeat.get('upper_95', 'n/e')}] | {dimension}-D | same source preset/MIDI; varied Vital seed |")
                    else:
                        lines.append(f"| {prefix} perturbation distance / fraction exceeding repeat | {_metric(perturb, 'estimate')} [{perturb.get('lower_95', 'n/e')}, {perturb.get('upper_95', 'n/e')}] / {_metric(exceeds, 'estimate')} [{exceeds.get('lower_95', 'n/e')}, {exceeds.get('upper_95', 'n/e')}] | {dimension}-D | distance from original at same MIDI/seed; bootstrap by MIDI |")
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
        if baselines.get("status") == "not_applicable":
            midi_text = f"not applicable: {baselines.get('reason', 'source split mismatch')}"
        else:
            midi_text = "not run" if not baseline else " / ".join(_metric(baseline, key) for key in ("recall_at_1", "recall_at_5", "mrr"))
        random_text = "not run" if not random_result else " / ".join(_metric(random_result, key) for key in ("recall_at_1", "recall_at_5", "mrr"))
        interval = "not applicable" if baselines.get("status") == "not_applicable" else _interval(baseline, "recall_at_1").strip() or "not run"
        lines.append(f"| {label} | {midi_text} | {interval} | {random_text} |")
    lines.extend(["", "## Evaluation protocol", "",
                  f"The primary condition uses one common, existing gallery MIDI per episode and a query render of the same target preset on another existing MIDI. Query/gallery MIDI IDs differ; selection rotates deterministically. Every admitted `{dataset_info.get('source_split', 'validation')}` preset is paired with all {dataset_info.get('unique_midi_excerpts', 0)} compatible MIDI performances from `{dataset_info.get('performance_source_split') or dataset_info.get('source_split', 'validation')}`, making multiple cross-renders per preset. Gallery candidates are all admitted source states. The mixed-gallery condition uses a deterministic candidate-specific MIDI assignment and is reported separately. A same-MIDI different-patch negative check compares each cross-MIDI positive against other patches rendered on the query MIDI. Known frozen lineage groups are reported as family-level retrieval; confidence intervals resample family, performance identity, and crossed family/performance units.", "",
                  "Metrics use cosine similarity over FP32 cached vectors; candidate sets and query episodes are paired across available arms. Acoustic hard negatives are selected independently of the tested encoders using cosine similarity over normalized 64-bin log-frequency power spectra of the target query and same-MIDI candidate renders; each encoder’s cross-MIDI positive is then compared against the five nearest such negatives on that query MIDI. The report includes exact-state and family-level Recall@1, Recall@5, MRR, family-cluster 95% bootstrap intervals, MIDI-only metadata and random baselines, and strata for polyphony, articulation, expression, source-lineage group size, pitch gap, duration, and note density. Sequence outputs include a valid-frame mean and an 8-bin ordered mean/std pyramid. No pooling or similarity setting is selected on reserved test labels.", "",
                  "This benchmark uses new Vital renders only; source WAVs are not used as model inputs or mixed with renders.", ""])
    document = "\n".join(lines)
    findings_start = document.find("## Comparative findings\n")
    if findings_start >= 0:
        section_ends = [position for marker in ("## Controlled preset perturbations\n", "## Baselines\n")
                        if (position := document.find(marker, findings_start)) >= 0]
        findings_end = min(section_ends) if section_ends else len(document)
        document = document[:findings_start] + document[findings_end:]
    research_start = min((position for marker in ("## Controlled preset perturbations\n", "## Baselines\n")
                          if (position := document.find(marker)) >= 0), default=-1)
    research_end = document.find("## Evaluation protocol\n", research_start) if research_start >= 0 else -1
    research_details = ""
    if research_start >= 0 and research_end >= 0:
        research_details = document[research_start:research_end].rstrip() + "\n\n"
        document = document[:research_start] + document[research_end:]
    visuals = _write_issue155_visualizations(destination.parent / "figures", evaluation,
        issue155_results, candidate_shortlist)
    reader_summary = _reader_summary(dataset_info, evaluation, comparison_info, comparison_eval, groups,
                                     inventory.name, availability, issue155_results, candidate_shortlist, visuals)
    status_start = document.find("## Execution status\n")
    if status_start >= 0:
        existing_execution_status = document[status_start:]
        protocol_start = existing_execution_status.find("## Evaluation protocol\n")
        evaluation_protocol = existing_execution_status[protocol_start:] if protocol_start >= 0 else ""
        document = document[:status_start] + "\n".join(reader_summary) + "\n" + research_details
        if evaluation and local_dataset:
            status_lines = _execution_status_lines(dataset_info, evaluation, local_dataset, arms, locked_models)
            if visuals:
                status_lines.extend(["", f"Report figures were generated with Matplotlib `{visuals['matplotlib_version']}` from cached evaluation and shortlist results."])
            document += evaluation_protocol + "\n" + "\n".join(status_lines) + "\n"
        else:
            document += existing_execution_status
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return {"models": len(MODEL_REGISTRY), "arms": sum(1 if spec["kind"] == "clip" else 2 for spec in MODEL_REGISTRY.values()),
            "control_records": inventory_count, "path": str(destination)}
