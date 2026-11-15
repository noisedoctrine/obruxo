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
             "| Frozen representation (output) | R@1 (%; preset-family / MIDI / crossed 95% CI) | R@5 (%) | MRR (%) | Same-MIDI negative win (%; preset-family / MIDI CI; queries) | Acoustic top-5 negative win (%; preset-family / MIDI CI; queries) | Query R@1 clean / warning (%) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
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
        dimension = spec["dimension"] if variant in {"official_clip", "masked_mean"} or spec["kind"] == "clip" else 16 * spec["dimension"]
        family_ci = primary.get("confidence_intervals", {}).get("recall_at_1", {})
        performance_ci = primary.get("performance_cluster_bootstrap", {}).get("recall_at_1", {})
        crossed_ci = primary.get("two_way_cluster_bootstrap", {}).get("recall_at_1", {})
        interval = lambda value: (f"[{100 * value['lower_95']:.1f}%, {100 * value['upper_95']:.1f}%]"
                                  if value and "lower_95" in value and "upper_95" in value else "not estimated")
        ci_text = (f"{_percent(primary, 'recall_at_1')} preset-family {interval(family_ci)} / MIDI {interval(performance_ci)} / "
                   f"2-way {interval(crossed_ci)}")
        strata = primary.get("strata", {}).get("query_render_quality", {})
        quality_cell = lambda key: (f"{_percent(strata[key], 'recall_at_1')} "
            f"{interval(strata[key].get('performance_cluster_bootstrap', {}))} "
            f"(q={strata[key].get('episodes', 0)}; "
            f"MIDI IDs={strata[key].get('performance_cluster_bootstrap', {}).get('performance_clusters', 0)})"
            if key in strata else "—")
        clean, warning = quality_cell("clean"), quality_cell("warning")
        def negative_cell(metric: str, count_key: str) -> str:
            rate_key = f"{metric}_rate"
            family_ci = primary.get("confidence_intervals", {}).get(metric, {})
            performance_ci = primary.get("diagnostic_performance_cluster_bootstrap", {}).get(metric, {})
            return (f"{_percent(primary, rate_key)} family {interval(family_ci)} / MIDI {interval(performance_ci)} "
                    f"(q={primary.get(count_key, 0)})")

        same = negative_cell("same_midi_negative_win", "same_midi_negative_win_episodes")
        acoustic = negative_cell("acoustic_hard_negative_win", "acoustic_hard_negative_win_episodes")
        lines.append(f"| {spec['name']} — {variant} ({dimension}-D) | {ci_text} | {_percent(primary, 'recall_at_5')} | {_percent(primary, 'mrr')} | {same} | {acoustic} | {clean} / {warning} |")
    if not ranked:
        lines.append("| No arm completed | not run | not run | not run | not run | not run | not run |")
    random_baseline = (evaluation or {}).get("midi_only_and_random_baselines", {}).get("random", {}).get(
        "cross_midi_common_gallery", {})
    if random_baseline:
        lines.append(f"| Random chance | {_percent(random_baseline, 'recall_at_1')} (fixed) | {_percent(random_baseline, 'recall_at_5')} | {_percent(random_baseline, 'mrr')} | — | — | — |")
    family_leaders = []
    seen_families = set()
    for arm, _, _ in ranked:
        family = arm.split(":", 1)[0]
        if family not in seen_families:
            seen_families.add(family)
            family_leaders.append(arm)
    if len(family_leaders) > 1:
        reference = family_leaders[0]
        lines.extend(["", f"Paired crossed-cluster Recall@1 differences from the leading family representation (`{reference}`; best pooling per family):", "",
                      "| Comparator | ΔR@1 (percentage points) | Crossed MIDI × preset-family 95% CI |",
                      "|---|---:|---:|"])
        for arm in family_leaders[1:]:
            delta = _paired_delta(evaluation, reference, arm)
            if delta:
                span = "includes zero" if delta["lower_95"] <= 0 <= delta["upper_95"] else ""
                lines.append(f"| `{arm}` | {100 * delta['difference']:+.1f} | [{100 * delta['lower_95']:+.1f}, {100 * delta['upper_95']:+.1f}] {span} |")
    lines.extend(["", "R@1 intervals are shown in preset-family, performance-ID, and crossed performance × preset-family order (2,000 bootstrap replicates); the heading gives the independent performance denominator. Negative-win cells show query denominators and preset-family/performance-cluster intervals. Clean/warning strata use query-render QA status; differences are descriptive because warning status can covary with preset acoustics. The full report also retains gallery quality strata.", ""])
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


def _reader_summary(current_dataset: dict, current_eval: dict | None, comparison_dataset: dict | None,
                    comparison_eval: dict | None, controls: dict, inventory_name: str,
                    availability: dict[str, dict]) -> list[str]:
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
        performance_label = ("held-out MIDI performances" if current_dataset.get("performance_source_split")
            and current_dataset.get("performance_source_split") != current_dataset.get("source_split")
            else "MIDI performances")
        finding = (f"**Finding:** {MODEL_REGISTRY[model_name]['name']} ({variant}) reached "
                   f"{100 * leader_result.get('recall_at_1', 0):.1f}% cross-MIDI Recall@1 "
                   f"(n={leader_result.get('episodes', 0):,} queries, {gallery_size} candidate presets, "
                   f"{current_dataset.get('unique_midi_excerpts', 0)} {performance_label}). "
                   "This ranks frozen representations across six model families; it is retrieval evidence, not a reconstruction result.")
        lines.extend([finding, ""])
    if comparison_dataset and comparison_eval is not None:
        lines.extend(_summary_cohort_lines("Original validation-paired cohort", comparison_dataset, comparison_eval))
    lines.extend(_summary_cohort_lines("Expanded held-out performance cohort", current_dataset, current_eval))
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
    family_coverage = (f"All six encoder families have expanded-cohort retrieval results on {common_episodes:,} identical common-gallery episodes per family."
                       if measured_families == len(MODEL_REGISTRY) else
                       f"{measured_families} of six encoder families have expanded-cohort retrieval results on {common_episodes:,} identical common-gallery episodes; incomplete families are listed below.")
    lines.extend([family_coverage,
                  f"No Vital controls are predicted or trained in retrieval. The linked [{inventory_name}](./{inventory_name}) inventories every exact source field/instance; each family is fixed with 0 enabled controls and 0 model outputs:", "",
                  "| Vital control family | Enabled / total | Output count | Update method / training signal |",
                  "|---|---:|---:|---|"])
    for group in GROUPS:
        total = controls.get(group, 0)
        lines.append(f"| {group} ❌ 0 / {total} | 0 | unchanged original preset; no training signal, loss, or gradient |")
    if current_dataset.get("performance_identity_overlap_excluded") is not None:
        lines.append(f"The expanded performance IDs came from the permitted `{current_dataset.get('performance_source_split')}` source split; {current_dataset.get('performance_identity_overlap_excluded', 0)} exact MIDI identities overlapping the original cohort were excluded. This is held out from the original MIDI IDs but remains within PresetShare source MIDI, not an external live-performance sample. The score is frozen retrieval on newly rendered audio, not reconstruction quality.")
    if current_eval:
        lines.append("The acoustic top-5 negatives are selected by an independent spectral descriptor, not human-verified perceptual labels. Query and gallery QA warning strata are reported to show whether render warnings shift ranking.")
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


def generate_report(destination: Path, inventory: Path, *, dataset_dir: Path | None = None,
                    checkpoint_root: Path | None = None, lock_path: Path | None = None,
                    execution_note: str | None = None, controlled_summary_path: Path | None = None,
                    comparison_dataset_dir: Path | None = None) -> dict:
    inventory_count, groups = _inventory_summary(inventory)
    controlled = json.loads(controlled_summary_path.read_text(encoding="utf-8")) if controlled_summary_path else None
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
             "Issue: [#152](https://github.com/noisedoctrine/obruxo/issues/152). This report covers frozen pretrained representations on original Vital source presets and existing MIDI performances. No synthesis controls are predicted, tuned, or optimized.", "",
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
            if primary and primary.get("episodes"):
                lines.extend(["", "Vital controls for this representation version:", "",
                    "| Vital control family | Enabled / exact instances | Vital output count | Update method / training signal |",
                    "|---|---:|---:|---|"])
                for group in GROUPS:
                    lines.append(f"| {group} ❌ 0 / {groups.get(group, 0)} | 0 | unchanged source preset; no head, optimization, loss, or gradient |")
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
            f"This secondary diagnostic uses one original validation preset, three single-control perturbations, {len(controlled.get('midi_strata', []))} MIDI performances, and source/independently randomized Vital renders for each version. It reports cross-MIDI identity retrieval, same-MIDI retrieval against a different random seed, and perturbation distance relative to the identical-preset repeat distance. Every query uses a source-seeded render and its gallery uses independent-seed renders; exact query audio is excluded from every gallery. BEATs and CIR controlled passes ran in isolated processes after a combined Windows XPU run faulted during a pandas/pyarrow import; their summaries were combined with the existing four-model results only after the same preset, MIDI IDs, controls, seeds, and 400 accepted renders were verified. MIDI-cluster intervals summarize variation for this one preset and do not generalize across presets or controls.",
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
    reader_summary = _reader_summary(dataset_info, evaluation, comparison_info, comparison_eval, groups,
                                     inventory.name, availability)
    status_start = document.find("## Execution status\n")
    if status_start >= 0:
        document = document[:status_start] + "\n".join(reader_summary) + "\n" + research_details + document[status_start:]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return {"models": len(MODEL_REGISTRY), "arms": sum(1 if spec["kind"] == "clip" else 2 for spec in MODEL_REGISTRY.values()),
            "control_records": inventory_count, "path": str(destination)}
