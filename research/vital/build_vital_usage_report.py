#!/usr/bin/env python3
"""Render figures and a human-readable report from a Vital usage census."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


repository_root = Path(__file__).resolve().parents[2]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))
from research.vital.build_vital_usage_census import MAJOR_FAMILIES


REPORT_VERSION = "1.0.0"
DEFAULT_CENSUS = Path("research") / "vital" / "vital_usage_census.json"
DEFAULT_REPORT = Path("research") / "vital" / "VITAL_USAGE_REPORT.md"
DEFAULT_FIGURES = Path("research") / "vital" / "vital_usage_figures"


def percentage(count: int | float, denominator: int | float) -> float:
    return 100.0 * count / denominator if denominator else 0.0


def count_distribution(values: dict[str, int]) -> list[tuple[int, int]]:
    return sorted((int(key), value) for key, value in values.items())


def save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def annotate_bars(ax: Any, bars: Any, labels: list[str], values: list[float], counts: list[int] | None = None) -> None:
    for index, (bar, label, value) in enumerate(zip(bars, labels, values)):
        count = f"  n={counts[index]:,}" if counts else ""
        ax.text(bar.get_width() + max(values, default=1.0) * 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%{count}", va="center", fontsize=8)


def component_family_rows(data: dict[str, Any], weight: str) -> list[tuple[str, float, int]]:
    aggregate = data[weight]
    denominator = aggregate["parsed_files"]
    rows = []
    for family, distribution in aggregate["family_count_distributions"].items():
        active = sum(count for key, count in distribution.items() if int(key) > 0)
        rows.append((family, percentage(active, denominator), active))
    return sorted(rows, key=lambda row: (-row[1], row[0]))


def parameter_rows(data: dict[str, Any], weight: str) -> list[dict[str, Any]]:
    rows = []
    for name, values in data[weight]["parameters"].items():
        if not values["default_known"] or not values["eligible"]:
            continue
        rows.append({
            "name": name,
            "non_default": percentage(values["non_default"], values["eligible"]),
            "conditional": percentage(values["active_non_default"], values["active_eligible"]),
            "modulated": percentage(values["modulated"], values["eligible"]),
            "non_default_count": values["non_default"],
            "eligible": values["eligible"],
            "family": values["introduced_feature"] or "shared",
        })
    return rows


def make_figures(data: dict[str, Any], figures: Path, weight: str) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    aggregate = data[weight]
    denominator = aggregate["parsed_files"]
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
    paths: dict[str, str] = {}

    versions = aggregate["versions"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2.0, 1.0]})
    version_labels = list(versions)
    version_values = list(versions.values())
    bars = axes[0].bar(version_labels, [percentage(value, denominator) for value in version_values], color="#315f8c")
    axes[0].set_title(f"Preset versions in the {weight.replace('_', ' ')} view")
    axes[0].set_ylabel("Presets (%)")
    axes[0].set_xlabel("Serialized synth_version")
    axes[0].tick_params(axis="x", rotation=55)
    for bar, count in zip(bars, version_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,}", ha="center", va="bottom", fontsize=8)
    duplicate_count = data["exact_duplicate_files"]
    unique_count = data["exact_unique_parsed_content_groups"]
    axes[1].bar(["duplicate\nfiles", "unique\ncontent"], [duplicate_count, unique_count], color=["#bd5b4b", "#5e9c76"])
    axes[1].set_title("Exact raw-byte duplicate sensitivity")
    axes[1].set_ylabel("Files / content groups")
    axes[1].text(0, duplicate_count, f"{duplicate_count:,}", ha="center", va="bottom")
    axes[1].text(1, unique_count, f"{unique_count:,}", ha="center", va="bottom")
    fig.suptitle(f"Corpus overview (n={denominator:,})", y=1.02)
    fig.tight_layout()
    path = figures / "01_corpus_overview.png"
    save_figure(fig, path)
    paths["corpus_overview"] = path.name

    family_rows = component_family_rows(data, weight)
    labels = [row[0] for row in family_rows]
    values = [row[1] for row in family_rows]
    counts = [row[2] for row in family_rows]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(labels[::-1], values[::-1], color="#467fa7")
    ax.set_title(f"Major component families used in presets (n={denominator:,})")
    ax.set_xlabel("Presets with at least one active/operational instance (%)")
    ax.set_xlim(0, max(values, default=100) * 1.25)
    annotate_bars(ax, bars, labels[::-1], values[::-1], counts[::-1])
    fig.tight_layout()
    path = figures / "02_component_family_prevalence.png"
    save_figure(fig, path)
    paths["component_family_prevalence"] = path.name

    patterns = aggregate["family_patterns"].get("oscillator", {})
    pattern_labels = list(patterns)
    pattern_values = [percentage(value, denominator) for value in patterns.values()]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(pattern_labels[::-1], pattern_values[::-1], color="#8c5b9e")
    ax.set_title(f"Oscillator identity patterns, not just oscillator count (n={denominator:,})")
    ax.set_xlabel("Presets (%)")
    ax.set_xlim(0, max(pattern_values, default=100) * 1.25)
    annotate_bars(ax, bars, pattern_labels[::-1], pattern_values[::-1], list(patterns.values())[::-1])
    fig.tight_layout()
    path = figures / "03_oscillator_patterns.png"
    save_figure(fig, path)
    paths["oscillator_patterns"] = path.name

    families = ["oscillator", "filter", "effect", "envelope", "lfo", "random"]
    slot_labels: list[str] = []
    slot_values: list[float] = []
    for family in families:
        for slot, count in aggregate["slot_prevalence"].get(family, {}).items():
            slot_labels.append(f"{family}[{slot}]")
            slot_values.append(percentage(count, denominator))
    order = sorted(range(len(slot_labels)), key=lambda index: (slot_values[index], slot_labels[index]))
    fig, ax = plt.subplots(figsize=(10, max(5.5, len(order) * 0.22)))
    bars = ax.barh([slot_labels[index] for index in order], [slot_values[index] for index in order], color="#d18a43")
    ax.set_title(f"Repeated-slot identity prevalence (n={denominator:,})")
    ax.set_xlabel("Presets with that specific slot active (%)")
    ax.set_xlim(0, max(slot_values, default=100) * 1.2)
    annotate_bars(ax, bars, [slot_labels[index] for index in order], [slot_values[index] for index in order])
    fig.tight_layout()
    path = figures / "04_repeated_slot_identity.png"
    save_figure(fig, path)
    paths["repeated_slot_identity"] = path.name

    count_families = [("oscillator", "Active oscillators"), ("lfo", "Routed LFOs"), ("effect", "Enabled effects"), ("modulation_slot", "Live modulation routes")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (family, title) in zip(axes.flat, count_families):
        distribution = aggregate["family_count_distributions"].get(family, {})
        items = count_distribution(distribution)
        ax.bar([str(key) for key, _ in items], [percentage(value, denominator) for _, value in items], color="#5d8c79")
        ax.set_title(title)
        ax.set_xlabel("Count per preset")
        ax.set_ylabel("Presets (%)")
    fig.suptitle(f"How many instances or routes does a preset use? (n={denominator:,})", y=1.02)
    fig.tight_layout()
    path = figures / "05_count_distributions.png"
    save_figure(fig, path)
    paths["count_distributions"] = path.name

    params = parameter_rows(data, weight)
    top = sorted(params, key=lambda row: (-row["non_default"], row["name"]))[:20]
    top_conditional = sorted(params, key=lambda row: (-row["conditional"], row["name"]))[:20]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    for ax, rows, title, field in (
        (axes[0], top, "Most often changed from atlas default", "non_default"),
        (axes[1], top_conditional, "Most often changed when owner is active", "conditional"),
    ):
        labels = [row["name"] for row in rows][::-1]
        values = [row[field] for row in rows][::-1]
        bars = ax.barh(labels, values, color="#426b9a")
        ax.set_title(title)
        ax.set_xlabel("Presets (%)")
        ax.set_xlim(0, 105)
        annotate_bars(ax, bars, labels, values)
    fig.suptitle(f"Scalar parameter prevalence (atlas-default comparison; n={denominator:,})", y=1.01)
    fig.tight_layout()
    path = figures / "06_parameter_prevalence.png"
    save_figure(fig, path)
    paths["parameter_prevalence"] = path.name

    ranked = sorted((row for row in params if row["non_default_count"] > 0), key=lambda row: (-row["non_default_count"], row["name"]))
    total_edits = sum(row["non_default_count"] for row in ranked)
    cumulative = []
    running = 0
    for row in ranked:
        running += row["non_default_count"]
        cumulative.append(percentage(running, total_edits))
    fig, ax = plt.subplots(figsize=(9, 5))
    if cumulative:
        ax.plot(range(1, len(cumulative) + 1), cumulative, color="#a44d58", linewidth=2)
    ax.set_title(f"Cumulative coverage of scalar non-default observations (n={denominator:,})")
    ax.set_xlabel("Parameters ranked by non-default count")
    ax.set_ylabel("Share of all counted non-default observations (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figures / "07_parameter_sparsity.png"
    save_figure(fig, path)
    paths["parameter_sparsity"] = path.name

    pair_counts = aggregate["routes"]["family_pairs"]
    source_families = sorted({key.split(" -> ", 1)[0] for key in pair_counts})
    destination_families = sorted({key.split(" -> ", 1)[1] for key in pair_counts})
    matrix = np.zeros((len(source_families), len(destination_families)))
    for row_index, source in enumerate(source_families):
        for column_index, destination in enumerate(destination_families):
            matrix[row_index, column_index] = pair_counts.get(f"{source} -> {destination}", 0)
    fig, ax = plt.subplots(figsize=(9, 6))
    if matrix.size:
        image = ax.imshow(matrix, cmap="Blues", aspect="auto")
        fig.colorbar(image, ax=ax, label="Live routes")
        ax.set_xticks(range(len(destination_families)), destination_families, rotation=45, ha="right")
        ax.set_yticks(range(len(source_families)), source_families)
        for row_index in range(len(source_families)):
            for column_index in range(len(destination_families)):
                if matrix[row_index, column_index]:
                    ax.text(column_index, row_index, f"{int(matrix[row_index, column_index]):,}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Live modulation source-family → destination-family routes (n={denominator:,})")
    fig.tight_layout()
    path = figures / "08_modulation_family_matrix.png"
    save_figure(fig, path)
    paths["modulation_family_matrix"] = path.name

    cooccurrence = aggregate["cooccurrence"]
    co_matrix = np.zeros((len(MAJOR_FAMILIES), len(MAJOR_FAMILIES)))
    for row_index, left in enumerate(MAJOR_FAMILIES):
        for column_index, right in enumerate(MAJOR_FAMILIES):
            if row_index == column_index:
                family_distribution = aggregate["family_count_distributions"].get(left, {})
                co_matrix[row_index, column_index] = percentage(
                    sum(count for key, count in family_distribution.items() if int(key) > 0),
                    denominator,
                )
            else:
                key = f"{left} + {right}" if row_index < column_index else f"{right} + {left}"
                co_matrix[row_index, column_index] = percentage(cooccurrence.get(key, 0), denominator)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(co_matrix, cmap="Purples", vmin=0, vmax=max(100.0, float(np.max(co_matrix))))
    fig.colorbar(image, ax=ax, label="Co-occurrence (%)")
    ax.set_xticks(range(len(MAJOR_FAMILIES)), MAJOR_FAMILIES, rotation=45, ha="right")
    ax.set_yticks(range(len(MAJOR_FAMILIES)), MAJOR_FAMILIES)
    for row_index in range(len(MAJOR_FAMILIES)):
        for column_index in range(len(MAJOR_FAMILIES)):
            ax.text(column_index, row_index, f"{co_matrix[row_index, column_index]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Major-family co-occurrence among presets (n={denominator:,})")
    fig.tight_layout()
    path = figures / "09_component_cooccurrence.png"
    save_figure(fig, path)
    paths["component_cooccurrence"] = path.name

    nested = aggregate["nested"]
    lfo_slot_total = sum(int(key) * count for key, count in aggregate["family_count_distributions"].get("lfo", {}).items())
    oscillator_slot_total = sum(int(key) * count for key, count in aggregate["family_count_distributions"].get("oscillator", {}).items())
    connected_route_total = aggregate["routes"].get("connected", 0)
    nested_rows = [
        ("preset with custom LFO shape", nested.get("lfo_shape", {}).get("custom", 0), denominator),
        ("routed slot with custom LFO", nested.get("lfo_shape", {}).get("operational_custom_slot", 0), lfo_slot_total),
        ("active named stock/unresolved wavetable", nested.get("wavetable", {}).get("named_stock_or_unresolved_content_slot", 0), oscillator_slot_total),
        ("active named nonstock/unresolved wavetable", nested.get("wavetable", {}).get("named_nonstock_or_unresolved_content_slot", 0), oscillator_slot_total),
        ("preset with named stock/unresolved sample", nested.get("sampler", {}).get("named_stock_or_unresolved_content", 0), denominator),
        ("preset with named nonstock/unresolved sample", nested.get("sampler", {}).get("named_nonstock_or_unresolved_content", 0), denominator),
        ("connected route with custom remap", aggregate["routes"].get("custom_mapping", 0), connected_route_total),
    ]
    labels = [row[0] for row in nested_rows]
    values = [percentage(row[1], row[2]) for row in nested_rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels[::-1], values[::-1], color="#7b6a4e")
    ax.set_title(f"Nested and custom-state prevalence (file-weighted parsed n={denominator:,})")
    ax.set_xlabel("Percentage within the denominator stated in the caption/report (%)")
    ax.set_xlim(0, max(values, default=100) * 1.25)
    annotate_bars(ax, bars, labels[::-1], values[::-1], [row[1] for row in nested_rows][::-1])
    fig.tight_layout()
    path = figures / "10_nested_state.png"
    save_figure(fig, path)
    paths["nested_state"] = path.name

    weighted_parameters = parameter_rows(data, "file_weighted")
    unique_parameters = {row["name"]: row for row in parameter_rows(data, "exact_deduplicated")}
    sensitivity_names = [row["name"] for row in sorted(weighted_parameters, key=lambda row: (-row["non_default"], row["name"]))[:10]]
    file_values = [next(row["non_default"] for row in weighted_parameters if row["name"] == name) for name in sensitivity_names]
    unique_values = [unique_parameters[name]["non_default"] for name in sensitivity_names]
    x = np.arange(len(sensitivity_names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2, file_values, width, label="file-weighted", color="#467fa7")
    ax.bar(x + width / 2, unique_values, width, label="exact-deduplicated", color="#bd5b4b")
    ax.set_xticks(x, sensitivity_names, rotation=60, ha="right")
    ax.set_ylabel("Non-default prevalence (%)")
    ax.set_title("Duplicate-weighting sensitivity for the most edited shared parameters")
    ax.legend()
    fig.tight_layout()
    path = figures / "11_duplicate_sensitivity.png"
    save_figure(fig, path)
    paths["duplicate_sensitivity"] = path.name

    return paths


def build_report(census_path: Path, report_path: Path, figures_path: Path, weight: str = "file_weighted") -> None:
    data = json.loads(census_path.read_text(encoding="utf-8"))
    aggregate = data[weight]
    denominator = aggregate["parsed_files"]
    unique_denominator = data["exact_deduplicated"]["parsed_files"]
    figures = make_figures(data, figures_path, weight)
    figure_link = lambda name: os.path.relpath(figures_path / figures[name], report_path.parent).replace(os.sep, "/")

    families = component_family_rows(data, weight)
    top_family = families[0] if families else ("unknown", 0.0, 0)
    patterns = aggregate["family_patterns"].get("oscillator", {})
    top_pattern = next(iter(patterns.items()), ("none", 0))
    params = parameter_rows(data, weight)
    top_parameter = sorted(params, key=lambda row: (-row["non_default"], row["name"]))[0] if params else None
    introduced = {
        name: values
        for name, values in aggregate["parameters"].items()
        if values["introduced_feature"]
    }
    route_count = aggregate["routes"]["live_count_distribution"]
    median_route_count = 0
    if route_count:
        midpoint = (denominator + 1) / 2
        running = 0
        for value, count in count_distribution(route_count):
            running += count
            if running >= midpoint:
                median_route_count = value
                break
    lines = [
        "# Vital Preset Usage Census",
        "",
        "## Main findings",
        "",
        f"The corpus contains **{denominator:,} parsed presets** in the file-weighted view and **{unique_denominator:,} exact raw-byte content groups** after duplicate collapse. Exact duplicates therefore account for **{data['exact_duplicate_files']:,} additional files** ({percentage(data['exact_duplicate_files'], denominator):.2f}% of parsed files).",
        "",
        f"The largest operational family is **{top_family[0]}** at **{top_family[1]:.1f}%** ({top_family[2]:,}/{denominator:,}). Oscillator identity is not reduced to a count: the most common exact mask is **{top_pattern[0]}**, occurring in **{percentage(top_pattern[1], denominator):.1f}%** ({top_pattern[1]:,}/{denominator:,}) of presets. The median number of live, non-bypassed, non-zero modulation routes is **{median_route_count}**.",
        "",
        f"The most frequently changed shared scalar is **`{top_parameter['name']}`** at **{top_parameter['non_default']:.1f}%** ({top_parameter['non_default_count']:,}/{top_parameter['eligible']:,}) of atlas-supported presets." if top_parameter else "No atlas-backed scalar parameters were available for ranking.",
        "",
        "These are descriptive storage/use measures, not claims about audibility or perceptual importance. A serialized key is not treated as evidence of use.",
        "",
        "## Corpus and version support",
        "",
        f"![Corpus overview]({figure_link('corpus_overview')})",
        "",
        f"Figure 1 uses the {weight.replace('_', ' ')} denominator (`n={denominator:,}`). Version bars are counts of parsed presets. The duplicate panel is a sensitivity check: later tables and plots retain both file-weighted and exact-deduplicated aggregates, rather than silently choosing one.",
        "",
        "The shared parameter space uses the pinned atlas defaults and treats a missing common scalar as its canonical default, matching Vital's load behavior. The 1.5.x spectral-morph phase fields, 1.6.x modulation-ramp fields, and the isolated `flanger_depth` extension are kept in version-eligible groups. Their defaults are not present in the pinned atlas, so the machine-readable output reports nonzero observations and marks default-qualified non-default percentages as unsupported rather than inventing a denominator or default.",
        "",
        "## Component activity and repeated-slot identity",
        "",
        f"![Component family prevalence]({figure_link('component_family_prevalence')})",
        "",
        "Figure 2 counts direct `*_on` state for oscillators, filters, effects, and the sampler. Envelopes, LFOs, and random sources use operational activity: their numbered slot is active only when it is the source of a live modulation route. This keeps direct enablement separate from routing.",
        "",
        f"![Oscillator patterns]({figure_link('oscillator_patterns')})",
        "",
        "Figure 3 preserves exact oscillator identity masks. `2` means oscillator 2 only; it is not merged with `1` merely because both masks have cardinality one. Any skipped-slot mask visible in the figure is therefore an observed non-prefix pattern.",
        "",
        f"![Repeated-slot identity]({figure_link('repeated_slot_identity')})",
        "",
        "Figure 4 gives the independent prevalence of each numbered slot. Figure 5 then switches back to cardinality, showing how many oscillators, routed LFOs, effects, and live modulation slots occur per preset. The two views answer different questions and should not be substituted for one another.",
        "",
        f"![Count distributions]({figure_link('count_distributions')})",
        "",
        "## Scalar parameter sparsity and modulation",
        "",
        f"![Parameter prevalence]({figure_link('parameter_prevalence')})",
        "",
        "Figure 6 ranks atlas-backed scalar changes globally and conditional on an active owner. For LFO, envelope, random, and modulation-slot parameters, the conditional denominator is the operationally routed/connected slot population. This highlights parameters that look globally rare because their component is rarely used.",
        "",
        f"![Parameter sparsity]({figure_link('parameter_sparsity')})",
        "",
        "Figure 7 is a cumulative head-versus-tail view. The x-axis is the rank of a parameter by non-default count; the y-axis is the share of all counted non-default observations covered by that prefix. It shows concentration without imposing an arbitrary prevalence cutoff.",
        "",
        f"![Modulation family matrix]({figure_link('modulation_family_matrix')})",
        "",
        f"Figure 8 counts live routes only: connected routes that are not bypassed and have non-zero amount. The census separately retains connected-route source/destination vocabularies, connected, bypassed, zero-amount, non-zero-amount, bipolar, stereo, explicit-linear, and custom-remap counts. In this view, the plotted source/destination prevalence is therefore about operational routing, while a populated but bypassed or zero-amount connection remains visible in the supporting JSON.",
        "",
        "## Co-occurrence and nested state",
        "",
        f"![Component co-occurrence]({figure_link('component_cooccurrence')})",
        "",
        "Figure 9 uses preset-level presence flags and normalized percentages. Diagonal cells are the family prevalence; off-diagonal cells are the percentage of presets containing both families. LFO, envelope, and random presence again means live source routing, not merely non-default scalar state.",
        "",
        f"![Nested state]({figure_link('nested_state')})",
        "",
        "Figure 10 treats nested structures semantically. LFO shape comparison ignores display names and compares the drawable shape fields. Wavetables retain a canonical-init descriptor comparison in the JSON, but the figure classifies active slots conservatively by known stock names: named stock content is reported as `named_stock_or_unresolved_content`, while unknown names remain unresolved rather than being called custom from a byte comparison. This avoids mistaking a version/rendering difference in a stock asset for a user-authored wavetable. Preset bars use `n` parsed presets; routed-LFO and wavetable bars use routed/active slot denominators; remap bars use connected-route denominators.",
        "",
        "## Version-introduced features and duplicate sensitivity",
        "",
        f"The version-introduced parameter groups contain **{len(introduced):,} scalar names** in the pooled output. Their eligible denominators are restricted to versions that can contain them; older presets are never put in the denominator. Because the pinned atlas does not define their defaults, use the `nonzero` field for the neutral-state observation and do not read their `non_default` field as zero.",
        "",
        f"![Duplicate sensitivity]({figure_link('duplicate_sensitivity')})",
        "",
        "Figure 11 compares the file-weighted and exact-deduplicated estimates for the most edited shared parameters. The JSON and parameter CSV retain the complete comparison, including numerator and denominator for every parameter; the figure is only a readable headline slice.",
        "",
        "## Method notes",
        "",
        "- The corpus is scanned recursively one `.vital` file at a time and is never rewritten.",
        "- File-weighted aggregates include every successfully parsed file. Exact-deduplicated aggregates include one representative for each raw-byte SHA-256; this is analysis-only and does not remove or alter source files.",
        "- Direct component usage is `*_on != 0`. Modulation `connected`, `bypassed`, `amount_zero`, `amount_nonzero`, and `live` are separate counters. A live route is connected, not bypassed, and non-zero amount.",
        "- Every scalar percentage has an eligible count in `vital_usage_parameters.csv` and the JSON. Atlas-backed defaults come from `VitalSchema.parameters`; missing common scalar keys are filled with that default for comparison.",
        "- Custom LFO shapes, wavetable name/content statuses, non-init wavetable descriptors, route remaps, and sampler statuses are aggregate semantic labels only. Raw preset paths, names, authors, payloads, and per-file records are not report content.",
        "- The pinned source/schema context is documented in [`PRESET_SCHEMA.md`](PRESET_SCHEMA.md) and [`VITAL_CORPUS_AUDIT.md`](VITAL_CORPUS_AUDIT.md).",
        "",
        "## Reproduction and artifacts",
        "",
        "```powershell",
        "conda activate py312",
        "python research/vital/build_vital_usage_census.py",
        "python research/vital/build_vital_usage_report.py",
        "```",
        "The aggregate source of truth is [`vital_usage_census.json`](vital_usage_census.json); the complete scalar lookup table is [`vital_usage_parameters.csv`](vital_usage_parameters.csv).",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Vital usage census report and figures.")
    parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--weight", choices=("file_weighted", "exact_deduplicated"), default="file_weighted")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.census.is_file():
        raise SystemExit(f"census file does not exist: {args.census}")
    build_report(args.census.resolve(), args.report.resolve(), args.figures.resolve(), args.weight)
    print(f"Wrote {args.report.resolve()}")
    print(f"Wrote figures under {args.figures.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
