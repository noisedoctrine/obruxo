#!/usr/bin/env python3
"""Render figures and a human-readable report from a Vital usage census."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


repository_root = Path(__file__).resolve().parents[2]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))
data_generation_root = repository_root / "research" / "data_generation"
if str(data_generation_root) not in sys.path:
    sys.path.insert(0, str(data_generation_root))
from research.data_generation.obruxo_data.vital.components import ComponentKind, ComponentRef, component_registry
from research.vital.build_vital_usage_census import MAJOR_FAMILIES


REPORT_VERSION = "1.1.0"
DEFAULT_CENSUS = Path("research") / "vital" / "vital_usage_census.json"
DEFAULT_REPORT = Path("research") / "vital" / "VITAL_USAGE_REPORT.md"
DEFAULT_FIGURES = Path("research") / "vital" / "vital_usage_figures"
COMPONENT_FAMILY_ORDER = ("global", "oscillator", "sampler", "filter", "envelope", "lfo", "random", "effect", "modulation_slot")
COMPONENT_FAMILY_LABELS = {
    "global": "Global controls",
    "oscillator": "Oscillators",
    "sampler": "Sampler",
    "filter": "Filters",
    "envelope": "Envelopes",
    "lfo": "LFOs",
    "random": "Random sources",
    "effect": "Effects",
    "modulation_slot": "Modulation matrix",
}
COMPONENT_FAMILY_DESCRIPTIONS = {
    "global": "Voice, performance, macro, routing, and other controls that do not belong to a repeated component.",
    "oscillator": "The three wavetable oscillators, including oscillator routing, spectral morph controls, and nested wavetable-editor state.",
    "sampler": "The sample oscillator's scalar controls and the sanitized stock/non-stock/unresolved content classification.",
    "filter": "The two main filters and their per-filter model, style, cutoff, resonance, drive, blend, and routing controls.",
    "envelope": "Six envelopes, with quartic-time controls, sustain, and operational routing prevalence.",
    "lfo": "Eight drawable LFOs, including rate/sync controls and nested custom-shape prevalence.",
    "random": "Four random modulation sources, including style, rate, sync, stereo, and keytracking controls.",
    "effect": "Each effect block separately, rather than pooling chorus, delay, distortion, EQ, and reverb controls.",
    "modulation_slot": "All 64 matrix slots, with per-slot scalar distributions plus the route source/destination and remap analysis.",
}


def percentage(count: int | float, denominator: int | float) -> float:
    return 100.0 * count / denominator if denominator else 0.0


def count_distribution(values: dict[str, int]) -> list[tuple[int, int]]:
    return sorted((int(key), value) for key, value in values.items())


def binned_count_distribution(values: dict[str, int], individual_max: int = 20, bin_width: int = 4) -> list[tuple[str, int]]:
    counts = {int(key): value for key, value in values.items()}
    maximum = max(counts, default=0)
    rows = [(str(value), counts.get(value, 0)) for value in range(min(individual_max, maximum) + 1)]
    lower = individual_max + 1
    while lower <= maximum:
        upper = min(lower + bin_width - 1, maximum)
        rows.append((f"{lower}–{upper}", sum(counts.get(value, 0) for value in range(lower, upper + 1))))
        lower = upper + 1
    return rows


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


def format_number(value: Any) -> str:
    if value is None:
        return "—"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 0.001:
        return f"{value:.4g}"
    return f"{value:.5f}".rstrip("0").rstrip(".")


def frequency_text(frequency: float | None) -> str:
    return f"{format_number(frequency * 100)}%" if frequency is not None else "—"


def raw_from_normalized(value: float | None, parameter: dict[str, Any]) -> float | None:
    if value is None or parameter.get("minimum") is None or parameter.get("maximum") is None:
        return value
    position = value
    scale = parameter.get("scale")
    if scale == "Quadratic":
        position = math.sqrt(max(0.0, value))
    elif scale == "Cubic":
        position = value ** (1.0 / 3.0)
    elif scale == "Quartic":
        position = value ** 0.25
    elif scale == "SquareRoot":
        position = value * value
    span = parameter["maximum"] - parameter["minimum"]
    return parameter["minimum"] + position * span


def value_text(row: dict[str, Any]) -> str:
    value = format_number(row.get("value"))
    label = row.get("label")
    return f"{value} ({label})" if label else value


def dominant_bin_text(parameter: dict[str, Any], row: dict[str, Any]) -> str:
    domain = parameter.get("distribution", {}).get("domain")
    lower = row.get("lower")
    upper = row.get("upper")
    if domain == "normalized":
        raw_lower = raw_from_normalized(lower, parameter)
        raw_upper = raw_from_normalized(upper, parameter)
        interval = f"{format_number(raw_lower)}–{format_number(raw_upper)} raw / {format_number(lower)}–{format_number(upper)} norm"
    else:
        interval = f"{format_number(lower)}–{format_number(upper)} raw"
    return f"{interval}: {row['count']:,} ({row['frequency'] * 100:.1f}%)"


def parameter_component_ref(name: str, registry: dict[ComponentRef, Any]) -> ComponentRef:
    for ref, definition in registry.items():
        if name in definition.owned_parameters:
            return ref
    if name == "flanger_depth":
        return ComponentRef(ComponentKind.EFFECT, "flanger")
    for pattern, kind in (
        (r"^osc_(\d+)_", ComponentKind.OSCILLATOR),
        (r"^sample_", ComponentKind.SAMPLER),
        (r"^filter_(\d+)_", ComponentKind.FILTER),
        (r"^env_(\d+)_", ComponentKind.ENVELOPE),
        (r"^lfo_(\d+)_", ComponentKind.LFO),
        (r"^random_(\d+)_", ComponentKind.RANDOM),
        (r"^modulation_(\d+)_", ComponentKind.MODULATION_SLOT),
    ):
        match = re.match(pattern, name)
        if match:
            return ComponentRef(kind, int(match.group(1)) if match.groups() else None)
    for effect in ("chorus", "compressor", "delay", "distortion", "eq", "filter_fx", "flanger", "phaser", "reverb"):
        if name.startswith(f"{effect}_"):
            return ComponentRef(ComponentKind.EFFECT, effect)
    return ComponentRef(ComponentKind.GLOBAL)


def component_display_name(ref: ComponentRef) -> str:
    if ref.kind == ComponentKind.GLOBAL:
        return "Global controls"
    if ref.kind == ComponentKind.SAMPLER:
        return "Sampler"
    if ref.kind == ComponentKind.EFFECT:
        return f"{str(ref.slot).replace('_', ' ').title()} effect"
    family_name = {
        ComponentKind.OSCILLATOR: "Oscillator",
        ComponentKind.FILTER: "Filter",
        ComponentKind.ENVELOPE: "Envelope",
        ComponentKind.LFO: "LFO",
        ComponentKind.RANDOM: "Random source",
        ComponentKind.MODULATION_SLOT: "Modulation slot",
    }.get(ref.kind, ref.kind.value.replace('_', ' ').title())
    return f"{family_name} {ref.slot}"


def component_parameter_groups(data: dict[str, Any], weight: str) -> tuple[dict[ComponentRef, Any], dict[ComponentRef, list[tuple[str, dict[str, Any]]]], dict[str, list[ComponentRef]]]:
    registry = component_registry()
    groups: dict[ComponentRef, list[tuple[str, dict[str, Any]]]] = {ref: [] for ref in registry}
    family_refs: dict[str, list[ComponentRef]] = defaultdict(list)
    for ref in registry:
        family_refs[ref.kind.value].append(ref)
    for name, values in data[weight]["parameters"].items():
        ref = parameter_component_ref(name, registry)
        groups.setdefault(ref, []).append((name, values))
        if ref not in family_refs[ref.kind.value]:
            family_refs[ref.kind.value].append(ref)
    for values in groups.values():
        values.sort(key=lambda item: item[0])
    for refs in family_refs.values():
        refs.sort(key=lambda ref: (ref.slot is None, ref.slot if isinstance(ref.slot, int) else str(ref.slot)))
    return registry, groups, family_refs


def component_parameter_label(ref: ComponentRef, name: str) -> str:
    return f"{component_display_name(ref)} · {name}"


def categorical_modal_text(parameter: dict[str, Any], limit: int = 3) -> str:
    frequencies = parameter.get("value_frequencies", [])
    return "; ".join(f"{value_text(row)}: {row['count']:,} ({row['frequency'] * 100:.1f}%)" for row in frequencies[:limit]) or "—"


def continuous_quantile_text(parameter: dict[str, Any]) -> str:
    distribution = parameter.get("distribution", {})
    quantiles = distribution.get("quantiles", {})
    values = (quantiles.get("p05"), quantiles.get("p50"), quantiles.get("p95"))
    if distribution.get("domain") == "normalized":
        values = tuple(raw_from_normalized(value, parameter) for value in values)
    return " / ".join(format_number(value) for value in values)


def component_state_text(ref: ComponentRef, definition: Any, component: dict[str, Any], denominator: int) -> str:
    eligible = component.get("eligible", denominator) or denominator
    details = []
    if definition.enable_parameter:
        details.append(f"enabled {percentage(component.get('enabled', 0), eligible):.1f}% ({component.get('enabled', 0):,}/{eligible:,})")
    elif ref.kind in {ComponentKind.ENVELOPE, ComponentKind.LFO, ComponentKind.RANDOM, ComponentKind.MODULATION_SLOT}:
        details.append(f"routed {percentage(component.get('routed', 0), eligible):.1f}% ({component.get('routed', 0):,}/{eligible:,})")
    if ref.kind != ComponentKind.GLOBAL:
        details.append(f"changed {percentage(component.get('changed', 0), eligible):.1f}% ({component.get('changed', 0):,}/{eligible:,})")
    return "; ".join(details) or "component-level enable/routing state is not applicable"


def component_parameter_lines(ref: ComponentRef, parameters: list[tuple[str, dict[str, Any]]]) -> list[str]:
    categorical = [(name, values) for name, values in parameters if values.get("parameter_type") == "categorical"]
    continuous = [(name, values) for name, values in parameters if values.get("parameter_type") == "continuous"]
    lines = []
    if categorical:
        lines.extend([
            "**Categorical / enum value frequencies**",
            "",
            "| Parameter | Observed | Distinct | Modal values |",
            "|---|---:|---:|---|",
        ])
        for name, values in categorical:
            lines.append(f"| `{name}` | {values.get('observed', 0):,} | {len(values.get('value_frequencies', [])):,} | {categorical_modal_text(values)} |")
        lines.append("")
    if continuous:
        lines.extend([
            "**Continuous value distributions**",
            "",
            "| Parameter | Observed | Raw range | p05 / p50 / p95 | Default | Zero | Dominant range |",
            "|---|---:|---|---|---:|---:|---|",
        ])
        for name, values in continuous:
            summary = values.get("observed_value_summary", {})
            distribution = values.get("distribution", {})
            dominant = distribution.get("dominant_bins", [])
            default_frequency = values.get("default_frequency")
            zero_frequency = (values.get("observed", 0) - values.get("nonzero", 0)) / values["observed"] if values.get("observed") else None
            dominant_text = dominant_bin_text(values, dominant[0]) if dominant else "—"
            lines.append(
                f"| `{name}` | {values.get('observed', 0):,} | {format_number(summary.get('min'))}–{format_number(summary.get('max'))} | {continuous_quantile_text(values)} | {frequency_text(default_frequency)} | {frequency_text(zero_frequency)} | {dominant_text} |"
            )
        lines.append("")
    return lines


def component_family_nested_lines(family: str, aggregate: dict[str, Any]) -> list[str]:
    lines = []
    nested = aggregate.get("nested", {})
    if family == "oscillator":
        wavetable = nested.get("wavetable", {})
        component_types = nested.get("wavetable_component_type", {})
        lines.extend([
            "**Nested wavetable-editor analysis**",
            "",
            f"Active oscillator slots contain **{wavetable.get('named_stock_or_unresolved_content_slot', 0):,}** named stock/unresolved slots and **{wavetable.get('named_nonstock_or_unresolved_content_slot', 0):,}** named non-stock/unresolved slots. The semantic non-init comparison flags **{wavetable.get('active_non_init_slot', 0):,}** active slots as different from the canonical init descriptor; that is not a claim that their embedded payload is user-authored.",
            "",
            "Wavetable editor component types are counted as sanitized active-component occurrences; waveform/keyframe payloads are not retained:",
            "",
            "| Editor component type | Occurrences |",
            "|---|---:|",
        ])
        lines.extend(f"| `{name}` | {count:,} |" for name, count in component_types.items())
        lines.append("")
    elif family == "sampler":
        states = nested.get("sampler", {})
        lines.extend([
            "**Nested sample-content analysis**",
            "",
            "| Sample state | Presets |",
            "|---|---:|",
        ])
        lines.extend(f"| `{name}` | {count:,} |" for name, count in states.items())
        lines.append("")
    elif family == "lfo":
        shapes = nested.get("lfo_shape", {})
        lines.extend([
            "**Nested drawable-shape analysis**",
            "",
            f"Across the eight drawable LFO objects, **{shapes.get('custom_slot', 0):,}** slots have a non-default shape and **{shapes.get('operational_custom_slot', 0):,}** of those are operationally routed. Display names are ignored; the comparison uses the serialized drawable fields.",
            "",
        ])
    elif family == "modulation_slot":
        routes = aggregate.get("routes", {})
        lines.extend([
            "**Modulation-matrix relationship analysis**",
            "",
            f"The corpus contains **{routes.get('connected', 0):,}** connected routes, **{routes.get('live', 0):,}** live routes, **{routes.get('bypassed', 0):,}** bypassed routes, **{routes.get('amount_zero', 0):,}** connected zero-amount routes, and **{routes.get('custom_mapping', 0):,}** custom remaps. Source/destination family pairs are shown in the modulation matrix figure; each slot's scalar controls are listed above.",
            "",
        ])
    return lines


def write_component_distribution_report(data: dict[str, Any], path: Path, weight: str) -> None:
    aggregate = data[weight]
    registry, groups, family_refs = component_parameter_groups(data, weight)
    continuous = [values for values in aggregate["parameters"].values() if values.get("parameter_type") == "continuous"]
    raw_fallback_count = sum("distribution_note" in values for values in continuous)
    lines = [
        "# Vital Component-by-Component Value Distributions",
        "",
        f"This companion report uses the **{weight.replace('_', ' ')}** aggregate ({aggregate['parsed_files']:,} parsed presets). It is organized by the pinned Vital component registry rather than pooling unrelated parameter names.",
        "",
        "The row-oriented exports are the complete machine-readable detail: [`vital_usage_categorical_values.csv`](vital_usage_categorical_values.csv) contains one row per categorical value, and [`vital_usage_continuous_bins.csv`](vital_usage_continuous_bins.csv) contains all 64 histogram bins for each continuous parameter, with both file-weighted and exact-deduplicated counts.",
        "",
        "Categorical values are Vital's raw numeric ordinals. Labels are included when the pinned atlas provides an option list. Their denominator is the observed value count under the existing policy that fills missing common scalar keys with the atlas default.",
        "",
        f"Continuous parameters include raw-value min/max/mean/stddev, histogram quantiles, default and zero prevalence, and dominant bins. Atlas-backed controls use 64 bins in normalized control position, preserving the parameter's scale metadata; version-introduced controls without atlas bounds use adaptive raw-value bins. {raw_fallback_count} atlas-backed controls also use complete raw-value bins because their corpus observations exceed the pinned bounds; their partial normalized histograms remain in the JSON. For an `Exponential` parameter, the raw Vital value is already the logarithmic storage domain.",
        "",
    ]
    for family in COMPONENT_FAMILY_ORDER:
        refs = family_refs.get(family, [])
        if not refs:
            continue
        lines.extend([f"## {COMPONENT_FAMILY_LABELS[family]}", "", COMPONENT_FAMILY_DESCRIPTIONS[family], ""])
        for ref in refs:
            definition = registry.get(ref)
            parameters = groups.get(ref, [])
            component = aggregate.get("components", {}).get(str(ref), {})
            categorical_count = sum(values.get("parameter_type") == "categorical" for _, values in parameters)
            continuous_count = sum(values.get("parameter_type") == "continuous" for _, values in parameters)
            lines.extend([
                f"### {component_display_name(ref)}",
                "",
                f"{len(parameters):,} scalar parameters: {categorical_count:,} categorical/enum and {continuous_count:,} continuous. {component_state_text(ref, definition, component, aggregate['parsed_files'])}.",
                "",
            ])
            lines.extend(component_parameter_lines(ref, parameters))
        lines.extend(component_family_nested_lines(family, aggregate))

    lines.extend([
        "## Interpretation cautions",
        "",
        "- A categorical mode is a storage-value mode, not a claim that the corresponding UI choice is perceptually dominant.",
        "- A continuous dominant bin is a range, not an exact mode. This avoids pretending that arbitrary floating-point values have exact categorical semantics.",
        "- The normalized histogram follows Vital's raw control scale. It is useful for comparing differently ranged controls, while the raw summary preserves the actual serialized values.",
        "- Wavetable and sample payloads are not included. Wavetable editor component-type counts are structural occurrences only; they do not expose embedded waveform/audio data.",
        "- These aggregates contain no preset paths, names, authors, raw wavetable/sample payloads, or per-file records.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def make_component_family_figure(
    data: dict[str, Any],
    weight: str,
    family: str,
    registry: dict[ComponentRef, Any],
    groups: dict[ComponentRef, list[tuple[str, dict[str, Any]]]],
    refs: list[ComponentRef],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    aggregate = data[weight]
    categorical_modes = []
    continuous_modes = []
    categorical_shares = []
    continuous_shares = []
    for ref in refs:
        for name, parameter in groups.get(ref, []):
            if parameter.get("parameter_type") == "categorical":
                frequencies = parameter.get("value_frequencies", [])
                if frequencies:
                    mode = frequencies[0]
                    share = mode["frequency"] * 100
                    categorical_shares.append(share)
                    categorical_modes.append({
                        "label": f"{component_display_name(ref)} · {name} = {value_text(mode)}",
                        "share": share,
                        "count": mode["count"],
                    })
            elif parameter.get("parameter_type") == "continuous":
                dominant = parameter.get("distribution", {}).get("dominant_bins", [])
                if dominant:
                    share = dominant[0]["frequency"] * 100
                    continuous_shares.append(share)
                    if parameter.get("observed", 0) >= 100:
                        continuous_modes.append({
                            "label": component_parameter_label(ref, name),
                            "share": share,
                            "count": dominant[0]["count"],
                        })

    categorical_modes.sort(key=lambda row: (row["share"], -row["count"], row["label"]))
    continuous_modes.sort(key=lambda row: (row["share"], -row["count"], row["label"]))
    categorical_top = list(reversed(categorical_modes[:12]))
    continuous_top = list(reversed(continuous_modes[:12]))
    height = max(9.0, len(refs) * (0.28 if family == "modulation_slot" else 0.22))
    fig, axes = plt.subplots(2, 2, figsize=(16, height), gridspec_kw={"height_ratios": [1.0, 1.2]})

    if family == "global":
        type_counts = [sum(values.get("parameter_type") == kind for ref in refs for _, values in groups.get(ref, [])) for kind in ("categorical", "continuous")]
        bars = axes[0, 0].bar(["Categorical / enum", "Continuous"], type_counts, color=["#8c5b9e", "#426b9a"])
        axes[0, 0].set_title("Global parameter inventory")
        axes[0, 0].set_ylabel("Parameters")
        for bar, count in zip(bars, type_counts):
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,}", ha="center", va="bottom")
    else:
        state_labels = [component_display_name(ref) for ref in refs]
        active_values = []
        changed_values = []
        for ref in refs:
            component = aggregate.get("components", {}).get(str(ref), {})
            definition = registry.get(ref)
            eligible = component.get("eligible", aggregate["parsed_files"]) or aggregate["parsed_files"]
            active = component.get("enabled", 0) if definition.enable_parameter else component.get("routed", 0)
            active_values.append(percentage(active, eligible))
            changed_values.append(percentage(component.get("changed", 0), eligible))
        order = list(range(len(refs)))[::-1]
        y_labels = [state_labels[index] for index in order]
        y = np.arange(len(order))
        axes[0, 0].barh(y + 0.18, [active_values[index] for index in order], height=0.34, label="enabled/routed", color="#5e9c76")
        axes[0, 0].barh(y - 0.18, [changed_values[index] for index in order], height=0.34, label="changed", color="#d18a43")
        axes[0, 0].set_yticks(y, y_labels, fontsize=8)
        axes[0, 0].set_xlim(0, 105)
        axes[0, 0].set_xlabel("Components (%)")
        axes[0, 0].set_title("Component state")
        axes[0, 0].legend(fontsize=8)

    if categorical_top:
        labels = [row["label"] for row in categorical_top]
        values = [row["share"] for row in categorical_top]
        counts = [row["count"] for row in categorical_top]
        bars = axes[0, 1].barh(labels, values, color="#8c5b9e")
        axes[0, 1].set_xlim(0, 105)
        axes[0, 1].set_xlabel("Modal value share (%)")
        axes[0, 1].set_title("Broadest categorical distributions")
        axes[0, 1].tick_params(axis="y", labelsize=7)
        annotate_bars(axes[0, 1], bars, labels, values, counts)
    else:
        axes[0, 1].text(0.5, 0.5, "No categorical / enum parameters", ha="center", va="center")
        axes[0, 1].set_axis_off()

    if continuous_top:
        labels = [row["label"] for row in continuous_top]
        values = [row["share"] for row in continuous_top]
        counts = [row["count"] for row in continuous_top]
        bars = axes[1, 0].barh(labels, values, color="#426b9a")
        axes[1, 0].set_xlim(0, 105)
        axes[1, 0].set_xlabel("Largest 64-bin share (%)")
        axes[1, 0].set_title("Broadest continuous distributions")
        axes[1, 0].tick_params(axis="y", labelsize=7)
        annotate_bars(axes[1, 0], bars, labels, values, counts)
    else:
        axes[1, 0].text(0.5, 0.5, "No continuous parameters", ha="center", va="center")
        axes[1, 0].set_axis_off()

    if categorical_shares:
        axes[1, 1].hist(categorical_shares, bins=np.linspace(0, 100, 11), alpha=0.7, color="#8c5b9e", edgecolor="white", label="categorical modal share")
    if continuous_shares:
        axes[1, 1].hist(continuous_shares, bins=np.linspace(0, 100, 11), alpha=0.7, color="#426b9a", edgecolor="white", label="continuous largest-bin share")
    axes[1, 1].set_xlim(0, 100)
    axes[1, 1].set_xlabel("Share of observations in modal value / largest bin (%)")
    axes[1, 1].set_ylabel("Parameters")
    axes[1, 1].set_title("Value concentration across this family")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(f"{COMPONENT_FAMILY_LABELS[family]}: component-level value analysis", y=1.01)
    fig.tight_layout()
    save_figure(fig, path)


def component_main_report_lines(data: dict[str, Any], weight: str, figure_link: Any) -> list[str]:
    aggregate = data[weight]
    registry, groups, family_refs = component_parameter_groups(data, weight)
    lines = [
        "## Component-by-component value analysis",
        "",
        "The global prevalence charts are orientation only. The sections below keep each component's scalar controls together, then separate categorical frequencies from continuous distributions. Every repeated slot has its own subsection; no oscillator, filter, effect, LFO, random source, or modulation slot is pooled into a misleading global top-20 list.",
        "",
        "The family chart above each section summarizes component state and the broadest observed value distributions within that family. The tables beneath it retain every parameter assigned to each component, including modal values, quantiles, default/zero prevalence, and dominant ranges.",
        "",
    ]
    for family in COMPONENT_FAMILY_ORDER:
        refs = family_refs.get(family, [])
        if not refs:
            continue
        lines.extend([
            f"### {COMPONENT_FAMILY_LABELS[family]}",
            "",
            COMPONENT_FAMILY_DESCRIPTIONS[family],
            "",
            f"**Figure {12 + COMPONENT_FAMILY_ORDER.index(family)}. {COMPONENT_FAMILY_LABELS[family]} component analysis.**",
            "",
            f"![{COMPONENT_FAMILY_LABELS[family]} component analysis]({figure_link(f'component_{family}_analysis')})",
            "",
        ])
        for ref in refs:
            definition = registry.get(ref)
            parameters = groups.get(ref, [])
            component = aggregate.get("components", {}).get(str(ref), {})
            categorical_count = sum(values.get("parameter_type") == "categorical" for _, values in parameters)
            continuous_count = sum(values.get("parameter_type") == "continuous" for _, values in parameters)
            lines.extend([
                f"#### {component_display_name(ref)}",
                "",
                f"{len(parameters):,} scalar parameters: {categorical_count:,} categorical/enum and {continuous_count:,} continuous. {component_state_text(ref, definition, component, aggregate['parsed_files'])}.",
                "",
            ])
            lines.extend(component_parameter_lines(ref, parameters))
        lines.extend(component_family_nested_lines(family, aggregate))
    return lines


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
    oscillator_counts = count_distribution(aggregate["family_count_distributions"].get("oscillator", {}))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.8, 1.0]})
    bars = axes[0].barh(pattern_labels[::-1], pattern_values[::-1], color="#8c5b9e")
    axes[0].set_title("Oscillator identity patterns")
    axes[0].set_xlabel("Presets (%)")
    axes[0].set_xlim(0, max(pattern_values, default=100) * 1.25)
    annotate_bars(axes[0], bars, pattern_labels[::-1], pattern_values[::-1], list(patterns.values())[::-1])
    count_labels = [str(value) for value, _ in oscillator_counts]
    count_values = [percentage(count, denominator) for _, count in oscillator_counts]
    bars = axes[1].bar(count_labels, count_values, color="#5d8c79")
    axes[1].set_title("Active oscillator count")
    axes[1].set_xlabel("Active oscillators per preset")
    axes[1].set_ylabel("Presets (%)")
    axes[1].tick_params(axis="x", rotation=45)
    for bar, count in zip(bars, [count for _, count in oscillator_counts]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Oscillator identity and count distributions (n={denominator:,})", y=1.01)
    fig.tight_layout()
    path = figures / "03_oscillator_patterns.png"
    save_figure(fig, path)
    paths["oscillator_patterns"] = path.name

    slot_groups = (
        ("Core components", ("oscillator", "sampler", "filter", "effect"), "#d18a43"),
        ("Modulation sources", ("envelope", "lfo", "random"), "#7b6a4e"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.15, 1.0]})
    for ax, (title, group_families, color) in zip(axes, slot_groups):
        rows = []
        for family in group_families:
            for slot, count in aggregate["slot_prevalence"].get(family, {}).items():
                label = family if family == "sampler" else f"{family}[{slot}]"
                rows.append((label, percentage(count, denominator)))
        rows.sort(key=lambda row: (row[1], row[0]))
        labels = [row[0] for row in rows]
        values = [row[1] for row in rows]
        bars = ax.barh(labels, values, color=color)
        ax.set_title(title)
        ax.set_xlabel("Presets with that specific component active (%)")
        ax.set_xlim(0, max(values, default=100) * 1.2)
        ax.tick_params(axis="y", labelsize=8)
        annotate_bars(ax, bars, labels, values)
    fig.suptitle(f"Repeated component identity prevalence (n={denominator:,})", y=1.01)
    fig.tight_layout()
    path = figures / "04_repeated_slot_identity.png"
    save_figure(fig, path)
    paths["repeated_slot_identity"] = path.name

    count_families = [("lfo", "Routed LFOs"), ("effect", "Enabled effects"), ("modulation_slot", "Live modulation routes")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, (family, title) in zip(axes, count_families):
        distribution = aggregate["family_count_distributions"].get(family, {})
        items = binned_count_distribution(distribution) if family == "modulation_slot" else [(str(key), value) for key, value in count_distribution(distribution)]
        bars = ax.bar([label for label, _ in items], [percentage(value, denominator) for _, value in items], color="#5d8c79")
        ax.set_title(title)
        ax.set_xlabel("Count per preset")
        ax.set_ylabel("Presets (%)")
        ax.tick_params(axis="x", rotation=45)
        for bar, (_, count) in zip(bars, items):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,}", ha="center", va="bottom", fontsize=7)
    fig.suptitle(f"How many routed LFOs, effects, and live modulation routes does a preset use? (n={denominator:,})", y=1.01)
    fig.tight_layout()
    path = figures / "05_count_distributions.png"
    save_figure(fig, path)
    paths["count_distributions"] = path.name

    params = parameter_rows(data, weight)
    top = sorted(params, key=lambda row: (-row["non_default"], row["name"]))[:20]
    top_conditional = sorted(params, key=lambda row: (-row["conditional"], row["name"]))[:20]
    fig, axes = plt.subplots(1, 3, figsize=(19, 8), gridspec_kw={"width_ratios": [1.0, 1.0, 0.9]})
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
    fig.suptitle(f"Scalar parameter prevalence: global and conditional views (n={denominator:,})", y=1.01)
    ranked = sorted((row for row in params if row["non_default_count"] > 0), key=lambda row: (-row["non_default_count"], row["name"]))
    total_edits = sum(row["non_default_count"] for row in ranked)
    cumulative = []
    running = 0
    for row in ranked:
        running += row["non_default_count"]
        cumulative.append(percentage(running, total_edits))
    ax = axes[2]
    if cumulative:
        ax.plot(range(1, len(cumulative) + 1), cumulative, color="#a44d58", linewidth=2)
    ax.set_title("Cumulative non-default observation coverage")
    ax.set_xlabel("Parameters ranked by non-default count")
    ax.set_ylabel("Share of all counted non-default observations (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figures / "06_07_parameter_prevalence_sparsity.png"
    save_figure(fig, path)
    paths["parameter_prevalence_sparsity"] = path.name

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

    registry, groups, family_refs = component_parameter_groups(data, weight)
    for figure_number, family in enumerate(COMPONENT_FAMILY_ORDER, start=12):
        refs = family_refs.get(family, [])
        if not refs:
            continue
        key = f"component_{family}_analysis"
        path = figures / f"{figure_number:02d}_{family}_component_analysis.png"
        make_component_family_figure(data, weight, family, registry, groups, refs, path)
        paths[key] = path.name

    return paths


def build_report(census_path: Path, report_path: Path, figures_path: Path, weight: str = "file_weighted") -> None:
    data = json.loads(census_path.read_text(encoding="utf-8"))
    aggregate = data[weight]
    denominator = aggregate["parsed_files"]
    unique_denominator = data["exact_deduplicated"]["parsed_files"]
    figures = make_figures(data, figures_path, weight)
    figure_link = lambda name: os.path.relpath(figures_path / figures[name], report_path.parent).replace(os.sep, "/")
    distribution_report_path = report_path.with_name("VITAL_PARAMETER_DISTRIBUTIONS.md")
    write_component_distribution_report(data, distribution_report_path, weight)
    distribution_report_link = os.path.relpath(distribution_report_path, report_path.parent).replace(os.sep, "/")

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
    component_lines = component_main_report_lines(data, weight, figure_link)
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
        "**Figure 1. Corpus overview.**",
        "",
        f"![Corpus overview]({figure_link('corpus_overview')})",
        "",
        f"Figure 1 uses the {weight.replace('_', ' ')} denominator (`n={denominator:,}`). Version bars are counts of parsed presets. The duplicate panel is a sensitivity check: later tables and plots retain both file-weighted and exact-deduplicated aggregates, rather than silently choosing one.",
        "",
        "The shared parameter space uses the pinned atlas defaults and treats a missing common scalar as its canonical default, matching Vital's load behavior. The 1.5.x spectral-morph phase fields, 1.6.x modulation-ramp fields, and the isolated `flanger_depth` extension are kept in version-eligible groups. Their defaults are not present in the pinned atlas, so the machine-readable output reports nonzero observations and marks default-qualified non-default percentages as unsupported rather than inventing a denominator or default.",
        "",
        "## Component activity and repeated-slot identity",
        "",
        "**Figure 2. Component family prevalence.**",
        "",
        f"![Component family prevalence]({figure_link('component_family_prevalence')})",
        "",
        "Figure 2 counts direct `*_on` state for oscillators, filters, effects, and the sampler. Envelopes, LFOs, and random sources use operational activity: their numbered slot is active only when it is the source of a live modulation route. This keeps direct enablement separate from routing.",
        "",
        "**Figure 3. Oscillator identity patterns and active oscillator counts.**",
        "",
        f"![Oscillator patterns]({figure_link('oscillator_patterns')})",
        "",
        "Figure 3 preserves exact oscillator identity masks and places the active-oscillator count distribution beside them. `2` means oscillator 2 only; it is not merged with `1` merely because both masks have cardinality one. Any skipped-slot mask visible in the figure is therefore an observed non-prefix pattern.",
        "",
        "**Figure 4. Core component versus modulation-source slot identity.**",
        "",
        f"![Repeated-slot identity]({figure_link('repeated_slot_identity')})",
        "",
        "Figure 4 separates independent numbered-slot prevalence into core components on the left (including filters) and modulation sources on the right. Figure 5 switches to per-preset count distributions for routed LFOs, effects, and live modulation routes; oscillator counts are shown beside the identity patterns in Figure 3.",
        "",
        "**Figure 5. Routed LFO, effect, and live modulation-route count distributions.**",
        "",
        f"![Count distributions]({figure_link('count_distributions')})",
        "",
        "The live-route panel keeps counts 0–20 as individual bars, then groups 21 and above into four-count bins so the long tail remains legible.",
        "",
        "## Scalar parameter sparsity and modulation",
        "",
        "**Figure 6. Scalar prevalence and conditional prevalence.**",
        "",
        "**Figure 7. Cumulative scalar sparsity (shown beside Figure 6).**",
        "",
        f"![Scalar prevalence and sparsity]({figure_link('parameter_prevalence_sparsity')})",
        "",
        "The left panels rank atlas-backed scalar changes globally and conditional on an active owner. The right panel is the cumulative head-versus-tail view: the x-axis is parameter rank by non-default count and the y-axis is the share of all counted non-default observations covered by that prefix. For LFO, envelope, random, and modulation-slot parameters, the conditional denominator is the operationally routed/connected slot population.",
        "",
        f"The detailed component-by-component value tables are in the companion [parameter distribution report]({distribution_report_link}), with complete categorical and continuous row exports beside it.",
        "",
        "**Figure 8. Live modulation source-family to destination-family routes.**",
        "",
        f"![Modulation family matrix]({figure_link('modulation_family_matrix')})",
        "",
        f"Figure 8 counts live routes only: connected routes that are not bypassed and have non-zero amount. The census separately retains connected-route source/destination vocabularies, connected, bypassed, zero-amount, non-zero-amount, bipolar, stereo, explicit-linear, and custom-remap counts. In this view, the plotted source/destination prevalence is therefore about operational routing, while a populated but bypassed or zero-amount connection remains visible in the supporting JSON.",
        "",
        "## Co-occurrence and nested state",
        "",
        "**Figure 9. Major-family co-occurrence.**",
        "",
        f"![Component co-occurrence]({figure_link('component_cooccurrence')})",
        "",
        "Figure 9 uses preset-level presence flags and normalized percentages. Diagonal cells are the family prevalence; off-diagonal cells are the percentage of presets containing both families. LFO, envelope, and random presence again means live source routing, not merely non-default scalar state.",
        "",
        "**Figure 10. Nested LFO, wavetable, sampler, and remap state.**",
        "",
        f"![Nested state]({figure_link('nested_state')})",
        "",
        "Figure 10 treats nested structures semantically. LFO shape comparison ignores display names and compares the drawable shape fields. Wavetables retain a canonical-init descriptor comparison in the JSON, but the figure classifies active slots conservatively by known stock names: named stock content is reported as `named_stock_or_unresolved_content`, while unknown names remain unresolved rather than being called custom from a byte comparison. This avoids mistaking a version/rendering difference in a stock asset for a user-authored wavetable. Preset bars use `n` parsed presets; routed-LFO and wavetable bars use routed/active slot denominators; remap bars use connected-route denominators.",
        "",
        "## Version-introduced features and duplicate sensitivity",
        "",
        f"The version-introduced parameter groups contain **{len(introduced):,} scalar names** in the pooled output. Their eligible denominators are restricted to versions that can contain them; older presets are never put in the denominator. Because the pinned atlas does not define their defaults, use the `nonzero` field for the neutral-state observation and do not read their `non_default` field as zero.",
        "",
        "**Figure 11. Duplicate-weighting sensitivity.**",
        "",
        f"![Duplicate sensitivity]({figure_link('duplicate_sensitivity')})",
        "",
        "Figure 11 compares the file-weighted and exact-deduplicated estimates for the most edited shared parameters. The JSON and parameter CSV retain the complete comparison, including numerator and denominator for every parameter; the figure is only a readable headline slice.",
        "",
        *component_lines,
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
        "The aggregate source of truth is [`vital_usage_census.json`](vital_usage_census.json); the complete scalar lookup table is [`vital_usage_parameters.csv`](vital_usage_parameters.csv), with row-level categorical and continuous distribution exports beside it.",
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
