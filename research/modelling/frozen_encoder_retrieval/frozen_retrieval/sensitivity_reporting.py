from __future__ import annotations

from pathlib import Path

from .core import MODEL_REGISTRY
from .sensitivity import MODEL_ORDER


def _arm_label(arm: str) -> str:
    model_name, variant = arm.split(":", 1)
    return f"{MODEL_REGISTRY[model_name]['name']} — `{variant}`"


def _format_ci(value: float | None, interval: list[float] | None, *, scale: float = 1.0,
               digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "not estimable"
    if interval is None:
        return f"{value * scale:.{digits}f}{suffix}"
    return f"{value * scale:.{digits}f}{suffix} [{interval[0] * scale:.{digits}f}, {interval[1] * scale:.{digits}f}]"


def _ratio_quantiles(arm: dict) -> str:
    values = arm["quantiles"]["ratio"]
    return f"{values['p50']:.3f} [{values['p10']:.3f}, {values['p90']:.3f}]"


def _relative_table(result: dict) -> str:
    lines = ["| Representation | Original common-gallery Recall@1 | Mean distance: same preset / different MIDI | Mean distance: different preset / same MIDI | Mean Δ (MIDI − preset) | Median ratio [P10, P90] | Mean normalized margin | Preset-dominant anchors |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm_name, arm in result["arms"].items():
        ci = arm["confidence_intervals_95"]
        lines.append(f"| {_arm_label(arm_name)} | {arm['retrieval_recall_at_1']:.1%} | "
                     f"{_format_ci(arm['mean_midi_change_distance'], ci['mean_midi_distance'])} | "
                     f"{_format_ci(arm['mean_preset_change_distance'], ci['mean_preset_distance'])} | "
                     f"{_format_ci(arm['mean_raw_difference_midi_minus_preset'], ci['mean_difference'])} | "
                     f"{_ratio_quantiles(arm)} | "
                     f"{_format_ci(arm['mean_normalized_margin'], ci['mean_margin'])} | "
                     f"{_format_ci(arm['preset_dominant_fraction'], ci['preset_dominant_fraction'], scale=100, digits=1, suffix='%')} |")
    return "\n".join(lines)


def _variance_table(result: dict) -> str:
    lines = ["| Representation | Preset variation | MIDI-performance variation | Interaction variation |",
             "|---|---:|---:|---:|"]
    for arm_name, arm in result["arms"].items():
        variance = arm["variance_decomposition"]
        intervals = variance["confidence_intervals_95"]
        shares = variance["shares"]
        lines.append(f"| {_arm_label(arm_name)} | "
                     f"{_format_ci(shares['preset'], intervals['preset'], scale=100, digits=1, suffix='%')} | "
                     f"{_format_ci(shares['midi'], intervals['midi'], scale=100, digits=1, suffix='%')} | "
                     f"{_format_ci(shares['interaction'], intervals['interaction'], scale=100, digits=1, suffix='%')} |")
    return "\n".join(lines)


def _cohort_spread_table(result: dict) -> str:
    lines = ["| Representation | Per-preset mean margin P10 / P50 / P90 | Per-performance mean margin P10 / P50 / P90 |",
             "|---|---:|---:|"]
    for arm_name, arm in result["arms"].items():
        preset = arm["per_preset_mean_margin_quantiles"]
        midi = arm["per_midi_mean_margin_quantiles"]
        lines.append(f"| {_arm_label(arm_name)} | {preset['p10']:+.3f} / {preset['p50']:+.3f} / {preset['p90']:+.3f} | "
                     f"{midi['p10']:+.3f} / {midi['p50']:+.3f} / {midi['p90']:+.3f} |")
    return "\n".join(lines)


def _paired_table(result: dict) -> str:
    baseline = "CIR single-timbre AST — `official_clip`"
    lines = [f"Paired against {baseline}; estimates are differences between observed full-cohort summaries, while intervals use the paired crossed-bootstrap differences. Positive changes mean a larger preset-over-MIDI margin or a larger preset-dominant share.",
             "",
             "| Representation | Δ mean margin [paired 95% CI] | Δ preset-dominant share, percentage points [paired 95% CI] |",
             "|---|---:|---:|"]
    for arm_name, metrics in result["paired_differences_vs_cir"].items():
        margin = metrics["mean_margin"]
        dominant = metrics["preset_dominant_fraction"]
        lines.append(f"| {_arm_label(arm_name)} | {_format_ci(margin['difference'], margin['confidence_interval_95'])} | "
                     f"{_format_ci(dominant['difference'], dominant['confidence_interval_95'], scale=100, digits=1, suffix=' pp')} |")
    return "\n".join(lines)


def _diagnostic_table(result: dict) -> str:
    lines = ["| Representation | Anchors | Undefined ratios | Near-zero preset-distance denominators | Both distances near zero | Ties | Zero-norm vectors | Nonfinite values |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm_name, arm in result["arms"].items():
        vector = arm["vector_validation"]
        lines.append(f"| {_arm_label(arm_name)} | {arm['anchored_observations']:,} | "
                     f"{arm['anchored_observations'] - arm['ratio_defined_count']:,} | "
                     f"{arm['near_zero_preset_distance_denominator_count']:,} | "
                     f"{arm['both_distances_near_zero_count']:,} | {arm['tie_count']:,} | "
                     f"{vector['zero_norm_vectors']:,} | {vector['nonfinite_values'] + arm['nonfinite_metric_values']:,} |")
    return "\n".join(lines)


def build_issue157_section(result: dict) -> str:
    arms = result["arms"]
    margin_leader = max(arms, key=lambda arm: arms[arm]["mean_normalized_margin"])
    retrieval_leader = max(arms, key=lambda arm: arms[arm]["retrieval_recall_at_1"])
    leading = arms[margin_leader]
    retrieval = arms[retrieval_leader]
    dac_masked = arms["dac_44khz:masked_mean"]
    dac_margin_ci = dac_masked["confidence_intervals_95"]["mean_margin"]
    dac_paired_ci = result["paired_differences_vs_cir"]["dac_44khz:masked_mean"]["mean_margin"]["confidence_interval_95"]
    encodec_masked = arms["encodec_24khz:masked_mean"]
    cohort = result["cohort"]
    bootstrap = result["statistics"]
    if margin_leader == retrieval_leader:
        ranking_readout = (f"The same tested representation led both measures: {_arm_label(margin_leader)} had the largest mean normalized margin "
                           f"({leading['mean_normalized_margin']:+.3f}) and the strongest original common-gallery Recall@1 ({retrieval['retrieval_recall_at_1']:.1%}).")
    else:
        ranking_readout = (f"The measures ranked representations differently: {_arm_label(retrieval_leader)} led original common-gallery Recall@1 "
                           f"({retrieval['retrieval_recall_at_1']:.1%}), while {_arm_label(margin_leader)} had the largest mean normalized margin "
                           f"({leading['mean_normalized_margin']:+.3f}; preset-dominant on {leading['preset_dominant_fraction']:.1%} of anchors).")
    figure_links = "\n\n".join(f"![{label}](figures/{filename})" for label, filename in (
        ("All eleven representations across sensitivity and variation metrics", "issue157_sensitivity_heatmap.png"),
        ("Anchored ratio and normalized-margin distributions", "issue157_distance_distributions.png"),
        ("Preset, MIDI, and interaction variation for every representation", "issue157_variance_decomposition.png"),
        ("Illustrative low- and higher-interaction geometry; synthetic points, not measurements", "issue157_interaction_geometry.png")))
    return f"""## Issue #157 — Preset versus MIDI sensitivity

**Finding:** The question is whether changing a Vital preset moves a frozen audio representation more than changing the MIDI performance. {ranking_readout} The comparison uses the same complete {cohort['preset_states']} preset states and {cohort['midi_performances']} distinct performances for every arm. These measurements describe representation geometry; they do not show that Vital settings can be decoded or that reconstructed audio would sound accurate.

The preset states come from the existing `validation` membership, and the 50 performances come from `train`; they are 129 source-family clusters and 50 distinct MIDI identities. Each matrix cell is one accepted cached render. The test changes several MIDI characteristics together; it does not isolate pitch.

### Four recordings make the comparison concrete

Take preset A playing performance 1 (`A1`). Compare it with A playing a different performance (`A2`) and preset B playing the same first performance (`B1`). The first distance asks how much the representation moves when the performance changes but the preset stays fixed. The second asks how much it moves when the preset changes but the performance stays fixed. Repeat the calculation from `A2`, `B1`, and `B2` separately; the four anchors are kept distinct until each metric is calculated.

Distances below are cosine distances after every cached feature vector is L2-normalized. A smaller distance means the two recordings land closer together in the representation. Let `dM` be the same-preset/different-performance distance and `dP` the different-preset/same-performance distance. The ratio is `dM / dP`, the raw difference is `dM − dP`, and the normalized margin is `(dP − dM) / (dP + dM + ε)`. A ratio below 1, a negative raw difference, and a positive margin all mean the preset change was larger. A ratio above 1, positive difference, and negative margin mean the performance change was larger. Values near 1, 0, and 0 indicate similar-sized changes.

**Illustration, not a measured pair:** if `dM = 0.10` and `dP = 0.50`, the ratio is `0.20`, the difference is `−0.40`, and the margin is about `+0.67`: this representation moved five times as far for the preset change. In the reverse example, `dM = 0.50` and `dP = 0.10` give a ratio of `5.0`, difference `+0.40`, and margin about `−0.67`; MIDI changed it more. If both distances are `0.25`, the ratio is `1`, the difference and margin are `0`, and the effects tie.

### Measured anchored comparisons

The two distance columns show their absolute scale as well as their contrast. This matters because an encoder whose normalized features barely move at all can have a favorable ratio without being a useful timbre representation. Original common-gallery Recall@1 is retained beside the new measures for that reason. It uses the primary single-assignment setup (81.4% for CIR), distinct from Issue #155's later three-assignment average (84.9%). Each mean and its interval is paired across the same crossed preset-family and MIDI-performance resamples.

{_relative_table(result)}

The practical ranking depends on the question. CIR had the strongest original common-gallery Recall@1 ({retrieval['retrieval_recall_at_1']:.1%}) and was preset-dominant on {retrieval['preset_dominant_fraction']:.1%} of anchors, with a mean margin of {retrieval['mean_normalized_margin']:+.3f}. DAC `masked_mean` had a larger mean margin ({dac_masked['mean_normalized_margin']:+.3f} [{dac_margin_ci[0]:+.3f}, {dac_margin_ci[1]:+.3f}]); its paired difference from CIR was {result['paired_differences_vs_cir']['dac_44khz:masked_mean']['mean_margin']['difference']:+.3f} [{dac_paired_ci[0]:+.3f}, {dac_paired_ci[1]:+.3f}]. But DAC's original common-gallery Recall@1 was only {dac_masked['retrieval_recall_at_1']:.1%}, so stronger average separation did not make it the better retrieval encoder. EnCodec `masked_mean` illustrates why ratios need an absolute scale: its median ratio was {encodec_masked['quantiles']['ratio']['p50']:.3f}, but its mean MIDI and preset distances were only {encodec_masked['mean_midi_change_distance']:.4f} and {encodec_masked['mean_preset_change_distance']:.4f}, with {encodec_masked['retrieval_recall_at_1']:.1%} original common-gallery Recall@1.

The population above is exhaustively derived from `{cohort['preset_states']} × {cohort['midi_performances']} = {cohort['matrix_cells']}` cached recordings per representation: `{cohort['unordered_2x2_combinations']:,}` unordered choices of two presets and two performances, with four separate anchors each. Those {cohort['anchored_observations_per_arm']:,} dependent anchor observations are not independent samples. The 95% intervals resample the {cohort['preset_family_clusters']} source-family clusters and 50 MIDI identities in crossed draws shared by every arm. For anchored distances, {bootstrap['sampled_unordered_2x2_combinations_per_bootstrap_replicate']:,} combinations approximate the full pair average using two distinct original presets and two distinct original performances. If a bootstrap draw contains preset A twice and preset B once, the A–B pair receives twice the weight; A–A is excluded. Variance-share intervals resample the full crossed matrix using the same cluster draws. Full-cohort point summaries use every combination. The subset is a deterministic, coverage-balanced performance cohort, so these intervals do not replace an independent confirmation cohort.

The next table shows how the average margin varies across individual presets and performances. These percentile ranges describe the selected cohort; they are not extra independent-sample confidence intervals.

{_cohort_spread_table(result)}

**Preset-dominant** means `dP` exceeds `dM` by more than `1e-6` cosine-distance units. Differences within that tolerance are ties. The ratio is undefined when `dP ≤ 1e-6`; those cases are excluded from ratio summaries and counted below. The normalized-margin denominator uses `ε = 1e-8`.

{_diagnostic_table(result)}

### Where the representation changes come from

The decomposition asks how normalized embedding positions vary over the full balanced preset-by-performance matrix. **Preset variation** means presets occupy consistently different regions even while the MIDI changes. **MIDI variation** means changing the performance shifts the representations in a similar way across presets. **Interaction** is what remains when preset and MIDI averages are removed: for example, A may move upward from MIDI 1 to MIDI 2 while B moves sideways. These are shares of squared Euclidean variation among the unit-length cached vectors, not percentages of recoverable information.

**Synthetic geometry examples, not measured data:** in the left panel, both presets move by the same vector when MIDI changes, so the expected decomposition is 80% preset, 20% MIDI, and 0% interaction. In the right panel, A moves upward while B moves sideways; the expected shares are 86.7% preset, 6.7% MIDI, and 6.7% interaction. Each arm’s measured components add to its total variation within the reported numeric tolerance.

{_variance_table(result)}

{_paired_table(result)}

The largest-margin arm is {"also" if margin_leader == retrieval_leader else "not"} the arm with the strongest existing retrieval R@1. A sensitivity ranking is not a replacement for retrieval: high preset sensitivity alone does not establish decodability, useful control predictions, or audio reconstruction. The interaction term is not automatically undesirable; it can represent real preset-specific responses to pitch, rhythm, duration, velocity, or articulation. With one cached render per preset-performance cell, it also includes render-specific variation, and there is no separate noise term.

### Coverage limits and next decision

- The 50 performances include pitch, rhythm, duration, velocity, articulation, and other events together; this experiment does not isolate pitch.
- No model-specific “difficult preset” subset was selected. Existing source-family metadata assigns one family to each preset but provides no independent difficulty labels, so using retrieval scores or distances to choose a hard subset would have made the comparison model-dependent.
- This is an exploratory analysis of one selected 129-preset × 50-performance cohort. No reserved test or independent-confirmation data were used.
- No decoder, prediction head, training, new render, or reconstruction test was run. High preset sensitivity does not show that a decoder can recover Vital settings.

Use these results to select a small, controlled reconstruction comparison; keep any architecture decision gated on held-out reconstruction quality. The [machine-readable results and provenance](issue157_sensitivity_results.json) retain all eleven arm summaries and the exact analysis settings.

### Issue #157 visualizations

Each heatmap cell is colored on a within-column relative scale; the printed cell value is the actual metric. Distribution lines summarize all defined ratios or all margins, not independent observations. The stacked bars include every representation arm. The geometry panels are synthetic examples only.

{figure_links}
"""


def build_execution_addendum(result: dict) -> str:
    provenance = result["cache_validation"]["checkpoint_provenance"]
    lines = ["### Issue #157 execution and reproducibility", "",
             "The sensitivity analysis reused the persisted frozen embeddings only. Cache verification ran before scoring and checked all archive SHA-256 values; each feature identity’s checkpoint and source revision, loader, tap, sample rate, preprocessing and pooling; the render request and WAV hash; and every array’s shape, dtype, finite values and nonzero norm. The seven families each contained 6,480 accepted cache entries; 6,450 mapped to the common matrix used here. No encoder weights were loaded and no new inference or rendering was run.", "",
             f"The analyzed matrix contains `{result['cohort']['preset_states']} × {result['cohort']['midi_performances']} × D` vectors per arm, normalized in-place on `{result['compute']['device']}` as `{result['compute']['dtype']}`. Pairwise cosine distances were precomputed in tensors shaped `[129, 50, 50]` and `[50, 129, 129]`; exhaustive combinations were processed in chunks of {result['compute']['preset_pair_chunk_size']} preset pairs. Total scoring and bootstrap time was {result['compute']['elapsed_seconds']:.1f} seconds. The bootstrap used {result['statistics']['bootstrap_replicates']:,} crossed resamples; pair metrics sampled {result['statistics']['sampled_unordered_2x2_combinations_per_bootstrap_replicate']:,} weighted distinct pairs, and variance shares used the full crossed matrix.",
             "",
             "| Encoder family | Checkpoint SHA-256 | Pinned source revision | Loader revision | Feature tap | Cached arms / dimensions | Pooling |",
             "|---|---|---|---|---|---|---|"]
    for model_name in MODEL_ORDER:
        row = provenance[model_name]
        outputs = "; ".join(f"`{variant}` {dimension:,}-D" for variant, dimension in row["dimensions_by_variant"].items())
        lines.append(f"| {MODEL_REGISTRY[model_name]['name']} | `{row['checkpoint_sha256']}` | `{row['source_revision']}` | "
                     f"`{row['loader_revision']}` | {row['tap']} | {outputs} | `{row['pooling']}` |")
    lines.extend(("", "The source preset membership remained `validation`; the distinct performance source remained `train`. Reserved test and confirmation splits were not accessed. Inputs consisted only of existing private manifests, receipts, cache metadata, and feature arrays; raw IDs, source paths, audio samples, and embeddings were not added to the report or repository.", "",
                  "Reproduce from the repository root with the existing local cache and pinned checkpoint lock:", "",
                  "```powershell", "python research/modelling/frozen_encoder_retrieval/run_issue157_sensitivity.py `",
                  "  --dataset research/modelling/frozen_encoder_retrieval/outputs/issue152-validation-train-midi-v3-50 `",
                  "  --checkpoint-root research/modelling/frozen_encoder_retrieval/local_checkpoints `",
                  f"  --device {result['compute']['device']} --bootstrap-replicates {result['statistics']['bootstrap_replicates']} `",
                  f"  --bootstrap-combinations {result['statistics']['sampled_unordered_2x2_combinations_per_bootstrap_replicate']}",
                  "```", ""))
    return "\n".join(lines)


def update_research_report(path: Path, result: dict) -> None:
    text = path.read_text(encoding="utf-8")
    section_marker = "## Issue #157 — Preset versus MIDI sensitivity"
    anchor = "\n### Visual summaries\n"
    if section_marker in text:
        start = text.index(section_marker)
        legacy_anchor = "\n### Visual summaries\n\n\n**Encoder × metric heatmap**"
        end = text.find(legacy_anchor, start)
        if end == -1:
            end = text.index(anchor, start)
        text = text[:start].rstrip() + "\n" + text[end + 1:]
    if anchor not in text:
        raise ValueError("research report does not contain the expected visual-summary insertion point")
    section = build_issue157_section(result)
    text = text.replace(anchor, f"\n{section}\n{anchor.lstrip()}", 1)

    execution_marker = "### Issue #157 execution and reproducibility"
    if execution_marker in text:
        text = text[:text.index(execution_marker)].rstrip() + "\n"
    text = text.rstrip() + "\n\n" + build_execution_addendum(result)
    path.write_text(text, encoding="utf-8", newline="\n")
