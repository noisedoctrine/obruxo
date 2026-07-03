"""Experiment 4 orchestration: phase-factorized mixed residual codebooks."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .phase4 import (
    PhaseChain,
    PhaseEncoding,
    circular_shift,
    compose_additive,
    compose_partitioned,
    compose_switch,
    decode_phase_chain,
    encode_phase_chain,
    quantize_phases,
    refine_encoding,
    train_phase_endpoints,
)
from .stacked import TOPOLOGY_NAMES, load_curve_dataset, load_stock_curves, metric_arrays


def conditions_for(dataset, chain: PhaseChain, validation: bool = True) -> np.ndarray:
    indices = dataset.validation_indices if validation else dataset.train_indices
    if chain.topology_conditioned:
        return dataset.topology[indices].astype(np.int32)
    return np.zeros(len(indices), np.int32)


def output_cost(chain: PhaseChain, *, base_phase: bool, residual_phase: bool, gains: bool) -> tuple[int, float]:
    dense = len(chain.bases) + sum(chain.stage_widths)
    bits = math.log2(len(chain.bases)) + sum(math.log2(width) for width in chain.stage_widths)
    if chain.topology_conditioned:
        dense += 3
        bits += math.log2(3)
    dense += int(base_phase) + len(chain.stages) * (int(residual_phase) + int(gains))
    return dense, bits


def configuration_name(chain: PhaseChain, base_phase: bool, residual_phase: bool, gains: bool) -> str:
    controls = []
    if gains:
        controls.append("gain")
    if base_phase:
        controls.append("basephase")
    if residual_phase:
        controls.append("resphase")
    return chain.name + "_" + ("_".join(controls) if controls else "categorical")


def evaluate_chain(
    dataset,
    chain: PhaseChain,
    *,
    base_phase: bool,
    residual_phase: bool,
    gains: bool,
    beam_width: int,
    refine: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, PhaseEncoding]:
    targets = dataset.validation_curves
    conditions = conditions_for(dataset, chain)
    encoding = encode_phase_chain(
        targets, chain, conditions, base_phase=base_phase,
        residual_phase=residual_phase, gains=gains, beam_width=beam_width,
    )
    if refine:
        encoding = refine_encoding(
            targets, chain, encoding, conditions, base_phase=base_phase,
            residual_phase=residual_phase, gains=gains,
        )
    reconstructed, _, _ = decode_phase_chain(chain, encoding, conditions)
    metrics = metric_arrays(targets, reconstructed)
    dense, bits = output_cost(
        chain, base_phase=base_phase, residual_phase=residual_phase, gains=gains
    )
    validation = dataset.frame.iloc[dataset.validation_indices].reset_index(drop=True)
    name = configuration_name(chain, base_phase, residual_phase, gains)
    result = pd.DataFrame(
        {
            "preset_id": validation.preset_id,
            "author_id": validation.author_id,
            "shape_signature": validation.shape_signature,
            "topology": [TOPOLOGY_NAMES[value] for value in dataset.topology[dataset.validation_indices]],
            "configuration": name,
            "strategy": chain.name,
            "base_phase_enabled": base_phase,
            "residual_phase_enabled": residual_phase,
            "gains_enabled": gains,
            "dense_dimensions": dense,
            "effective_bits": bits,
            "stored_floats": chain.stored_floats,
            "base_index": encoding.base_indices,
            "base_phase": encoding.base_phases,
            **metrics,
        }
    )
    for index, label in enumerate(chain.stage_labels):
        result[f"stage_{index+1}_index"] = encoding.stage_indices[index]
        result[f"stage_{index+1}_phase"] = encoding.stage_phases[index]
        result[f"stage_{index+1}_gain"] = encoding.stage_gains[index]
        result[f"stage_{index+1}_label"] = label

    usage = []
    for stage_index, indices in enumerate(encoding.stage_indices):
        for condition in range(chain.stages[stage_index].shape[0]):
            members = conditions == condition
            counts = np.bincount(indices[members], minlength=chain.stage_widths[stage_index])
            for code, count in enumerate(counts):
                usage.append(
                    {
                        "configuration": name,
                        "stage": chain.stage_labels[stage_index],
                        "condition": TOPOLOGY_NAMES[condition] if chain.topology_conditioned else "shared",
                        "code": code,
                        "uses": int(count),
                        "is_noop": code == 0,
                    }
                )
    return result, pd.DataFrame(usage), encoding


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "configuration", "strategy", "base_phase_enabled", "residual_phase_enabled",
        "gains_enabled", "dense_dimensions", "effective_bits", "stored_floats",
    ]
    return (
        results.groupby(keys, as_index=False)
        .agg(
            shapes=("shape_signature", "size"), rmse_median=("rmse", "median"),
            rmse_mean=("rmse", "mean"), rmse_p90=("rmse", lambda x: x.quantile(.90)),
            rmse_p95=("rmse", lambda x: x.quantile(.95)), rmse_p99=("rmse", lambda x: x.quantile(.99)),
            max_error_p95=("max_abs_error", lambda x: x.quantile(.95)),
        )
        .sort_values(["dense_dimensions", "rmse_p95", "rmse_median"])
        .reset_index(drop=True)
    )


def pareto(summary: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for index, row in summary.iterrows():
        dominated = (
            (summary.dense_dimensions <= row.dense_dimensions)
            & (summary.rmse_median <= row.rmse_median)
            & (summary.rmse_p95 <= row.rmse_p95)
            & ((summary.dense_dimensions < row.dense_dimensions)
               | (summary.rmse_median < row.rmse_median)
               | (summary.rmse_p95 < row.rmse_p95))
        ).any()
        if not dominated:
            selected.append(index)
    return summary.loc[selected].sort_values("dense_dimensions").reset_index(drop=True)


def phase_quantization(dataset, chain: PhaseChain, encoding: PhaseEncoding) -> pd.DataFrame:
    conditions = conditions_for(dataset, chain)
    rows = []
    for bins in (8, 16, 32, 64):
        quantized = quantize_phases(encoding, bins)
        reconstructed, _, _ = decode_phase_chain(chain, quantized, conditions)
        rmse = metric_arrays(dataset.validation_curves, reconstructed)["rmse"]
        rows.append({"bins": bins, "rmse_median": np.median(rmse), "rmse_p95": np.quantile(rmse, .95)})
    return pd.DataFrame(rows)


def _select_visual_rows(results: pd.DataFrame) -> list[int]:
    selected = []
    for topology in TOPOLOGY_NAMES:
        members = results[results.topology == topology]
        if members.empty:
            continue
        ordered = members.sort_values("rmse")
        common_signature = members.shape_signature.value_counts().index[0]
        common = members[members.shape_signature == common_signature].sort_values("rmse")
        selected.extend([
            int(common.index[len(common) // 2]),
            int(ordered.index[len(ordered) // 2]),
            int(ordered.index[min(len(ordered) - 1, int(len(ordered) * .95))]),
        ])
    return selected


def create_layer_visuals(
    dataset,
    chain: PhaseChain,
    results: pd.DataFrame,
    encoding: PhaseEncoding,
    output_dir: Path,
) -> None:
    visual_dir = output_dir / "layer_visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    conditions = conditions_for(dataset, chain)
    reconstructed, cumulative, transformed = decode_phase_chain(chain, encoding, conditions)
    phase = np.linspace(0.0, 1.0, dataset.validation_curves.shape[1], endpoint=False)
    rows = _select_visual_rows(results)
    colors = plt.cm.tab10(np.linspace(0, 1, max(5, len(chain.stages) + 1)))

    overview, axes = plt.subplots(len(rows), 2, figsize=(14, max(3, 2.2 * len(rows))), squeeze=False)
    for plot_row, row in enumerate(rows):
        target = dataset.validation_curves[row]
        axes[plot_row, 0].plot(phase, target, color="black", lw=1.6, label="target")
        for index, value in enumerate(cumulative):
            axes[plot_row, 0].plot(phase, value[row], color=colors[index], alpha=.8, label=f"prefix {index}")
        axes[plot_row, 0].axvline(0, color="gray", ls="--", lw=.8)
        axes[plot_row, 0].set_title(f"{results.iloc[row].topology} | RMSE {results.iloc[row].rmse:.4f}")
        axes[plot_row, 1].plot(phase, target, color="black", lw=1.8, label="target")
        axes[plot_row, 1].plot(phase, reconstructed[row], color="#d62728", lw=1.3, label="final")
        axes[plot_row, 1].fill_between(phase, target, reconstructed[row], color="#d62728", alpha=.12)
        for axis in axes[plot_row]:
            axis.set_xlim(0, 1)
            axis.set_ylim(-.05, 1.05)
            axis.grid(alpha=.15)
    axes[0, 0].legend(ncol=min(6, len(chain.stages) + 2), fontsize=7)
    axes[0, 1].legend(fontsize=8)
    overview.tight_layout()
    overview.savefig(output_dir / "layer_decomposition_examples.png", dpi=170)
    plt.close(overview)

    for ordinal, row in enumerate(rows):
        figure, axes = plt.subplots(len(chain.stages) + 2, 2, figsize=(14, 2.35 * (len(chain.stages) + 2)))
        base = chain.bases[encoding.base_indices[row]]
        shifted_base = circular_shift(base, encoding.base_phases[row])
        axes[0, 0].plot(phase, base, color=colors[0])
        axes[0, 0].set_title(f"Canonical base {encoding.base_indices[row]}")
        axes[0, 1].plot(phase, shifted_base, color=colors[0])
        axes[0, 1].set_title(f"Base shifted {encoding.base_phases[row] * 360:.1f}°")
        for stage_index, stage in enumerate(chain.stages):
            code_index = encoding.stage_indices[stage_index][row]
            code = stage[conditions[row], code_index]
            axes[stage_index + 1, 0].plot(phase, code, color=colors[stage_index + 1])
            axes[stage_index + 1, 0].set_title(
                f"{chain.stage_labels[stage_index]} code {code_index}"
                + (" (no-op)" if code_index == 0 else "")
            )
            axes[stage_index + 1, 1].plot(phase, transformed[stage_index][row], color=colors[stage_index + 1])
            axes[stage_index + 1, 1].set_title(
                f"Applied: gain {encoding.stage_gains[stage_index][row]:.3f}, "
                f"phase {encoding.stage_phases[stage_index][row] * 360:.1f}°"
            )
        axes[-1, 0].plot(phase, dataset.validation_curves[row], color="black", label="target")
        for index, value in enumerate(cumulative):
            axes[-1, 0].plot(phase, value[row], color=colors[index], alpha=.75, label=f"prefix {index}")
        axes[-1, 0].set_title("Cumulative reconstruction")
        axes[-1, 1].plot(phase, dataset.validation_curves[row], color="black", label="target")
        axes[-1, 1].plot(phase, reconstructed[row], color="#d62728", label="reconstruction")
        axes[-1, 1].set_title(f"Final RMSE {results.iloc[row].rmse:.6f}")
        for axis in axes.flat:
            axis.axvline(0, color="gray", ls="--", lw=.7)
            axis.set_xlim(0, 1)
            axis.grid(alpha=.15)
        axes[-1, 1].legend()
        figure.tight_layout()
        figure.savefig(visual_dir / f"example_{ordinal+1:02d}_{results.iloc[row].topology}.svg")
        plt.close(figure)

    atlas, axes = plt.subplots(len(chain.stages), 1, figsize=(13, 2.5 * len(chain.stages)), squeeze=False)
    for stage_index, stage in enumerate(chain.stages):
        axis = axes[stage_index, 0]
        for code in range(1, min(9, stage.shape[1])):
            value = stage[0, code]
            axis.plot(phase, value + (code - 1) * 1.1, lw=1, label=f"code {code}")
            axis.plot(phase, circular_shift(value, .125) + (code - 1) * 1.1, lw=.7, ls="--", alpha=.65)
        axis.set_title(f"{chain.stage_labels[stage_index]}: canonical (solid) and +45° use (dashed)")
        axis.set_yticks([])
        axis.axvline(0, color="gray", ls="--", lw=.7)
    atlas.tight_layout()
    atlas.savefig(output_dir / "codebook_phase_atlas.png", dpi=170)
    plt.close(atlas)


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    for column in selected.select_dtypes(include=["float"]).columns:
        selected[column] = selected[column].map(lambda x: f"{x:.6f}")
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
        *["| " + " | ".join(map(str, row)) + " |" for row in selected.itertuples(index=False, name=None)],
    ])


def verify_pytorch_environment(output_dir: Path) -> dict[str, object]:
    import torch

    available = bool(torch.xpu.is_available())
    device = torch.device("xpu:0" if available else "cpu")
    report: dict[str, object] = {
        "torch_version": torch.__version__, "xpu_available": available,
        "xpu_device_count": int(torch.xpu.device_count()), "selected_device": str(device),
    }
    if available:
        report["device_name"] = torch.xpu.get_device_name(0)
    torch.manual_seed(20260622)
    if available:
        torch.xpu.manual_seed_all(20260622)
    convolution = torch.nn.Conv1d(
        1, 32, 9, padding=4, padding_mode="circular"
    ).to(device)
    values = torch.rand(64, 1, 1024, device=device, requires_grad=True)
    if available:
        torch.xpu.synchronize()
    started = time.perf_counter()
    output = convolution(values)
    spectrum = torch.fft.rfft(output, dim=-1)
    restored = torch.fft.irfft(spectrum, n=output.shape[-1], dim=-1)
    loss = restored.square().mean()
    loss.backward()
    if available:
        torch.xpu.synchronize()
    report.update({
        "circular_conv_forward_backward": True,
        "fft_roundtrip_max_error": float(torch.max(torch.abs(restored - output)).detach().cpu()),
        "smoke_elapsed_seconds": time.perf_counter() - started,
        "output_shape": list(output.shape),
        "finite_gradients": bool(torch.isfinite(convolution.weight.grad).all().cpu()),
    })
    report["smoke_passed"] = bool(
        report["fft_roundtrip_max_error"] < 1e-4 and report["finite_gradients"]
    )
    (output_dir / "pytorch_environment.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def write_findings(summary: pd.DataFrame, frontier: pd.DataFrame, quantized: pd.DataFrame,
                   environment: dict[str, object], output: Path) -> None:
    best_median = summary.loc[summary.rmse_median.idxmin()]
    best_tail = summary.loc[summary.rmse_p95.idxmin()]
    content = f"""# Experiment 4 Findings: Phase-Factorized Residual Codebooks

## What did this experiment ask?

Can a base or residual correction be reused at an arbitrary circular phase instead of spending separate categorical codes on translated copies, and can shared and topology-specific dictionaries be combined economically?

## How was phase handled?

Codebook fitting and encoding used circular correlation at 128 phases, followed by full-resolution local refinement. Every canonical code was oriented so phase zero produced its largest frequency-weighted global improvement. One continuous phase scalar was used for the base and each active residual stage.

## Which configurations are Pareto-efficient?

{_table(frontier, ['configuration', 'dense_dimensions', 'stored_floats', 'rmse_median', 'rmse_p95'])}

Best median: `{best_median.configuration}` at {best_median.rmse_median:.6f}.  
Best P95: `{best_tail.configuration}` at {best_tail.rmse_p95:.6f}.

## How much phase precision was needed?

{_table(quantized, ['bins', 'rmse_median', 'rmse_p95'])}

## Is the PyTorch environment ready for neural experiments?

The neural predictor was intentionally deferred to Experiment 5. Environment verification selected `{environment['selected_device']}` ({environment.get('device_name', 'CPU')}) using PyTorch `{environment['torch_version']}`. Circular Conv1d forward/backward and FFT round-trip tests passed: `{environment['smoke_passed']}`. The full machine-readable check is in `pytorch_environment.json`.

## Layer decomposition

![Representative LFO layer decompositions](layer_decomposition_examples.png)

Solid curves in the codebook atlas are canonical codes; dashed curves show the same code translated by 45 degrees. Detailed SVGs include canonical and transformed components, gains, phase offsets, cumulative prefixes, and final error.

## Full geometry summary

{_table(summary, ['configuration', 'dense_dimensions', 'stored_floats', 'rmse_median', 'rmse_p95', 'rmse_p99'])}

## Proposed research direction for Experiment 5

Experiment 5 should test inferability rather than expand the oracle codec grid again:

1. Freeze Grid64 and the three useful phase-aware points: shared at 105 outputs, the best shared/topology hybrid near 108, and additive K8+K8 at 116.
2. Train circular 1D CNN and parameter-matched non-convolutional baselines on XPU, first from dense curves and then from controlled rendered modulation audio.
3. Optimize decoded reconstruction and rendered-effect loss as primary objectives. Exact code accuracy should remain diagnostic because multiple code/phase/gain paths can reconstruct the same curve.
4. Measure median and tail reconstruction, circular phase error, topology routing, code stability under small perturbations, and performance by modulation destination/rate/depth.
5. Fit predicted dense curves back to valid Vital points and powers before deciding which representation enters the full audio-to-preset model.
"""
    output.write_text(content, encoding="utf-8")


def encoding_from_results(frame: pd.DataFrame, chain: PhaseChain) -> PhaseEncoding:
    return PhaseEncoding(
        frame.base_index.to_numpy(dtype=np.int16), frame.base_phase.to_numpy(dtype=np.float32),
        [frame[f"stage_{i+1}_index"].to_numpy(dtype=np.int16) for i in range(len(chain.stages))],
        [frame[f"stage_{i+1}_phase"].to_numpy(dtype=np.float32) for i in range(len(chain.stages))],
        [frame[f"stage_{i+1}_gain"].to_numpy(dtype=np.float32) for i in range(len(chain.stages))],
    )


def finalize_from_artifacts(output_dir: Path) -> dict[str, object]:
    summary = pd.read_csv(output_dir / "summary.csv")
    frontier = pd.read_csv(output_dir / "pareto.csv")
    quantized = pd.read_csv(output_dir / "phase_quantization.csv")
    environment = verify_pytorch_environment(output_dir)
    write_findings(summary, frontier, quantized, environment, output_dir / "EXPERIMENT_4_FINDINGS.md")
    return environment


def refresh_phase_disabled_artifacts(dataset, output_dir: Path, beam_width: int = 64) -> None:
    """Re-evaluate endpoint controls with full-resolution prefix no-op fallback."""
    results = pd.read_csv(output_dir / "results.csv", low_memory=False)
    usage = pd.read_csv(output_dir / "code_usage.csv")
    replacements = []
    usage_replacements = []
    names = []
    for strategy in ("phase_shared", "phase_topology"):
        chain = PhaseChain.load(output_dir / "codebooks" / strategy)
        frame, use, _ = evaluate_chain(
            dataset, chain, base_phase=False, residual_phase=False,
            gains=True, beam_width=beam_width, refine=True,
        )
        names.append(frame.configuration.iloc[0])
        replacements.append(frame)
        usage_replacements.append(use)
    results = pd.concat([results[~results.configuration.isin(names)], *replacements], ignore_index=True)
    usage = pd.concat([usage[~usage.configuration.isin(names)], *usage_replacements], ignore_index=True)
    current_summary = summarize(results)
    current_frontier = pareto(current_summary)
    results.to_csv(output_dir / "results.csv", index=False)
    usage.to_csv(output_dir / "code_usage.csv", index=False)
    current_summary.to_csv(output_dir / "summary.csv", index=False)
    current_frontier.to_csv(output_dir / "pareto.csv", index=False)
    results.groupby(["configuration", "topology"], as_index=False).agg(
        rmse_median=("rmse", "median"), rmse_p95=("rmse", lambda x: x.quantile(.95))
    ).to_csv(output_dir / "topology_summary.csv", index=False)
    environment = json.loads((output_dir / "pytorch_environment.json").read_text(encoding="utf-8"))
    quantized = pd.read_csv(output_dir / "phase_quantization.csv")
    write_findings(
        current_summary, current_frontier, quantized, environment,
        output_dir / "EXPERIMENT_4_FINDINGS.md",
    )


def run_experiment4(
    catalog_path: Path,
    codebook_path: Path,
    output_dir: Path,
    *,
    resolution: int = 1024,
    beam_width: int = 64,
    quick: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_curve_dataset(catalog_path, resolution=resolution)
    if (
        not quick
        and (output_dir / "summary.csv").exists()
        and (output_dir / "results.csv").exists()
        and not (output_dir / "pytorch_environment.json").exists()
    ):
        print("Finalizing Experiment 4 from serialized geometry artifacts", flush=True)
        finalize_from_artifacts(output_dir)
        return pd.read_csv(output_dir / "results.csv"), pd.read_csv(output_dir / "summary.csv")
    stock_names, stock = load_stock_curves(codebook_path, resolution)
    width = 4 if quick else 16
    depth = 2 if quick else 4
    shared, topology, fitting_audit = train_phase_endpoints(
        dataset, stock, residual_width=width, depth=depth
    )
    if quick:
        chains = [shared, topology]
    else:
        chains = [
            shared, topology,
            compose_partitioned(shared, topology, [4, 4, 4, 4], name="phase_partition_s4"),
            compose_partitioned(shared, topology, [8, 8, 8, 8], name="phase_partition_s8"),
            compose_partitioned(shared, topology, [12, 12, 12, 12], name="phase_partition_s12"),
            compose_partitioned(shared, topology, [12, 9, 6, 3], name="phase_partition_taper_12_9_6_3"),
            compose_partitioned(shared, topology, [15, 10, 5, 0], name="phase_partition_taper_15_10_5_0"),
            *(compose_switch(shared, topology, switch) for switch in (1, 2, 3)),
            compose_additive(shared, topology, 8),
            compose_additive(shared, topology, 16),
        ]
    for chain in chains:
        chain.save(output_dir / "codebooks" / chain.name)
    np.savez_compressed(output_dir / "fitting_audit.npz", **fitting_audit)
    manifest = {
        "stock_names": stock_names,
        "train_instances": len(dataset.train_indices),
        "validation_instances": len(dataset.validation_indices),
        "train_authors": int(dataset.frame.iloc[dataset.train_indices].author_id.nunique()),
        "validation_authors": int(dataset.frame.iloc[dataset.validation_indices].author_id.nunique()),
        "phase_search_resolution": 128,
        "full_resolution": resolution,
        "beam_width": min(8, beam_width) if quick else beam_width,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    evaluations = []
    usages = []
    encodings: dict[str, PhaseEncoding] = {}
    chain_by_name = {chain.name: chain for chain in chains}
    for chain in chains:
        modes = [(True, True, True)]
        if chain.name in {"phase_shared", "phase_topology"}:
            modes = [(False, False, True), (True, False, True), (False, True, True), (True, True, True)]
        for base_phase, residual_phase, gains in modes:
            print(f"Evaluating {chain.name}: base_phase={base_phase}, residual_phase={residual_phase}", flush=True)
            result, usage, encoding = evaluate_chain(
                dataset, chain, base_phase=base_phase, residual_phase=residual_phase,
                gains=gains, beam_width=min(8, beam_width) if quick else beam_width,
                refine=not quick,
            )
            evaluations.append(result)
            usages.append(usage)
            encodings[result.configuration.iloc[0]] = encoding

    results = pd.concat(evaluations, ignore_index=True)
    usage = pd.concat(usages, ignore_index=True)
    summary = summarize(results)
    frontier = pareto(summary)
    full_phase = summary[
        summary.base_phase_enabled & summary.residual_phase_enabled & summary.gains_enabled
    ]
    under_budget = full_phase[full_phase.dense_dimensions < 120]
    visual_row = under_budget.loc[under_budget.rmse_p95.idxmin()] if len(under_budget) else full_phase.loc[full_phase.rmse_p95.idxmin()]
    visual_name = str(visual_row.configuration)
    visual_chain = chain_by_name[str(visual_row.strategy)]
    visual_results = results[results.configuration == visual_name].reset_index(drop=True)
    visual_encoding = encodings[visual_name]
    quantized = phase_quantization(dataset, visual_chain, visual_encoding)

    # Beam audit on a deterministic 10% sample. This records whether a wider rerun is needed.
    audit_count = max(1, len(dataset.validation_indices) // 10)
    subset = replace(
        dataset,
        validation_indices=dataset.validation_indices[:audit_count],
    )
    _, _, wide_encoding = evaluate_chain(
        subset, visual_chain, base_phase=True, residual_phase=True, gains=True,
        beam_width=32 if quick else 256, refine=False,
    )
    wide_reconstruction, _, _ = decode_phase_chain(
        visual_chain, wide_encoding, conditions_for(subset, visual_chain)
    )
    narrow = visual_results.iloc[:audit_count].rmse.to_numpy()
    wide = metric_arrays(subset.validation_curves, wide_reconstruction)["rmse"]
    improved_share = float(np.mean(wide < narrow - 1e-5))
    pd.DataFrame({"narrow_rmse": narrow, "wide_rmse": wide}).to_csv(output_dir / "beam_audit.csv", index=False)
    if improved_share > .001 and not quick:
        print(f"Beam audit improved {improved_share:.2%}; rerunning selected codec at width 128", flush=True)
        rerun, rerun_usage, rerun_encoding = evaluate_chain(
            dataset, visual_chain, base_phase=True, residual_phase=True, gains=True,
            beam_width=128, refine=True,
        )
        results = pd.concat([results[results.configuration != visual_name], rerun], ignore_index=True)
        usage = pd.concat([usage[usage.configuration != visual_name], rerun_usage], ignore_index=True)
        visual_results, visual_encoding = rerun.reset_index(drop=True), rerun_encoding
        summary, frontier = summarize(results), pareto(summarize(results))

    create_layer_visuals(dataset, visual_chain, visual_results, visual_encoding, output_dir)
    results.to_csv(output_dir / "results.csv", index=False)
    usage.to_csv(output_dir / "code_usage.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    frontier.to_csv(output_dir / "pareto.csv", index=False)
    quantized.to_csv(output_dir / "phase_quantization.csv", index=False)
    topology_summary = results.groupby(["configuration", "topology"], as_index=False).agg(
        rmse_median=("rmse", "median"), rmse_p95=("rmse", lambda x: x.quantile(.95))
    )
    topology_summary.to_csv(output_dir / "topology_summary.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for strategy, group in summary.groupby("strategy"):
        axes[0].scatter(group.dense_dimensions, group.rmse_median, label=strategy, alpha=.8)
        axes[1].scatter(group.dense_dimensions, group.rmse_p95, label=strategy, alpha=.8)
    axes[0].set_title("Median RMSE")
    axes[1].set_title("95th-percentile RMSE")
    for axis in axes:
        axis.set(xlabel="Dense output dimensions", ylabel="Curve RMSE")
        axis.grid(alpha=.2)
    axes[1].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(output_dir / "pareto.png", dpi=170)
    plt.close(figure)

    environment = verify_pytorch_environment(output_dir)
    write_findings(summary, frontier, quantized, environment, output_dir / "EXPERIMENT_4_FINDINGS.md")
    print(summary.to_string(index=False))
    return results, summary
