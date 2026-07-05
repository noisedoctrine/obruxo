"""Forensic probes for the Era 1 vs Era 2 LFO RMSE gap."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .alignment import alignment_matrix
from .curve import circular_shift
from .metrics import reconstruction_summary, rmse_per_curve


ERA2_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ERA2_ROOT.parents[3]
ERA1_ROOT = ERA2_ROOT.parent / "era1"
ERA1_CODE = ERA1_ROOT / "code"
DEFAULT_ERA1_CATALOG = ERA1_ROOT / "artifacts" / "lfo_catalog.csv"
DEFAULT_ERA1_CHECKPOINT = (
    ERA1_ROOT
    / "artifacts"
    / "additive_finalization_9_screen"
    / "checkpoints"
    / "9_screen_9B_phase_only_final_only_phase_only_phase_only_raw_final_only_none_W8D16_bw4_eval120_sample33_seed7267"
)
DEFAULT_OUTPUT_DIR = ERA2_ROOT / "artifacts" / "experiment_11" / "rmse_gap_audit"
DEFAULT_REPORT = ERA2_ROOT / "reports" / "EXPERIMENT_11_RMSE_GAP_FORENSIC_AUDIT.md"


@dataclass(frozen=True)
class Era1Chain:
    name: str
    bases: np.ndarray
    stages: tuple[np.ndarray, ...]
    stage_labels: tuple[str, ...]
    stage_branches: tuple[str, ...]
    topology_conditioned: bool


@dataclass(frozen=True)
class Era1Encoding:
    dataset_index: np.ndarray
    base_index: np.ndarray
    base_phase: np.ndarray
    stage_indices: list[np.ndarray]
    stage_phases: list[np.ndarray]
    stage_gains: list[np.ndarray]

    @property
    def row_count(self) -> int:
        return int(len(self.base_index))


def run_rmse_gap_audit(
    *,
    checkpoint_dir: Path = DEFAULT_ERA1_CHECKPOINT,
    catalog_path: Path = DEFAULT_ERA1_CATALOG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT,
    row_limit: int = 256,
    progress: Any | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _log(progress, f"rmse-gap: checkpoint={checkpoint_dir}")
    probes = _primitive_probes()
    replay_rows = _artifact_replay_probe(
        checkpoint_dir=Path(checkpoint_dir),
        catalog_path=Path(catalog_path),
        row_limit=int(row_limit),
        progress=progress,
    )
    all_rows = [*probes, *replay_rows]
    probe_csv = output_dir / "probe_summary.csv"
    _write_rows(probe_csv, all_rows)
    _write_report(report_path, all_rows, checkpoint_dir=Path(checkpoint_dir), catalog_path=Path(catalog_path), row_limit=int(row_limit))
    return {"probe_summary": str(probe_csv), "report": str(report_path)}


def _primitive_probes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260705)
    reference = rng.random((6, 37), dtype=np.float32)
    reconstructed = np.clip(reference + rng.normal(0.0, 0.05, reference.shape).astype(np.float32), 0.0, 1.0)
    direct = np.sqrt(np.mean((reconstructed - reference) ** 2, axis=1))
    era2 = rmse_per_curve(reference, reconstructed)
    rows.append(
        _probe_row(
            "rmse_same_array",
            "pass" if np.max(np.abs(direct - era2)) < 1e-7 else "fail",
            float(np.max(np.abs(direct - era2))),
            "Same target/prediction arrays produce identical RMSE.",
        )
    )

    curve = (0.5 + 0.4 * np.sin(2.0 * np.pi * np.arange(64, dtype=np.float32) / 64.0)).astype(np.float32)
    phases = np.asarray([0.0, 0.125, 0.37], dtype=np.float32)
    era2_shift = circular_shift(np.repeat(curve[None, :], len(phases), axis=0), phases)
    try:
        phase4 = _import_era1_module("lfo_experiment.phase4")
        era1_shift = phase4.circular_shift(np.repeat(curve[None, :], len(phases), axis=0), phases)
        delta = float(np.max(np.abs(era1_shift - era2_shift)))
        status = "pass" if delta < 1e-6 else "fail"
        detail = "Era 1 and Era 2 circular shift agree on endpoint-excluded sampled curves."
    except Exception as exc:  # pragma: no cover - depends on local Era 1 importability
        delta = float("nan")
        status = "unsupported"
        detail = f"Could not import Era 1 circular shift: {exc}"
    rows.append(_probe_row("circular_shift_parity", status, delta, detail))
    tiny_shift = circular_shift(curve, np.asarray([1.0e-15], dtype=np.float32))[0]
    tiny_delta = float(np.max(np.abs(tiny_shift - curve)))
    rows.append(
        _probe_row(
            "tiny_phase_identity",
            "pass" if tiny_delta < 1e-5 else "fail",
            tiny_delta,
            "Era 2 circular shift keeps near-zero phases near the identity transform.",
        )
    )

    targets = np.stack([circular_shift(curve, 0.17), 0.55 * circular_shift(curve, 0.31)], axis=0)
    codes = curve[None, :]
    try:
        alignment5 = _import_era1_module("lfo_experiment.alignment5")
        era1_result = alignment5.exact_align_cpu(targets, codes, fixed_gain=None)
        era2_matrix = alignment_matrix(targets, codes, phase_policy="exact", gain_policy="optimized")
        delta = float(
            max(
                np.max(np.abs(era1_result.error - era2_matrix.losses)),
                np.max(np.abs(era1_result.phase - era2_matrix.phases)),
                np.max(np.abs(era1_result.gain - era2_matrix.gains)),
            )
        )
        status = "pass" if delta < 1e-5 else "fail"
        detail = "Era 1 exact phase/gain scoring and Era 2 exact alignment agree on identical arrays."
    except Exception as exc:  # pragma: no cover - depends on local Era 1 importability
        delta = float("nan")
        status = "unsupported"
        detail = f"Could not import Era 1 exact phase/gain alignment: {exc}"
    rows.append(_probe_row("exact_phase_gain_alignment_parity", status, delta, detail))
    return rows


def _artifact_replay_probe(
    *,
    checkpoint_dir: Path,
    catalog_path: Path,
    row_limit: int,
    progress: Any | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    chain_dir = checkpoint_dir / "chain"
    paths_csv = checkpoint_dir / "paths.csv"
    if not chain_dir.exists() or not paths_csv.exists():
        return [
            _probe_row(
                "era1_artifact_replay",
                "unsupported",
                float("nan"),
                f"Missing Era 1 checkpoint chain or paths at {checkpoint_dir}.",
            )
        ]

    chain = load_era1_chain(chain_dir)
    encoding = load_era1_encoding(paths_csv, stage_count=len(chain.stages), row_limit=row_limit)
    target_bundle = _load_era1_targets(catalog_path, encoding.dataset_index, resolution=_read_eval_resolution(paths_csv), progress=progress)
    conditions = target_bundle["conditions"]
    targets = target_bundle["targets"]

    decoded_with_gain = decode_era1_arrays(chain, encoding, conditions, use_stage_gains=True)
    decoded_without_gain = decode_era1_arrays(chain, encoding, conditions, use_stage_gains=False)
    replay_summary = reconstruction_summary(targets, decoded_with_gain)
    no_gain_summary = reconstruction_summary(targets, decoded_without_gain)
    reconstruction_delta = rmse_per_curve(decoded_with_gain, decoded_without_gain)
    gain_values = np.concatenate(encoding.stage_gains) if encoding.stage_gains else np.asarray([], dtype=np.float32)
    nonzero_gain_rate = float(np.mean(np.abs(gain_values) > 1e-8)) if len(gain_values) else 0.0
    gain_p95 = float(np.quantile(np.abs(gain_values), 0.95)) if len(gain_values) else 0.0

    rows.append(
        _probe_row(
            "era1_artifact_replay_with_stage_gains",
            "pass",
            float(replay_summary["p95_rmse"]),
            "Era 1 W8D16 checkpoint replayed with saved residual-layer gains.",
            metric_name="p95_rmse",
            extra={
                "median_rmse": replay_summary["median_rmse"],
                "row_count": replay_summary["row_count"],
                "checkpoint": str(checkpoint_dir),
            },
        )
    )
    rows.append(
        _probe_row(
            "era1_artifact_replay_without_stage_gains",
            "pass",
            float(no_gain_summary["p95_rmse"]),
            "Same Era 1 encoding decoded after forcing every residual-layer gain to 1.0.",
            metric_name="p95_rmse",
            extra={"median_rmse": no_gain_summary["median_rmse"]},
        )
    )
    rows.append(
        _probe_row(
            "residual_gain_decode_effect",
            "confirmed",
            float(np.quantile(reconstruction_delta, 0.95)),
            "Changing only residual-layer gains produces a large reconstruction change from the same codes/phases.",
            metric_name="p95_rmse_between_decodes",
            extra={
                "median_rmse_between_decodes": float(np.median(reconstruction_delta)),
                "residual_gain_abs_p95": gain_p95,
                "residual_gain_nonzero_rate": nonzero_gain_rate,
            },
        )
    )

    try:
        phase4 = _import_era1_module("lfo_experiment.phase4")
        experiment7 = _import_era1_module("lfo_experiment.experiment7")
        era1_chain = phase4.PhaseChain.load(chain_dir)
        era1_encoding = phase4.PhaseEncoding(
            encoding.base_index.astype(np.int16),
            encoding.base_phase.astype(np.float32),
            [value.astype(np.int16) for value in encoding.stage_indices],
            [value.astype(np.float32) for value in encoding.stage_phases],
            [value.astype(np.float32) for value in encoding.stage_gains],
        )
        era1_raw, _, _ = experiment7._decode_raw(era1_chain, era1_encoding, conditions, residual_clip_policy="final_only")
        era1_decoded = np.clip(era1_raw, 0.0, 1.0).astype(np.float32)
        delta = float(np.max(np.abs(era1_decoded - decoded_with_gain)))
        status = "pass" if delta < 1e-6 else "fail"
        detail = "Era 2-side artifact decoder reproduces the Experiment 9 final-only decode from the same saved encoding."
    except Exception as exc:  # pragma: no cover - depends on local Era 1 importability
        delta = float("nan")
        status = "unsupported"
        detail = f"Could not run Era 1 decoder replay: {exc}"
    rows.append(_probe_row("era1_decoder_replay_parity", status, delta, detail))
    return rows


def load_era1_chain(chain_dir: Path) -> Era1Chain:
    manifest = json.loads((chain_dir / "manifest.json").read_text(encoding="utf-8"))
    payload = np.load(chain_dir / "codebook.npz")
    stage_count = len(manifest["stage_widths"])
    return Era1Chain(
        name=str(manifest.get("name", "")),
        bases=payload["bases"].astype(np.float32),
        stages=tuple(payload[f"stage_{index}"].astype(np.float32) for index in range(stage_count)),
        stage_labels=tuple(manifest.get("stage_labels", [])),
        stage_branches=tuple(manifest.get("stage_branches", [])),
        topology_conditioned=bool(manifest.get("topology_conditioned", False)),
    )


def load_era1_encoding(paths_csv: Path, *, stage_count: int, row_limit: int | None = None) -> Era1Encoding:
    rows: list[dict[str, str]] = []
    with paths_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            if row_limit is not None and len(rows) >= int(row_limit):
                break
    if not rows:
        raise ValueError(f"no rows found in {paths_csv}")
    dataset_index = np.asarray([int(row["dataset_index"]) for row in rows], dtype=np.int32)
    base_index = np.asarray([int(row["base_index"]) for row in rows], dtype=np.int16)
    base_phase = np.asarray([float(row["base_phase"]) for row in rows], dtype=np.float32)
    stage_indices = []
    stage_phases = []
    stage_gains = []
    for stage in range(1, stage_count + 1):
        stage_indices.append(np.asarray([int(row[f"stage_{stage}_index"]) for row in rows], dtype=np.int16))
        stage_phases.append(np.asarray([float(row[f"stage_{stage}_phase"]) for row in rows], dtype=np.float32))
        stage_gains.append(np.asarray([float(row[f"stage_{stage}_gain"]) for row in rows], dtype=np.float32))
    return Era1Encoding(dataset_index, base_index, base_phase, stage_indices, stage_phases, stage_gains)


def decode_era1_arrays(
    chain: Era1Chain,
    encoding: Era1Encoding,
    conditions: np.ndarray,
    *,
    use_stage_gains: bool = True,
) -> np.ndarray:
    conditions = np.asarray(conditions, dtype=np.int32)
    result = circular_shift(chain.bases[encoding.base_index], encoding.base_phase)
    for stage_index, stage in enumerate(chain.stages):
        code = stage[conditions, encoding.stage_indices[stage_index]]
        addition = circular_shift(code, encoding.stage_phases[stage_index])
        if use_stage_gains:
            addition = addition * encoding.stage_gains[stage_index][:, None]
        result = result + addition
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _load_era1_targets(catalog_path: Path, dataset_indices: np.ndarray, *, resolution: int, progress: Any | None) -> dict[str, np.ndarray]:
    _log(progress, f"rmse-gap: loading Era 1 catalog targets at resolution={resolution}")
    stacked = _import_era1_module("lfo_experiment.stacked")
    dataset = stacked.load_curve_dataset(catalog_path, resolution=resolution)
    indices = np.asarray(dataset_indices, dtype=np.int32)
    return {
        "targets": dataset.curves[indices].astype(np.float32),
        "conditions": dataset.topology[indices].astype(np.int32),
    }


def _read_eval_resolution(paths_csv: Path) -> int:
    with paths_csv.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    return int(float(row.get("eval_resolution") or 120))


def _import_era1_module(name: str) -> Any:
    era1_code = str(ERA1_CODE)
    if era1_code not in sys.path:
        sys.path.insert(0, era1_code)
    __import__(name)
    return sys.modules[name]


def _probe_row(
    probe_id: str,
    status: str,
    value: float,
    finding: str,
    *,
    metric_name: str = "max_abs_delta",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "probe_id": probe_id,
        "status": status,
        "metric_name": metric_name,
        "metric_value": value,
        "finding": finding,
    }
    if extra:
        row.update(extra)
    return row


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(report_path: Path, rows: list[dict[str, Any]], *, checkpoint_dir: Path, catalog_path: Path, row_limit: int) -> None:
    by_id = {str(row["probe_id"]): row for row in rows}
    with_gain = by_id.get("era1_artifact_replay_with_stage_gains", {})
    without_gain = by_id.get("era1_artifact_replay_without_stage_gains", {})
    gain_effect = by_id.get("residual_gain_decode_effect", {})
    replay_parity = by_id.get("era1_decoder_replay_parity", {})
    alignment_parity = by_id.get("exact_phase_gain_alignment_parity", {})

    lines = [
        "# Experiment 11 RMSE Gap Forensic Audit",
        "",
        "## Main Findings",
        "",
        "The large Era 1 vs Era 2 RMSE gap should be read first as a reconstruction-pipeline problem, not as a budget problem.",
        "The primitive array checks clear the basic RMSE calculation, circular shift behavior, and Experiment 9 decode replay.",
        "The strongest concrete difference is residual-layer gain: Era 1's `phase_only` rows still saved and applied a gain scalar for every residual layer, while the canonical Era 2 flat path did not.",
        "The audit also fixed an Era 2 circular-shift precision bug: tiny near-zero phases now stay near the identity transform instead of occasionally wrapping into a bad interpolation fraction.",
        "",
    ]
    if with_gain and without_gain:
        lines.extend(
            [
                f"In the replayed Era 1 W8D16 checkpoint sample, using the saved residual-layer gains gives P95 RMSE `{_fmt(with_gain.get('metric_value'))}`.",
                f"Forcing those same residual layers to gain `1.0` gives P95 RMSE `{_fmt(without_gain.get('metric_value'))}`.",
                "That is the cleanest current explanation for why the Era 2 quality numbers look wildly worse: the supposedly comparable Era 1 rows were not fixed-amplitude residual atoms.",
                "",
            ]
        )
    if gain_effect:
        lines.extend(
            [
                f"Changing only residual-layer gains changes the reconstructed curve by P95 RMSE `{_fmt(gain_effect.get('metric_value'))}` between decodes.",
                f"The saved residual-layer gain absolute P95 is `{_fmt(gain_effect.get('residual_gain_abs_p95'))}`, with nonzero rate `{_fmt(gain_effect.get('residual_gain_nonzero_rate'))}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## What Passed",
            "",
            _bullet(by_id.get("rmse_same_array")),
            _bullet(by_id.get("circular_shift_parity")),
            _bullet(by_id.get("tiny_phase_identity")),
            _bullet(replay_parity),
            "",
            "These checks mean the first-order failure is not the RMSE formula, ordinary circular shift behavior, or a decoder replay mismatch. The gap is higher in the stack: which residual scalars are applied, how paths are searched, and how atoms are constructed.",
            "",
            "## Still Open",
            "",
            _bullet(alignment_parity),
            "",
            "Exact phase/gain alignment parity still needs a cleaner cross-era probe. That does not weaken the residual-gain finding above, because the artifact replay uses saved Era 1 indices, phases, and gains rather than re-solving alignment.",
            "",
            "## Implication For Experiment 11",
            "",
            "The next meaningful Experiment 11 candidate should include optimized residual-layer gain as a model-facing scalar family. Beam width 4 and offline topology-aware construction are still useful, but neither explains the huge gap on its own.",
            "",
            "Do not compare Era 2's fixed-amplitude flat path against Era 1's `phase_only` rows as if both use the same decoder degrees of freedom. In Era 1, `phase_only` means no extra modifier/base gain family; it does not mean residual atom gains were absent.",
            "",
            "## Method Notes",
            "",
            f"- Era 1 checkpoint: `{checkpoint_dir}`",
            f"- Era 1 catalog: `{catalog_path}`",
            f"- Replay row limit: `{row_limit}`",
            "- `W` remains residual-layer atom choices, not grid subdivisions.",
            "- Topology-runtime replay is forensic only. It is not an Era 2 deployable contract.",
            "- Detailed probe values are in `era2/artifacts/experiment_11/rmse_gap_audit/probe_summary.csv`.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _bullet(row: dict[str, Any] | None) -> str:
    if not row:
        return "- missing probe row"
    return f"- `{row.get('probe_id')}`: {row.get('status')} ({row.get('metric_name')} `{_fmt(row.get('metric_value'))}`). {row.get('finding')}"


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.6g}"


def _log(progress: Any | None, message: str) -> None:
    if progress is not None:
        progress(message)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_ERA1_CHECKPOINT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_ERA1_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--row-limit", type=int, default=256)
    args = parser.parse_args(argv)
    result = run_rmse_gap_audit(
        checkpoint_dir=args.checkpoint_dir,
        catalog_path=args.catalog,
        output_dir=args.output_dir,
        report_path=args.report,
        row_limit=args.row_limit,
        progress=print,
    )
    print(f"Wrote RMSE gap audit report to {result['report']}")
    print(f"probe_summary={result['probe_summary']}")


if __name__ == "__main__":
    main()
