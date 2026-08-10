"""Thin command-line wrapper for Basic Pitch import and parity."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from obruxo_basic_pitch.benchmark import run_benchmark_cli
from obruxo_basic_pitch.evaluation.corpus import build_evaluation_manifest
from obruxo_basic_pitch.evaluation.report import write_sanitized_reports
from obruxo_basic_pitch.evaluation.runner import (
    BackendUnavailable,
    EvaluationInputError,
    evaluate_corpus,
)
from obruxo_basic_pitch.parity import (
    assert_parity,
    audio_to_windows,
    compare_windows,
    synthetic_windows,
    write_reports,
)
from obruxo_basic_pitch.weights import write_imported_checkpoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OBRUXO Basic Pitch conversion and parity tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    importer = subparsers.add_parser("import-onnx", help="verify and import the pinned public ONNX checkpoint")
    importer.add_argument("--onnx", type=Path, required=True)
    importer.add_argument("--checkpoint", type=Path, required=True)
    importer.add_argument("--metadata", type=Path, required=True)
    importer.add_argument("--force", action="store_true")

    parity = subparsers.add_parser("parity", help="compare the native module with ONNX Runtime CPU")
    parity.add_argument("--onnx", type=Path, required=True)
    parity.add_argument("--checkpoint", type=Path, required=True)
    parity.add_argument("--json", type=Path, required=True)
    parity.add_argument("--markdown", type=Path, required=True)
    parity.add_argument("--audio", type=Path, action="append", default=[])
    parity.add_argument("--force", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="run the fixed CPU/XPU/OpenVINO benchmark")
    benchmark.add_argument("--config", type=Path, required=True)
    benchmark.add_argument("--manifest", type=Path, required=True)
    benchmark.add_argument("--checkpoint", type=Path, required=True)
    benchmark.add_argument("--json", type=Path, required=True)
    benchmark.add_argument("--markdown", type=Path, required=True)
    benchmark.add_argument("--xpu-index", type=int, default=0)
    benchmark.add_argument("--openvino-gpu-device", default="GPU")
    benchmark.add_argument(
        "--allow-derived-render",
        action="store_true",
        help="opt in to missing-WAV renders from an unambiguous Vital patch and MIDI pair",
    )
    benchmark.add_argument("--force", action="store_true")

    manifest = subparsers.add_parser("build-eval-manifest", help="resolve the local PresetShare evaluation pairs")
    manifest.add_argument("--corpus-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--audit", type=Path, required=True)
    manifest.add_argument(
        "--allow-derived-render",
        action="store_true",
        help="opt in to derived WAVs for unambiguous Vital patch and MIDI pairs",
    )
    manifest.add_argument("--force", action="store_true")

    evaluation = subparsers.add_parser("evaluate-corpus", help="evaluate the fixed stock Basic Pitch corpus manifest")
    evaluation.add_argument("--manifest", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "import-onnx":
        metadata = write_imported_checkpoint(args.onnx, args.checkpoint, args.metadata, force=args.force)
        print(f"imported {metadata.model_id} from the pinned public artifact")
        return 0

    if args.command == "benchmark":
        return run_benchmark_cli(
            args.config,
            args.manifest,
            args.checkpoint,
            args.json,
            args.markdown,
            xpu_index=args.xpu_index,
            openvino_gpu_device=args.openvino_gpu_device,
            allow_derived_render=args.allow_derived_render,
            force=args.force,
        )

    if args.command == "build-eval-manifest":
        try:
            summary = build_evaluation_manifest(
                args.corpus_root,
                output=args.output,
                audit=args.audit,
                allow_derived_render=args.allow_derived_render,
                force=args.force,
            )
        except (EvaluationInputError, FileExistsError, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"paired {summary['eligible_count']} of {summary['candidate_count']} candidates")
        return 0

    if args.command == "evaluate-corpus":
        try:
            result = evaluate_corpus(args.manifest, args.output, force=args.force)
            manifest_path = args.manifest.resolve(strict=True)
            output_dir = args.output.resolve(strict=False)
            write_sanitized_reports(
                manifest_path.with_name("pairing_audit.json"),
                output_dir / "run.json",
                output_dir / "aggregates.json",
                Path(__file__).resolve().parent / "reports" / "presetshare_baseline.json",
                Path(__file__).resolve().parent / "reports" / "presetshare_baseline.md",
                force=args.force,
            )
        except (BackendUnavailable, EvaluationInputError, FileExistsError, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 3 if isinstance(exc, BackendUnavailable) else 2
        print(f"evaluated {result['successful_pair_count']} of {result['pair_count']} pairs")
        return 0 if result["status"] == "ok" else 3

    public = synthetic_windows()
    local = [audio_to_windows(path) for path in args.audio]
    windows = np.concatenate((public, *local), axis=0) if local else public
    summary = compare_windows(args.onnx, args.checkpoint, windows)
    summary = replace(summary, synthetic_windows=public.shape[0], private_local_windows=windows.shape[0] - public.shape[0])
    assert_parity(summary)
    write_reports(summary, args.json, args.markdown, private_local_clips=len(args.audio), force=args.force)
    print(f"parity passed for {windows.shape[0]} windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
