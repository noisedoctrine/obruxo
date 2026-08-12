"""Thin command-line wrapper for Basic Pitch import and parity."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


def _preload_conda_openmp_runtime() -> None:
    """Load the py312 Conda OpenMP runtime before PyTorch and SciPy coexist."""
    if sys.platform != "win32":
        return
    import ctypes

    runtime = Path(sys.prefix) / "Library" / "bin" / "libiomp5md.dll"
    if runtime.is_file():
        ctypes.CDLL(str(runtime))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OBRUXO Basic Pitch conversion and parity tools"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    importer = subparsers.add_parser(
        "import-onnx", help="verify and import the pinned public ONNX checkpoint"
    )
    importer.add_argument("--onnx", type=Path, required=True)
    importer.add_argument("--checkpoint", type=Path, required=True)
    importer.add_argument("--metadata", type=Path, required=True)
    importer.add_argument("--force", action="store_true")

    parity = subparsers.add_parser(
        "parity", help="compare the native module with ONNX Runtime CPU"
    )
    parity.add_argument("--onnx", type=Path, required=True)
    parity.add_argument("--checkpoint", type=Path, required=True)
    parity.add_argument("--json", type=Path, required=True)
    parity.add_argument("--markdown", type=Path, required=True)
    parity.add_argument("--audio", type=Path, action="append", default=[])
    parity.add_argument("--force", action="store_true")

    benchmark = subparsers.add_parser(
        "benchmark", help="run the fixed CPU/XPU/OpenVINO benchmark"
    )
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

    manifest = subparsers.add_parser(
        "build-eval-manifest", help="resolve the local PresetShare evaluation pairs"
    )
    manifest.add_argument("--corpus-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--audit", type=Path, required=True)
    manifest.add_argument(
        "--allow-derived-render",
        action="store_true",
        help="opt in to derived WAVs for unambiguous Vital patch and MIDI pairs",
    )
    manifest.add_argument("--force", action="store_true")

    evaluation = subparsers.add_parser(
        "evaluate-corpus", help="evaluate the fixed stock Basic Pitch corpus manifest"
    )
    evaluation.add_argument("--manifest", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--force", action="store_true")

    parity_diagnostic = subparsers.add_parser(
        "parity-diagnostic",
        help="run the fixed synthetic parity gate for every inference route",
    )
    parity_diagnostic.add_argument("--checkpoint", type=Path, required=True)
    parity_diagnostic.add_argument(
        "--json",
        type=Path,
        required=True,
        help="existing backend benchmark JSON to augment",
    )
    parity_diagnostic.add_argument(
        "--markdown",
        type=Path,
        required=True,
        help="existing backend benchmark Markdown to augment",
    )
    parity_diagnostic.add_argument("--xpu-index", type=int, default=0)
    parity_diagnostic.add_argument("--openvino-gpu-device", default="GPU")
    parity_diagnostic.add_argument("--repetitions", type=int, default=3)
    parity_diagnostic.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "import-onnx":
        from obruxo_basic_pitch.weights import write_imported_checkpoint

        metadata = write_imported_checkpoint(
            args.onnx, args.checkpoint, args.metadata, force=args.force
        )
        print(f"imported {metadata.model_id} from the pinned public artifact")
        return 0

    if args.command == "benchmark":
        from obruxo_basic_pitch.benchmark import run_benchmark_cli

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

    if args.command == "parity-diagnostic":
        from obruxo_basic_pitch.benchmark import run_parity_diagnostic_cli

        return run_parity_diagnostic_cli(
            args.checkpoint,
            args.json,
            args.markdown,
            xpu_index=args.xpu_index,
            openvino_gpu_device=args.openvino_gpu_device,
            process_repetitions=args.repetitions,
            force=args.force,
        )

    if args.command == "build-eval-manifest":
        from obruxo_basic_pitch.evaluation.corpus import (
            CorpusInputError,
            build_evaluation_manifest,
        )

        try:
            summary = build_evaluation_manifest(
                args.corpus_root,
                output=args.output,
                audit=args.audit,
                allow_derived_render=args.allow_derived_render,
                force=args.force,
            )
        except (CorpusInputError, FileExistsError, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(
            f"paired {summary['eligible_count']} of {summary['candidate_count']} candidates"
        )
        return 0

    if args.command == "evaluate-corpus":
        _preload_conda_openmp_runtime()
        from obruxo_basic_pitch.evaluation.report import write_sanitized_reports
        from obruxo_basic_pitch.evaluation.runner import (
            BackendUnavailable,
            EvaluationInputError,
            evaluate_corpus,
        )

        try:
            result = evaluate_corpus(args.manifest, args.output, force=args.force)
            manifest_path = args.manifest.resolve(strict=True)
            output_dir = args.output.resolve(strict=False)
            write_sanitized_reports(
                manifest_path.with_name("pairing_audit.json"),
                output_dir / "run.json",
                output_dir / "aggregates.json",
                Path(__file__).resolve().parent
                / "reports"
                / "presetshare_baseline.json",
                Path(__file__).resolve().parent / "reports" / "presetshare_baseline.md",
                force=args.force,
            )
        except (
            BackendUnavailable,
            EvaluationInputError,
            FileExistsError,
            OSError,
            ValueError,
        ) as exc:
            print(str(exc), file=sys.stderr)
            return 3 if isinstance(exc, BackendUnavailable) else 2
        print(
            f"evaluated {result['successful_pair_count']} of {result['pair_count']} pairs"
        )
        return 0 if result["status"] == "ok" else 3

    from obruxo_basic_pitch.inference import prepare_wav
    from obruxo_basic_pitch.parity import (
        assert_parity,
        compare_windows_and_audio,
        synthetic_windows,
        write_reports,
    )

    public = synthetic_windows()
    local = [prepare_wav(path) for path in args.audio]
    summary = compare_windows_and_audio(
        args.onnx,
        args.checkpoint,
        public,
        local,
    )
    summary = replace(
        summary,
        synthetic_windows=public.shape[0],
        private_local_windows=sum(clip.windows.shape[0] for clip in local),
    )
    assert_parity(summary)
    write_reports(
        summary,
        args.json,
        args.markdown,
        private_local_clips=len(args.audio),
        force=args.force,
    )
    print(
        f"parity passed for {summary.synthetic_windows + summary.private_local_windows} windows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
