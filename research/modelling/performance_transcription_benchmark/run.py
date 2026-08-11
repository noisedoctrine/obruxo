"""CLI for the bounded comparative performance-transcription benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from obruxo_performance_benchmark.adapters import adapter_for
from obruxo_performance_benchmark.artifacts import (
    ArtifactError,
    ArtifactUnavailable,
    load_model_specs,
)
from obruxo_performance_benchmark.benchmark import run_benchmark
from obruxo_performance_benchmark.evaluate import evaluate_model
from obruxo_performance_benchmark.report import write_public_report


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--force", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OBRUXO comparative performance-transcription benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify-model", help="verify pinned source and checkpoint identity")
    _common(verify)
    evaluate = sub.add_parser("evaluate", help="evaluate one fixed #25 manifest")
    _common(evaluate)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--quantized", action="store_true")
    benchmark = sub.add_parser("benchmark", help="run one fixed #24 smoke workload")
    _common(benchmark)
    benchmark.add_argument("--smoke-manifest", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--quantized", action="store_true")
    report = sub.add_parser("report", help="write sanitized aggregate reports")
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--json", type=Path, required=True)
    report.add_argument("--markdown", type=Path, required=True)
    report.add_argument("--force", action="store_true")
    return parser


def _spec(args: argparse.Namespace):
    specs = load_model_specs(args.config)
    try:
        return specs[args.model_id]
    except KeyError as exc:
        raise ArtifactError(f"unknown model ID: {args.model_id}") from exc


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "report":
            write_public_report(args.input, args.json, args.markdown, force=args.force)
            print("wrote sanitized performance-transcription reports")
            return 0
        spec = _spec(args)
        if args.command == "verify-model":
            if not spec.is_available:
                print(f"{spec.model_id}: unavailable ({spec.unavailability_reason})")
                return 0
            adapter = adapter_for(spec, args.source_root, args.checkpoint)
            adapter.preflight()
            print(f"verified {spec.model_id}")
            return 0
        adapter = adapter_for(spec, args.source_root, args.checkpoint)
        if args.command == "evaluate":
            result = evaluate_model(spec, adapter, args.manifest, args.output, quantized=args.quantized, force=args.force)
            print(f"{result.model_id}: {result.status}")
            return 0 if result.status in {"ok", "unavailable"} else 3
        result = run_benchmark(spec, adapter, args.smoke_manifest, args.output, quantized=args.quantized, force=args.force)
        print(f"{result['model_id']}: {result['status']}")
        return 0 if result["status"] in {"ok", "unavailable"} else 3
    except (ArtifactUnavailable, ArtifactError, FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
