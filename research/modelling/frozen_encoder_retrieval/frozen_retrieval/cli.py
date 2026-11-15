from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PACKAGE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_DIR.parents[2]
for candidate in (REPO_ROOT / "research/data_generation", REPO_ROOT / "research/training",
                  REPO_ROOT / "research/modelling/basic_pitch", PACKAGE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def _defaults() -> dict:
    root = REPO_ROOT / "research/modelling/frozen_encoder_retrieval"
    return {"root": root, "inventory": root / "reports/vital_control_inventory.jsonl",
            "report": root / "reports/FROZEN_ENCODER_RETRIEVAL_RESULTS.md",
            "renderer": REPO_ROOT / "research/data_generation/configs/renderer_full_corpus.yaml"}


def _parser() -> argparse.ArgumentParser:
    defaults = _defaults()
    parser = argparse.ArgumentParser(description="Frozen pretrained audio-encoder retrieval benchmark (#152)")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="check exact local model/dependency availability without downloading weights")
    doctor.add_argument("--checkpoint-root", type=Path)
    doctor.add_argument("--lock", type=Path)
    inventory = commands.add_parser("inventory", help="build the one per-control frozen experiment inventory")
    inventory.add_argument("--source", type=Path, default=REPO_ROOT / "research/modelling/control_scope/stage_a_controls.jsonl")
    inventory.add_argument("--output", type=Path, default=defaults["inventory"])
    prepare = commands.add_parser("prepare", help="audit allowed source inputs and freeze a private render/evaluation plan")
    prepare.add_argument("--snapshot", type=Path, required=True, help="complete audited frozen corpus snapshot")
    prepare.add_argument("--data-root", type=Path, required=True, help="original corpus root paired with snapshot relative paths")
    prepare.add_argument("--renderer-config", type=Path, default=defaults["renderer"])
    prepare.add_argument("--output", type=Path, required=True, help="ignored local output outside the raw data root")
    prepare.add_argument("--split", choices=("train", "validation"), default="validation")
    prepare.add_argument("--performance-source-split", choices=("train", "validation"),
                         help="use compatible MIDI paired to this permitted source split while keeping the preset cohort unchanged")
    prepare.add_argument("--max-performances", type=int,
                         help="select a deterministic diversity-balanced MIDI subset after exact performance deduplication")
    prepare.add_argument("--exclude-performance-manifest", type=Path,
                         help="exclude normalized MIDI IDs already present in another private performance manifest")
    prepare.add_argument("--gallery-midi-pool-size", type=int, default=4)
    prepare.add_argument("--max-render-requests", type=int, default=50_000,
                         help="fail before episode/plan creation if the full matrix exceeds this budget; never truncates")
    render = commands.add_parser("render", help="pre-render a frozen manifest with one serial Vital consumer")
    render.add_argument("--dataset", type=Path, required=True)
    render.add_argument("--renderer-config", type=Path, default=defaults["renderer"])
    render.add_argument("--plugin-path", type=Path)
    render.add_argument("--batch-size", type=int, default=32)
    render.add_argument("--queue-depth", type=int, default=2)
    embed = commands.add_parser("embed", help="extract or reuse frozen feature caches without fetching weights")
    embed.add_argument("--dataset", type=Path, required=True)
    embed.add_argument("--model", required=True)
    embed.add_argument("--checkpoint-root", type=Path, required=True)
    embed.add_argument("--lock", type=Path, required=True)
    embed.add_argument("--device", choices=("cpu", "cuda", "xpu"), default="cpu")
    evaluate = commands.add_parser("evaluate", help="evaluate identical cached retrieval episodes and baselines")
    evaluate.add_argument("--dataset", type=Path, required=True)
    report = commands.add_parser("report", help="regenerate the sanitized reader-facing comparison report")
    report.add_argument("--dataset", type=Path)
    report.add_argument("--comparison-dataset", type=Path,
                        help="optional earlier frozen cohort to compare with the selected result dataset")
    report.add_argument("--checkpoint-root", type=Path)
    report.add_argument("--lock", type=Path)
    report.add_argument("--controlled-summary", type=Path,
                        help="private JSON produced by run_controlled_perturbation.py; adds a controlled-results section and inventory")
    report.add_argument("--issue155-results", type=Path,
                        help="private issue #155 assignment and audio-quality summary produced by run_issue155_validation.py")
    report.add_argument("--candidate-shortlist", type=Path,
                        help="private automated controlled-preset shortlist summary; no definitive controlled run is started")
    report.add_argument("--output", type=Path, default=defaults["report"])
    report.add_argument("--execution-note", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            from .encoders import available_candidates, pipeline_dependency_status
            from .core import MODEL_REGISTRY
            result = {"models": available_candidates(args.checkpoint_root, args.lock),
                      "pipeline_dependencies": pipeline_dependency_status(),
                      "vital_plugin_path_configured": bool(os.environ.get("OBRUXO_VITAL_PLUGIN")),
                      "weight_downloads_performed": False,
                      "checkpoint_lock_required_fields": ["path", "sha256", "revision", "loader_revision", "runtime_versions", "license_status"]}
        elif args.command == "inventory":
            from .reporting import build_control_inventory
            result = build_control_inventory(args.source, args.output)
        elif args.command == "prepare":
            from .dataset import prepare_dataset
            result = prepare_dataset(args.snapshot, args.data_root, args.renderer_config, args.output,
                                     split=args.split, gallery_cap=args.gallery_midi_pool_size,
                                     max_render_requests=args.max_render_requests,
                                     performance_source_split=args.performance_source_split,
                                     max_performances=args.max_performances,
                                     exclude_performance_manifest=args.exclude_performance_manifest)
        elif args.command == "render":
            from .rendering import render_dataset
            result = render_dataset(args.dataset, args.renderer_config, plugin_path=args.plugin_path,
                                    batch_size=args.batch_size, queue_depth=args.queue_depth)
        elif args.command == "embed":
            from .feature_stage import extract_dataset
            result = extract_dataset(args.dataset, args.model, args.checkpoint_root, args.lock, device=args.device)
        elif args.command == "evaluate":
            from .evaluation import evaluate_dataset
            result = evaluate_dataset(args.dataset)
        elif args.command == "report":
            from .reporting import build_control_inventory, generate_report
            defaults = _defaults()
            if not defaults["inventory"].is_file():
                build_control_inventory(REPO_ROOT / "research/modelling/control_scope/stage_a_controls.jsonl", defaults["inventory"])
            result = generate_report(args.output, defaults["inventory"], dataset_dir=args.dataset,
                                     checkpoint_root=args.checkpoint_root, lock_path=args.lock,
                                     execution_note=args.execution_note, controlled_summary_path=args.controlled_summary,
                                     comparison_dataset_dir=args.comparison_dataset,
                                     issue155_results_path=args.issue155_results,
                                     candidate_shortlist_path=args.candidate_shortlist)
        else:
            raise ValueError(f"unsupported command {args.command}")
    except Exception as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}, ensure_ascii=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
