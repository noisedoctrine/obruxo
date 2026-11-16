from __future__ import annotations

import argparse
import json
from pathlib import Path

from frozen_retrieval.sensitivity import (analyze_cache, generate_figures, verify_embedding_cache)
from frozen_retrieval.sensitivity_reporting import update_research_report


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "FROZEN_ENCODER_RETRIEVAL_RESULTS.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze cached preset-by-MIDI embeddings for issue #157")
    parser.add_argument("--dataset", type=Path, default=ROOT / "outputs" / "issue152-validation-train-midi-v3-50")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "local_checkpoints")
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--device", choices=("cpu", "xpu", "cuda"), default="cpu")
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--bootstrap-combinations", type=int, default=32_768)
    parser.add_argument("--chunk-preset-pairs", type=int, default=64)
    args = parser.parse_args(argv)
    if args.bootstrap_replicates < 1 or args.bootstrap_combinations < 1 or args.chunk_preset_pairs < 1:
        parser.error("bootstrap counts and chunk size must be positive integers")

    progress = lambda message: print(message, flush=True)
    audit = verify_embedding_cache(args.dataset, args.checkpoint_root, progress=progress)
    print(f"Verified {audit['evaluation_arms']} cached arms across {audit['common_cells']} common preset-performance cells.")
    print("All seven encoder-family manifests, archive digests, checkpoint pins, taps, pooling, preprocessing and vectors passed.")
    result = analyze_cache(audit, device_name=args.device, bootstrap_replicates=args.bootstrap_replicates,
                           bootstrap_combinations=args.bootstrap_combinations, chunk_preset_pairs=args.chunk_preset_pairs,
                           progress=progress)
    figure_paths = generate_figures(result, args.report.parent / "figures")
    result["figures"] = figure_paths
    result_path = args.report.parent / "issue157_sensitivity_results.json"
    result_path.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    update_research_report(args.report, result)
    print(f"Saved machine-readable results to {result_path}.")
    print(f"Updated research report at {args.report}.")
    print(f"Generated figures: {', '.join(figure_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
