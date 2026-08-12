"""Thin command-line wrapper for Basic Pitch import and parity."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from obruxo_basic_pitch.inference import prepare_wav
from obruxo_basic_pitch.parity import (
    assert_parity,
    compare_windows_and_audio,
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
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "import-onnx":
        metadata = write_imported_checkpoint(args.onnx, args.checkpoint, args.metadata, force=args.force)
        print(f"imported {metadata.model_id} from the pinned public artifact")
        return 0

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
    write_reports(summary, args.json, args.markdown, private_local_clips=len(args.audio), force=args.force)
    print(f"parity passed for {summary.synthetic_windows + summary.private_local_windows} windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
