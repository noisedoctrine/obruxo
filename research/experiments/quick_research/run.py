"""Explicit entry point; importing this file never starts a research task."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for directory in (ROOT / "research/data_generation", ROOT / "research/modelling/basic_pitch"):
    sys.path.insert(0, str(directory))

if __name__ == "__main__":
    from quickresearch.cli import main
    raise SystemExit(main())
