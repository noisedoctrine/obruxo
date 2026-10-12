from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[4]
for directory in (REPO / "research/experiments/quick_research", REPO / "research/data_generation", REPO / "research/modelling/basic_pitch"):
    sys.path.insert(0, str(directory))
