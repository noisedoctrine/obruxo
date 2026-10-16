"""Compatibility import for the shared full 1.5.5 preset contract."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_generation"))

from obruxo_data.vital.full_contract import FullPresetContract, json_digest, strict_json  # noqa: E402, F401
