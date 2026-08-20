from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = WORKSPACE
BASIC_PITCH_ROOT = WORKSPACE.parent / "basic_pitch"
for path in (PACKAGE_ROOT, BASIC_PITCH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
