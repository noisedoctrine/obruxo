from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
for path in (WORKSPACE, WORKSPACE.parent / "basic_pitch"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
