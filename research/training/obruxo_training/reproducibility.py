"""Explicit oscillator-phase controls for native pipeline correctness checks."""

import json


OSCILLATOR_PHASE_CONTROL = {f"osc_{index}_random_phase": 0. for index in range(1, 4)}


def fixed_oscillator_phases(raw: bytes) -> bytes:
    """Create a diagnostic variant; preserve the patch's fixed phase knobs and other settings."""
    document = json.loads(raw)
    document["settings"].update(OSCILLATOR_PHASE_CONTROL)
    return json.dumps(document, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
