"""Explicit dispatch for the four candidate families; no plugin mechanism."""

from __future__ import annotations

from typing import Any

from ..artifacts import ModelSpec
from .basic_pitch import BasicPitchAdapter
from .muscriptor import MuScriptorAdapter
from .timbre_trap import TimbreTrapAdapter
from .yourmt3 import YourMT3Adapter


def adapter_for(
    spec: ModelSpec,
    source_root: Any = None,
    checkpoint: Any = None,
    segment_batch_size: int | None = None,
) -> object:
    if spec.family == "basic_pitch":
        return BasicPitchAdapter(spec, source_root, checkpoint)
    if spec.family == "timbre_trap":
        return TimbreTrapAdapter(spec, source_root, checkpoint)
    if spec.family == "yourmt3":
        return YourMT3Adapter(
            spec, source_root, checkpoint, segment_batch_size=segment_batch_size
        )
    if spec.family == "muscriptor":
        return MuScriptorAdapter(spec, source_root, checkpoint)
    raise ValueError(f"unsupported model family: {spec.family}")
