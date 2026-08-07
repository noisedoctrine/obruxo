from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RendererCapabilities:
    notes: bool = True
    polyphony: bool = True
    tempo_changes: bool = False
    pitch_bend: bool = False
    channel_pressure: bool = False
    control_changes: frozenset[int] = frozenset()
    max_channels: int = 16

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RendererCapabilities":
        return cls(
            notes=bool(value.get("notes", False)),
            polyphony=bool(value.get("polyphony", False)),
            tempo_changes=bool(value.get("tempo_changes", False)),
            pitch_bend=bool(value.get("pitch_bend", False)),
            channel_pressure=bool(value.get("channel_pressure", False)),
            control_changes=frozenset(int(item) for item in value.get("control_changes", [])),
            max_channels=int(value.get("max_channels", 16)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "notes": self.notes,
            "polyphony": self.polyphony,
            "tempo_changes": self.tempo_changes,
            "pitch_bend": self.pitch_bend,
            "channel_pressure": self.channel_pressure,
            "control_changes": sorted(self.control_changes),
            "max_channels": self.max_channels,
        }
