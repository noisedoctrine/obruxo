"""Validated in-memory representation of Vital drawable LFO state."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LfoShape:
    name: str
    points: np.ndarray
    powers: np.ndarray
    smooth: bool

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "LfoShape":
        raw_points = np.asarray(value.get("points", []), dtype=np.float64)
        if raw_points.ndim != 1 or raw_points.size % 2:
            raise ValueError("LFO points must be a flat sequence of x/y pairs")

        points = raw_points.reshape(-1, 2)
        declared = int(value.get("num_points", len(points)))
        if declared != len(points):
            raise ValueError(
                f"num_points={declared} does not match {len(points)} coordinate pairs"
            )
        if len(points) < 2:
            raise ValueError("an LFO shape needs at least two points")
        if np.any(np.diff(points[:, 0]) < 0.0):
            raise ValueError("LFO x coordinates must be non-decreasing")

        powers = np.asarray(value.get("powers", []), dtype=np.float64)
        if powers.shape != (len(points),):
            raise ValueError("LFO powers must contain one value per point")

        return cls(
            name=str(value.get("name", "")),
            points=points,
            powers=powers,
            smooth=bool(value.get("smooth", False)),
        )

    @classmethod
    def from_serialized(cls, points: str, powers: str, **kwargs: Any) -> "LfoShape":
        point_values = json.loads(points)
        power_values = json.loads(powers)
        return cls.from_json(
            {
                "name": kwargs.get("name", ""),
                "num_points": len(point_values) // 2,
                "points": point_values,
                "powers": power_values,
                "smooth": kwargs.get("smooth", False),
            }
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_points": len(self.points),
            "points": self.points.reshape(-1).tolist(),
            "powers": self.powers.tolist(),
            "smooth": self.smooth,
        }

    def signature(self) -> str:
        """Hash rendered-relevant serialized geometry, excluding the mutable name."""
        state = self.to_json()
        state.pop("name")
        payload = json.dumps(
            state, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

