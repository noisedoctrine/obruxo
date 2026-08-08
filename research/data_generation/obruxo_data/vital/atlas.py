from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_SCHEMA_DIR = Path(__file__).with_name("schema") / "vital-1.0.8-vita-0.1.0"


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    minimum: float
    maximum: float
    default: float
    scale: str
    is_discrete: bool
    options: tuple[str, ...] = ()
    display_units: str = ""
    display_name: str = ""
    post_offset: float = 0.0
    display_multiply: float = 1.0

    @classmethod
    def from_dict(cls, name: str, value: dict[str, Any]) -> "ParameterSpec":
        return cls(
            name=name,
            minimum=float(value["min"]),
            maximum=float(value["max"]),
            default=float(value["default"]),
            scale=str(value["scale"]),
            is_discrete=bool(value["is_discrete"]),
            options=tuple(str(item) for item in value.get("options", [])),
            display_units=str(value.get("display_units", "")),
            display_name=str(value.get("display_name", name)),
            post_offset=float(value.get("post_offset", 0.0)),
            display_multiply=float(value.get("display_multiply", 1.0)),
        )

    def validate_raw(self, value: float) -> str | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "must be a number"
        if not math.isfinite(value):
            return "must be finite"
        if value < self.minimum or value > self.maximum:
            return f"must be between {self.minimum:g} and {self.maximum:g}"
        if (self.is_discrete or self.scale == "Indexed") and not float(value).is_integer():
            return "must be an integral ordinal"
        return None

    def raw_from_normalized(self, value: float) -> float:
        self._validate_normalized(value)
        span = self.maximum - self.minimum
        if self.scale == "Indexed":
            count = int(round(span)) + 1
            return self.minimum + round(value * (count - 1))
        if self.scale == "Quadratic":
            position = math.sqrt(value)
        elif self.scale == "Cubic":
            position = value ** (1.0 / 3.0)
        elif self.scale == "SquareRoot":
            position = value * value
        else:
            position = value
        return self.minimum + position * span

    def normalized_from_raw(self, value: float) -> float:
        message = self.validate_raw(value)
        if message:
            raise ValueError(f"{self.name} {message}")
        span = self.maximum - self.minimum
        if span == 0:
            return 0.0
        position = (float(value) - self.minimum) / span
        if self.scale == "Indexed":
            count = int(round(span)) + 1
            return round(float(value) - self.minimum) / (count - 1) if count > 1 else 0.0
        if self.scale == "Quadratic":
            return position * position
        if self.scale == "Cubic":
            return position * position * position
        if self.scale == "SquareRoot":
            return math.sqrt(position)
        return position

    @staticmethod
    def _validate_normalized(value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("normalized value must be finite")
        if value < 0.0 or value > 1.0:
            raise ValueError("normalized value must be between 0 and 1")


@dataclass(frozen=True)
class VitalSchema:
    schema_id: str
    vita_revision: str
    vital_revision: str
    init_preset_path: Path
    parameter_atlas_path: Path
    modulation_vocab_path: Path
    init_preset_sha256: str
    engine_identity: dict[str, Any]

    @classmethod
    def load(cls, directory: Path | None = None) -> "VitalSchema":
        root = (directory or DEFAULT_SCHEMA_DIR).resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        return cls(
            schema_id=manifest["schema_id"],
            vita_revision=manifest["vita_revision"],
            vital_revision=manifest["vital_revision"],
            init_preset_path=root / manifest["init_preset"],
            parameter_atlas_path=root / manifest["parameter_atlas"],
            modulation_vocab_path=root / manifest["modulation_vocab"],
            init_preset_sha256=manifest["init_preset_sha256"],
            engine_identity=dict(manifest["engine_identity"]),
        )

    @property
    def parameters(self) -> dict[str, ParameterSpec]:
        values = json.loads(self.parameter_atlas_path.read_text(encoding="utf-8"))
        return {name: ParameterSpec.from_dict(name, value) for name, value in values["parameters"].items()}

    @property
    def modulation_sources(self) -> frozenset[str]:
        values = json.loads(self.modulation_vocab_path.read_text(encoding="utf-8"))
        return frozenset(values["sources"])

    @property
    def modulation_destinations(self) -> frozenset[str]:
        values = json.loads(self.modulation_vocab_path.read_text(encoding="utf-8"))
        return frozenset(values["destinations"])

    def load_init_document(self) -> dict[str, Any]:
        import hashlib

        raw = self.init_preset_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != self.init_preset_sha256:
            raise ValueError(f"schema init hash mismatch: expected {self.init_preset_sha256}, got {digest}")
        return json.loads(raw.decode("utf-8-sig"))
