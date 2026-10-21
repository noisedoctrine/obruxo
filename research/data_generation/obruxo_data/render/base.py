from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any

from obruxo_data.errors import Diagnostic, OutputExistsError
from obruxo_data.hashing import canonical_json, canonical_sha256
from obruxo_data.render.capabilities import RendererCapabilities

if TYPE_CHECKING:
    import numpy as np

    from obruxo_data.midi import Performance


DEFAULT_RENDERER_ID = (
    "vital-1.6.4-vst3-sha256-"
    "a622a2c99b4066cd7945a4ab9bbdd698e7632a30702f6f0a7ccbf26a56b576e1-dawdreamer-0.8.3-state-verified-v3"
)
LEGACY_TRAINING_RENDERER_ID = DEFAULT_RENDERER_ID.removesuffix("v3") + "v2"
FULL_PRESET_RENDERER_ID = DEFAULT_RENDERER_ID.removesuffix("state-verified-v3") + "full-1.5.5-state-verified-v1"
FULL_SEED_RENDERER_ID = FULL_PRESET_RENDERER_ID.removesuffix("v1") + "seed-bits-v2"
FULL_EXPRESSION_RENDERER_ID = FULL_SEED_RENDERER_ID + "-expression-midi-v1"
FULL_CURVE_RENDERER_ID = FULL_EXPRESSION_RENDERER_ID + "-curve-input-v1"
FULL_CORPUS_RENDERER_ID = FULL_CURVE_RENDERER_ID + "-corpus-readback-v1"


@dataclass(frozen=True, init=False)
class RenderRequest:
    preset_json: str
    performance: "Performance"
    sample_rate: int = 44_100
    channels: int = 2
    end_tick: int | None = None
    tail_seconds: float = 2.0
    renderer_id: str = DEFAULT_RENDERER_ID
    preroll_frames: int = 0

    def __init__(self, *, performance: "Performance", preset: Any | None = None, preset_json: str | None = None,
                 sample_rate: int = 44_100, channels: int = 2, end_tick: int | None = None,
                 tail_seconds: float = 2.0, renderer_id: str = DEFAULT_RENDERER_ID, preroll_frames: int = 0):
        if (preset is None) == (preset_json is None):
            raise ValueError("provide exactly one of preset or preset_json")
        if preset is not None:
            preset_json = preset.to_json(canonical=True)
        assert preset_json is not None
        if type(preroll_frames) is not int or not 0 <= preroll_frames <= 44100:
            raise ValueError("preroll_frames must be an integer in [0, 44100]")
        if renderer_id in {FULL_PRESET_RENDERER_ID, FULL_SEED_RENDERER_ID, FULL_EXPRESSION_RENDERER_ID, FULL_CURVE_RENDERER_ID, FULL_CORPUS_RENDERER_ID}:
            from obruxo_data.vital.full_contract import FullPresetContract
            document, report = FullPresetContract().loads(preset_json)
            report.require_valid()
        else:
            document = json.loads(preset_json)
        canonical_preset = canonical_json(document)
        object.__setattr__(self, "preset_json", canonical_preset)
        object.__setattr__(self, "performance", performance)
        object.__setattr__(self, "sample_rate", sample_rate)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "end_tick", end_tick)
        object.__setattr__(self, "tail_seconds", tail_seconds)
        object.__setattr__(self, "renderer_id", renderer_id)
        object.__setattr__(self, "preroll_frames", preroll_frames)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "preset_json": self.preset_json,
            "performance": self.performance.to_dict(),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "end_tick": self.performance.end_tick if self.end_tick is None else self.end_tick,
            "tail_seconds": self.tail_seconds,
            "renderer_id": self.renderer_id,
        }
        if self.preroll_frames:
            payload["preroll_frames"] = self.preroll_frames
        return payload

    @property
    def request_id(self) -> str:
        return canonical_sha256(self.identity_payload())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RenderRequest":
        from obruxo_data.midi import Performance

        return cls(
            preset_json=value["preset_json"], performance=Performance.from_dict(value["performance"]),
            sample_rate=int(value["sample_rate"]), channels=int(value["channels"]),
            end_tick=None if value.get("end_tick") is None else int(value["end_tick"]),
            tail_seconds=float(value["tail_seconds"]), renderer_id=str(value["renderer_id"]),
            preroll_frames=value.get("preroll_frames", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.identity_payload()


@dataclass(frozen=True)
class RenderProvenance:
    request_id: str
    renderer_id: str
    backend_version: str
    engine_fingerprint: str
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "renderer_id": self.renderer_id,
            "backend_version": self.backend_version,
            "engine_fingerprint": self.engine_fingerprint,
            "settings": self.settings,
        }


@dataclass
class RenderResult:
    audio: "np.ndarray"
    sample_rate: int
    diagnostics: tuple[Diagnostic, ...]
    provenance: RenderProvenance
    qa: dict[str, Any] = field(default_factory=dict)

    def write_wav(self, path: Path | str, *, force: bool = False) -> None:
        destination = Path(path)
        if destination.exists() and not force:
            raise OutputExistsError(f"refusing to overwrite {destination}; pass force=True")
        from scipy.io import wavfile

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
            wavfile.write(temporary, self.sample_rate, self.audio)
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def write_json(self, path: Path | str, *, force: bool = False) -> None:
        destination = Path(path)
        if destination.exists() and not force:
            raise OutputExistsError(f"refusing to overwrite {destination}; pass force=True")
        value = {
            "sample_rate": self.sample_rate,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "provenance": self.provenance.to_dict(),
            "qa": self.qa,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=destination.parent,
                                             prefix=f".{destination.name}.", suffix=".tmp", delete=False) as stream:
                stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                temporary = Path(stream.name)
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


class Renderer(ABC):
    @property
    def max_workers(self) -> int | None:
        return None

    @property
    @abstractmethod
    def capabilities(self) -> RendererCapabilities: ...

    @abstractmethod
    def render(self, request: RenderRequest) -> RenderResult: ...
