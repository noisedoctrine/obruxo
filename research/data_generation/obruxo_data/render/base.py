from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from obruxo_data.errors import Diagnostic
from obruxo_data.hashing import canonical_sha256
from obruxo_data.render.capabilities import RendererCapabilities

if TYPE_CHECKING:
    from obruxo_data.midi import Performance


@dataclass(frozen=True)
class RenderRequest:
    preset_json: str
    performance: "Performance"
    sample_rate: int = 44_100
    channels: int = 2
    end_tick: int | None = None
    tail_seconds: float = 2.0
    renderer_id: str = "vital-vst3-dawdreamer-0.8.3"

    def identity_payload(self) -> dict[str, Any]:
        return {
            "preset_json": self.preset_json,
            "performance": self.performance.to_dict(),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "end_tick": self.performance.end_tick if self.end_tick is None else self.end_tick,
            "tail_seconds": self.tail_seconds,
            "renderer_id": self.renderer_id,
        }

    @property
    def request_id(self) -> str:
        return canonical_sha256(self.identity_payload())


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
    audio: np.ndarray
    sample_rate: int
    diagnostics: tuple[Diagnostic, ...]
    provenance: RenderProvenance
    qa: dict[str, Any] = field(default_factory=dict)


class Renderer(ABC):
    @property
    @abstractmethod
    def capabilities(self) -> RendererCapabilities: ...

    @abstractmethod
    def render(self, request: RenderRequest) -> RenderResult: ...
