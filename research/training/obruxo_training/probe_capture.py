"""Retain unmodified diagnostic audio without changing any native QA decision."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .stream import NativeRendererFactory


@dataclass(frozen=True)
class CaptureFactory:
    directory: str
    renderer_id: str

    def __call__(self):
        return CaptureRenderer(NativeRendererFactory(renderer_id=self.renderer_id)(), Path(self.directory))


class CaptureRenderer:
    def __init__(self, renderer, directory):
        self.renderer, self.directory = renderer, directory

    def render(self, request):
        result = self.renderer.render(request)
        np.save(self.directory / f"{request.request_id}.npy", result.audio, allow_pickle=False)
        return result
