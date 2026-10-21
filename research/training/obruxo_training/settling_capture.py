"""Diagnostic observer for native control state after processing; no audio changes."""

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

from .audit import write_json
from .codec import CATEGORICAL, CONTINUOUS
from .stream import NativeRendererFactory


@dataclass(frozen=True)
class SettlingCaptureFactory:
    directory: str
    renderer_id: str

    def __call__(self):
        return SettlingCaptureRenderer(NativeRendererFactory(renderer_id=self.renderer_id)(), Path(self.directory))


class SettlingCaptureRenderer:
    def __init__(self, renderer, directory):
        self.renderer, self.directory = renderer, directory

    def render(self, request):
        from obruxo_data.render.vita import VitalVst3StateTemplate
        code = type(self.renderer).render.__code__
        previous = sys.getprofile()
        names = [name for name, *_ in (*CONTINUOUS, *CATEGORICAL)] + ["volume", "osc_1_random_phase"]

        def observe(frame, event, value):
            if event != "return" or frame.f_code is not code or value is None:
                return
            path = self.directory / f"{request.request_id}.post.state"
            frame.f_locals["synth"].save_state(str(path))
            after = VitalVst3StateTemplate(path.read_bytes()).preset_document["settings"]
            before = frame.f_locals["native_document"]["settings"]
            loaded = frame.f_locals["loaded"]["settings"]
            write_json(self.directory / f"{request.request_id}.state.json",
                       {"requested": {name: before.get(name) for name in names}, "loaded": {name: loaded.get(name) for name in names},
                        "after_render": {name: after.get(name) for name in names}})

        try:
            sys.setprofile(observe)
            result = self.renderer.render(request)
        finally:
            sys.setprofile(previous)
        np.save(self.directory / f"{request.request_id}.npy", result.audio, allow_pickle=False)
        return result
