"""Profile-checking fake workers without importing Torch in every child."""

from dataclasses import dataclass, replace
import json

from obruxo_training.reproducibility import FULL_PHASE_CONTROL
from test_stream import FakeRenderer


class ProfileRenderer(FakeRenderer):
    def render(self, request):
        assert request.preroll_frames == 32768
        assert all(json.loads(request.preset_json)["settings"][name] == value for name, value in FULL_PHASE_CONTROL.items())
        result = super().render(request)
        result.provenance = replace(result.provenance, settings={"preroll_frames": request.preroll_frames})
        return result


@dataclass(frozen=True)
class ProfileRendererFactory:
    def __call__(self):
        return ProfileRenderer("valid")
