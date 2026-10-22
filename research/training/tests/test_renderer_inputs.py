import copy
import json
from pathlib import Path
import runpy

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "reports/scripts/compare_renderer_inputs.py"
HELPERS = runpy.run_path(str(SCRIPT))


def test_comparison_refuses_unpaired_recipes_initial_weights_and_inference_sources():
    plan = json.loads((SCRIPT.parent.parent / "data/calibrated_native_inverse_fit_summary.json").read_text())["plan"]
    cold = copy.deepcopy(plan)
    cold["render_profile"]["preroll_frames"] = 0
    HELPERS["require_pair_contract"](cold, plan)
    for field in ("training", "initial_file_sha256", "frontend"):
        changed = copy.deepcopy(plan)
        changed[field] = [] if field == "training" else "different"
        with pytest.raises(ValueError, match="identity"):
            HELPERS["require_pair_contract"](cold, changed)
    changed = copy.deepcopy(plan)
    changed["source"]["features.py"] = "different"
    with pytest.raises(ValueError, match="source differs"):
        HELPERS["require_pair_contract"](cold, changed)
    with pytest.raises(ValueError, match="profiles"):
        HELPERS["require_pair_contract"](plan, plan)


def test_feature_differences_retain_absolute_scale_and_refuse_invalid_pairs():
    before = np.array([[1., 2.]], dtype=np.float32)
    same = HELPERS["feature_change"](before, before.copy())
    assert same["exact_equal"] and same["relative_l2"] == same["mean_absolute_delta"] == 0
    changed = HELPERS["feature_change"](before, before * 2)
    assert changed["relative_l2"] == pytest.approx(1.)
    assert changed["mean_absolute_delta"] == pytest.approx(1.5) and not changed["exact_equal"]
    for invalid in (before[0], before * np.nan):
        with pytest.raises(ValueError, match="shape or finiteness"):
            HELPERS["feature_change"](before, invalid)
