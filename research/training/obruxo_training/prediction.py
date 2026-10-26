"""Load inference artifacts without changing the qualified training loaders."""

import torch

from .basic_direct import FORMAT as DIRECT_FORMAT, restore_direct
from .trainable_basic import FORMAT as INPUT_FORMAT
from .trainer import restore_model


def load_predictor(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict):
        raise ValueError("unsupported inference artifact")
    format_name = saved.get("format")
    qualified = saved.get("qualified_for_release")
    if qualified is not None and type(qualified) is not bool:
        raise ValueError("inference artifact has an invalid release status")
    if format_name in (DIRECT_FORMAT, INPUT_FORMAT):
        model, frontend, codec = restore_direct(saved)
        model = model.to(device).eval()
        threads = 2  # Match the singleton forwards used by both native-fit audits.
    elif format_name == "obruxo_pilot_inference_v1":
        model, frontend, codec = restore_model(saved, device)
        threads = 4  # Preserve the existing CLI's setting for historical artifacts.
    else:
        raise ValueError("unsupported inference artifact; provide an exported model rather than a training checkpoint")
    return model, frontend, codec, {"format": format_name, "cpu_threads": threads, "qualified_for_release": qualified}
