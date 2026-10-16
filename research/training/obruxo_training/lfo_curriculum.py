"""The fixed authored curve population used for the train-only Era2 bank."""
import hashlib
from pathlib import Path

import numpy as np

POLICY = "authored_piecewise_float32_index_rng_v1"


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def authored_lfos(index):
    rng, shapes = np.random.default_rng(index), []
    for slot in range(1, 9):
        count = 3 + (index + slot) % 6
        x = np.linspace(0, 1, count, dtype=np.float32)
        y = rng.uniform(.05, .95, count).astype(np.float32)
        shapes.append({"name": f"Authored {index} slot {slot}", "num_points": count,
                       "points": np.column_stack((x, y)).reshape(-1).tolist(),
                       "powers": rng.uniform(-4, 4, count).astype(np.float32).tolist(), "smooth": bool((index + slot) % 2)})
    return shapes
