"""Designer-selected execution batches, independent of the optimizer's example weights."""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import torch

EXECUTION_BATCH_SOURCES = {"batching.py", "native_gradient.py", "native_inverse.py", "native_acceptance.py", "spectral.py", "features.py",
    "basic_direct_fit.py", "basic_direct_fit_data.py", "basic_direct_fit_step.py", "basic_direct_fit_audit.py", "paired_native_evidence.py",
    "shared_basic_auxiliary.py", "balanced_basic_fit.py", "balanced_basic_audit.py", "unison_sound_fit.py", "unison_fit_data.py",
    "evaluate.py", "trainable_basic_fit.py", "balanced_basic_data.py",
    "reports/scripts/run_basic_direct_fit.py", "reports/scripts/run_balanced_basic_fit.py"}


def compatible_infrastructure_source(name, historical, current):
    """Only audited PR #141 migrations; unknown codec/trainer edits remain refused."""
    name = name.removeprefix("obruxo_training/")
    migrations = {
        "codec.py": ("62e915fd0aae9d7e77644771deb4fc216ac2f614f930628f7aaa9fa63e23edb8",
                     "6b2d4b37b7674d80bc98825d6b4efae2d13a01d4808c175826edafa1c230aaee"),
        "trainer.py": ("92cae41f88618e34f02f624aa3086bf9b1c7c5a9334b19409bbc8fe633d24f3f",
                       "c795223a25d7a7c6ae67769f75eaea43cae161ad8a8126a99f8ad209d3543173")}
    if (historical, current) != migrations.get(name):
        return False
    if name == "trainer.py":
        return True  # Removal of the fresh-worker-only guard, qualified by the native lifetime checks.
    shared = Path(__file__).parents[2] / "data_generation/obruxo_data/render/basic_profile.py"
    return hashlib.sha256(shared.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == "217066f01b51e45828aeef2e5b11cdce8059278b9baa9e408f9084a49ed6e188"


@dataclass(frozen=True)
class BatchConfig:
    model: int = 32
    spectral: int = 16
    pitch: int = 32

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise ValueError("execution batch sizes must be positive integers")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_plan(cls, plan):
        return cls(**plan.get("batching", {}))


def prediction_row(prediction, index):
    return {key: [item[index:index + 1] for item in value] if key == "categorical" else value[index:index + 1]
            for key, value in prediction.items()}


def predict_features(model, features, batch_size):
    from .model import batch_features
    BatchConfig(model=batch_size)
    rows = []
    for start in range(0, len(features), batch_size):
        chunk = features[start:start + batch_size]
        output = model(batch_features([{"features": feature} for feature in chunk]))
        rows.extend(prediction_row(output, index) for index in range(len(chunk)))
    return rows


def predict_cells(inputs, model, cells):
    if hasattr(inputs, "predict_many"):
        return inputs.predict_many(model, cells)
    # Preserve custom validators in historical/external scalar prediction interfaces.
    return [inputs.predict(model, cell) for cell in cells]


def concatenate_predictions(rows):
    return {key: [torch.cat([row[key][index] for row in rows]) for index in range(len(value))]
            if key == "categorical" else torch.cat([row[key] for row in rows]) for key, value in rows[0].items()}
