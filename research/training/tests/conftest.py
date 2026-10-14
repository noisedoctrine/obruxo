from pathlib import Path
import sys

import pytest

RESEARCH = Path(__file__).resolve().parents[2]
for path in (RESEARCH / "training", RESEARCH / "data_generation"):
    sys.path.insert(0, str(path))

from obruxo_data.vital import VitalSchema  # noqa: E402
from obruxo_training.assets import AssetCatalog  # noqa: E402


@pytest.fixture(scope="session")
def schema():
    return VitalSchema.load()


@pytest.fixture(scope="session")
def catalog(schema):
    return AssetCatalog.reviewed_init(schema)


@pytest.fixture
def init_raw(schema):
    return schema.init_preset_path.read_bytes()
