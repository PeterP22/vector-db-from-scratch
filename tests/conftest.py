import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def small_data():
    rng = np.random.default_rng(1234)
    base = rng.normal(size=(500, 16)).astype(np.float32)
    queries = rng.normal(size=(20, 16)).astype(np.float32)
    return base, queries
