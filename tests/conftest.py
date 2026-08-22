from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from habitat_monitor.generate import generate_split  # noqa: E402
from habitat_monitor.schema import LabeledRecord  # noqa: E402


@pytest.fixture(scope="session")
def labeled_split() -> tuple[list[LabeledRecord], list[LabeledRecord]]:
    return generate_split(n_train=200, n_test=160, seed=7)
