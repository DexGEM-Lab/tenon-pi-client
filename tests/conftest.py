from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="session")
def contract_fixtures() -> dict[str, Any]:
    path = Path(__file__).parent / "fixtures" / "rest-contract.json"
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def dataset_v3_source(tmp_path_factory):
    from dataset_v3_fixture import write_v3_fixture

    return write_v3_fixture(tmp_path_factory.mktemp("robot-v3"))
