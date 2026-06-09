"""Shared fixtures for the test suite."""

import importlib.util
import json
from pathlib import Path

import pytest

# Project root (one level up from tests/)
ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers for importing digit-prefixed module names (e.g. "1_create_dataset")
# ---------------------------------------------------------------------------

def _import(script_name: str):
    """Import a script by filename using spec_from_file_location.

    This handles digit-prefixed names that importlib.import_module rejects.
    """
    path = ROOT / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(script_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def mod_1():
    return _import("1_create_dataset")


@pytest.fixture()
def mod_2():
    return _import("2_embed_and_cluster")


@pytest.fixture()
def mod_3():
    return _import("3_label_and_visualize")


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

SAMPLE_RECORDS = [
    {"text": f"Treść artykułu {i}", "meta": {"title": f"Artykuł {i}", "length": 100 + i}}
    for i in range(10)
]


@pytest.fixture()
def sample_records():
    return [dict(r) for r in SAMPLE_RECORDS]


@pytest.fixture()
def tmp_jsonl(tmp_path):
    """Factory fixture: writes records to a JSONL file and returns the path."""
    def _write(records, name="test.jsonl"):
        path = tmp_path / name
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return path
    return _write
