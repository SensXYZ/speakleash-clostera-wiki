"""Tests for 2_embed_and_cluster.py — JSONL iteration, checkpoint logic, embedding."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


class TestIterJsonl:
    """iter_jsonl should parse valid JSONL and skip blanks."""

    def test_valid_records(self, mod_2, tmp_jsonl, sample_records):
        path = tmp_jsonl(sample_records)
        result = list(mod_2.iter_jsonl(path))
        assert len(result) == 10
        assert result[0]["text"] == "Treść artykułu 0"

    def test_empty_lines_skipped(self, mod_2, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text(
            '{"text": "a"}\n\n  \n{"text": "b"}\n',
            encoding="utf-8",
        )
        result = list(mod_2.iter_jsonl(path))
        assert len(result) == 2

    def test_missing_file_raises(self, mod_2, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(mod_2.iter_jsonl(tmp_path / "nonexistent.jsonl"))


class TestCkptMetaPath:
    """_ckpt_meta_path should swap .npy for .meta.json."""

    def test_basic(self, mod_2):
        result = mod_2._ckpt_meta_path(Path("out/vectors_partial.npy"))
        assert result == Path("out/vectors_partial.meta.json")

    def test_already_meta(self, mod_2):
        # Edge: what if someone passes a non-.npy path?
        result = mod_2._ckpt_meta_path(Path("out/checkpoint.bin"))
        assert result.suffix == ".json"


class TestCkptMeta:
    """_ckpt_meta should capture input file stats and embedding config."""

    def test_returns_expected_keys(self, mod_2, tmp_jsonl, sample_records):
        path = tmp_jsonl(sample_records)
        meta = mod_2._ckpt_meta(path, n_records=10, model="test-model", max_chars=2000, batch_size=32)
        assert set(meta.keys()) == {
            "input_path", "input_mtime", "input_size",
            "n_records", "model", "max_chars", "batch_size",
        }
        assert meta["model"] == "test-model"
        assert meta["n_records"] == 10
        assert meta["input_size"] == path.stat().st_size


class TestCkptValid:
    """_ckpt_valid should accept matching metadata and reject mismatches."""

    def test_matching_meta(self, mod_2, tmp_path):
        meta_path = tmp_path / "vectors_partial.meta.json"
        expected = {"model": "x", "n_records": 5}
        meta_path.write_text(json.dumps(expected), encoding="utf-8")
        assert mod_2._ckpt_valid(meta_path, expected) is True

    def test_mismatched_meta(self, mod_2, tmp_path):
        meta_path = tmp_path / "vectors_partial.meta.json"
        meta_path.write_text(json.dumps({"model": "x"}), encoding="utf-8")
        assert mod_2._ckpt_valid(meta_path, {"model": "y"}) is False

    def test_missing_file(self, mod_2, tmp_path):
        assert mod_2._ckpt_valid(tmp_path / "missing.json", {}) is False

    def test_corrupt_json(self, mod_2, tmp_path):
        meta_path = tmp_path / "vectors_partial.meta.json"
        meta_path.write_text("not json at all{{{", encoding="utf-8")
        assert mod_2._ckpt_valid(meta_path, {}) is False


class TestEmbedAll:
    """embed_all with mocked OpenAI client — fresh run, resume, config mismatch."""

    @staticmethod
    def _mock_client(dim=8):
        """Build a mock OpenAI client that returns fixed embeddings."""
        client = MagicMock()
        rng = np.random.default_rng(42)

        def fake_create(model, input):
            resp = MagicMock()
            resp.data = []
            for i, text in enumerate(input):
                item = MagicMock()
                item.embedding = rng.standard_normal(dim).tolist()
                item.index = i
                resp.data.append(item)
            return resp

        client.embeddings.create = fake_create
        return client

    def test_fresh_run_shape(self, mod_2, sample_records, tmp_path):
        client = self._mock_client(dim=16)
        ckpt = tmp_path / "vectors_partial.npy"
        input_path = tmp_path / "test.jsonl"
        input_path.write_text("{}", encoding="utf-8")

        vectors = mod_2.embed_all(
            sample_records, client, "test-model",
            batch_size=3, max_chars=500,
            checkpoint_path=ckpt, input_path=input_path,
        )
        assert vectors.shape == (10, 16)
        assert vectors.dtype == np.float32

    def test_fresh_run_no_residual_checkpoint(self, mod_2, sample_records, tmp_path):
        client = self._mock_client(dim=4)
        ckpt = tmp_path / "vectors_partial.npy"
        input_path = tmp_path / "test.jsonl"
        input_path.write_text("{}", encoding="utf-8")

        mod_2.embed_all(
            sample_records, client, "test-model",
            batch_size=5, max_chars=500,
            checkpoint_path=ckpt, input_path=input_path,
        )
        # Checkpoint should be saved during embedding
        assert ckpt.exists()
        saved = np.load(ckpt)
        assert saved.shape[0] == 10

    def test_resume_from_checkpoint(self, mod_2, sample_records, tmp_path):
        dim = 8
        client = self._mock_client(dim=dim)
        ckpt = tmp_path / "vectors_partial.npy"
        input_path = tmp_path / "test.jsonl"
        input_path.write_text("{}", encoding="utf-8")

        # Pre-create a checkpoint with 5 rows already embedded
        partial = np.random.randn(5, dim).astype(np.float32)
        np.save(ckpt, partial)
        meta = mod_2._ckpt_meta(input_path, 10, "test-model", 500, 5)
        meta_path = mod_2._ckpt_meta_path(ckpt)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        # The mock client should only be called for the remaining 5
        vectors = mod_2.embed_all(
            sample_records, client, "test-model",
            batch_size=5, max_chars=500,
            checkpoint_path=ckpt, input_path=input_path,
        )
        assert vectors.shape == (10, dim)
        # The first 5 rows should come from the checkpoint
        np.testing.assert_array_almost_equal(vectors[:5], partial)

    def test_config_mismatch_starts_fresh(self, mod_2, sample_records, tmp_path):
        dim = 4
        client = self._mock_client(dim=dim)
        ckpt = tmp_path / "vectors_partial.npy"
        input_path = tmp_path / "test.jsonl"
        input_path.write_text("{}", encoding="utf-8")

        # Pre-create checkpoint with WRONG model name
        partial = np.random.randn(5, dim).astype(np.float32)
        np.save(ckpt, partial)
        meta = mod_2._ckpt_meta(input_path, 10, "WRONG-model", 500, 5)
        meta_path = mod_2._ckpt_meta_path(ckpt)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        vectors = mod_2.embed_all(
            sample_records, client, "test-model",
            batch_size=5, max_chars=500,
            checkpoint_path=ckpt, input_path=input_path,
        )
        assert vectors.shape == (10, dim)
        # First 5 rows should NOT match the stale checkpoint
        with pytest.raises(AssertionError):
            np.testing.assert_array_almost_equal(vectors[:5], partial)


class TestCLISmoke:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, "2_embed_and_cluster.py", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "embed" in result.stdout.lower()
