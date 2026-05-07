"""Tests for 3_label_and_visualize.py — labeling, sampling, visualization helpers."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


class TestArticleTitle:
    """article_title should extract title from meta, record, or fallback."""

    def test_meta_title(self, mod_3):
        rec = {"text": "abc", "meta": {"title": "Polska"}}
        assert mod_3.article_title(rec) == "Polska"

    def test_fallback_to_rec_title(self, mod_3):
        rec = {"text": "abc", "title": "Warszawa", "meta": {}}
        assert mod_3.article_title(rec) == "Warszawa"

    def test_no_meta_key(self, mod_3):
        rec = {"text": "abc", "title": "Kraków"}
        assert mod_3.article_title(rec) == "Kraków"

    def test_empty_fallback(self, mod_3):
        rec = {"text": "abc"}
        assert mod_3.article_title(rec) == "(bez tytułu)"

    def test_none_meta(self, mod_3):
        rec = {"text": "abc", "meta": None}
        assert mod_3.article_title(rec) == "(bez tytułu)"


class TestRepresentativeSamples:
    """representative_samples should pick N articles closest to each centroid."""

    def _make_data(self, n_records=20, n_clusters=3):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((n_records, 8)).astype(np.float32)
        labels = np.array([i % n_clusters for i in range(n_records)])
        records = [{"text": f"text {i}", "meta": {"title": f"art {i}"}} for i in range(n_records)]
        return vectors, labels, records

    def test_returns_correct_clusters(self, mod_3):
        vectors, labels, records = self._make_data(n_clusters=3)
        result = mod_3.representative_samples(vectors, labels, records, n_per_cluster=5)
        assert set(result.keys()) == {0, 1, 2}

    def test_returns_correct_count_per_cluster(self, mod_3):
        vectors, labels, records = self._make_data(n_clusters=3)
        result = mod_3.representative_samples(vectors, labels, records, n_per_cluster=4)
        for cid, samples in result.items():
            assert len(samples) == 4

    def test_fewer_records_than_requested(self, mod_3):
        # Cluster with only 2 articles but n_per_cluster=5
        vectors = np.random.randn(2, 4).astype(np.float32)
        labels = np.array([0, 0])
        records = [{"text": "a"}, {"text": "b"}]
        result = mod_3.representative_samples(vectors, labels, records, n_per_cluster=5)
        assert len(result[0]) == 2


class TestDiscretePalette:
    """discrete_palette should generate n visually distinct colors."""

    def test_returns_n_colors(self, mod_3):
        colors = mod_3.discrete_palette(10)
        assert len(colors) == 10

    def test_colors_are_rgb_strings(self, mod_3):
        colors = mod_3.discrete_palette(3)
        for c in colors:
            assert c.startswith("rgb(")
            assert c.endswith(")")

    def test_all_unique(self, mod_3):
        colors = mod_3.discrete_palette(50)
        assert len(set(colors)) == 50

    def test_zero(self, mod_3):
        assert mod_3.discrete_palette(0) == []

    def test_one(self, mod_3):
        colors = mod_3.discrete_palette(1)
        assert len(colors) == 1


class TestLoadData:
    """load_data should validate alignment of vectors, labels, and records."""

    def test_length_mismatch_raises(self, mod_3, tmp_jsonl, sample_records, tmp_path):
        # Write a clustered JSONL with 10 records
        records_with_cluster = [dict(r, cluster=0) for r in sample_records]
        jsonl_path = tmp_jsonl(records_with_cluster, name="test_clustered.jsonl")

        # But vectors and labels have wrong size
        vectors = np.random.randn(5, 8).astype(np.float32)
        np.save(tmp_path / "vectors.npy", vectors)
        np.save(tmp_path / "labels.npy", np.array([0, 0, 0, 0, 0]))

        with pytest.raises(ValueError, match="length mismatch"):
            mod_3.load_data(tmp_path, jsonl_path)

    def test_auto_detect_clustered_jsonl(self, mod_3, tmp_jsonl, sample_records, tmp_path):
        records_with_cluster = [dict(r, cluster=0) for r in sample_records]
        jsonl_path = tmp_jsonl(records_with_cluster, name="wiki_clustered.jsonl")

        n = len(sample_records)
        vectors = np.random.randn(n, 4).astype(np.float32)
        np.save(tmp_path / "vectors.npy", vectors)
        np.save(tmp_path / "labels.npy", np.zeros(n, dtype=int))

        v, l, r, p = mod_3.load_data(tmp_path, None)
        assert v.shape == (n, 4)
        assert len(r) == n


class TestNameCluster:
    """name_cluster with mocked OpenAI — verify prompt structure and parsing."""

    def _mock_client(self, response_text: str):
        client = MagicMock()
        msg = MagicMock()
        msg.content = response_text
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        client.chat.completions.create.return_value = resp
        return client

    def test_basic_response(self, mod_3):
        client = self._mock_client("Historia Polski")
        samples = [
            {"text": "Treść o historii", "meta": {"title": "Historia"}},
            {"text": "Więcej historii", "meta": {"title": "Polska"}},
        ]
        name = mod_3.name_cluster(client, "test-model", samples, max_chars=200, temperature=0.2)
        assert name == "Historia Polski"

    def test_strips_quotes(self, mod_3):
        client = self._mock_client('"Historia Polski"')
        samples = [{"text": "abc", "meta": {"title": "T"}}]
        name = mod_3.name_cluster(client, "test-model", samples, max_chars=200, temperature=0.2)
        assert name == "Historia Polski"

    def test_strips_trailing_dot(self, mod_3):
        client = self._mock_client("Historia Polski.")
        samples = [{"text": "abc", "meta": {"title": "T"}}]
        name = mod_3.name_cluster(client, "test-model", samples, max_chars=200, temperature=0.2)
        assert name == "Historia Polski"

    def test_sends_polish_prompt(self, mod_3):
        client = self._mock_client("Test")
        samples = [{"text": "abc", "meta": {"title": "T"}}]
        mod_3.name_cluster(client, "test-model", samples, max_chars=200, temperature=0.2)
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "polskiej Wikipedii" in user_msg
        assert "TYLKO nazwą" in user_msg


class TestReduceDims:
    """reduce_dims should produce the correct number of components."""

    def test_pca_2d(self, mod_3):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((50, 8)).astype(np.float32)
        coords = mod_3.reduce_dims(vectors, "pca", 2)
        assert coords.shape == (50, 2)

    def test_pca_3d(self, mod_3):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((50, 8)).astype(np.float32)
        coords = mod_3.reduce_dims(vectors, "pca", 3)
        assert coords.shape == (50, 3)


class TestCLISmoke:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, "3_label_and_visualize.py", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "label" in result.stdout.lower()
