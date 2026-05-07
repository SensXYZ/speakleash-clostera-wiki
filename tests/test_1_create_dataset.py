"""Tests for 1_create_dataset.py — parse_count and CLI smoke tests."""

import subprocess
import sys

import pytest


class TestParseCount:
    """Unit tests for parse_count() — accepts human-friendly count strings."""

    def test_plain_integer(self, mod_1):
        assert mod_1.parse_count("50000") == 50_000

    def test_k_suffix(self, mod_1):
        assert mod_1.parse_count("10k") == 10_000

    def test_k_uppercase(self, mod_1):
        assert mod_1.parse_count("100K") == 100_000

    def test_m_suffix(self, mod_1):
        assert mod_1.parse_count("1M") == 1_000_000

    def test_float_k(self, mod_1):
        assert mod_1.parse_count("1.5k") == 1_500

    def test_underscore_separator(self, mod_1):
        assert mod_1.parse_count("10_000") == 10_000

    def test_whitespace(self, mod_1):
        assert mod_1.parse_count(" 10k ") == 10_000

    def test_negative_raises(self, mod_1):
        with pytest.raises(Exception):  # argparse.ArgumentTypeError
            mod_1.parse_count("-5")

    def test_zero_raises(self, mod_1):
        with pytest.raises(Exception):
            mod_1.parse_count("0")

    def test_just_k_raises(self, mod_1):
        with pytest.raises(Exception):
            mod_1.parse_count("k")


class TestCLISmoke:
    """--help should exit cleanly."""

    def test_help(self):
        result = subprocess.run(
            [sys.executable, "1_create_dataset.py", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "sample" in result.stdout.lower() or "count" in result.stdout.lower()
