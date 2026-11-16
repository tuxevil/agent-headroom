"""Offline tests for IA Solar's provisional hybrid KV catalog."""
import tempfile
import unittest
from pathlib import Path

from bench.solar_catalog import PROFILES, build_catalog, growing_kv_bytes_per_token, parse_artifacts


class SolarCatalogTests(unittest.TestCase):
    def test_growing_kv_bf16_bytes_per_token(self):
        expected = {
            "gemma4-e2b": 6 * 1024,
            "qwen3.5-2b": 12 * 1024,
            "lfm3-1.7b": 12 * 1024,
            "minicpm5-2b": 42 * 1024,
            "nemotron3-nano-4b": 16 * 1024,
        }
        self.assertEqual({key: growing_kv_bytes_per_token(p) for key, p in PROFILES.items()}, expected)

    def test_bounded_kv_gemma_and_lfm(self):
        self.assertEqual(PROFILES["gemma4-e2b"]["bounded_kv_bytes"], 6 * 1024 * 1024)
        self.assertEqual(PROFILES["lfm3-1.7b"]["bounded_kv_bytes"], int(2.5 * 1024 * 1024))

    def test_catalog_validates_and_flags_estimates(self):
        base = {
            "catalog_version": "test-1",
            "engines": {"llama.cpp": {"overhead_bytes": 1024}},
            "models": [],
        }
        cat = build_catalog(base)
        self.assertEqual(len(cat["models"]), 5)
        self.assertTrue(all(not m["verified"] for m in cat["models"]))
        self.assertTrue(all(m["files"][0]["bytes"] > 0 for m in cat["models"]))
        self.assertTrue(all(not m["_solar"]["file_size_measured"] for m in cat["models"]))

    def test_artifact_size_overrides_estimate(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "placeholder.gguf"
            p.write_bytes(b"not-a-real-gguf")
            artifacts = parse_artifacts([f"gemma4-e2b={p}"])
            cat = build_catalog({
                "catalog_version": "test-1",
                "engines": {"llama.cpp": {"overhead_bytes": 1024}},
                "models": [],
            }, artifacts)
            gemma = next(m for m in cat["models"] if m["id"] == "gemma4-e2b")
            self.assertEqual(gemma["files"][0]["bytes"], p.stat().st_size)
            self.assertTrue(gemma["_solar"]["file_size_measured"])
            self.assertIsNone(gemma["files"][0]["sha256"])

    def test_invalid_artifact_input_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_artifacts(["not-in-catalog=/tmp/model.gguf"])


if __name__ == "__main__":
    unittest.main()
