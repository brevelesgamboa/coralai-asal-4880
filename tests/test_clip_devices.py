"""Check preprocessing parity, explicit-device failure, and ranking comparisons."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.clip_scoring import load_clip_image, select_device
from scripts.compare_clip_devices import compare_scores, near_ties, write_tables


class ClipDeviceTests(unittest.TestCase):
    def test_progression_preprocessing_matches_existing_rgb_lanczos_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.png"
            pixels = np.arange(31 * 29 * 4, dtype=np.uint8).reshape(31, 29, 4)
            Image.fromarray(pixels, "RGBA").save(path)
            with Image.open(path) as image:
                expected = np.asarray(image.convert("RGB").resize(
                    (224, 224), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
            actual = load_clip_image(path, resize=True)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, np.float32)
            with self.assertRaisesRegex(ValueError, "224 x 224"):
                load_clip_image(path)

    def test_unavailable_gpu_never_silently_uses_cpu(self):
        with patch("jax.devices", side_effect=RuntimeError("only CPU is installed")) as devices:
            with self.assertRaisesRegex(ValueError, "GPU is unavailable.*only CPU"):
                select_device("gpu")
            devices.assert_called_once_with("gpu")

    def test_comparison_distinguishes_near_tie_from_material_rank_reversal(self):
        cpu = np.array([[-0.5], [-0.500001], [-0.2]])
        gpu = np.array([[-0.500002], [-0.5], [-0.2]])
        result = compare_scores(cpu, gpu, ["cells"], 1e-5)
        self.assertAlmostEqual(result["max_absolute_difference"], 2e-6)
        rank = result["rankings"][0]
        self.assertFalse(rank["strict_order_agrees"])
        self.assertTrue(rank["agrees_outside_near_ties"])
        self.assertTrue(rank["inversions"][0]["near_tie"])
        gpu = np.array([[-0.1], [-0.5], [-0.2]])
        self.assertFalse(compare_scores(cpu, gpu, ["cells"], 1e-5)["rankings"][0]["agrees_outside_near_ties"])
        self.assertEqual(near_ties(cpu[:, 0], 1e-5), [True, True, False])

    def test_a_large_reference_gap_is_not_excused_by_a_gpu_near_tie(self):
        cpu = np.array([[-0.5], [-0.7]])
        gpu = np.array([[-0.500001], [-0.5]])
        result = compare_scores(cpu, gpu, ["cells"], 1e-5)
        self.assertFalse(result["rankings"][0]["agrees_outside_near_ties"])

    def test_unavailable_gpu_has_blank_scores_and_no_claimed_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            comparison = write_tables(output, ["frame.png"], ["cells"], {
                "cpu": {"status": "ok", "scores": [[-0.3]], "model_load_seconds": 1.0,
                        "median_scoring_seconds": 0.1, "median_ms_per_image": 100},
                "gpu": {"status": "unavailable", "error": "no backend"},
            }, 1e-5)
            self.assertIsNone(comparison)
            with (output / "scores.csv").open() as stream:
                row, = list(csv.DictReader(stream))
            self.assertEqual(float(row["cpu_loss"]), -0.3)
            self.assertEqual(row["gpu_loss"], "")
            self.assertEqual(row["abs_difference"], "")
            self.assertEqual(row["gpu_rank"], "")


if __name__ == "__main__":
    unittest.main()
