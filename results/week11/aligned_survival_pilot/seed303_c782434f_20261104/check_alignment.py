"""Focused recording checks; no simulations or model inference."""

import ast
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from pilot import RUNNER, measure_pair, renderer, validate_keys
import pilot


SOURCE_TREE = Path("/tmp/coralai-aligned-pilot-source")


class AlignmentChecks(unittest.TestCase):
    def test_completed_clock_precedes_both_recording_operations(self):
        tree = ast.parse((SOURCE_TREE / RUNNER).read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        loop = next(node for node in ast.walk(main) if isinstance(node, ast.For) and ast.unparse(node.target) == "step")
        code = [ast.unparse(node) for node in loop.body]
        clock = code.index("evolver.timestep = step + 1")
        capture = next(i for i, text in enumerate(code) if "capture(evolver.timestep)" in text)
        checkpoint = next(i for i, text in enumerate(code) if "evolver.save_checkpoint(run_dir, evolver.timestep)" in text)
        self.assertLess(clock, capture)
        self.assertLess(capture, checkpoint)
        # Only read-only metric logging occurs between capture and checkpoint.
        self.assertEqual(checkpoint - capture, 2)
        self.assertIn("log.writerow", code[capture + 1])
        self.assertNotIn("substrate.mem[", code[capture + 1].split("log.writerow")[1])

    def test_missing_duplicate_and_wrong_update_are_rejected(self):
        rows = [{"configuration": c, "seed": 303, "completed_update": u}
                for c in ("default", "selected") for u in range(50, 1001, 50)]
        validate_keys(rows)
        for bad in (rows[:-1], rows[:-1] + [rows[0]], rows[:-1] + [dict(rows[-1], completed_update=1001)]):
            with self.subTest(length=len(bad)), self.assertRaisesRegex(ValueError, "identities"):
                validate_keys(bad)

    def test_verification_rejects_incomplete_recording_manifest(self):
        output = Path(__file__).resolve().parent
        original_read = pilot.read_json

        def read_with_missing_row(path):
            data = original_read(path)
            if Path(path).name == "recording_manifest.json":
                data["frames"] = data["frames"][:-1]
            return data

        # The real evidence remains untouched. Before the coverage guard, zip
        # silently skipped the last scored row and reported all 40 as checked.
        with patch.object(pilot, "read_json", side_effect=read_with_missing_row), patch.object(pilot, "save_json"):
            with self.assertRaisesRegex(ValueError, "Missing/duplicate/wrong update identities"):
                pilot.verify(SimpleNamespace(output_dir=output, source_tree=SOURCE_TREE))

    def test_occupancy_ignores_resource_brightness_and_preserves_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path, image_path = Path(directory) / "state.pt", Path(directory) / "image.png"
            state = torch.ones((1, 14, 100, 100))
            state[0, 13] = -1
            state[0, 13, 7, 29] = 0
            torch.save(state, state_path)
            render = renderer(SOURCE_TREE)
            from types import SimpleNamespace
            pixels = render(SimpleNamespace(mem=state, w=100, h=100, windex={"energy": 0, "infra": 1, "genome": 13}))
            self.assertEqual(int(pixels[29, 7, 2]), 255)
            Image.fromarray(pixels).save(image_path)
            self.assertEqual(measure_pair(state_path, image_path, {"n_genomes": 1}, render), (1, 0.0001))

    def test_wrong_rgb_checkpoint_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path, image_path = Path(directory) / "state.pt", Path(directory) / "image.png"
            torch.save(torch.ones((1, 14, 100, 100)), state_path)
            Image.new("RGB", (100, 100), "white").save(image_path)
            with self.assertRaisesRegex(ValueError, "Image/state mismatch"):
                measure_pair(state_path, image_path, {"n_genomes": 2}, renderer(SOURCE_TREE))


if __name__ == "__main__":
    unittest.main(verbosity=2)
