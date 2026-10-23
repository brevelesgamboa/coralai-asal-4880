"""Check config/seed overrides, checkpoint counts, and existing-output protection."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "external/coralai/experiments/coral_dev/headless_coral.py"
HAS_RUNTIME = all(importlib.util.find_spec(name) for name in (
    "torch", "taichi", "neat", "pytorch_neat", "coralai", "PIL",
))


@unittest.skipUnless(HAS_RUNTIME, "Install Coralai and its vendored PyTorch-NEAT package first")
class HeadlessRunnerTests(unittest.TestCase):
    def test_default_config_is_found_before_rejecting_an_invalid_seed(self):
        result = subprocess.run([sys.executable, str(RUNNER), "--seed", "-1"],
                                capture_output=True, text=True, timeout=120)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("seed must be an integer", result.stderr)
        self.assertNotIn("config file not found", result.stderr)

    def test_saved_config_seed_final_checkpoint_and_existing_output_protection(self):
        import torch
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            config = json.loads((ROOT / "configs/headless_cpu_repro.json").read_text())
            config.update(shape=32, steps=3, checkpoint_interval=2, screenshot_interval=2)
            config["output_dir"] = str(folder / "configured_output")
            config_path = folder / "config.json"
            config_path.write_text(json.dumps(config))
            output = folder / "override_output"
            command = [sys.executable, str(RUNNER), "--config", str(config_path),
                       "--seed", "7", "--output-dir", str(output)]
            result = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse((folder / "configured_output").exists())
            effective = json.loads((output / "config.json").read_text())
            self.assertEqual(effective["seed"], 7)
            metadata = json.loads((output / "meta.json").read_text())
            self.assertEqual(metadata["effective_config"], effective)
            self.assertEqual(metadata["completed_steps"], 3)
            self.assertEqual(metadata["final_checkpoint"], "checkpoint_0000003")
            for step in (2, 3):
                checkpoint = output / f"checkpoint_{step:07d}"
                self.assertEqual(json.loads((checkpoint / "meta.json").read_text())["step"], step)
                state = torch.load(checkpoint / "substrate.pt", weights_only=True)
                self.assertEqual(tuple(state.shape), (1, 14, 32, 32))
                self.assertTrue(torch.isfinite(state).all())
                self.assertTrue((checkpoint / "population.pkl").is_file())
            with Image.open(output / "screenshots/step_0000003.png") as image:
                self.assertEqual(image.size, (32, 32))
            self.assertTrue((output / "history").is_dir())
            self.assertFalse((folder / "history").exists())
            before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
            repeated = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=120)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("FileExistsError", repeated.stderr)
            after = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
            self.assertEqual(after, before)

    def test_disabled_periodic_checkpoints_still_save_final_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            config = json.loads((ROOT / "configs/headless_cpu_repro.json").read_text())
            config.update(shape=32, steps=1, checkpoint_interval=0, output_dir=str(folder / "run"))
            path = folder / "config.json"
            path.write_text(json.dumps(config))
            result = subprocess.run([sys.executable, str(RUNNER), "--config", str(path)],
                                    cwd=folder, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            checkpoints = list((folder / "run").glob("checkpoint_*"))
            self.assertEqual([p.name for p in checkpoints], ["checkpoint_0000001"])


if __name__ == "__main__":
    unittest.main()
