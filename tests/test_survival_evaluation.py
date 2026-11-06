"""Guard historical update alignment, occupancy semantics, and missing-data claims."""

from pathlib import Path
import tempfile
import unittest

import torch
from PIL import Image

from scripts.evaluate_survival import (
    ROOT, join_trajectory, load_population_log, measure_state, zero_status,
)


class SurvivalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def write_log(self, rows):
        path = self.directory / "step_log.csv"
        path.write_text("step,n_alive,n_genomes\n" + "\n".join(rows) + "\n")
        return path

    def test_log_labels_follow_completed_updates_not_screenshot_labels(self):
        path = self.write_log(["0,2,10", "50,3,11"])
        populations = load_population_log(path, 4, 100)
        self.assertEqual(list(populations), [1, 51])
        self.assertNotIn(0, populations)
        self.assertNotIn(50, populations)
        self.assertEqual(populations[51]["logged_step"], 50)
        self.assertEqual(populations[51]["living_cell_fraction"], 0.75)
        self.assertEqual(populations[51]["retained_genome_count"], 11)
        self.assertIsNone(populations[51]["occupied_genome_ids"])

    def test_invalid_or_duplicate_population_log_rejected(self):
        cases = (["0,5,10"], ["0,-1,10"], ["0,2.5,10"], ["0,2,10", "0,2,10"],
                 ["100,2,10"], ["-1,2,10"], ["10,2,10", "0,2,10"])
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                load_population_log(self.write_log(rows), 4, 100)

    def test_occupancy_uses_genome_sentinel_not_energy_or_threshold(self):
        state = torch.zeros((1, 14, 2, 2))
        state[0, 13] = torch.tensor([[0, -1], [-1, 0]])
        state[0, 0, 0, 1] = 100  # Empty, although resource-rich.
        path = self.directory / "substrate.pt"
        torch.save(state, path)
        measured, diagnostic = measure_state(path, [2, 2], retained_genomes=1)
        self.assertEqual(measured["living_cell_count"], 2)
        self.assertEqual(measured["living_cell_fraction"], 0.5)
        self.assertEqual(measured["occupied_genome_ids"], 1)  # Cells != genomes.
        self.assertEqual(diagnostic["unoccupied_cells_with_energy_plus_infra_above_death_threshold"], 1)
        self.assertEqual(diagnostic["occupied_cells_at_or_below_death_threshold"], 2)

    def test_invalid_grid_or_genome_reference_rejected(self):
        state = torch.zeros((1, 14, 2, 2))
        path = self.directory / "substrate.pt"
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, "State/grid"):
            measure_state(path, [4, 4])
        with self.assertRaisesRegex(ValueError, "Invalid genome reference"):
            measure_state(path, [2, 2], retained_genomes=0)
        state[0, 13, 0, 0] = 0.5
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, "integers"):
            measure_state(path, [2, 2])

    def test_state_must_reproduce_its_scored_screenshot_pixels(self):
        state = torch.ones((1, 14, 2, 2))
        path, image = self.directory / "substrate.pt", self.directory / "frame.png"
        torch.save(state, path)
        # Constant channels normalize to zero in the historical renderer.
        Image.new("RGB", (2, 2), "black").save(image)
        _, diagnostic = measure_state(path, [2, 2], image_path=image)
        self.assertTrue(diagnostic["matches_saved_screenshot_pixels"])
        Image.new("RGB", (2, 2), "white").save(image)
        with self.assertRaisesRegex(ValueError, "pixel mismatch"):
            measure_state(path, [2, 2], image_path=image)

    def test_outer_join_preserves_missing_measurements_and_real_zero(self):
        populations = load_population_log(self.write_log(["50,0,10"]), 4, 100)
        score = {"clip_alignment": 0.2, "clip_loss": -0.2, "siglip_logit": -8,
                 "image": "step_0000050.png", "image_sha256": "hash"}
        rows = join_trajectory("seed_101/default", 101, "default", [2, 2], populations, {50: score, 100: score})
        self.assertEqual([r["update"] for r in rows], [50, 51, 100])
        self.assertEqual([r["match_status"] for r in rows], ["score_only", "population_only", "score_only"])
        self.assertIsNone(rows[0]["living_cell_count"])
        self.assertEqual(rows[1]["living_cell_count"], 0)
        self.assertIsNone(rows[1]["clip_alignment"])

    def test_exact_join_requires_same_update(self):
        populations = load_population_log(self.write_log(["49,2,10"]), 4, 100)
        score = dict.fromkeys(("clip_alignment", "clip_loss", "siglip_logit", "image", "image_sha256"), 1)
        rows = join_trajectory("seed_101/default", 101, "default", [2, 2], populations, {50: score})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["match_status"], "exact_match")

    def test_zero_recovery_is_distinct_from_zero_through_last_observation(self):
        def observations(values):
            return {u: {"living_cell_count": n} for u, n in zip((0, 10, 20, 30), values)}
        self.assertEqual(zero_status(observations([1, 0, 0, 2])), (10, "temporary_zero_observed"))
        self.assertEqual(zero_status(observations([1, 0, 0, 0])), (10, "zero_through_last_observation"))
        self.assertEqual(zero_status(observations([1, 1, 1, 1])), (None, "not_observed"))

    def test_saved_runs_have_only_endpoint_population_score_overlap(self):
        root = ROOT / "results/week9/supervised_coralai_20261022/seed_comparison"
        for seed in (101, 202, 303):
            for configuration in ("default", "selected"):
                run = root / f"seed_{seed}" / configuration / "run"
                populations = load_population_log(run / "step_log.csv", 10000, 1000)
                self.assertEqual(list(populations), list(range(1, 1000, 10)))
                initial, _ = measure_state(run / "initial_state.pt", [100, 100])
                final, _ = measure_state(run / "checkpoint_0001000/substrate.pt", [100, 100])
                populations.update({0: initial, 1000: final})
                self.assertEqual(set(populations) & set(range(50, 1001, 50)), {1000})
                self.assertNotEqual(populations[0]["living_cell_count"], populations[1]["living_cell_count"])
                self.assertEqual(len(populations), 102)


if __name__ == "__main__":
    unittest.main()
