"""Controller assignment, neighborhood layout, and registry lifecycle regressions."""

from pathlib import Path
import pickle
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import taichi as ti
import torch

from scripts.verify_controller_lookup import ROOT, fixture, assigned_outputs, check, layout_check

sys.path.insert(0, str(ROOT / "external/coralai/experiments/coral_dev"))
from coralai import evolver as module
from experiment import EXPERIMENT
from evolution import apply_radiation_mutation


class ControllerLookupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, cpu_max_num_threads=1, debug=True)

    def test_known_outputs_sparse_active_ids_zero_and_inactive_sentinels(self):
        self.assertTrue(check(module)["passed"])
        self.assertTrue(layout_check(module)["passed"])

    def test_missing_and_noninteger_active_ids_fail_clearly(self):
        ev = fixture(module)
        for value in (8, 99, 1.5, float("nan")):
            ev.substrate.mem[0, 13, 0, 0] = value
            with self.assertRaisesRegex(ValueError, "genome IDs"):
                assigned_outputs(module, ev)
        ev.substrate.mem[0, 13, 0, 0] = 2
        ev._get_key_to_local()[2] = -1
        with self.assertRaisesRegex(ValueError, "missing controller rows"):
            assigned_outputs(module, ev)

    def test_weighted_neighborhood_matches_independent_reference_for_all_rotations(self):
        ev = fixture(module)
        ev.sense_chinds = [0, 1]
        ev.dir_kernel = torch.tensor([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])
        ev.dir_order = torch.tensor([0, -1, 1])
        coefficients = torch.tensor([1., 2., 3., 4., 11., 12., 13., 14.]) / 10
        ev.combined_weights = [(coefficients * (key + 1)).reshape(1, 1, 8) for key in range(8)]
        ev._invalidate_weight_cache()
        s = ev.substrate.mem[0]
        s[0] = torch.arange(64).reshape(8, 8) / 64
        s[1] = 1 + torch.arange(64).reshape(8, 8) / 32
        s[12] = torch.arange(64).reshape(8, 8) % 8
        s[13] = torch.tensor([0, 2, 7, -1, 0, 2, 7, -2]).repeat(8, 1)
        expected = torch.zeros(8, 8)
        for x in range(8):
            for y in range(8):
                key, rot = int(s[13, x, y]), int(s[12, x, y])
                if key < 0: continue
                values = []
                for channel in (0, 1):
                    values.append(s[channel, x, y])
                    for offset in (0, -1, 1):
                        dx, dy = ev.dir_kernel[(rot + offset) % 8]
                        values.append(s[channel, (x + int(dx)) % 8, (y + int(dy)) % 8])
                expected[x, y] = torch.dot(torch.stack(values), coefficients * (key + 1)) + 10 + key
        torch.testing.assert_close(assigned_outputs(module, ev)[0], expected)

    def test_cleanup_registration_and_caches_preserve_controller_identity(self):
        ev = fixture(module)
        for x, key in enumerate((0, 2, 7)): ev.substrate.mem[0, 13, x, 0] = key
        old = ev._get_key_to_local()
        self.assertEqual(ev.remove_extinct_genomes(), 5)
        self.assertEqual(ev.genome_keys, [0, 1, 2])
        self.assertEqual([float(assigned_outputs(module, ev)[0, x, 0]) for x in range(3)], [10, 12, 17])
        net = SimpleNamespace(weights=torch.zeros(1, 1, 2), biases=torch.tensor([[[42.]]]))
        self.assertEqual(ev.add_organism_get_key(net), 3)
        self.assertEqual(ev._get_key_to_local().tolist(), [0, 1, 2, 3])
        ev.substrate.mem[0, 13, 3, 0] = 3
        self.assertEqual(float(assigned_outputs(module, ev)[0, 3, 0]), 42)
        self.assertIsNot(ev._get_key_to_local(), old)
        self.assertEqual(ev.local_idx_for_key(3), 3)
        self.assertEqual(ev.local_idx_for_key(7), -1)

    def test_population_cull_remaps_cells_and_controller_rows_together(self):
        ev = fixture(module)
        for x, key in enumerate((0, 2, 2, 7, 7, 7)): ev.substrate.mem[0, 13, x, 0] = key
        ev.reduce_population_to_threshold(2)
        self.assertEqual(ev.genome_keys, [0, 1])
        self.assertEqual([float(assigned_outputs(module, ev)[0, x, 0]) for x in range(6)], [0, 12, 12, 17, 17, 17])

    def test_checkpoint_restore_new_and_legacy_registry_formats(self):
        ev = fixture(module)
        for x, key in enumerate((0, 2, 7)): ev.substrate.mem[0, 13, x, 0] = key
        with tempfile.TemporaryDirectory() as folder:
            ev.save_checkpoint(folder, 0)
            checkpoint = Path(folder) / "checkpoint_0000000"
            for legacy in (False, True):
                if legacy:
                    path = checkpoint / "population.pkl"
                    with path.open("rb") as f: data = pickle.load(f)
                    data.pop("genome_keys"); data.pop("genome_id_semantics")
                    with path.open("wb") as f: pickle.dump(data, f)
                restored = fixture(module, n=2)
                restored._get_key_to_local()
                restored.load_checkpoint(checkpoint)
                self.assertEqual(restored.genome_keys, list(range(8)))
                self.assertEqual([float(assigned_outputs(module, restored)[0, x, 0]) for x in range(3)], [10, 12, 17])

    def test_normal_initialization_and_radiation_registration(self):
        import os
        with tempfile.TemporaryDirectory() as folder:
            previous = Path.cwd()
            try:
                os.chdir(folder)
                torch.manual_seed(42); random.seed(42); np.random.seed(42)
                sub = EXPERIMENT.make_substrate((16, 16), torch.device("cpu"))
                ev = EXPERIMENT.make_evolver(sub)
                self.assertEqual(ev.genome_keys, list(range(50)))
                for mutation_branch in (0.0, 1.0):
                    sub.mem[0, 13] = 0
                    count = len(ev.genomes)
                    with patch("evolution.random.random", return_value=mutation_branch):
                        apply_radiation_mutation(ev, n_spots=1)
                    self.assertEqual(len(ev.genomes), count + 1)
                    self.assertEqual(ev._get_key_to_local().tolist(), list(range(count + 1)))
            finally: os.chdir(previous)


if __name__ == "__main__": unittest.main()
