"""Focused release-schedule regressions; reuse the existing environment fixture."""
import copy
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'external/coralai'))
sys.path.insert(0, str(ROOT / 'tests'))

import torch
from test_dynamic_energy import substrate, TrailEnvironment, make_env

SCHEDULE = dict(cycle_updates=200, feeding_updates=100, rate_multiplier=2.0)


class FeedingCycleTests(unittest.TestCase):
    def test_default_matches_original_continuous_release(self):
        # Execute the source revision as an independent backward-compatibility oracle.
        import subprocess
        namespace = {'__name__': 'original_trail_environment'}
        source = subprocess.check_output(['git', 'show',
            '54de9f5:external/coralai/experiments/coral_dev/environments.py'], cwd=ROOT, text=True)
        exec(compile(source, 'original_environments.py', 'exec'), namespace)
        for interval in (0, 125):
            old, new = namespace['TrailEnvironment'](shift_interval=interval), TrailEnvironment(shift_interval=interval)
            a, b = substrate(), substrate()
            old.seed(a); new.seed(b)
            self.assertEqual(old.to_dict(), new.to_dict())
            for _ in range(201):
                old.step(a); new.step(b)
                self.assertTrue(torch.equal(a.mem, b.mem))
                self.assertTrue(torch.equal(old._capacity, new._capacity))
                for k, v in old.resource_state().items():
                    other = new.resource_state()[k]
                    self.assertTrue(torch.equal(v, other) if torch.is_tensor(v) else v == other)

    def test_phases_and_pause_movement_preserve_stored_channels(self):
        env = TrailEnvironment(release_schedule=SCHEDULE, shift_interval=125, shift_cells=2)
        sub = substrate(); env.seed(sub)
        unchanged = sub.mem[:, 1:].clone()
        for update in range(1, 1001):
            before, reserve = sub.mem[0, 0].clone(), env._capacity.clone()
            env.step(sub)
            feeding = any(lo <= update <= hi for lo, hi in
                          ((1, 100), (201, 300), (401, 500), (601, 700), (801, 900)))
            self.assertEqual(bool(env._last_injection.any()), feeding)
            self.assertTrue(torch.equal(sub.mem[:, 1:], unchanged))
            if not feeding:
                self.assertTrue(torch.equal(sub.mem[0, 0], before))
                self.assertTrue(torch.equal(env._capacity,
                    torch.roll(reserve, 2, dims=1) if update % 125 == 0 else reserve))

    def test_complete_cycle_budgets_and_finite_reserve_conservation(self):
        envs = [TrailEnvironment(release_schedule=s, shift_interval=i, shift_cells=2)
                for s in (None, SCHEDULE) for i in (0, 125)]
        subs = [substrate() for _ in envs]
        for env, sub in zip(envs, subs): env.seed(sub)
        self.assertTrue(all(torch.equal(subs[0].mem, sub.mem) for sub in subs))
        initial = envs[0]._initial_capacity
        for update in range(1, 1001):
            for env, sub in zip(envs, subs): env.step(sub)
            if update % 200 == 0:
                budgets = [e.resource_state() for e in envs]
                self.assertLess(max(b['cumulative_requested'] for b in budgets) -
                                min(b['cumulative_requested'] for b in budgets), 1e-6 * initial)
        for env in envs:
            b = env.resource_state()
            bound = 0.5 * torch.finfo(torch.float32).eps * 64 * 1000 * int((env._capacity > 0).sum())
            self.assertLess(abs(initial - b['remaining_capacity'] - b['cumulative_requested']), bound)
            self.assertEqual(b['cumulative_clipped'], 0)
        # Exhaustion cannot create reserves, including when receiving energy clips.
        env = TrailEnvironment(max_capacity=0.15, release_schedule=SCHEDULE)
        sub = substrate(); env.seed(sub); sub.mem[0, 0] = 100
        initial = env._initial_capacity
        for _ in range(401): env.step(sub)
        self.assertEqual(env.resource_state()['remaining_capacity'], 0)
        self.assertAlmostEqual(env._requested, initial, delta=1e-5)
        self.assertEqual(env._delivered, 0)

    def test_configuration_roundtrip_and_validation(self):
        options = {'release_schedule': copy.deepcopy(SCHEDULE)}
        env = make_env('trail', options=options)
        self.assertEqual(env.to_dict()['release_schedule'], SCHEDULE)
        options['release_schedule']['feeding_updates'] = 1
        self.assertEqual(env.release_schedule, SCHEDULE)
        for schedule in ({}, SCHEDULE | {'cycle_updates': 0},
                         SCHEDULE | {'feeding_updates': 201}, SCHEDULE | {'feeding_updates': True},
                         SCHEDULE | {'rate_multiplier': float('nan')}, SCHEDULE | {'extra': 1}):
            with self.assertRaises(ValueError): make_env('trail', options={'release_schedule': schedule})

    def test_compact_analysis_needs_no_raw_data_and_preserves_execution_record(self):
        import contextlib
        import io
        import json
        import shutil
        import tempfile
        from unittest.mock import patch
        from optional.feeding_cycles import run as comparison

        results = ROOT / 'optional/feeding_cycles/results'
        source = results / 'seed42_20261127'
        with tempfile.TemporaryDirectory(dir=results) as directory:
            output = Path(directory)
            for path in source.iterdir():
                if path.is_file(): shutil.copy2(path, output / path.name)
            before = json.loads((output / 'summary.json').read_text())
            with patch.object(sys, 'argv', ['run.py', '--output-dir', str(output), '--analyze-only']), \
                    patch.object(comparison, 'run', side_effect=AssertionError('Unexpected simulation')), \
                    patch.object(comparison, 'analyze', side_effect=AssertionError('Unexpected raw-data analysis')), \
                    patch.object(comparison, 'render', side_effect=AssertionError('Unexpected media regeneration')), \
                    contextlib.redirect_stdout(io.StringIO()):
                comparison.main()
            after = json.loads((output / 'summary.json').read_text())
            for key in ('provenance', 'checks', 'resource_budgets', 'focused_validation'):
                self.assertEqual(before[key], after[key])
            self.assertFalse((output / 'local').exists())
            self.assertEqual(after['figure']['plotted_measurements_verified'], 164)
            self.assertEqual(after['figure']['feeding_spans_half_update_edges'][0], [0.5, 100.5])
            self.assertEqual(after['figure']['verified_counts']['900'], dict(zip(comparison.NAMES, (23454, 27190, 24376, 27578))))
            self.assertEqual(after['figure']['verified_counts']['1000'], dict(zip(comparison.NAMES, (23787, 17599, 24382, 18399))))


if __name__ == '__main__':
    unittest.main()
