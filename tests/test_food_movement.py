"""Controls that prevent boundary and moving-measurement-region confounds."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.analyze_food_movement import same_mask_counts
from scripts.compare_food_movement import configurations, preflight
from scripts.compare_trail_reference import read_json


class FoodMovementTests(unittest.TestCase):
    def test_moving_mask_over_stationary_ecosystem_has_zero_treatment_effect(self):
        occupied = np.zeros((6, 6), dtype=bool)
        occupied[:, 0] = True
        initial = np.zeros_like(occupied)
        initial[:, 0] = True
        moved = np.zeros_like(occupied)
        moved[:, 2] = True
        result = same_mask_counts(occupied, occupied.copy(), moved, initial)
        self.assertEqual(result['moving_occupied_in_initial_source'], 6)
        self.assertEqual(result['moving_occupied_in_current_source'], 0)
        self.assertEqual(result['fixed_occupied_in_same_moving_source'], 0)
        self.assertEqual(result['moving_minus_fixed_in_same_source'], 0)

    def test_same_mask_control_detects_growth_separately_from_mask_motion(self):
        fixed = np.zeros((6, 6), dtype=bool)
        fixed[:, 0] = True
        moving = fixed.copy()
        moving[:3, 2] = True
        source = np.zeros_like(fixed)
        source[:, 2] = True
        result = same_mask_counts(fixed, moving, source, fixed)
        self.assertEqual(result['moving_minus_fixed_in_same_source'], 3)
        self.assertEqual(result['moving_occupied_in_initial_source'], 6)
        self.assertEqual(result['current_source_cells'], 6)

    def test_selected_schedules_preserve_domain_without_physics_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            preflight(output, configurations())
            design = read_json(output / 'design.json')
        self.assertEqual(design['initial_y_bounds_inclusive'], [18, 382])
        self.assertEqual(design['schedules']['slower']['final_y_bounds_inclusive'], [26, 390])
        self.assertEqual(design['schedules']['faster']['final_y_bounds_inclusive'], [34, 398])
        self.assertEqual(design['resource_wraps'], 0)

    def test_unsupported_faster_schedule_is_rejected_before_rollout(self):
        configs = configurations()
        configs['faster']['env_options']['shift_interval'] = 100
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'cross the boundary'):
                preflight(Path(temp), configs)


if __name__ == '__main__':
    unittest.main()
