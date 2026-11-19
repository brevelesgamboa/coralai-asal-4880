"""Coordinate, normalization, and timeline guards for the shared saved-state renderer."""
from collections import Counter
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.compare_trail_reference import gui_values, rgb_image, read_json
from scripts.compare_food_movement import configurations as speed_configs
from scripts.compare_food_direction import configurations as direction_configs
from scripts.render_food_gui_style import (ROOT, CLOSEUP, ORANGE, state_panel, outline_pixels,
    cue, timeline_for, export)


class FoodPresentationTests(unittest.TestCase):
    def test_full_world_normalization_precedes_crop_and_xy_transpose(self):
        state = torch.zeros(1, 14, 400, 400)
        state[0, 13] = -1
        state[0, 1] = 1
        state[0, 0, 375, 150] = 2
        state[0, 0, 10, 10] = 4  # Maximum outside the crop must still set its scale.
        state[0, 12, 375, 150] = 3
        state[0, 12, 10, 10] = 7
        rgb = np.asarray(rgb_image(gui_values(state)))
        source = np.zeros((400, 400), dtype=bool)
        source[150, 375] = True
        image, outline = state_panel(rgb, source, 'closeup')
        pixels = np.asarray(image)
        y, x = (150-100)*8+4, (375-300)*8+4
        self.assertFalse(outline[y, x])
        np.testing.assert_array_equal(pixels[y, x], [127, 255, 109])
        self.assertEqual(image.size, (800, 800))
        self.assertEqual(int(rgb[0, 0, 1]), 255)  # Colored background can be unoccupied.
        np.testing.assert_array_equal(pixels[outline], np.tile(ORANGE, (outline.sum(), 1)))

    def test_crop_does_not_invent_source_boundaries(self):
        source = np.ones((400, 400), dtype=bool)
        outline = outline_pixels(source, CLOSEUP, 8, 2)
        self.assertFalse(outline[:, 0].any())
        self.assertFalse(outline[0, :].any())
        source[:] = False
        source[120:130, 340:350] = True
        outline = outline_pixels(source, CLOSEUP, 8, 2)
        self.assertTrue(outline[20*8, 45*8])
        self.assertTrue(outline[25*8, 40*8])
        self.assertFalse(outline[10, 10])

    def test_cues_and_holds_keep_turn_enabling_distinct_from_first_reversed_shift(self):
        speed, direction = Counter(timeline_for('speeds')), Counter(timeline_for('direction'))
        self.assertEqual(sorted(speed), list(range(0, 1001, 25)))
        self.assertEqual(sorted(direction), sorted(speed))
        self.assertEqual(sum(speed.values()), 159)
        self.assertEqual(sum(direction.values()), 174)
        self.assertEqual([direction[u] for u in (0, 500, 625, 1000)], [12, 12, 9, 30])
        self.assertIn('next source shift', cue(500))
        self.assertEqual(cue(625), 'First upward source shift')

    def test_maintained_configs_equal_recorded_effective_configs(self):
        for group, configs in (('food_movement_comparison', speed_configs()),
                               ('food_direction_change', direction_configs())):
            for name, config in configs.items():
                self.assertEqual(config, read_json(ROOT/'results/week13'/group/'seed42_20261118/configs'/f'{name}.json'))

    def test_existing_exports_are_refused_before_loading_any_raw_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                export('speeds', None, False, results_dir=path/'no-input', output_dir=path, view='occupancy')


if __name__ == '__main__':
    unittest.main()
