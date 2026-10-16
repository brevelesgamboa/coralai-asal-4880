"""Check aggregation across seeds and agreement with the endpoint reference."""

import csv
import math
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_entropy_over_time import (
    CONDITIONS, COUNTS, LABELS, METRICS, summarize, verify_endpoints,
)


def observations():
    rows = []
    for condition in CONDITIONS:
        for seed in range(42, 47):
            for label in LABELS:
                row = {"condition": condition, "seed": seed, "checkpoint_label": label,
                       "bin_width": 1.0, "alive_cells": 10, "occupied_genome_ids": 2}
                row.update({metric: seed - 42 for metric in METRICS})
                rows.append(row)
    return rows


class EntropyOverTimeTests(unittest.TestCase):
    def test_summary_uses_five_seeds_and_sample_standard_deviation(self):
        result = summarize(observations())
        self.assertEqual(len(result), 30)
        for row in result:
            self.assertEqual(row["n_runs"], 5)
            self.assertEqual(row["mean"], 2.0)
            self.assertAlmostEqual(row["sample_sd"], math.sqrt(2.5))

    def test_missing_or_duplicate_observations_are_rejected(self):
        original = observations()
        cases = [original[:-1], original[:-1] + [original[0]],
                 [row for row in original if row["checkpoint_label"] != 250]]
        for rows in cases:
            with self.subTest(count=len(rows)):
                with self.assertRaisesRegex(ValueError, "one observation"):
                    summarize(rows)

    def test_summary_rejects_nonfinite_metrics(self):
        rows = observations()
        rows[0][next(iter(METRICS))] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            summarize(rows)

    def test_endpoint_values_are_checked_against_reference(self):
        rows = observations()
        final = [row for row in rows if row["checkpoint_label"] == 1000]
        with tempfile.TemporaryDirectory() as folder:
            reference = Path(folder) / "reference.csv"
            with reference.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(final[0]))
                writer.writeheader()
                writer.writerows(final)
            verify_endpoints(rows, reference)
            for metric in (*COUNTS, *METRICS):
                changed = [dict(row) for row in rows]
                target = next(row for row in changed if row["checkpoint_label"] == 1000)
                target[metric] += 0.01
                with self.subTest(metric=metric):
                    with self.assertRaisesRegex(ValueError, "Endpoint differs"):
                        verify_endpoints(changed, reference)


if __name__ == "__main__":
    unittest.main()
