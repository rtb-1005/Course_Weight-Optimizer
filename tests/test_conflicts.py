from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Course_Weight-Optimizer"))

import main  # noqa: E402
from utils import load_global_state  # noqa: E402


class ConflictInputTests(unittest.TestCase):
    def test_load_global_state_reads_and_normalizes_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "global_state.json"
            path.write_text(
                json.dumps(
                    {
                        "grade_size": 100,
                        "courses": [
                            {"course_id": "A", "capacity": 10, "bidders": 12},
                            {"course_id": "B", "capacity": 10, "bidders": 12},
                            {"course_id": "C", "capacity": 10, "bidders": 12},
                        ],
                        "conflicts": [["B", "A"], ["A", "B"], ["B", "C"]],
                    }
                ),
                encoding="utf-8",
            )

            state = load_global_state(str(path))

        self.assertEqual(state.conflicts, [("A", "B"), ("B", "C")])


class ConflictAllocationTests(unittest.TestCase):
    def test_iter_conflict_free_subsets_never_contains_a_conflict_pair(self) -> None:
        subsets = list(main.iter_conflict_free_subsets(["A", "B", "C"], [("A", "B")], 3))

        self.assertIn(("A",), subsets)
        self.assertIn(("B", "C"), subsets)
        self.assertNotIn(("A", "B"), subsets)

    def test_select_allocation_drops_one_of_two_conflicting_courses(self) -> None:
        utilities = {"A": 10.0, "B": 9.0, "C": 1.0}
        alphas = {"A": 10.0, "B": 10.0, "C": 10.0}

        result = main.select_allocation(
            desired_ids=["A", "B", "C"],
            utilities=utilities,
            robust_safe=[],
            competitive=["A", "B", "C"],
            alphas_design=alphas,
            conflicts=[("A", "B")],
        )

        selected = set(result.selected_ids)
        self.assertFalse({"A", "B"}.issubset(selected))
        self.assertTrue(selected & {"A", "B"})
        self.assertTrue(all(bid >= 0.0 for bid in result.final_bids.values()))


if __name__ == "__main__":
    unittest.main()
