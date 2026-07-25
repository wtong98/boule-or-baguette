import unittest

from token_budget import choose_rt_schedule


class ChooseRtScheduleTest(unittest.TestCase):
    def test_preserves_original_accumulation_when_one_step_fits(self):
        schedule = choose_rt_schedule(
            dp_counts=[4] * 2_048,
            rt_counts=[1_004] * 2_048,
            dp_steps=500,
            batch_size=4,
            dp_accum_steps=8,
        )

        self.assertEqual(schedule.rt_steps, 2)
        self.assertEqual(schedule.rt_accum_steps, 8)
        self.assertAlmostEqual(schedule.achieved_ratio, 1.004, places=3)

    def test_reduces_accumulation_when_one_step_overshoots(self):
        schedule = choose_rt_schedule(
            dp_counts=[4] * 2_048,
            rt_counts=[18_576] * 2_048,
            dp_steps=500,
            batch_size=1,
            dp_accum_steps=32,
        )

        self.assertEqual(schedule.rt_steps, 1)
        self.assertEqual(schedule.rt_accum_steps, 3)

    def test_uses_exact_prefix_counts_for_candidate_examples(self):
        schedule = choose_rt_schedule(
            dp_counts=[4] * 100,
            rt_counts=[100, 100, 100, 900] + [100] * 96,
            dp_steps=1,
            batch_size=1,
            dp_accum_steps=4,
        )

        self.assertEqual(schedule.estimated_rt_tokens, 100)


if __name__ == "__main__":
    unittest.main()
