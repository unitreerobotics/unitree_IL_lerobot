import unittest

import numpy as np

from unitree_lerobot.eval_robot.hybrid_arm_utils import (
    chunk_timestep_range,
    compose_hybrid_state,
    extract_arm_chunk,
    limit_arm_target,
)


class HybridArmUtilsTest(unittest.TestCase):
    def test_compose_hybrid_state_replaces_only_arm(self):
        dataset = np.arange(28, dtype=np.float32)
        real_arm = np.arange(100, 114, dtype=np.float32)

        hybrid = compose_hybrid_state(dataset, real_arm)

        np.testing.assert_array_equal(hybrid[:14], real_arm)
        np.testing.assert_array_equal(hybrid[14:], dataset[14:])
        np.testing.assert_array_equal(dataset, np.arange(28, dtype=np.float32))

    def test_extract_arm_chunk(self):
        chunk = np.arange(3 * 28, dtype=np.float32).reshape(3, 28)
        np.testing.assert_array_equal(extract_arm_chunk(chunk), chunk[:, :14])

    def test_limit_arm_target_clamps_each_joint(self):
        current = np.zeros(14, dtype=np.float32)
        predicted = np.linspace(-0.2, 0.2, 14, dtype=np.float32)

        limited = limit_arm_target(predicted, current, max_delta_rad=0.05)

        self.assertLessEqual(float(np.max(np.abs(limited - current))), 0.050001)
        np.testing.assert_allclose(limited[5:9], predicted[5:9])

    def test_rejects_wrong_shapes_and_non_finite_values(self):
        with self.assertRaises(ValueError):
            compose_hybrid_state(np.zeros(27), np.zeros(14))
        with self.assertRaises(ValueError):
            extract_arm_chunk(np.zeros((100, 14)))
        with self.assertRaises(ValueError):
            limit_arm_target(np.full(14, np.nan), np.zeros(14), 0.05)

    def test_chunk_timestep_range_skips_stale_actions(self):
        self.assertEqual(list(chunk_timestep_range(40, 100, 75, 70)), list(range(70, 115)))
        self.assertEqual(list(chunk_timestep_range(0, 100, 75, 0)), list(range(75)))
        self.assertEqual(list(chunk_timestep_range(40, 100, 75, 120)), [])


if __name__ == "__main__":
    unittest.main()
