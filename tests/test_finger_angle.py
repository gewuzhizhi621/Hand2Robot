"""Basic tests for finger angle calculation."""

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hand2robot.finger_angle import angle_between_three_points, compute_finger_angles


def test_angle_between_three_points():
    angle = angle_between_three_points([1, 0, 0], [0, 0, 0], [0, 1, 0])
    assert 89.0 <= angle <= 91.0


def test_compute_finger_angles_shape_and_range():
    landmarks = np.zeros((21, 3), dtype=float)
    for index in range(21):
        landmarks[index] = [index * 0.01, index * 0.02, 0.0]

    result = compute_finger_angles(landmarks)
    assert set(result) == {"thumb", "index", "middle", "ring", "pinky"}
    for value in result.values():
        assert 0.0 <= value <= 1.0


if __name__ == "__main__":
    test_angle_between_three_points()
    test_compute_finger_angles_shape_and_range()
    print("test_finger_angle passed")
