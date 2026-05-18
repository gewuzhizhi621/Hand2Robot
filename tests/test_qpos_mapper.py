"""Basic tests for qpos mapping."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hand2robot.qpos_mapper import flatten_qpos, map_finger_angles_to_qpos


def test_map_finger_angles_to_qpos():
    finger_angles = {
        "thumb": 0.3,
        "index": 0.8,
        "middle": 0.7,
        "ring": 0.6,
        "pinky": 0.5,
    }
    qpos = map_finger_angles_to_qpos(finger_angles)

    assert set(qpos) == {"thumb", "index", "middle", "ring", "pinky"}
    for joints in qpos.values():
        assert len(joints) == 3
        for angle in joints:
            assert 0.0 <= angle <= 1.57

    assert len(flatten_qpos(qpos)) == 15


if __name__ == "__main__":
    test_map_finger_angles_to_qpos()
    print("test_qpos_mapper passed")
