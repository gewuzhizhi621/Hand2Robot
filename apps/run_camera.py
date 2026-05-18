"""Real-time camera demo for Hand2Robot."""

from __future__ import annotations

from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hand2robot.detector import HandDetector
from hand2robot.export_utils import (
    ensure_dir,
    save_finger_angles_json,
    save_landmarks_json,
    save_qpos_json,
    timestamp_string,
)
from hand2robot.finger_angle import compute_finger_angles
from hand2robot.qpos_mapper import map_finger_angles_to_qpos
from hand2robot.robot_hand_visualizer import plot_robot_hand
from hand2robot.smoothing import ExponentialSmoother
from hand2robot.visualizer import (
    draw_finger_angles,
    draw_hand_landmarks,
    draw_qpos,
    draw_status_text,
)


OUTPUT_DIR = PROJECT_ROOT / "examples" / "outputs"
WINDOW_TITLE = "Hand2Robot - Camera Retargeting Demo"


def main() -> None:
    output_dir = ensure_dir(OUTPUT_DIR)
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Unable to open camera. Try changing cv2.VideoCapture(0) to cv2.VideoCapture(1).")
        return

    detector = HandDetector()
    smoother = ExponentialSmoother(alpha=0.6)
    latest_detection = None
    latest_angles = None
    latest_qpos = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Failed to read a frame from the camera.")
                break

            frame = cv2.flip(frame, 1)
            detection = detector.detect(frame)
            latest_detection = detection

            if detection["has_hand"]:
                angles = compute_finger_angles(detection["landmarks"])
                angles = smoother.smooth_dict(angles)
                qpos = map_finger_angles_to_qpos(angles)
                latest_angles = angles
                latest_qpos = qpos
                draw_hand_landmarks(frame, detection)
                draw_finger_angles(frame, angles)
                draw_qpos(frame, qpos)
            else:
                latest_angles = None
                latest_qpos = None

            draw_status_text(
                frame,
                ["S: Save landmarks", "Q: Save qpos", "P: Save robot hand image"],
            )
            cv2.imshow(WINDOW_TITLE, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("s"), ord("S")):
                if latest_detection and latest_detection["has_hand"]:
                    save_landmarks_json(output_dir, latest_detection)
                else:
                    print("No hand detected.")
            if key in (ord("q"), ord("Q")):
                if latest_detection and latest_detection["has_hand"] and latest_qpos and latest_angles:
                    save_qpos_json(output_dir, latest_qpos)
                    save_finger_angles_json(output_dir, latest_angles)
                else:
                    print("No hand detected.")
            if key in (ord("p"), ord("P")):
                if latest_detection and latest_detection["has_hand"] and latest_qpos:
                    save_path = output_dir / f"robot_hand_{timestamp_string()}.png"
                    plot_robot_hand(latest_qpos, save_path=save_path, show=False)
                else:
                    print("No hand detected.")
    finally:
        capture.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
