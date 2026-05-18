"""Process one image with Hand2Robot."""

from __future__ import annotations

import argparse
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
    save_image,
    save_landmarks_json,
    save_qpos_json,
    timestamp_string,
)
from hand2robot.finger_angle import compute_finger_angles
from hand2robot.qpos_mapper import map_finger_angles_to_qpos
from hand2robot.robot_hand_visualizer import plot_robot_hand
from hand2robot.visualizer import draw_finger_angles, draw_hand_landmarks, draw_qpos, draw_status_text


def parse_args():
    parser = argparse.ArgumentParser(description="Run Hand2Robot on one image.")
    parser.add_argument("--image", required=True, type=Path, help="Input image path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "examples" / "outputs",
        help="Directory for generated files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image
    if not image_path.exists():
        print(f"Image file not found: {image_path}")
        return

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    output_dir = ensure_dir(args.output_dir)
    detector = HandDetector()
    try:
        detection = detector.detect(image)
    finally:
        detector.close()

    if not detection["has_hand"]:
        print("No hand detected in the image.")
        return

    finger_angles = compute_finger_angles(detection["landmarks"])
    qpos = map_finger_angles_to_qpos(finger_angles)

    annotated = image.copy()
    draw_hand_landmarks(annotated, detection)
    draw_status_text(annotated, ["Image mode"])
    draw_finger_angles(annotated, finger_angles)
    draw_qpos(annotated, qpos)

    stamp = timestamp_string()
    save_image(output_dir, annotated, f"annotated_{stamp}.png")
    save_landmarks_json(output_dir, detection, f"landmarks_{stamp}.json")
    save_finger_angles_json(output_dir, finger_angles, f"finger_angles_{stamp}.json")
    save_qpos_json(output_dir, qpos, f"qpos_{stamp}.json")
    plot_robot_hand(qpos, save_path=output_dir / f"robot_hand_{stamp}.png", show=False)


if __name__ == "__main__":
    main()
