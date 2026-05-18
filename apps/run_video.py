"""Process a video with Hand2Robot and export annotated results."""

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
    save_finger_angles_csv,
    save_qpos_sequence_json,
    timestamp_string,
)
from hand2robot.finger_angle import compute_finger_angles
from hand2robot.qpos_mapper import map_finger_angles_to_qpos
from hand2robot.smoothing import ExponentialSmoother
from hand2robot.visualizer import draw_finger_angles, draw_hand_landmarks, draw_qpos, draw_status_text


def parse_args():
    parser = argparse.ArgumentParser(description="Run Hand2Robot on a video.")
    parser.add_argument("--video", required=True, type=Path, help="Input video path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "examples" / "outputs",
        help="Directory for generated files.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        print(f"Video file not found: {args.video}")
        return

    output_dir = ensure_dir(args.output_dir)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        print(f"Failed to open video: {args.video}")
        return

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stamp = timestamp_string()
    video_path = output_dir / f"result_video_{stamp}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    detector = HandDetector()
    smoother = ExponentialSmoother(alpha=0.6)
    qpos_sequence = []
    finger_angle_rows = []
    frame_id = 0

    try:
        while True:
            if args.max_frames is not None and frame_id >= args.max_frames:
                break

            ok, frame = capture.read()
            if not ok:
                break

            print(f"Processing frame {frame_id + 1}/{total_frames or '?'}")
            detection = detector.detect(frame)
            row = {"frame_id": frame_id, "has_hand": detection["has_hand"]}

            if detection["has_hand"]:
                angles = compute_finger_angles(detection["landmarks"])
                angles = smoother.smooth_dict(angles)
                qpos = map_finger_angles_to_qpos(angles)
                row.update(angles)
                qpos_sequence.append({"frame_id": frame_id, "has_hand": True, "qpos": qpos})
                draw_hand_landmarks(frame, detection)
                draw_finger_angles(frame, angles)
                draw_qpos(frame, qpos)
            else:
                qpos_sequence.append({"frame_id": frame_id, "has_hand": False, "qpos": None})

            finger_angle_rows.append(row)
            draw_status_text(frame, ["Video mode"])
            writer.write(frame)
            frame_id += 1
    finally:
        capture.release()
        writer.release()
        detector.close()

    print(f"Saved: {video_path}")
    save_qpos_sequence_json(output_dir, qpos_sequence, f"qpos_sequence_{stamp}.json")
    save_finger_angles_csv(output_dir, finger_angle_rows, f"finger_angles_{stamp}.csv")


if __name__ == "__main__":
    main()
