"""Export helpers for Hand2Robot data, images, and videos."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import cv2


LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]


def ensure_dir(path) -> Path:
    """Create a directory if needed and return it as a Path."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def timestamp_string() -> str:
    """Return a filesystem-friendly timestamp such as 20260518_153012."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, data) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {path}")
    return path


def save_landmarks_json(output_dir, detection_result, filename=None, frame_id: int = 0):
    """Save detected hand landmarks to JSON."""
    output_dir = ensure_dir(output_dir)
    filename = filename or f"landmarks_{timestamp_string()}.json"
    landmarks = detection_result.get("landmarks", [])
    data = {
        "frame_id": frame_id,
        "handedness": detection_result.get("handedness", "Unknown"),
        "landmark_names": LANDMARK_NAMES,
        "landmarks": [
            {
                "id": index,
                "name": LANDMARK_NAMES[index],
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
            }
            for index, point in enumerate(landmarks)
        ],
    }
    return _write_json(output_dir / filename, data)


def save_finger_angles_json(output_dir, finger_angles, filename=None):
    output_dir = ensure_dir(output_dir)
    filename = filename or f"finger_angles_{timestamp_string()}.json"
    return _write_json(output_dir / filename, finger_angles)


def save_qpos_json(output_dir, qpos, filename=None):
    output_dir = ensure_dir(output_dir)
    filename = filename or f"qpos_{timestamp_string()}.json"
    path = _write_json(output_dir / filename, qpos)
    if filename != "qpos_latest.json":
        _write_json(output_dir / "qpos_latest.json", qpos)
    return path


def save_qpos_sequence_json(output_dir, qpos_sequence, filename=None):
    output_dir = ensure_dir(output_dir)
    filename = filename or f"qpos_sequence_{timestamp_string()}.json"
    return _write_json(output_dir / filename, qpos_sequence)


def save_finger_angles_csv(output_dir, sequence, filename=None):
    output_dir = ensure_dir(output_dir)
    filename = filename or f"finger_angles_{timestamp_string()}.csv"
    path = output_dir / filename
    fieldnames = ["frame_id", "has_hand", "thumb", "index", "middle", "ring", "pinky"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in sequence:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    print(f"Saved: {path}")
    return path


def save_image(output_dir, image_bgr, filename=None):
    output_dir = ensure_dir(output_dir)
    filename = filename or f"image_{timestamp_string()}.png"
    path = output_dir / filename
    cv2.imwrite(str(path), image_bgr)
    print(f"Saved: {path}")
    return path
