"""Render README-friendly demo pose images for Hand2Robot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hand2robot.export_utils import ensure_dir
from hand2robot.robot_hand_visualizer import plot_robot_hand


DEMO_POSES = {
    "open": "qpos_open_demo.json",
    "half_fist": "qpos_half_fist_demo.json",
    "fist": "qpos_fist_demo.json",
    "pinch": "qpos_pinch_demo.json",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Render Hand2Robot demo pose images.")
    parser.add_argument(
        "--extra-views",
        action="store_true",
        help="Also render side and isometric images for each pose.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(PROJECT_ROOT / "examples" / "outputs")
    views = ["first_person"]
    if args.extra_views:
        views.extend(["side", "isometric"])

    for pose_name, filename in DEMO_POSES.items():
        qpos_path = output_dir / filename
        if not qpos_path.exists():
            print(f"Missing demo qpos: {qpos_path}")
            continue

        qpos = json.loads(qpos_path.read_text(encoding="utf-8-sig"))
        for view in views:
            suffix = "" if view == "first_person" else f"_{view}"
            save_path = output_dir / f"robot_{pose_name}_demo{suffix}.png"
            plot_robot_hand(qpos, save_path=save_path, show=False, view_mode=view)


if __name__ == "__main__":
    main()
