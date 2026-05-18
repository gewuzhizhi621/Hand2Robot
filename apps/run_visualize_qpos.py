"""Visualize a saved Hand2Robot qpos JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hand2robot.export_utils import ensure_dir, timestamp_string
from hand2robot.robot_hand_visualizer import plot_robot_hand


VIEW_CHOICES = ["first_person", "front", "side", "top", "isometric", "all"]
ALL_VIEWS = ["first_person", "front", "side", "top", "isometric"]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a qpos JSON file.")
    parser.add_argument("--qpos", required=True, type=Path, help="Path to qpos JSON.")
    parser.add_argument(
        "--view",
        default="first_person",
        choices=VIEW_CHOICES,
        help="Camera view mode for the Matplotlib visualization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.qpos.exists():
        print(f"qpos file not found: {args.qpos}")
        return

    qpos = json.loads(args.qpos.read_text(encoding="utf-8-sig"))
    output_dir = ensure_dir(PROJECT_ROOT / "examples" / "outputs")
    stamp = timestamp_string()

    if args.view == "all":
        for view in ALL_VIEWS:
            save_path = output_dir / f"robot_hand_{view}_{stamp}.png"
            plot_robot_hand(qpos, save_path=save_path, show=False, view_mode=view)
        return

    save_path = output_dir / f"robot_hand_{args.view}_{stamp}.png"
    plot_robot_hand(qpos, save_path=save_path, show=True, view_mode=args.view)


if __name__ == "__main__":
    main()
