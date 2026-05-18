"""Reusable frame and video processing pipeline for Hand2Robot apps."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from hand2robot.finger_angle import compute_finger_angles
from hand2robot.hand_geometry import FINGER_COLORS, VIEW_MODES
from hand2robot.qpos_mapper import map_finger_angles_to_qpos
from hand2robot.realtime_robot_view import OPEN_HAND_QPOS
from hand2robot.robot_hand_model import compute_robot_hand_points
from hand2robot.visualizer import draw_finger_angles, draw_hand_landmarks, draw_qpos, draw_status_text


FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


class RobotHandFrameRenderer:
    """Offscreen Matplotlib renderer for robot hand videos."""

    def __init__(self, width: int = 640, height: int = 480, view_mode: str = "first_person") -> None:
        self.width = int(width)
        self.height = int(height)
        self.view_mode = view_mode if view_mode in VIEW_MODES else "first_person"
        self.figure = Figure(figsize=(self.width / 100, self.height / 100), dpi=100)
        self.canvas = FigureCanvasAgg(self.figure)
        self.ax = self.figure.add_subplot(111, projection="3d")

    def render(self, qpos) -> np.ndarray:
        self.ax.clear()
        model = compute_robot_hand_points(qpos or OPEN_HAND_QPOS)
        self._draw_palm(model["palm"])
        for finger, points in model["fingers"].items():
            color = FINGER_COLORS.get(finger, "tab:gray")
            self.ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=4.4, solid_capstyle="round")
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=30)
            self.ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color=color, s=80, edgecolors="black")
        self._set_axes()
        view = VIEW_MODES[self.view_mode]
        self.ax.view_init(elev=view["elev"], azim=view["azim"])
        self.figure.tight_layout()
        self.canvas.draw()
        rgba = np.asarray(self.canvas.buffer_rgba())
        rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _draw_palm(self, palm_faces) -> None:
        collection = Poly3DCollection(
            palm_faces,
            facecolors=(0.72, 0.74, 0.78, 0.36),
            edgecolors=(0.18, 0.18, 0.20, 1.0),
            linewidths=1.1,
        )
        self.ax.add_collection3d(collection)
        top = palm_faces[0]
        closed_top = list(top) + [top[0]]
        self.ax.plot(
            [point[0] for point in closed_top],
            [point[1] for point in closed_top],
            [point[2] for point in closed_top],
            color="black",
            linewidth=1.8,
        )

    def _set_axes(self) -> None:
        self.ax.set_xlim(-1.4, 1.4)
        self.ax.set_ylim(-0.6, 1.8)
        self.ax.set_zlim(-1.4, 0.6)
        self.ax.set_box_aspect((2.8, 2.4, 2.0))
        self.ax.set_title("Hand2Robot Robot Hand", fontsize=11)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.grid(alpha=0.15)


def process_frame(
    frame_bgr,
    detector,
    mirror: bool = False,
    smoother=None,
    draw_overlay: bool = True,
    draw_status: bool = True,
    status_lines=None,
):
    """Detect one hand, compute finger bends/qpos, and return annotated output."""
    working_frame = cv2.flip(frame_bgr, 1) if mirror else frame_bgr.copy()
    detection = detector.detect(working_frame)
    finger_angles = None
    qpos = None

    annotated_frame = working_frame.copy()
    if detection["has_hand"]:
        finger_angles = compute_finger_angles(detection["landmarks"])
        if smoother is not None:
            finger_angles = smoother.smooth_dict(finger_angles)
        qpos = map_finger_angles_to_qpos(finger_angles)
        if draw_overlay:
            draw_hand_landmarks(annotated_frame, detection)
            draw_finger_angles(annotated_frame, finger_angles)
            draw_qpos(annotated_frame, qpos)

    if draw_overlay and draw_status:
        draw_status_text(annotated_frame, status_lines)

    return {
        "annotated_frame": annotated_frame,
        "detection": detection,
        "finger_angles": finger_angles,
        "qpos": qpos,
        "has_hand": detection["has_hand"],
        "handedness": detection.get("handedness", "Unknown"),
    }


def process_video_to_files(video_path, output_dir, progress_callback=None, view_mode: str = "first_person"):
    """Process a whole video and save human-hand and robot-hand videos."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = None
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    try:
        from hand2robot.detector import HandDetector

        detector = HandDetector()
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video = output_dir / "annotated_video.mp4"
        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        robot_video = output_dir / "robot_hand_video.mp4"
        robot_size = (640, 480)
        robot_writer = cv2.VideoWriter(str(robot_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, robot_size)
        robot_renderer = RobotHandFrameRenderer(width=robot_size[0], height=robot_size[1], view_mode=view_mode)
        qpos_sequence = []
        angle_rows = []
        detected_frames = 0
        frame_index = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = process_frame(frame, detector, mirror=False, smoother=None, draw_overlay=True)
            writer.write(result["annotated_frame"])
            robot_frame = robot_renderer.render(result["qpos"] if result["has_hand"] else OPEN_HAND_QPOS)
            robot_writer.write(robot_frame)
            has_hand = bool(result["has_hand"])
            if has_hand:
                detected_frames += 1

            qpos_sequence.append(
                {
                    "frame_index": frame_index,
                    "has_hand": has_hand,
                    "handedness": result["handedness"],
                    "qpos": result["qpos"],
                }
            )
            row = {"frame_index": frame_index, "has_hand": has_hand, "handedness": result["handedness"]}
            if result["finger_angles"]:
                row.update(result["finger_angles"])
            angle_rows.append(row)

            frame_index += 1
            if progress_callback and (frame_index % 10 == 0 or frame_index == total_frames):
                progress_callback(frame_index, total_frames)

        writer.release()
        robot_writer.release()

        qpos_path = output_dir / "qpos_sequence.json"
        qpos_path.write_text(json.dumps(qpos_sequence, ensure_ascii=False, indent=2), encoding="utf-8")

        csv_path = output_dir / "finger_angles_sequence.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            fieldnames = ["frame_index", "has_hand", "handedness", *FINGER_ORDER]
            writer_csv = csv.DictWriter(file, fieldnames=fieldnames)
            writer_csv.writeheader()
            for row in angle_rows:
                writer_csv.writerow({name: row.get(name, "") for name in fieldnames})

        summary = {
            "source_video": str(video_path),
            "total_frames": frame_index,
            "detected_frames": detected_frames,
            "missing_frames": frame_index - detected_frames,
            "human_hand_video": "annotated_video.mp4",
            "robot_hand_video": "robot_hand_video.mp4",
            "output_video": "annotated_video.mp4",
            "qpos_sequence": "qpos_sequence.json",
            "finger_angles_sequence": "finger_angles_sequence.csv",
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    finally:
        capture.release()
        if detector is not None:
            detector.close()
