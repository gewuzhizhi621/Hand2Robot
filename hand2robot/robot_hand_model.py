"""Simplified biomimetic 3D robot hand kinematic model."""

from __future__ import annotations

import math

import numpy as np

from hand2robot.hand_geometry import (
    FINGER_CONFIGS,
    FINGER_ORDER,
    PALM_BOTTOM_VERTICES,
    PALM_CENTER,
    PALM_TOP_VERTICES,
)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Smooth correction curve used for fist realism."""
    if edge0 == edge1:
        return 0.0
    t = clamp01((x - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def _palm_faces() -> list[np.ndarray]:
    top = PALM_TOP_VERTICES
    bottom = PALM_BOTTOM_VERTICES
    return [
        top,
        bottom[[0, 3, 2, 1]],
        np.array([top[0], top[1], bottom[1], bottom[0]]),
        np.array([top[1], top[2], bottom[2], bottom[1]]),
        np.array([top[2], top[3], bottom[3], bottom[2]]),
        np.array([top[3], top[0], bottom[0], bottom[3]]),
    ]


def _yaw_direction(yaw: float) -> np.ndarray:
    return np.array([math.sin(yaw), math.cos(yaw), 0.0], dtype=float)


def _finger_segment(config: dict, length: float, theta: float, curl_level: float) -> np.ndarray:
    yaw = float(config["base_yaw"]) + float(config["spread"]) * (1.0 - 0.45 * curl_level)
    forward = _yaw_direction(yaw)

    visual_theta = min(theta + float(config.get("base_pitch", 0.0)), 2.85)
    dy_scale = math.cos(visual_theta)
    z_drop = math.sin(visual_theta)

    # At high curl, distal segments should roll back toward the palm instead
    # of simply diving downward.
    palm_return = smoothstep(0.48, 0.95, curl_level) * max(0.0, visual_theta - 1.15)
    horizontal = length * (dy_scale - 0.34 * palm_return)
    vertical = -length * (z_drop + 0.18 * palm_return)
    return forward * horizontal + np.array([0.0, 0.0, vertical], dtype=float)


def _thumb_segment(config: dict, length: float, theta: float, curl_level: float) -> np.ndarray:
    base_yaw = float(config["base_yaw"])
    opposition = smoothstep(0.12, 0.95, curl_level)
    yaw = base_yaw + 0.95 * opposition
    forward = _yaw_direction(yaw)

    visual_theta = min(theta * 0.9 + float(config.get("base_pitch", 0.0)), 2.25)
    horizontal = length * math.cos(0.62 * visual_theta)
    vertical = -length * (0.18 + 0.86 * math.sin(visual_theta)) * opposition
    across_palm = np.array([0.42 * opposition * length, 0.20 * opposition * length, 0.0])
    return forward * horizontal + across_palm + np.array([0.0, 0.0, vertical])


def _apply_fist_realism(finger: str, points: np.ndarray, curl_level: float) -> np.ndarray:
    if finger == "thumb":
        return points

    strength = smoothstep(0.55, 0.95, curl_level)
    if strength <= 0.0:
        return points

    corrected = points.copy()
    for index, weight in ((2, 0.45), (3, 0.86)):
        target = PALM_CENTER.copy()
        target[0] += {"index": -0.16, "middle": 0.0, "ring": 0.12, "pinky": 0.22}[finger]
        target[1] += 0.04
        target[2] -= 0.34
        corrected[index] = (1.0 - strength * weight) * corrected[index] + strength * weight * target
    return corrected


def _apply_thumb_opposition(points: np.ndarray, curl_level: float) -> np.ndarray:
    strength = smoothstep(0.45, 0.95, curl_level)
    if strength <= 0.0:
        return points

    corrected = points.copy()
    target = np.array([-0.18, 0.38, -0.38], dtype=float)
    corrected[2] = (1.0 - 0.28 * strength) * corrected[2] + 0.28 * strength * target
    corrected[3] = (1.0 - 0.72 * strength) * corrected[3] + 0.72 * strength * target
    return corrected


def _finger_points(finger: str, qpos_values: list[float]) -> tuple[np.ndarray, dict]:
    config = FINGER_CONFIGS[finger]
    base = np.asarray(config["base"], dtype=float)
    lengths = config["lengths"]
    joints = [float(value) for value in qpos_values[:3]]
    while len(joints) < 3:
        joints.append(0.0)

    curl_level = clamp01(sum(joints) / (1.57 * 3.0))
    points = [base.copy()]
    current = base.copy()
    theta = 0.0
    joint_records = []
    for joint_index, (length, angle) in enumerate(zip(lengths, joints)):
        theta += angle
        if config.get("is_thumb"):
            segment = _thumb_segment(config, length, theta, curl_level)
        else:
            segment = _finger_segment(config, length, theta, curl_level)
        current = current + segment
        points.append(current.copy())
        joint_records.append(
            {
                "joint": joint_index,
                "angle": angle,
                "curl_level": curl_level,
                "position": current.copy(),
            }
        )

    point_array = np.vstack(points)
    if config.get("is_thumb"):
        point_array = _apply_thumb_opposition(point_array, curl_level)
    else:
        point_array = _apply_fist_realism(finger, point_array, curl_level)
    return point_array, {"curl_level": curl_level, "joints": joint_records}


def compute_robot_hand_points(qpos: dict[str, list[float]]):
    """Compute palm body, finger chains, tips, and joint metadata from qpos."""
    fingers = {}
    tips = {}
    joints = {}

    for finger in FINGER_ORDER:
        points, metadata = _finger_points(finger, qpos.get(finger, [0.0, 0.0, 0.0]))
        fingers[finger] = points
        tips[finger] = points[-1].copy()
        joints[finger] = metadata

    return {
        "palm": _palm_faces(),
        "palm_top": PALM_TOP_VERTICES.copy(),
        "palm_bottom": PALM_BOTTOM_VERTICES.copy(),
        "palm_center": PALM_CENTER.copy(),
        "fingers": fingers,
        "tips": tips,
        "joints": joints,
    }
