"""Finger bend estimation from MediaPipe's 21 normalized hand landmarks."""

from __future__ import annotations

import math

import numpy as np


FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
STRAIGHT_ANGLE = 170.0
FULL_BEND_ANGLE = 65.0
ANGLE_RANGE = STRAIGHT_ANGLE - FULL_BEND_ANGLE


def clamp01(x: float) -> float:
    """Clamp a number to the closed interval [0, 1]."""
    return max(0.0, min(1.0, float(x)))


def angle_between_three_points(a, b, c) -> float:
    """Return angle ABC in degrees for three 3D points."""
    point_a = np.asarray(a, dtype=float)
    point_b = np.asarray(b, dtype=float)
    point_c = np.asarray(c, dtype=float)

    ba = point_a - point_b
    bc = point_c - point_b
    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm_product < 1e-8:
        return 180.0

    cosine = float(np.dot(ba, bc) / norm_product)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _joint_bend(landmarks: np.ndarray, a: int, b: int, c: int) -> float:
    angle = angle_between_three_points(landmarks[a], landmarks[b], landmarks[c])
    return clamp01((STRAIGHT_ANGLE - angle) / ANGLE_RANGE)


def _enhance_bend(value: float) -> float:
    """Boost mid-range MediaPipe bends so a fist reads clearly as closed."""
    value = clamp01(value)
    return clamp01((value**0.70) * 1.20)


def _palm_center(landmarks: np.ndarray) -> np.ndarray:
    return np.mean(landmarks[[0, 5, 9, 13, 17]], axis=0)


def _tip_closure_bonus(landmarks: np.ndarray, mcp_index: int, tip_index: int) -> float:
    """Add bend when the fingertip folds back toward the palm area."""
    wrist = landmarks[0]
    palm_center = _palm_center(landmarks)
    mcp_distance = max(float(np.linalg.norm(landmarks[mcp_index] - wrist)), 1e-6)
    tip_to_wrist = float(np.linalg.norm(landmarks[tip_index] - wrist))
    tip_to_palm = float(np.linalg.norm(landmarks[tip_index] - palm_center))
    wrist_bonus = clamp01((mcp_distance * 1.28 - tip_to_wrist) / (mcp_distance * 0.86))
    palm_bonus = clamp01((mcp_distance * 1.10 - tip_to_palm) / (mcp_distance * 0.82))
    return max(wrist_bonus, palm_bonus)


def _finger_bend(
    landmarks: np.ndarray,
    mcp_triplet: tuple[int, int, int],
    pip_triplet: tuple[int, int, int],
    dip_triplet: tuple[int, int, int],
    mcp_index: int,
    tip_index: int,
) -> float:
    """Weighted bend for non-thumb fingers.

    PIP and DIP carry more weight because a fist is mostly visible in those
    middle and distal joints, while MCP can remain comparatively open.
    """
    mcp_bend = _joint_bend(landmarks, *mcp_triplet)
    pip_bend = _joint_bend(landmarks, *pip_triplet)
    dip_bend = _joint_bend(landmarks, *dip_triplet)
    raw_bend = 0.2 * mcp_bend + 0.5 * pip_bend + 0.3 * dip_bend
    bonus = _tip_closure_bonus(landmarks, mcp_index, tip_index)
    return clamp01(_enhance_bend(raw_bend) + 0.18 * bonus)


def _thumb_bend(landmarks: np.ndarray) -> float:
    """Estimate thumb closure with joint angles and distance to index finger."""
    mcp_bend = _joint_bend(landmarks, 1, 2, 3)
    ip_bend = _joint_bend(landmarks, 2, 3, 4)

    thumb_tip = landmarks[4]
    index_mcp = landmarks[5]
    index_pip = landmarks[6]
    hand_scale = max(float(np.linalg.norm(landmarks[0] - landmarks[9])), 1e-6)
    distance_to_index = min(
        float(np.linalg.norm(thumb_tip - index_mcp)),
        float(np.linalg.norm(thumb_tip - index_pip)),
    )

    # When the thumb tip approaches the index root area, it is likely扣合.
    distance_bend = clamp01((0.95 * hand_scale - distance_to_index) / (0.65 * hand_scale))
    raw_bend = 0.42 * mcp_bend + 0.34 * ip_bend + 0.24 * distance_bend
    return clamp01(_enhance_bend(raw_bend) * 1.18)


def compute_finger_angles(landmarks: np.ndarray) -> dict[str, float]:
    """Compute normalized bend values for thumb, index, middle, ring, pinky."""
    landmarks = np.asarray(landmarks, dtype=float)
    if landmarks.shape != (21, 3):
        raise ValueError("landmarks must be a numpy array with shape (21, 3)")

    bends = {
        "thumb": _thumb_bend(landmarks),
        "index": _finger_bend(landmarks, (0, 5, 6), (5, 6, 7), (6, 7, 8), 5, 8),
        "middle": _finger_bend(landmarks, (0, 9, 10), (9, 10, 11), (10, 11, 12), 9, 12),
        "ring": _finger_bend(landmarks, (0, 13, 14), (13, 14, 15), (14, 15, 16), 13, 16),
        "pinky": _finger_bend(landmarks, (0, 17, 18), (17, 18, 19), (18, 19, 20), 17, 20),
    }
    return {finger: round(clamp01(value), 4) for finger, value in bends.items()}


def pretty_finger_angles(finger_angles: dict[str, float]) -> list[str]:
    """Return display-friendly lines for normalized finger bend values."""
    return [f"{name}: {finger_angles.get(name, 0.0):.2f}" for name in FINGER_NAMES]
