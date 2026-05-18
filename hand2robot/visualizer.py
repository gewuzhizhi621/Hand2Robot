"""OpenCV drawing helpers for camera, image, and video demos."""

from __future__ import annotations

import cv2
import mediapipe as mp


def draw_hand_landmarks(frame, detection_result):
    """Draw MediaPipe hand skeleton and landmark indices on a BGR frame."""
    if not detection_result.get("has_hand"):
        return frame

    raw_landmarks = detection_result.get("raw_landmarks")
    if raw_landmarks is None:
        return frame

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    mp_drawing.draw_landmarks(
        frame,
        raw_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_styles.get_default_hand_landmarks_style(),
        mp_styles.get_default_hand_connections_style(),
    )

    height, width = frame.shape[:2]
    for index, landmark in enumerate(raw_landmarks.landmark):
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.putText(
            frame,
            str(index),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


def draw_status_text(frame, lines=None):
    """Draw compact multi-line status text in the upper-left corner."""
    default_lines = ["Hand2Robot | Press ESC to quit"]
    text_lines = default_lines + list(lines or [])

    x, y0 = 18, 32
    for row, text in enumerate(text_lines):
        y = y0 + row * 26
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return frame


def draw_finger_angles(frame, finger_angles):
    """Draw five normalized finger bend values on the right side."""
    if not finger_angles:
        return frame

    height, width = frame.shape[:2]
    x = max(20, width - 230)
    y0 = 40
    cv2.putText(
        frame,
        "finger bend",
        (x, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for index, finger in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
        value = finger_angles.get(finger, 0.0)
        cv2.putText(
            frame,
            f"{finger}: {value:.2f}",
            (x, y0 + 26 * (index + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 220, 80),
            1,
            cv2.LINE_AA,
        )
    return frame


def draw_qpos(frame, qpos):
    """Draw a short qpos summary near the lower-right corner."""
    if not qpos:
        return frame

    height, width = frame.shape[:2]
    x = max(20, width - 300)
    y0 = max(180, height - 145)
    cv2.putText(
        frame,
        "qpos mean(rad)",
        (x, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for index, finger in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
        values = qpos.get(finger, [0.0, 0.0, 0.0])
        mean_value = sum(values) / len(values)
        cv2.putText(
            frame,
            f"{finger}: {mean_value:.2f}",
            (x, y0 + 24 * (index + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (180, 230, 255),
            1,
            cv2.LINE_AA,
        )
    return frame
