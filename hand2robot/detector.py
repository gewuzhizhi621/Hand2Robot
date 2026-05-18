"""MediaPipe hand detection wrapper used by Hand2Robot.

This module only performs detection and data conversion. It intentionally does
not contain any OpenCV window loop, so it can be reused by camera, image, and
video entry points.
"""

from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    """Detect one hand and return normalized 21-point landmarks."""

    def __init__(
        self,
        max_num_hands: int = 1,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        """Detect the first hand in an OpenCV BGR image.

        Returns a dictionary with normalized landmarks of shape ``(21, 3)``.
        If no hand is found, ``has_hand`` is ``False`` and ``landmarks`` is
        an empty array.
        """
        image_height, image_width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        base_result: dict[str, Any] = {
            "has_hand": False,
            "handedness": "Unknown",
            "landmarks": np.empty((0, 3), dtype=float),
            "raw_landmarks": None,
            "image_width": image_width,
            "image_height": image_height,
        }

        if not results.multi_hand_landmarks:
            return base_result

        raw_landmarks = results.multi_hand_landmarks[0]
        landmarks = np.array(
            [[point.x, point.y, point.z] for point in raw_landmarks.landmark],
            dtype=float,
        )

        handedness = "Unknown"
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        base_result.update(
            {
                "has_hand": True,
                "handedness": handedness,
                "landmarks": landmarks,
                "raw_landmarks": raw_landmarks,
            }
        )
        return base_result

    def close(self) -> None:
        """Release the MediaPipe Hands object."""
        self._hands.close()
