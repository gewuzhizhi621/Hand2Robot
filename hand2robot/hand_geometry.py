"""Geometry constants for the lightweight Hand2Robot 3D hand model."""

from __future__ import annotations

import math

import numpy as np


FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

FINGER_COLORS = {
    "thumb": "tab:red",
    "index": "tab:blue",
    "middle": "tab:green",
    "ring": "tab:purple",
    "pinky": "tab:orange",
}

PALM_WIDTH_TOP = 1.6
PALM_WIDTH_BOTTOM = 1.1
PALM_HEIGHT = 1.1
PALM_THICKNESS = 0.18
WRIST_Y = -0.35
FINGER_BASE_Y = WRIST_Y + PALM_HEIGHT
PALM_TOP_Z = 0.05
PALM_BOTTOM_Z = PALM_TOP_Z - PALM_THICKNESS
PALM_CENTER = np.array([0.0, 0.12, -0.04], dtype=float)

PALM_TOP_VERTICES = np.array(
    [
        [-PALM_WIDTH_BOTTOM / 2, WRIST_Y, PALM_TOP_Z],
        [PALM_WIDTH_BOTTOM / 2, WRIST_Y, PALM_TOP_Z],
        [PALM_WIDTH_TOP / 2, FINGER_BASE_Y, PALM_TOP_Z],
        [-PALM_WIDTH_TOP / 2, FINGER_BASE_Y, PALM_TOP_Z],
    ],
    dtype=float,
)
PALM_BOTTOM_VERTICES = PALM_TOP_VERTICES.copy()
PALM_BOTTOM_VERTICES[:, 2] = PALM_BOTTOM_Z

FINGER_CONFIGS = {
    "thumb": {
        "base": np.array([-0.75, 0.05, 0.05], dtype=float),
        "lengths": [0.35, 0.28, 0.22],
        "base_yaw": math.radians(-55),
        "base_pitch": math.radians(10),
        "spread": -0.45,
        "is_thumb": True,
    },
    "index": {
        "base": np.array([-0.38, 0.62, 0.08], dtype=float),
        "lengths": [0.50, 0.36, 0.26],
        "base_yaw": math.radians(-12),
        "base_pitch": 0.0,
        "spread": -0.18,
        "is_thumb": False,
    },
    "middle": {
        "base": np.array([0.0, 0.70, 0.08], dtype=float),
        "lengths": [0.55, 0.40, 0.30],
        "base_yaw": 0.0,
        "base_pitch": 0.0,
        "spread": 0.0,
        "is_thumb": False,
    },
    "ring": {
        "base": np.array([0.35, 0.62, 0.08], dtype=float),
        "lengths": [0.50, 0.36, 0.26],
        "base_yaw": math.radians(10),
        "base_pitch": 0.0,
        "spread": 0.12,
        "is_thumb": False,
    },
    "pinky": {
        "base": np.array([0.65, 0.45, 0.08], dtype=float),
        "lengths": [0.42, 0.30, 0.22],
        "base_yaw": math.radians(18),
        "base_pitch": 0.0,
        "spread": 0.22,
        "is_thumb": False,
    },
}

VIEW_MODES = {
    "first_person": {"elev": 68, "azim": -90},
    "front": {"elev": 18, "azim": -90},
    "side": {"elev": 16, "azim": 0},
    "top": {"elev": 90, "azim": -90},
    "isometric": {"elev": 28, "azim": -55},
}
