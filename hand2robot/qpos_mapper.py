"""Map normalized finger bends to a simplified robot hand qpos."""

from __future__ import annotations


FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


def _clamp(value: float, low: float, high: float) -> float:
    return round(max(float(low), min(float(high), float(value))), 4)


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 0.0
    t = max(0.0, min(1.0, (float(x) - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def _display_bend(bend: float) -> float:
    bend2 = _smoothstep(0.05, 0.95, bend)
    return max(0.0, min(1.0, bend2**0.75))


def map_finger_angles_to_qpos(
    finger_angles: dict[str, float],
    max_angle: float = 1.57,
) -> dict[str, list[float]]:
    """Convert five normalized bend values to 15 robot hand joint angles."""
    qpos: dict[str, list[float]] = {}

    for finger in FINGER_ORDER:
        bend = max(0.0, min(1.0, float(finger_angles.get(finger, 0.0))))
        bend2 = _display_bend(bend)
        if finger == "thumb":
            qpos[finger] = [
                _clamp(bend2 * 0.85, 0.0, 1.20),
                _clamp(bend2 * 1.25, 0.0, 1.45),
                _clamp(bend2 * 0.95, 0.0, 1.25),
            ]
        else:
            qpos[finger] = [
                _clamp(bend2 * 1.05, 0.0, 1.35),
                _clamp(bend2 * 1.45, 0.0, max_angle),
                _clamp(bend2 * 1.15, 0.0, 1.35),
            ]

    return qpos


def flatten_qpos(qpos: dict[str, list[float]]) -> list[float]:
    """Flatten qpos into thumb/index/middle/ring/pinky joint order."""
    values: list[float] = []
    for finger in FINGER_ORDER:
        values.extend(qpos.get(finger, [0.0, 0.0, 0.0]))
    return values


def pretty_qpos(qpos: dict[str, list[float]]) -> list[str]:
    """Return compact display lines for qpos."""
    lines = []
    for finger in FINGER_ORDER:
        values = qpos.get(finger, [0.0, 0.0, 0.0])
        lines.append(f"{finger}: [{values[0]:.2f}, {values[1]:.2f}, {values[2]:.2f}]")
    return lines
