"""Small smoothing helpers for real-time Hand2Robot demos."""

from __future__ import annotations


class ExponentialSmoother:
    """Exponential moving average for dictionaries of numeric values."""

    def __init__(self, alpha: float = 0.6) -> None:
        self.alpha = max(0.0, min(1.0, float(alpha)))
        self.previous = None

    def smooth_dict(self, current_dict):
        """Smooth a dict of floats or a dict of lists of floats."""
        if self.previous is None:
            self.previous = current_dict.copy()
            return current_dict.copy()

        smoothed = {}
        for key, current_value in current_dict.items():
            previous_value = self.previous.get(key, current_value)
            if isinstance(current_value, list):
                smoothed[key] = [
                    self.alpha * float(cur) + (1.0 - self.alpha) * float(prev)
                    for cur, prev in zip(current_value, previous_value)
                ]
            else:
                smoothed[key] = (
                    self.alpha * float(current_value)
                    + (1.0 - self.alpha) * float(previous_value)
                )

        self.previous = smoothed.copy()
        return smoothed
