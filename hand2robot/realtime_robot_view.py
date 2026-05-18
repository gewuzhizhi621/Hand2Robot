"""Tkinter-embedded Matplotlib realtime robot hand view."""

from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from hand2robot.hand_geometry import FINGER_COLORS, VIEW_MODES
from hand2robot.robot_hand_model import compute_robot_hand_points


OPEN_HAND_QPOS = {
    "thumb": [0.05, 0.05, 0.05],
    "index": [0.05, 0.05, 0.05],
    "middle": [0.05, 0.05, 0.05],
    "ring": [0.05, 0.05, 0.05],
    "pinky": [0.05, 0.05, 0.05],
}


class RealtimeRobotHandView:
    """A lightweight realtime 3D hand view embedded inside Tkinter."""

    def __init__(self, parent, view_mode: str = "first_person") -> None:
        self.view_mode = view_mode
        self.current_qpos = OPEN_HAND_QPOS
        self.show_axes = True
        self.show_labels = True
        self.figure = Figure(figsize=(5.2, 5.2), dpi=100)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.widget = self.canvas.get_tk_widget()
        self.widget.pack(fill="both", expand=True)
        self.draw_robot_hand(self.current_qpos)

    def update_qpos(self, qpos) -> None:
        """Update the current qpos and redraw without creating a new Figure."""
        if not qpos:
            return
        self.current_qpos = qpos
        self.draw_robot_hand(qpos)

    def set_view_mode(self, view_mode: str) -> None:
        """Switch camera view and redraw the current robot hand pose."""
        if view_mode not in VIEW_MODES:
            view_mode = "first_person"
        self.view_mode = view_mode
        self.draw_robot_hand(self.current_qpos)

    def set_display_options(self, show_axes: bool = True, show_labels: bool = True) -> None:
        """Toggle axes and finger labels."""
        self.show_axes = bool(show_axes)
        self.show_labels = bool(show_labels)
        self.draw_robot_hand(self.current_qpos)

    def save_current_image(self, save_path) -> Path:
        """Save the currently displayed robot hand image."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(save_path, dpi=180, bbox_inches="tight")
        return save_path

    def draw_robot_hand(self, qpos) -> None:
        """Redraw palm, fingers, joints, and tips for the current qpos."""
        self.ax.clear()
        model = compute_robot_hand_points(qpos)
        self._draw_palm(model["palm"])

        for finger, points in model["fingers"].items():
            color = FINGER_COLORS.get(finger, "tab:gray")
            self.ax.plot(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=color,
                linewidth=4.4,
                solid_capstyle="round",
            )
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=32)
            self.ax.scatter(
                points[-1, 0],
                points[-1, 1],
                points[-1, 2],
                color=color,
                s=82,
                edgecolors="black",
            )
            if self.show_labels:
                tip = points[-1]
                self.ax.text(tip[0], tip[1] + 0.03, tip[2] + 0.03, finger, fontsize=8, color=color)

        self._set_axes()
        self._apply_view()
        self.figure.tight_layout()
        self.canvas.draw_idle()

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
            linewidth=1.9,
        )

    def _set_axes(self) -> None:
        self.ax.set_xlim(-1.4, 1.4)
        self.ax.set_ylim(-0.6, 1.8)
        self.ax.set_zlim(-1.4, 0.6)
        self.ax.set_box_aspect((2.8, 2.4, 2.0))
        self.ax.set_title("Realtime Robot Hand", fontsize=11)
        if self.show_axes:
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")
            self.ax.grid(alpha=0.15)
        else:
            self.ax.set_axis_off()

    def _apply_view(self) -> None:
        view = VIEW_MODES.get(self.view_mode, VIEW_MODES["first_person"])
        self.ax.view_init(elev=view["elev"], azim=view["azim"])
