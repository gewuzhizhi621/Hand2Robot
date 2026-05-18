"""Matplotlib 3D visualization for the simplified robot hand."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from hand2robot.hand_geometry import FINGER_COLORS, VIEW_MODES
from hand2robot.robot_hand_model import compute_robot_hand_points


def _set_axes(ax) -> None:
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.6, 1.8)
    ax.set_zlim(-1.4, 0.6)
    ax.set_box_aspect((2.8, 2.4, 2.0))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(alpha=0.18)


def _apply_view(ax, view_mode: str) -> str:
    if view_mode not in VIEW_MODES:
        print(f"Unknown view mode '{view_mode}', using first_person.")
        view_mode = "first_person"
    view = VIEW_MODES[view_mode]
    ax.view_init(elev=view["elev"], azim=view["azim"])
    return view_mode


def _draw_palm(ax, palm_faces) -> None:
    collection = Poly3DCollection(
        palm_faces,
        facecolors=(0.72, 0.74, 0.78, 0.36),
        edgecolors=(0.18, 0.18, 0.20, 1.0),
        linewidths=1.4,
    )
    ax.add_collection3d(collection)

    # Add a darker top outline so the wrist-to-finger direction reads clearly.
    top = palm_faces[0]
    closed_top = list(top) + [top[0]]
    xs = [point[0] for point in closed_top]
    ys = [point[1] for point in closed_top]
    zs = [point[2] for point in closed_top]
    ax.plot(xs, ys, zs, color="black", linewidth=2.2)


def plot_robot_hand(qpos, save_path=None, show=True, view_mode="first_person"):
    """Plot a simplified 3D robot hand pose from qpos.

    ``first_person`` is the default view for README-friendly images: wrist at
    the bottom, fingers toward the top, and thumb opened to the side.
    """
    model = compute_robot_hand_points(qpos)
    fig = plt.figure(figsize=(7.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    _draw_palm(ax, model["palm"])

    for finger, points in model["fingers"].items():
        color = FINGER_COLORS.get(finger, "tab:gray")
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=5, solid_capstyle="round")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=45, depthshade=True)
        ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color=color, s=110, marker="o", edgecolors="black")
        tip = points[-1]
        ax.text(tip[0], tip[1] + 0.035, tip[2] + 0.035, finger, fontsize=9, color=color)

    center = model["palm_center"]
    ax.scatter(center[0], center[1], center[2], color="black", s=25, alpha=0.55)

    ax.set_title("Hand2Robot - Simplified Robot Hand Pose")
    _set_axes(ax)
    view_mode = _apply_view(ax, view_mode)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved: {save_path} ({view_mode})")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
