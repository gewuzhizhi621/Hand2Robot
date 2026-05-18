"""Chinese Tkinter GUI for Hand2Robot realtime visualization."""

from __future__ import annotations

import json
import os
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import cv2
from PIL import Image, ImageTk

from hand2robot.detector import HandDetector
from hand2robot.export_utils import (
    ensure_dir,
    save_finger_angles_json,
    save_image,
    save_landmarks_json,
    save_qpos_json,
    timestamp_string,
)
from hand2robot.pipeline import process_frame, process_video_to_files
from hand2robot.realtime_robot_view import OPEN_HAND_QPOS, RealtimeRobotHandView
from hand2robot.smoothing import ExponentialSmoother


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "examples" / "outputs"
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
VIEW_LABEL_TO_MODE = {
    "第一人称视角": "first_person",
    "正面视角": "front",
    "侧面视角": "side",
    "俯视视角": "top",
    "等轴测视角": "isometric",
}


def show_bgr_image_on_label(label, frame_bgr, max_size=(720, 520)) -> None:
    """Display an OpenCV BGR image on a Tkinter label."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(image)
    label.configure(image=photo)
    label.image = photo


class Hand2RobotGUI:
    """Hand2Robot Chinese GUI."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Hand2Robot 人手到机器人手姿态映射系统")
        self.root.geometry("1280x760")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.detector = HandDetector()
        self.output_root = ensure_dir(DEFAULT_OUTPUT_ROOT)
        self.current_run_dir: Path | None = None

        self.camera_cap = None
        self.video_cap = None
        self.camera_running = False
        self.video_running = False
        self.video_paused = False
        self.video_export_running = False

        self.selected_image_path: Path | None = None
        self.selected_video_path: Path | None = None
        self.current_mode = "待机"
        self.current_frame_index = 0
        self.total_video_frames = 0
        self.save_counter = 0
        self.frame_counter = 0
        self.last_frame_time = time.perf_counter()
        self.fps = 0.0

        self.current_frame_bgr = None
        self.current_annotated_frame_bgr = None
        self.current_detection = None
        self.current_finger_angles = None
        self.current_qpos = OPEN_HAND_QPOS
        self.smoother = ExponentialSmoother(alpha=0.6)

        self._build_layout()
        self.robot_view.update_qpos(self.current_qpos)
        self._update_status_text("待机", "未检测到手")

    def _build_layout(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        center = ttk.Frame(self.root, padding=8)
        center.grid(row=0, column=1, sticky="nsew")
        right = ttk.Frame(self.root, padding=8)
        right.grid(row=0, column=2, sticky="nsew")
        bottom = ttk.Frame(self.root, padding=8)
        bottom.grid(row=1, column=0, columnspan=3, sticky="ew")

        center.rowconfigure(1, weight=1)
        center.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=1)
        bottom.columnconfigure(2, weight=2)

        self._build_left_controls(left)

        ttk.Label(center, text="视觉预览区").grid(row=0, column=0, sticky="w")
        self.preview_label = ttk.Label(center, anchor="center", relief="groove")
        self.preview_label.grid(row=1, column=0, sticky="nsew")

        self._build_robot_panel(right)
        self._build_bottom_panel(bottom)

    def _build_left_controls(self, parent) -> None:
        ttk.Label(parent, text="输入方式").pack(anchor="w")
        self.mode_var = tk.StringVar(value="摄像头")
        ttk.Combobox(parent, textvariable=self.mode_var, values=["摄像头", "照片", "视频"], state="readonly", width=16).pack(fill="x", pady=(0, 8))

        camera_box = ttk.LabelFrame(parent, text="摄像头设置", padding=6)
        camera_box.pack(fill="x", pady=4)
        ttk.Label(camera_box, text="摄像头编号").pack(anchor="w")
        self.camera_index_var = tk.StringVar(value="0")
        ttk.Combobox(camera_box, textvariable=self.camera_index_var, values=["0", "1", "2"], state="readonly", width=8).pack(fill="x")
        self.mirror_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(camera_box, text="镜像显示", variable=self.mirror_var).pack(anchor="w")
        ttk.Button(camera_box, text="开始摄像头", command=self.start_camera).pack(fill="x", pady=2)
        ttk.Button(camera_box, text="停止摄像头", command=self.stop_camera).pack(fill="x", pady=2)

        image_box = ttk.LabelFrame(parent, text="照片输入", padding=6)
        image_box.pack(fill="x", pady=4)
        ttk.Button(image_box, text="选择照片", command=self.select_image).pack(fill="x", pady=2)
        ttk.Button(image_box, text="运行照片", command=self.run_image).pack(fill="x", pady=2)

        video_box = ttk.LabelFrame(parent, text="视频输入", padding=6)
        video_box.pack(fill="x", pady=4)
        ttk.Button(video_box, text="选择视频", command=self.select_video).pack(fill="x", pady=2)
        ttk.Button(video_box, text="运行视频", command=self.run_video).pack(fill="x", pady=2)
        ttk.Button(video_box, text="暂停视频", command=self.pause_video).pack(fill="x", pady=2)
        ttk.Button(video_box, text="停止视频", command=self.stop_video).pack(fill="x", pady=2)
        ttk.Button(video_box, text="保存所有帧组成视频", command=self.save_processed_video).pack(fill="x", pady=2)

        output_box = ttk.LabelFrame(parent, text="输出目录", padding=6)
        output_box.pack(fill="x", pady=4)
        ttk.Button(output_box, text="选择输出目录", command=self.choose_output_root).pack(fill="x", pady=2)

        save_box = ttk.LabelFrame(parent, text="保存操作", padding=8)
        save_box.pack(fill="x", pady=(8, 4))
        ttk.Button(save_box, text="保存当前帧完整结果", command=self.save_current_full_result).pack(fill="x", pady=3)
        ttk.Button(save_box, text="打开当前输出文件夹", command=self.open_current_run_dir).pack(fill="x", pady=3)

    def _build_robot_panel(self, parent) -> None:
        options = ttk.Frame(parent)
        options.grid(row=0, column=0, sticky="ew")
        ttk.Label(options, text="仿真视角").grid(row=0, column=0, sticky="w")
        self.view_label_var = tk.StringVar(value="第一人称视角")
        view_box = ttk.Combobox(options, textvariable=self.view_label_var, values=list(VIEW_LABEL_TO_MODE), state="readonly", width=14)
        view_box.grid(row=0, column=1, padx=4)
        view_box.bind("<<ComboboxSelected>>", self.on_view_changed)

        self.smooth_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text="姿态平滑", variable=self.smooth_var, command=self.reset_smoother).grid(row=0, column=2, padx=4)
        ttk.Label(options, text="仿真刷新间隔").grid(row=1, column=0, sticky="w")
        self.robot_interval_var = tk.IntVar(value=2)
        ttk.Spinbox(options, from_=1, to=10, textvariable=self.robot_interval_var, width=5).grid(row=1, column=1, sticky="w", padx=4)
        self.show_axes_var = tk.BooleanVar(value=True)
        self.show_labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text="显示坐标轴", variable=self.show_axes_var, command=self.on_display_options_changed).grid(row=1, column=2, padx=4)
        ttk.Checkbutton(options, text="显示手指名称", variable=self.show_labels_var, command=self.on_display_options_changed).grid(row=1, column=3, padx=4)

        robot_frame = ttk.Frame(parent)
        robot_frame.grid(row=1, column=0, sticky="nsew")
        self.robot_view = RealtimeRobotHandView(robot_frame, view_mode="first_person")

    def _build_bottom_panel(self, parent) -> None:
        self.status_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.status_var).grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        self.finger_text = tk.Text(parent, height=8, width=28)
        self.finger_text.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.qpos_text = tk.Text(parent, height=8, width=38)
        self.qpos_text.grid(row=1, column=1, sticky="nsew", padx=(0, 6))
        self.log_text = scrolledtext.ScrolledText(parent, height=8)
        self.log_text.grid(row=1, column=2, sticky="nsew")

    def start_camera(self) -> None:
        self.stop_video()
        self.stop_camera(log_message=False)
        self.current_mode = "摄像头"
        self.mode_var.set("摄像头")
        self.current_run_dir = self._create_run_dir("camera_run")
        self.save_counter = 0
        self.current_frame_index = 0
        self.reset_smoother()
        camera_index = int(self.camera_index_var.get())
        self.camera_cap = cv2.VideoCapture(camera_index)
        if not self.camera_cap.isOpened():
            self.camera_cap.release()
            self.camera_cap = None
            self.log("无法打开摄像头，请尝试切换摄像头编号为 1 或 2。")
            return
        self.camera_running = True
        self.log(f"摄像头已启动：编号 {camera_index}")
        self.update_camera_frame()

    def stop_camera(self, log_message: bool = True) -> None:
        self.camera_running = False
        if self.camera_cap is not None:
            self.camera_cap.release()
            self.camera_cap = None
        if log_message:
            self.log("摄像头已停止。")

    def update_camera_frame(self) -> None:
        if not self.camera_running or self.camera_cap is None:
            return
        ok, frame = self.camera_cap.read()
        if not ok:
            self.log("读取摄像头画面失败。")
            self.stop_camera()
            return
        self.current_frame_index += 1
        self._process_and_show_frame(frame, mirror=self.mirror_var.get())
        self.root.after(30, self.update_camera_frame)

    def select_image(self) -> None:
        path = filedialog.askopenfilename(title="选择照片", filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")])
        if path:
            self.selected_image_path = Path(path)
            self.mode_var.set("照片")
            self.log(f"已选择照片：{self.selected_image_path}")

    def run_image(self) -> None:
        if self.selected_image_path is None:
            self.log("请先选择照片。")
            return
        image = cv2.imread(str(self.selected_image_path))
        if image is None:
            self.log(f"读取照片失败：{self.selected_image_path}")
            return
        self.stop_camera(log_message=False)
        self.stop_video(log_message=False)
        self.current_mode = "照片"
        self.current_run_dir = self._create_run_dir("image_run")
        self.save_counter = 0
        self.current_frame_index = 1
        self.reset_smoother()
        self._process_and_show_frame(image, mirror=False, use_smoothing=False)
        self._save_image_run(image)
        self.log("照片处理完成，并已自动保存完整结果。")

    def select_video(self) -> None:
        path = filedialog.askopenfilename(title="选择视频", filetypes=[("视频文件", "*.mp4 *.avi *.mov"), ("所有文件", "*.*")])
        if path:
            self.selected_video_path = Path(path)
            self.mode_var.set("视频")
            self.log(f"已选择视频：{self.selected_video_path}")

    def run_video(self) -> None:
        if self.selected_video_path is None:
            self.log("请先选择视频。")
            return
        self.stop_camera(log_message=False)
        self.stop_video(log_message=False)
        self.video_cap = cv2.VideoCapture(str(self.selected_video_path))
        if not self.video_cap.isOpened():
            self.log(f"无法打开视频：{self.selected_video_path}")
            return
        self.current_mode = "视频"
        self.current_run_dir = self._create_run_dir("video_run")
        self.save_counter = 0
        self.current_frame_index = 0
        self.total_video_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.video_running = True
        self.video_paused = False
        self.reset_smoother()
        self.log("视频已开始逐帧播放。")
        self.update_video_frame()

    def pause_video(self) -> None:
        if not self.video_running:
            return
        self.video_paused = not self.video_paused
        self.log("视频已暂停。" if self.video_paused else "视频继续播放。")
        if not self.video_paused:
            self.update_video_frame()

    def stop_video(self, log_message: bool = True) -> None:
        self.video_running = False
        self.video_paused = False
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        if log_message:
            self.log("视频已停止。")

    def update_video_frame(self) -> None:
        if not self.video_running or self.video_paused or self.video_cap is None:
            return
        ok, frame = self.video_cap.read()
        if not ok:
            self.stop_video()
            self.log("视频播放完成。")
            return
        self.current_frame_index += 1
        self._process_and_show_frame(frame, mirror=False)
        delay = int(1000 / max(self.video_cap.get(cv2.CAP_PROP_FPS) or 25, 1))
        self.root.after(max(15, delay), self.update_video_frame)

    def save_processed_video(self) -> None:
        if self.current_mode != "视频" or self.selected_video_path is None or self.current_run_dir is None:
            self.log("请先选择并运行视频。")
            return
        if self.video_export_running:
            self.log("视频结果正在保存，请稍候。")
            return
        self.video_export_running = True
        self.log("开始保存所有帧组成视频。")

        def worker() -> None:
            try:
                def progress(current, total):
                    self.root.after(0, lambda: self.log(f"正在保存视频结果：{current}/{total or '?'}"))

                view_mode = VIEW_LABEL_TO_MODE[self.view_label_var.get()]
                summary = process_video_to_files(
                    self.selected_video_path,
                    self.current_run_dir,
                    progress_callback=progress,
                    view_mode=view_mode,
                )
                self.root.after(0, lambda: self.log(f"视频结果保存完成：检测 {summary['detected_frames']}/{summary['total_frames']} 帧"))
            except Exception as error:
                message = str(error)
                self.root.after(0, lambda: self.log(f"保存视频失败：{message}"))
            finally:
                self.video_export_running = False

        threading.Thread(target=worker, daemon=True).start()

    def choose_output_root(self) -> None:
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_root = ensure_dir(Path(path))
            self.log(f"输出根目录已设置：{self.output_root}")
            self._update_status_text()

    def open_current_run_dir(self) -> None:
        if self.current_run_dir is None or not self.current_run_dir.exists():
            self.log("当前没有可打开的输出文件夹。")
            return
        os.startfile(self.current_run_dir)

    def on_view_changed(self, _event=None) -> None:
        view_mode = VIEW_LABEL_TO_MODE[self.view_label_var.get()]
        self.robot_view.set_view_mode(view_mode)
        self.log(f"仿真视角已切换：{self.view_label_var.get()}")

    def on_display_options_changed(self) -> None:
        self.robot_view.set_display_options(self.show_axes_var.get(), self.show_labels_var.get())

    def reset_smoother(self) -> None:
        self.smoother = ExponentialSmoother(alpha=0.6)

    def _process_and_show_frame(self, frame_bgr, mirror: bool, use_smoothing: bool | None = None) -> None:
        self.current_frame_bgr = frame_bgr.copy()
        smoother = self.smoother if (self.smooth_var.get() if use_smoothing is None else use_smoothing) else None
        result = process_frame(
            frame_bgr,
            self.detector,
            mirror=mirror,
            smoother=smoother,
            draw_overlay=True,
            draw_status=False,
        )
        self.current_annotated_frame_bgr = result["annotated_frame"]
        self.current_detection = result["detection"]
        self.current_finger_angles = result["finger_angles"]
        if result["qpos"] is not None:
            self.current_qpos = result["qpos"]

        now = time.perf_counter()
        dt = max(now - self.last_frame_time, 1e-6)
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
        self.last_frame_time = now
        self._draw_runtime_overlay(self.current_annotated_frame_bgr, result)

        self.frame_counter += 1
        interval = max(1, int(self.robot_interval_var.get()))
        if result["qpos"] is not None and self.frame_counter % interval == 0:
            self.robot_view.update_qpos(self.current_qpos)

        show_bgr_image_on_label(self.preview_label, self.current_annotated_frame_bgr, max_size=(720, 520))
        self._update_data_panels()
        self._update_status_text(detection_status=self._detection_status(result))

    def _draw_runtime_overlay(self, frame_bgr, result) -> None:
        status = self._overlay_status(result)
        cv2.putText(frame_bgr, f"FPS: {self.fps:.1f}", (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"Status: {status}", (18, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)

    def _overlay_status(self, result=None) -> str:
        """English-only status text for cv2.putText to avoid Chinese mojibake."""
        detection = result["detection"] if result else self.current_detection
        if not detection or not detection.get("has_hand"):
            return "No Hand"
        handedness = detection.get("handedness", "Unknown")
        if handedness == "Right":
            return "Right Hand"
        if handedness == "Left":
            return "Left Hand"
        return "Stable"

    def _detection_status(self, result=None) -> str:
        detection = result["detection"] if result else self.current_detection
        if not detection or not detection.get("has_hand"):
            return "未检测到手"
        handedness = detection.get("handedness", "Unknown")
        if handedness == "Right":
            return "检测到右手"
        if handedness == "Left":
            return "检测到左手"
        return "检测稳定"

    def _update_status_text(self, mode: str | None = None, detection_status: str | None = None) -> None:
        mode = mode or self.current_mode
        detection_status = detection_status or self._detection_status()
        frame_text = f"{self.current_frame_index}/{self.total_video_frames}" if self.current_mode == "视频" else str(self.current_frame_index)
        self.status_var.set(
            f"当前模式：{mode} | 检测状态：{detection_status} | FPS：{self.fps:.1f} | 当前帧：{frame_text} | "
            f"输出根目录：{self.output_root} | 当前输出文件夹：{self.current_run_dir or '未创建'}"
        )

    def _update_data_panels(self) -> None:
        self.finger_text.delete("1.0", tk.END)
        self.finger_text.insert(tk.END, "finger_angles\n")
        if self.current_finger_angles:
            for name in FINGER_ORDER:
                self.finger_text.insert(tk.END, f"{name}: {self.current_finger_angles.get(name, 0.0):.3f}\n")
        else:
            self.finger_text.insert(tk.END, "未检测到手\n")

        self.qpos_text.delete("1.0", tk.END)
        self.qpos_text.insert(tk.END, "qpos\n")
        for name in FINGER_ORDER:
            values = self.current_qpos.get(name, [0.0, 0.0, 0.0])
            self.qpos_text.insert(tk.END, f"{name}: [{values[0]:.3f}, {values[1]:.3f}, {values[2]:.3f}]\n")

    def save_current_full_result(self) -> None:
        if not self.current_detection or not self.current_detection.get("has_hand"):
            self.log("当前没有检测到手，无法保存完整结果。")
            return
        if self.current_run_dir is None:
            self.current_run_dir = self._create_run_dir("manual_run")
        self.save_counter += 1
        index = f"{self.save_counter:04d}"
        try:
            frame_path = save_image(self.current_run_dir, self.current_annotated_frame_bgr, f"frame_{index}.png")
            save_landmarks_json(self.current_run_dir, self.current_detection, f"landmarks_{index}.json", frame_id=self.current_frame_index)
            save_finger_angles_json(self.current_run_dir, self.current_finger_angles, f"finger_angles_{index}.json")
            save_qpos_json(self.current_run_dir, self.current_qpos, f"qpos_{index}.json")
            robot_path = self.robot_view.save_current_image(self.current_run_dir / f"robot_hand_{index}.png")
            self.log(f"已保存当前帧完整结果：{frame_path}")
            self.log(f"机器人手图像：{robot_path}")
        except Exception as error:
            self.log(f"保存失败：{error}")

    def _save_image_run(self, original_image) -> None:
        if self.current_run_dir is None:
            return
        cv2.imwrite(str(self.current_run_dir / "original_image.png"), original_image)
        save_image(self.current_run_dir, self.current_annotated_frame_bgr, "annotated_image.png")
        if self.current_detection and self.current_detection.get("has_hand"):
            save_landmarks_json(self.current_run_dir, self.current_detection, "landmarks.json", frame_id=1)
            save_finger_angles_json(self.current_run_dir, self.current_finger_angles, "finger_angles.json")
            save_qpos_json(self.current_run_dir, self.current_qpos, "qpos.json")
            self.robot_view.update_qpos(self.current_qpos)
            self.robot_view.save_current_image(self.current_run_dir / "robot_hand.png")
        else:
            self.log("未检测到手，仅保存原图和标注图。")

    def _create_run_dir(self, prefix: str) -> Path:
        run_dir = ensure_dir(self.output_root / f"{prefix}_{timestamp_string()}")
        self.log(f"已创建输出文件夹：{run_dir}")
        self._update_status_text()
        return run_dir

    def log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{stamp}] {message}\n")
        self.log_text.see(tk.END)

    def on_close(self) -> None:
        self.stop_camera(log_message=False)
        self.stop_video(log_message=False)
        try:
            self.detector.close()
        except Exception:
            pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        Hand2RobotGUI(root)
        root.mainloop()
    except Exception as error:
        messagebox.showerror("Hand2Robot GUI 错误", str(error))
        raise
