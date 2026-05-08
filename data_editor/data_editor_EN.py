import os
import re
import sys
import json
import shutil
from collections import defaultdict

from PyQt5.QtCore import Qt, QTimer, QRect, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QBrush
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
    QMessageBox,
    QFrame,
    QFileDialog,
)


class ImageLabel(QLabel):
    """QLabel with automatic image scaling support."""

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: black;
                color: white;
                border: 1px solid #444;
            }
        """)
        self.setMinimumSize(320, 240)
        self._original_pixmap = None
        self._title = title
        self.set_placeholder(f"{title}\nWaiting for playback")

    def set_placeholder(self, text):
        self._original_pixmap = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def set_pixmap(self, pixmap: QPixmap):
        self._original_pixmap = pixmap
        self.setText("")
        self._update_scaled_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self._original_pixmap is not None and not self._original_pixmap.isNull():
            scaled = self._original_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.setPixmap(scaled)
        elif self.text() == "":
            self.setText(f"{self._title}\nNo image")


class RangeSlider(QFrame):
    """
    Progress bar + range selection bar:
    - Left click: jump to the frame
    - Left drag: preview frames in real time
    - Shift + left drag: select a frame range
    """
    rangeChanged = pyqtSignal(int, int)
    frameChanged = pyqtSignal(int)
    sliderPressed = pyqtSignal()
    sliderReleased = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(90)
        self.setStyleSheet("background-color: #f2f2f2; border: 1px solid #bbb;")

        self.total_frames = 0
        self.current_frame = 0

        self.sel_start = 0
        self.sel_end = 0

        self.drag_mode = None   # None / "scrub" / "select"
        self.drag_anchor = 0

    def set_total_frames(self, total_frames):
        self.total_frames = max(0, total_frames)
        if self.total_frames <= 1:
            self.current_frame = 0
            self.sel_start = 0
            self.sel_end = 0
        else:
            max_index = self.total_frames - 1
            self.current_frame = min(self.current_frame, max_index)
            self.sel_start = min(self.sel_start, max_index)
            self.sel_end = min(self.sel_end, max_index)
        self.update()

    def set_current_frame(self, frame_index):
        self.current_frame = max(0, min(frame_index, max(0, self.total_frames - 1)))
        self.update()

    def reset_selection(self):
        if self.total_frames > 0:
            self.sel_start = 0
            self.sel_end = 0
        else:
            self.sel_start = 0
            self.sel_end = 0
        self.rangeChanged.emit(self.sel_start, self.sel_end)
        self.update()

    def get_selected_range(self):
        return min(self.sel_start, self.sel_end), max(self.sel_start, self.sel_end)

    def frame_from_x(self, x):
        if self.total_frames <= 1:
            return 0
        left = 15
        right = self.width() - 15
        width = max(1, right - left)
        x = max(left, min(x, right))
        ratio = (x - left) / width
        frame = int(round(ratio * (self.total_frames - 1)))
        return max(0, min(frame, self.total_frames - 1))

    def x_from_frame(self, frame):
        if self.total_frames <= 1:
            return 15
        left = 15
        right = self.width() - 15
        width = max(1, right - left)
        ratio = frame / (self.total_frames - 1)
        return int(left + ratio * width)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self.total_frames <= 0:
            return

        frame = self.frame_from_x(event.x())
        self.sliderPressed.emit()

        if event.modifiers() & Qt.ShiftModifier:
            self.drag_mode = "select"
            self.drag_anchor = frame
            self.sel_start = frame
            self.sel_end = frame
            self.rangeChanged.emit(*self.get_selected_range())
        else:
            self.drag_mode = "scrub"
            self.current_frame = frame
            self.frameChanged.emit(frame)

        self.update()

    def mouseMoveEvent(self, event):
        if self.drag_mode is None or self.total_frames <= 0:
            return

        frame = self.frame_from_x(event.x())

        if self.drag_mode == "select":
            self.sel_end = frame
            self.rangeChanged.emit(*self.get_selected_range())
        elif self.drag_mode == "scrub":
            self.current_frame = frame
            self.frameChanged.emit(frame)

        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or self.drag_mode is None:
            return

        frame = self.frame_from_x(event.x())

        if self.drag_mode == "select":
            self.sel_end = frame
            self.rangeChanged.emit(*self.get_selected_range())
        elif self.drag_mode == "scrub":
            self.current_frame = frame
            self.frameChanged.emit(frame)
            self.sliderReleased.emit(frame)

        self.drag_mode = None
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        left = 15
        right = self.width() - 15
        bar_y = 36
        bar_h = 12
        bar_w = max(1, right - left)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#cccccc"))
        painter.drawRoundedRect(left, bar_y, bar_w, bar_h, 6, 6)

        if self.total_frames > 0:
            start, end = self.get_selected_range()
            x1 = self.x_from_frame(start)
            x2 = self.x_from_frame(end)
            sel_x = min(x1, x2)
            sel_w = max(6, abs(x2 - x1))
            painter.setBrush(QColor("#6fa8dc"))
            painter.drawRoundedRect(sel_x, bar_y, sel_w, bar_h, 6, 6)

        if self.total_frames > 0:
            cur_x = self.x_from_frame(self.current_frame)
            painter.setPen(QPen(QColor("red"), 2))
            painter.drawLine(cur_x, 16, cur_x, 58)

        if self.total_frames > 0:
            start, end = self.get_selected_range()
            for frame in [start, end]:
                x = self.x_from_frame(frame)
                painter.setPen(QPen(QColor("#1c4587"), 2))
                painter.setBrush(QBrush(QColor("#ffffff")))
                painter.drawEllipse(x - 5, bar_y - 4, 10, 20)

        painter.setPen(QColor("#222222"))
        top_text = "Drag directly: seek playback    Shift+drag: select trim/playback range"
        painter.drawText(QRect(10, 6, self.width() - 20, 20), Qt.AlignCenter, top_text)

        if self.total_frames > 0:
            start, end = self.get_selected_range()
            bottom_text = (
                f"Selected range: {start} - {end}    "
                f"Current frame: {self.current_frame} / {self.total_frames - 1}"
            )
        else:
            bottom_text = "No available frames"
        painter.drawText(QRect(10, 62, self.width() - 20, 20), Qt.AlignCenter, bottom_text)


class DatasetPlayer(QWidget):
    """
    Dataset playback interface:
    - Selectable root_dir
    - Left and right arrows to switch episodes
    - 2x2 multi-camera display in the center
    - Bottom progress bar + range selection + range playback + trimming
    - Delete current dataset
    """

    CAM_LAYOUT = [0, 1, 2, 3]
    FILE_PATTERN = re.compile(r"^(\d+)_color_(\d+)\.jpg$", re.IGNORECASE)

    def __init__(self, root_dir, interval_ms=100):
        super().__init__()
        self.root_dir = root_dir
        self.interval_ms = interval_ms

        self.episodes = []
        self.current_episode_index = 0
        self.current_episode_name = ""
        self.frame_keys = []
        self.frames_map = defaultdict(dict)
        self.current_frame_index = 0

        self.is_playing = True
        self.play_selection_only = False
        self.was_playing_before_drag = False

        self.init_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.play_next_frame)
        self.timer.start(self.interval_ms)

        self.reload_dataset_root(self.root_dir, show_message=False)

    def init_ui(self):
        self.setWindowTitle("Unitree Dataset Editor V1.0")
        self.resize(1450, 1030)

        self.select_root_btn = QPushButton("Select Dataset Path")
        self.select_root_btn.clicked.connect(self.select_root_dir)
        self.select_root_btn.setStyleSheet("""
            QPushButton {
                font-size: 15px;
                font-weight: bold;
                padding: 10px 18px;
                border-radius: 8px;
                background-color: #2e86de;
                color: white;
            }
            QPushButton:hover:!disabled {
                background-color: #1f6fc1;
            }
        """)

        self.root_dir_label = QLabel("")
        self.root_dir_label.setWordWrap(True)
        self.root_dir_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.root_dir_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #222;
                padding: 8px 12px;
                border: 1px solid #cccccc;
                background: #fafafa;
                border-radius: 6px;
            }
        """)

        root_layout = QHBoxLayout()
        root_layout.addWidget(self.select_root_btn)
        root_layout.addWidget(self.root_dir_label, 1)

        self.prev_btn = QPushButton("◀")
        self.next_btn = QPushButton("▶")

        for btn in (self.prev_btn, self.next_btn):
            btn.setFixedWidth(90)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 36px;
                    font-weight: bold;
                    background-color: #2d2d2d;
                    color: white;
                    border-radius: 10px;
                }
                QPushButton:disabled {
                    background-color: #777;
                    color: #bbb;
                }
                QPushButton:hover:!disabled {
                    background-color: #444;
                }
            """)

        self.prev_btn.clicked.connect(self.prev_episode)
        self.next_btn.clicked.connect(self.next_episode)

        self.episode_label = QLabel("Current Episode:")
        self.episode_label.setAlignment(Qt.AlignCenter)
        self.episode_label.setStyleSheet("""
            QLabel {
                font-size: 22px;
                font-weight: bold;
                padding: 8px;
            }
        """)

        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #333;
                padding-bottom: 8px;
            }
        """)

        self.cam_labels = {
            0: ImageLabel("Camera 0"),
            1: ImageLabel("Camera 1"),
            2: ImageLabel("Camera 2"),
            3: ImageLabel("Camera 3"),
        }

        grid = QGridLayout()
        grid.setSpacing(8)
        grid.addWidget(self.cam_labels[0], 0, 0)
        grid.addWidget(self.cam_labels[1], 0, 1)
        grid.addWidget(self.cam_labels[2], 1, 0)
        grid.addWidget(self.cam_labels[3], 1, 1)

        self.range_slider = RangeSlider()
        self.range_slider.rangeChanged.connect(self.on_range_changed)
        self.range_slider.frameChanged.connect(self.on_slider_frame_changed)
        self.range_slider.sliderPressed.connect(self.on_slider_pressed)
        self.range_slider.sliderReleased.connect(self.on_slider_released)

        self.range_info_label = QLabel("No range selected")
        self.range_info_label.setAlignment(Qt.AlignCenter)
        self.range_info_label.setStyleSheet("""
            QLabel {
                font-size: 15px;
                color: #333;
                padding-top: 4px;
                padding-bottom: 4px;
            }
        """)

        self.play_pause_btn = QPushButton("Pause")
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)

        self.play_all_btn = QPushButton("Loop Full Episode")
        self.play_all_btn.clicked.connect(self.set_play_all_mode)

        self.play_selection_btn = QPushButton("Play Selected Range Only")
        self.play_selection_btn.clicked.connect(self.set_play_selection_mode)

        self.reset_range_btn = QPushButton("Reset Range")
        self.reset_range_btn.clicked.connect(self.reset_trim_range)

        self.trim_btn = QPushButton("Trim Selected Range")
        self.trim_btn.clicked.connect(self.trim_selected_frames)

        self.delete_episode_btn = QPushButton("Delete Current Episode")
        self.delete_episode_btn.clicked.connect(self.delete_current_episode)

        for btn in [
            self.play_pause_btn,
            self.play_all_btn,
            self.play_selection_btn,
            self.reset_range_btn,
            self.trim_btn,
            self.delete_episode_btn
        ]:
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 15px;
                    padding: 10px 18px;
                    border-radius: 8px;
                    background-color: #555555;
                    color: white;
                }
                QPushButton:hover:!disabled {
                    background-color: #444444;
                }
                QPushButton:disabled {
                    background-color: #999999;
                }
            """)

        self.trim_btn.setStyleSheet("""
            QPushButton {
                font-size: 15px;
                font-weight: bold;
                padding: 10px 18px;
                border-radius: 8px;
                background-color: #c0392b;
                color: white;
            }
            QPushButton:hover:!disabled {
                background-color: #a93226;
            }
            QPushButton:disabled {
                background-color: #999999;
            }
        """)

        self.delete_episode_btn.setStyleSheet("""
            QPushButton {
                font-size: 15px;
                font-weight: bold;
                padding: 10px 18px;
                border-radius: 8px;
                background-color: #8e44ad;
                color: white;
            }
            QPushButton:hover:!disabled {
                background-color: #7d3c98;
            }
            QPushButton:disabled {
                background-color: #999999;
            }
        """)

        control_layout = QHBoxLayout()
        control_layout.addStretch()
        control_layout.addWidget(self.play_pause_btn)
        control_layout.addWidget(self.play_all_btn)
        control_layout.addWidget(self.play_selection_btn)
        control_layout.addWidget(self.reset_range_btn)
        control_layout.addWidget(self.trim_btn)
        control_layout.addWidget(self.delete_episode_btn)
        control_layout.addStretch()

        center_layout = QVBoxLayout()
        center_layout.addLayout(root_layout)
        center_layout.addWidget(self.episode_label)
        center_layout.addWidget(self.info_label)
        center_layout.addLayout(grid)
        center_layout.addSpacing(12)
        center_layout.addWidget(self.range_slider)
        center_layout.addWidget(self.range_info_label)
        center_layout.addLayout(control_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.prev_btn)
        main_layout.addLayout(center_layout, 1)
        main_layout.addWidget(self.next_btn)

        self.setLayout(main_layout)

    def update_root_dir_label(self):
        self.root_dir_label.setText(f"Current Dataset Path: {self.root_dir}")

    def clear_player_state(self, message="No available data"):
        self.episodes = []
        self.current_episode_index = 0
        self.current_episode_name = ""
        self.frame_keys = []
        self.frames_map = defaultdict(dict)
        self.current_frame_index = 0
        self.play_selection_only = False

        self.episode_label.setText("Current Episode: None")
        self.info_label.setText(message)

        for cam_id, label in self.cam_labels.items():
            label.set_placeholder(f"Camera {cam_id}\nNo image")

        self.range_slider.set_total_frames(0)
        self.range_info_label.setText(message)
        self.trim_btn.setEnabled(False)
        self.delete_episode_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

    def find_episodes(self):
        episode_dirs = []
        if not os.path.isdir(self.root_dir):
            return episode_dirs

        for name in os.listdir(self.root_dir):
            full_path = os.path.join(self.root_dir, name)
            if os.path.isdir(full_path) and re.match(r"^episode_(\d+)$", name):
                episode_dirs.append(name)

        episode_dirs.sort(key=lambda x: int(re.match(r"^episode_(\d+)$", x).group(1)))
        return episode_dirs

    def reload_dataset_root(self, new_root_dir, show_message=True):
        if not new_root_dir or not os.path.isdir(new_root_dir):
            return

        self.is_playing = False
        self.update_play_button_text()

        self.root_dir = new_root_dir
        self.update_root_dir_label()

        self.episodes = self.find_episodes()

        if not self.episodes:
            self.clear_player_state("No episode_0001 / episode_0002 ... found in the current path")
            if show_message:
                QMessageBox.warning(
                    self,
                    "Warning",
                    f"No episode_XXXX dataset directories found in:\n{self.root_dir}"
                )
            return

        self.current_episode_index = 0
        self.load_episode(self.current_episode_index)
        self.is_playing = True
        self.update_play_button_text()
        self.update_range_info()

        if show_message:
            QMessageBox.information(
                self,
                "Done",
                f"Dataset path switched to:\n{self.root_dir}"
            )

    def select_root_dir(self):
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Root Directory",
            self.root_dir if os.path.isdir(self.root_dir) else os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if selected_dir:
            self.reload_dataset_root(selected_dir, show_message=False)

    def update_nav_buttons(self):
        if not self.episodes:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            return
        self.prev_btn.setEnabled(self.current_episode_index > 0)
        self.next_btn.setEnabled(self.current_episode_index < len(self.episodes) - 1)

    def update_play_button_text(self):
        self.play_pause_btn.setText("Pause" if self.is_playing else "Start")

    def update_range_info(self):
        if not self.frame_keys:
            self.range_info_label.setText("No available frames")
            return

        start, end = self.range_slider.get_selected_range()
        mode_text = "Play selected range only" if self.play_selection_only else "Loop full episode"
        state_text = "Playing" if self.is_playing else "Paused"
        self.range_info_label.setText(
            f"Mode: {mode_text}    State: {state_text}    Selected range: frame {start} ~ frame {end}"
        )

    def on_range_changed(self, start, end):
        self.trim_btn.setEnabled(len(self.frame_keys) > 0 and end >= start)
        self.update_range_info()

    def on_slider_pressed(self):
        self.was_playing_before_drag = self.is_playing
        self.is_playing = False
        self.update_play_button_text()
        self.update_range_info()

    def on_slider_released(self, frame_index):
        self.current_frame_index = frame_index
        if self.was_playing_before_drag:
            self.is_playing = True
        self.update_play_button_text()
        self.update_range_info()

    def on_slider_frame_changed(self, frame_index):
        if not self.frame_keys:
            return
        self.current_frame_index = max(0, min(frame_index, len(self.frame_keys) - 1))
        self.show_frame(self.current_frame_index)

    def toggle_play_pause(self):
        if not self.frame_keys:
            return
        self.is_playing = not self.is_playing
        self.update_play_button_text()
        self.update_range_info()

    def set_play_all_mode(self):
        self.play_selection_only = False
        self.update_range_info()

    def set_play_selection_mode(self):
        if not self.frame_keys:
            return

        start, end = self.range_slider.get_selected_range()
        if start == end:
            QMessageBox.information(
                self,
                "Info",
                "The selected range currently contains only 1 frame. It can still be played. To select a longer range, hold Shift and drag on the progress bar."
            )

        self.play_selection_only = True
        self.current_frame_index = start
        self.show_frame(self.current_frame_index)
        self.update_range_info()

    def reset_trim_range(self):
        self.range_slider.reset_selection()
        self.update_range_info()

    def load_episode(self, episode_index):
        if not self.episodes:
            self.clear_player_state("No available datasets")
            return

        self.current_episode_index = episode_index
        self.current_episode_name = self.episodes[episode_index]
        colors_dir = os.path.join(self.root_dir, self.current_episode_name, "colors")

        if not os.path.isdir(colors_dir):
            self.frames_map = defaultdict(dict)
            self.frame_keys = []
            self.current_frame_index = 0
            self.episode_label.setText(f"Current Episode: {self.current_episode_name}")
            self.info_label.setText(f"Directory not found: {colors_dir}")
            for cam_id, label in self.cam_labels.items():
                label.set_placeholder(f"Camera {cam_id}\ncolors directory not found")
            self.range_slider.set_total_frames(0)
            self.range_info_label.setText("colors directory not found")
            self.trim_btn.setEnabled(False)
            self.delete_episode_btn.setEnabled(True)
            self.update_nav_buttons()
            return

        frames_map = defaultdict(dict)

        for filename in os.listdir(colors_dir):
            match = self.FILE_PATTERN.match(filename)
            if not match:
                continue

            frame_id = int(match.group(1))
            cam_id = int(match.group(2))

            if cam_id not in self.CAM_LAYOUT:
                continue

            full_path = os.path.join(colors_dir, filename)
            frames_map[frame_id][cam_id] = full_path

        frame_keys = sorted(frames_map.keys())

        self.frames_map = frames_map
        self.frame_keys = frame_keys
        self.current_frame_index = 0
        self.play_selection_only = False

        self.episode_label.setText(f"Current Episode: {self.current_episode_name}")

        if not self.frame_keys:
            self.info_label.setText("No playable frames found in the current dataset")
            for cam_id, label in self.cam_labels.items():
                label.set_placeholder(f"Camera {cam_id}\nNo image")
            self.range_slider.set_total_frames(0)
            self.range_info_label.setText("No available frames")
            self.trim_btn.setEnabled(False)
        else:
            self.range_slider.set_total_frames(len(self.frame_keys))
            self.range_slider.reset_selection()
            self.show_frame(self.current_frame_index)
            self.trim_btn.setEnabled(True)

        self.delete_episode_btn.setEnabled(True)
        self.update_nav_buttons()
        self.update_play_button_text()
        self.update_range_info()

    def show_frame(self, frame_index):
        if not self.frame_keys:
            return

        frame_index = max(0, min(frame_index, len(self.frame_keys) - 1))
        frame_id = self.frame_keys[frame_index]
        cam_file_map = self.frames_map[frame_id]

        mode_text = "Selected range loop" if self.play_selection_only else "Full episode loop"
        state_text = "Playing" if self.is_playing else "Paused"

        self.info_label.setText(
            f"Dataset: {self.current_episode_name}    "
            f"Current frame: {frame_index}/{len(self.frame_keys) - 1}    "
            f"Frame ID: {frame_id:06d}    "
            f"Mode: {mode_text}    State: {state_text}"
        )

        for cam_id in self.CAM_LAYOUT:
            label = self.cam_labels[cam_id]
            image_path = cam_file_map.get(cam_id)

            if image_path and os.path.isfile(image_path):
                pixmap = QPixmap(image_path)
                if pixmap.isNull():
                    label.set_placeholder(f"Camera {cam_id}\nFailed to read image")
                else:
                    label.set_pixmap(pixmap)
            else:
                label.set_placeholder(f"Camera {cam_id}\nMissing frame")

        self.range_slider.set_current_frame(frame_index)

    def play_next_frame(self):
        if not self.frame_keys or not self.is_playing:
            return

        if self.play_selection_only:
            start, end = self.range_slider.get_selected_range()
            start = max(0, min(start, len(self.frame_keys) - 1))
            end = max(0, min(end, len(self.frame_keys) - 1))

            if self.current_frame_index < start or self.current_frame_index > end:
                self.current_frame_index = start

            self.show_frame(self.current_frame_index)
            self.current_frame_index += 1

            if self.current_frame_index > end:
                self.current_frame_index = start
        else:
            self.show_frame(self.current_frame_index)
            self.current_frame_index += 1

            if self.current_frame_index >= len(self.frame_keys):
                self.current_frame_index = 0

    def prev_episode(self):
        if self.current_episode_index > 0:
            self.load_episode(self.current_episode_index - 1)

    def next_episode(self):
        if self.current_episode_index < len(self.episodes) - 1:
            self.load_episode(self.current_episode_index + 1)

    def trim_selected_frames(self):
        if not self.frame_keys:
            QMessageBox.warning(self, "Warning", "There are no frames that can be trimmed in the current episode.")
            return

        start_idx, end_idx = self.range_slider.get_selected_range()

        if start_idx < 0 or end_idx >= len(self.frame_keys):
            QMessageBox.warning(self, "Warning", "Invalid selected range.")
            return

        delete_frame_ids = self.frame_keys[start_idx:end_idx + 1]
        delete_count = len(delete_frame_ids)
        remain_count = len(self.frame_keys) - delete_count

        if remain_count <= 0:
            QMessageBox.warning(
                self,
                "Warning",
                "You cannot trim all frames from the current episode. At least 1 frame must remain."
            )
            return

        reply = QMessageBox.question(
            self,
            "Confirm Trim",
            (
                f"Current dataset: {self.current_episode_name}\n"
                f"Frames to delete: {start_idx} ~ {end_idx} (total {delete_count} frames)\n"
                f"{remain_count} frames will remain after deletion and be renumbered.\n\n"
                f"This operation will directly modify files on disk. Continue?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self.is_playing = False
        self.update_play_button_text()

        try:
            self.delete_and_renumber_frames(delete_frame_ids)
            self.load_episode(self.current_episode_index)
            QMessageBox.information(
                self,
                "Done",
                f"Trim completed.\n{delete_count} frames were deleted and the remaining frames were renumbered."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Trim failed:\n{str(e)}")

    def delete_current_episode(self):
        if not self.episodes or not self.current_episode_name:
            QMessageBox.warning(self, "Warning", "There is no dataset to delete.")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            "This operation will delete the current dataset!",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel
        )

        if reply != QMessageBox.Yes:
            return

        self.is_playing = False
        self.update_play_button_text()

        episode_path = os.path.join(self.root_dir, self.current_episode_name)
        current_index = self.current_episode_index

        try:
            if not os.path.isdir(episode_path):
                raise RuntimeError(f"Dataset directory does not exist: {episode_path}")

            shutil.rmtree(episode_path)

            self.episodes = self.find_episodes()

            if not self.episodes:
                self.clear_player_state("No datasets remain in the current path")
                QMessageBox.information(self, "Done", "The current dataset has been deleted.")
                return

            next_index = min(current_index, len(self.episodes) - 1)
            self.load_episode(next_index)
            self.is_playing = True
            self.update_play_button_text()
            self.update_range_info()

            QMessageBox.information(self, "Done", "The current dataset has been deleted.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete the current dataset:\n{str(e)}")

    def delete_and_renumber_frames(self, delete_frame_ids):
        colors_dir = os.path.join(self.root_dir, self.current_episode_name, "colors")
        json_path = os.path.join(self.root_dir, self.current_episode_name, "data.json")

        if not os.path.isdir(colors_dir):
            raise RuntimeError(f"colors directory does not exist: {colors_dir}")

        if not os.path.isfile(json_path):
            raise RuntimeError(f"data.json does not exist: {json_path}")

        delete_frame_set = set(delete_frame_ids)

        all_files = []
        for filename in os.listdir(colors_dir):
            match = self.FILE_PATTERN.match(filename)
            if not match:
                continue

            frame_id = int(match.group(1))
            cam_id = int(match.group(2))
            old_path = os.path.join(colors_dir, filename)
            all_files.append((frame_id, cam_id, old_path, filename))

        if not all_files:
            raise RuntimeError("No valid jpg frame files were found in the colors directory.")

        for frame_id, cam_id, old_path, filename in all_files:
            if frame_id in delete_frame_set and os.path.exists(old_path):
                os.remove(old_path)

        remaining = []
        for filename in os.listdir(colors_dir):
            match = self.FILE_PATTERN.match(filename)
            if not match:
                continue

            frame_id = int(match.group(1))
            cam_id = int(match.group(2))
            old_path = os.path.join(colors_dir, filename)
            remaining.append((frame_id, cam_id, old_path, filename))

        if not remaining:
            raise RuntimeError("No image files remain after trimming.")

        grouped = defaultdict(list)
        for frame_id, cam_id, old_path, filename in remaining:
            grouped[frame_id].append((cam_id, old_path, filename))

        old_frame_ids = sorted(grouped.keys())

        new_id_map = {
            old_frame_id: new_frame_id
            for new_frame_id, old_frame_id in enumerate(old_frame_ids)
        }

        temp_records = []

        for old_frame_id in old_frame_ids:
            for cam_id, old_path, old_filename in grouped[old_frame_id]:
                temp_name = f"__tmp__{old_frame_id:06d}_color_{cam_id}.jpg"
                temp_path = os.path.join(colors_dir, temp_name)
                os.rename(old_path, temp_path)
                temp_records.append((old_frame_id, cam_id, temp_path))

        for old_frame_id, cam_id, temp_path in temp_records:
            new_frame_id = new_id_map[old_frame_id]
            new_name = f"{new_frame_id:06d}_color_{cam_id}.jpg"
            new_path = os.path.join(colors_dir, new_name)
            os.rename(temp_path, new_path)

        with open(json_path, "r", encoding="utf-8") as f:
            json_obj = json.load(f)

        if "data" not in json_obj or not isinstance(json_obj["data"], list):
            raise RuntimeError("Invalid data.json format: missing data array.")

        old_data_list = json_obj["data"]
        new_data_list = []

        for item in old_data_list:
            if not isinstance(item, dict):
                continue

            old_idx = item.get("idx", None)
            if old_idx is None:
                continue

            if old_idx in delete_frame_set:
                continue

            if old_idx in new_id_map:
                new_idx = new_id_map[old_idx]
                new_item = json.loads(json.dumps(item, ensure_ascii=False))
                new_item["idx"] = new_idx

                if isinstance(new_item.get("colors"), dict):
                    new_colors = {}
                    for color_key, color_path in new_item["colors"].items():
                        new_colors[color_key] = f"colors/{new_idx:06d}_{color_key}.jpg"
                    new_item["colors"] = new_colors

                if isinstance(new_item.get("depths"), dict):
                    new_depths = {}
                    for depth_key, depth_path in new_item["depths"].items():
                        new_depths[depth_key] = f"depths/{new_idx:06d}_{depth_key}.jpg"
                    new_item["depths"] = new_depths

                if isinstance(new_item.get("audios"), dict):
                    new_audios = {}
                    for audio_key, audio_path in new_item["audios"].items():
                        new_audios[audio_key] = f"audios/audio_{new_idx:06d}_{audio_key}.npy"
                    new_item["audios"] = new_audios

                new_data_list.append(new_item)

        if not new_data_list:
            raise RuntimeError("No frame data remains in data.json after trimming.")

        json_obj["data"] = new_data_list

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_obj, f, ensure_ascii=False, indent=4)


def main():
    app = QApplication(sys.argv)

    root_dir = ""

    player = DatasetPlayer(root_dir=root_dir, interval_ms=100)
    player.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()