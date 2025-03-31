import datetime
import json
import sys
import time
from pathlib import Path
from queue import SimpleQueue

import PySide6.QtGui as QtGui
import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QEvent
from PySide6.QtGui import QAction, QMouseEvent, QStatusTipEvent
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QPushButton,
    QMessageBox,
    QWidget,
    QHBoxLayout,
    QLabel,
    QMenuBar,
    QMenu,
)

from drawing import draw_clicks
from database import active_db, set_db, DatabaseFrame
from image_label import ImageLabel
from mark_canvas import MarkCanvas
from serialization import deserialize_database, serialize_database
from side_menu import SideMenu
from help import HelpMenu
from motion_detector_ui import MotionDetectorUi
from utils import pretty_time_delta


class MainWindow(QMainWindow):
    def __init__(
        self, work_queue: SimpleQueue, videos_path: Path, labels_path: Path
    ) -> None:
        super().__init__()

        # Initialize variables
        self.work_queue_ = work_queue
        self.video_reader_ = None
        self.video_fps_ = 1
        self.last_advance_time_ms = 0
        self.image_: np.ndarray | None = None
        self.timer_ = QTimer()
        self.timer_.setInterval(int(1000 / 240))  # More or less 10x per frame ~240fps
        self.timer_.timeout.connect(self.advance_frame)
        self.frame_index_ = 0
        self.playback_speed_: int = 0  # Playback speed multiplier
        self.videos_path = videos_path
        self.labels_path = labels_path
        self.current_video_index_ = 0
        self.frame_count_ = 0

        # Load video files
        self.video_files_ = list(self.videos_path.glob("**/*.mp4"))
        if not self.video_files_:
            QMessageBox.critical(
                self, "Error", f"No .mp4 files found in the {str(self.videos_path)}."
            )
            sys.exit()

        self.create_menu_bar()

        # Create the central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main layout (Horizontal)
        main_layout = QHBoxLayout(self.central_widget)

        # Left side layout (70% of the width)
        left_layout = QVBoxLayout()

        # Video display
        self.image_label_ = ImageLabel()
        self.image_label_.mousePressEvent = self.image_clicked
        left_layout.addWidget(self.image_label_, alignment=Qt.AlignmentFlag.AlignLeft)

        # Scrubber (QSlider)
        self.position_slider_ = QSlider(Qt.Horizontal)  # type:ignore
        self.position_slider_.setTickPosition(QSlider.TicksBelow)  # type:ignore
        self.position_slider_.setMinimum(0)
        self.position_slider_.setMaximum(100)
        self.position_slider_.setValue(0)
        self.position_slider_.valueChanged.connect(self.set_position)

        # Buttons Layout (Horizontal layout for buttons)
        button_layout = QHBoxLayout()

        # Play/Pause Button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play_pause)
        button_layout.addWidget(self.play_button)

        # Next/Previous Video Buttons
        self.next_button = QPushButton("Next Video")
        self.next_button.clicked.connect(self.next_video)
        button_layout.addWidget(self.next_button)

        self.prev_button = QPushButton("Previous Video")
        self.prev_button.clicked.connect(self.prev_video)
        button_layout.addWidget(self.prev_button)

        # Add MarkCanvas to the left layout
        self.mark_canvas_ = MarkCanvas([], self.position_slider_)

        left_layout.addWidget(self.mark_canvas_)
        left_layout.addWidget(self.position_slider_)
        left_layout.addLayout(button_layout)

        # Add left layout to the main layout
        main_layout.addLayout(left_layout, 7)  # 70% width

        # Right side layout (30% of the width)
        right_layout = QVBoxLayout()

        # JSON content display
        self.side_menu = SideMenu(self.position_slider_, self.work_queue_)
        right_layout.addWidget(self.side_menu)

        # Add right layout to the main layout
        main_layout.addLayout(right_layout, 3)  # 30% width

        statusBar = self.statusBar()
        self.statusLabels_ = {}
        self.statusLabels_["sam2"] = QLabel("")
        self.statusLabels_["video_speed"] = QLabel("0x")
        self.statusLabels_["video_time"] = QLabel("0s")
        for label in self.statusLabels_.values():
            statusBar.addPermanentWidget(label)

        # Keyboard Shortcuts
        self.addAction(
            self.create_action("Play/Pause", self.toggle_play_pause, "Space")
        )
        self.addAction(
            self.create_action("Speed →", self.increase_speed, Qt.Key.Key_Right)
        )
        self.addAction(
            self.create_action("Speed ←", self.decrease_speed, Qt.Key.Key_Left)
        )
        self.addAction(
            self.create_action("Name ↑", self.side_menu.prev_name, Qt.Key.Key_Up)
        )
        self.addAction(
            self.create_action("Name ↓", self.side_menu.next_name, Qt.Key.Key_Down)
        )

        # Load the first video
        self.load_video(self.current_video_index_)

    def create_action(self, name, func, shortcut):
        action = QAction(name, self)
        action.triggered.connect(func)
        action.setShortcut(shortcut)
        return action

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if active_db().is_dirty:
            serialize_database()

    def event(self, event: QEvent) -> bool:
        if isinstance(event, QStatusTipEvent):
            msg = event.tip()
            parts = msg.split(":")
            if len(parts) == 2:
                key = parts[0]
                assert key in self.statusLabels_
                self.statusLabels_[key].setText(parts[1])
            else:
                self.statusBar().showMessage(msg)
            return True
        return super().event(event)

    def load_video(self, index: int) -> None:
        if active_db().is_dirty:
            serialize_database()

        video_path = self.video_files_[index]
        print(f"Loading {str(video_path)}")

        # Use decord for video reading
        self.video_reader_ = cv2.VideoCapture(str(video_path))
        self.frame_count_ = int(self.video_reader_.get(cv2.CAP_PROP_FRAME_COUNT))

        # Reset database
        set_db(
            deserialize_database(
                video_path=video_path,
                labels_path=self.labels_path,
                video_reader=self.video_reader_,
            )
        )

        # Submit all frames to background segmenter
        for frame in active_db().frames.values():
            self.work_queue_.put(frame.frame)

        # Set the timer interval based on the framerate
        self.video_fps_ = self.video_reader_.get(cv2.CAP_PROP_FPS)
        if self.video_fps_ < 1:
            self.video_fps_ = 24  # Assume if invalid
        print(f"FPS: {self.video_fps_}")

        # Get the first frame to determine the size
        self.original_width = self.video_reader_.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.original_height = self.video_reader_.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Image size: {(self.original_width, self.original_height)}")

        self.position_slider_.setMaximum(self.frame_count_ - 1)
        self.display_image_by_index(0)

        json_file_path_for_movement = (
            self.labels_path / video_path.with_suffix(".json").name
        )

        self.side_menu.display_records()

        try:
            with json_file_path_for_movement.open("r") as f:
                json_data = json.load(f)
            self.mark_canvas_.json_data = json_data
            self.mark_canvas_.update()  # Trigger a repaint
        except Exception as e:
            print(f"Error: Failed to load movment JSON file: {e}")

        self.update_window_title()

    def next_video(self):
        self.current_video_index_ += 1
        if self.current_video_index_ >= len(self.video_files_):
            self.current_video_index_ = 0
        self.load_video(self.current_video_index_)

    def prev_video(self):
        self.current_video_index_ -= 1
        if self.current_video_index_ < 0:
            self.current_video_index_ = len(self.video_files_) - 1
        self.load_video(self.current_video_index_)

    def update_window_title(self):
        self.setWindowTitle(
            f"Video {self.current_video_index_ + 1} out of {len(self.video_files_)} - {str(active_db().video_path.name)}"
        )

    def toggle_play_pause(self):
        if self.playback_speed_ == 0:
            self.set_play_speed(1)  # Always start playing forward
        else:
            self.set_play_speed(0)

    def ensure_playing(self):
        self.last_advance_time_ms = 0
        self.timer_.start()
        self.play_button.setText("Pause")

    def ensure_stopped(self):
        self.timer_.stop()
        self.play_button.setText("Play")

    def set_play_speed(self, speed: int) -> None:
        QApplication.sendEvent(
            self,
            QStatusTipEvent(f"video_speed:{speed}x"),
        )
        self.playback_speed_ = speed

        if speed == 0:
            self.ensure_stopped()
        else:
            self.ensure_playing()

    def increase_speed(self):
        if self.playback_speed_ == 0:
            self.set_play_speed(1)
        elif self.playback_speed_ > 0:
            self.set_play_speed(self.playback_speed_ + 1)
        else:
            self.set_play_speed(0)

    def decrease_speed(self):
        if self.playback_speed_ == 0:
            self.set_play_speed(-1)
        elif self.playback_speed_ < 0:
            self.set_play_speed(self.playback_speed_ - 1)
        else:
            self.set_play_speed(0)

    def set_position(self, position):
        if self.video_reader_:
            self.display_image_by_index(position)

    def advance_frame(self):
        now_ms = time.time() * 1000
        if self.last_advance_time_ms == 0:
            self.last_advance_time_ms = now_ms - 1

        duration_s = (now_ms - self.last_advance_time_ms) / 1000

        time_factor = self.video_fps_ * duration_s
        delta_frame = int(self.playback_speed_ * time_factor)
        if delta_frame != 0:
            new_frame_index = self.frame_index_ + delta_frame
            self.display_image_by_index(new_frame_index)
            self.last_advance_time_ms += (
                delta_frame / self.video_fps_ * 1000 / self.playback_speed_
            )

    def display_image_by_index(self, index: int):
        if not self.video_reader_:
            return

        if index < 0:
            self.set_play_speed(0)
            index = 0
        elif index >= self.frame_count_:
            self.set_play_speed(0)
            index = self.frame_count_ - 1

        self.frame_index_ = index
        db_frame = active_db().frames.get(self.frame_index_)
        if db_frame is not None:
            # Display image from the database
            self.image_ = (
                db_frame.segmented_image
                if db_frame.segmented_image is not None
                else db_frame.original_image
            )
        else:
            # Nothing in db, just display raw from video
            self.video_reader_.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index_)
            ret, cv_frame = self.video_reader_.read()
            if not ret:
                return
            self.image_ = cv_frame
        assert self.image_ is not None
        self.image_label_.set_image(self.image_)
        self.position_slider_.setValue(self.frame_index_)

        video_time = datetime.timedelta(seconds=self.frame_index_ / self.video_fps_)
        QApplication.sendEvent(
            self,
            QStatusTipEvent(
                f"video_time:frame {self.frame_index_} ({pretty_time_delta(video_time.total_seconds())})"
            ),
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def image_clicked(self, ev: QMouseEvent):
        if ev.button() in [Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton]:
            assert self.image_ is not None
            if self.timer_.isActive():
                self.toggle_play_pause()

            pixelPos = self.image_label_.event_to_image_position(ev.position())

            active_db().add_point(
                self.frame_index_,
                self.side_menu.get_selected_name(),
                pixelPos,
                is_positive=ev.button() == Qt.MouseButton.LeftButton,
                original_image=self.image_,
            )
            active_db().is_dirty = True

            # Do a quick draw with only clicks for feedback
            frame = active_db().frames[self.frame_index_]
            frame.segmented_image = None
            draw_clicks(frame)
            self.update_ui(self.frame_index_)

            # Request background processing
            self.work_queue_.put(self.frame_index_)
            self.side_menu.on_database_changed()

    def update_ui(self, frame_index: int) -> None:
        if self.frame_index_ == frame_index:
            frame = active_db().frames.get(frame_index)
            if frame is None:
                # Frame was deleted, show from video
                self.display_image_by_index(frame_index)
            elif frame.segmented_image is None:
                # No segmentation, show cached original image
                self.image_label_.set_image(frame.original_image)
            else:
                # Show segmenation result
                self.image_label_.set_image(frame.segmented_image)

    def create_menu_bar(self):
        """Create the top menu bar with File and Help menus"""
        menu_bar = QMenuBar(self)

        # File menu
        file_menu = QMenu("File", self)
        motion_detection_action = QAction("Motion Detection", self)
        motion_detection_action.triggered.connect(self.detect_motion)
        file_menu.addAction(motion_detection_action)

        # Help menu
        help_menu = QMenu("Help", self)
        help_action = QAction("Help", self)
        help_action.triggered.connect(self.open_help_menu)
        help_menu.addAction(help_action)

        # Add menus to the menu bar
        menu_bar.addMenu(file_menu)
        menu_bar.addMenu(help_menu)

        # Set the menu bar for the main window
        self.setMenuBar(menu_bar)

    def detect_motion(self):
        self.motion_detection_ui = MotionDetectorUi(self.videos_path, self.labels_path)
        self.motion_detection_ui.show()

    def open_help_menu(self):
        """Open the Help menu dialog"""
        help_dialog = HelpMenu()
        help_dialog.exec()
