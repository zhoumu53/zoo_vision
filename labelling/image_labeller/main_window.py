import sys
from pathlib import Path
from queue import SimpleQueue

import PySide6.QtGui as QtGui
import numpy as np
import numpy.typing as npt
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QAction, QMouseEvent, QStatusTipEvent
from PySide6.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QPushButton,
    QMessageBox,
    QWidget,
    QHBoxLayout,
    QLabel,
    QMenuBar,
    # QMenu,
)

from project_root import PROJECT_ROOT  # type: ignore
from labelling.common.drawing import draw_clicks
from labelling.common.image_label import ImageLabel
from database import active_db, set_db
from serialization import deserialize_database, serialize_database


class MainWindow(QMainWindow):
    def __init__(self, work_queue: SimpleQueue[int], image_files: list[Path]) -> None:
        super().__init__()

        # Initialize variables
        self.work_queue_ = work_queue
        self.image_: npt.NDArray[np.uint8] | None = None
        self.image_files_ = image_files
        self.current_image_index_ = 0
        self.instance_id_ = 0

        # Load video files
        if not self.image_files_:
            QMessageBox.critical(self, "Error", f"No image files found.")
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

        # Buttons Layout (Horizontal layout for buttons)
        button_layout = QHBoxLayout()

        # Next/Previous Video Buttons
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.prev_image)
        button_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_image)
        button_layout.addWidget(self.next_button)

        left_layout.addLayout(button_layout)

        # Add left layout to the main layout
        main_layout.addLayout(left_layout)

        statusBar = self.statusBar()
        self.statusLabels_ = {}
        self.statusLabels_["sam2"] = QLabel("")
        self.statusLabels_["instance"] = QLabel("0")
        for label in self.statusLabels_.values():
            statusBar.addPermanentWidget(label)

        # Keyboard Shortcuts
        self.addAction(self.create_action("Inst ↑", self.next_instance, Qt.Key.Key_Up))
        self.addAction(
            self.create_action("Inst ↓", self.prev_instance, Qt.Key.Key_Down)
        )
        self.addAction(self.create_action("Next", self.next_image, Qt.Key.Key_Right))
        self.addAction(self.create_action("Previous", self.prev_image, Qt.Key.Key_Left))

        # Load the first video
        self.load_image(self.current_image_index_)

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

    def load_image(self, index: int) -> None:
        if active_db().is_dirty:
            serialize_database()

        image_path = self.image_files_[index]
        print(f"Loading {str(image_path)}")

        # Reset database
        db = deserialize_database(image_path=image_path)
        set_db(db)

        # Submit all frames to background segmenter
        self.work_queue_.put(0)

        # Get the first frame to determine the size
        self.original_width = db.image.shape[1]
        self.original_height = db.image.shape[0]
        print(f"Image size: {(self.original_width, self.original_height)}")

        self.update_window_title()

        # Display image from the database
        self.update_image()

        # Reset instance id
        self.set_instance(0)

    def set_instance(self, id):
        self.instance_id_ = id
        self.statusLabels_["instance"].setText(f"{self.instance_id_}")

    def prev_instance(self):
        if self.instance_id_ > 0:
            self.set_instance(self.instance_id_ - 1)

    def next_instance(self):
        self.set_instance(self.instance_id_ + 1)

    def next_image(self):
        self.current_image_index_ += 1
        if self.current_image_index_ >= len(self.image_files_):
            self.current_image_index_ = 0
        self.load_image(self.current_image_index_)

    def prev_image(self):
        self.current_image_index_ -= 1
        if self.current_image_index_ < 0:
            self.current_image_index_ = len(self.image_files_) - 1
        self.load_image(self.current_image_index_)

    def update_window_title(self):
        self.setWindowTitle(
            f"Image {self.current_image_index_ + 1} out of {len(self.image_files_)} - {str(active_db().image_path.name)}"
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def image_clicked(self, ev: QMouseEvent):
        if ev.button() in [Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton]:
            assert self.image_ is not None
            pixelPos = self.image_label_.event_to_image_position(ev.position())

            db = active_db()
            db.add_point(
                self.instance_id_,
                pixelPos,
                is_positive=ev.button() == Qt.MouseButton.LeftButton,
            )

            # Do a quick draw with only clicks for feedback
            db.segmented_image = draw_clicks(
                db.image.copy(),
                [r.positive_points for r in db.records.values()],
                [r.negative_points for r in db.records.values()],
            )
            self.update_image()

            # Request background processing
            self.work_queue_.put(0)

    def update_image(self) -> None:
        db = active_db()
        self.image_ = db.segmented_image if db.segmented_image is not None else db.image
        self.image_label_.set_image(self.image_)

    def create_menu_bar(self):
        """Create the top menu bar with File and Help menus"""
        menu_bar = QMenuBar(self)

        # File menu
        # file_menu = QMenu("File", self)
        # motion_detection_action = QAction("Motion Detection", self)
        # motion_detection_action.triggered.connect(self.detect_motion)
        # file_menu.addAction(motion_detection_action)

        # Help menu
        # help_menu = QMenu("Help", self)
        # help_action = QAction("Help", self)
        # help_action.triggered.connect(self.open_help_menu)
        # help_menu.addAction(help_action)

        # Add menus to the menu bar
        # menu_bar.addMenu(file_menu)
        # menu_bar.addMenu(help_menu)

        # Set the menu bar for the main window
        self.setMenuBar(menu_bar)
