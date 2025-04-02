from pathlib import Path

from PySide6.QtGui import QColor
from tqdm import tqdm
from PySide6.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QPushButton,
    QMessageBox,
    QWidget,
    QListWidget,
    QTextEdit,
    QListWidgetItem,
)

from motion_detector import detect_motion


class MotionDetectorUi(QWidget):
    def __init__(self, folder_path: Path, labels_path: Path):
        super().__init__()

        self.folder_path_ = folder_path
        self.labels_path_ = labels_path
        self.video_files_ = list(self.folder_path_.glob("*.mp4"))

        # Initialize UI elements
        self.init_uI()

    def init_uI(self):
        self.setWindowTitle("Video File Processor")

        # Layout setup
        layout = QVBoxLayout()

        # ListWidget to display video files
        self.file_list = QListWidget()
        layout.addWidget(self.file_list)

        # Button to process missing files
        self.process_button = QPushButton("Process All Missing Files")
        self.process_button.clicked.connect(self.process_missing_files)
        layout.addWidget(self.process_button)

        # Console output for tqdm progress
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        layout.addWidget(self.console_output)

        # Populate the list with files and color them
        self.populate_file_list()

        # Set layout and show window
        self.setLayout(layout)
        self.resize(600, 400)

    def populate_file_list(self):
        for video_file in self.video_files_:
            json_file = video_file.with_suffix(".json")
            item = QListWidgetItem(video_file.name)
            if json_file.exists():
                item.setBackground(QColor("green"))  # Green if .json exists
            else:
                item.setBackground(QColor("red"))  # Red if .json is missing
            self.file_list.addItem(item)

    def process_missing_files(self, labels_path: Path):
        missing_files = [
            video_file
            for video_file in self.video_files_
            if not video_file.with_suffix(".json").exists()
        ]

        if not missing_files:
            QMessageBox.information(
                self,
                "Info",
                "No missing files to process.",
                QMessageBox.StandardButton.Ok,
            )
            return

        self.console_output.clear()
        for video_file in tqdm(missing_files, file=self.get_console_writer()):
            detect_motion(video_file, labels_path)

    def get_console_writer(self):
        # Writer for tqdm to write into the QTextEdit
        class TqdmWriter:
            def __init__(self, text_widget):
                self.text_widget = text_widget

            def write(self, message):
                self.text_widget.append(message)
                QApplication.processEvents()  # Update the UI

            def flush(self):
                pass  # No need to flush for this example

        return TqdmWriter(self.console_output)
