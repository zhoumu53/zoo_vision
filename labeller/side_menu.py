import json
import sys
from pathlib import Path
from queue import SimpleQueue
from typing import cast

import PySide6.QtWidgets as QtWidgets
from PySide6.QtWidgets import (
    QVBoxLayout,
    QComboBox,
    QLabel,
    QListWidget,
    QWidget,
    QHBoxLayout,
    QPushButton,
    QListWidgetItem,
    QSpinBox,
)

from database import active_db, Record
from serialization import serialize_database


class RecordItemWidget(QWidget):
    def __init__(self, record: Record):
        super().__init__()
        self.record = record

        item_layout = QHBoxLayout()

        # Display the record info
        label = QLabel(
            f"{record.name} at {record.frame} +{record.positive_points.shape[0]} -{record.negative_points.shape[0]}"
        )
        item_layout.addWidget(label)

        # Add a "Delete" button for each record
        self.delete_button = QPushButton("Delete")
        item_layout.addWidget(self.delete_button)

        # Set the layout to the custom widget
        self.setLayout(item_layout)


class SideMenu(QWidget):
    def __init__(self, slider, work_queue: SimpleQueue):
        super().__init__()

        self.slider = slider
        self.work_queue_ = work_queue

        # Main Layout
        self.file_name = None
        layout = QVBoxLayout()

        # Instance id
        self.instance_id_box = QSpinBox()
        layout.addWidget(QLabel("Instance id:"))
        layout.addWidget(self.instance_id_box)

        # Dropdown for names
        self.name_dropdown = QComboBox()
        self.load_names()  # Load names from the JSON file
        layout.addWidget(QLabel("Select Name:"))
        layout.addWidget(self.name_dropdown)

        # Save status indicator
        self.save_status_label = QLabel("")

        # List widget for records
        self.record_list = QListWidget()
        self.record_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.record_list.itemClicked.connect(self.display_item)
        records_label_layout = QHBoxLayout()
        records_label = QLabel("Records List:")
        records_label_layout.addWidget(records_label)
        records_label_layout.addWidget(self.save_status_label)
        layout.addLayout(records_label_layout)
        layout.addWidget(self.record_list)

        # Save button at the bottom
        self.save_button = QPushButton("Save Records")
        self.save_button.clicked.connect(self.save_records)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

        self.update_save_status()  # Update the save status indicator

    def load_names(self) -> None:
        path = Path("names.json")
        try:
            with path.open("r") as file:
                names = json.load(file)
            self.name_dropdown.addItems(names)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to load names from {str(path)}: \n{e}"
            )
            sys.exit()

    def next_name(self) -> None:
        index = self.name_dropdown.currentIndex() + 1
        if index < self.name_dropdown.count():
            self.name_dropdown.setCurrentIndex(index)

    def prev_name(self) -> None:
        index = self.name_dropdown.currentIndex() - 1
        if index >= 0:
            self.name_dropdown.setCurrentIndex(index)

    def display_records(self):
        self.record_list.clear()

        # Display each record as a list item with "Display" and "Delete" buttons
        for frame in active_db().frames.values():
            for record in frame.records.values():
                # Create a custom widget for each list item
                item_widget = RecordItemWidget(record)
                item_widget.delete_button.clicked.connect(
                    lambda _, r=record: self.delete_record(r)
                )

                # Create a QListWidgetItem
                list_item = QListWidgetItem(self.record_list)
                list_item.setSizeHint(item_widget.sizeHint())

                # Add the custom widget to the QListWidget
                self.record_list.addItem(list_item)
                self.record_list.setItemWidget(list_item, item_widget)

    def display_item(self, list_item: QListWidgetItem) -> None:
        item = cast(RecordItemWidget, self.record_list.itemWidget(list_item))
        record: Record = item.record
        self.instance_id_box.setValue(record.instance_id)
        # print(record)
        self.slider.setValue(record.frame)

    def delete_record(self, record: Record) -> None:
        # Remove the record from the list and refresh the UI
        frame_data = active_db().frames[record.frame]
        frame_data.records.pop(record.instance_id)
        frame_data.segmented_image = None

        if len(frame_data.records) == 0:
            active_db().frames.pop(record.frame)

        active_db().is_dirty = True
        # Request background processing
        self.work_queue_.put(record.frame)

        self.display_records()
        self.update_save_status()

    def get_selected_name(self):
        return self.name_dropdown.currentText()

    def on_database_changed(self) -> None:
        self.display_records()
        self.update_save_status()

    def update_save_status(self):
        if active_db().is_dirty:
            self.save_status_label.setText("Unsaved Changes")
            self.save_status_label.setStyleSheet("color: red;")
        else:
            self.save_status_label.setText("All changes saved")
            self.save_status_label.setStyleSheet("color: green;")

    def save_records(self):
        serialize_database()
        self.update_save_status()
