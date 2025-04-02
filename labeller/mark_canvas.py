from PySide6.QtGui import QPainter, QColor, QMouseEvent, Qt, QPen
from PySide6.QtWidgets import QWidget


class MarkCanvas(QWidget):
    def __init__(self, json_data, slider, parent=None):
        super().__init__(parent)
        self.json_data = json_data
        self.slider = slider
        self.setMinimumHeight(5)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)

        pen = QPen(QColor("red"))
        pen.setWidth(6)
        painter.setPen(pen)

        width = self.width()

        for item in self.json_data:
            start_frame = item["start_frames"]
            end_frame = item["end_frames"]

            # Calculate the start and end positions on the canvas
            start_pos = (start_frame / (self.slider.maximum() - 1)) * width
            end_pos = (end_frame / (self.slider.maximum() - 1)) * width

            # Draw a horizontal line from start_pos to end_pos
            painter.drawLine(start_pos, self.height() // 2, end_pos, self.height() // 2)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            x = event.position().x()
            width = self.width()
            frame = int((x / width) * (self.slider.maximum() - 1))
            self.slider.setValue(frame)
