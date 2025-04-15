from PySide6.QtCore import QPointF
import numpy as np
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QLabel,
)


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()

        self.image_: np.ndarray | None = None
        self.qimage_: QImage | None = None
        self.aspect_ = 1.0

        self.setScaledContents(True)

    def set_image(self, image: np.ndarray):
        self.image_ = image

        # Ensure the image is in the correct format (BGR888 or RGB888)
        height = image.shape[0]
        width = image.shape[1]
        assert image.shape == (height, width, 3)

        # Convert NumPy array to bytes
        image_bytes = image.tobytes()

        # Create QImage from the byte array
        qimage = QImage(
            image_bytes,
            width,
            height,
            image.strides[0],  # bytes per line
            QImage.Format.Format_BGR888,
        )
        self.qimage_ = qimage

        self.aspect_ = width / height

        # Create a QPixmap from the QImage
        pixmap = QPixmap.fromImage(qimage)
        self.setPixmap(pixmap)
        self.setMinimumSize(1, 1)
        self.update_margins()

    def resizeEvent(self, event: QResizeEvent) -> None:
        self.update_margins()
        return super().resizeEvent(event)

    def update_margins(self) -> None:
        width = self.width()
        height = self.height()
        label_aspect = width / height

        if label_aspect > self.aspect_:
            hmargin = int((width - height * self.aspect_) / 2)
            vmargin = 0
        else:
            hmargin = 0
            vmargin = int((height - width / self.aspect_) / 2)
        self.setContentsMargins(hmargin, vmargin, hmargin, vmargin)

    def event_to_image_position(self, event_position: QPointF) -> np.ndarray:
        margins = self.contentsMargins()
        localPos = np.array(event_position.toTuple()) - np.array(
            [margins.left(), margins.top()]
        )
        labelSize = np.array(self.size().toTuple()) - np.array(
            [margins.left() + margins.right(), margins.top() + margins.bottom()]
        )

        relPos = localPos / labelSize
        relPos[0], relPos[1] = relPos[1], relPos[0]
        assert self.image_ is not None
        pixelPos = relPos * self.image_.shape[0:2]
        pixelPos[0], pixelPos[1] = pixelPos[1], pixelPos[0]
        pixelPos = pixelPos.reshape(1, 2)
        return pixelPos
