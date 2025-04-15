from queue import SimpleQueue

import numpy as np
from PySide6.QtGui import QStatusTipEvent
from PySide6.QtWidgets import QApplication

from database import DatabaseFrame, Record, active_db
from drawing import draw_clicks, update_frame_image
from main_window import MainWindow
from sam2_processor import Sam2Processor


class BackgroundSegmenter:
    def __init__(self, window: MainWindow, work_queue: SimpleQueue) -> None:
        self.should_stop = False
        self.window = window

        self.run_with_sam = True
        if self.run_with_sam:
            self.sam2_ = Sam2Processor()
        else:
            self.sam2_ = None

        self.work_queue_ = work_queue

    def run(self) -> None:
        while not self.should_stop:
            window = QApplication.activeWindow()
            if window is not None:
                if self.work_queue_.empty():
                    QApplication.sendEvent(window, QStatusTipEvent("sam2:Ready"))

            try:
                try:
                    frame_index: int = self.work_queue_.get(block=True, timeout=0.5)
                except:
                    # No work item, not an error, simply continue
                    continue

                frame = active_db().frames.get(frame_index)

                # Check that frame was not deleted
                if frame is not None:
                    if window is not None:
                        QApplication.sendEvent(
                            window,
                            QStatusTipEvent(
                                f"sam2:Segmenting {self.work_queue_.qsize() + 1} frames..."
                            ),
                        )

                    # Do a slow segmentation
                    for record in frame.records.values():
                        if record.segmentation is None:
                            # Segment!
                            self.segment_record(frame, record)

                    # Combine segmentations into a single image
                    update_frame_image(frame)

                # Trigger UI update
                self.window.update_ui(frame_index)
            except Exception as e:
                print(f"Error background segmenter queue item: {e}")
                continue

    def segment_record(self, frame: DatabaseFrame, record: Record) -> None:
        mask = np.full(
            (frame.original_image.shape[0], frame.original_image.shape[1]), 0
        )
        try:
            if self.sam2_:
                mask = self.sam2_.process_click(
                    frame.original_image, record.positive_points, record.negative_points
                )
                mask = mask.astype(np.uint8)
        except Exception as e:
            print(f"Error processing click: {e}")

        record.segmentation = mask
