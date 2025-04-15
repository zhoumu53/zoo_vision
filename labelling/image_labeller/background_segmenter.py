from queue import SimpleQueue

import numpy as np
from PySide6.QtGui import QStatusTipEvent
from PySide6.QtWidgets import QApplication

import project_root  # type: ignore
from database import Record, active_db, Database
from main_window import MainWindow
from labelling.common.drawing import update_frame_image
from labelling.common.sam2_processor import Sam2Processor
from labelling.common.utils import unwrap


class BackgroundSegmenter:
    def __init__(self, window: MainWindow, work_queue: SimpleQueue[int]) -> None:
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
                    _ = self.work_queue_.get(block=True, timeout=0.5)
                except:
                    # No work item, not an error, simply continue
                    continue

                # Check that frame was not deleted
                db = active_db()
                if db.image.size != 0:
                    if window is not None:
                        QApplication.sendEvent(
                            window,
                            QStatusTipEvent(
                                f"sam2:Segmenting {self.work_queue_.qsize() + 1} frames..."
                            ),
                        )

                    # Do a slow segmentation
                    for record in db.records.values():
                        if record.segmentation is None:
                            # Segment!
                            self.segment_record(db, record)

                    # Combine segmentations into a single image
                    db.segmented_image = update_frame_image(
                        db.image,
                        [unwrap(r.segmentation) for r in db.records.values()],
                        [r.positive_points for r in db.records.values()],
                        [r.negative_points for r in db.records.values()],
                    )

                # Trigger UI update
                self.window.update_image()
            except Exception as e:
                print(f"Error background segmenter queue item: {e}")
                continue

    def segment_record(self, db: Database, record: Record) -> None:
        mask = np.full((db.image.shape[0], db.image.shape[1]), 0)
        try:
            if self.sam2_:
                mask = self.sam2_.process_click(
                    db.image, record.positive_points, record.negative_points
                )
                mask = mask.astype(np.uint8)
        except Exception as e:
            print(f"Error processing click: {e}")

        record.segmentation = mask
