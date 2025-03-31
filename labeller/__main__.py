import sys
import threading
import argparse
from pathlib import Path
from queue import SimpleQueue
from PySide6.QtWidgets import QApplication, QFileDialog

from background_segmenter import BackgroundSegmenter
from main_window import MainWindow


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", help="Directory containing the videos to label.")
    parser.add_argument("--label_dir", help="Directory containing the labels.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create the application instance
    app = QApplication(sys.argv)

    videos_path = Path(
        QFileDialog.getExistingDirectory(None, "Select the folder with videos")
        if args.video_dir is None
        else args.video_dir
    )
    if not videos_path.exists():
        raise RuntimeError(f"Video dir does not exist: {str(videos_path)}")
    elif len(list(videos_path.glob("**/*.mp4"))) == 0:
        raise RuntimeError(f"No videos found in dir: {str(videos_path)}")

    labels_path = Path(args.label_dir) if args.label_dir is not None else videos_path
    labels_path.mkdir(parents=True, exist_ok=True)

    # Create the work queue and main window
    work_queue = SimpleQueue()
    main_win = MainWindow(work_queue, videos_path, labels_path)

    # Initialize and start the background segmenter thread
    segmenter = BackgroundSegmenter(main_win, work_queue)
    segmenter_thread = threading.Thread(target=segmenter.run, daemon=False)
    segmenter_thread.start()

    # Resize and display the main window
    available_geometry = main_win.screen().availableGeometry()
    main_win.resize(available_geometry.width() // 3, available_geometry.height() // 2)

    # Show the main application window
    main_win.show()
    ret_value = app.exec()

    segmenter.should_stop = True

    sys.exit(ret_value)
