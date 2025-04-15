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
    parser.add_argument("--image_dir", help="Directory containing the images to label.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create the application instance
    app = QApplication(sys.argv)

    images_path = Path(
        QFileDialog.getExistingDirectory(None, "Select the folder with images")
        if args.image_dir is None
        else args.image_dir
    )
    if not images_path.exists():
        raise RuntimeError(f"Video dir does not exist: {str(images_path)}")

    print("Searching for files...")
    image_files = [
        f
        for f in images_path.glob("**/*")
        if f.suffix in [".jpg", ".png"] and not f.with_suffix("").name.endswith("_seg")
    ]
    if len(image_files) == 0:
        raise RuntimeError(f"No videos found in dir: {str(images_path)}")
    image_files = sorted(image_files)
    print(f"Found {len(image_files)} images.")

    # Create the work queue and main window
    work_queue = SimpleQueue()
    main_win = MainWindow(work_queue, image_files)

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
