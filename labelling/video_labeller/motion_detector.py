import json
import logging
from pathlib import Path
import argparse

import cv2
from tqdm import tqdm

from utils import pretty_time_delta

logger = logging.getLogger(__name__)


def detect_motion(
    video_path: Path,
    labels_path: Path,
    movement_threshold=0.01,
    min_contour_area=0.001,
    step_sec=3,
    debug=False,
    cut_top=None,
    cut_bottom=None,
    cut_right=None,
    cut_left=None,
):
    logger.info(f"Loading {str(video_path)}")
    assert video_path.exists()

    # Open the video file
    video_reader = cv2.VideoCapture(str(video_path))
    frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_reader.get(cv2.CAP_PROP_FPS)

    frame_step = int(
        fps * step_sec
    )  # Number of frames to skip to achieve the desired interval

    iter_count = frame_count // frame_step

    motion_periods = []

    ret, frame1 = video_reader.read()
    if not ret:
        logger.error("Failed to read first frame, aborting video")
        return
    frame_index = 0

    height = frame1.shape[0]
    width = frame1.shape[1]
    logger.info(
        f"Original size ({width},{height}), {pretty_time_delta(frame_count/fps)}"
    )

    # crop_x = int(cut_left / 100 * width if cut_left else 0)
    # crop_y = int(cut_top / 100 * height if cut_top else 0)
    # crop_xx = width - int(cut_right / 100 * width if cut_right else 0)
    # crop_yy = height - int(cut_bottom / 100 * height if cut_bottom else 0)

    def frame_preprocess(frame):
        # frame = frame[crop_x:crop_xx, crop_y:crop_yy]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        down_scale = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        return down_scale

    frame1 = frame_preprocess(frame1)

    height = frame1.shape[0]
    width = frame1.shape[1]
    logger.info(
        f"Subsampled size ({width},{height}), {pretty_time_delta((frame_count/frame_step)/fps)}"
    )

    movement_threshold_px = movement_threshold * width * height
    min_contour_area_px = min_contour_area * width * height

    motion_start_str = None
    motion_start = None
    total_moving_frame_count = 0

    for _ in tqdm(range(iter_count), desc=f"{video_path.name}", unit="frames"):
        frame_index += frame_step

        # Skip to the next frame after the desired interval
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame2 = video_reader.read()
        if not ret:
            logger.error("Error reading frame {frame_index}")
            continue

        frame2 = frame_preprocess(frame2)

        diff = cv2.absdiff(frame1, frame2)
        _, threshold_binary_image = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        # dilated = cv2.dilate(threshold_binary_image, None, iterations=3)
        dilated = threshold_binary_image

        if debug:
            cv2.imshow("diff", diff)
            cv2.imshow("down_scale1", frame1)
            cv2.imshow("down_scale2", frame2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate the total movement
        total_movement = sum(
            cv2.contourArea(c)
            for c in contours
            if cv2.contourArea(c) > min_contour_area_px
        )

        # Convert frame count to timestamp
        timestamp = frame_index / fps
        timestamp_str = pretty_time_delta(timestamp)

        # Detect motion start and end times
        if total_movement > movement_threshold_px:
            total_moving_frame_count += 1
            if (total_moving_frame_count % 100) == 0:
                logger.info(f"Moving frames: {total_moving_frame_count}/{frame_index}")
            if motion_start_str is None:
                motion_start_str = timestamp_str  # Start a new motion period
                motion_start = frame_index  # Start a new motion period

                if debug:
                    # Create a blank image with the same dimensions as the dilated image
                    contour_img = cv2.cvtColor(
                        dilated, cv2.COLOR_GRAY2BGR
                    )  # Convert to BGR to draw colored contours
                    # Draw the contours on the blank image
                    cv2.drawContours(
                        contour_img, contours, -1, (0, 255, 0), 2
                    )  # Green contours with thickness of 2
                    # Display the image with contours
                    cv2.imshow("Contours", contour_img)
                    cv2.imshow("down_scale1", frame1)
                    cv2.imshow("frame1", frame1)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        else:
            if motion_start_str is not None:
                motion_periods.append(
                    {
                        "start": motion_start_str,
                        "end": timestamp_str,
                        "start_frames": motion_start,
                        "end_frames": frame_index,
                    }
                )
                motion_start_str = None  # Reset motion start
                motion_start = None

        # Update the frame and frame count
        frame1 = frame2

    # If motion was ongoing at the end of the video
    if motion_start_str is not None:
        timestamp = frame_index / fps
        motion_periods.append(
            {
                "start": motion_start_str,
                "end": pretty_time_delta(timestamp),
                "start_frames": motion_start,
                "end_frames": frame_index,
            }
        )

    # Save the motion periods to a JSON file
    output_json = labels_path / video_path.with_suffix(".json").name
    with open(output_json, "w") as f:
        json.dump(motion_periods, f, indent=4)

    print(f"Motion detection complete. Motion periods saved to {output_json}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", help="Video or directory containing the videos to detect motion on."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Detect motion even if motion json already exists",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    path = Path(args.path)
    if path.exists() and path.is_file():
        logger.info(f"Processing single file: {str(path)}")
        video_files = [path]
    else:
        video_files = list(path.glob("**/*.mp4"))
        logger.info(f"Found {len(video_files)} video files")

    if not args.force:
        total_video_files = len(video_files)
        video_files = [f for f in video_files if not f.with_suffix(".json").exists()]
        logger.info(
            f"Skipping {total_video_files - len(video_files)} because they already have motion info"
        )

    if len(video_files) == 0:
        print("No videos to process!")

    for file in video_files:
        detect_motion(file, file.parent)


if __name__ == "__main__":
    main()
