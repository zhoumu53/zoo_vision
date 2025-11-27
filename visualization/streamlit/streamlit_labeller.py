from __future__ import annotations

import sys
import threading
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from ultralytics import YOLO

THIS_DIR = Path(__file__).resolve().parent
VIS_DIR = THIS_DIR.parent
ROOT = VIS_DIR.parent
for path in (THIS_DIR, VIS_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analysis  # type: ignore
import fixer  # type: ignore
# import video_tracks_reid_improved as tracker  # type: ignore
import video_tracks_reid_improved_with_behavior as tracker  # type: ignore


st.set_page_config(page_title="Tracking Labeller", layout="wide")


def _init_state() -> None:
    if "frame_queue" not in st.session_state:
        st.session_state.frame_queue = Queue(maxsize=4)
    if "latest_frame" not in st.session_state:
        st.session_state.latest_frame: Optional[tuple[int, np.ndarray, list]] = None
    if "runner_thread" not in st.session_state:
        st.session_state.runner_thread: Optional[threading.Thread] = None
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "tracks_json_path" not in st.session_state:
        st.session_state.tracks_json_path = ""
    if "frames_dir" not in st.session_state:
        st.session_state.frames_dir = ""
    if "output_video" not in st.session_state:
        st.session_state.output_video = ""
    if "tracks_df" not in st.session_state:
        st.session_state.tracks_df: Optional[pd.DataFrame] = None
    if "track_meta" not in st.session_state:
        st.session_state.track_meta: Dict = {}
    if "fixes" not in st.session_state:
        st.session_state.fixes: Dict[int, int] = {}
    if "fixes_path" not in st.session_state:
        st.session_state.fixes_path = ""
    if "yolo_stream_running" not in st.session_state:
        st.session_state.yolo_stream_running = False
    if "yolo_stream_stop" not in st.session_state:
        st.session_state.yolo_stream_stop = False


def _drain_frame_queue() -> None:
    """Move the newest frame from the queue into session_state."""
    q: Queue = st.session_state.frame_queue
    latest = st.session_state.get("latest_frame")
    while True:
        try:
            latest = q.get_nowait()
        except Empty:
            break
    if latest is not None:
        st.session_state.latest_frame = latest


def _make_frame_callback() -> callable:
    q: Queue = st.session_state.frame_queue

    def on_frame(frame_idx: int, frame_bgr: np.ndarray, tracks: list) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            q.put_nowait((frame_idx, frame_rgb, tracks))
        except Full:
            try:
                q.get_nowait()
            except Empty:
                pass
            q.put_nowait((frame_idx, frame_rgb, tracks))

    return on_frame


def _start_tracker(args) -> None:
    if st.session_state.is_running and st.session_state.runner_thread is not None:
        st.warning("Tracker already running.")
        return

    cb = _make_frame_callback()
    runner = threading.Thread(target=tracker.run_tracking, args=(args,), kwargs={"frame_callback": cb}, daemon=True)
    runner.start()
    st.session_state.runner_thread = runner
    st.session_state.is_running = True
    st.session_state.tracks_json_path = args.tracks_json
    st.session_state.frames_dir = args.frames_dir or str(Path(args.output) / Path(args.video).stem)
    st.session_state.output_video = str(Path(args.output) / f"{Path(args.video).stem}_tracks.mp4")


def _build_args_from_ui(
    video_path: str,
    output_dir: str,
    yolo_model: str,
    class_names: str,
    reid_config: str,
    reid_ckpt: str,
    max_frames: int | None,
    frame_skip: int,
    disable_stitching: bool,
    behavior_model: str = "",
) -> object:
    cli_args = [
        "--video",
        video_path,
        "--output",
        output_dir,
        "--yolo-model",
        yolo_model,
        "--class-names",
        class_names,
        "--reid-config",
        reid_config,
        "--reid-checkpoint",
        reid_ckpt,
        "--frame-skip",
        str(frame_skip),
        "--jpg-interval",
        "1",
        "--frames-dir",
        str(Path(output_dir) / "frames"),
        "--tracks-json",
        str(Path(output_dir) / "tracks.jsonl"),
        "--log-level",
        "INFO",
        "--behavior-model",
        behavior_model,
    ]
    if max_frames is not None and max_frames > 0:
        cli_args.extend(["--max-frames", str(max_frames)])
    if disable_stitching:
        cli_args.append("--no-new-stitching")
    cli_args.append("--save-jpg")
    args = tracker.parse_args(cli_args)
    return args


def _load_tracks_if_available(path_str: str) -> None:
    if not path_str:
        return
    try:
        df, meta = fixer.load_track_log(path_str)
    except FileNotFoundError:
        st.warning(f"Could not find track log at {path_str}")
        return
    st.session_state.tracks_df = df
    st.session_state.track_meta = meta
    fixes_path = Path(path_str).with_name(fixer.DEFAULT_FIXES_NAME)
    st.session_state.fixes_path = str(fixes_path)
    st.session_state.fixes = fixer.load_fixes(fixes_path)


def _show_frame_image(frames_dir: str, frame_idx: int) -> None:
    if not frames_dir:
        st.info("No frames directory configured yet.")
        return
    frame_path = Path(frames_dir) / f"frame_{frame_idx:06d}.jpg"
    if not frame_path.exists():
        st.info(f"Frame image not found at {frame_path}")
        return
    img = cv2.imread(str(frame_path))
    if img is None:
        st.warning(f"Failed to load {frame_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img, caption=f"Frame {frame_idx}", use_column_width=True)


def main() -> None:
    _init_state()
    _drain_frame_queue()

    st.title("Streamlit Tracking Labeller")
    st.caption(
        "Run the lightweight tracker, preview annotated frames live, relabel bad tracks, and export summaries."
    )

    with st.sidebar:
        st.header("Run tracker")
        video_path = st.text_input(
            "Video path",
            "/mnt/camera_nas/ZAG-ELP-CAM-016/20240906PM/ZAG-ELP-CAM-016-20240906-184716-1725641236715-7.mp4",
        )
        output_dir = st.text_input(
            "Output directory",
            "/home/mu/Desktop/comparison_videos/test_steamlit",
        )
        yolo_model = st.text_input(
            "YOLO weights (.pt / .onnx)",
            "/media/mu/zoo_vision/models/segmentation/yolo/all_v3/weights/best.pt",
        )
        class_names = st.text_input(
            "Class names txt",
            "/media/mu/zoo_vision/models/segmentation/yolo/class_names.txt",
        )
        reid_config = st.text_input(
            "ReID config (.yml)",
            "/media/mu/zoo_vision/training/PoseGuidedReID/configs/elephant_resnet.yml",
        )
        reid_ckpt = st.text_input(
            "ReID checkpoint (.pth)",
            "/media/mu/zoo_vision/training/PoseGuidedReID/logs/elephant_resnet/lr001_bs16_softmax_triplet/net_best.pth",
        )
        behavior_model = st.text_input(
            "Behavior model (.pt) (optional)",
            "/media/mu/zoo_vision/models/sleep/vit/v2_no_validation/config.ptc",
        )
        max_frames_val = st.number_input("Max frames (optional)", min_value=0, value=100, step=10)
        frame_skip = st.number_input("Frame skip", min_value=1, value=5, step=1)
        disable_stitch = st.checkbox("Disable ReID stitching (raw ByteTrack IDs only)", value=False)
        start_btn = st.button("Start tracking", type="primary")

        if start_btn:
            if not all([video_path, output_dir, yolo_model, class_names]):
                st.error("Fill video, output, YOLO weights, and class names to start.")
            else:
                max_frames = None if max_frames_val == 0 else int(max_frames_val)
                args = _build_args_from_ui(
                    video_path,
                    output_dir,
                    yolo_model,
                    class_names,
                    reid_config,
                    reid_ckpt,
                    max_frames,
                    int(frame_skip),
                    disable_stitch,
                    behavior_model
                )
                _start_tracker(args)

        if st.session_state.runner_thread is not None and not st.session_state.runner_thread.is_alive():
            st.session_state.is_running = False

        st.write(f"Tracker status: {'Running' if st.session_state.is_running else 'Idle'}")
        if st.session_state.tracks_json_path:
            st.write(f"Current track log: {st.session_state.tracks_json_path}")

    tab_live, tab_analysis = st.tabs(
        ["Live preview","Analysis"]
    )

    with tab_live:
        st.subheader("Live annotated frames")
        auto_refresh = st.checkbox("Auto-refresh while tracking", value=True)
        if st.session_state.is_running and auto_refresh:
            # Trigger a rerun every second while the tracker is running
            st_autorefresh(interval=1000, key="live_autorefresh")
        latest = st.session_state.latest_frame
        if latest is None:
            st.info("Start the tracker or load a track log to see frames.")
        else:
            frame_idx, frame_rgb, tracks = latest
            st.image(frame_rgb, caption=f"Live frame {frame_idx}", use_column_width=True)
            if tracks:
                st.dataframe(pd.DataFrame(tracks))
        if st.session_state.is_running and not auto_refresh:
            st.info("Tracker running in background. Click this button to refresh live frame.")
            if st.button("Refresh live frame"):
                _drain_frame_queue()
                st.experimental_rerun()

if __name__ == "__main__":
    main()
