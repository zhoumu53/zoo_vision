from sam2.build_sam import build_sam2, build_sam2_video_predictor
import sam2.sam2_video_predictor
import numpy as np
import torch
import cv2
import json
import matplotlib.pyplot as plt
from pathlib import Path
import enlighten
import gc
from typing import Any
import argparse

from project_root import PROJECT_ROOT
from segmentation_utils import grey_from_label


def points_json_path_from_video_path(video_path: Path) -> Path:
    # Avoid .with_suffix() because the filenames contain dots
    points_json_path = Path(str(video_path).replace(".mp4", "_points.json"))
    return points_json_path


def add_label_points(
    predictor: sam2.sam2_video_predictor.SAM2VideoPredictor,
    inference_state: dict[Any, Any],
    obj_id: int,
    record: dict[str, Any],
):
    ann_frame_idx = 0  # the frame index we interact with
    positive_points = np.array(record["ppoints"], dtype=np.float32).reshape((-1, 2))
    negative_points = np.array(record["npoints"], dtype=np.float32).reshape((-1, 2))

    points = np.concatenate([positive_points, negative_points])
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array(
        [1] * positive_points.shape[0] + [0] * negative_points.shape[0], np.int32
    )
    _, _, mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
    )
    return points, labels, mask_logits


def mask_iou(a, b) -> float:
    overlap_area = (a & b).sum()
    union_area = (a | b).sum()
    iou = overlap_area / union_area
    return iou


def segmentation_from_logits(logits: torch.Tensor, obj_ids: list[int]) -> torch.Tensor:
    logits = logits[:, 0, :, :]  # Drop batch dimension
    logit_union, mask_idx = logits.max(dim=0)

    h, w = logit_union.shape
    segmentation = torch.zeros([h, w], dtype=torch.uint8, device=logits.device)
    valid_idx = logit_union > 0

    for idx, obj_id in enumerate(obj_ids):
        segmentation[valid_idx & (mask_idx == idx)] = grey_from_label(obj_id)
    return segmentation


def save_frame(
    path_prefix: Path, index: int, frame: np.ndarray, segmentation: np.ndarray
) -> None:
    path_prefix.parent.mkdir(exist_ok=True, parents=True)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(
        f"{str(path_prefix)}_{index:08d}_img.jpg",
        frame_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    # cv2.imwrite(f"{str(path_prefix)}_{index:08d}_img.png", frame_bgr)
    cv2.imwrite(f"{str(path_prefix)}_{index:08d}_seg.png", segmentation)


def make_path_prefix(input_path: Path, video_file: Path, output_path: Path) -> Path:
    relative_path = video_file.relative_to(input_path)
    relative_dir = relative_path.parent
    prefix = output_path / relative_dir / video_file.name.replace(".mp4", "")
    return prefix


def are_results_present(input_path: Path, video_path: Path, output_path: Path):
    prefix = make_path_prefix(input_path, video_path, output_path)
    if len(list(prefix.parent.glob(f"{prefix.name}*"))) > 0:
        return True
    else:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_count", "-f", type=int, default=200)
    parser.add_argument("--input_path", "-i", type=Path)
    parser.add_argument("--output_path", "-o", type=Path)
    return parser.parse_args()


class App:
    def __init__(self):
        self.pbar_manager = enlighten.get_manager()
        self.predictor: sam2.sam2_video_predictor.SAM2VideoPredictor | None = None
        self.id_from_name = {}  # {name: id}

    def main(self) -> None:
        args = parse_args()
        propagate_frame_count = args.frame_count
        input_path = args.input_path
        output_path = args.output_path

        print("Loading identity labels...")
        with (PROJECT_ROOT / "data/config.json").open() as f:
            config = json.load(f)
        self.id_from_name = {
            name: props["id"] for name, props in config["individuals"].items()
        }
        # Correct misspelling
        self.id_from_name["Farha"] = self.id_from_name["Fahra"]
        print(self.id_from_name)

        print("Loading SAM2...")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        sam2_checkpoint = str(PROJECT_ROOT / "models/sam2/sam2.1_hiera_large.pt")
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        self.predictor = build_sam2_video_predictor(
            model_cfg, sam2_checkpoint, device="cuda"
        )

        print("Gathering inputs...")

        videos = [f for f in input_path.glob("**/*.mp4")]
        labelled_videos = [
            f for f in videos if points_json_path_from_video_path(f).exists()
        ]

        # Count annotations for early summary
        annotation_count = 0
        for f in labelled_videos:
            with points_json_path_from_video_path(f).open("r") as f:
                points_data = json.load(f)
            annotation_count += len(points_data)

        print(
            f"{str(input_path)}: total videos={len(videos)}, labelled={len(labelled_videos)}, annotations={annotation_count}"
        )

        print("Processing...")
        pbar = self.pbar_manager.counter(
            total=len(labelled_videos), desc="Processing videos", unit="video"
        )
        for video_path in pbar(labelled_videos):
            # Check to see if there are already images
            if are_results_present(input_path, video_path, output_path):
                print(f"Found existing images for {str(video_path)}, skipping video")
            else:
                self.process_video(
                    video_path, input_path, output_path, propagate_frame_count
                )
        pbar.close()

    def process_video(
        self,
        video_path: Path,
        input_path: Path,
        output_path: Path,
        propagate_frame_count: int,
    ):
        print(f"Processing {str(video_path)}...")

        with points_json_path_from_video_path(video_path).open("r") as f:
            points_data = json.load(f)

        # Sort data
        points_data = sorted(points_data, key=lambda x: x["frame"])

        video = cv2.VideoCapture(video_path)
        video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video frame count: {video_frame_count}, labelled: {len(points_data)}")
        if len(points_data) == 0:
            return  # Early exit to avoid issues

        obj_count = np.max([len(x["records"]) for x in points_data])
        print(f"Objects: {obj_count}")

        distance_to_next_label = [
            points_data[i + 1]["frame"] - points_data[i]["frame"]
            for i in range(len(points_data) - 1)
        ]
        distance_to_next_label.append(video_frame_count - points_data[-1]["frame"])
        print(
            f"distance_to_next_label: median={np.median(distance_to_next_label)}, min={np.min(distance_to_next_label)}, max={np.max(distance_to_next_label)}"
        )

        prefix = make_path_prefix(input_path, video_path, output_path)

        pbar_anno = self.pbar_manager.counter(
            total=len(points_data),
            desc=f"Annotations for {video_path.name}",
            unit="annot",
        )
        for i, data_i in enumerate(points_data):
            frame_count = min(propagate_frame_count, distance_to_next_label[i])

            self.process_annotation(video, data_i, frame_count, prefix)

            pbar_anno.update()
        pbar_anno.close()

    def load_frames(self, video, ref_frame_index, frame_count):
        video.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_index)
        frames = []
        pbar = self.pbar_manager.counter(
            total=frame_count, desc="Loading frames", unit="frame", leave=False
        )
        for i in range(frame_count):
            valid, frame = video.read()
            if not valid:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            pbar.update()
        pbar.close()
        return frames

    def process_annotation(self, video, annotation, frame_count, prefix):
        IOU_THRESHOLD = 0.9

        ref_frame_idx = annotation["frame"]

        frames = self.load_frames(video, ref_frame_idx, frame_count)
        if "inference_state" in locals():
            del inference_state
        gc.collect()
        inference_state = self.predictor.init_state(video_path=frames)

        for record in annotation["records"]:
            label = self.id_from_name[record["name"]]
            add_label_points(self.predictor, inference_state, label, record)

        ref_masks = None

        SKIP_FRAME_COUNT = 10
        pbar_frames = self.pbar_manager.counter(
            total=len(frames),
            desc=f"Propagating frame {ref_frame_idx}",
            unit="frame",
            leave=False,
        )
        overlap_skip_count = 0
        total_mask_count = 0
        for (
            sam_frame_idx,
            obj_ids,
            mask_logits,
        ) in self.predictor.propagate_in_video(inference_state):
            pbar_frames.update()
            if sam_frame_idx % SKIP_FRAME_COUNT != 0:
                continue

            total_mask_count += 1
            video_frame_idx = ref_frame_idx + sam_frame_idx

            segmentation_image = segmentation_from_logits(mask_logits, obj_ids)
            mask_all = segmentation_image > 0
            if mask_all.sum() == 0:
                break

            if ref_masks is None:
                ref_masks = mask_all
            else:
                iou = mask_iou(ref_masks, mask_all).cpu().item()
                if iou > IOU_THRESHOLD:
                    overlap_skip_count += 1
                    continue
                ref_masks = mask_all

            segmentation_image = segmentation_image.cpu().numpy()

            save_frame(
                prefix, video_frame_idx, frames[sam_frame_idx], segmentation_image
            )
        pbar_frames.close()
        print(
            f"Skipped {overlap_skip_count}/{total_mask_count} due to high mask overlap"
        )


def fake_tqdm(x, desc):
    return x


if __name__ == "__main__":
    # Disable progress bar
    sam2.sam2_video_predictor.tqdm = fake_tqdm

    app = App()
    app.main()
