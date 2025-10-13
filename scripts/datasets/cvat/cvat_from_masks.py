import xml.etree.ElementTree as ET
from pathlib import Path
from argparse import ArgumentParser
from cvat_sdk import masks
import cv2
import numpy as np

from project_root import PROJECT_ROOT


def sortchildrenby(parent, attr):
    parent[:] = sorted(parent, key=lambda child: int(child.get(attr)))


def poly_from_mask(mask):
    # Find contours in the mask
    # RETR_EXTERNAL retrieves only the extreme outer contours
    # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    areas = np.array([cv2.contourArea(x) for x in contours])
    biggest_index = np.argmax(areas)
    polygon = contours[biggest_index]

    # Optional: Simplify the contour into a polygon with fewer vertices
    # The epsilon parameter determines the approximation accuracy.
    # A smaller epsilon means a more precise approximation (more vertices).
    # A larger epsilon means a coarser approximation (fewer vertices).
    epsilon = 0.001 * cv2.arcLength(polygon, True)  # Adjust epsilon as needed
    polygon = cv2.approxPolyDP(polygon, epsilon, True)

    polygon = [p[0] for p in polygon]
    # 'polygons' now contains a list of arrays, where each array represents a polygon's vertices.
    # Each vertex is typically an array like [[x, y]].
    return polygon


def main():
    arg_parser = ArgumentParser("cvat_from_masks.py")
    arg_parser.add_argument(
        "-d", "--dir", help="Directory where tracks with masks are.", required=True
    )
    arg_parser.add_argument(
        "-sx", "--width", help="Original video width.", required=True, type=int
    )
    arg_parser.add_argument(
        "-sy", "--height", help="Original video height.", required=True, type=int
    )
    args = arg_parser.parse_args()

    tracks_dir = Path(args.dir)
    assert tracks_dir.exists()

    original_width = args.width
    original_height = args.height

    root = ET.parse(PROJECT_ROOT / "scripts/datasets/cvat/cvat_dataset_template.xml")

    TRACK_DIR_PREFIX = "track_"
    for track_dir in tracks_dir.glob(TRACK_DIR_PREFIX + "*"):
        track_id = int(track_dir.name[len(TRACK_DIR_PREFIX) :])

        track_xml = ET.SubElement(
            root.getroot(),
            "track",
            id=str(track_id),
            label="elephant",
            source="auto",
        )

        MASK_PREFIX = "frame_"
        masks_dir = tracks_dir / track_dir
        mask_filenames = sorted(masks_dir.glob(MASK_PREFIX + "*"))

        for mask_filename in mask_filenames:
            frame_id = int(mask_filename.with_suffix("").name[len(MASK_PREFIX) :])
            mask_data = cv2.imread(
                str(tracks_dir / track_dir / mask_filename), cv2.IMREAD_GRAYSCALE
            )
            assert mask_data is not None
            # Resize the mask because detection in zoo_vision happens at lower resolution than original video
            mask_data = cv2.resize(mask_data, [original_width, original_height])

            # Note: this uploads masks in rle format but cvat does not support tracks for masks,
            #       so we have to transform them to polygons instead
            # mask_rle = masks.encode_mask(mask_data != 0)
            # x1, y1, x2, y2 = mask_rle[-4:]
            # ET.SubElement(
            #     track_xml,
            #     "mask",
            #     frame=str(frame_id),
            #     keyframe="1",
            #     occluded="0",
            #     outside="0" if (frame_id % 2) == 0 else "1",
            #     rle=",".join(str(x) for x in mask_rle),
            #     left=str(x1),
            #     top=str(y1),
            #     width=str(x2 - x1 + 1),
            #     height=str(y2 - y1 + 1),
            #     z_order="0",
            # )

            poly = poly_from_mask(mask_data)
            ET.SubElement(
                track_xml,
                "polygon",
                frame=str(frame_id),
                keyframe="1",
                occluded="0",
                outside="0",
                points=";".join([",".join([str(x) for x in p]) for p in poly]),
            )
        # Write a final polygon marking it as outside
        ET.SubElement(
            track_xml,
            "polygon",
            frame=str(frame_id + 1),
            keyframe="1",
            occluded="0",
            outside="1",
            points="0,0;1,1;0,1",
        )
    ET.indent(root)
    root.write("dataset.xml")


if __name__ == "__main__":
    main()
