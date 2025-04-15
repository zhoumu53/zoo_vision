import numpy as np
from database import DatabaseFrame, Record

COLOR_GREEN = np.array([0, 255, 0], dtype=np.uint8).reshape(1, 1, 3)
COLOR_RED = np.array([0, 0, 255], dtype=np.uint8).reshape(1, 1, 3)
COLOR_BLACK = np.array([0, 0, 0], dtype=np.uint8).reshape(1, 1, 3)

MASK_COLORS = np.array(
    [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
    ],
    dtype=np.uint8,
)


def draw_clicks(frame: DatabaseFrame) -> None:
    image = frame.segmented_image
    if image is None:
        image = frame.original_image.copy()

    # Add clicks
    for i, record in enumerate(frame.records.values()):
        color = MASK_COLORS[i]
        for pixelPos in record.positive_points:
            pixelPos = pixelPos.reshape(-1).astype(np.int32)
            image[
                pixelPos[1] - 5 : pixelPos[1] + 5, pixelPos[0] - 5 : pixelPos[0] + 5
            ] = color

        for pixelPos in record.negative_points:
            pixelPos = pixelPos.reshape(-1).astype(np.int32)
            image[
                pixelPos[1] - 5 : pixelPos[1] + 5, pixelPos[0] - 5 : pixelPos[0] + 5
            ] = COLOR_BLACK
    frame.segmented_image = image


def update_frame_image(frame: DatabaseFrame) -> None:

    base_alpha = 0.4
    masked_image = frame.original_image.astype(dtype=np.float32, copy=True)
    masked_image *= 0.8
    for i, record in enumerate(frame.records.values()):
        assert record.segmentation is not None
        mask = record.segmentation
        alpha_mask = base_alpha * mask[:, :, np.newaxis]
        color_mask = MASK_COLORS[i].reshape(1, 1, 3) * mask[:, :, np.newaxis]
        masked_image = masked_image * (1 - alpha_mask) + alpha_mask * color_mask
    masked_image = masked_image.astype(np.uint8)

    frame.segmented_image = masked_image

    draw_clicks(frame)
