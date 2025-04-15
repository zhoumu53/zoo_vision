import numpy as np
import numpy.typing as npt

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


def draw_clicks(
    image: npt.NDArray[np.uint8],
    positive_points: list[npt.NDArray[np.float32]],
    negative_points: list[npt.NDArray[np.float32]],
) -> npt.NDArray[np.uint8]:
    # image = frame.segmented_image
    # if image is None:
    #     image = frame.original_image.copy()

    # Add clicks
    for i, (positives, negatives) in enumerate(zip(positive_points, negative_points)):
        color = MASK_COLORS[i]
        for pixelPos in positives:
            pixelPos = pixelPos.reshape(-1).astype(np.int32)
            image[
                pixelPos[1] - 5 : pixelPos[1] + 5, pixelPos[0] - 5 : pixelPos[0] + 5
            ] = color

        for pixelPos in negatives:
            pixelPos = pixelPos.reshape(-1).astype(np.int32)
            image[
                pixelPos[1] - 5 : pixelPos[1] + 5, pixelPos[0] - 5 : pixelPos[0] + 5
            ] = COLOR_BLACK
    return image


def update_frame_image(
    image: npt.NDArray[np.uint8],
    masks: list[npt.NDArray[np.uint8]],
    positive_points: list[npt.NDArray[np.float32]],
    negative_points: list[npt.NDArray[np.float32]],
) -> npt.NDArray[np.uint8]:

    base_alpha = 0.4
    masked_image = image.astype(dtype=np.float32, copy=True)
    masked_image *= 0.8
    for i, mask in enumerate(masks):
        assert mask is not None
        alpha_mask = base_alpha * mask[:, :, np.newaxis]
        color_mask = MASK_COLORS[i].reshape(1, 1, 3) * mask[:, :, np.newaxis]
        masked_image = masked_image * (1 - alpha_mask) + alpha_mask * color_mask
    masked_image = masked_image.astype(np.uint8)

    masked_image = draw_clicks(masked_image, positive_points, negative_points)
    return masked_image
