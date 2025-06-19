import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass


def imread_rgb(name):
    m = cv2.imread(name)
    return cv2.cvtColor(m, cv2.COLOR_BGR2RGB)


# Homogenous coordinates utilities
def to_h(x, hvalue=1):
    if len(x.shape) == 1:
        return np.concatenate([x, np.full((1,), hvalue)])
    else:
        count = x.shape[0]
        return np.concatenate([x, np.full((count, 1), hvalue)], axis=1)


def from_h(x):
    if len(x.shape) == 1:
        return x[0:-1] / x[-1]
    else:
        return x[:, 0:-1] / x[:, [-1]]


def hmult(A, b, keep_h=False):
    np.testing.assert_equal(len(A.shape), 2)
    np.testing.assert_equal(A.shape[1], b.shape[-1] + 1)
    bh = to_h(b)
    Abh = (A @ bh.T).T
    if not keep_h:
        return from_h(Abh)
    else:
        return Abh


def K_from_fov(fov, center, width):
    fx = width / np.tan(fov / 2)
    K = np.array([[fx, 0, center[0]], [0, fx, center[1]], [0, 0, 1]])
    return K


def test_hfuncs():
    print("Homogenous utilities tests")
    a2 = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    print(to_h(a2))
    print(from_h(to_h(a2)))
    A = np.eye(3)
    hmult(A, np.ones((2,)))
    hmult(A, np.ones((10, 2)))


def plot_projections_generic(
    H_camera_from_world2: np.ndarray,
    T_submap_from_world2: np.ndarray,
    im_submap: np.ndarray,
    im_camera: np.ndarray,
    points_in_image: np.ndarray,
    points_in_world2: np.ndarray,
    floor_polygon_in_camera: np.ndarray,
    undistort=lambda x: x,
    distort=lambda x: x,
):
    H_world2_from_camera = np.linalg.inv(H_camera_from_world2)
    H_submap_from_camera = T_submap_from_world2 @ H_world2_from_camera

    floor_polygon_in_submap = hmult(
        H_submap_from_camera, undistort(floor_polygon_in_camera)
    )
    submap_points = hmult(T_submap_from_world2, points_in_world2)
    camera_points_exp = distort(hmult(H_camera_from_world2, points_in_world2))
    world2_points_exp = hmult(H_world2_from_camera, undistort(points_in_image))
    submap_points_exp = hmult(T_submap_from_world2, points_in_world2)

    errors = np.linalg.norm(submap_points - submap_points_exp, axis=1)
    print(
        f"Error (submap units): mean={np.mean(errors)}, max={np.max(errors)}, sum={np.sum(errors)}"
    )

    errors = np.linalg.norm(points_in_world2 - world2_points_exp, axis=1)
    print(
        f"Error (world units): mean={np.mean(errors)}, max={np.max(errors)}, sum={np.sum(errors)}"
    )

    errors = np.linalg.norm(points_in_image - points_in_image, axis=1)
    print(
        f"Error (camera units): mean={np.mean(errors)}, max={np.max(errors)}, sum={np.sum(errors)}"
    )

    fig, axs = plt.subplots(1, 2, figsize=(20, 15))
    axs[0].imshow(im_submap, alpha=0.5)
    axs[1].imshow(im_camera, alpha=0.5)
    ax = axs[0]
    ax.plot(
        floor_polygon_in_submap[:, 0],
        floor_polygon_in_submap[:, 1],
        "-",
        color="purple",
        label="Transformed floor polygon",
    )
    ax = axs[1]
    ax.plot(
        floor_polygon_in_camera[:, 0],
        floor_polygon_in_camera[:, 1],
        "-",
        color="purple",
        label="Selected floor polygon",
    )
    for i in range(points_in_image.shape[0]):
        ax = axs[0]
        ax.plot(
            submap_points[i, 0],
            submap_points[i, 1],
            "*",
            markersize=20,
            color="darkgreen",
            alpha=1,
            label="Selected map points" if i == 0 else None,
        )
        ax.plot(
            submap_points_exp[i, 0],
            submap_points_exp[i, 1],
            "+",
            markersize=20,
            color="red",
            alpha=1,
            label="Transformed camera points" if i == 0 else None,
        )

        ax = axs[1]
        ax.plot(
            points_in_image[i, 0],
            points_in_image[i, 1],
            "*",
            markersize=20,
            color="darkgreen",
            alpha=1,
            label="Selected camera points" if i == 0 else None,
        )
        ax.plot(
            camera_points_exp[i, 0],
            camera_points_exp[i, 1],
            "+",
            markersize=20,
            color="red",
            alpha=1,
            label="Transformed map points" if i == 0 else None,
        )

    fig.tight_layout()
    return axs


def plot_camera(
    ax,
    unproject_points,
    camera_position_in_world,
    width=int,
    height=int,
    scale=1,
    T_axes_from_world2: np.ndarray = np.eye(3),
):
    center = hmult(T_axes_from_world2, camera_position_in_world[0:2])
    h, w = height, width

    N = 10
    u = np.linspace(0, w - 1, N)
    v = np.linspace(0, h - 1, N)
    uv, vv = np.meshgrid(u, v)
    grid_shape = uv.shape
    points_image = np.stack([uv.reshape(-1), vv.reshape(-1)], axis=1)

    xyz = unproject_points(points_image, scale)

    # Drop z
    xy = xyz[:, 0:2]

    xy = hmult(T_axes_from_world2, xy)

    # Back to grid
    xv = xy[:, 0].reshape(grid_shape)
    yv = xy[:, 1].reshape(grid_shape)

    ax.plot([center[0]], [center[1]], "*")
    for i, j in [(0, 0), (0, -1), (-1, -1), (-1, 0)]:
        ax.plot([center[0], xv[i, j]], [center[1], yv[i, j]], "-", color="gray")
    # Rows
    for i in range(xv.shape[0]):
        ax.plot(xv[i, :], yv[i, :], "-", color="black")
    # Cols
    for i in range(xv.shape[1]):
        ax.plot(xv[:, i], yv[:, i], "-", color="black")


class MouseHandler:
    """
    Helper class to click on images
    Example:
       image_clicker = MouseHandler(im, "test_name", is_polygon=True)
       cv2.startWindowThread()
       image_clicker.start()
       cv2.waitKey(0)
       cv2.destroyAllWindows()

       polygon = np.array(image_clicker.positions)

    """

    def __init__(self, img: np.ndarray, window_name: str, is_polygon: bool = False):
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.window_name = window_name
        self.is_polygon = is_polygon
        self.positions = []

    def start(self, window_flags: int = cv2.WINDOW_AUTOSIZE):
        cv2.namedWindow(self.window_name, window_flags)
        cv2.setMouseCallback(self.window_name, self)
        cv2.imshow(self.window_name, self.img)

    def __call__(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.positions.append((x, y))
            count = len(self.positions)
            if self.is_polygon:
                color = [0, 0, 200]
            else:
                color = [0, 0, 0]
                color[count % 3] = 200
            self.img = cv2.circle(self.img, (x, y), radius=10, color=color, thickness=5)
            if count > 1 and self.is_polygon:
                prev_x, prev_y = self.positions[count - 2]
                self.img = cv2.line(
                    self.img, (x, y), (prev_x, prev_y), color=color, thickness=3
                )
            cv2.imshow(self.window_name, self.img)
