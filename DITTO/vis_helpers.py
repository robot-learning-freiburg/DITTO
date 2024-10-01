import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import open3d as o3d
import casino
import copy
from typing import Optional, Tuple

from flow_control.flow.flow_plot import FlowPlot
from CNOS.model.utils import Detections as CNOSSAMDetections


def smooth_line(arr, samples=100):
    print("XXX", arr.shape, arr.dtype)
    x, y = zip(*arr)
    # create spline function
    f, u = interpolate.splprep([x, y], s=0)
    # create interpolated lists of points
    xint, yint = interpolate.splev(np.linspace(0, 1, samples), f)
    return np.stack((xint, yint), axis=1)


def convert_cnos_detections_to_original(cnos_detections: CNOSSAMDetections):
    masks = cnos_detections.masks.cpu().numpy().astype(bool)
    # TODO are these all?
    return [
        {"segmentation": masks[i], "area": np.count_nonzero(masks[i])}
        for i in range(len(cnos_detections))
    ]


def show_sam_results(
    anns,
    ax,
    rgb_img=None,
    object_mask=None,
    sam_alpha: float = 0.35,
    object_alpha: float = 0.7,
):
    """
    Plots SAM result onto an image
    """
    if len(anns) == 0:
        return

    # TODO Why needed?
    # ax.set_autoscale_on(False)

    if not rgb_img is None:
        ax.imshow(rgb_img)

    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0

    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [sam_alpha]])
        img[m] = color_mask

    if not object_mask is None:
        color_mask = np.array([1.0, 0.0, 0.0, object_alpha])
        img[object_mask] = color_mask

    # Overlay
    ax.imshow(img)
    return ax


def get_bbox_center(x_start, x_stop, y_start, y_stop):
    return int((x_start + x_stop) / 2), int((y_start + y_stop) / 2)


def pred_bbox2rect(pred_bbox, c="r"):
    y_start, x_start, y_stop, x_stop = [float(x) for x in pred_bbox]
    rect = patches.Rectangle(
        (y_start, x_start),
        (y_stop - y_start),
        (x_stop - x_start),
        linewidth=1,
        edgecolor=c,
        facecolor="none",
    )
    return rect


def mask_edge(mask: np.ndarray):
    assert len(mask.shape) == 2
    edge = np.gradient(mask.astype(float))
    edge = (np.abs(edge[0]) + np.abs(edge[1])) > 0
    return edge


def to_uint8_image(image: np.ndarray):
    image = image.copy()
    is_uint8_img = image.dtype == np.uint8
    if is_uint8_img:
        return image
    return (image * 255).astype(np.uint8)


def to_float32_image(image: np.ndarray):
    image = image.copy()
    is_float_img = image.dtype in [np.float32, np.float64]
    if is_float_img:
        return image.astype(np.float32)
    return image.astype(np.float32) / 255.0


def overlay_mask_edge(
    image: np.ndarray, mask: np.ndarray, color: Tuple[int] = (0, 255, 0)
):
    image = to_uint8_image(image)
    edge = mask_edge(mask)
    image[edge] = color
    return to_float32_image(image)


def overlay_points(
    image: np.ndarray, points: np.ndarray, color: Tuple[int] = (0, 255, 0)
):
    assert image.ndim == 3 and image.shape[2] == 3
    assert points.ndim == 2 and points.shape[1] == 2
    image = to_uint8_image(image)
    image[points[:, 0], points[:, 1], :] = color
    return to_float32_image(image)


def overlay_on_image(
    img, seg: Optional[np.ndarray] = None, points: Optional[np.ndarray] = None
):
    if not seg is None:
        img = overlay_mask_edge(img, seg)
    elif not points is None:
        img = overlay_points(img, points)
    return to_float32_image(img)


def show_registration2d(
    image_a,
    image_b,
    flow: Optional[np.ndarray] = None,
    seg_a: Optional[np.ndarray] = None,
    points_a: Optional[np.ndarray] = None,
    seg_b: Optional[np.ndarray] = None,
    points_b: Optional[np.ndarray] = None,
):
    plot_flow = not flow is None
    if plot_flow:
        fp = FlowPlot()
        flow_image = fp.compute_image(flow)
    fig, ax = plt.subplots(1, 3 if plot_flow else 2, dpi=200)

    [x.set_axis_off() for x in ax]

    ax[0].imshow(overlay_on_image(image_a, seg_a, points_a))
    ax[1].imshow(overlay_on_image(image_b, seg_b, points_b))
    if plot_flow:
        ax[2].imshow(flow_image)
    fig.suptitle("Registration 2D")
    fig.tight_layout()
    fig.show()


def show_registration3d(pcd_a_valid, pcd_b_valid, trf_est, pcd_correspond: bool = True):
    pcd_a_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_a_valid[:, :3]))
    pcd_a_o3d.paint_uniform_color([1, 0, 0])  # RED

    pcd_b_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_b_valid[:, :3]))
    pcd_b_o3d.paint_uniform_color([0, 0, 1])  # BLUE

    pcd_a_o3d_trf = copy.deepcopy(pcd_a_o3d)
    pcd_a_o3d_trf.transform(trf_est)
    pcd_a_o3d_trf.paint_uniform_color([0, 1, 0])  # GREEN

    to_visualize = [pcd_a_o3d, pcd_b_o3d, pcd_a_o3d_trf]

    # Draw lines between corresponding points
    if pcd_correspond:
        points = np.concatenate(
            (
                casino.pointcloud.make_non_homoegeneous(pcd_a_valid),
                casino.pointcloud.make_non_homoegeneous(pcd_b_valid),
            ),
            axis=0,
        )
        lines = np.array(
            [
                np.arange(pcd_a_valid.shape[0]),
                pcd_a_valid.shape[0] + np.arange(pcd_a_valid.shape[0]),
            ],
            dtype=int,
        ).T
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        to_visualize.append(line_set)
    # line_set.colors = o3d.utility.Vector3dVector()

    o3d.visualization.draw_geometries(to_visualize)


def show_selected_points(xyz, rgb, points, color=np.array([1.0, 0.0, 0.0])):
    full_mask_a = np.ones(xyz.shape[:2], dtype=bool)
    o3d.visualization.draw_geometries(
        [
            casino.pointcloud.to_o3d(xyz[full_mask_a], color=rgb[full_mask_a] / 255.0),
            casino.pointcloud.to_o3d(
                casino.pointcloud.make_non_homoegeneous(points),
                color=color,
            ),
        ]
    )
