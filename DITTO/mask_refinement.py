import numpy as np
from typing import List, Dict, Callable, Union
import functools
import open3d as o3d

from casino.pointcloud import to_o3d

from skimage.measure import label


def calculate_uni_overlap(mask_1, mask_2, m1_to_m2: bool = True):
    """
    mask_1 --> mask_2; m1_to_m2=True
    or
    mask_2 --> mask_1; m2_to_m1=True
    """
    assert mask_1.shape == mask_2.shape
    norm = np.sum(mask_1 if m1_to_m2 else mask_2)
    return np.sum(mask_1 * mask_2) / norm


def calculate_bi_overlap(mask_1, mask_2):
    assert mask_1.shape == mask_2.shape
    return np.sum(mask_1 * mask_2) / np.sum(np.logical_or(mask_1, mask_2))


def get_refined_mask_idx(
    given_mask: np.ndarray,
    sam_output_dict: List[Dict[str, np.ndarray]],
    overlap_calculator: Union[
        calculate_uni_overlap, calculate_bi_overlap
    ] = calculate_bi_overlap,
    threshold: float = 0.25,
):
    """
    Returns the index of mask with the most overlap uni-directional measured to the given mask
    """
    assert "segmentation" in sam_output_dict[0].keys()

    all_overlaps = [
        overlap_calculator(given_mask, mask["segmentation"]) for mask in sam_output_dict
    ]
    idx = np.argmax(all_overlaps)
    print(all_overlaps[idx])
    return idx if all_overlaps[idx] >= threshold else None


def get_refined_mask_indices_above_threshold(
    given_mask: np.ndarray,
    sam_output_dict: List[Dict[str, np.ndarray]],
    return_above_threshold: float = 0.7,
    overlap_calculator: Union[
        calculate_uni_overlap, calculate_bi_overlap
    ] = calculate_uni_overlap,
):
    assert "segmentation" in sam_output_dict[0].keys()
    all_overlaps = [
        overlap_calculator(given_mask, mask["segmentation"]) for mask in sam_output_dict
    ]
    return list(np.argwhere(np.array(all_overlaps) > return_above_threshold)[:, 0])


def min_mean_reduction(points: np.ndarray, K: Union[float, int]):
    """
    returns the mean of the first N points, where K could be
    - K <= 1. (float): percentage of smallest points used for calculating the min
    - K >= 1  (int)  : fixed N points
    """
    if K < 1.0 or type(K) == float:
        K = int(K * len(points))
    return np.mean(sorted(points)[:K])


def get_closest_pointcloud_idx(
    given_pcd: np.ndarray,
    all_pcds: List[np.ndarray],
    skip_pcds_idxs: List[int] = [],
    reduction_fn: Callable = functools.partial(min_mean_reduction, K=0.01),
    tolerance: float = 1e-6,
    visualize_K_closest: int = 0,
):
    """
    reduction_fn: np.min, np.mean
    """
    distances = []

    given_pcd_o3d = to_o3d(given_pcd)

    for idx, current_pcd in enumerate(all_pcds):
        if idx in skip_pcds_idxs:
            distances.append(np.inf)
            continue

        # Returns closest distance for each point in the object point cloud
        # to the given point cloud
        dists = given_pcd_o3d.compute_point_cloud_distance(to_o3d(current_pcd))
        # Reduction
        dist = reduction_fn(dists)
        # If pointcloud has NaNs this will return 0.0
        # or if we found the same pointcloud as the given one!
        dist = np.inf if dist < tolerance else dist
        distances.append(dist)

    min_indices = np.argsort(distances)

    if visualize_K_closest > 0:
        o3d.visualization.draw_geometries(
            [
                to_o3d(
                    all_pcds[idx],
                    # TODO do this with virdis?
                    color=np.array([0.0, 1.0, (i + 1) / visualize_K_closest]),
                )
                for i, idx in enumerate(min_indices[1:visualize_K_closest])
            ]
            + [to_o3d(given_pcd, color=np.array([1.0, 0.0, 0.0]))]
            + [
                to_o3d(
                    all_pcds[min_indices[0]],
                    # TODO do this with virdis?
                    color=np.array([0.0, 0.0, 1.0]),
                )
            ]
        )

    return min_indices[0]


def get_largest_cc(seg_mask):
    labels = label(seg_mask)
    assert labels.max() != 0  # assume at least 1 connected component
    # Select the largest connected component
    largest_mask = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largest_mask
