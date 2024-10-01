from typing import List, Callable, Optional

import numpy as np


def warp_2D_trajectory(
    demo_rgb: np.ndarray,
    live_rgb: np.ndarray,
    demo_trajectory: np.ndarray,
    step_function: Callable,
    with_keypoints: Optional[np.ndarray] = None,
):
    """
    Finds the displacement of the first keypoint(s) in the trajectory from the demo_rgb
    to live_rgb and shifts the whole trajectory with that

    If `with_keypoints` is specified, we will use them as input to the step_function
    """
    # Extract the change of the first keypoints when no keypoints are given
    # Otherwise we will calculate the change of the given keypoints
    demo_keypoints = (
        demo_trajectory[0, ...] if with_keypoints is None else with_keypoints
    )
    live_keypoints = step_function(
        demo_keypoints,
        demo_rgb,
        live_rgb,
    )
    # Calculate the change for the keypoints
    change = live_keypoints - demo_keypoints
    # Apply change to all
    warped_demo_trajectory = demo_trajectory.copy().astype(np.float32)
    warped_demo_trajectory += change
    return warped_demo_trajectory


def track_2D_keypoints(
    keypoints: np.ndarray, rgb_images: List[np.ndarray], step_function
) -> np.ndarray:
    """
    Given an initial set of keypoints, this functions tracks them through all provided images
    This function should be independent of where the data comes from etc.

    step_function gets called multiple times repeadetly
    """
    assert keypoints.shape[1] == 2
    assert keypoints.ndim == 2

    keypoints_list = [keypoints.copy()]
    for image_i, image_i_1 in zip(rgb_images[:-1], rgb_images[1:]):
        keypoints = step_function(keypoints, image_i, image_i_1).round().astype(int)
        keypoints_list.append(keypoints)
    return np.array(keypoints_list)
