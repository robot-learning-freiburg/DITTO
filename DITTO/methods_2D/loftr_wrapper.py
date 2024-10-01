import numpy as np
import cv2
import torch
import casino

import matplotlib.pyplot as plt
import skimage

import torch
import gc

try:
    import kornia
except Exception as e:
    import logging

    logging.error(f"Couldn't load kornia {e = }")

LOFTR_CFG_DICT = kornia.feature.loftr.loftr.default_cfg.copy()
LOFTR_CFG_DICT["match_coarse"]["thr"] = (
    LOFTR_CFG_DICT["match_coarse"]["thr"] * 0.1
)  # default is 0.2 --> lower threshold to 10%
loftr = kornia.feature.LoFTR(pretrained="indoor_new", config=LOFTR_CFG_DICT)


# TODO Move to vis_helpers.py
def visualize_loftr(image_a, image_b, seg_a, points_a, points_b, confidences):
    cmap = "gray" if image_a.ndim == 2 or image_a.shape[-1] == 1 else None

    fig, axes = plt.subplots(nrows=1, ncols=3, dpi=200)

    axes[0].imshow(image_a, cmap=cmap)
    axes[0].scatter(points_a[:, 1], points_a[:, 0], s=1, c=confidences)

    axes[1].imshow(seg_a)

    axes[2].imshow(image_b)
    axes[2].scatter(points_b[:, 1], points_b[:, 0], s=1, c=confidences)
    fig.suptitle("LoFTR Visualizaiton")
    fig.show()


def get_loftr_keypoints_in_mask(
    image_a, image_b, seg_a, debug_vis: bool = False, clean_up: bool = False
):
    loftr.to(casino.learning.DEFAULT_TORCH_DEVICE)

    gray_img_a = cv2.cvtColor(image_a, cv2.COLOR_RGB2GRAY)
    gray_img_b = cv2.cvtColor(image_b, cv2.COLOR_RGB2GRAY)

    input_dict = {
        "image0": torch.from_numpy(gray_img_a)[None][None].to(
            casino.learning.DEFAULT_TORCH_DEVICE
        )
        / 255.0,
        "image1": torch.from_numpy(gray_img_b)[None][None].to(
            casino.learning.DEFAULT_TORCH_DEVICE
        )
        / 255.0,
        "mask0": torch.from_numpy(seg_a[None, ...]).to(
            casino.learning.DEFAULT_TORCH_DEVICE, torch.uint8
        ),
    }
    with torch.no_grad():
        return_dict = loftr(input_dict)

    del input_dict

    all_keypoints_a = (
        np.asarray(return_dict["keypoints0"].cpu()).round().astype(np.uint16)
    )
    all_keypoints_b = (
        np.asarray(return_dict["keypoints1"].cpu()).round().astype(np.uint16)
    )
    all_confidences = np.asarray(return_dict["confidence"].cpu())

    object_keypoints_idx = seg_a[all_keypoints_a[..., 1], all_keypoints_a[..., 0]]

    points_a = all_keypoints_a[object_keypoints_idx, ::-1]
    points_b = all_keypoints_b[object_keypoints_idx, ::-1]
    confidences = all_confidences[object_keypoints_idx, ...]

    # TODO Filter to be in a known mask?

    if debug_vis:
        visualize_loftr(image_a, image_b, seg_a, points_a, points_b, confidences)

    loftr.to("cpu")
    casino.hardware.clear_torch_memory()

    return points_a, points_b, confidences

# TODO Make this a general function? --> 
# get_mask_based_on_keypoints(
#   image_a, 
#   image_b, 
#   seg_a, 
#   keypoint_extractor: CorrespondencesInMaskMethod, 
#   debug_vis: bool = False
# )

def get_mask_based_on_loftr_keypoints(image_a, image_b, seg_a, debug_vis: bool = False):
    points_a, points_b, confidence = get_loftr_keypoints_in_mask(
        image_a, image_b, seg_a, debug_vis=debug_vis
    )
    # Fit a convex hull around the points
    seg_b = np.zeros_like(seg_a, dtype=bool)
    seg_b[points_b[:, 0], points_b[:, 1]] = True
    seg_b = skimage.morphology.convex_hull_image(seg_b)

    # TODO Should be refactored out
    new_masks = casino.masks.equal_max_bbox(np.array([seg_a, seg_b]))
    seg_a_square, seg_b_square = new_masks[0], new_masks[1]

    image_a_cropped = casino.masks.get_segment_crop(image_a.copy(), seg_a_square)
    image_b_cropped = casino.masks.get_segment_crop(image_b.copy(), seg_b_square)
    seg_a_cropped = casino.masks.get_segment_crop(seg_a.copy(), seg_a_square)

    offset_points_a = casino.masks.mask_top_left_bbox(seg_a_square)
    offset_points_b = casino.masks.mask_top_left_bbox(seg_b_square)

    return (
        image_a_cropped,
        image_b_cropped,
        seg_a_cropped,
        offset_points_a,
        offset_points_b,
        seg_b,
    )
