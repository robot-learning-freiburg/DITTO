import numpy as np

import casino

from flow_control.flow.flow_plot import FlowPlot
from flow_control.flow.module_raft import FlowModule

import matplotlib.pyplot as plt


flow_module = FlowModule()
flow_module.model.to("cpu")


def track_2D_keypoints_flow_step(
    keypoints,
    image_a,
    image_b,
    flow_module: FlowModule = flow_module,
    debug_vis: bool = False,
):
    """
    Arguments:
        keypoints: list or array of (x,y) keypoints
        image_a: start image
        image_b: end image
     Returns:
        keypoints in image_b: how keypoints on image_a moved to on image_b
            potentially returns non-int
    """
    flow_module.flow_prev = None

    # RAFT requires to be of size 128!
    og_h, og_w = image_a.shape[:-1]
    pad_h = max(128 - og_h, 0)
    pad_w = max(128 - og_w, 0)
    image_a_padded = np.pad(image_a, pad_width=((0, pad_h), (0, pad_w), (0, 0)))
    image_b_padded = np.pad(image_b, pad_width=((0, pad_h), (0, pad_w), (0, 0)))

    flow_module.model.to(casino.learning.DEFAULT_TORCH_DEVICE)
    flow_padded = flow_module.step(image_a_padded, image_b_padded, use_prev=False)

    flow = flow_padded[:og_h, :og_w]
    if not isinstance(keypoints, np.ndarray):
        keypoints = np.asarray(keypoints)
    change = flow[keypoints[:, 0], keypoints[:, 1]][..., ::-1]  # Convert back to xy

    if debug_vis:
        # TODO Merge with show_registration2d in vis_helpers.py
        fp = FlowPlot()
        flow_image = fp.compute_image(flow)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        [x.set_axis_off() for x in ax]
        ax[0].imshow(image_a)
        ax[1].imshow(image_b)
        ax[2].imshow(flow_image)
        plt.show()

    flow_module.model.to("cpu")
    casino.hardware.clear_torch_memory()

    return (keypoints.copy() + change).round().astype(int)


def get_flow_keypoints_in_mask(
    image_a,
    image_b,
    seg_a,
    flow_module: FlowModule = flow_module,
    debug_vis: bool = False,
):
    keypoints_a = casino.masks.mask_to_coords(seg_a)
    keypoints_b = track_2D_keypoints_flow_step(
        keypoints_a, image_a, image_b, flow_module, debug_vis=debug_vis
    )
    assert keypoints_a.shape == keypoints_b.shape
    confidences = np.ones((keypoints_a.shape[0]))
    # TODO Filter to be in a known mask?
    return keypoints_a, keypoints_b, confidences


def warp_masks_flow(
    image_a,
    image_b,
    seg_a,
    flow_module: FlowModule = flow_module,
    debug_vis: bool = False,
):
    coords_b = get_flow_keypoints_in_mask(
        image_a, image_b, seg_a, flow_module, debug_vis=debug_vis
    )
    coords_b = casino.masks.filter_coords(
        coords_b, image_b.shape[:2], return_mask=False
    )
    seg_b = casino.masks.coords_to_mask(
        coords_b,
        seg_a.shape,
        use_convex_hull=False,
        use_dilation=False,
        use_opening=True,
        use_aa_bbox=False,
    )
    return seg_b
