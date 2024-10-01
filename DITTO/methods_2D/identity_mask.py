import numpy as np


def identity_mask_extraction(image_a, image_b, seg_a, debug_vis: bool = False):
    image_a_cropped = image_a
    image_b_cropped = image_b
    seg_a_cropped = seg_a

    offset_points_a = np.zeros((2,), dtype=int)
    offset_points_b = np.zeros((2,), dtype=int)

    # TODO Baseline that finds the mask?
    seg_b = None

    return (
        image_a_cropped,
        image_b_cropped,
        seg_a_cropped,
        offset_points_a,
        offset_points_b,
        seg_b,
    )
