from casino.enums import CallableEnum
import functools

from .flow_wrapper import get_flow_keypoints_in_mask, warp_masks_flow
from .loftr_wrapper import (
    get_loftr_keypoints_in_mask,
    get_mask_based_on_loftr_keypoints,
)

# from .superglue_wrapper import get_superglue_keypoints_in_mask
from .dino_wrapper import get_dino_keypoints_in_mask
from .cnos_wrapper import cnos_mask_extraction
from .identity_mask import identity_mask_extraction


# These methods share the same interface
# input: image_a, image_b, seg_a, debug_vis: bool = False
# output: keypoints_a, keypoints_b, confidance
class CorrespondencesInMaskMethod(CallableEnum):
    flow = functools.partial(get_flow_keypoints_in_mask)
    loftr = functools.partial(get_loftr_keypoints_in_mask)
    dino = functools.partial(get_dino_keypoints_in_mask)


# These methods share the same interface
# input: image_a, image_b, seg_a, debug_vis: bool = False
# output: seg_b
class MaskDetector(CallableEnum):
    cnos = functools.partial(cnos_mask_extraction)
    loftr = functools.partial(get_mask_based_on_loftr_keypoints)
    identity = functools.partial(identity_mask_extraction)
