import logging

logging.basicConfig(level=logging.INFO)

import gc
import os
from casino.learning import DEFAULT_TORCH_DEVICE
from casino.hardware import clear_torch_memory

from hydra import initialize, compose
from hydra.utils import instantiate
import torch
import numpy as np
import matplotlib.pyplot as plt
import pathlib

import casino
import CNOS
import matplotlib.pyplot as plt
import numpy as np
import torch
from casino.learning import DEFAULT_TORCH_DEVICE
from CNOS.model.loss import Similarity
from CNOS.model.utils import Detections
from CNOS.utils.bbox_utils import CropResizePad
from hydra import compose, initialize
from hydra.utils import instantiate

from DITTO.config import REPO_ROOT, SAMConfig
from DITTO.mask_refinement import get_largest_cc
from DITTO.vis_helpers import (
    convert_cnos_detections_to_original,
    overlay_mask_edge,
    show_sam_results,
)

CNOS_DIR = pathlib.Path(CNOS.__path__[0])
from PIL import Image


def create_template(rgb_img: np.array, masks: np.array):
    """
    Creates input for CNOS. Specifially returns th

    It also permutes and changes
    """
    # Create new image
    templates = []
    boxes = []

    for mask in masks:
        masked_rgb_img = rgb_img.copy()
        masked_rgb_img[np.logical_not(mask)] = (0, 0, 0)
        # Convert to float
        if masked_rgb_img.dtype == np.uint8:
            masked_rgb_img = (masked_rgb_img / 255.0).astype(np.float32)
        # Make it a proper torch tensor image
        masked_rgb_img = torch.from_numpy(masked_rgb_img)
        masked_rgb_img = masked_rgb_img.permute(2, 0, 1)
        templates.append(masked_rgb_img)

        boxes.append(
            Image.fromarray(mask).getbbox()
        )  # TODO use this function in casino!

    return torch.stack(templates), torch.tensor(np.array(boxes))


class CNOSDetect:
    def __init__(self, sam_cfg: SAMConfig = SAMConfig()):
        self.metric = Similarity()

        stability_score_thresh = 0.97
        conf_dir = (CNOS_DIR / "configs").absolute()
        cur_dir = pathlib.Path(__file__).parent.resolve()
        relative_path_to_cnos_config_dir = os.path.relpath(conf_dir, cur_dir)

        with initialize(config_path=relative_path_to_cnos_config_dir):
            cfg = compose(config_name="run_inference.yaml")
        # This creates a reference!
        cfg_segmentor = cfg.model.segmentor_model

        if "fast_sam" in cfg_segmentor._target_:
            logging.info("Using FastSAM, ignore stability_score_thresh!")
        else:
            cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
            cfg.model.segmentor_model.sam.checkpoint_dir = sam_cfg.model_weights.parent

        logging.info("Initializing model")
        self.model = instantiate(cfg.model)

        self.proposal_processor = CropResizePad(
            cfg.model.descriptor_model.image_size
        )  # 244

    def _descriptor_to_device(self, device):
        logging.debug(f"Moving model to {device}. (descriptor)")
        self.model.descriptor_model.model = self.model.descriptor_model.model.to(device)
        self.model.descriptor_model.model.device = device

    def _segmentor_to_device(self, device):
        logging.debug(f"Moving model to {device}. (segmentor)")
        # if there is predictor in the model, move it to device
        if hasattr(self.model.segmentor_model, "predictor"):
            self.model.segmentor_model.predictor.model = (
                self.model.segmentor_model.predictor.model.to(device)
            )
        else:
            self.model.segmentor_model.model.setup_model(device=device, verbose=True)
        # logging.info(f"Moving models to {device} done!")

        # Manually overwrite some parameters for the segmentor to get more masks
        # TODO This is yet to verify if causing any problems
        self.model.segmentor_model.box_nms_thresh = 0.97
        self.model.segmentor_model.pred_iou_thresh = 0.5
        self.model.segmentor_model.crop_nms_thresh = 0.97
        self.model.segmentor_model.stability_score_thres = 0.9

    def models_to_device(self, device):
        self._descriptor_to_device(device)
        self._segmentor_to_device(device)

    def clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()

    def warp_masks(
        self,
        rgb_img_a: np.ndarray,
        rgb_img_b: np.ndarray,
        input_mask_a: np.ndarray,
        return_connected_single_mask: bool = True,
        debug_vis: bool = False,
    ):
        casino.hardware.clear_torch_memory()
        self.models_to_device(DEFAULT_TORCH_DEVICE)

        rgb_img_a = np.array(rgb_img_a)
        rgb_img_b = np.array(rgb_img_b)
        input_mask_a = np.array(input_mask_a)

        # Make sure we have a batch on the masks
        single_mask = input_mask_a.ndim == 2
        if single_mask:
            input_mask_a = input_mask_a[None, ...]

        # TODO Activate write_to_disk
        # Create template
        masked_rgb_imgs, boxes = create_template(rgb_img_a, input_mask_a)
        # Does this needs to be batched?
        templates = self.proposal_processor(images=masked_rgb_imgs, boxes=boxes)

        ref_feats = self.model.descriptor_model.compute_features(
            templates.cuda(), token_name="x_norm_clstoken"
        )
        casino.hardware.clear_torch_memory()

        raw_detections = self.model.segmentor_model.generate_masks(rgb_img_b)
        # Why did this work on the desktop comptuer?
        # for key, value in raw_detections.items():
        #     raw_detections[key] = value.detach().to("cpu")
        detections = Detections(raw_detections)
        casino.hardware.clear_torch_memory()

        descriptors = self.model.descriptor_model.forward(rgb_img_b, detections)
        casino.hardware.clear_torch_memory()

        scores = self.metric(descriptors[:, None, :], ref_feats[None, :, :])

        matches = torch.argmax(scores, axis=0).to("cpu")

        # TODO Iterate over all matches
        warped_masks = detections.masks[matches].detach().cpu().numpy()

        # Just to make sure SAM did not fuck up with small pixels somewhere
        if return_connected_single_mask:
            warped_masks = np.array([get_largest_cc(mask) for mask in warped_masks])

        casino.hardware.clear_torch_memory()

        if debug_vis:
            tmps = templates.permute(0, 2, 3, 1)

            static_rows = 2
            fig, axes = plt.subplots(nrows=static_rows + len(tmps), ncols=2, dpi=200)

            # TODO Set titles
            axes[0][0].set_title("Source image")
            axes[0][0].imshow(rgb_img_a)
            axes[0][1].set_title("Target image")
            axes[0][1].imshow(rgb_img_b)

            axes[1][0].set_title("Input Masks")
            rgb_a_mask_edges = rgb_img_a.copy()
            for mask in input_mask_a:
                rgb_a_mask_edges = overlay_mask_edge(rgb_a_mask_edges, mask)
            axes[1][0].imshow(rgb_a_mask_edges)

            axes[1][1].set_title("SAM Masks")
            show_sam_results(
                convert_cnos_detections_to_original(detections), axes[1][1], rgb_img_b
            )

            for idx, (tmp, warped_mask) in enumerate(zip(tmps, warped_masks)):
                axes[static_rows + idx][0].set_title("Input Template")
                axes[static_rows + idx][0].imshow(tmp)
                axes[static_rows + idx][1].set_title("Matched mask")
                axes[static_rows + idx][1].imshow(warped_mask)

            fig.suptitle("CNOS Visualization")
            fig.tight_layout()

        self.models_to_device("cpu")
        casino.hardware.clear_torch_memory()

        return warped_masks if not single_mask else warped_masks[0]


# Initialize globally
cnos_pipeline = CNOSDetect()


def cnos_mask_extraction(image_a, image_b, seg_a, debug_vis: bool = False):
    seg_b = cnos_pipeline.warp_masks(image_a, image_b, seg_a, debug_vis=debug_vis)

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
