import functools
import logging
import pathlib
import traceback
from typing import List

import cv2
import numpy as np
import open3d as o3d
import tqdm
import tyro
from casino.files import rmtree
from casino.learning import DEFAULT_TORCH_DEVICE
from casino.masks import mask_to_coords
from casino.pointcloud import to_o3d
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from skimage.measure import label

from DITTO.config import BASE_RECORDING_PATH, SAMConfig
from DITTO.data import Hands23Dataset, get_all_runs
from DITTO.mask_refinement import (
    calculate_bi_overlap,
    calculate_uni_overlap,
    get_closest_pointcloud_idx,
    get_largest_cc,
    get_refined_mask_idx,
    get_refined_mask_indices_above_threshold,
)
from DITTO.vis_helpers import overlay_mask_edge, show_sam_results


def main(
    folder_name: str = "sam_refined_v4",
    overwrite: bool = False,
    visualize: bool = False,
    only_sessions: List[str] = [],
    base_path: pathlib.Path = BASE_RECORDING_PATH,
    logging_file: pathlib.Path = pathlib.Path("logs") / "refine_with_sam.log",
):
    """
    This scripts parses the given Hands23 dataset
    and creates a refined mask for the object and container
    """
    # Load a sam model
    sam_config = SAMConfig()
    sam = sam_model_registry[sam_config.model_type](checkpoint=sam_config.model_weights)
    sam.to(device=DEFAULT_TORCH_DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)

    logging_file.parent.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        filename=str(logging_file),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
        force=True,  # Removes all previously handlers of the logger and forces to use the file
    )

    all_runs = get_all_runs(base_path, only_keywords=only_sessions)

    for run in tqdm.tqdm(all_runs):
        try:
            process_run(
                run,
                folder_name=folder_name,
                overwrite=overwrite,
                mask_generator=mask_generator,
                visualize=visualize,
            )
        except Exception as e:
            logging.warn(f"Skipping {str(run)} because of {traceback.format_exc()}")
            return


def process_run(run, folder_name, overwrite, mask_generator, visualize):
    logging.debug(f"Processing {str(run)}")

    dataset: Hands23Dataset = Hands23Dataset(run)

    out_path = dataset.recording_path / folder_name

    if out_path.exists() and not overwrite:
        logging.warn(f"Skipping {str(out_path)} because we do not overwrite.")
        return

    rmtree(out_path)
    out_path.mkdir(exist_ok=True, parents=True)

    t_first: int = 0

    if "t_h23_container_manual" in dataset.time_steps_dict.keys():
        container_segmentor = refine_with_container_mask
    elif "t_h23_object_is_goal" in dataset.time_steps_dict.keys():
        container_segmentor = refine_with_object_mask
    elif "t_object_close_to_goal" in dataset.time_steps_dict.keys():
        container_segmentor = refine_with_pointcloud_distance
    else:
        logging.info(
            f"{str(run)} missing container timestep annotation/no goal for this task"
        )
        container_segmentor = lambda dataset, mask_generator, visualize: np.zeros(
            (dataset.get_resolution())
        )

    container_seg_mask_object_i = container_segmentor(
        dataset, mask_generator, visualize=visualize
    )

    # Get refined object + goal mask in first image
    rgb_img_first = dataset.get_rgb(t_first)
    sam_masks_first = mask_generator.generate(rgb_img_first)

    t_start, _ = dataset.get_start_stop()

    # --- Object ---
    object_hands23_seg_mask_start = dataset.get_object_mask(
        t_start, refined=False, refined_cnos=False
    )[..., 0]
    object_mask_idx = get_refined_mask_idx(
        object_hands23_seg_mask_start,
        sam_masks_first,
        # If m1_to_m2=true: prefer larger masks
        # If m1_to_m2=false: prefer smaller masks
        # overlap_calculator=functools.partial(calculate_uni_overlap, m1_to_m2=True),
        # Or maximum overlap --> true IoU
        overlap_calculator=calculate_bi_overlap,
        threshold=0.25,
    )

    if not object_mask_idx is None:
        # Only write if redected the object
        object_seg_mask_first = sam_masks_first[object_mask_idx]["segmentation"]
        object_seg_mask_first = get_largest_cc(object_seg_mask_first)

    else:
        # Otherwise we will transfer the initial mask and hope for the best? :)
        object_seg_mask_first = object_hands23_seg_mask_start

    cv2.imwrite(
        str(out_path / f"object_seg_{t_first:03}.png"),
        object_seg_mask_first * 255,
    )

    # --- Goal ---
    container_mask_idx = get_refined_mask_idx(
        container_seg_mask_object_i,
        sam_masks_first,
        overlap_calculator=calculate_uni_overlap,  # Goal is potentially heavily overlapped
        threshold=0.25,
    )

    if not container_mask_idx is None and np.any(container_seg_mask_object_i):
        container_seg_mask_first = sam_masks_first[container_mask_idx]["segmentation"]
        container_seg_mask_first = get_largest_cc(container_seg_mask_first)
        cv2.imwrite(
            str(out_path / f"container_seg_{t_first:03}.png"),
            container_seg_mask_first * 255,
        )
    else:
        # Only write container if there is actually one
        container_seg_mask_first = np.zeros(dataset.get_resolution(), dtype=bool)

    # Done with the main part, rest is visualizing code
    if visualize:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(5, 5), dpi=200)
        for ax in axes.flat:
            ax.axis("off")

        fig.suptitle(f"Run: {run.parent.stem}/{run.stem}")

        current_row = 0
        # Inputs #
        axes[current_row][0].set_title("Input Object Mask")
        axes[current_row][0].imshow(
            overlay_mask_edge(rgb_img_first, object_hands23_seg_mask_start)
        )

        axes[current_row][1].set_title("Input Container Mask")
        axes[current_row][1].imshow(
            overlay_mask_edge(rgb_img_first, container_seg_mask_object_i)
        )

        current_row += 1
        # Row #

        show_sam_results(
            anns=sam_masks_first,
            ax=axes[current_row][0],
            rgb_img=rgb_img_first,
            object_mask=container_seg_mask_first,
        )
        axes[current_row][0].set_title(f"SAM Predictions @ t={t_first}")
        axes[current_row][1].imshow(container_seg_mask_first)
        axes[current_row][1].set_title(f"Container Seg Mask @ t={t_first}")
        current_row += 1

        # Row #
        show_sam_results(
            anns=sam_masks_first,
            ax=axes[current_row][0],
            rgb_img=rgb_img_first,
            object_mask=object_seg_mask_first,
        )
        axes[current_row][0].set_title(f"SAM Predictions @ t={t_first}")

        axes[current_row][1].imshow(object_seg_mask_first)
        axes[current_row][1].set_title(f"Object Seg Mask @ t={t_first}")

        fig.tight_layout()

        plt.show()


def refine_with_container_mask(
    dataset: Hands23Dataset, mask_generator, t_first: int = 0, visualize: bool = False
):
    t_container = dataset.time_steps_dict["t_h23_container_manual"]
    return dataset.get_goal_mask(t_container, refined=False)[..., 0]


def refine_with_object_mask(
    dataset: Hands23Dataset, mask_generator, t_first: int = 0, visualize: bool = False
):
    t_object_i = dataset.time_steps_dict["t_h23_object_is_goal"]
    return dataset.get_object_mask(t_object_i, refined=False, refined_cnos=False)[
        ..., 0
    ]


def refine_with_pointcloud_distance(
    dataset: Hands23Dataset, mask_generator, t_first: int = 0, visualize: bool = False
):
    t_object_i = dataset.time_steps_dict["t_object_close_to_goal"]
    object_hands23_seg_mask_object_i = dataset.get_object_mask(
        t_object_i, refined=False, refined_cnos=False
    )[..., 0]
    rgb_img_object_i = dataset.get_rgb(t_object_i)

    sam_masks_object_i = mask_generator.generate(rgb_img_object_i)

    object_mask_idx = get_refined_mask_idx(
        object_hands23_seg_mask_object_i,
        sam_masks_object_i,
        overlap_calculator=calculate_bi_overlap,
        threshold=0.0001,  # We assume there will always be some overlap
    )
    object_sam_seg_mask_object_i = sam_masks_object_i[object_mask_idx]["segmentation"]

    hand_hands23_seg_mask = dataset.get_hand_masks(t_object_i)[..., 0]
    hand_mask_indices = get_refined_mask_indices_above_threshold(
        hand_hands23_seg_mask, sam_masks_object_i
    )

    # We then look for the closest object in terms of point cloud distance
    object_pcd = dataset.get_pointcloud(
        t_object_i, mask_to_coords(object_sam_seg_mask_object_i)
    )
    # TODO Maybe we should use directly the 0-th image?
    all_pcds = [
        dataset.get_pointcloud(
            t_object_i,
            mask_to_coords(
                # Do largest CC to filter out big disconnected table etc.
                get_largest_cc(mask["segmentation"])
            ),
        )
        for mask in sam_masks_object_i
    ]

    container_mask_idx = get_closest_pointcloud_idx(
        object_pcd,
        all_pcds,
        skip_pcds_idxs=hand_mask_indices + [object_mask_idx],
        visualize_K_closest=10 if visualize else 0,
    )
    container_seg_mask_object_i = sam_masks_object_i[container_mask_idx]["segmentation"]
    # Refine the SAM mask
    container_seg_mask_object_i = get_largest_cc(container_seg_mask_object_i)

    if visualize:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(5, 5), dpi=200)
        # Row #
        show_sam_results(
            anns=sam_masks_object_i,
            ax=axes[0][0],
            rgb_img=rgb_img_object_i,
            object_mask=object_sam_seg_mask_object_i,
        )
        axes[0][0].set_title(f"SAM Predictions @ t={t_object_i}")

        axes[0][1].imshow(object_sam_seg_mask_object_i)
        axes[0][1].set_title(f"Object Seg Mask @ t={t_object_i}")

        # Row #
        show_sam_results(
            anns=sam_masks_object_i,
            ax=axes[1][0],
            rgb_img=rgb_img_object_i,
            object_mask=container_seg_mask_object_i,
        )
        axes[1][0].set_title(f"SAM Predictions @ t={t_object_i}")

        axes[1][1].imshow(container_seg_mask_object_i)
        axes[1][1].set_title(f"Container Seg Mask @ t={t_object_i}")

        fig.tight_layout()

        # 3D Visualization
        o3d_object_pcd = to_o3d(object_pcd, color=np.array([1.0, 0.0, 0.0]))
        o3d_all_pcds = [
            to_o3d(pcd, color=np.array([np.random.random(3)])) for pcd in all_pcds
        ]
        o3d.visualization.draw_geometries([o3d_object_pcd] + o3d_all_pcds)

    return container_seg_mask_object_i


if __name__ == "__main__":
    tyro.cli(main)
