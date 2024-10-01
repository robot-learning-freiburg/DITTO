import logging
import pathlib
from typing import List

import cv2
import numpy as np
import tqdm
import tyro

from DITTO.mask_refinement import get_largest_cc
from DITTO.config import BASE_RECORDING_PATH
from DITTO.data import Hands23Dataset, get_all_runs
from DITTO.methods_2D.cnos_wrapper import cnos_pipeline


def main(
    folder_name: str = "cnos_redetected_masks_v2",
    overwrite: bool = False,
    visualize: bool = False,
    only_sessions: List[str] = [],
    base_path: pathlib.Path = BASE_RECORDING_PATH,
):
    all_runs = get_all_runs(base_path, only_keywords=only_sessions)

    for run in tqdm.tqdm(all_runs):
        print(f"processing {str(run)}")
        dataset = Hands23Dataset(run)

        out_path = dataset.recording_path / folder_name

        # if out_path.exists() and len(list(out_path.glob("*"))) and not overwrite:
        #     logging.warn(f"Skipping {out_path} because we do not overwrite.")
        #     continue

        out_path.mkdir(exist_ok=True, parents=True)

        frame_index_good_mask = dataset.time_steps_dict.get(
            "t_h23_good_object_mask", -1
        )
        if frame_index_good_mask == -1:  # Returned invalid time step
            # Will override with t_start, this is potenyially already overwritten with t_h23_object_manual
            frame_index_good_mask = dataset.time_steps_dict["t_start"]

        # Assume we had a good re-detection with SAM overlap?
        # frame_index_good_mask = 0

        rgb = dataset.get_rgb(frame_index_good_mask)
        object_mask = dataset.get_object_mask(
            frame_index_good_mask, refined=True, refined_cnos=False
        )[..., 0]

        if not np.any(object_mask):
            print(f"No object mask for {run} at {frame_index_good_mask = }")

        object_mask = get_largest_cc(object_mask)

        # TODO Maybe condition on this?
        # hand_bbox = dataset.get_bbox(frame_index_good_mask, "hand_bbox")
        # hand_center_2D = np.array(get_bbox_center(*hand_bbox))

        for t in tqdm.tqdm(range(len(dataset))):
            file_path = out_path / f"object_seg_{t:03}.png"

            if file_path.exists() and not overwrite:
                continue

            rgb_i = dataset.get_rgb(t)
            try:
                warped_mask = cnos_pipeline.warp_masks(
                    rgb_img_a=rgb,
                    rgb_img_b=rgb_i,
                    input_mask_a=object_mask,
                    return_connected_single_mask=False,  # Otherwise hand intersection won't work
                    debug_vis=visualize,
                )

                cv2.imwrite(
                    str(file_path),
                    warped_mask * 255,
                )
            except Exception as e:
                print(f"Encountered {e = } for {run} @ {t}")
                continue


if __name__ == "__main__":
    tyro.cli(main)
