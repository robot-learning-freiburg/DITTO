import logging
import pathlib
import time
import traceback
from typing import List, Optional

import casino
import numpy as np
import tqdm
import tyro
import yaml

import wandb
from DITTO.config import BASE_RECORDING_PATH, RESULTS_DIR, TIME_STEPS
from DITTO.data import Hands23Dataset, get_all_runs
from DITTO.methods_2D import CorrespondencesInMaskMethod, MaskDetector
from DITTO.trajectory import Trajectory
from DITTO import wandb_helper

METHODS_TO_TEST = [
    (MaskDetector.identity, CorrespondencesInMaskMethod.flow),
    (MaskDetector.cnos, CorrespondencesInMaskMethod.flow),
    (MaskDetector.identity, CorrespondencesInMaskMethod.loftr),
    (MaskDetector.cnos, CorrespondencesInMaskMethod.loftr),
]


def get_method_key(mask_warp_method, correspondence_method):
    return f"{mask_warp_method.name}/{correspondence_method.name}"


def process_episode(run_path: pathlib.Path, log_wandb: bool = True):
    trajectory: Trajectory = Trajectory.from_hands23(
        Hands23Dataset(run_path, lazy_loading=True), n_frames=TIME_STEPS
    )

    # Evaluate the following metrics within an episode
    # % Fully Tracked (i.e. at leat three keypoints between two timesteps) --> No
    # Mean of: absolute keypoints between two timesteps
    # Mean of: % keypoints in GT mask

    # Methods: 2D Correspondces Matchers

    # Input: rgb_img_a, rgb_img_b, mask_a (i.e. seed keypoints)
    # Output: keypoints (in rgb_img_a AND rgb_img_b)
    # GT: mask_b

    if log_wandb:
        config = {
            "task": run_path.parent.stem,
            "episode": run_path.stem,
            "run_path": run_path,
        }
        wandb.init(
            project="[VideoImitation] Intra Episode Correspondence Tracker",
            job_type="eval",
            name=config["task"] + "/" + config["episode"],  # Is this a good idea?
            config=config,
        )

    results = casino.special_dicts.AccumulatorDict(
        accumulator=lambda dict_entry, new_value: dict_entry + [new_value]
    )

    for mask_warp_method, correspondence_method in METHODS_TO_TEST:
        method_key = get_method_key(mask_warp_method, correspondence_method)
        results[f"{method_key}/N_absolute_keypoints"] = []
        results[f"{method_key}/percentage_in_mask"] = []
        results[f"{method_key}/duration"] = []

        if log_wandb:
            wandb.define_metric(
                f"{method_key}/N_absolute_keypoints",
                summary="mean",
            )
            wandb.define_metric(
                f"{method_key}/percentage_in_mask",
                summary="mean",
            )
            wandb.define_metric(
                f"{method_key}/duration",
                summary="mean",
            )

    for frame_i in tqdm.tqdm(range(0, trajectory.n_frames - 1)):
        rgb_a = trajectory.rgb_imgs[frame_i]
        rgb_b = trajectory.rgb_imgs[frame_i + 1]
        seg_a = trajectory.object_masks[frame_i, ..., 0]

        # Get "GT" Mask --> TODO: Make sure this is actually correct
        seg_b = trajectory.object_masks[frame_i + 1, ..., 0]

        for mask_warp_method, correspondence_method in METHODS_TO_TEST:
            start = time.time()
            (
                rgb_a_cropped,
                rgb_b_cropped,
                seg_a_cropped,
                offset_points_a,
                offset_points_b,
                _,
            ) = mask_warp_method(rgb_a, rgb_b, seg_a)
            keypoints_a_cropped, keypoints_b_cropped, confidences = (
                correspondence_method(rgb_a_cropped, rgb_b_cropped, seg_a_cropped)
            )
            keypoints_a = keypoints_a_cropped + offset_points_a
            keypoints_b = keypoints_b_cropped + offset_points_b
            duration = time.time() - start

            N_absolute_keypoints = keypoints_b.shape[0]
            safe_keypoints_b = casino.masks.filter_coords(
                keypoints_b, shape=seg_a.shape
            )
            percentage_in_mask = (
                (
                    np.count_nonzero(
                        seg_b[safe_keypoints_b[:, 0], safe_keypoints_b[:, 1]]
                    )
                    / N_absolute_keypoints
                )
                if N_absolute_keypoints > 0
                else 0.0
            )

            method_key = get_method_key(mask_warp_method, correspondence_method)
            local_result_dict = {
                f"{method_key}/N_absolute_keypoints": N_absolute_keypoints,
                f"{method_key}/percentage_in_mask": percentage_in_mask,
                f"{method_key}/duration": duration,
            }

            if log_wandb:
                wandb.log({"frame_i": frame_i, **local_result_dict})

            results.increment_dict(local_result_dict)

        if log_wandb:
            wandb.log(
                {
                    "frame_i": frame_i,
                    "imgs": wandb_helper.create_rgb_with_masks(
                        [rgb_a, rgb_b], [seg_a, seg_b]
                    ),
                }
            )

    return results.as_default_dict()


def main(
    base_path: pathlib.Path = BASE_RECORDING_PATH,
    only_sessions: Optional[List[str]] = ["24_01_04", "24_01_05"],
    output_dir: pathlib.Path = RESULTS_DIR / "intra_correspondence",
    logging_file: pathlib.Path = pathlib.Path("logs")
    / "eval_correspondences_within_episode.log",
    log_wandb: bool = False, # This is no longer supported, but might still work
    allow_debugging: bool = False
):
    output_dir.mkdir(exist_ok=True, parents=True)
    single_dir = output_dir / "single_runs"
    single_dir.mkdir(exist_ok=True, parents=True)

    all_runs = get_all_runs(base_path, only_keywords=only_sessions)
    all_results = {}

    logging_file.parent.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        filename=str(logging_file),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d, %H:%M:%S",
        level=logging.DEBUG,
        force=True,  # Removes all previously handlers of the logger and forces to use the file
    )

    logging.info(f"Evaluating on {len(all_runs)} with {all_runs = }")

    for episode_path in tqdm.tqdm(all_runs):
        # Comment me out to let debugger catch try..excepts
        if allow_debugging:
            episode_result = process_episode(episode_path, log_wandb=log_wandb)

        try:
            episode_result = process_episode(episode_path, log_wandb=log_wandb)
            all_results[str(episode_path)] = episode_result

            if log_wandb:
                wandb.finish()

            with (single_dir / ("_".join(episode_path.parts[-2:]) + ".yaml")).open(
                "w"
            ) as f:
                yaml.dump(episode_result, f)

            with (output_dir / "all_results.yaml").open("w") as f:
                yaml.dump(all_results, f)

        except Exception as e:
            logging.warn(
                f"Encoutered exception {e = } while processing {episode_path = }\n{traceback.format_exc()}"
            )


if __name__ == "__main__":
    tyro.cli(main)
