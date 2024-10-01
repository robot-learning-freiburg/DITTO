import logging
import pathlib
import time
import traceback
from typing import Dict, List, Optional

import casino
import numpy as np
import tqdm
import tyro
import yaml

import wandb
from DITTO.config import BASE_RECORDING_PATH, RESULTS_DIR, TIME_STEPS
from DITTO.data import Hands23Dataset, get_all_runs
from DITTO.eval import eval_trajectories_3d, eval_trajectories_3d_old, eval_trajectories_3d_one_direction
from DITTO.geometry import extract_relative_transforms
from DITTO.methods_2D import CorrespondencesInMaskMethod, MaskDetector
from DITTO.mixing import mix_3D
from DITTO.tracking_3D import Step3DMethod
from DITTO.trajectory import Trajectory
from DITTO import wandb_helper

METHODS_TO_TEST: List[Step3DMethod] = [
    Step3DMethod.flow_ransac,  
    Step3DMethod.loftr_ransac,  
    Step3DMethod.cnos_flow_ransac,
    Step3DMethod.cnos_loftr_ransac,  
    Step3DMethod.dino_ransac,  
]


def get_method_key(method: Step3DMethod):
    return method.name


def process_episode(
    demo_run_path: pathlib.Path,
    live_run_paths: List[pathlib.Path],
    all_trajectories: Dict[pathlib.Path, Trajectory],
    log_wandb: bool = True,
):
    demo_trajectory: Trajectory = all_trajectories[demo_run_path]
    # Pre-Compute for fairness
    demo_trajectory.trajectory_3D

    rgb_demo = demo_trajectory.rgb_start
    xyz_demo = demo_trajectory.xyz_start

    if log_wandb:
        config = {
            "task": demo_run_path.parent.stem,
            "episode": demo_run_path.stem,
            "run_path": demo_run_path,
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

    # Setup accumalators
    for method in METHODS_TO_TEST:
        method_key = get_method_key(method)
        results[f"{method_key}/translation_error"] = []
        results[f"{method_key}/rotation_error"] = []
        results[f"{method_key}/duration"] = []

        if log_wandb:
            wandb.define_metric(
                f"{method_key}/translation_error",
                summary="mean",
            )
            wandb.define_metric(
                f"{method_key}/rotation_error",
                summary="mean",
            )
            wandb.define_metric(
                f"{method_key}/duration",
                summary="mean",
            )

    # Iterate over all other trajectories
    for live_run_path in tqdm.tqdm(live_run_paths):
        live_trajectory: Trajectory = all_trajectories[live_run_path]
        # TODO Set warp+step functions?
        # Pre-Compute for fairness
        live_trajectory.trajectory_3D

        # Get live observations
        rgb_live = live_trajectory.rgb_start
        xyz_live = live_trajectory.xyz_start

        # rgb_b = live_trajectory.rgb_start
        # seg_a = demo_trajectory.object_mask_start

        # Get "GT" Mask --> TODO: Make sure this is actually correct
        # seg_b = live_trajectory.object_mask_start

        for method in METHODS_TO_TEST:
            logging.debug(f"Running {method = }")
            # demo_trajectory.reset_cached()
            demo_trajectory._warp_function_3D = method

            # -------- Perform actual computation ---------------
            start = time.time()
            object_warped_3D_trajectory = demo_trajectory.warp_3D_onto_live_frame(
                rgb_live, xyz_live, use_goal_mask=False, debug_vis=False
            )
            if demo_trajectory.has_goal:
                # tennisball_cup/003
                goal_warped_3D_trajectory = demo_trajectory.warp_3D_onto_live_frame(
                    rgb_live, xyz_live, use_goal_mask=True, debug_vis=False
                )

                warped_3D_trajectory = mix_3D(
                    object_warped_3D_trajectory, goal_warped_3D_trajectory
                )
            else:
                warped_3D_trajectory = object_warped_3D_trajectory
            duration = time.time() - start
            # --------------- Computation Done ---------------

            # Start evaluation
            method_key = get_method_key(method)

            trans_errors, rot_errors = eval_trajectories_3d(
                live_trajectory.trajectory_3D, warped_3D_trajectory
            )

            local_result_dict = {
                f"{method_key}/translation_error": trans_errors,
                f"{method_key}/rotation_error": rot_errors,
                f"{method_key}/duration": duration,
            }

            if log_wandb:
                wandb.log({"live_episode": live_run_path.stem, **local_result_dict})

            results.increment_dict(local_result_dict)

    return results.as_default_dict()


def main(
    base_path: pathlib.Path = BASE_RECORDING_PATH,
    only_sessions: Optional[List[str]] = ["24_01_04", "24_01_05"],
    output_dir: pathlib.Path = RESULTS_DIR / "inter_poses",
    logging_file: pathlib.Path = pathlib.Path("logs")
    / "eval_relative_poses_between_episodes.log",
    log_wandb: bool = False, # This is no longer supported, but might still work
    allow_debugging: bool = False
):
    print(logging_file)
    output_dir.mkdir(exist_ok=True, parents=True)
    single_dir = output_dir / "single_runs"
    single_dir.mkdir(exist_ok=True, parents=True)

    logging_file.parent.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        filename=str(logging_file),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d, %H:%M:%S",
        level=logging.DEBUG,
        force=True,  # Removes all previously handlers of the logger and forces to use the file
    )

    all_runs = get_all_runs(base_path, only_keywords=only_sessions)
    all_trajectories = {
        run_path: Trajectory.from_hands23(
            Hands23Dataset(run_path, lazy_loading=True), n_frames=TIME_STEPS
        )
        for run_path in all_runs
    }

    all_results = {}
    logging.info(f"Evaluating on {len(all_runs)} with {all_runs = }")

    for episode_path in tqdm.tqdm(all_runs):
        # Extract other demonstrations
        all_episode_runs = get_all_runs(
            base_path, only_keywords=[f"{episode_path.parent.stem}"]
        )
        all_runs_N = len(all_episode_runs)
        all_episode_runs.remove(episode_path)
        assert len(all_episode_runs) == (all_runs_N - 1)
        all_episode_runs = sorted(all_episode_runs, key=lambda path: int(path.parts[-1]))

        if allow_debugging:
            # This is needed otherwise the debugger won't stop due the try .. catch-block
            episode_result = process_episode(
                episode_path, all_episode_runs, all_trajectories, log_wandb=log_wandb
            )

        try:
            episode_result = process_episode(
                episode_path, all_episode_runs, all_trajectories, log_wandb=log_wandb
            )

            print(f"Saving results for episode_path: {episode_path}")
            #logging.debug(f"Results out of process_episode: {episode_result}")
            all_results[str(episode_path)] = episode_result

            if log_wandb:
                wandb.finish()

            print(f"Saving single run results to {single_dir / ('_'.join(episode_path.parts[-2:]) + '.yaml')}")
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
