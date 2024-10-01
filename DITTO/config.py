import dataclasses
import enum
import os
import pathlib
import platform
import logging
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
CONFIG_DIR = REPO_ROOT / "config"


BASE_RECORDING_PATH = None  # pathlib.Path() # set correct path?

if BASE_RECORDING_PATH is None:
    BASE_RECORDING_PATH = REPO_ROOT / "demonstrations"
    logging.warn(
        f"BASE_RECORDING_PATH in config.py is not set, defaulting to {BASE_RECORDING_PATH = }"
    )

# This means we do 10 steps
TIME_STEPS: int = 11

with (CONFIG_DIR / "valid_episodes.yaml").open("r") as f:
    _all_checked_episodes = yaml.load(f, Loader=yaml.Loader)
checked_episodes = _all_checked_episodes["selected_episodes"]


# TODO More configs?
### Parametrize RANSAC
# @dataclasses.dataclass
# class RANSACConfig:
#     outlier: float = 0.2
#     ...

### Trajectory Tracking Config Class ###
# class KeypointSelector(enum.Enum):
#     all = enum.auto()
#     mean = enum.auto()
#     random = enum.auto()


# @dataclasses.dataclass
# class TrajectoryConfig:
#     keypoints_selection: KeypointSelector = KeypointSelector.mean
#     num_frames: int = TIME_STEPS  # Equals to 10 steps

#### Hands23/hodetector Config Class ###
try:
    import hodetector

    @dataclasses.dataclass
    class Hand23Config:
        # TODO Figure all of these out
        thresh: float = 0.3
        hand_thresh: float = 0.8
        first_obj_thresh: float = 0.3
        second_obj_thresh: float = 0.3
        model_weights: str = (
            pathlib.Path(hodetector.CHECKPOINTS_DIR) / "final_on_blur_model_0399999.pth"
        )

except:
    print("Hands23/hodetector not installed")


try:
    import segment_anything

    @dataclasses.dataclass
    class SAMConfig:
        model_weights: pathlib.Path = (
            REPO_ROOT / "data" / "sam_checkpoints" / "sam_vit_h_4b8939.pth"
        )
        model_type: str = "vit_h"

except:
    print("SAM not installed.")
