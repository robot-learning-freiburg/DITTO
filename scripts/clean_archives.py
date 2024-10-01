import pathlib
from typing import List, Optional

import numpy as np
import tqdm
import tyro
import yaml
import shutil

from DITTO.config import BASE_RECORDING_PATH
from DITTO.data import Hands23Dataset, get_all_runs

DEFAULT_INTRINSICS = {
    "c_x": 644.7646484375,
    "c_y": 361.5267333984375,
    "f_x": 521.2882690429688,
    "f_y": 521.2882690429688,
}


def process_episode(
    run_path: pathlib.Path,
    write_rgb: bool = False,
    write_depth: bool = True,
    write_confidence: bool = True,
    write_intrinsics: bool = True,
):
    dataset = Hands23Dataset(run_path)

    if write_rgb:
        rgb_dir = dataset.recording_path / "rgb"  # dataset.archive["rgbs_left"]

    if write_depth:
        depth_dir = dataset.recording_path / "depth"  # dataset.archive["depths"]
        depth_dir.mkdir(exist_ok=True, parents=True)

    if write_confidence:
        confidence_dir = (
            dataset.recording_path / "confidence"
        )  # dataset.archive["confidence"]
        confidence_dir.mkdir(exist_ok=True, parents=True)

    for frame_i in range(len(dataset)):
        if write_rgb:
            raise NotImplementedError()

        if write_depth:
            np.save(depth_dir / f"{frame_i:05}", dataset.get_depth(frame_i))

        if write_confidence:
            np.save(confidence_dir / f"{frame_i:05}", dataset.get_confidence(frame_i))

    if write_intrinsics and dataset.intrinsics_dict["c_x"] == 0.0:
        shutil.copyfile(
            dataset.recording_path / "intrinsics.yaml",
            dataset.recording_path / "intrinsics.yaml.bak",
        )
        with (dataset.recording_path / "intrinsics.yaml").open("w") as f:
            yaml.dump(DEFAULT_INTRINSICS, f)


def main(
    base_path: pathlib.Path = BASE_RECORDING_PATH,
    only_sessions: Optional[List[str]] = [],  # ["24_01_04", "24_01_05"],
):
    all_runs = get_all_runs(base_path, only_keywords=only_sessions)

    for episode_path in tqdm.tqdm(all_runs):
        process_episode(episode_path)


if __name__ == "__main__":
    print(
        "This script will write out the archive and potentially fix missing intrinsics."
    )
    tyro.cli(main)
