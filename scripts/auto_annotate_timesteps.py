import logging
import pathlib
import traceback
import shutil
from typing import List

import tqdm
import tyro
import yaml

from DITTO.config import BASE_RECORDING_PATH
from DITTO.data import Hands23Dataset, get_all_runs
from DITTO.hands23_helpers import extract_start_stop


def main(
    overwrite: bool = False,
    base_path: pathlib.Path = BASE_RECORDING_PATH,
    only_sessions: List[str] = [],
    consecutive_valid: int = 2,
):
    all_runs = get_all_runs(base_path, only_keywords=only_sessions)

    for run_path in tqdm.tqdm(all_runs):
        file_outpath = run_path / "time_steps.yaml.auto"
        T = len(list((run_path / "rgb").glob("*.png")))

        if file_outpath.exists() and not overwrite:
            logging.warn(f"Skipping {file_outpath} because we do not overwrite.")
            continue

        hands23_path = run_path / "hands23"
        time_steps_dict = extract_start_stop(hands23_path, T)

        with file_outpath.open("w") as f:
            yaml.dump(time_steps_dict, f)


if __name__ == "__main__":
    tyro.cli(main)
