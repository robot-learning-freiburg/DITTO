import numpy as np
import cv2

from typing import Dict, Literal, Union


import pathlib
import yaml


# TODO Make this an enum?
def is_valid_graps_state(grasp_state: str):
    return grasp_state != "" and not "NP" in grasp_state


def extract_start_stop(
    hands23_path: pathlib.Path, T: int, consecutive_valid: int = 2
) -> Dict[str, int]:
    # Create grasp timeline
    grasp_states = [""] * T

    t_start = -1
    t_stop = -1

    for hand_str in ["left_hand", "right_hand"]:
        hands_path = hands23_path / hand_str / "preds"
        for hand_file in hands_path.glob("*.yaml"):
            t = int(hand_file.stem)
            with hand_file.open("r") as f:
                hand_dict = yaml.load(f, Loader=yaml.Loader)
            grasp_states[t] = hand_dict["grasp"]

    for t in range(T - (consecutive_valid - 1)):
        if any(
            not is_valid_graps_state(grasp_states[_t])
            for _t in range(t, t + consecutive_valid)
        ):
            continue
        t_start = t
        break

    for t in range(T - consecutive_valid, 0, -1):
        if any(
            not is_valid_graps_state(grasp_states[_t])
            for _t in range(t, t + consecutive_valid)
        ):
            continue
        t_stop = t + (consecutive_valid - 1)
        break
    return {"t_start": t_start, "t_stop": t_stop}


def get_mask_at_frame(
    hands23_dir, frame_index, mask_type: Union[Literal[2], Literal[3], Literal[5]], default_size
):
    """
    Naming conventions for masks are:
        <entity>_<hand_id>_<img_idx>
            <entity>
                `2`: Hand
                `3`: First object
                `5`: Second object
            <hand_id>
                this is either 0 or 1, depending on how many hands were detected
    """
    out_mask = np.zeros(default_size + (1,)).astype(bool)
    # Scan left and right hand folder
    for hand_str in ["left_hand", "right_hand"]:
        file_path = (
            hands23_dir / hand_str / "masks" / f"{mask_type}_0_{frame_index:05}.png"
        )
        if not file_path.exists():
            continue

        mask = np.array(cv2.imread(str(file_path)))
        mask = np.any(mask > 0, axis=-1)[..., None]  # Convert png's 3 channels
        out_mask = np.logical_or(out_mask, mask)
    return out_mask
