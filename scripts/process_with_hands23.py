import pathlib
from typing import List
import tqdm

import tyro
import yaml
from hodetector.utils.demo import Hands, deal_output, get_hands23
from hodetector.utils.vis_utils import vis_per_image

from DITTO import config, data
from DITTO.config import BASE_RECORDING_PATH, Hand23Config
from DITTO.event_series import visualize_hands23


def main(
    folder_name: str = "hands23",
    overwrite: bool = False,
    base_path: pathlib.Path = BASE_RECORDING_PATH,
):
    hands23_cfg = Hand23Config()
    predictor = get_hands23(
        model_weights=str(hands23_cfg.model_weights),
        thresh=hands23_cfg.thresh,
        hand_thresh=hands23_cfg.hand_thresh,
        first_obj_thresh=hands23_cfg.first_obj_thresh,
        second_obj_thresh=hands23_cfg.second_obj_thresh,
    )

    all_runs = data.get_all_runs(base_path)

    for run in tqdm.tqdm(all_runs):
        simple_data_loader = data.SimpleLoader(run)
        vis_dir = simple_data_loader.recording_path / folder_name

        if vis_dir.exists() and not overwrite:
            print(f"Skipping {vis_dir}")
            continue

        all_pred_results = []

        for idx in range(len(simple_data_loader)):
            rgb_img = simple_data_loader.get_rgb(idx)
            input_img = rgb_img[..., ::(-1)].copy()
            hand_lists: List[Hands] = deal_output(im=input_img, predictor=predictor)

            vis_dir.mkdir(exist_ok=True, parents=True)

            left_dir = vis_dir / "left_hand"
            left_dir.mkdir(exist_ok=True, parents=True)
            (left_dir / "preds").mkdir(exist_ok=True, parents=True)

            right_dir = vis_dir / "right_hand"
            right_dir.mkdir(exist_ok=True, parents=True)
            (right_dir / "preds").mkdir(exist_ok=True, parents=True)

            idx_str = f"{idx:05}"

            for hand in hand_lists:
                hand_dir = (
                    left_dir if hand.hand_side == "left_hand" else right_dir
                )  # must be right then :)
                hand.save_masks(hand_dir, rgb_img, idx_str)
                with open(hand_dir / "preds" / f"{idx_str}.yaml", "w") as f:
                    yaml.dump(hand.get_json(), f)  # technically this is a dict

            im = vis_per_image(
                rgb_img[:, :, ::-1], hand_lists, None, vis_dir, use_simple=False
            )
            save_path = vis_dir / f"{idx_str}.png"
            im.save(save_path)

            all_pred_results.append(hand_lists)

        fig = visualize_hands23(all_pred_results)
        fig.savefig(vis_dir / "events.png", dpi=400, bbox_inches="tight")


if __name__ == "__main__":
    tyro.cli(main)
