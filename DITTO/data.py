import dataclasses
import logging
import pathlib
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml
from casino import pointcloud as pc_utils

from DITTO import hands23_helpers
from DITTO.config import checked_episodes

try:
    from Zed2Utils import ZED2Config
    import tyro
except ModuleNotFoundError:
    ZED2Config = None
    fake_ZED2Config = dataclasses.make_dataclass(
        "ZED2Config", [("max_distance", float)]
    )

try:
    import open3d as o3d
except ModuleNotFoundError:
    o3d = None

from .config import BASE_RECORDING_PATH


def get_all_runs(
    base_path=BASE_RECORDING_PATH,
    skip_keywords: List[str] = ["debug"],
    only_keywords: Optional[List[str]] = None,
    only_checked: bool = True,
) -> List[pathlib.Path]:
    _all_runs = base_path.glob("**/config.yaml")
    all_runs = [
        run.parent  # As we searched for config.yaml
        for run in _all_runs
        if (not any([keyword in str(run) for keyword in skip_keywords]))
        and (
            only_keywords is None
            or len(only_keywords) == 0
            or any([keyword in str(run) for keyword in only_keywords])
        )
        and (
            not only_checked
            or (only_checked and ("/").join(run.parts[-3:-1]) in checked_episodes)
        )
    ]

    return all_runs


class SimpleLoader:
    def __init__(self, recording_path: pathlib.Path, lazy_loading: bool = True) -> None:
        self.recording_path: pathlib.Path = recording_path
        self.archive: Dict[str, np.ndarray] = np.load(
            str(self.recording_path / "images.np.npz")
        )
        if not lazy_loading:
            self.archive = dict(self.archive)

        if ZED2Config is None:
            self.recording_cfg = fake_ZED2Config(max_distance=10.0)
        else:
            with (self.recording_path / "config.yaml").open("r") as f:
                self.recording_cfg: ZED2Config = tyro.from_yaml(
                    cls=ZED2Config, stream=f.read()
                )

        with (self.recording_path / "intrinsics.yaml").open("r") as f:
            self.intrinsics_dict = yaml.load(f, Loader=yaml.Loader)

            if any(
                [
                    intrinsic_value < 1e-5
                    for intrinsic_value in self.intrinsics_dict.values()
                ]
            ):
                logging.warn(
                    f"Found zero intrisic value when loading from {recording_path}: {self.intrinsics_dict.values() = }"
                )

    @property
    def intrinsics(self):
        return pc_utils.Intrinsics(
            self.intrinsics_dict["f_x"],
            self.intrinsics_dict["f_y"],
            self.intrinsics_dict["c_x"],
            self.intrinsics_dict["c_y"],
        )

    def __len__(self) -> int:
        return self.get_len()

    def get_len(self) -> int:
        return len(self.archive["rgbs_left"])

    def get_resolution(self) -> Tuple[int, int]:
        """
        Returns height, width
        """
        return self.get_rgb(0).shape[:2]

    def get_rgb(self, frame_index) -> np.ndarray:
        try:
            return cv2.cvtColor(
                cv2.imread(str(self.recording_path / "rgb" / f"{frame_index:05}.png")),
                cv2.COLOR_BGR2RGB,
            )
        except:
            return self.archive["rgbs_left"][frame_index]

    def get_depth(self, frame_index) -> np.ndarray:
        try:
            return np.load(self.recording_path / "depth" / f"{frame_index:05}.npy")
        except:
            return self.archive["depths"][frame_index]

    def get_confidence(self, frame_index) -> np.ndarray:
        try:
            return np.load(self.recording_path / "confidence" / f"{frame_index:05}.npy")
        except:
            return self.archive["confidences"][frame_index]

    def get_pointcloud(
        self,
        frame_index,
        at_points: Optional[np.ndarray] = None,
        in_mask: Optional[np.ndarray] = None,
        return_color: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Given points superseeds given mask, given mask superseed full image
        """
        rgb = self.get_rgb(frame_index) if return_color else None

        if not at_points is None:
            return pc_utils.get_points(
                at_points,
                self.get_depth(frame_index),
                intrinsics=self.intrinsics,
                rgb=rgb,
            )

        return pc_utils.get_pc(
            self.get_depth(frame_index), self.intrinsics, mask=in_mask, rgb=rgb
        )

    def get_xyz(self, frame_index):
        raise NotImplementedError()

    def get_pointcloud_o3d(self, frame_index) -> "o3d.geometry.PointCloud":
        rgb_np = self.get_rgb(frame_index)
        rgb = o3d.geometry.Image(rgb_np)
        depth = o3d.geometry.Image(self.get_depth(frame_index))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=self.recording_cfg.max_distance,
            convert_rgb_to_intensity=False,
        )

        height, width = self.get_resolution()

        K_o3d: o3d.camera.PinholeCameraInstrinsics = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            self.intrinsics_dict["f_x"],
            self.intrinsics_dict["f_y"],
            self.intrinsics_dict["c_x"],
            self.intrinsics_dict["c_y"],
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, K_o3d)
        return pcd


class Hands23Dataset(SimpleLoader):
    def __init__(self, recording_path: pathlib.Path, lazy_loading: bool = True) -> None:
        super().__init__(recording_path, lazy_loading=lazy_loading)

        time_step_path = self.recording_path / "time_steps.yaml"
        assert time_step_path.exists()
        with time_step_path.open("r") as f:
            self.time_steps_dict = yaml.load(f, Loader=yaml.Loader)

            # Check if we manually annoted the object in the first few frames as good
            _new_start = self.time_steps_dict.get("t_h23_object_manual", -1)
            if _new_start > -1:
                self.time_steps_dict["t_start"] = _new_start

            # Check if we manually annotated the sequence stop
            _new_stop = self.time_steps_dict.get("t_stop_manual", -1)
            if _new_stop > -1:
                self.time_steps_dict["t_stop"] = _new_stop

    @property
    def hands23_dir(self):
        return self.recording_path / "hands23"

    @property
    def sam_refined_dir(self):
        return self.recording_path / "sam_refined"

    @property
    def cnos_refined_dir(self):
        return self.recording_path / "cnos_redetected_masks"

    @property
    def has_goal(self):
        return self.get_goal_mask(frame_index=0).any()

    def get_cnos_object_mask(self, frame_index: int = 0) -> Optional[np.ndarray]:
        refined_cnos_mask_path = (
            self.cnos_refined_dir / f"object_seg_{frame_index:03}.png"
        )
        if not refined_cnos_mask_path.exists():
            return None
        mask = np.array(cv2.imread(str(refined_cnos_mask_path)))
        mask = np.any(mask > 0, axis=-1)[..., None]  # Convert png's 3 channels
        return mask

    def get_timesteps(self, n_frames: int):
        demo_start, demo_end = self.get_start_stop()
        # Check if we need to get the mask at a different timestep
        linear_samples = np.linspace(demo_start, demo_end, n_frames).astype(int)
        if not "redirect_to_other_step" in self.time_steps_dict.keys():
            return linear_samples

        # TODO Log this?
        non_linear = [
            self.time_steps_dict["redirect_to_other_step"].get(frame_index, frame_index)
            for frame_index in linear_samples
        ]
        return np.array(non_linear)

    def get_object_mask(
        self, frame_index=0, refined: bool = True, refined_cnos: bool = True
    ) -> np.ndarray:
        """
        Arguments:
            frame_index: the frame for which we want the mask
        Returns:
            out_mask: int array
        """
        # Check if there is refined mask for the first time step
        refined_mask_path = self.sam_refined_dir / f"object_seg_{frame_index:03}.png"
        if frame_index == 0 and refined and refined_mask_path.exists():
            # TODO This functionality should be pulled out into casino?
            mask = np.array(cv2.imread(str(refined_mask_path)))
            mask = np.any(mask > 0, axis=-1)[..., None]  # Convert png's 3 channels
            return mask

        # Check if there is a CNOS mask and we should use this for this timestep
        cnos_mask = self.get_cnos_object_mask(frame_index=frame_index)
        if (
            not cnos_mask is None
            and "cnos_over_hands23" in self.time_steps_dict
            and frame_index in self.time_steps_dict["cnos_over_hands23"]
        ):
            logging.info("Using refined CNOS mask")
            return cnos_mask
            # TODO This functionality should be pulled out into casino?

        # Last option, use the actual Hands23 mask (most likely)
        return hands23_helpers.get_mask_at_frame(
            self.hands23_dir, frame_index, 3, self.get_resolution()
        )

    def get_goal_mask(
        self,
        frame_index: int = 0,
        refined: bool = True,
    ) -> np.ndarray:
        """
        Will attempt to load a container mask from the sam_refined folder. By default returns the mask from the earliest timestep
        Arguments:
            last: will return the latest timestep
            frame_index: deprecated!
        Returns:
            mask: binary array of size (w, h, 1)
        """
        refined_mask_path = self.sam_refined_dir / f"container_seg_{frame_index:03}.png"
        if refined and refined_mask_path.exists():
            mask = np.array(cv2.imread(str(refined_mask_path)))
            mask = np.any(mask > 0, axis=-1)[..., None]  # Convert png's 3 channels
        else:
            mask = hands23_helpers.get_mask_at_frame(
                self.hands23_dir, frame_index, 5, self.get_resolution()
            )
        return mask

    def get_hand_preds(self, frame_index) -> np.ndarray:
        hands23_dir = self.recording_path / "hands23"
        for hand_str in ["right_hand", "left_hand"]:
            file_path = hands23_dir / hand_str / "preds" / f"{frame_index:05}.yaml"
            if not file_path.exists():
                continue

            with (file_path).open("r") as f:
                return yaml.load(f, Loader=yaml.Loader)

    def get_hand_masks(self, frame_index) -> np.ndarray:
        return hands23_helpers.get_mask_at_frame(
            self.hands23_dir, frame_index, 2, self.get_resolution()
        )

    def get_bbox(self, frame_index, name):
        """
        Return x, y coordinates, use as follows image[x, y], plt.scatter(y, x)
        """
        assert name in ("hand_bbox", "obj_bbox")
        y_start, x_start, y_stop, x_stop = [
            float(x) for x in self.get_hand_preds(frame_index)[name]
        ]
        return x_start, x_stop, y_start, y_stop

    def get_start_stop(self):
        return self.time_steps_dict["t_start"], self.time_steps_dict["t_stop"]
