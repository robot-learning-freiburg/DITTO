import copy
import dataclasses
import functools
from typing import Callable, List, Optional

import casino
import matplotlib.pyplot as plt
import numpy as np
from casino import pointcloud
from casino.masks import make_mask_aa_bbox
from casino.hardware import clear_torch_memory

# from .config import TrajectoryConfig
from .data import Hands23Dataset
from .geometry import pre_apply_relative_transforms
from .methods_2D import MaskDetector
from .methods_2D.flow_wrapper import track_2D_keypoints_flow_step
from .tracking_2D import track_2D_keypoints, warp_2D_trajectory
from .tracking_3D import Step3DMethod, track_3D_masks, warp_3D_trajectory

# TODO Move this to a non "vis" module?
from .vis_helpers import get_bbox_center, overlay_mask_edge

# All different variants


# TODO Remove optional stuff and make positional
@dataclasses.dataclass
class Trajectory:
    # Extracted frames at these time steps
    frames: np.ndarray = None

    # Images for the demonstration sequence
    depth_imgs: Optional[np.ndarray] = None
    rgb_imgs: Optional[np.ndarray] = None
    object_masks: Optional[np.ndarray] = None

    # Zero-th frame, i.e. initial scene
    rgb_start: Optional[np.ndarray] = None
    depth_start: Optional[np.ndarray] = None
    object_mask_start: Optional[np.ndarray] = None

    # Goal mask
    goal_mask: Optional[np.ndarray] = None

    intrinsics: Optional[pointcloud.Intrinsics] = None

    # Keypoints used for tracking
    keypoints_2D: Optional[np.ndarray] = None
    goal_keypoints_2D: Optional[np.ndarray] = None

    # Hand keypoints for calculating position
    hand_keypoints_2D: Optional[np.ndarray] = None

    # TODO Do we need this? --> Save the step functions in there?
    # cfg: TrajectoryConfig = TrajectoryConfig

    # Internal members as we can't cache numpy
    # TODO Find a way to do this?
    _trajectory_2D: Optional[np.ndarray] = None
    _lifted_2D_trajectory: Optional[np.ndarray] = None
    _relative_trajectory_3d: Optional[np.ndarray] = None

    # Fuction that takes as input two images and a set of keypoints in
    # the first image and re-detects them in the second image
    # used for tracking the 2D center points --> not actually needed for full execution
    _step_function_2D: Callable = track_2D_keypoints_flow_step

    # Function used to warp the demo mask onto the live image to extract grasping region
    _warp_mask_function: MaskDetector = MaskDetector.loftr

    # Function that calculate the relative transform from a given mask in an image
    # in another image; Used within the demonstration
    _step_function_3D: Step3DMethod = Step3DMethod.flow_ransac

    # Same as above but used with the live image
    # _warp_function_3D: Optional[Step3DMethod] = Step3DMethod.cnos_loftr_ransac
    _warp_function_3D: Optional[Step3DMethod] = Step3DMethod.loftr_ransac

    @property
    def n_frames(self):
        return len(self.frames)

    @property
    def warp_function_3D(self):
        return (
            self._warp_function_3D.value
            if not self._warp_function_3D is None
            else self.step_function_3D
        )

    @property
    def step_function_3D(self):
        return self._step_function_3D.value

    @property
    def n_keypoints(self):
        return self.keypoints_2D.shape[0]

    @property
    def xyz_start(self):
        return pointcloud.get_xyz(self.depth_start, self.intrinsics)

    @property
    def trajectory_2D(self) -> np.ndarray:
        assert not self.keypoints_2D is None

        if self._trajectory_2D is None:
            self._trajectory_2D = track_2D_keypoints(
                keypoints=self.keypoints_2D,
                rgb_images=self.rgb_imgs,
                step_function=self._step_function_2D,
            )
            clear_torch_memory

        return self._trajectory_2D

    @property
    def lifted_2D_trajectory(self) -> np.ndarray:
        if self._lifted_2D_trajectory is None:
            self._lifted_2D_trajectory = lift_2D_trajectory_to_3d(
                self.trajectory_2D, self.depth_imgs, self.intrinsics
            )
        return self._lifted_2D_trajectory

    @property
    def relative_trajectory_3D(self) -> np.ndarray:
        if self._relative_trajectory_3d is None:
            self._relative_trajectory_3d = track_3D_masks(
                rgb_imgs=self.rgb_imgs,
                xyz_imgs=np.array(
                    [
                        pointcloud.get_xyz(depth_img, self.intrinsics)
                        for depth_img in self.depth_imgs
                    ]
                ),
                masks=self.object_masks,
                step_function=self.step_function_3D,
            )
        # TODO Add an assert that ensures that we have a valid trajectory here?
        return self._relative_trajectory_3d

    @property
    def object_start_pose(self) -> np.ndarray:
        start_pose = np.eye(4)
        # TODO This takes the mean over all keypoints. We might want to adapt this later
        start_pose[0:3, 3] += self.lifted_2D_trajectory[0, ...].mean(axis=0).copy()
        return start_pose

    @property
    def trajectory_3D(self) -> np.ndarray:
        return pre_apply_relative_transforms(
            self.object_start_pose, self.relative_trajectory_3D
        )

    # TODO I think I don't even need this
    @property
    def object_T_hand(self) -> np.ndarray:
        """
        A rough position of the hand center given in the object frame
        """
        return np.linalg.pinv(self.object_start_pose) @ self.hand_pose_3D

    @property
    def hand_position_3D(self) -> np.ndarray:
        """
        A rough position of the hand in the camera frame
        """
        # It is important that we use self.depth_imgs[0] and not depth_start here
        # As depth_start is the first image, i.e. the one where no hands are visible
        # The hand keypoints are extracted from t_start, which is the first time step
        # in depth_imgs

        # It could happen that we invalid depth at that pixel,
        # we will subquently make bigger windows until we have no Nans values
        window_size: int = 0
        _hand_position_3D: np.ndarray
        while True:
            xx, yy = np.meshgrid(
                np.arange(-window_size, window_size + 1),
                np.arange(-window_size, window_size + 1),
            )
            xx += self.hand_keypoints_2D[0]
            yy += self.hand_keypoints_2D[1]
            window_points = np.stack([xx.flatten(), yy.flatten()], axis=-1)

            _hand_position_3D = np.nanmean(
                pointcloud.get_points(
                    window_points, self.depth_imgs[0], self.intrinsics
                ),
                axis=0,
            )

            if not np.any(np.isnan(_hand_position_3D)):
                break

            window_size += 1

        return _hand_position_3D

    @property
    def hand_pose_3D(self) -> np.ndarray:
        hand_pose = np.eye(4)
        hand_pose[0:3, 3] = self.hand_position_3D
        return hand_pose

    @property
    def has_goal(self) -> bool:
        return self.goal_mask.any()

    def reset_cached(self):
        self._trajectory_2D = None
        self._lifted_2D_trajectory = None
        self._relative_trajectory_3d = None

    def warp_2D_onto_live_frame(
        self, rgb_live: np.ndarray, use_goal_mask: bool = False
    ):
        # TODO: Delete, no longer maintained
        print("warp_2D_onto_live_frame -- no longer maintained")
        assert self.rgb_start.shape == rgb_live.shape
        return warp_2D_trajectory(
            self.rgb_start,
            rgb_live,
            self.trajectory_2D,
            self._step_function_2D,
            with_keypoints=self.goal_keypoints_2D if use_goal_mask else None,
        )

    def warp_3D_onto_live_frame(
        self,
        rgb_live: np.ndarray,
        xyz_live: np.ndarray,
        use_goal_mask: bool = False,
        debug_vis: bool = False,
    ):
        assert self.rgb_start.shape == rgb_live.shape

        xyz_demo = pointcloud.get_xyz(self.depth_start, self.intrinsics)
        assert xyz_demo.shape == xyz_live.shape

        seg_demo = self.object_mask_start if not use_goal_mask else self.goal_mask

        return warp_3D_trajectory(
            self.rgb_start,
            rgb_live,
            xyz_demo,
            xyz_live,
            seg_demo,
            self.trajectory_3D,
            self.warp_function_3D,
            debug_vis=debug_vis,
        )

    def warp_3D_hand_onto_live_frame(
        self, rgb_live: np.ndarray, xyz_live: np.ndarray, debug_vis: bool = False
    ):
        assert self.rgb_start.shape == rgb_live.shape

        xyz_demo = pointcloud.get_xyz(self.depth_start, self.intrinsics)
        assert xyz_demo.shape == xyz_live.shape

        # We are overloading the warp functionality by making it a trajectory with a single
        # pose and then returning just a single index
        return warp_3D_trajectory(
            self.rgb_start,
            rgb_live,
            xyz_demo,
            xyz_live,
            self.object_mask_start,
            self.hand_pose_3D[None, ...],
            self.warp_function_3D,
            debug_vis=debug_vis,
        )[0]

    def warp_object_mask_onto_live_frame(self, rgb_live: np.ndarray):
        # Warp the object mask from demo to live
        _, _, _, _, _, object_mask_live = self._warp_mask_function(
            self.rgb_start, rgb_live, self.object_mask_start
        )
        # Fit bounding box around it --> actually needed?
        object_mask_live = casino.masks.equal_max_bbox(object_mask_live, multiplier=1.2)
        return object_mask_live

    def translate_lifted_2D_trajectory(self, new_start: np.ndarray):
        assert new_start.shape[0] == 3
        moved_trajectory = self.lifted_2D_trajectory.copy()
        moved_trajectory += new_start - self.lifted_2D_trajectory[0, ...]
        return moved_trajectory

    @staticmethod
    def from_hands23(loader: Hands23Dataset, n_frames: int) -> "Trajectory":
        # TODO Should this be here or should this be in Hands23Dataset?
        # TODO If in Hands23Dataset.get_trajectory_blueprint(cfg: TrajectoryCfg) maybe?
        """
        Instantiates a trajectory from a Hands23Dataset.

        As our object keypoints we use the center mean.
        """

        # TODO Make this an argument?
        def keypoint_reduction_fn(kps):
            # Option 1: all points: (don't need to do anything)
            # return np.array(kps).T
            # Option 2: center point TODO Assumes this is inside the object
            return np.array([np.mean(kps, axis=0).astype(int)])
            # Option 3: random points
            # return kps[np.random.choice(len(kps), 5, replace=False)]

        demo_start, _ = loader.get_start_stop()
        frames = loader.get_timesteps(n_frames)

        object_segmentation = loader.get_object_mask(demo_start)[..., 0]
        # No longer maintained
        keypoints = np.argwhere(object_segmentation)
        keypoints = keypoint_reduction_fn(keypoints)
        # ---

        goal_mask = loader.get_goal_mask(
            0,
        )[..., 0]
        # No longer maintained
        goal_keypoints = np.argwhere(goal_mask)
        goal_keypoints = keypoint_reduction_fn(goal_keypoints)
        # ---

        # TODO More sophisticated hand extraction?
        hand_bbox = loader.get_bbox(demo_start, "hand_bbox")
        hand_keypoints = np.array(get_bbox_center(*hand_bbox))

        # Get all images through the sequence
        rgb_images = np.array([loader.get_rgb(frame) for frame in frames])
        depth_images = np.array([loader.get_depth(frame) for frame in frames])
        object_masks = np.array(
            [loader.get_object_mask(frame, refined=False) for frame in frames]
        )

        object_mask_start = loader.get_object_mask(0, refined=True)[..., 0]
        # It could happen, that our object mask is not set initially
        # Thus, we set it to the first object mask from Hands23
        if not np.any(object_mask_start):
            object_mask_start = copy.deepcopy(object_masks[0])

        return Trajectory(
            frames=frames,
            rgb_imgs=rgb_images,
            depth_imgs=depth_images,
            object_masks=object_masks,
            rgb_start=loader.get_rgb(0),
            depth_start=loader.get_depth(0),
            object_mask_start=object_mask_start,
            goal_mask=goal_mask,
            intrinsics=loader.intrinsics,
            keypoints_2D=keypoints,
            goal_keypoints_2D=goal_keypoints,
            hand_keypoints_2D=hand_keypoints,
        )


# TODO where to place this?
def lift_2D_trajectory_to_3d(
    trajectory_2D: np.ndarray,
    depth_imgs: np.ndarray,
    intrinsics: pointcloud.Intrinsics,
) -> np.ndarray:
    trajectory_3d = [
        pointcloud.get_points(
            keypoints_2D,
            depth_img,
            intrinsics,
            remove_out_of_bounds=False,  # We do not remove points that end up out of bounds as we need constant shapes for the trajectory
        )
        for keypoints_2D, depth_img in zip(trajectory_2D, depth_imgs)
    ]
    return np.array(trajectory_3d)


def visualize_trajectory(
    trajectory: Trajectory, identifier: Optional[str] = None, show: bool = True
):
    # Show object and goal masks in 0 frame
    fig_start_masks, axes = plt.subplots(nrows=2, ncols=2, squeeze=False)
    for ax in axes.flatten():
        ax.axis("off")
    axes[0][0].imshow(
        overlay_mask_edge(trajectory.rgb_start, trajectory.object_mask_start)
    )
    axes[0][0].set_title("0th RGB + Object Mask")
    axes[1][0].imshow(trajectory.object_mask_start)
    axes[1][0].set_title("Object Mask")

    axes[0][1].imshow(overlay_mask_edge(trajectory.rgb_start, trajectory.goal_mask))
    axes[0][1].set_title("0th RGB + Goal Mask")
    axes[1][1].imshow(trajectory.goal_mask)
    axes[1][1].set_title("Goal Mask")

    fig_start_masks.suptitle(
        (f"{identifier}: " if not identifier is None else "") + "Masks in Start Frame"
    )
    fig_start_masks.tight_layout()

    # Show object masks through time
    rgbs = trajectory.rgb_imgs
    object_segs = trajectory.object_masks
    N_images = rgbs.shape[0]
    fig_object_masks, axes = plt.subplots(
        nrows=2, ncols=N_images, squeeze=False, figsize=(15, 2), dpi=200
    )
    for ax in axes.flatten():
        ax.axis("off")

    for n in range(N_images):
        axes[0][n].imshow(overlay_mask_edge(rgbs[n], object_segs[n, ..., 0]))
        axes[1][n].imshow(object_segs[n, ..., 0], vmin=0, vmax=1)
        axes[0][n].set_title(f"Frame: {trajectory.frames[n]}")

    fig_object_masks.suptitle(
        (f"{identifier}: " if not identifier is None else "") + "All object masks"
    )
    fig_object_masks.tight_layout()

    if show:
        plt.show()

    return fig_start_masks, fig_object_masks
