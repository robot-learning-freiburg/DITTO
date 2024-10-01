import numpy as np


def extract_relative_transforms(absolute_poses: np.ndarray):
    assert absolute_poses.ndim == 3 and absolute_poses.shape[1:] == (4, 4)
    return np.array(
        [
            np.linalg.pinv(pose_i) @ pose_i_1
            for pose_i, pose_i_1 in zip(absolute_poses[:-1], absolute_poses[1:])
        ]
    )


def pre_apply_relative_transforms(
    absolute_pose: np.ndarray, relative_transforms: np.ndarray
):
    return apply_relative_transforms(
        absolute_pose, relative_transforms, post_apply=False
    )


def post_apply_relative_transforms(
    absolute_pose: np.ndarray, relative_transforms: np.ndarray
):
    return apply_relative_transforms(
        absolute_pose, relative_transforms, post_apply=True
    )


def apply_relative_transforms(
    absolute_pose: np.ndarray, relative_transforms: np.ndarray, post_apply: bool
):
    assert absolute_pose.ndim == 2 and absolute_pose.shape == (4, 4)
    assert relative_transforms.ndim == 3 and relative_transforms.shape[1:] == (4, 4)

    absolute_trajectory_3D = [absolute_pose.copy()]
    for transform in relative_transforms:
        new_pose = (
            absolute_trajectory_3D[-1] @ transform
            if post_apply
            else transform @ absolute_trajectory_3D[-1]
        )
        absolute_trajectory_3D.append(new_pose.copy())

    return np.array(absolute_trajectory_3D)


def transform_poses(poses: np.ndarray, transform: np.ndarray):
    assert poses.ndim == 3 and poses.shape[1:] == (4, 4)
    assert transform.ndim == 2 and transform.shape == (4, 4)
    raise NotImplementedError
    
