import numpy as np
import logging

from casino.math import multiply_along_axis
from casino.geometry import ensure_valid_rotation, to_transformation_matrix
import spatialmath as sm


def get_interp_coefficients(lt, scale: float = 5.0):
    # this defines a mixing function based on a gauss curve.
    # scales defines the steepness of the curve
    def gaussian(x, mu, sig):
        return (
            1.0
            / (np.sqrt(2.0 * np.pi) * sig)
            * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
        )

    gs = gaussian(np.arange(lt), 0, lt / scale)
    mix = gs / (gs + gs[::-1])
    return mix, mix[::-1]


def _mix_2D(
    trajectory_a: np.ndarray,
    trajectory_b: np.ndarray,
    mix_a: np.ndarray,
    mix_b: np.ndarray,
    axis: int = 0,
):
    """
    By default mixes along first axis, i.e. time
    """
    return multiply_along_axis(trajectory_a, mix_a, axis) + multiply_along_axis(
        trajectory_b, mix_b, axis
    )


def mix_2D(trajectory_a: np.ndarray, trajectory_b: np.ndarray, scale: float = 5.0):
    assert trajectory_a.shape == trajectory_b.shape
    mix_a, mix_b = get_interp_coefficients(trajectory_a.shape[0], scale=scale)
    return _mix_2D(trajectory_a, trajectory_b, mix_a, mix_b)


def _mix_3D(
    trajectory_a: np.ndarray,
    trajectory_b: np.ndarray,
    mix_a: np.ndarray,
    mix_b: np.ndarray,
):
    # TODO Can we batch this?
    trajectory_mixed = []

    for pose_a_t, mix_a_t, pose_b_t, mix_b_t in zip(
        trajectory_a, mix_a, trajectory_b, mix_b
    ):
        # TODO Replace with?
        # sm.base.trnorm()

        pose_a_t_matrix = to_transformation_matrix(
            pose_a_t[:3, 3], ensure_valid_rotation(pose_a_t[:3, :3])
        )

        pose_b_t_matrix = to_transformation_matrix(
            pose_b_t[:3, 3], ensure_valid_rotation(pose_b_t[:3, :3])
        )
        # # TODO Refactor into function mix_poses(...)?
        # Checking sometimes marginally fails because .
        # --> Do a manual check here
        if not (
            sm.base.ishom(pose_a_t_matrix, check=True, tol=20)
            and sm.base.ishom(pose_b_t_matrix, check=True, tol=20)
        ):
            logging.error("Invalid transformation in trajectory.")
            return

        pose_mixed = sm.base.trinterp(pose_a_t_matrix, pose_b_t_matrix, mix_b_t)

        pose_mixed_valid = to_transformation_matrix(
            pose_mixed[:3, 3], ensure_valid_rotation(pose_mixed[:3, :3])
        )

        trajectory_mixed.append(pose_mixed_valid)

        # translation_mixed = mix_a_t * pose_a_t.t + mix_b_t * pose_b_t.t
        # quat_a = sm.UnitQuaternion(pose_a_t)
        # quat_b = sm.UnitQuaternion(pose_b_t)
        # quat_mixed = sm.UnitQuaternion.exp(
        #     float(mix_a_t) * quat_a.log() + float(mix_b_t) * quat_b.log()
        # )
        # pose_mixed =
    return np.array(trajectory_mixed)


def mix_3D(trajectory_a: np.ndarray, trajectory_b: np.ndarray, scale: float = 5.0):
    assert trajectory_a.shape == trajectory_b.shape
    mix_a, mix_b = get_interp_coefficients(trajectory_a.shape[0], scale=scale)
    return _mix_3D(trajectory_a, trajectory_b, mix_a, mix_b)
