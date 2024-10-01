import numpy as np
import spatialmath as sm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import numpy as np
import spatialmath as sm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def eval_trajectories_3d(all_poses_1, all_poses_2):
    """
    Give a list of trajectories, we will compare them at each timestep.
    Returns the translational and rotational error
    """
    assert all_poses_1.ndim == 3 and all_poses_2.ndim == 3
    assert all_poses_1.shape[0] == all_poses_2.shape[0]
    assert all_poses_1.shape[1:] == (4, 4) and all_poses_2.shape[1:] == (4, 4)

    # Normalize trajectories to the origin
    traj_1_to_origin = normalize_trajectory_to_origin(all_poses_1)
    traj_2_to_origin = normalize_trajectory_to_origin(all_poses_2)

    # Check if the first pose is identity
    if not np.allclose(traj_1_to_origin[0], np.eye(traj_1_to_origin[0].shape[0]), atol=1e-9) or not np.allclose(traj_2_to_origin[0], np.eye(traj_2_to_origin[0].shape[0]), atol=1e-9):
        print("Warning: no identity matrix in beginning")

    # Bidirectional distance computation
    trans_errors_1, interpolated_rots_1 = compute_translation_errors(traj_1_to_origin, traj_2_to_origin, debug=True)
    rot_errors_1 = compute_rotation_errors(interpolated_rots_1, traj_1_to_origin)
    trans_errors_2, interpolated_rots_2 = compute_translation_errors(traj_2_to_origin, traj_1_to_origin, debug=True)
    rot_errors_2 = compute_rotation_errors(interpolated_rots_2, traj_2_to_origin)

    # Concatenate errors from both directions
    rot_errors = np.concatenate([rot_errors_1, rot_errors_2])
    trans_errors = np.concatenate([trans_errors_1, trans_errors_2])

    return trans_errors, rot_errors


def eval_trajectories_3d_one_direction(all_poses_1, all_poses_2):
    """
    Give a list of trajectories, we will compare them at each timestep.
    Returns the translational and rotational error
    """
    assert all_poses_1.ndim == 3 and all_poses_2.ndim == 3
    assert all_poses_1.shape[0] == all_poses_2.shape[0]
    assert all_poses_1.shape[1:] == (4, 4) and all_poses_2.shape[1:] == (4, 4)

    # Normalize trajectories to the origin
    traj_1_to_origin = normalize_trajectory_to_origin(all_poses_1)
    traj_2_to_origin = normalize_trajectory_to_origin(all_poses_2)

    # Check if the first pose is identity
    if not np.allclose(traj_1_to_origin[0], np.eye(traj_1_to_origin[0].shape[0]), atol=1e-9) or not np.allclose(traj_2_to_origin[0], np.eye(traj_2_to_origin[0].shape[0]), atol=1e-9):
        print("Warning: no identity matrix in beginning")

    # Bidirectional distance computation
    trans_errors_1, interpolated_rots_1 = compute_translation_errors(traj_1_to_origin, traj_2_to_origin, debug=True)
    rot_errors_1 = compute_rotation_errors(interpolated_rots_1, traj_1_to_origin)

    # Concatenate errors from both directions
    rot_errors = rot_errors_1
    trans_errors = trans_errors_1

    return trans_errors, rot_errors


def eval_trajectories_3d_old(all_poses_1, all_poses_2):
    """
    Give a list of trajectories, we will compare them at each timestep.

    Returns the translational and rotational error
    """
    assert all_poses_1.ndim == 3 and all_poses_2.ndim == 3
    assert all_poses_1.shape[0] == all_poses_2.shape[0]
    assert all_poses_1.shape[1:] == (4, 4) and all_poses_2.shape[1:] == (4, 4)

    trans_errors = []
    rot_errors = []

    all_poses_1 = normalize_trajectory_to_origin(all_poses_1)
    all_poses_2 = normalize_trajectory_to_origin(all_poses_2)

    for pose_1, pose_2 in zip(all_poses_1, all_poses_2):
        # compute Euclidean norm between the 3D poses
        trans_errors.append(np.linalg.norm(pose_1[:3, 3] - pose_2[:3, 3]))
        # compute rotational change in axis-angle representation
        rot_errors.append(compute_rotation_diff(pose_1[:3, :3], pose_2[:3, :3]))
    return trans_errors, rot_errors


def normalize_trajectory_to_origin(poses):
    """
     Normalize the trajectory so that the first pose is at the origin.
     """
    initial_pose_inv = np.linalg.inv(poses[0])
    return np.array([initial_pose_inv @ pose for pose in poses])


def compute_translation_errors(traj1, traj2, debug=False):
    """
    Compute the shortest distance from the keypoints of one trajectory to the line segments of another trajectory.
    Args:
        traj1: Poses of the first trajectory (Nx4x4). Use its keypoints to compute the distance.
        traj2: Poses of the second trajectory (Mx4x4). Use its line segments to compute the distance.
        debug: Enable debugging information.

    Returns: The minimum distances (translation error) and the according interpolated rotations of the second trajectory.

    """
    min_distances = []
    rotations = []

    if debug:
        min_dist_before = [np.linalg.norm(pose1[:3, 3] - pose2[:3, 3]) for pose1, pose2 in zip(traj1, traj2)]

    for pose1 in traj1:
        min_distance = np.inf
        rotation = None
        for pose2, next_pose2 in zip(traj2[:-1], traj2[1:]):
            distance, _, proportion = shortest_distance_from_point_to_line(pose1, pose2, next_pose2)
            if distance < min_distance:
                min_distance = distance
                # interpolate rotation
                if proportion == 1:
                    rotation = next_pose2[:3, :3]
                elif proportion == 0:
                    rotation = pose2[:3, :3]
                else:
                    rotation = interpolate_rotation(pose2, next_pose2, proportion)
        min_distances.append(min_distance)
        rotations.append(rotation)

    if debug:
        # print("Difference:", np.array(min_dist_before) - np.array(min_distances))
        if np.any(np.array(min_dist_before) - np.array(min_distances) < 0):
            print("Difference:", np.array(min_dist_before) - np.array(min_distances))
            print("Warning: Distance increased!")

    return np.array(min_distances), np.array(rotations)


def compute_rotation_errors(interpolated_rots, traj1):
    """
    Compute the rotational error given the interpolated rotations and the first trajectory.
    """
    rot_errors = []
    for rot1, rot2 in zip(interpolated_rots, traj1):
        rot_error = compute_rotation_diff(rot1, rot2[:3, :3])
        rot_errors.append(rot_error)
    return np.array(rot_errors)


def shortest_distance_from_point_to_line(pose1, pose2, next_pose2):
    """
    Compute the shortest distance from a point to a line segment.
    It computes the perpendicular distance from the point to the line segment defined by the start and end points.
    If the point is outside the line segment, it returns the distance to the closest endpoint.
    See Also: https://math.stackexchange.com/questions/1905533/find-perpendicular-distance-from-point-to-line-in-3d
    Args:
        pose1: point to compute the distance from (3D transformation matrix)
        pose2: start of the line segment (3D transformation matrix)
        next_pose2: end of the line segment (3D transformation matrix)

    Returns:
        distance: shortest distance from the point to the line segment
        closest_point: closest point on the line segment to the point
        proportion: proportion of the projection along the line segment
    """
    pos1 = pose1[:3, 3]  # traj1
    pos2 = pose2[:3, 3]  # traj2
    next_pos2 = next_pose2[:3, 3]  # traj2

    segment_vector = next_pos2 - pos2  # vector from start to end of line segment
    segment_length = np.linalg.norm(segment_vector)
    if segment_length == 0:
        return np.linalg.norm(pos1 - pos2), pos2, 0

    d = segment_vector / segment_length  # direction vector d of line segment
    v = pos1 - pos2  # vector from start of line segment to point
    projection = np.dot(v, segment_vector)
    projection = np.clip(projection, 0, segment_length)

    closest_point = pos2 + d * projection  # projection of point to closest point on line segment
    distance = np.linalg.norm(pos1 - closest_point)

    # check if the closest point is the end point (rn excluded from segment length (TO-DO: fix it))
    distance_end_point = np.linalg.norm(pos1 - next_pos2)
    if distance_end_point < distance:
        distance = distance_end_point
        closest_point = next_pos2
        projection = segment_length

    proportion = projection / segment_length  # proportion of projection along line segment

    # print(f"Distance: {distance}, Closest Point: {closest_point}, Proportion: {proportion}")
    return distance, closest_point, proportion


def interpolate_rotation(pose2, next_pose2, proportion):
    """
    Interpolate rotation between two poses based on a proportion.
    """
    # Check if proportion is within the correct range
    if not (0 <= proportion <= 1):
        raise ValueError("Proportion must be between 0 and 1.")

    if not is_rotation_matrix(pose2[:3, :3]):
        raise ValueError("pose2 does not contain a valid rotation matrix.")
    if not is_rotation_matrix(next_pose2[:3, :3]):
        raise ValueError("next_pose2 does not contain a valid rotation matrix.")

    r1 = R.from_matrix(pose2[:3, :3])
    r2 = R.from_matrix(next_pose2[:3, :3])

    # Ensure quaternions are aligned to take the shortest path
    r1_quat = r1.as_quat()
    r2_quat = r2.as_quat()

    if np.dot(r1_quat, r2_quat) < 0.0:
        r2_quat = -r2_quat

    # Create Slerp object for interpolation
    slerp = Slerp([0, 1], R.from_quat([r1_quat, r2_quat]))

    interpolated_rotation_matrix = slerp(proportion).as_matrix()  # convert quaternion back to rotation matrix

    if not is_rotation_matrix(interpolated_rotation_matrix):
        raise ValueError("Interpolated rotation is not a valid rotation matrix.")

    return interpolated_rotation_matrix


def compute_rotation_diff(rot1, rot2):
    relative_rotation = (
            np.linalg.pinv(rot1) @ rot2
    )  # or mat1[i, :3, :3] @ np.linalg.inv(mat2[i, :3, :3])
    angle, axis = sm.SO3(relative_rotation, check=False).angvec()
    return angle

def is_rotation_matrix(matrix):
    """ Check if a matrix is a valid rotation matrix. """
    should_be_identity = np.dot(matrix.T, matrix)
    identity = np.identity(3, dtype=matrix.dtype)
    n = np.linalg.norm(identity - should_be_identity)
    return n < 1e-6
