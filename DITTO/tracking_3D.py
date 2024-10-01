import copy
import functools
from typing import Callable, List, Literal, Optional, Type
import open3d as o3d

import casino
import numpy as np
import tqdm

from flow_control.servoing.fitting import eval_fit, solve_transform
from flow_control.servoing.fitting_ransac import Ransac

from .methods_2D import CorrespondencesInMaskMethod, MaskDetector
from .vis_helpers import show_registration2d, show_registration3d, show_selected_points

try:
    from DITTO.methods_2D.cnos_wrapper import CNOSDetect, cnos_pipeline
except Exception as e:
    import logging

    logging.error(f"Couldn't load CNOS {e = }")


def track_3D_masks(
    rgb_imgs: np.ndarray, xyz_imgs: np.ndarray, masks: np.ndarray, step_function
) -> np.ndarray:
    """
    Returns the relative transformations between each frame
    """
    n_frames = rgb_imgs.shape[0]
    relative_transformations = []

    for frame_i, frame_i_1 in tqdm.tqdm(
        zip(range(0, n_frames - 1), range(1, n_frames)), total=n_frames - 1
    ):
        rel_trans = step_function(
            rgb_imgs[frame_i, ...],
            rgb_imgs[frame_i_1, ...],
            xyz_imgs[frame_i, ...],
            xyz_imgs[frame_i_1, ...],
            masks[frame_i],
            # debug_vis=frame_i == 2,
        )
        relative_transformations.append(rel_trans)

    return np.array(relative_transformations)


def warp_3D_trajectory(
    rgb_demo: np.ndarray,
    rgb_live: np.ndarray,
    xyz_demo: np.ndarray,
    xyz_live: np.ndarray,
    seg_demo: np.ndarray,
    trajectory: np.ndarray,
    step_function,
    debug_vis: bool = False,
):
    assert trajectory.shape[-2:] == (4, 4)
    rel_trans = step_function(
        rgb_demo,
        rgb_live,
        xyz_demo,
        xyz_live,
        seg_demo,
        debug_vis_2D=debug_vis,
        debug_vis_3D=False,
    )
    warped_3D_trajectory = np.array([rel_trans @ transform for transform in trajectory])
    return warped_3D_trajectory


def _universal_3D_step(
    image_a: np.ndarray,
    image_b: np.ndarray,
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    seg_a: np.ndarray,
    correspondence_calculator: Optional[CorrespondencesInMaskMethod] = None,
    use_ransac: bool = True,
    global_mask_detector: Optional[Type[CNOSDetect.warp_masks]] = None,
    filter_correspondences: bool = False,
    debug_vis_2D: bool = False,
    debug_vis_3D: bool = False,
):
    """
    This does the heavy lifting!
    Generally speaking, we will input two rgb images, two corresponding xyz maps and a segmentation mask corresponding to the first image
    This function will return the relative 3D transformation to transform the points in the mask to the second image

    def calculate_3D(
        image_a, image_b, xyz_a, xyz_b, seg_a,
        mask_cropping: Callable, (None, CNOS)
        correspondence: Callable, (Flow, LoFTR, SuperGlue)
        pose_estimator: Callable, (RANSAC, ICP)
    ) -> SE(3) as 4x4:
        if mask_cropping exists:
            seg_b = mask_cropping (...)

        if pose_estimator == ICP:
            assert seg_b exists
            3D points from mask
            return ICP(3D points)

        elif pose_estimator == RANSAC
            if seg_b exists
                cropped patches
                cropped correspondence = correspondence(cropped patches)
                correspondence = cropped correspondence + cropped offset
            else
                correspondence = correspondence(images)

            3D correspondence from correspondence
            return RANSAC(3D correspondence)

    mask_cropping(image_a, image_b, seg_a) -> seg_b
    correspondence(image_a, image_b, seg_a) -> points_a, points_b, confidence
    pose_estimator(points_a, points_b) -> SE(3)
    """
    assert image_a.ndim == 3
    assert image_b.ndim == 3
    assert xyz_a.ndim == 3
    assert xyz_b.ndim == 3
    if seg_a.ndim == 3:
        seg_a = seg_a[..., 0].copy()  # Ignore last channel
    if seg_a.dtype != bool:
        print("Warning: segmentation mask should be bool.")
        seg_a = seg_a.astype(bool)
    assert seg_a.ndim == 2
    assert seg_a.dtype == bool

    # We optionally do a global mask detection step
    (
        image_a_cropped,
        image_b_cropped,
        seg_a_cropped,
        offset_points_a,
        offset_points_b,
        seg_b,
    ) = global_mask_detector(image_a, image_b, seg_a, debug_vis=debug_vis_2D)

    if use_ransac:
        # When using RANSAC we need to calculate the correspondences
        assert not correspondence_calculator is None
        points_a_cropped, points_b_cropped, confidence = correspondence_calculator(
            image_a_cropped, image_b_cropped, seg_a_cropped, debug_vis=debug_vis_2D
        )
        # And move them to the global frame
        points_a = points_a_cropped + offset_points_a
        points_b = points_b_cropped + offset_points_b
        # TODO Move this into the correspondence function?
        # Then making sure the new correspondnces are within bounds
        points_b, point_mask = casino.masks.filter_coords(
            points_b, xyz_b.shape[:2], return_mask=True
        )

        # TODO Move this into the correspondnces functions?
        # # Filter correspondnces to be in the new segmentation mask
        # if filter_correspondences:
        #     point_mask & seg_b[points_b[:, 0], points_b[:, 1]]

        points_a = points_a[point_mask]

        pcd_a = xyz_a[points_a[:, 0], points_a[:, 1], :]
        pcd_b = xyz_b[points_b[:, 0], points_b[:, 1], :]

        mask_pc = np.logical_and.reduce(
            (
                # NH: I don't like this, depending on the frame of the point cloud this could potentially happen
                # pcd_a[:, 2] != 0,
                # pcd_b[:, 2] != 0,
                # NH: casino marks invalid points with NaN
                ~np.any(np.isnan(pcd_a), axis=-1),
                ~np.any(np.isnan(pcd_b), axis=-1),
            )
        )
        pcd_a_valid = pcd_a[mask_pc]
        pcd_b_valid = pcd_b[mask_pc]

        # Already move based on the mask tracking results
        # Probably doesn't have much impact since we substract the means anyway
        # TODO Re-Add to have better prior?
        # shift_a = trajectories_3d[demo_index][i + 1] - trajectories_3d[demo_index][i]
        # shift_a = np.zeros((3,))
        # pcd_a_valid += shift_a

        # Append homogeneous dimension
        pcd_a_valid = casino.pointcloud.make_homogeneous(pcd_a_valid)
        pcd_b_valid = casino.pointcloud.make_homogeneous(pcd_b_valid)

        ransac = Ransac(
            pcd_a_valid,
            pcd_b_valid,
            solve_transform,
            eval_fit,
            thresh=0.005,
            num_pts_needed=3,
            percentage_thresh=0.9999,
            # outlier_ratio=0.5
            min_runs=1000,
        )
        try:
            fit_q_pos, trf_est = ransac.run()
        except ValueError:
            print("Ransac Failed! Returning identity matrix.")
            trf_est = np.eye(4)

        # Vis stuff at the end
        if debug_vis_2D:
            show_registration2d(image_a, image_b, points_a=points_a, points_b=points_b)
    else:
        assert not seg_b is None
        # For ICP we directly use the masks
        pcd_a_valid = xyz_a[seg_a]
        pcd_b_valid = xyz_b[seg_b]

        o3d_a = casino.pointcloud.to_o3d(pcd_a_valid)
        o3d_b = casino.pointcloud.to_o3d(pcd_b_valid)

        try:
            reg_p2p = o3d.pipelines.registration.registration_icp(
                o3d_a,
                o3d_b,
                0.02,  # threshold
                np.eye(4),  # trans_init
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )
            trf_est = reg_p2p.transformation.copy()
        except:
            print("ICP Failed! Returning identiy matrix")
            trf_est = np.eye(4)

    if debug_vis_3D:
        show_selected_points(xyz_a, image_a, pcd_a_valid)
        show_selected_points(xyz_b, image_b, pcd_b_valid)
        show_registration3d(
            pcd_a_valid, pcd_b_valid, trf_est, pcd_correspond=use_ransac
        )

    return trf_est


## All different variants we can use with universal 3D step
# i.e. a model registry


class Step3DMethod(casino.enums.CallableEnum):
    flow_ransac = functools.partial(
        _universal_3D_step,
        correspondence_calculator=CorrespondencesInMaskMethod.flow,
        use_ransac=True,
        global_mask_detector=MaskDetector.identity,
    )
    loftr_ransac = functools.partial(
        _universal_3D_step,
        correspondence_calculator=CorrespondencesInMaskMethod.loftr,
        use_ransac=True,
        global_mask_detector=MaskDetector.identity,
    )
    cnos_flow_ransac = functools.partial(
        _universal_3D_step,
        correspondence_calculator=CorrespondencesInMaskMethod.flow,
        use_ransac=True,
        global_mask_detector=MaskDetector.cnos,
    )
    cnos_loftr_ransac = functools.partial(
        _universal_3D_step,
        correspondence_calculator=CorrespondencesInMaskMethod.loftr,
        use_ransac=True,
        global_mask_detector=MaskDetector.cnos,
    )
    dino_ransac = functools.partial(
        _universal_3D_step,
        correspondence_calculator=CorrespondencesInMaskMethod.dino,
        use_ransac=True,
        global_mask_detector=MaskDetector.identity,
    )
    cnos_icp = functools.partial(
        _universal_3D_step,
        correspondence_calculator=None,
        use_ransac=False,  # We will use ICP
        global_mask_detector=MaskDetector.cnos,
    )
