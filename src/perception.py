"""Pure per-frame perception computations for the anticipation pipeline.

These functions hold no cross-frame state: given a frame's inputs they return
freshly-computed geometry. They were extracted verbatim from the per-frame loop
of ``main.run_experiment`` — the maths is unchanged.
"""
from dataclasses import dataclass
from collections import deque
from typing import Any, Dict

import numpy as np

from utils.tools import transform_point, visibility_mask


# ----------------------------------------------------------------------------
# Objects in the scene + visibility (loop sections "Objects Poses" / "Visual
# Objects in Camera Frame")
# ----------------------------------------------------------------------------
@dataclass
class VisibleObjects:
    bboxes3d: Any
    obj_ids: np.ndarray
    obj_names: np.ndarray
    obj_positions_scene: np.ndarray
    T_Cam_Scene: Any
    obj_positions_cam: np.ndarray
    valid_mask: np.ndarray
    T_scene_object: Dict
    visible_obj_ids: np.ndarray
    visible_obj_names: np.ndarray
    visible_obj_positions_scene: np.ndarray
    visible_obj_positions_cam: np.ndarray


def get_visible_objects(gt_provider, timestamp_ns, T_Scene_Cam, rgb_camera_calibration):
    bbox3d_with_dt = gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(timestamp_ns)
    assert bbox3d_with_dt.is_valid(), "3D bounding box is not available"
    bboxes3d = bbox3d_with_dt.data()

    obj_ids = np.array(list(bboxes3d.keys()))
    obj_names = np.array([gt_provider.get_instance_info_by_id(obj_id).name for obj_id in obj_ids])
    obj_positions_scene = np.array([bbox_3d.transform_scene_object.translation() for bbox_3d in bboxes3d.values()])

    T_Cam_Scene = T_Scene_Cam.inverse()
    obj_positions_cam = np.array([transform_point(T_Cam_Scene, pos.reshape(1, 3)) for pos in obj_positions_scene])
    obj_positions_cam_reshaped = obj_positions_cam.reshape(-1, 3, 1)
    valid_mask = visibility_mask(obj_positions_cam_reshaped, rgb_camera_calibration)

    T_scene_object = {}
    for key, (include, value) in zip(bboxes3d.keys(), zip(valid_mask, bboxes3d.values())):
        if True:  # if include: to take only the visible objects
            T_scene_object[key] = value.transform_scene_object

    return VisibleObjects(
        bboxes3d=bboxes3d,
        obj_ids=obj_ids,
        obj_names=obj_names,
        obj_positions_scene=obj_positions_scene,
        T_Cam_Scene=T_Cam_Scene,
        obj_positions_cam=obj_positions_cam,
        valid_mask=valid_mask,
        T_scene_object=T_scene_object,
        visible_obj_ids=obj_ids[valid_mask],
        visible_obj_names=obj_names[valid_mask],
        visible_obj_positions_scene=obj_positions_scene[valid_mask],
        visible_obj_positions_cam=obj_positions_cam[valid_mask],
    )


# ----------------------------------------------------------------------------
# Camera / device / world axes in the scene frame
# ----------------------------------------------------------------------------
@dataclass
class SceneAxes:
    cam_x_axis_scene: np.ndarray
    cam_y_axis_scene: np.ndarray
    cam_z_axis_scene: np.ndarray
    cam_z_axis_rotation: np.ndarray
    device_x_axis_scene: np.ndarray
    device_y_axis_scene: np.ndarray
    device_z_axis_scene: np.ndarray
    world_x_axis: np.ndarray
    world_y_axis: np.ndarray
    world_z_axis: np.ndarray


def compute_scene_axes(T_Scene_Cam, T_Scene_Device):
    device_x_axis = cam_x_axis = np.array([1, 0, 0]).reshape(3, 1)
    device_y_axis = cam_y_axis = np.array([0, 1, 0]).reshape(3, 1)
    device_z_axis = cam_z_axis = np.array([0, 0, 1]).reshape(3, 1)

    cam_x_axis_scene = (T_Scene_Cam @ cam_x_axis).reshape(1, 3)[0]
    cam_y_axis_scene = (T_Scene_Cam @ cam_y_axis).reshape(1, 3)[0]
    cam_z_axis_scene = (T_Scene_Cam @ cam_z_axis).reshape(1, 3)[0]

    cam_z_axis_rotation = (T_Scene_Cam.rotation().to_matrix() @ cam_z_axis)[:, 0]

    device_x_axis_scene = (T_Scene_Device @ device_x_axis).reshape(1, 3)[0]
    device_y_axis_scene = (T_Scene_Device @ device_y_axis).reshape(1, 3)[0]
    device_z_axis_scene = (T_Scene_Device @ device_z_axis).reshape(1, 3)[0]

    return SceneAxes(
        cam_x_axis_scene=cam_x_axis_scene,
        cam_y_axis_scene=cam_y_axis_scene,
        cam_z_axis_scene=cam_z_axis_scene,
        cam_z_axis_rotation=cam_z_axis_rotation,
        device_x_axis_scene=device_x_axis_scene,
        device_y_axis_scene=device_y_axis_scene,
        device_z_axis_scene=device_z_axis_scene,
        world_x_axis=np.array([1, 0, 0]),
        world_y_axis=np.array([0, 1, 0]),
        world_z_axis=np.array([0, 0, 1]),
    )


# ----------------------------------------------------------------------------
# Gaze-object dot products and user-object distances
# ----------------------------------------------------------------------------
@dataclass
class DotsDistances:
    camera_position_scene: np.ndarray
    dot_products: list
    distances: list
    visible_dot_products: np.ndarray
    visible_vector_camera_objects_scene: np.ndarray
    visible_distance_camera_objects_scene: np.ndarray
    visible_distance_device_objects_scene: np.ndarray


def compute_dots_and_distances(obj_positions_scene, user_position_scene, T_Scene_Cam,
                               cam_z_axis_rotation, valid_mask, work_in_xz_plane=True):
    camera_position_scene = T_Scene_Cam.translation()
    vector_camera_objects_scene = obj_positions_scene[:, 0] - camera_position_scene

    if work_in_xz_plane:
        vector_camera_objects_scene_xz = np.copy(vector_camera_objects_scene)
        vector_camera_objects_scene_xz[:, 1] = 0
        unit_vector_camera_objects_scene_xz = vector_camera_objects_scene_xz / np.linalg.norm(
            vector_camera_objects_scene_xz, axis=1, keepdims=True)

        cam_z_axis_rotation_xz = np.copy(cam_z_axis_rotation)
        cam_z_axis_rotation_xz[1] = 0
        cam_z_axis_rotation_xz /= np.linalg.norm(cam_z_axis_rotation_xz)

        dot_products_array = np.dot(unit_vector_camera_objects_scene_xz, cam_z_axis_rotation_xz)
    else:
        unit_vector_camera_objects_scene = vector_camera_objects_scene / np.linalg.norm(
            vector_camera_objects_scene, axis=1, keepdims=True)
        dot_products_array = np.dot(unit_vector_camera_objects_scene, cam_z_axis_rotation)

    # Visible vectors / dot products / distances
    visible_obj_positions_scene = obj_positions_scene[valid_mask]
    visible_vector_camera_objects_scene = visible_obj_positions_scene[:, 0] - camera_position_scene
    visible_vector_devive_objects_scene = visible_obj_positions_scene[:, 0] - user_position_scene

    visible_dot_products = dot_products_array[valid_mask]
    dot_products = visible_dot_products.tolist()

    visible_distance_camera_objects_scene = np.linalg.norm(visible_vector_camera_objects_scene, axis=1)
    visible_distance_device_objects_scene = np.linalg.norm(visible_vector_devive_objects_scene, axis=1)
    distances = visible_distance_device_objects_scene.tolist()

    return DotsDistances(
        camera_position_scene=camera_position_scene,
        dot_products=dot_products,
        distances=distances,
        visible_dot_products=visible_dot_products,
        visible_vector_camera_objects_scene=visible_vector_camera_objects_scene,
        visible_distance_camera_objects_scene=visible_distance_camera_objects_scene,
        visible_distance_device_objects_scene=visible_distance_device_objects_scene,
    )


# ----------------------------------------------------------------------------
# Filter visible objects by their windowed-average dot/distance
# ----------------------------------------------------------------------------
@dataclass
class FilteredObjects:
    filtered_obj_ids: np.ndarray
    filtered_obj_names: np.ndarray
    filtered_obj_positions_scene: np.ndarray
    filtered_dot_products: np.ndarray
    filtered_distances: np.ndarray
    filtered_high_dot_counts: Dict
    filtered_low_distance_counts: Dict
    filtered_names_high_dot_counts: Dict
    filtered_names_low_distance_counts: Dict
    filtered_names_time_to_approach: Dict
    filtered_names_duration: Dict


def filter_objects_by_average(parameters, gt_provider, visible_obj_ids, visible_obj_names,
                              visible_obj_positions_scene, visible_dot_products,
                              visible_distance_device_objects_scene,
                              visible_avg_dots, visible_avg_distances,
                              visible_visibility_duration, visible_high_dot_counts,
                              visible_low_distance_counts, visible_time_to_approach):
    high_dot_mask = np.array([visible_avg_dots[obj_id] > parameters["avg_dot_high"] for obj_id in visible_obj_ids])
    high_distance_mask = np.array([visible_avg_distances[obj_id] < parameters["avg_distance_high"] for obj_id in visible_obj_ids])
    low_dot_mask = np.array([visible_avg_dots[obj_id] > parameters["avg_dot_low"] for obj_id in visible_obj_ids])
    low_distance_mask = np.array([visible_avg_distances[obj_id] < parameters["avg_distance_low"] for obj_id in visible_obj_ids])

    combined_high_high_mask = high_dot_mask & high_distance_mask
    combined_low_low_mask = low_dot_mask & low_distance_mask
    combined_mask = combined_high_high_mask | combined_low_low_mask

    filtered_obj_ids = visible_obj_ids[combined_mask]
    filtered_obj_names = visible_obj_names[combined_mask]
    filtered_obj_positions_scene = visible_obj_positions_scene[combined_mask]
    filtered_dot_products = visible_dot_products[combined_mask]
    filtered_distances = visible_distance_device_objects_scene[combined_mask]

    filtered_duration = {obj_id: visible_visibility_duration.get(obj_id, deque([(0.0, 0)])) for obj_id in filtered_obj_ids}
    filtered_high_dot_counts = {obj_id: visible_high_dot_counts[obj_id] for obj_id in filtered_obj_ids if obj_id in visible_high_dot_counts}
    filtered_low_distance_counts = {obj_id: visible_low_distance_counts[obj_id] for obj_id in filtered_obj_ids if obj_id in visible_low_distance_counts}

    filtered_names_high_dot_counts = {gt_provider.get_instance_info_by_id(obj_id).name: len(visible_high_dot_counts[obj_id]) for obj_id in filtered_obj_ids if obj_id in visible_high_dot_counts}
    filtered_names_low_distance_counts = {gt_provider.get_instance_info_by_id(obj_id).name: len(visible_low_distance_counts[obj_id]) for obj_id in filtered_obj_ids if obj_id in visible_low_distance_counts}
    filtered_names_time_to_approach = {gt_provider.get_instance_info_by_id(obj_id).name: visible_time_to_approach[obj_id] for obj_id in filtered_obj_ids}
    filtered_names_duration = {gt_provider.get_instance_info_by_id(obj_id).name: filtered_duration[obj_id][-1][1] for obj_id in filtered_obj_ids}

    return FilteredObjects(
        filtered_obj_ids=filtered_obj_ids,
        filtered_obj_names=filtered_obj_names,
        filtered_obj_positions_scene=filtered_obj_positions_scene,
        filtered_dot_products=filtered_dot_products,
        filtered_distances=filtered_distances,
        filtered_high_dot_counts=filtered_high_dot_counts,
        filtered_low_distance_counts=filtered_low_distance_counts,
        filtered_names_high_dot_counts=filtered_names_high_dot_counts,
        filtered_names_low_distance_counts=filtered_names_low_distance_counts,
        filtered_names_time_to_approach=filtered_names_time_to_approach,
        filtered_names_duration=filtered_names_duration,
    )
