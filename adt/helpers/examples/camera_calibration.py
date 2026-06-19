import argparse

from math import tan
from typing import Dict, Set


import numpy as np
import rerun as rr
import os # added by Petros

from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection 

from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.calibration import CameraCalibration, DeviceCalibration
from projectaria_tools.core.sophus import SE3
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataPathsProvider,
    AriaDigitalTwinDataProvider,
    AriaDigitalTwinSkeletonProvider,
    bbox3d_to_line_coordinates,
    DYNAMIC,
    STATIC,
)
from projectaria_tools.utils.rerun_helpers import AriaGlassesOutline, ToTransform3D
from tqdm import tqdm

    
def test_batch_projection(camera_calibration: CameraCalibration, points: np.ndarray):
    """
    Test batch projection of 3D points to 2D pixel coordinates.

    Args:
        camera_calibration (CameraCalibration): The camera calibration object.
        points (np.ndarray): An array of 3D points in the camera frame of shape (n, 3).
    
    Returns:
        np.ndarray: An array of 2D projected points of shape (n, 2), with None for non-projectable points.
    """
    # Reshape points to the required shape (3, n)
    points_reshaped = points.T.reshape(3, -1, 1)  # Shape (3, n, 1)

    # Project points in batch
    projected_points = camera_calibration.project(points_reshaped)

    # Assuming the project method returns None for non-projectable points, we need to handle that.
    valid_mask = np.array([proj is not None for proj in projected_points])
    projected_points_valid = np.array([proj.squeeze() if proj is not None else [None, None] for proj in projected_points])

    return projected_points_valid, valid_mask

def main():
    
    device_number = 0
    sequence_path = "/Users/petrospolydorou/ETH_Thesis/coding/Actionanticipation/projectaria_sandbox/projectaria_tools/projects/AriaDigitalTwinDatasetTools/data/adt_data/Apartment_release_clean_seq150"
    base_path = "/Users/petrospolydorou/ETH_Thesis/coding/Actionanticipation/projectaria_sandbox/projectaria_tools/projects/AriaDigitalTwinDatasetTools/data/adt_data/Apartment_release_clean_seq150/1WM103600M1292_optitrack_release_clean_seq150"
    vrsfile = os.path.join(base_path, "video.vrs")
    ADT_trajectory_file = os.path.join(base_path, "aria_trajectory.csv")
    MPS_trajectory_file = os.path.join(base_path, "mps/slam/closed_loop_trajectory.csv")
    print(" sequence_path: ", sequence_path)
    print(" VRS File Path: ", vrsfile)  
    print(" GT trajectory path", ADT_trajectory_file)
    
    try:
        paths_provider = AriaDigitalTwinDataPathsProvider(sequence_path)
        data_paths = paths_provider.get_datapaths_by_device_num(device_number)
        gt_provider = AriaDigitalTwinDataProvider(data_paths)
    except Exception as e:
        print("Error: ", str(e))
        exit(-1)
        
    # Load the device trajectory timestamps
    aria_pose_start_timestamp = gt_provider.get_start_time_ns() # get the start time of the Aria poses in nanoseconds
    aria_pose_end_timestamp = gt_provider.get_end_time_ns()
    rgb_stream_id = StreamId("214-1")

    # Load the camera calibration
    rgb_camera_calibration = gt_provider.get_aria_camera_calibration(rgb_stream_id) # Get the camera calibration of an Aria camera, including intrinsics, distortion params,and projection functions.
    T_Device_Cam = rgb_camera_calibration.get_transform_device_camera() # Τhis does not change based on time
    
    # Assuming the camera_calibration object is available from the ADT data provider
    # Initialize some example 3D points in the camera frame
    example_points = np.array([
        [0.5, 0.5, 2.0],
        [-0.5, -0.5, 2.0],
        [0.0, 0.0, 3.0],
        [1.0, 1.0, 1.5],
        [-1.0, -1.0, 1.5]
    ])

    # Test the batch projection
    projected_points = test_batch_projection(rgb_camera_calibration, example_points)

    # Print the results
    for i, point in enumerate(example_points):
        print(f"3D Point: {point} -> 2D Projection: {projected_points[i]}")

if __name__ == "__main__":
    main()