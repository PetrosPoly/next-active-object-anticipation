# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from math import tan
from typing import Dict, Set
import numpy as np
import rerun as rr
from collections import deque
from projectaria_tools.core.sophus import SE3


def initialize_rerun_viewer(rr, args):
    # Initialize Rerun viewer
    rr.init("ADT Sequence Viewer Object", spawn=(not args.rrd_output_path))
    
    # Check if there's an output path for saving the .rrd file
    if args.rrd_output_path:
        print(f"Saving .rrd file to {args.rrd_output_path}")
        rr.save(args.rrd_output_path)

    # Log the world coordinates
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, timeless=True)


def log_camera_calibration(rr, rgb_camera_calibration, args):
    try:
        # Calculate the resolution based on the down sampling factor
        resolution = [
            rgb_camera_calibration.get_image_size()[0] / args.down_sampling_factor,
            rgb_camera_calibration.get_image_size()[1] / args.down_sampling_factor,
        ]
        
        # Calculate the focal length based on the down sampling factor
        focal_length = float(
            rgb_camera_calibration.get_focal_lengths()[0] / args.down_sampling_factor
        )
        
        # Log the camera calibration with the specified parameters
        rr.log(
            "world/device/rgb",
            rr.Pinhole(
                resolution=resolution,
                focal_length=focal_length,
            ),
            timeless=True,
        )
    except AttributeError as e:
        print(f"AttributeError: {e}. Please ensure that 'rgb_camera_calibration' and 'args' have the necessary attributes.")
    except Exception as e:
        print(f"An error occurred: {e}")

def log_aria_glasses(rr, aria_glasses_point_outline):
    rr.log(
        "world/device/glasses_outline",
        rr.LineStrips3D([aria_glasses_point_outline]),
        timeless=True,
    )
    
def set_rerun_time(rr, timestamp_ns):
    # Set device time in nanoseconds
    rr.set_time_nanos("device_time", timestamp_ns)
    # Set time sequence with the given timestamp
    rr.set_time_sequence("timestamp", timestamp_ns)

def process_and_log_image(rr, args, image_with_dt):
    if args.down_sampling_factor > 1:
        # Downsample the image using the down sampling factor
        img = image_with_dt.data().to_numpy_array()[
            ::args.down_sampling_factor, ::args.down_sampling_factor
        ]
        
        # Log the downsampled RGB image with the specified JPEG quality
        rr.log(
            "world/device/rgb",
            rr.Image(img).compress(jpeg_quality=args.jpeg_quality),
        )

# Example usage (ensure 'rr', 'args', and 'image_with_dt' are defined appropriately):
# process_and_log_image(rr, args, image_with_dt)

def log_device_transformations(rr, aria_3d_pose, device_to_rgb, ToTransform3D):
    # Log the transformation of the scene device
    rr.log(
        "world/device",
        ToTransform3D(aria_3d_pose.transform_scene_device, False),
    )
    
    # Log the inverse transformation of the device to RGB
    rr.log(
        "world/device/rgb",
        ToTransform3D(device_to_rgb.inverse(), True),
        # ToTransform3D(device_to_rgb, True),
    )

    # Log the device location (translation)
    rr.log(
        "world/device_translation",
        rr.Points3D(aria_3d_pose.transform_scene_device.translation()),
    )

# Example usage (ensure 'rr', 'aria_3d_pose', and 'device_to_rgb' are defined appropriately):
# log_device_transformations(rr, aria_3d_pose, device_to_rgb, ToTransform3D)

def log_dynamic_object(rr, instance_info, obb):
    # Log the static object with its name and oriented bounding box (OBB)
    rr.log(
        f"world/objects/dynamic/{instance_info.name}",
        rr.LineStrips3D([obb]),
        timeless=True,
    )

def log_object(rr, instance_info, obb):
    # Log the static object with its name and oriented bounding box (OBB)
    rr.log(
        f"world/objects/{instance_info.name}",
        rr.LineStrips3D([obb]),
     #    timeless=True,
    )
    
def log_vector(rr, vector_type,  start_point, end_point):
    # Log the line from the camera frame to the object
    rr.log(
        f"world/lines/{vector_type}", 
        rr.LineStrips3D([start_point, end_point])
    )
    
def log_object_line(rr, instance_info, camera_position, object_position):
    # Log the line from the camera frame to the object
    rr.log(
        f"world/object_lines/{instance_info.name}", 
        rr.LineStrips3D([camera_position, object_position])
    )

# Clear the log files
def clear_logs_names(rr, names):
    for name in names:
        rr.log(f"world/objects/{name}", rr.Clear(recursive=False))
        rr.log(f"world/object_lines/{name}", rr.Clear(recursive=False))
        
def clear_logs_ids(rr, obj_ids):
    for obj_id in obj_ids:
        rr.log(f"world/object_lines/{obj_id}", rr.Clear(recursive=False))