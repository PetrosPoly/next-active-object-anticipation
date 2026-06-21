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

import numpy as np

# rerun-sdk exposes `rerun` via a path-style .pth that uv-managed venvs may not
# process; add rerun_sdk to sys.path so visualization works.
import sys as _sys, os as _os
for _p in list(_sys.path):
    if _os.path.isdir(_os.path.join(_p, "rerun_sdk")):
        _sys.path.insert(0, _os.path.join(_p, "rerun_sdk")); break
try:
    import rerun as rr
except ImportError:
    rr = None
import os                                   # added by Petros
from collections import deque               # added by Petros 

import logging
import os
import json
import time

from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataPathsProvider,
    AriaDigitalTwinDataProvider,
)

from tqdm import tqdm    

from projectaria_tools.utils.rerun_helpers import (
    AriaGlassesOutline,                                        # Me: Return a list of points to be used to draw the outline of the glasses (line strip).
    ToTransform3D                                              # Me: Helper function to convert Sophus SE3D pose to a Rerun Transform3D
)                                                               

from visualization.rr import (                                # Me: added by Petros                        
    initialize_rerun_viewer,     
    log_camera_calibration,
    log_aria_glasses,
    set_rerun_time,
    process_and_log_image,
    log_device_transformations,
)

from utils.tools import (
    transform_point,                                          # Me: Transformation point from scene to camera frame
    visibility_mask,                                          # Me: Check which points are visible and which are not visible
    exponential_filter,                                       # Me: Filter the velocity with exponential mean average 
    user_movement_calculation,                                # Me: User's movement calculation
    calculate_relative_pose_difference                        
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_path", type=str, required=True, help="path to the ADT sequence")
    parser.add_argument("--device_number", type=int, default=0, help="Device_number you want to visualize, default is 0")
    parser.add_argument("--down_sampling_factor", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--jpeg_quality", type=int, default=75, help=argparse.SUPPRESS)
    parser.add_argument("--rrd_output_path", type=str, default="", help=argparse.SUPPRESS  )                                  # Me: If this path is set, we will save the rerun (.rrd) file to the given path
    parser.add_argument("--runrr", action='store_true',help="Run the the visualization part..same as above")   

    return parser.parse_args()

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
main_logger = logging.getLogger(__name__)

def main():
    
    # ==============================================
    # Filenames & Paths
    # ==============================================
    
    start_time = time.time()
    args = parse_args()
    
    # Project path (src/) and repo root.
    project_path = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(project_path)
    sequence_path = args.sequence_path

    # Dataset folder: accept either a full path to the sequence or a name under
    # ADT_DATA_PATH (env override; default <repo>/data/adt). `data/` is dataset-only.
    datasets_path = os.environ.get("ADT_DATA_PATH", os.path.join(repo_root, "data", "adt"))
    dataset_folder = sequence_path if os.path.isdir(sequence_path) else os.path.join(datasets_path, sequence_path)

    # Ground-truth outputs go under <repo>/results/ (never inside data/).
    seq_name = os.path.basename(os.path.normpath(sequence_path))
    gt_folder = os.path.join(repo_root, 'results', 'gt', seq_name)
    os.makedirs(gt_folder, exist_ok=True)
    
    # Print the paths
    print("Sequence_path: ", dataset_folder)
    
    # ==============================================
    # Load the data 
    # ==============================================
    try:
        paths_provider = AriaDigitalTwinDataPathsProvider(dataset_folder)
        data_paths = paths_provider.get_datapaths_by_device_num(args.device_number)
        gt_provider = AriaDigitalTwinDataProvider(data_paths)
    except Exception as e:
        print("Error: ", str(e))
        exit(-1)

    # True to run the rerun.io 
    args.runrr and initialize_rerun_viewer(rr, args)

    # Load the device trajectory timestamps
    aria_pose_start_timestamp = gt_provider.get_start_time_ns()                     # Me: Get the start time of the Aria poses in nanoseconds
    aria_pose_end_timestamp = gt_provider.get_end_time_ns()                         # Me: Get the end time
    rgb_stream_id = StreamId("214-1")
    
    # Load the camera calibration
    rgb_camera_calibration = gt_provider.get_aria_camera_calibration(rgb_stream_id) # Me: Get the camera calibration of an Aria camera, including intrinsics, distortion params,and projection functions.
    T_Device_Cam = rgb_camera_calibration.get_transform_device_camera()             # Me: Τhis does not change based on time
    args.runrr and log_camera_calibration(rr, rgb_camera_calibration, args)
    
    # Get all timestamps (in ns) of all observations of an Aria sensor
    img_timestamps_ns = gt_provider.get_aria_device_capture_timestamps_ns(rgb_stream_id)    
    img_timestamps_ns = [
        img_timestamp_ns
        for i, img_timestamp_ns in enumerate(img_timestamps_ns)
        if (
            img_timestamp_ns >= aria_pose_start_timestamp
            and img_timestamp_ns <= aria_pose_end_timestamp
        )
    ]
    start_time = img_timestamps_ns[0] / 1e9
    
    # Log Aria Glasses outline
    raw_data_provider_ptr = gt_provider.raw_data_provider_ptr()
    device_calibration = raw_data_provider_ptr.get_device_calibration()
    aria_glasses_point_outline = AriaGlassesOutline(device_calibration)
    args.runrr and log_aria_glasses(rr, aria_glasses_point_outline)
        
    # ==============================================
    # Initialization 
    # ==============================================
                                        
    # Movement detection phase / we need the old poses of the objects
    T_scene_object_before = None                                                    # Me: Objects' poses to check a relative transformation between poses for each round
    original_objects_that_moved_dict = {}                                           # Me: Dict --keys: time of movement   --values: object name 
    movement_time_dict = {}                                                         # Me: Dict --keys: objects that moved --values: start/end time of movement
    indexes_of_objects_that_moved = []                                              # Me: List --indexes of the objects that moved (index from a list of all objects in the sequence)
    previous_moved_names = []                                                       # Me: List --objects that previously moved
    distances = {}                                                                  # Me: Dict --keys: names of objects in motion -- values: a list with the distances of object from the user 
    
    # Users motion based on each movement with 
    user_velocity_before = None                                                     # Me: Initialize previous velocity
    user_position_before = None                                                     # Me: Calculate the user's position
    user_object_position = {}                                                       # User's position when a specific object is inside a radius of 1.5 meters 
    user_object_movement = {}                                                       # User's distance from initial position
    
    print("Variables have been initialized")
    
    # For loop from the start of the time till the end of it
    for timestamp_ns in tqdm(img_timestamps_ns):
        args.runrr  and set_rerun_time(rr, timestamp_ns)
       
        ## Print the current time
        current_time_ns = timestamp_ns
        current_time_s = round((current_time_ns / 1e9 - start_time), 3)
        
        ## Log RGB image
        image_with_dt = gt_provider.get_aria_image_by_timestamp_ns(timestamp_ns, rgb_stream_id)
        args.runrr and process_and_log_image(rr, args, image_with_dt)
                
        # ==============================================
        # Users poses - position / movement (scene)
        # ==============================================                                                          
                                                                                                                   
        ## Device pose in scene frame                                                                          
        aria_3d_pose_with_dt = gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamp_ns)                                 # Me: Pose of the device

        if aria_3d_pose_with_dt.is_valid():
            p_dev_scene = aria_3d_pose_with_dt.data()                                                                     # Me: Pose of the user's device on Scene frame
            T_Scene_Device = p_dev_scene.transform_scene_device                                                           # Me: SE3 of the device 
            T_Scene_Cam = T_Scene_Device @ T_Device_Cam                                                                   # Me: SE3 from Camera to Scene
            
            ## User's position and velocity in the scene frame 
            user_position_scene = aria_3d_pose_with_dt.data().transform_scene_device.translation()[0]      
            user_velocity_scene = aria_3d_pose_with_dt.data().device_linear_velocity
            
            ## User filtered position - velocity 
            if user_position_before is None and user_velocity_before is None:
                
                # user's position & velocity 
                user_ema_position = user_position_before = user_position_scene  
                user_ema_velocity = user_velocity_before = user_velocity_scene   
                
                # Initialise user's movement 
                user_total_movement = 0                                                                                   
                user_relative_total_movement = 0                                                                          # Me: This is the movement of the user every two
                
                # user's position store 
                users_position = deque([(current_time_s, 0)])           
                users_velocity = deque([(current_time_s, 0)])     

            else:
                # user's position and velocity
                user_ema_position = exponential_filter(user_position_scene, user_position_before, alpha = 0.9)            # Me: Exponential filter for the position to reduce the noise 
                user_ema_velocity = exponential_filter(user_velocity_scene, user_velocity_before, alpha = 0.9)            # Me: Apply exponential filter 
                
                # user's movement at each timestamp 
                user_movement_timestep = user_movement_calculation(user_position_before, user_ema_position) 
                user_total_movement += user_movement_timestep
                user_relative_total_movement += user_movement_timestep
                
                # user's position store
                users_position.append((current_time_s, user_ema_position[:2]))                                            # Me: collect user's position on x,y 
                users_velocity.append((current_time_s, user_ema_velocity[:2]))       
                
                # recalculate the position & velocity before
                user_position_before = user_ema_position
                user_velocity_before = user_ema_velocity
                        
            ## Insert this info in the runio.rr software
            args.runrr and log_device_transformations(rr, p_dev_scene, T_Device_Cam, ToTransform3D) 
                
        # ==============================================
        # Objects Poses
        # ==============================================      
                
        ## Objects informations (orientation) for each timestamp in nanoseconds 
        bbox3d_with_dt = gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(timestamp_ns)
        assert bbox3d_with_dt.is_valid(), "3D bounding box is not available"
        bboxes3d = bbox3d_with_dt.data()                                                                                         # Me: Objects data
                                                                                                                    
        ## Extract object IDs and their positions
        obj_ids = np.array(list(bboxes3d.keys()))                                                                                # Me: Ids of the objects
        obj_number = len(obj_ids)
        obj_names = np.array([gt_provider.get_instance_info_by_id(obj_id).name for obj_id in obj_ids])                           # Me: Names of the objects
        obj_positions_scene = np.array([bbox_3d.transform_scene_object.translation() for bbox_3d in bboxes3d.values()])          # Me: Positions on Scene frame #TODO: maybe add the [0] to take the first elemeent 
        distance_user_objects_scene = np.linalg.norm(obj_positions_scene.reshape(obj_number,3) - user_position_scene, axis=1)    # Me: Distances from user's device to objects on scene frame   
                
        # ==============================================
        # Visual Objects in Camera Frame
        # ==============================================  
                
        ## Objects in Camera Frame
        T_Cam_Scene = T_Scene_Cam.inverse()                                                                               # Me: Transform positions to the camera frame
        obj_positions_cam = np.array([transform_point(T_Cam_Scene, pos.reshape(1, 3)) for pos in obj_positions_scene])    # Me: Positions on Camera frame
        obj_positions_cam_reshaped = obj_positions_cam.reshape(-1, 3, 1)                                                  # Me: Prepare positions for projection
        
        ## Produce the visible mask to work only with objects that are in the camera frame
        valid_mask = visibility_mask(obj_positions_cam_reshaped, rgb_camera_calibration)                                  # Me: Produce the Mask
        
        ## Take the only poses of the visible objects   
        T_scene_object = {}
        for key, (include, value) in zip(bboxes3d.keys(), zip(valid_mask, bboxes3d.values())):                            # Me: Filter the dictionary based on the boolean array
            if True: # if include:
                T_scene_object[key] = value.transform_scene_object
                        
        # ==============================================
        # Object Interaction / Detection Phase
        # ==============================================  

        # Get the L2 norm of the angle-axis vector and translation vector (after the log operation in the SE3 for the relative transformation)
        norm_relative_T_object, T_scene_object_before = calculate_relative_pose_difference(T_scene_object_before, T_scene_object)

        # Find the object indexes that satisfy your condition for potential interaction 
        indexes_activation = np.where((norm_relative_T_object) > 0.004)[0] # 4mm
        
        # If Condition to check interaction took place (apart from norm of relative transformation need to check also the distance which should be relative close)
        if indexes_activation.size > 0: 
            
            # Using the indexes to find THE potential objects names that moved and respective distance from them:
            objects_that_moved_names = list(obj_names[indexes_activation])
            
            for i, (name, index) in enumerate(zip(objects_that_moved_names, indexes_activation)):
                
                """
                1. loop over the objects that moved based on pose change and the list of respective indexes calculated from the object names list 
                2. store the distance of the user from the object
                3. store the position of the user and the movement of the user for the object that has been moved 
                4. check if in the previous step the object has been moved in order to avoid insert it in the dictionary original_objects_that_moved_dict  
                """
                # work with the ground truth 
                if name not in user_object_movement:
                    user_pos = np.array([user_ema_position[0], 0, user_ema_position[2]])   # movement on xz plane
                    user_object_position[name] = [user_pos]
                    user_object_movement[name] = 0
                else:
                    user_pos = np.array([user_ema_position[0], 0, user_ema_position[2]]) 
                    user_object_position[name].append(user_pos) # only xz plane
                    if name == 'WoodenSpoon':
                        user_object_movement[name] += np.linalg.norm(user_object_position[name][-1] - user_object_position[name][-2]) 
                    else: 
                        user_object_movement[name] += np.linalg.norm(user_object_position[name][-1] - user_object_position[name][-2])
                # fill the ground truth 
                if name not in previous_moved_names:
                    original_objects_that_moved_dict[current_time_s] = name
                    previous_moved_names = objects_that_moved_names 
                    indexes_of_objects_that_moved.append(indexes_activation[i])
                    print(f"\tObjects that have been moved so far: {original_objects_that_moved_dict}")
                    print(f"\tUser motion while aforementioned object is moving: {user_object_movement}")
                    
                # Track the start and end times of the object movement
                if name not in movement_time_dict:
                    movement_time_dict[name] = {"start_time": current_time_s, "end_time": None} # Me: Object starts moving, record the start time
                else:
                    movement_time_dict[name]["end_time"] = current_time_s                       # Me: Object is already moving, update the end time

    # ==============================================
    # Write the results in Json Files
    # ==============================================  
    
    # Movement of object X with start / end time
    with open(os.path.join(gt_folder,'movement_time_dict.json'), 'w') as json_file:
        json.dump(movement_time_dict, json_file)
    
     # Movement of object X with start time 
    with open(os.path.join(gt_folder,'objects_that_moved.json'), 'w') as json_file:
        json.dump(original_objects_that_moved_dict, json_file)

    # User's movement while an object is moving - We use this to identify objects that the user interacts with 
    with open(os.path.join(gt_folder,'user_object_movement.json'), 'w') as json_file:
        json.dump(user_object_movement, json_file)
    
    end_time = time.time()
            
    # ==============================================
    # Prints
    # ==============================================  
        
    print(f"\t The ground truth interactions {movement_time_dict}")
    print(f"\t Total time taken: {end_time - start_time:.2f} seconds")
    
if __name__ == "__main__":
    main()


