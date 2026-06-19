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
import os                                   # added by Petros
from collections import deque, defaultdict  # added by Petros
from typing import Dict, List, Tuple, Deque # added by Petros

from itertools import product               # added by Petros ()

import logging
import os
import csv
import json

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

from tqdm import tqdm    

from helpers import write_to_excel # Me: added by Petros and is only for debugging

from projectaria_tools.utils.rerun_helpers import (
    AriaGlassesOutline,                                      # Me: Return a list of points to be used to draw the outline of the glasses (line strip).
    ToTransform3D                                            # Me: Helper function to convert Sophus SE3D pose to a Rerun Transform3D
)                                                               

from visualization.rr import (                               # Me: added by Petros
    initialize_rerun_viewer,                                 # Me: Initialize the rerun software    
    log_camera_calibration,                                  # Me: Log the camera features
    log_aria_glasses,                                        # Me: Log the aria glasses
    set_rerun_time,                                           
    process_and_log_image,
    log_device_transformations,                             
    log_dynamic_object,
    log_object,                                              # Me: Log an object 
    log_object_line,                                         # Me: Log an object Line 
    log_vector,                                              # Me: Log the velocity line
    clear_logs_names,                                        # Me: At each timestep clear the objects from the visualization tool 
    clear_logs_ids,
)

from utils.tools import (
    transform_point,                                          # Me: Transformation point from scene to camera frame
    visibility_mask,                                          # Me: Check which points are visible and which are not visible
    exponential_filter,                                       # Me: Filter the velocity with exponential mean average 
    object_within_radius,                                     # Me: Check the objects that are close to a user
    user_movement_calculation,                                # Me: User's movement calculation      
)

from utils.openai_models_work import ( 
    activate_llm,                                             # Me: Query the LLM
    setup_logger,                                             # Me: Setup the logger
    append_to_history_string,                                 # Me: Write the history in a string
    process_llm_response,                                     # Me: Post processing of LLM output 
                                               
)

from utils.objectsGroup_user import (
    ObjectGroupAnalyzer                                       # Me: Class to analyse the objects around the user and specify if the user is changing areas
)

from utils.stats import (
    Statistics                                                # Me: Keep statistics for high dot value and low distance
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_path", type=str, required=True, help="path to the ADT sequence")
    parser.add_argument("--device_number", type=int, default=0, help="Device_number you want to visualize, default is 0")
    parser.add_argument("--down_sampling_factor", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--jpeg_quality", type=int, default=75, help=argparse.SUPPRESS)
    parser.add_argument("--rrd_output_path", type=str, default="", help=argparse.SUPPRESS  )                                  # Me: If this path is set, we will save the rerun (.rrd) file to the given path
    parser.add_argument("--use_llm", action='store_true',help="If you include it in arguments becomes True")                              # Me: added by Petros, if there is a value that 
    parser.add_argument("--runrr", action='store_true',help="Run the the visualization part..same as above")   
    parser.add_argument("--visualize_objects", action='store_true',help="Visualize the objects in the rerun.io")   
    return parser.parse_args()

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
main_logger = logging.getLogger(__name__)

# ==============================================
# Parameters Settting
# ==============================================

time_thresholds = [2] # [1, 2, 3, 4, 5]          # Me: how many seconds before interaction activate LLM
avg_dot_threshold_highs = [0.7]                  # Me: To filter out visible objects. Keep only those that have average dot value higher than this
avg_dot_threshold_lows = [0.2]                      
avg_distance_threshold_highs = [3]               # Me: To filter out visible objects. Keep only those that have average distance value higher than this 
avg_distance_threshold_lows = [1]                   
high_dot_thresholds = [0.9]                      # Me: Count only for these values
distance_thresholds = [2]                        
low_distance_thresholds = [1]
high_dot_counters_threshold = [45]               # Me: Keep only objects with values with counters more than these values
low_distance_counters_threshold = [30]           
time_counters_threshold = [60]                   # Me: 
variables_window_times = [3.0]

# Generate all combinations of the parameters
param_combinations = [
    {
        "time_threshold": t,
        "avg_dot_high": adh,
        "avg_dot_low": adl,
        "avg_distance_high": adhg,
        "avg_distance_low": adlg,
        "high_dot_threshold": hdt,
        "distance_threshold": dt,
        "low_distance_threshold": ldt,
        "high_dot_counters_threshold": hdct,
        "low_distance_counters_threshold": ldct,
        "time_counters_threshold": tc,
        "window_time": w
    }
    for t, adh, adl, adhg, adlg, hdt, dt, ldt, hdct, ldct, tc, w in product(
        time_thresholds, avg_dot_threshold_highs, avg_dot_threshold_lows, 
        avg_distance_threshold_highs, avg_distance_threshold_lows,
        high_dot_thresholds, distance_thresholds, low_distance_thresholds, 
        high_dot_counters_threshold, low_distance_counters_threshold, time_counters_threshold, 
        variables_window_times
    )
]

# ==============================================
# Run for different parameter combinatons
# ==============================================

for parameters in param_combinations: # TODO parallel 4 loop
        
    def main():
        
        work_in_xz_plane = True
        
        # ==============================================
        # Filenames / Paths  & Load of the Data
        # ==============================================
        args = parse_args()
        
        # Base folder path for saving predictions
        project_path = "Documents/projectaria_sandbox/projectaria_tools/projects/AriaDigitalTwinDatasetTools/object_anticipation/adt/"
        sequence_path = args.sequence_path

        # Datasets path
        datasets_path = 'Documents/projectaria_tools_adt_data/'
        dataset_folder = os.path.join(datasets_path, sequence_path)
        os.makedirs(dataset_folder, exist_ok=True)                                          # Me: Ensure the entire directory 
        
        # VRS file and Ground 
        vrsfile = os.path.join(dataset_folder, "video.vrs")
        ADT_trajectory_file = os.path.join(dataset_folder, "aria_trajectory.csv")
        
        # Path to log items for the LLM - Define the CSV file to log the items and check if it exists to write the header
        csv_file = os.path.join(project_path,'utils','txt_files','interaction_log.csv')

        # Save the list to a file
        json_folder = os.path.join(project_path,'utils','json')
        os.makedirs(json_folder, exist_ok=True)                        
        json_file = os.path.join(json_folder,'param_combinations.json')
        
        with open(json_file, 'w') as file:
            json.dump(param_combinations, file)
            
        # Parameters folder name
        parameter_folder_name = (
                f"time_{parameters['time_threshold']}_"
                f"highdot_{parameters['high_dot_threshold']}_"
                f"highdotcount_{parameters['high_dot_counters_threshold']}_"
                f"dist_{parameters['distance_threshold']}_"
                f"distcount_{parameters['low_distance_counters_threshold']}"
            )
    
        # Print the paths
        print("Sequence_path: ", dataset_folder)
        print("Project_path", project_path)
        print("VRS File Path: ", vrsfile)  
        print("GT trajectory path: ", ADT_trajectory_file)
        
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
        dynamic_obj_pose_cache: Dict[str, SE3] = {}                                     # Me: initializes a dictionary maps string keys (likely object identifiers) to SE3 objects
        static_obj_ids: Set[int] = set()                                                # Me: initializes a set intended to store the IDs of static objects. 
        dynamic_obj_moved: Set[str] = set()                                             # Me: initializes a set intended to store the IDS of dynamic objects.

        # Intialize variable for visualizations part 
        previous_obj_ids = set()                                                        # Me: keep track of previously logged objects ids 
        previous_obj_names = set()                                                      # Me: keep track of previously logged objects names
        
        # Collection for the time window variables
        average = True                                                                  # Me: use the accumulated average dot value and distance to filter the objects
        previous_time_ns = aria_pose_start_timestamp                                    # Me: Initialize time to store duration of each object
        
        # Activation of LLM
        llm_activated = False                                                           # Me: LLM activated or not
        history_log: Dict   
                                                                                        # Me: This is for now 
        objects_possibilities=[]                                                        # Me: Posibilities of objects of the LLM   
        rationales = []                                                                 # Me: Rationales of objects of the LLM   
        predictions = []                                                                # Me: Predictions of objects of the LLM   
        goals = []                                                                      # Me: Goals of objects of the LLM  

        objects_possibility_dict = {}
        rationale_dict = {}
        predictions_dict = {}                                                           # Μe: Predictions of the LLM as a dictionary with the timestamps
        goals_dict = {}

        llm_times = []                                                                  # Me: Timestamps that LLM predicted each output
        user_velocity_before = None                                                     # Me: Initialize previous velocity
        user_position_before = None                                                     # Me: Calculate the user's position
        objects_within_radius = []                                                      # Me: Objects in the vicinity of the user
        previous_objects_within_radius = []                                             # Me: Is used to check if the user is stll in the same area that LLM has been activated in order to avoid reactivatiomn of the LLM 
        last_activation_time = 0                                                        # Me: the time activatiom of an LLM
        all_unique_object_names_with_high_dot = set()                                               
    
        # Initialize classes
        group_analyzer = ObjectGroupAnalyzer(history_size=15, future_size=5, change_threshold=0.7)                                         
        statistics = Statistics(
                                parameters['window_time'], 
                                parameters["high_dot_threshold"], 
                                parameters["distance_threshold"], 
                                parameters["low_distance_threshold"], 
                                parameters["time_threshold"]
                                )      # Me: Initialize the ObjectStatistics instance
        
        # ==============================================
        # Load the Ground truth data 
        # ==============================================
        
        with open(os.path.join(project_path,'data','gt',args.sequence_path,'movement_time_dict.json'), 'r') as json_file:
            movement_time_dict = json.load(json_file)
        
        gt_object_names = np.array(list(movement_time_dict.keys()))
        gt_start_times = np.array([movement_time_dict[obj]['start_time'] for obj in gt_object_names])
        gt_end_times = np.array([movement_time_dict[obj]['end_time'] for obj in gt_object_names])
        
        # ==============================================
        # Loop over all timestamps in the sequence
        # ==============================================

        for timestamp_ns in tqdm(img_timestamps_ns):
            args.runrr  and set_rerun_time(rr, timestamp_ns)
        
            ## Current time in seconds
            current_time_ns = timestamp_ns
            current_time_s = round((current_time_ns / 1e9 - start_time), 3)
            
            if current_time_s > 70 and current_time_s < 71 :
                print('stop')      
                  
            ## Time Difference
            time_difference_ns = (current_time_ns - previous_time_ns) / 1e9                                                  # Me: Calculate the time difference in seconds
            previous_time_ns = current_time_ns

            ## Clear previously logged objects and lines
            if args.runrr:
                clear_logs_ids(rr, previous_obj_ids)
                clear_logs_names(rr, previous_obj_names)
            
            previous_obj_ids.clear()
            previous_obj_names.clear()
            
            ## Log RGB image
            image_with_dt = gt_provider.get_aria_image_by_timestamp_ns(timestamp_ns, rgb_stream_id)
            args.runrr and process_and_log_image(rr, args, image_with_dt)
            
            # ==============================================
            # Users poses - position / velocity / movement (scene)
            # ==============================================                                                          
                                                                                    
            aria_3d_pose_with_dt = gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamp_ns)                                 # Me: Pose of the device

            if aria_3d_pose_with_dt.is_valid():
                p_dev_scene = aria_3d_pose_with_dt.data()                                                                     # Me: Pose of the user's device on Scene frame
                T_Scene_Device = p_dev_scene.transform_scene_device                                                           # Me: SE3 of the device 
                T_Scene_Cam = T_Scene_Device @ T_Device_Cam                                                                   # Me: SE3 from Camera to Scene
                   
                ## User's position and velocity in the scene frame 
                user_position_scene = aria_3d_pose_with_dt.data().transform_scene_device.translation()[0]      
                user_velocity_device = aria_3d_pose_with_dt.data().device_linear_velocity # given in the device frame as per https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/mps_trajectory
                
                # VELOCITY - From Device to Scene (only rotation is necessary) 
                """
                    Take only the ROTATION / Velocity vectors represent rates of change of position, so they are not anchored to a specific position in space
                    
                    3 ways to do it
                    
                    1. user_velocity_scene    = T_Scene_Device.rotation().to_matrix() @ user_velocity_device
                    2. user_velocity_scene_v2 = (T_Scene_Device @ user_velocity_device).reshape(1,3)[0] - user_position_scene
                    3. user_velocity_scene_v3 = (T_Scene_Device.to_matrix() @ np.append(user_velocity_device, [1]))[0:3] - user_position_scene
                """ 
                user_velocity_scene = T_Scene_Device.rotation().to_matrix() @ user_velocity_device
                
                #  EXPONENTIAL MOVEMENT AVERAGE (EMA) & TOTAL MOVEMENT
                if user_position_before is None and user_velocity_before is None:
                    
                    # USER/DEVICE POSITION 
                    user_ema_position = user_position_before = user_position_scene  
                    user_ema_velocity = user_velocity_before = user_velocity_scene   
                    
                    # Initialise user's movement 
                    user_total_movement = 0                                                                                  # Me: Total movement along the sequene (lenght of trajectory)
                    user_relative_total_movement = 0                                                                         # Me: Is used to activate LLM again

                    # USER/DEVICE POSITION at each timestep
                    users_position = deque([(current_time_s, 0)])           
                    users_velocity = deque([(current_time_s, 0)])     
                    
                else:
                    # EXPONENTIAL MOVEMENT AVERAGE POSITION AND AVERAGE 
                    user_ema_position = exponential_filter(user_position_scene, user_position_before, alpha = 0.9)            # Me: Exponential filter for the position to reduce the noise 
                    user_ema_velocity = exponential_filter(user_velocity_scene, user_velocity_before, alpha = 0.9)            # Me: Apply exponential filter 
                    
                    # MOVEMENT 
                    user_movement_timestep = user_movement_calculation(user_position_before, user_ema_position) 
                    user_total_movement += user_movement_timestep
                    user_relative_total_movement += user_movement_timestep
                    
                    # DEQUE POSITION & DEQUE VELOCITY
                    users_position.append((current_time_s, user_ema_position))                                                 
                    users_velocity.append((current_time_s, user_ema_velocity))       
                    
                    # recalculate the position & velocity before
                    user_position_before = user_ema_position
                    user_velocity_before = user_ema_velocity
                    
                ## Insert this info in the runio.rr software
                args.runrr and log_device_transformations(rr, p_dev_scene, T_Device_Cam, ToTransform3D) 
            
            # ==============================================
            # Objects Poses
            # ==============================================      
            
            bbox3d_with_dt = gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(timestamp_ns)
            assert bbox3d_with_dt.is_valid(), "3D bounding box is not available"
            bboxes3d = bbox3d_with_dt.data()                                                                                        # Me: Objects data

            # TODO: check where the centroid is located
            
            ## Extract object IDs and their positions
            obj_ids = np.array(list(bboxes3d.keys()))                                                                                # Me: Ids of the objects
            obj_names = np.array([gt_provider.get_instance_info_by_id(obj_id).name for obj_id in obj_ids])                           # Me: Names of the objects
            obj_positions_scene = np.array([bbox_3d.transform_scene_object.translation() for bbox_3d in bboxes3d.values()])          # Me: Positions on Scene frame #TODO: maybe add the [0] to take the first elemeent   
            
            # ==============================================
            # Visual Objects in Camera Frame
            # ==============================================  
            
            T_Cam_Scene = T_Scene_Cam.inverse()                                                                               # Me: Transform positions to the camera frame
            
            ## Objects in Camera Frame
            obj_positions_cam = np.array([transform_point(T_Cam_Scene, pos.reshape(1, 3)) for pos in obj_positions_scene])    # Me: Positions on Camera frame
            obj_positions_cam_reshaped = obj_positions_cam.reshape(-1, 3, 1)                                                  # Me: Prepare positions for projection
            
            ## Produce the visible mask to work only with objects that are in the camera frame
            valid_mask = visibility_mask(obj_positions_cam_reshaped, rgb_camera_calibration)                                  # Me: Produce the Mask
            
            ## Take the only poses of the visible objects   
            T_scene_object = {}
            for key, (include, value) in zip(bboxes3d.keys(), zip(valid_mask, bboxes3d.values())):                            # Me: Filter the dictionary based on the boolean array
                if True: # if include: to take only the visible objects
                    T_scene_object[key] = value.transform_scene_object
            
            ## Take ids, names, position and distances for the visible objects 
            object_ids = visible_obj_ids = obj_ids[valid_mask]                                                                # Me: Filter the ids of visible objects
            visible_obj_names = obj_names[valid_mask]                                                                         # Me: Filter the names of visible objects
            object_positions = visible_obj_positions_scene = obj_positions_scene[valid_mask]                                  # Me; Filter the positions visible objects in scene 
            visible_obj_positions_cam = obj_positions_cam[valid_mask]                                                         # Me: Filter the positions visible objects in camera
            
            # ==============================================
            # Camera and Device Axes Transformation in Scene and World Frames
            # ==============================================  
            
            # Device and Camera
            device_x_axis = cam_x_axis = np.array([1, 0, 0]).reshape(3, 1)
            device_y_axis = cam_y_axis = np.array([0, 1, 0]).reshape(3, 1)
            device_z_axis = cam_z_axis = np.array([0, 0, 1]).reshape(3, 1)
            
            # CAMERA coordinates in the scene frame (having the traslation into account)
            cam_x_axis_scene = (T_Scene_Cam @ cam_x_axis).reshape(1,3)[0]  
            cam_y_axis_scene = (T_Scene_Cam @ cam_y_axis).reshape(1,3)[0]
            cam_z_axis_scene = (T_Scene_Cam @ cam_z_axis).reshape(1,3)[0] # these value points the end of z axis. so from the origin has an opposite direction to where the user is looking and that's why the dot value is negative
            
            # CAMERA coordinates in the scene frame (having the traslation into account)
            cam_x_axis_rotation = (T_Scene_Cam.rotation().to_matrix() @ cam_x_axis)[:,0]
            cam_y_axis_rotation = (T_Scene_Cam.rotation().to_matrix() @ cam_y_axis)[:,0]
            cam_z_axis_rotation = (T_Scene_Cam.rotation().to_matrix() @ cam_z_axis)[:,0]
            
            # DEVICE coordinates in the scene frame
            device_x_axis_scene = (T_Scene_Device @ device_x_axis).reshape(1,3)[0]
            device_y_axis_scene = (T_Scene_Device @ device_y_axis).reshape(1,3)[0]
            device_z_axis_scene = (T_Scene_Device @ device_z_axis).reshape(1,3)[0]
            
            # WORLD coordinates 
            world_x_axis = np.array([1,0,0])
            world_y_axis = np.array([0,1,0])
            world_z_axis = np.array([0,0,1])
            
            # ==============================================
            # Dot Products - Distances
            # ==============================================  
            
            # VECTORS FROM CAMERA AND DEVICE TO OBJECTS
            camera_position_scene = T_Scene_Cam.translation()                                                                                    # Me: Camera position in scene
            vector_camera_objects_scene = obj_positions_scene[:, 0] - camera_position_scene                                     
            vector_devive_objects_scene = obj_positions_scene[:, 0] - user_position_scene
            
            # CALCULATE DOT PRODUCT IN 2D
            if work_in_xz_plane:
                
                # PROJECT VECTORS ONTO XZ PLANE (ignore Y component)
                vector_camera_objects_scene_xz = np.copy(vector_camera_objects_scene)
                vector_camera_objects_scene_xz[:, 1] = 0  # Set Y component to 0
                unit_vector_camera_objects_scene_xz = vector_camera_objects_scene_xz / np.linalg.norm(vector_camera_objects_scene_xz, axis=1, keepdims=True)

                # PROJECT CAMERA Z AXIS ONTO XZ PLANE (ignore Y component)
                cam_z_axis_rotation_xz = np.copy(cam_z_axis_rotation)
                cam_z_axis_rotation_xz[1] = 0  # Set Y component to 0

                # Normalize the camera Z axis vector on the XZ plane
                cam_z_axis_rotation_xz /= np.linalg.norm(cam_z_axis_rotation_xz)

                # DOT PRODUCT IN XZ PLANE
                dot_products_array = np.dot(unit_vector_camera_objects_scene_xz, cam_z_axis_rotation_xz)

            else:
                unit_vector_camera_objects_scene = vector_camera_objects_scene / np.linalg.norm(vector_camera_objects_scene, axis=1, keepdims=True)  # Me: Normalize the vectors
            
                # DOT PRODUCT
                dot_products_array = np.dot(unit_vector_camera_objects_scene, cam_z_axis_rotation)                                              
            
            all_dot_products = dot_products_array.tolist()                                                                                                 # Me: Dot Product (camera z axis / camera to object vector
                
            # DISTANCES 
            distance_camera_objects_scene = np.linalg.norm(vector_camera_objects_scene, axis=1)                      
            distance_device_objects_scene = np.linalg.norm(vector_devive_objects_scene, axis=1)
            all_distances = distance_device_objects_scene.tolist()  
            
            # VISIBLE VECTORS FROM CAMERA AND DEVICE TO OBJECTS
            visible_vector_camera_objects_scene = visible_obj_positions_scene[:, 0] - camera_position_scene                                     
            visible_vector_devive_objects_scene = visible_obj_positions_scene[:, 0] - user_position_scene
            visible_unit_vector_camera_objects_scene = visible_vector_camera_objects_scene / np.linalg.norm(visible_vector_camera_objects_scene, axis=1, keepdims=True)  # Me: Normalize the vectors

            # VISIBLE DOT PRODUCT
            visible_dot_products = dot_products_array[valid_mask]                                        
            dot_products = visible_dot_products.tolist()   
            
            # VISIBLE DISTANCES
            visible_distance_camera_objects_scene = np.linalg.norm(visible_vector_camera_objects_scene, axis=1)                      
            visible_distance_device_objects_scene = np.linalg.norm(visible_vector_devive_objects_scene, axis=1)                     
            distances =  visible_distance_device_objects_scene.tolist()                                                                            # Me: Filter the visible objects in scene

            # ==============================================
            # Time Window - Accumulated / Average Values & Counts
            # ==============================================  
            
            (visible_past_dots, 
            visible_past_distances, 
            visible_avg_dots, 
            visible_avg_distances, 
            visible_avg_dots_list,
            visoble_avg_distances_list,
            visible_visibility_counter, 
            visible_visibility_duration, 
            visible_high_dot_counts, 
            visible_low_distance_counts,
            visible_very_low_distance_counts,
            visible_time_to_approach, 
            visible_time_to_approach_counts) = statistics.time_window(
                                                        current_time_s, 
                                                        time_difference_ns, 
                                                        user_ema_position, 
                                                        user_velocity_device,
                                                        object_ids, 
                                                        object_positions,
                                                        dot_products, 
                                                        distances,
                                                        T_Scene_Device
                                                    )

            # ==============================================
            # Filter Visible Objects with Average Values
            # ==============================================  
            
            # MASK
            if average == True:
                
                # high dot mask and relaxed distance
                high_dot_mask = np.array([visible_avg_dots[obj_id] > parameters["avg_dot_high"] for obj_id in visible_obj_ids])                      # Me: high dot mask shape for those objects that have high accummulated dot
                high_distance_mask = np.array([visible_avg_distances[obj_id] < parameters["avg_distance_high"] for obj_id in visible_obj_ids])       # Me: high distance mask
                
                # low average dot mask but also low distance
                low_dot_mask = np.array([visible_avg_dots[obj_id] > parameters["avg_dot_low"] for obj_id in visible_obj_ids])                   # Me: low dot mask
                low_distance_mask = np.array([visible_avg_distances[obj_id] < parameters["avg_distance_low"] for obj_id in visible_obj_ids])         # Me: low distance mask (because the minimum distace is 0.56 from the object)
                
                # combined masks
                combined_high_high_mask = high_dot_mask & high_distance_mask                                       
                combined_low_low_mask = low_dot_mask & low_distance_mask                                        
                combined_mask = combined_high_high_mask | combined_low_low_mask                           
                
            else:                                                                                              
                
                # high dot mask and relaxed distance
                high_dot_mask = dot_products > 0.8                                                                      # Me: high dot mask shape is shape (189,) 1D array
                high_distance_mask = distances < 3                                                                      # Me: high distance mask
                
                # low average dot mask but also low distance
                low_dot_mask = dot_products > 0.2                                                                       # Me: low dot mask 
                low_distance_mask = distances < 0.9                                                                     # Me: low distance mask (because the minimum distace is 0.56 from the object)
                
                # combined masks
                combined_high_high_mask = high_dot_mask & high_distance_mask                                            # Me: if the distance is high the dot should be high to accept 
                combined_low_low_mask = low_dot_mask & low_distance_mask                                                # Me: if the dot is low the distance should be low to accept 
                combined_mask = combined_high_high_mask | combined_low_low_mask

            # IDs, NAMES
            filtered_obj_ids = visible_obj_ids[combined_mask]
            filtered_obj_names = visible_obj_names[combined_mask]
            filtered_vector_camera_objects_scene = visible_vector_camera_objects_scene[combined_mask]
            
            # POSITIONS
            filtered_obj_positions_scene = visible_obj_positions_scene[combined_mask]
            filtered_obj_positions_cam = visible_obj_positions_cam[combined_mask]
            
            # DOT PRODUCTS
            filtered_dot_products = visible_dot_products[combined_mask]
            filtered_names_dot = {gt_provider.get_instance_info_by_id(obj_id).name: filtered_dot_products[i] for i, obj_id in enumerate(filtered_obj_ids)} # use it for the excel
            
            # DISTANCES 
            filtered_distances_cam = visible_distance_camera_objects_scene[combined_mask]
            filtered_distances = visible_distance_device_objects_scene[combined_mask]
            filtered_names_distances = {gt_provider.get_instance_info_by_id(obj_id).name: filtered_distances[i] for i, obj_id in enumerate(filtered_obj_ids)}   
            
            # VISIBILITY COUNTER & DURATION                   
            filtered_counter = {obj_id: visible_visibility_counter.get(obj_id, deque([(0.0,0)])) for obj_id in filtered_obj_ids}     
            filtered_duration = {obj_id: visible_visibility_duration.get(obj_id, deque([(0.0,0)])) for obj_id in filtered_obj_ids}
            
            # HIGH DOT / LOW DISTANCE / TIME COUNTERS 
            filtered_high_dot_counts = {obj_id: visible_high_dot_counts[obj_id] for obj_id in filtered_obj_ids if obj_id in visible_high_dot_counts}
            filtered_low_distance_counts = {obj_id: visible_low_distance_counts[obj_id] for obj_id in filtered_obj_ids if obj_id in visible_low_distance_counts}
                   
            # ==============================================
            # Keep the Important Context Information for the feasible objects
            # ==============================================  
            
            # HIGH DOT COUNTS / LOW DISTANCE COUNTS / TIME TO APPROACH- DICTIONARIES {NAME: COUNT}
            filtered_names_high_dot_counts = {gt_provider.get_instance_info_by_id(obj_id).name: len(visible_high_dot_counts[obj_id]) for obj_id in filtered_obj_ids if obj_id in visible_high_dot_counts}
            filtered_names_low_distance_counts = {gt_provider.get_instance_info_by_id(obj_id).name: len(visible_low_distance_counts[obj_id]) for obj_id in filtered_obj_ids if obj_id in visible_low_distance_counts}
            filtered_names_time_to_approach = {gt_provider.get_instance_info_by_id(obj_id).name: visible_time_to_approach[obj_id] for obj_id in filtered_obj_ids} 
            filtered_names_duration = {gt_provider.get_instance_info_by_id(obj_id).name: filtered_duration[obj_id][-1][1] for obj_id in filtered_obj_ids}
            
            # ==============================================
            # 4 Criteria to enable LLM (1. High dot products duration 2. Low distance duration 3. Time to contact 4. High visibility duration)
            # ==============================================  
            
            """
            Object with high dot counts 

            1. Initialize dictionaries and lists to be used for the LLM activation 
                - We want to have objects that presents high dot values consistently 
                - we want to have objects that are close to the user consistently
                - We want to have objects that are approachable in less than 2 seconds (this list is a subset of the above list)
                - We want to have objects that are visible to the user's camera for a significant amount of time over 2 seconds 
                - We want objects that have shown high dot history in the past 
            
            2. Assummption: 
                - User will interact with objects that are consistently in the user's focus
                - User will interact with objects that are close to him 
                - User will interact with objects is close to one of these in less a certain amount of time (e.g. 2 seconds)
                - User will interact with objects that have been seen consistently (so for this reason we need to add the history)
                - User will interact with objects that are visible to the user for a certain amount of time 
                
            2. Store the objects 
                - High dot counts > 45 counts or 1.5 seconds than 3
                - Close distance  < 30 counts or 1 second from than 3
                - Time to approach
            """
            
            # high dot values & counts
            high_dot_counts = {}
            
            # high dot values & counts but also distance
            high_dot_counts_but_also_distance = {}
      
            # distance values & counts
            low_distance_counts = {}
            
            # distance values & counts
            low_distance_counts_but_also_high_dot = {}
            
            # time to approach dictionaries 
            time_to_approach_dict = {}  
            time_to_approach_list = []      

            # Objects with time less than 2 seconds
            less_than_2_seconds_dict = {}
            less_than_2_seconds_list = []

            # Objects with time < 2 seconds and count threshold
            filtered_names_high_dot_counts_and_distance_counts = {}
            filtered_names_high_dot_counts_and_distance_values = {}
            
            filtered_names_low_distance_counts_and_high_dot_counts = {}
            filtered_names_low_distance_counts_and_high_dot_values = {}

            # High duration objects
            high_duration_objects = {}
            
            # Identify objects in motion using NumPy
            objects_in_motion = gt_object_names[(gt_start_times <= current_time_s) & (current_time_s <= gt_end_times)].tolist()
            
            # Combine the logic for high dot counts and low distance counts into one loop
            for index, object_id in enumerate(filtered_obj_ids):
                object_name = gt_provider.get_instance_info_by_id(object_id).name

                # Skip objects in motion
                if object_name in objects_in_motion:
                    continue

                # Time to approach based on user's velocity, user's position, objects position
                object_time_xyz, object_time_xz = statistics.interaction_time_user_object(user_velocity_device, user_ema_position, filtered_obj_positions_scene[index][0], T_Scene_Device)

                # Check if the object meets the high dot counts threshold
                if object_id in filtered_high_dot_counts and len(filtered_high_dot_counts[object_id]) >= parameters["high_dot_counters_threshold"]:
                    # Store high dot values and counts
                    high_dot_counts[object_name] = filtered_names_high_dot_counts[object_name]

                    # If the object also meets the low distance count threshold, store additional data
                    if object_id in filtered_low_distance_counts:
                        high_dot_counts_but_also_distance[object_name] = high_dot_counts[object_name]

                # Check if the object meets the low distance counts threshold
                if object_id in filtered_low_distance_counts and len(filtered_low_distance_counts[object_id]) >= parameters["low_distance_counters_threshold"]:
                    # Store low distance values and counts
                    low_distance_counts[object_name] = filtered_names_low_distance_counts[object_name]

                    # If the object also meets the high dot count threshold, store additional data
                    if object_id in filtered_high_dot_counts:
                        low_distance_counts_but_also_high_dot[object_name] = low_distance_counts[object_name]
                
                # This is if the object has at least one count in the whole time period of high dot, low distance and duration over 1 second
                if (object_name in filtered_names_high_dot_counts and  
                    object_name in filtered_names_low_distance_counts and 
                    filtered_names_duration[object_name] > 1):
                    
                    # Object with high dot counts and but also have distance counts 
                    filtered_names_high_dot_counts_and_distance_counts[object_name] = filtered_names_high_dot_counts[object_name]
                    filtered_names_high_dot_counts_and_distance_values[object_name] = f"{float(filtered_dot_products[index]):.3f}"
                    
                    # Object with low distance counts and but also have high dot counts 
                    filtered_names_low_distance_counts_and_high_dot_counts[object_name] = filtered_names_low_distance_counts[object_name]
                    filtered_names_low_distance_counts_and_high_dot_values[object_name] = f"{float(filtered_distances[index]):.3f}"
                    
                    # Store time to approach for objects that meet all criteria
                    time_to_approach_dict[object_name] = object_time_xz
                    time_to_approach_list.append(object_time_xz)

                    # If time is below the threshold, store the time
                    if object_time_xz < parameters["time_threshold"]:
                        less_than_2_seconds_dict[object_name] = object_time_xz
                        less_than_2_seconds_list.append(object_time_xz)
                        print(f"\t Time to approach {object_name} is less than 2 seconds: {object_time_xz}")

            # Maintain history of objects with high dot values
            all_unique_object_names_with_high_dot |= set(high_dot_counts.keys()) 
            high_dot_history = list(all_unique_object_names_with_high_dot)
        
            # ==============================================
            # LLM Query and Activation
            # ==============================================  
            
            """
            TODO: THIS IS THE NEW ONE
            In summary the conditions to activate the LLM are the following: 
            
            Object with high dot counts  > threshold but also has distance counts --> visibility is certain as high dot counts over theshold
            Object with distance counts  > threshold but also has high dot counts --> visibility is certain as distance counts over the threshold
            Object with time to approach < threshold but has some high dot and distance counts and visibility duration --> visibility duration over 1 second
            
            ===== 
            
            What we pass to the LLM? 
            
            For the above mentioned objects we pass the 
            1. Dot Counts 
            2. Distance Counts 
            3. Dot Value
            4. Distance Value
            5. Time to approach
            
            =====
            
            Reasoning process of the LLMs 
            
            Objects that satisfy all thresholds (dot counts, distance, time to approach) are deemed highly probable.
            Objects meeting the distance and time thresholds but showing partial dot counts, or objects meeting the dot and time thresholds with partial distance counts, are the next most likely.
            Objects that do not meet the time threshold but satisfy both the dot and distance thresholds (since proximity typically suggests a reduced time to approach) are considered probable.
            Objects that fail to meet the time threshold but satisfy either the dot or distance thresholds (though not both) are considered less likely but still possibl
           
            """

            if (high_dot_counts_but_also_distance
                and low_distance_counts_but_also_high_dot
                and less_than_2_seconds_dict  # this list contains only objects that have duration visibility over
                ):
        
                # Print statement 
                print("the 3 criteria have been satisfied")

                # Write information only if LLM is ON and is ready to activated 
                if args.use_llm and not llm_activated:  
                    
                    # Additional condition to activate the LLM. The object that is approachable in less than 2 seconds should belong in the closed list
                    if any(object in high_dot_history for object in less_than_2_seconds_dict.keys()): # TODO: make this soft having it as a group of objects that was looking, Red Clock was within a group 
                        
                        """
                        1. objects names --- high dot counts                                name:  high_focus_objects_measured_in_counts
                        2. objects names --- distance counts                                name:  nearby_objects_measured_in_counts
                        3. objects names --- high dot value before activation               name:  objects_names_and_latest_focus_intensity 
                        4. objects names --- distance value before actication               name:  objects_names_and_latest_distance_from_the_user
                        5. objects names --- list of names with time less than 2 seconds    name:  quick_access_object 
                        """
                        
                        # write the log
                        history_log = append_to_history_string(current_time_s, 
                                                "Living Room", 
                                                filtered_names_high_dot_counts_and_distance_counts,
                                                filtered_names_low_distance_counts_and_high_dot_counts,  
                                                filtered_names_high_dot_counts_and_distance_values,
                                                filtered_names_low_distance_counts_and_high_dot_values,
                                                time_to_approach_dict,
                                                predictions_dict,
                        )
            
                        # Convert history log to a string
                        history_log_string = str(history_log)
                        
                        # use the LLM
                        llm_response = activate_llm(history_log_string, parameters)
                        
                        # write the lllm 
                        llm_path = "llm/llm_response.yaml"
                        llm_folder = os.path.join(project_path, llm_path)   
                        os.makedirs(os.path.dirname(llm_folder), exist_ok=True)   # check existence 
                        with open(llm_folder, "a") as file:
                            file.write("\n\n")
                            file.write(llm_response)
                        print(f"LLM response has been written to {llm_folder}")

                        # Update the last time LLM was activated 
                        last_activation_time = current_time_s
                        
                        if True: # if llm_response:
                            
                            objects_possibility, rationale, predicted_objects, goal = process_llm_response(llm_response)

                            # dictionaries
                            objects_possibility_dict[current_time_s] = objects_possibility
                            rationale_dict[current_time_s] = rationale
                            predictions_dict[current_time_s] = predicted_objects
                            goals_dict[current_time_s] = goal

                            # lists 
                            objects_possibilities.append(objects_possibility)
                            rationales.append(rationale)
                            predictions.append(predicted_objects)
                            goals.append(goal)
                            llm_times.append(current_time_s)

                            # Keep the output of LLM in log file   # TODO: update the log file for 1 second
                            log_filename = f'logs/time_{current_time_s}.log'    
                            log_folder = os.path.join(project_path, log_filename)               
                            os.makedirs(os.path.dirname(log_folder), exist_ok=True)
                            logger = setup_logger(log_folder)
                            logger.info(f"LLM Response: {llm_response}")
                            
                            # Log history_log content until the LLM activation and then initialize it again 
                            history_log_filename = f'logs/history_{current_time_s}.log'
                            history_log_folder = os.path.join(project_path, history_log_filename)
                            os.makedirs(os.path.dirname(history_log_folder), exist_ok=True)
                            history_logger = setup_logger(history_log_folder)
                            history_logger.info(f"History Log: {history_log}")

                            # LLM has been activated, so llm should be false and relative total movememt zero
                            llm_activated = True
                            user_relative_total_movement = 0

                            write_to_excel(filtered_names_high_dot_counts, filtered_names_low_distance_counts, filtered_names_time_to_approach, filtered_names_dot, filtered_names_distances, predictions_dict, goals_dict, args.sequence_path, parameter_folder_name, current_time_s)
            
            # ==============================================
            # Objects Inside the radius & LLM activation conditions
            # ==============================================  
            
            # Get objects within 1.5 meter radius 
            """
            At each timestampe returns a list with all the objects inside the radius of 1.5 meters
            """
            current_objects_within_radius = object_within_radius(visible_distance_device_objects_scene, visible_obj_names, radius = 1.5) 
            
            # Calculate conditions
            """
            
            """
            group_analyzer.add_objects(current_time_s, current_objects_within_radius)
            user_objects = group_analyzer.compare_objects()                                   # Me: Boolean value (True/False) if user moved to different area based on objects around
            users_move = group_analyzer.user_move(user_relative_total_movement)               # Me: Boolean value (True/False) if user moved to different area based on movement
            time_since_last_activation = current_time_s - last_activation_time                # Me: Time 
        
            # Conditions to enable the LLM
            """
            4 conditions for the LLM
            - Not earlier than 2 seconds 
            - Not later than 5 seconds 
            - user's movement significant 
            - user's is not surrounding by the same of objects 
            """
            
            if time_since_last_activation > 2: 
                if users_move or user_objects or time_since_last_activation > 5:
                    user_relative_total_movement = 0
                    last_activation_time = current_time_s
                    llm_activated = False
        
        # ==============================================
        # Store the predictions
        # ==============================================  
        
        # # Define the path for saving the predictions
        # predictions_folder = os.path.join(project_path, 'data', 'predictions', sequence_path, parameter_folder_name)
        # os.makedirs(predictions_folder, exist_ok=True)

        # # Save the predictions to a JSON file
        # prediction_file = os.path.join(predictions_folder, 'large_language_model_prediction.json')
        # with open(prediction_file, 'w') as json_file:
        #     json.dump(predictions_dict, json_file, indent=4)

         # Define the path for saving the predictions
        predictions_folder = os.path.join(project_path, 'data', 'predictions', sequence_path, parameter_folder_name)
        os.makedirs(predictions_folder, exist_ok=True)

        # Save the predictions to a JSON file
        possibilities_file = os.path.join(predictions_folder, 'large_language_model_possbilities.json')
        with open(possibilities_file, 'w') as json_file:
            json.dump(objects_possibilities, json_file, indent=4)

        rationale_file = os.path.join(predictions_folder, 'large_language_model_rationale.json')
        with open(prediction_file, 'w') as json_file:
            json.dump(rationale_dict, json_file, indent=4)

        prediction_file = os.path.join(predictions_folder, 'large_language_model_prediction.json')
        with open(prediction_file, 'w') as json_file:
            json.dump(predictions_dict, json_file, indent=4)

        goals_file = os.path.join(predictions_folder, 'large_language_model_goals.json')
        with open(goals_file, 'w') as json_file:
            json.dump(goals_dict, json_file, indent=4)

        print(f"Saved predictions for parameters to {prediction_file}")
        
        # ==============================================
        # Prints
        # ==============================================  
        
        # Print basic sequence characteristics
        print(f"Loaded scene: {dataset_folder}")
        print("Scene characteristics:")
        print(f"\t Aria RGB frames count: {len(img_timestamps_ns)}")
        print(f"\t Skeleton count: {len(gt_provider.get_skeleton_ids())}")
        
        # Print objects info
        # print(f"\t Static object count: {len(static_obj_ids)}")
        # print(f"\t Dynamic object count (tracked): {len(dynamic_obj_pose_cache)}")
        # print(f"\t Dynamic object count (tracked and moved - estimated): {len(dynamic_obj_moved)}")

    if __name__ == "__main__":
        main()


