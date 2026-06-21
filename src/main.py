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

import time
import os                                   

from collections import deque, defaultdict  
from typing import Dict, List, Tuple, Deque 

from itertools import product           

import logging
import os
import csv
import json

# from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection 

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
    log_vector_2,
    clear_logs_names,                                        # Me: At each timestep clear the objects from the visualization tool 
    clear_logs_ids,
)

from utils.tools import (
    transform_point,                                          # Me: Transformation point from scene to camera frame
    visibility_mask,                                          # Me: Check which points are visible and which are not visible
    exponential_filter,                                       # Me: Filter the velocity with exponential mean average 
    object_within_radius,                                     # Me: Check the objects that are close to a user
    user_movement_calculation,                                # Me: User's movement calculation   
    load_config,                                              # Me: Load the configuration file   
)

from utils.openai_models_work import ( 
    activate_llm,                                             # Me: Query the LLM
    setup_logger,                                             # Me: Setup the logger
    append_to_history_string,                                 # Me: Write the history in a string
    process_llm_response,                                     # Me: Post processing of LLM output                                           
)

from utils.llama import (
    activate_llama
)

from utils.objectsGroup_user import (
    ObjectGroupAnalyzer                                       # Me: Class to analyse the objects around the user and specify if the user is changing areas
)

from utils.stats import (
    Statistics                                                # Me: Keep statistics for high dot value and low distance
)

# Optional deps for video generation
try:
    import imageio.v3 as iio
except Exception:
    iio = None
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_path", type=str, required=True, help="path to the ADT sequence")
    parser.add_argument("--device_number", type=int, default=0, help="Device_number you want to visualize, default is 0")
    parser.add_argument("--down_sampling_factor", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--jpeg_quality", type=int, default=75, help=argparse.SUPPRESS)
    parser.add_argument("--rrd_output_path", type=str, default="", help=argparse.SUPPRESS)                                                # Me: If this path is set, we will save the rerun (.rrd) file to the given path
    parser.add_argument("--use_llm", action='store_true',help="If you include it in arguments becomes True")                              # Me: added by Petros, if there is a value that 
    parser.add_argument("--runrr", action='store_true',help="Run the the visualization part..same as above")   
    parser.add_argument("--visualize_objects", action='store_true',help="Visualize the objects in the rerun.io")   
    parser.add_argument("--make_video", action='store_true', help="Export annotated video with LLM pauses")
    parser.add_argument("--video_out", type=str, default="", help="Output mp4 path; defaults under predictions folder")
    parser.add_argument("--pause_duration", type=float, default=1.5, help="Seconds to pause-overlay on LLM activation")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    return parser.parse_args()

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
main_logger = logging.getLogger(__name__)

# ==============================================
# Parameters Settting
# ==============================================

# Parameters for the language model module       
time_thresholds = [2]                       # Time (in seconds) before interaction to activate the LLM
avg_dot_threshold_highs = [0.9]             # Filter objects: keep only those with an average dot product above this value
avg_dot_threshold_lows = [0.2]              # Filter objects: keep only those with an average dot product above this minimum value
avg_distance_threshold_highs = [3]          # Filter objects: keep only those with an average distance below this value
avg_distance_threshold_lows = [1]           # Filter objects: keep only those with an average distance above this minimum value
high_dot_thresholds = [0.9] # [0.9]         # Count objects with dot product values exceeding this threshold
distance_thresholds = [2] # [2]             # Count objects with distance values below this threshold
high_dot_counters_threshold = [60] #[45]    # Keep objects that exceed this count for dot product values above the threshold
distance_counters_threshold = [30] #[60]    # Keep objects that exceed this count for distance values below the threshold
variables_window_times = [3.0]              # Sliding time window (in seconds) for tracking these parameters
minimum_time_deactivated = [2.0]            # Minimum time (in seconds) the LLM remains deactivated after querying
maximum_time_deactivated = [5.0]            # Maximum time (in seconds) before the LLM is activated again after querying
user_relative_movement = [2.0]              # Threshold for user's relative movement (distance) after LLM is queried
object_percentage_overlap = [0.7]           # Percentage of overlap required for objects near the user to trigger reactivation

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
        "high_dot_counters_threshold": hdct,
        "distance_counters_threshold": dct,       # Corrected to match the earlier definition
        "window_time": w, 
        "minimum_time_deactivated": mintd,                
        "maximum_time_deactivated": maxtd,             
        "user_relative_movement": urm,                 
        "object_percentage_overlap" : obo,   
    }
    for t, adh, adl, adhg, adlg, hdt, dt, hdct, dct, w, mintd, maxtd, urm, obo in product(
        time_thresholds, avg_dot_threshold_highs, avg_dot_threshold_lows, 
        avg_distance_threshold_highs, avg_distance_threshold_lows,
        high_dot_thresholds, distance_thresholds,
        high_dot_counters_threshold, distance_counters_threshold, variables_window_times, 
        minimum_time_deactivated, maximum_time_deactivated, user_relative_movement, object_percentage_overlap   
    )
]

# ==============================================
# Run through the parameter combinatons in parallel
# ==============================================
    
for parameters in param_combinations: # TODO parallel 4 loop

    def main():
        
        work_in_xz_plane = True
        
        # ==============================================
        # Filenames / Paths  & Load of the Data
        # ==============================================
        args = parse_args()
        
        # Base folder path for saving predictions
        config = load_config()
        project_path = os.path.expanduser(config["project_path"])
        sequence_path = args.sequence_path

        print ("Sequence path: ", sequence_path)
        # Datasets path
        dataset_folder = os.path.join(sequence_path)
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
                f"distcount_{parameters['distance_counters_threshold']}"
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
        input("Press Enter to continue...") # Me: Wait for user input before proceeding. This is useful for debugging or pausing the script at a specific point.
        
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
        llama = False                                                                   # Me: Llama is activated or not      
        llm_activated = False                                                           # Me: LLM activated or not
        history_log: Dict                                                               # Me: This is for now 

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
        group_analyzer = ObjectGroupAnalyzer(parameters["object_percentage_overlap"], parameters["user_relative_movement"], history_size=15, future_size=5)                                         
        statistics = Statistics(
                                parameters['window_time'], 
                                parameters["high_dot_threshold"], 
                                parameters["distance_threshold"], 
                                parameters["distance_threshold"], 
                                parameters["time_threshold"]
                                )      # Me: Initialize the Object Statistics instance

        # =============================
        # Video writer setup
        # =============================
        writer = None
        frame_size = None
        pause_frames = 0

        def draw_overlay(frame_np, lines):
            if Image is None:
                return frame_np
            img = Image.fromarray(frame_np)
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("Arial.ttf", 22)
            except Exception:
                font = ImageFont.load_default()
            padding = 8
            line_height = 24
            box_width = max([draw.textlength(l, font=font) for l in lines] + [0]) + 2 * padding
            box_height = line_height * len(lines) + 2 * padding
            draw.rectangle([0, 0, box_width, box_height], fill=(0, 0, 0, 180))
            for idx, text in enumerate(lines):
                draw.text((padding, padding + idx * line_height), text, fill=(255, 255, 255), font=font)
            return np.array(img)
        
        # ==============================================
        # Load the Ground truth data 
        # ==============================================
        
        # Locate the ground-truth file. `sequence_path` may be a full path or a name,
        # so try the project gt folder first, then the sequence folder itself.
        _seq_name = os.path.basename(os.path.normpath(args.sequence_path))
        _gt_candidates = [
            os.path.join(project_path, 'data', 'gt', _seq_name, 'movement_time_dict.json'),
            os.path.join(args.sequence_path, 'movement_time_dict.json'),
        ]
        gt_file = next((p for p in _gt_candidates if os.path.exists(p)), None)
        if gt_file is None:
            raise FileNotFoundError(
                "movement_time_dict.json not found. Run gt.py first, or place it in "
                + " or ".join(_gt_candidates)
            )
        with open(gt_file, 'r') as json_file:
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

            # Get numpy frame for video export (RGB)
            frame_np = None
            if args.make_video and image_with_dt.is_valid():
                try:
                    frame_np = image_with_dt.data().to_numpy_array()
                    if frame_np.dtype != np.uint8:
                        frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                    # initialize writer on first frame
                    if writer is None and iio is not None:
                        frame_size = (frame_np.shape[1], frame_np.shape[0])
                        # video_out path will be finalized after predictions folder init; temporarily hold
                        pass
                except Exception as _:
                    frame_np = None
            
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
                low_dot_mask = np.array([visible_avg_dots[obj_id] > parameters["avg_dot_low"] for obj_id in visible_obj_ids])                        # Me: low dot mask
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
                if object_id in filtered_low_distance_counts and len(filtered_low_distance_counts[object_id]) >= parameters["distance_counters_threshold"]:
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
                        # print(f"\t Time to approach {object_name} is less than 2 seconds: {object_time_xz}")

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
            Object with distance counts  > threshold but also has high dot counts --> visibility is certain as distance counts over threshold
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
                # print("the 3 criteria have been satisfied")

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
                        
                        if llama == True:
                            print ("Llama is activated")
                        else: 
                            print ("Llama is not activated")

                        # use the LLM
                        if llama == True:
                            llm_response = activate_llama(history_log_string, parameters)
                        else: 
                            llm_response = activate_llm(history_log, parameters)
                        # process the output of the LLM
                        objects_possibility, rationale, predicted_objects, goal = process_llm_response(llm_response)
                        
                        print(f"Objects Possibility: {objects_possibility}")
                        print(f"Rationale: {rationale}")
                        print(f"Predicted Objects: {predicted_objects}")
                        print(f"Goal: {goal}")

                        # pause the execution and wait for the user to press Enter
                        # input("Press Enter to continue...")

                        # Update the last time LLM was activated 
                        last_activation_time = current_time_s

                        # LLM has been activated, so llm should be false and relative total movememt zero
                        llm_activated = True
                        user_relative_total_movement = 0

                        # dictionaries & lists
                        objects_possibility_dict[current_time_s] = objects_possibility
                        rationale_dict[current_time_s] = rationale
                        predictions_dict[current_time_s] = predicted_objects
                        goals_dict[current_time_s] = goal

                        objects_possibilities.append(objects_possibility)
                        rationales.append(rationale)
                        predictions.append(predicted_objects)
                        goals.append(goal)
                        llm_times.append(current_time_s)

                        # Prepare pause overlay text and schedule pause frames
                        if args.make_video and frame_np is not None:
                            # Compute lead-time for first predicted object that exists in GT
                            lead_time_txt = "n/a"
                            if isinstance(predicted_objects, list) and len(predicted_objects) > 0:
                                for cand in predicted_objects:
                                    if cand in movement_time_dict:
                                        lead = movement_time_dict[cand]["start_time"] - current_time_s
                                        lead_time_txt = f"{max(0.0, float(lead)):.2f}s"
                                        break
                            overlay_lines = [
                                f"t={current_time_s:.2f}s",
                                f"LLM predicted: {', '.join(predicted_objects) if isinstance(predicted_objects, list) else str(predicted_objects)}",
                                f"Lead-time: {lead_time_txt}"
                            ]
                            frame_np = draw_overlay(frame_np, overlay_lines)
                            pause_frames = int(max(0.0, args.pause_duration) * max(1, args.fps))

                        # Rerun overlay and pause simulation on timeline
                        if args.runrr:
                            # Log a textual overlay entity with prediction summary
                            try:
                                rr.log("overlay/llm", rr.TextLog(f"t={current_time_s:.2f}s | Pred: {', '.join(predicted_objects) if isinstance(predicted_objects, list) else str(predicted_objects)} | Lead: {lead_time_txt}"))
                            except Exception:
                                pass
                            # Simulate pause by emitting repeated frames at synthetic timestamps
                            try:
                                synthetic_pause_frames = int(max(0.0, args.pause_duration) * max(1, args.fps))
                                if synthetic_pause_frames > 0 and image_with_dt.is_valid():
                                    for i in range(synthetic_pause_frames):
                                        synth_ts_ns = int(timestamp_ns + (i + 1) * (1e9 / max(1, args.fps)))
                                        set_rerun_time(rr, synth_ts_ns)
                                        process_and_log_image(rr, args, image_with_dt)
                            except Exception:
                                pass

                        # Log the output of LLM in log file 
                        log_filename = f'logs/time_{current_time_s}.log'    
                        log_folder = os.path.join(project_path, log_filename)               
                        os.makedirs(os.path.dirname(log_folder), exist_ok=True)
                        logger = setup_logger(log_folder)
                        logger.info(f"LLM Response: {llm_response}")
                        
                        # Log history_log content in the log file
                        history_log_filename = f'logs/history_{current_time_s}.log'
                        history_log_folder = os.path.join(project_path, history_log_filename)
                        os.makedirs(os.path.dirname(history_log_folder), exist_ok=True)
                        history_logger = setup_logger(history_log_folder)
                        history_logger.info(f"History Log: {history_log}")

                        # Write the conditions to excel for debugging purposes
                        # write_to_excel(filtered_names_high_dot_counts, filtered_names_low_distance_counts, filtered_names_time_to_approach, filtered_names_dot, filtered_names_distances, predictions_dict, goals_dict, args.sequence_path, parameter_folder_name, current_time_s)
            
            # ==============================================
            # Log Only Predicted Objects in rerun.io
            # ==============================================

            # Clear previously logged objects and lines
            if args.runrr:
                clear_logs_ids(rr, previous_obj_ids)
                clear_logs_names(rr, previous_obj_names)

            previous_obj_ids.clear()
            previous_obj_names.clear()

            # Log only the predicted objects
            if args.runrr and args.visualize_objects:
                for obj_id, obj_position_scene in zip(filtered_obj_ids, filtered_obj_positions_scene):
                    instance_info = gt_provider.get_instance_info_by_id(obj_id)
                    object_name = instance_info.name

                    # Check if the object is in the predicted list
                    if object_name in predictions_dict.get(current_time_s, []):
                        # Log the bounding box
                        bbox_3d = bboxes3d[obj_id]
                        aabb_coords = bbox3d_to_line_coordinates(bbox_3d.aabb)
                        obb = np.zeros(shape=(len(aabb_coords), 3))
                        for i in range(len(aabb_coords)):
                            aabb_pt = aabb_coords[i]
                            aabb_pt_homo = np.append(aabb_pt, [1])
                            obb_pt = (bbox_3d.transform_scene_object.to_matrix() @ aabb_pt_homo)[0:3]
                            obb[i] = obb_pt

                        log_object(rr, instance_info, obb)

                        # Log the line connecting the user camera to the object
                        log_object_line(rr, instance_info, T_Scene_Cam.translation(), obj_position_scene[0])

                        # Add to previous_obj_ids to clear in the next timestep
                        previous_obj_names.add(instance_info.name)
                        previous_obj_ids.add(obj_id)
                        
            # ==============================================
            # Objects Inside the radius & LLM activation conditions
            # ==============================================  
            
            if llm_activated == True:

                """
                This module verifies if reactivating the LLM is necessary by checking whether it has 
                been previously queried. If not, running the algorithm is unnecessary, allowing for 
                computational cost savings.
                """

                # Get objects within 1.5 meter radius 
                """
                At each timestampe returns a list with all the objects inside the radius of 1.5 meters
                """
                current_objects_within_radius = object_within_radius(visible_distance_device_objects_scene, visible_obj_names, radius = 1.5) 
                
                # Calculate conditions
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
                
                if time_since_last_activation >  parameters["minimum_time_deactivated"]: 
                    if users_move or user_objects or time_since_last_activation > parameters["maximum_time_deactivated"]:
                        user_relative_total_movement = 0
                        last_activation_time = current_time_s
                        llm_activated = False

            # =============================
            # Write video frame (and pause copies)
            # =============================
            if args.make_video and iio is not None and frame_np is not None:
                try:
                    if writer is None:
                        # finalize output path now that predictions_folder is known later; default temp here
                        # we defer opening until predictions_folder is created at end; instead collect frames would be heavy.
                        # So open now: if args.video_out empty, put under project_path/tmp first; will mirror finale path name.
                        if args.video_out:
                            video_path = os.path.expanduser(args.video_out)
                        else:
                            tmp_dir = os.path.join(project_path, 'data', 'predictions', sequence_path, 'tmp')
                            os.makedirs(tmp_dir, exist_ok=True)
                            video_path = os.path.join(tmp_dir, 'preview.mp4')
                        writer = iio.get_writer(video_path, fps=max(1, args.fps), codec='libx264')
                    writer.append_data(frame_np)
                    # append pause frames if any
                    if pause_frames > 0:
                        for _ in range(pause_frames):
                            writer.append_data(frame_np)
                        pause_frames = 0
                except Exception as _:
                    pass
            

            # if args.runrr and args.visualize_objects:

                # for obj_id, dot_product, distance, obj_position_cam, obj_position_scene in zip(
                #                                                                         filtered_obj_ids, 
                #                                                                         filtered_dot_products,
                #                                                                         filtered_distances,
                #                                                                         filtered_obj_positions_cam, 
                #                                                                         filtered_obj_positions_scene,
                #                                                                         # objects_time_approach_list,
                #                                                                         ):
                    
                #     instance_info = gt_provider.get_instance_info_by_id(obj_id) # instance info 
                #     object_name = instance_info.name   # name

                #     # ==============================================
                #     # Handling the object coordinates 
                #     # ==============================================

                #     """
                #     ** bbox_3d.aabb --> represents the minimum and maximum coordinates that define the AABB --> [x_min, y_min, z_min, x_max, y_max, z_max]
                #     ** bbox_3d.transform_scene_object is a transformation from οbject coordinate system to scene coordinate system (4x4 matrix)
                #     ** AABB - Axis-Aligned Bounding Box ---> 16 points that form 8 pairs (edges), which define a 3D bounding box. Some of these points are repeated.
                #     ** OBB  - Oriented bounding box     ---> same dimension with aabb_coords. it will store the values on the new coordinate system
                #     """
                #     bbox_3d = bboxes3d[obj_id]                                                     
                #     aabb_coords = bbox3d_to_line_coordinates(bbox_3d.aabb)                         
                #     obb = np.zeros(shape=(len(aabb_coords), 3))                                    
                #     for i in range(0, len(aabb_coords)):                                           
                #         aabb_pt = aabb_coords[i]                                                  
                #         aabb_pt_homo = np.append(aabb_pt, [1])                                    
                #         obb_pt = (bbox_3d.transform_scene_object.to_matrix() @ aabb_pt_homo)[0:3]  
                #         obb[i] = obb_pt
                        
                #     # ==============================================
                #     # VECTORS - POSITION / VELOCITY / DISTANCE / PROJECTION  
                #     # ==============================================  
                    
                #     # POSITION - vectors
                #     position_vector_xyz = T_Scene_Device.translation()[0]   
                #     position_vector_xz = np.array([position_vector_xyz[0], 0, position_vector_xyz[2]])
                    
                #     # VELOCITY - vector in 3D
                #     velocity_vector_xyz = T_Scene_Device.rotation().to_matrix() @ user_velocity_device
                #     velocity_vector_xz = np.array([velocity_vector_xyz[0], 0, velocity_vector_xyz[2]])
                #     velocity_vector_xyz_on_user = velocity_vector_xyz + position_vector_xyz     # End of the velocity vector (scaled by velocity)
                #     velocity_vector_xz_on_user = velocity_vector_xz + position_vector_xyz       # Velocity vector in 2D 
                    
                #     # DISTANCE vector in 3D from user to object
                #     displacement_vector = (obj_position_scene[0] - position_vector_xyz) 
                #     displacement_unit_vector = displacement_vector / np.linalg.norm(displacement_vector)
                #     displacement_vector_xz = np.array([displacement_vector[0], 0, displacement_vector[2]])
                #     displacement_unit_vector_xz = np.array([displacement_unit_vector[0], 0, displacement_unit_vector[2]])
                    
                #     # DISTANCE vector in 3D from user to object ON USER
                #     displacement_vector_on_user = (obj_position_scene[0] - position_vector_xyz) + position_vector_xyz
                #     displacement_unit_vector_on_user = displacement_vector / np.linalg.norm(displacement_vector) + position_vector_xyz
                #     displacement_vector_xz_on_user = np.array([displacement_vector[0], 0, displacement_vector[2]]) + position_vector_xyz
                #     displacement_unit_vector_xz_on_user = np.array([displacement_unit_vector[0], 0, displacement_unit_vector[2]]) + position_vector_xyz
                    
                #     # PROJECTED VELOCITY vector in 3D from user to object  (Project the user's velocity onto the displacement vector)
                #     projected_velocity_xyz = np.dot(velocity_vector_xyz, displacement_unit_vector) * displacement_unit_vector 
                #     projected_velocity_xz = np.array([projected_velocity_xyz[0], 0, projected_velocity_xyz[2]])
                #     projected_velocity_xyz_on_user = projected_velocity_xyz + position_vector_xyz 
                #     projected_velocity_xz_on_user = projected_velocity_xz + position_vector_xyz
                    
                #     # ==============================================
                #     # VISUALISATION ORIGIN
                #     # ==============================================  
                    
                #     # POSITION - Visualize the position vector (3D & 2D)
                #     args.runrr and log_vector_2(rr, "origin_position_xyz", np.array([0,0,0]), position_vector_xyz)
                #     args.runrr and log_vector_2(rr, "origin_position_ema_xyz", np.array([0,0,0]), user_ema_position)
                #     args.runrr and log_vector_2(rr, "origin_position_xz", np.array([0,0,0]), position_vector_xz)
                    
                #     # VELOCITY - Visualize the velocity vector as a line in the GUI
                #     args.runrr and log_vector_2(rr, "origin_velocity_xyz", np.array([0,0,0]), velocity_vector_xyz)
                #     args.runrr and log_vector_2(rr, "origin_velocity_ema_xyz", np.array([0,0,0]), user_ema_velocity)
                #     args.runrr and log_vector_2(rr, "origin_velocity_xz", np.array([0,0,0]), velocity_vector_xz)
                    
                #     # PROJECTED VELOCITY - Visualize the velocity vector projected towards the  
                #     args.runrr and log_vector_2(rr, "origin_projected_velocity_xyz", np.array([0,0,0]), projected_velocity_xyz)
                #     args.runrr and log_vector_2(rr, "origin_projected_velocity_xz", np.array([0,0,0]), projected_velocity_xz)
                    
                #     # DISTANCE 3D - Visualize the position vector (3D & 2D)
                #     args.runrr and log_vector_2(rr, "origin_distance_vector_xyz", np.array([0,0,0]), displacement_vector)
                #     args.runrr and log_vector_2(rr, "origin_distance_unit_vector_xyz", np.array([0,0,0]), displacement_unit_vector)
                #     args.runrr and log_vector_2(rr, "origin_distance_vector_xz", np.array([0,0,0]), displacement_vector_xz)
                #     args.runrr and log_vector_2(rr, "origin_distance_unit_vector_xz", np.array([0,0,0]), displacement_unit_vector_xz)
                    
                #     # OBJECT POSITION VECTOR - Visualise the position of the vector in the space 
                #     args.runrr and log_vector_2(rr, "origin_object_position", np.array([0,0,0]), obj_position_scene[0])
                #     args.runrr and log_vector_2(rr, "camera_to_object", camera_position_scene, filtered_vector_camera_objects_scene[0] + camera_position_scene)
                #     args.runrr and log_vector_2(rr, "origin_camera_to_object", np.array([0,0,0]), filtered_vector_camera_objects_scene[0])
                    
                #     # ==============================================
                #     # VISUALISATION ON USER 
                #     # ==============================================  
                    
                #     # VELOCITY - Visualize the velocity vector as a line in the GUI
                #     args.runrr and log_vector_2(rr, "device_velocity_xyz", position_vector_xyz, velocity_vector_xyz_on_user)
                #     args.runrr and log_vector_2(rr, "device_velocity_xz", position_vector_xyz, velocity_vector_xz_on_user)
                    
                #     # PROJECTED VELOCITY - Visualize the velocity vector projected towards the  
                #     args.runrr and log_vector_2(rr, f"device_projected_velocity_xyz", position_vector_xyz, projected_velocity_xyz_on_user)
                #     args.runrr and log_vector_2(rr, f"device_projected_velocity_xz", position_vector_xyz, projected_velocity_xz_on_user)
                    
                #     # DISTANCE 3D - Visualize the position vector (3D & 2D) - ON USER 
                #     args.runrr and log_vector_2(rr, f"device_distance_vector_xy", position_vector_xyz, displacement_vector_on_user)
                #     args.runrr and log_vector_2(rr, f"device_distance_unit_vector_xyz", position_vector_xyz, displacement_unit_vector_on_user)
                #     args.runrr and log_vector_2(rr, f"device_distance_vector_xz", position_vector_xyz, displacement_vector_xz_on_user)
                #     args.runrr and log_vector_2(rr, f"device_distance_unit_vector_xz", position_vector_xyz, displacement_unit_vector_xz_on_user)
                    
                #     # OBJECT LINE - Visualize the line from camera to object
                #     args.runrr and log_object_line(rr, instance_info, position_vector_xyz, obj_position_scene[0])
                #     args.runrr and log_object_line(rr, instance_info, camera_position_scene[0], obj_position_scene[0])

                #     # OBJECT BOUNDING BOX - Visualize the object
                #     args.runrr and log_object(rr, instance_info, obb)
                    
                #     # ==============================================
                #     # VISUALISATION FRAMES
                #     # ==============================================    
                    
                #     # CAMERA Z-AXIS ONLY ROTATION TO SCENE
                #     cam_z_axis_rotation = (T_Scene_Cam.rotation().to_matrix() @ cam_z_axis)[:,0]
                #     cam_z_axis_rotation_xz =  np.array([cam_z_axis_rotation[0], 0, cam_z_axis_rotation[2]]) 
                #     args.runrr and log_vector_2(rr, "origin_camera_z_axis_rotation_only", np.array([0,0,0]), cam_z_axis_rotation)
                #     args.runrr and log_vector_2(rr, "origin_camera_z_axis_rotation_only_xz", np.array([0,0,0]), cam_z_axis_rotation_xz)
                #     args.runrr and log_vector_2(rr, "origin_camera_z_axis", np.array([0,0,0]), cam_z_axis_scene)

                #     # CAMERA Z-AXIS ONLY ROTATION TO SCENE
                #     # cam_z_axis_rotation = T_Scene_Cam.rotation().to_matrix() @ cam_z_axis     
                #     # args.runrr and log_vector(rr, "origin_camera_z_axis_rotation_only", np.array([0,0,0]), cam_z_axis_rotation)
                #     # args.runrr and log_vector(rr, "origin_camera_z_axis", np.array([0,0,0]), cam_z_axis_scene)
                    
                #     # CAMERA
                #     args.runrr and log_vector_2(rr, "camera_x_axis", camera_position_scene[0], cam_x_axis_scene)
                #     args.runrr and log_vector_2(rr, "camera_y_axis", camera_position_scene[0], cam_y_axis_scene)
                #     args.runrr and log_vector_2(rr, "camera_z_axis", camera_position_scene[0], cam_z_axis_scene)
                    
                #     # DEVICE
                #     args.runrr and log_vector_2(rr, "device_x_axis", position_vector_xyz, device_x_axis_scene)
                #     args.runrr and log_vector_2(rr, "device_y_axis", position_vector_xyz, device_y_axis_scene)
                #     args.runrr and log_vector_2(rr, "device_z_axis", position_vector_xyz, device_z_axis_scene)
                    
                #     # WORLD
                #     args.runrr and log_vector_2(rr, "world_x_axis", np.array([0,0,0]), world_x_axis)
                #     args.runrr and log_vector_2(rr, "world_y_axis", np.array([0,0,0]), world_y_axis)
                #     args.runrr and log_vector_2(rr, "world_z_axis", np.array([0,0,0]), world_z_axis)
                    
                #     # Add to previous_obj_ids to clear in the next timestep
                #     previous_obj_names.add(instance_info.name)
                #     previous_obj_ids.add(obj_id)

                #     if gt_provider.get_instance_info_by_id(object_id).name == "ChoppingBoard":
                #         print('stop')
        # ==============================================
        # Store the predictions of the LLM
        # ==============================================  
        
        # Define the path for saving the predictions
        predictions_folder = os.path.join(project_path, 'data', 'predictions', sequence_path, parameter_folder_name)
        os.makedirs(predictions_folder, exist_ok=True)

        # Save the predictions to a JSON file
        possibilities_file = os.path.join(predictions_folder, 'large_language_model_possbilities.json')
        with open(possibilities_file, 'w') as json_file:
            json.dump(objects_possibility_dict, json_file, indent=4)

        rationale_file = os.path.join(predictions_folder, 'large_language_model_rationale.json')
        with open(rationale_file, 'w') as json_file:
            json.dump(rationale_dict, json_file, indent=4)

        prediction_file = os.path.join(predictions_folder, 'large_language_model_prediction.json')
        with open(prediction_file, 'w') as json_file:
            json.dump(predictions_dict, json_file, indent=4)

        goals_file = os.path.join(predictions_folder, 'large_language_model_goals.json')
        with open(goals_file, 'w') as json_file:
            json.dump(goals_dict, json_file, indent=4)

        # Finalize video output: if writer used without explicit path, move tmp preview under predictions folder
        try:
            if 'writer' in locals() and writer is not None:
                writer.close()
                if not args.video_out and frame_size is not None:
                    src_tmp = os.path.join(project_path, 'data', 'predictions', sequence_path, 'tmp', 'preview.mp4')
                    dst_path = os.path.join(predictions_folder, f"preview_{parameter_folder_name}.mp4")
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    if os.path.exists(src_tmp):
                        try:
                            os.replace(src_tmp, dst_path)
                        except Exception:
                            pass
        except Exception:
            pass

        print(f"Saved predictions for parameters to {prediction_file}")

# ==============================================
# Run for different parameter combinations in parallel
# ==============================================
if __name__ == "__main__":

    start_time = time.time()
    main()
    end_time = time.time()
    
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

