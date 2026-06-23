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

# rerun-sdk exposes `rerun` via a path-style .pth that uv-managed venvs may not
# process; add rerun_sdk to sys.path so --runrr / --make_video visualization works.
import sys as _sys, os as _os
for _p in list(_sys.path):
    if _os.path.isdir(_os.path.join(_p, "rerun_sdk")):
        _sys.path.insert(0, _os.path.join(_p, "rerun_sdk")); break
try:
    import rerun as rr
except ImportError:
    rr = None

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

# Visualization imports are optional — only used when --runrr is passed.
try:
    from projectaria_tools.utils.rerun_helpers import (
        AriaGlassesOutline,                                  # Me: Return a list of points to be used to draw the outline of the glasses (line strip).
        ToTransform3D                                        # Me: Helper function to convert Sophus SE3D pose to a Rerun Transform3D
    )
    from visualization.rr import (                           # Me: added by Petros
        initialize_rerun_viewer,                             # Me: Initialize the rerun software
        log_camera_calibration,                              # Me: Log the camera features
        log_aria_glasses,                                    # Me: Log the aria glasses
        set_rerun_time,
        process_and_log_image,
        log_device_transformations,
        log_dynamic_object,
        log_object,                                          # Me: Log an object
        log_object_line,                                     # Me: Log an object Line
        log_vector,                                          # Me: Log the velocity line
        log_vector_2,
        clear_logs_names,                                    # Me: At each timestep clear the objects from the visualization tool
        clear_logs_ids,
    )
except ImportError:
    # rerun not available: visualization disabled. These names are only ever
    # referenced behind `args.runrr and ...`, so they stay unused without --runrr.
    AriaGlassesOutline = ToTransform3D = None
    initialize_rerun_viewer = log_camera_calibration = log_aria_glasses = None
    set_rerun_time = process_and_log_image = log_device_transformations = None
    log_dynamic_object = log_object = log_object_line = log_vector = log_vector_2 = None
    clear_logs_names = clear_logs_ids = None

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

from experiment_config import build_param_combinations, make_parameter_folder_name
from perception import (
    get_visible_objects, compute_scene_axes, compute_dots_and_distances, filter_objects_by_average,
)
from activation import evaluate_activation_criteria, check_reactivation
from llm_step import query_and_log_llm
from state import UserMotionState, ActivationState


def save_predictions(folder, possibilities, rationale, predictions, goals):
    """Write the four LLM output dicts as JSON into ``folder``.

    Returns the path of the prediction file.
    """
    os.makedirs(folder, exist_ok=True)
    outputs = {
        "large_language_model_possbilities.json": possibilities,
        "large_language_model_rationale.json": rationale,
        "large_language_model_prediction.json": predictions,
        "large_language_model_goals.json": goals,
    }
    for name, data in outputs.items():
        with open(os.path.join(folder, name), "w") as f:
            json.dump(data, f, indent=4)
    return os.path.join(folder, "large_language_model_prediction.json")


def run_experiment(parameters):
    
    work_in_xz_plane = True
    
    # ==============================================
    # Filenames / Paths  & Load of the Data
    # ==============================================
    args = parse_args()
    
    # Base folder path for saving predictions
    config = load_config()
    project_path = os.path.expanduser(config["project_path"])
    repo_root = os.path.dirname(project_path)          # outputs go under <repo>/results/
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
        json.dump(build_param_combinations(), file)
        
    # Parameters folder name
    parameter_folder_name = make_parameter_folder_name(parameters)

    # Log the paths
    main_logger.info("Sequence folder: %s", dataset_folder)
    main_logger.info("Project path:    %s", project_path)
    main_logger.info("VRS file:        %s", vrsfile)
    main_logger.info("GT trajectory:   %s", ADT_trajectory_file)

    try:
        paths_provider = AriaDigitalTwinDataPathsProvider(dataset_folder)
        data_paths = paths_provider.get_datapaths_by_device_num(args.device_number)
        gt_provider = AriaDigitalTwinDataProvider(data_paths)
    except Exception as e:
        main_logger.error("Failed to load ADT provider: %s", e)
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
    llama = False                                                                   # Me: Llama is activated or not      
    activation = ActivationState()                                                  # Me: LLM activation flags + output archives
    motion = UserMotionState()                                                      # Me: cross-frame user motion state
    user_total_movement = 0                                                         # Me: total trajectory length
    user_relative_total_movement = 0                                                # Me: movement since last LLM activation
    objects_within_radius = []                                                      # Me: Objects in the vicinity of the user
    previous_objects_within_radius = []                                             # Me: Is used to check if the user is stll in the same area that LLM has been activated in order to avoid reactivatiomn of the LLM 
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
        os.path.join(repo_root, 'results', 'gt', _seq_name, 'movement_time_dict.json'),
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
                                                                                
        aria_3d_pose_with_dt = gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamp_ns)
        if aria_3d_pose_with_dt.is_valid():
            p_dev_scene = aria_3d_pose_with_dt.data()
            (T_Scene_Device, T_Scene_Cam, user_position_scene, user_velocity_device,
             user_ema_position, _movement_timestep) = motion.update(aria_3d_pose_with_dt, T_Device_Cam, current_time_s)
            user_total_movement += _movement_timestep
            user_relative_total_movement += _movement_timestep
            args.runrr and log_device_transformations(rr, p_dev_scene, T_Device_Cam, ToTransform3D)
        
        # ============================================== Objects + visibility (perception.py)
        _vo = get_visible_objects(gt_provider, timestamp_ns, T_Scene_Cam, rgb_camera_calibration)
        bboxes3d = _vo.bboxes3d
        obj_ids = _vo.obj_ids
        obj_names = _vo.obj_names
        obj_positions_scene = _vo.obj_positions_scene
        T_Cam_Scene = _vo.T_Cam_Scene
        obj_positions_cam = _vo.obj_positions_cam
        valid_mask = _vo.valid_mask
        T_scene_object = _vo.T_scene_object
        object_ids = visible_obj_ids = _vo.visible_obj_ids
        visible_obj_names = _vo.visible_obj_names
        object_positions = visible_obj_positions_scene = _vo.visible_obj_positions_scene
        visible_obj_positions_cam = _vo.visible_obj_positions_cam
        
        # ============================================== Scene / camera axes (perception.py)
        _axes = compute_scene_axes(T_Scene_Cam, T_Scene_Device)
        cam_x_axis_scene = _axes.cam_x_axis_scene
        cam_y_axis_scene = _axes.cam_y_axis_scene
        cam_z_axis_scene = _axes.cam_z_axis_scene
        cam_z_axis_rotation = _axes.cam_z_axis_rotation
        device_x_axis_scene = _axes.device_x_axis_scene
        device_y_axis_scene = _axes.device_y_axis_scene
        device_z_axis_scene = _axes.device_z_axis_scene
        world_x_axis = _axes.world_x_axis
        world_y_axis = _axes.world_y_axis
        world_z_axis = _axes.world_z_axis
        
        # ============================================== Dot products + distances (perception.py)
        _dd = compute_dots_and_distances(obj_positions_scene, user_position_scene, T_Scene_Cam, cam_z_axis_rotation, valid_mask)
        camera_position_scene = _dd.camera_position_scene
        dot_products = _dd.dot_products
        distances = _dd.distances
        visible_dot_products = _dd.visible_dot_products
        visible_vector_camera_objects_scene = _dd.visible_vector_camera_objects_scene
        visible_distance_camera_objects_scene = _dd.visible_distance_camera_objects_scene
        visible_distance_device_objects_scene = _dd.visible_distance_device_objects_scene

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

        # ============================================== Filter visible objects by windowed average (perception.py)
        _fo = filter_objects_by_average(
            parameters, gt_provider, visible_obj_ids, visible_obj_names,
            visible_obj_positions_scene, visible_dot_products, visible_distance_device_objects_scene,
            visible_avg_dots, visible_avg_distances, visible_visibility_duration,
            visible_high_dot_counts, visible_low_distance_counts, visible_time_to_approach,
        )
        filtered_obj_ids = _fo.filtered_obj_ids
        filtered_obj_names = _fo.filtered_obj_names
        filtered_obj_positions_scene = _fo.filtered_obj_positions_scene
        filtered_dot_products = _fo.filtered_dot_products
        filtered_distances = _fo.filtered_distances
        filtered_high_dot_counts = _fo.filtered_high_dot_counts
        filtered_low_distance_counts = _fo.filtered_low_distance_counts
        filtered_names_high_dot_counts = _fo.filtered_names_high_dot_counts
        filtered_names_low_distance_counts = _fo.filtered_names_low_distance_counts
        filtered_names_time_to_approach = _fo.filtered_names_time_to_approach
        filtered_names_duration = _fo.filtered_names_duration
        
        # ============================================== LLM activation criteria (activation.py)
        _crit, all_unique_object_names_with_high_dot = evaluate_activation_criteria(
            _fo, parameters, statistics, gt_provider,
            user_velocity_device, user_ema_position, T_Scene_Device,
            gt_object_names, gt_start_times, gt_end_times,
            current_time_s, all_unique_object_names_with_high_dot,
        )
        high_dot_counts_but_also_distance = _crit.high_dot_counts_but_also_distance
        low_distance_counts_but_also_high_dot = _crit.low_distance_counts_but_also_high_dot
        less_than_2_seconds_dict = _crit.less_than_2_seconds_dict
        filtered_names_high_dot_counts_and_distance_counts = _crit.names_high_dot_counts_and_distance_counts
        filtered_names_low_distance_counts_and_high_dot_counts = _crit.names_low_distance_counts_and_high_dot_counts
        filtered_names_high_dot_counts_and_distance_values = _crit.names_high_dot_counts_and_distance_values
        filtered_names_low_distance_counts_and_high_dot_values = _crit.names_low_distance_counts_and_high_dot_values
        time_to_approach_dict = _crit.time_to_approach_dict
        high_dot_history = _crit.high_dot_history
    
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

        # ============================================== LLM activation step (llm_step.py)
        predicted_objects = query_and_log_llm(activation, _crit, args, parameters, llama, current_time_s, project_path)
        if predicted_objects is not None:
            user_relative_total_movement = 0

            # Pause overlay + scheduled pause frames (video export)
            if args.make_video and frame_np is not None:
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
                    f"Lead-time: {lead_time_txt}",
                ]
                frame_np = draw_overlay(frame_np, overlay_lines)
                pause_frames = int(max(0.0, args.pause_duration) * max(1, args.fps))

            # Rerun overlay + pause simulation on the timeline
            if args.runrr:
                try:
                    rr.log("overlay/llm", rr.TextLog(f"t={current_time_s:.2f}s | Pred: {', '.join(predicted_objects) if isinstance(predicted_objects, list) else str(predicted_objects)} | Lead: {lead_time_txt}"))
                except Exception:
                    pass
                try:
                    synthetic_pause_frames = int(max(0.0, args.pause_duration) * max(1, args.fps))
                    if synthetic_pause_frames > 0 and image_with_dt.is_valid():
                        for i in range(synthetic_pause_frames):
                            synth_ts_ns = int(timestamp_ns + (i + 1) * (1e9 / max(1, args.fps)))
                            set_rerun_time(rr, synth_ts_ns)
                            process_and_log_image(rr, args, image_with_dt)
                except Exception:
                    pass
        
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
                if object_name in activation.prediction_dict.get(current_time_s, []):
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
        
        activation.llm_activated, activation.last_activation_time, user_relative_total_movement = check_reactivation(
            activation.llm_activated, activation.last_activation_time, user_relative_total_movement,
            group_analyzer, visible_distance_device_objects_scene, visible_obj_names,
            current_time_s, parameters,
        )

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
                        tmp_dir = os.path.join(repo_root, 'results', 'predictions', os.path.basename(os.path.normpath(sequence_path)), 'tmp')
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
    
    # Define the path for saving the predictions. Outputs go under
    # <repo>/results/predictions/<seq_name>/ — data/ stays dataset-only and
    # the dataset folder is never overwritten.
    _seq_name = os.path.basename(os.path.normpath(sequence_path))
    predictions_folder = os.path.join(repo_root, 'results', 'predictions', _seq_name, parameter_folder_name)
    prediction_file = save_predictions(
        predictions_folder, activation.possibility_dict, activation.rationale_dict, activation.prediction_dict, activation.goal_dict
    )

    # Finalize video output: if writer used without explicit path, move tmp preview under predictions folder
    try:
        if 'writer' in locals() and writer is not None:
            writer.close()
            if not args.video_out and frame_size is not None:
                src_tmp = os.path.join(repo_root, 'results', 'predictions', os.path.basename(os.path.normpath(sequence_path)), 'tmp', 'preview.mp4')
                dst_path = os.path.join(predictions_folder, f"preview_{parameter_folder_name}.mp4")
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                if os.path.exists(src_tmp):
                    try:
                        os.replace(src_tmp, dst_path)
                    except Exception:
                        pass
    except Exception:
        pass

    main_logger.info("Saved predictions to %s", prediction_file)

# ==============================================
# Run all parameter combinations (one experiment each)
# ==============================================
if __name__ == "__main__":
    start_time = time.time()
    for parameters in build_param_combinations():
        run_experiment(parameters)
    main_logger.info("Total time taken: %.2f seconds", time.time() - start_time)
