import numpy as np

from projectaria_tools.core.sophus import SE3
     
import logging

# Function to transform points using SE3
def transform_point(se3: SE3, point: np.ndarray) -> np.ndarray:
    # Ensure point is in the correct shape [3, n] for transformation
    if point.shape == (1, 3):
        point = point.T                         # Me: Transpose to [3, 1]
    elif point.ndim == 1:                       
        point = point.reshape(3, 1)             # Me: Reshape to [3, 1]
    transformed_point = se3 @ point             # Me; Apply transformation
    return transformed_point.T                  # Me: Transpose back to [1, 3] if necessary

def user_movement_calculation(user_position_before, user_position_current):
    """
    args: Previous position of the user (x,y components) and current position
    return: the abs distance from one position to the other
    """
    # convert to plane xz
    user_position_before_xz = np.array([user_position_before[0], 0, user_position_before[2]])
    user_position_current_xz = np.array([user_position_current[0], 0, user_position_current[2]])
   
    # movement vector and distance (euclideian norm)
    user_movement_xz_vector = user_position_before_xz - user_position_current_xz
    user_movement_xz_distance = np.linalg.norm(user_movement_xz_vector) # Me: take the x,z components only
    
    return user_movement_xz_distance

def write_to_file(file, content):
     with open(file, 'w') as file:
            # Write the content of the list
            file.write("Content of objects_within_radius:\n")
            for index, sublist in enumerate(content, start=1):
                file.write(f"{index}: {', '.join(sublist)}\n")
                
def calculate_overlap(list_1, list_2):
            set1, set2 = set(list_1), set(list_2)
            overlap = set1.intersection(set2)
            percentage = (len(overlap) / max(len(set1), len(set2))) * 100            
            if percentage < 60:
                print ('stop')
            return percentage
        
def object_within_radius (object_distances, objects_names, radius = 1.5):
    obj_names = []
    for name, distance in zip(objects_names, object_distances):
        if distance < radius: 
            obj_names.append(name)
    return obj_names

def calculate_centroid(objects):
    """Calculate the centroid of a group of objects based on their coordinates."""
    return np.mean(objects, axis=0)

def visibility_mask(obj_positions_cam_reshaped, rgb_camera_calibration):
    valid_mask = []
    # Project to 2D pixel coordinates and filter visible objects!     
    for pos in obj_positions_cam_reshaped:
        pixel_coord = rgb_camera_calibration.project(pos)
        if pixel_coord is not None:
            # projected_points.append(pixel_coord.squeeze())
            valid_mask.append(True)
        else:
            valid_mask.append(False)
    # projected_points = np.array(projected_points)
    valid_mask = np.array(valid_mask)
    return valid_mask

def is_object_in_group(objects, new_object, radius=1.5):
    """
    Determine if the new object is within a specified radius from the group's centroid.
    
    Args:
    - objects: A list of numpy arrays representing the coordinates of the objects.
    - new_object: A numpy array representing the coordinates of the new object.
    - radius: The distance threshold to determine group membership (default is 1.5 meters).
    
    Returns:
    - True if the new object is within the radius of the group's centroid, False otherwise.
    """
    centroid = calculate_centroid(objects)
    distance = np.linalg.norm(centroid[:2], new_object[:2]) # calculate the distance in the plane xy
    return distance <= radius

def unique_in_order(sequence):
    seen = set()
    unique_list = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def refine_dict_keep_unique(original_dict, result_dict, last_value=None):
        # Iterate over the items in the original dictionary
        for key, value in original_dict.items():
            # If the value changes, add it to the result dictionary
            if value != last_value:
                result_dict[key] = value
                last_value = value   
        return result_dict
        
def normalize_measuremnt(measurement):
    norm = np.linalg.norm(measurement)
    if norm == 0:
        return measurement
    return measurement / norm

def exponential_filter(new_value, prev_value, alpha=0.9):
    return alpha * new_value + (1 - alpha) * prev_value

# Function to check for values greater than 6 and store them
def get_objects_with_high_duration(data, threshold=1):
    high_value_objects = {}
    for key, values in data.items():
        if values[-1][1] > threshold:
            high_value_objects[key] = f"{float(values[-1][1]):.3f}"
    return high_value_objects

def generate_indexes(gt_list, names):
    """This function assigns names to indexes. Takes as input a list with names and respective indexes"""
    indexes_gt = []
    for item in gt_list:
        index = np.where(names == item)[0][0]
        indexes_gt.append(index)
    return indexes_gt

## Check the center of the bounding box 
def check_point_in_3D_bbox_of_object_position(bboxes3d):
    for key, bbox_3d in bboxes3d.items():
        x_min, y_min, z_min, x_max, y_max, z_max = bbox_3d.aabb

        # Transformation matrix from object to scene
        transform_matrix = bbox_3d.transform_scene_object.to_matrix()
        translation = bbox_3d.transform_scene_object.translation()[0]
        
        # Compute the center of the bounding box
        center_of_bbox_local =[np.array([
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            (z_min + z_max) / 2
        ])]    
        
        center_of_bottom_face_local = np.array([
            (x_min + x_max) / 2,  # Midpoint in the x direction
            (y_min + y_max) / 2,  # Midpoint in the y direction
            z_min                 # Bottom face in the z direction
        ])
        
        # Conver the center point to scene frame
        center_of_bbox_local_homo = np.append(center_of_bbox_local, [1])
        center_of_bbox_scene = (transform_matrix @ center_of_bbox_local_homo)[0:3]  # [x, y, z]
        
        # Convert the bottom face center point to scene frame
        center_of_bottom_face_local_homo = np.append(center_of_bottom_face_local, [1])
        center_of_bottom_face_scene = (transform_matrix @ center_of_bottom_face_local_homo)[0:3]      

        # Compare the translation with the center of the bounding box in global coordinates
        if np.allclose(translation, center_of_bbox_scene):
            print("Translation corresponds to the center of the bounding box in global coordinates")
        else:
            print("Translation does not correspond to the center of the bounding box")

        if np.allclose(translation, center_of_bottom_face_scene):
            print("Translation corresponds to the center of the bottom face in global coordinates")
        else:
            print("Translation does not correspond to the center of the bottom face")
            
## Movement 
def calculate_relative_pose_difference(T_scene_object_before, T_scene_object):
        
        # convert to list
        T_scene_object = list(T_scene_object.values())
        
        ## initialize the poses before in case I don't know 
        if T_scene_object_before is None:
            T_scene_object_before = [None] * len(T_scene_object)  
     
        ## Initialization of lists of the objects                                                         
        T_scene_object_ema = [None] * len(T_scene_object)                                                                         # Me: It's important to specify dtype=object so that np.array be capable of holding any object
        relative_T_object = [None] * len(T_scene_object)                                                                          # Me: I would use np.array to exploit vectorization and avoid for loops but I need to use the inverse method of the objects 
        norm_relative_T_object = [None] * len(T_scene_object)                                                                     # Me: Initialization of calculating the relative pose
            
        ## Calculate the relative transformation between poses for the same objects betweeen two consecutive timesteps 
        for i in range(len(T_scene_object)):
            
            # Initialize previous pose for each object that we work
            if T_scene_object_before[i] is None:
                
                ## Initialise the previous 
                T_scene_object_before[i] = T_scene_object[i] 
                
                ## Calculations 
                vector_pose_6d = T_scene_object[i].log()[0]                                                                        # Me: Calculate the 6D vector (log representation) of the current and previous SE3 transformations
                vector_pose_6d_before = T_scene_object_before[i].log()[0]
                
                ## Ema 
                vector_pose_6d = exponential_filter(vector_pose_6d, vector_pose_6d_before, alpha=0.7)                              # Apply exponential moving average (EMA) to reduce noise
                translational_vector, rotation_vector = vector_pose_6d[:3], vector_pose_6d[3:]                                     # Decompose the EMA result into translational and rotational components
                T_scene_object_ema[i] = SE3.exp(translational_vector, rotation_vector)                                             # Reconstruct the SE3 object from the smoothed 6D vector
                
                ## Relative
                relative_T_object_matrix = T_scene_object_before[i].inverse().to_matrix() @ T_scene_object_ema[i].to_matrix()      # Compute the relative pose transformation matrix and convert back to SE3
                relative_T_object[i] = SE3.from_matrix(relative_T_object_matrix)
                norm_relative_T_object[i] = np.linalg.norm(relative_T_object[i].log())                                             # Optionally, calculate the norm of the relative pose transformation's logarithm
        
            else:
                
                ## Calculations 
                vector_pose_6d = T_scene_object[i].log()[0]                                                                        # Me: Calculate the 6D vector (log representation) of the current and previous SE3 transformations
                vector_pose_6d_before = T_scene_object_before[i].log()[0]
                
                ## Ema 
                vector_pose_6d = exponential_filter(vector_pose_6d, vector_pose_6d_before, alpha=0.7)                              # Apply exponential moving average (EMA) to reduce noise
                translational_vector, rotation_vector = vector_pose_6d[:3], vector_pose_6d[3:]                                     # Decompose the EMA result into translational and rotational components
                T_scene_object_ema[i] = SE3.exp(translational_vector, rotation_vector)                                             # Reconstruct the SE3 object from the smoothed 6D vector
                
                ## Relative
                relative_T_object_matrix = T_scene_object_before[i].inverse().to_matrix() @ T_scene_object_ema[i].to_matrix()      # Compute the relative pose transformation matrix and convert back to SE3
                relative_T_object[i] = SE3.from_matrix(relative_T_object_matrix)
                norm_relative_T_object[i] = np.linalg.norm(relative_T_object[i].log())                                             # Optionally, calculate the norm of the relative pose transformation's logarithm                                         
                    
        # Update of the poses in the past 
        T_scene_object_before = T_scene_object_ema.copy()   
        
        # Tranform to np array
        norm_relative_T_object = np.array(norm_relative_T_object)
        return np.round(norm_relative_T_object, 8), T_scene_object_before
    
## Distance 
def detect_user_interaction_proximity():
    proximity_sensor_data = is_user_consistently_near_object()
    return handle_distance_data(proximity_sensor_data)

def is_user_consistently_near_object(user_position, object_position, threshold_distance=0.05, consistency_duration=5):
    distances = []
    for timestamp in range(consistency_duration):
        distance = np.linalg.norm(user_position[timestamp], object_position[timestamp])
        distances.append(distance)

    return all(d < threshold_distance for d in distances)     # Check if all distances are below the threshold

def handle_distance_data(proximity_sensor_data):
    for data in proximity_sensor_data:
        if is_user_near(data):
            return data['object_name']
    return None

def is_user_near(data, interaction_distance = 0.6):
    return data['distance'] < interaction_distance

def is_interacting(hand, obj, interaction_threshold = 0.05 ):
    distance = np.linalg.norm(hand['coordinates'], obj['coordinates'])
    return distance < interaction_threshold

"""
Detection function if use vision or sensors in hands 
"""
# ## Vision 
# def detect_user_interaction_vision():
#     # Vision-based interaction detection
#     return None

# ## Sensors 
# def detect_user_interaction_sensors(sensor_data):
#     # Sensor-based interaction detection / if the user wears gloves can detect interaction events 
#     for data in sensor_data:
#         if is_touch_event(data):
#             return data['object_name']
#     return None

# def is_touch_event(data, threshold = 2):
#     return data['pressure'] > threshold

# def detect_user_hand_pose():
#     detected_objects = get_detected_objects()  # List of detected objects with their coordinates
#     user_hands = track_user_hands()  # Coordinates of the user's hands

#     for obj in detected_objects:
#         for hand in user_hands:
#             if is_interacting(hand, obj):
#                 return obj['name']
#     return None

"""
Detect user interaction 
"""
# def detect_user_interaction():
#     """ Placeholder for actual interaction detection logic
#         This could be event-driven or polled based on application design
#     """
#     # Integrate different detection methods here
#     proximity_interaction = detect_user_interaction_proximity()
#     movable_objects_interaction = detect_user_interaction_object_motion()
#     # vision_interaction = detect_user_interaction_vision()
#     # sensor_interaction = detect_user_interaction_sensors()
    
#     # Prioritize interactions if multiple detections occur
#     if proximity_interaction:
#         return proximity_interaction
#     elif movable_objects_interaction:
#         return movable_objects_interaction
#     # if vision_interaction:
#     #     return vision_interaction
#     # elif sensor_interaction:
#     #     return sensor_interaction
#     return None

