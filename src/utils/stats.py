import numpy as np

from collections import deque
from typing import List, Dict, Tuple

 
class Statistics:
    def __init__(self, window_time, high_dot_threshold, distance_threshold, low_distance_threshold, time_threshold):
        
        self.window_time = window_time
        self.high_dot_threshold = high_dot_threshold
        self.distance_threshold = distance_threshold
        self.low_distance_threshold = low_distance_threshold
        self.time_threshold = time_threshold
        
        self.past_dots = {}                    
        self.past_distances = {}               
        self.visibility_counter = {}           
        self.visibility_duration = {}
        self.high_dot_counts = {}
        self.close_distance_counts = {}
        self.very_close_distance_counts = {}
        self.time_to_approach_counts = {}
        self.avg_dots = {}
        self.avg_distances = {}
        self.object_time_interaction = {} 
        self.avg_dots_list = []                 # for debugging
        self.avg_distances_list = []            # for debugging
        
    
    @property
    def parameters(self):
        """Dynamic parameters list containing internal state mappings."""
        return [
            (self.past_dots, 'dots'),
            (self.past_distances, 'distances'),
            (self.visibility_counter, 'counter'),
            (self.visibility_duration, 'duration'),
            (self.high_dot_counts, 'dot_counts'),
            (self.close_distance_counts, 'low_distance_counts'),
            (self.very_close_distance_counts, 'very_low_distance_counts'),
            (self.time_to_approach_counts, 'time'),
        ]
        
    def calculate_counter(self, obj_id: int, current_time_s: int) -> int:
        """Calculate the counter for an object."""
        counters = self.visibility_counter[obj_id][-1][1] + 1
        return counters
    
    def calculate_duration(self, obj_id: int, time_difference: int, current_time_s: int) -> float:
        """Calculate the visibility duration for an object."""
        if len(self.visibility_counter[obj_id]) > 1:
            earliest_count = self.visibility_counter[obj_id][0][1]
            latest_count = self.visibility_counter[obj_id][-1][1]
            visible_duration = round(((latest_count - earliest_count) * time_difference), 3)
        else:
            visible_duration = 0
        return visible_duration
    
    def calculate_dot_distance_time_counters(self, obj_id: int, streaks: Dict[int, deque], typedict: str) -> int:
        """Calculate the count of high dot or low distance streaks."""
        streak_count = len(streaks[obj_id])
        return streak_count
    
    def remove_outdated_entries(self, data_dict, current_time_s, type_dict, VARIABLES_WINDOW_TIME):
        """Remove outdated entries beyond the VARIABLES_WINDOW_TIME."""
    
        # print('The dictionary inside the removal of outdated entries is', type_dict)
        cutoff_time_s = current_time_s - VARIABLES_WINDOW_TIME
        
        # Delete the keys 
        keys_to_delete = []
        for key, dq in data_dict.items():
            data_dict[key] = deque([(t, v) for t, v in dq if t >= cutoff_time_s]) # move the window on the write 
            if not data_dict[key]:
                keys_to_delete.append(key)
        
        # Remove keys with empty deques
        for key in keys_to_delete:
            del data_dict[key]
    
    def interaction_time_user_object(self, user_velocity_device, user_position_scene, object_position_scene, T_Scene_Device):
        
        """
        Input: 
        1. User Velocity on scene coordinate frame
        2. User Position on Scene coordinate frame (EMA)
        3. Object Position on Scene coordinate frame
        4. Transformaton from Device to Scene coordinate frame
        
        Calculate: 
        1. Velocity from device to Scene frame
        2. Displacement Vector
        3. Projection Velocity to Displacement vector
            a. Unit displaceemnt vector --> orientation from user to object
            b. Velocity Vector component to the direction of the unit displacement vector --> dot product
        4. Time --> norm(dispacement vector) / norm(projected velocity vector) 
        """      
        # ==============================================
        # 3D MOTION 
        # ==============================================  

        # VELOCITY 
        velocity_xyz = T_Scene_Device.rotation().to_matrix() @ user_velocity_device
        
        # DISTANCE
        displacement_vector_xyz = object_position_scene - user_position_scene
        distance_xyz = np.linalg.norm(displacement_vector_xyz)
        # guard: user already at the object -> no meaningful direction
        if distance_xyz <= 1e-9:
            return float('inf'), float('inf')
        displacement_unit_vector_xyz = displacement_vector_xyz / distance_xyz

        # PROJECTED VELOCITY
        projected_velocity_xyz = np.dot(velocity_xyz, displacement_unit_vector_xyz) * displacement_unit_vector_xyz
        speed_xyz = np.linalg.norm(projected_velocity_xyz)
        # guard: near-stationary user -> effectively unreachable (was inf/NaN)
        time_xyz = distance_xyz / speed_xyz if speed_xyz > 1e-6 else float('inf')
        
        # ==============================================
        # PLANAR MOTION ON XZ PLANE
        # ==============================================

        # DISTANCE
        displacement_vector_xz = np.array([displacement_vector_xyz[0], 0, displacement_vector_xyz[2]])
        displacement_unit_vector_xz = displacement_vector_xz / np.linalg.norm(displacement_vector_xz)
        distance_xz = np.linalg.norm(displacement_vector_xz)
        
        # VELOCITY 
        velocity_xz = np.array([velocity_xyz[0], 0, velocity_xyz[2]])
    
        # PROJECTED VELOCITY
        projected_velocity_xz= np.dot(velocity_xz, displacement_unit_vector_xz) * displacement_unit_vector_xz
        speed_xz= np.linalg.norm(projected_velocity_xz)
        time_xz = distance_xz / speed_xz if speed_xz > 1e-6 else float('inf')
    
        return time_xyz, time_xz

    def time_window(
        self,
        current_time_s: int,
        time_difference: int,
        users_position: np.ndarray,
        users_velocity: np.ndarray, 
        object_ids: List[int],
        object_positions: List[np.ndarray],  # List of numpy arrays
        dot_products: List[float],
        distances: List[float], 
        T_Scene_Device, 
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, deque], Dict[int, deque], Dict[int, deque], Dict[int, deque], Dict[int, deque], Dict[int, deque]]:
        
        """initialiase every time a function is called to have only the objects that are visible"""
        self.avg_dots = {}
        self.avg_distances = {}
        self.avg_dots_list = []
        self.avg_distances_list = []
        
        """Update statistics based on the current time window."""
        for parameter, name in self.parameters:
            self.remove_outdated_entries(parameter, current_time_s, name, self.window_time)
        
        for obj_id, dot_product, distance, obj_position in zip(object_ids, dot_products, distances, object_positions):
            
            # Time to approach an object 
            user_object_time_xyz, user_object_time_xz = self.interaction_time_user_object(users_velocity, users_position, obj_position[0], T_Scene_Device)
            
            # if obj_id == 4404207983027294 and current_time_s > 4.5:
            #     print(f"[INSIDE CLASS] - Time to approach the Chopping Board is {user_object_time_xz}")
            
            # Dictionary having the time of with the user
            self.object_time_interaction[obj_id] = user_object_time_xz
            
            # Deque for dots
            if obj_id in self.past_dots:
                self.past_dots[obj_id].append((current_time_s, dot_product))
            else:
                self.past_dots[obj_id] = deque([(current_time_s, dot_product)])
            
            # Deque for distances (user / objects)
            if obj_id in self.past_distances:
                self.past_distances[obj_id].append((current_time_s, distance))
            else:
                self.past_distances[obj_id] = deque([(current_time_s, distance)])
            
            # Counter for Visibility
            if obj_id in self.visibility_counter:
                self.visibility_counter[obj_id].append((current_time_s, self.calculate_counter(obj_id, current_time_s)))
            else:
                self.visibility_counter[obj_id] = deque([(current_time_s, 0)])
            
            # Duration visibility
            if obj_id in self.visibility_duration:
                self.visibility_duration[obj_id].append((current_time_s, self.calculate_duration(obj_id, time_difference, current_time_s)))
            else:
                self.visibility_duration[obj_id] = deque([(current_time_s, 0)])
            
            # Counter for DOT
            if obj_id not in self.high_dot_counts and dot_product > self.high_dot_threshold:                  # Me: 0.9 
                self.high_dot_counts[obj_id] = deque([(current_time_s, 0)])
            elif dot_product > self.high_dot_threshold:
                self.high_dot_counts[obj_id].append((current_time_s, self.calculate_dot_distance_time_counters(obj_id, self.high_dot_counts, typedict='dot'))) 
            
            # Counter for DISTANCE
            if obj_id not in self.close_distance_counts and distance < self.distance_threshold:           # Me: 2 meters
                self.close_distance_counts[obj_id] = deque([(current_time_s, 0)])
            elif distance < self.distance_threshold:
                self.close_distance_counts[obj_id].append((current_time_s, self.calculate_dot_distance_time_counters(obj_id, self.close_distance_counts, typedict='low_distance')))  

            # Counter for LOW DISTANCE
            if obj_id not in self.very_close_distance_counts and distance < self.low_distance_threshold: # Me: 0.7 meters
                self.very_close_distance_counts[obj_id] = deque([(current_time_s, 0)])
            elif distance < self.low_distance_threshold:
                self.very_close_distance_counts[obj_id].append((current_time_s, self.calculate_dot_distance_time_counters(obj_id, self.very_close_distance_counts, typedict='very_low_distance')))  

            # Counter for TIME
            if obj_id not in self.time_to_approach_counts and user_object_time_xz < self.time_threshold:         # Me: 2 seconds
                self.time_to_approach_counts[obj_id] = deque([(current_time_s, 0)])
            elif user_object_time_xz < self.time_threshold:
                self.time_to_approach_counts[obj_id].append((current_time_s, self.calculate_dot_distance_time_counters(obj_id, self.time_to_approach_counts, typedict='time')))  

        # calculate the average Dots and the average distances 
        for obj_id, dots in self.past_dots.items():
            if len(dots) > 0:
                total_dots = sum(dot for _, dot in dots)
                count_dots = len(dots)
                self.avg_dots[obj_id] = total_dots / count_dots
                self.avg_dots_list.append(self.avg_dots[obj_id])
                
        for obj_id, dists in self.past_distances.items():
            if len(dists) > 0:
                total_distance = sum(dist for _, dist in dists)
                count_distance = len(dists)
                self.avg_distances[obj_id] = total_distance / count_distance
                self.avg_distances_list.append(self.avg_distances[obj_id])
        
        # Return
        return (self.past_dots, self.past_distances,
                self.avg_dots, self.avg_distances, self.avg_dots_list, self.avg_distances_list, 
                self.visibility_counter, self.visibility_duration, 
                self.high_dot_counts, self.close_distance_counts, self.very_close_distance_counts, 
                self.object_time_interaction, self.time_to_approach_counts,
                )
    
    def get_count_at_timestamp(self, obj_id: int, target_timestamp: float) -> int:
        
        """Retrieve the count for the object at a specific timestamp."""
        thedeque = self.close_distance_counts.get(obj_id, deque())
        
        for timestamp, count in thedeque:
            if timestamp == target_timestamp:
                return count
        return None  
    


    # def time_window(self,
    #     current_time_s: int,
    #     time_difference: int,
    #     users_position: np.ndarray,
    #     users_velocity: np.ndarray, 
    #     object_ids: List[int],
    #     object_positions: List[np.ndarray],  # List of numpy arrays
    #     dot_products: List[float],
    #     distances: List[float], 
    #     T_Scene_Device, 
    # ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, deque], Dict[int, deque], Dict[int, deque], Dict[int, deque], Dict[int, deque], Dict[int, deque]]:
        
    #     """initialiase every time a function is called to have only the objects that are visible"""
    #     self.avg_dots = {}
    #     self.avg_distances = {}
    #     self.avg_dots_list = []
    #     self.avg_distances_list = []
        
    #     """Update statistics based on the current time window."""
    #     for parameter, name in self.parameters:
    #         self.remove_outdated_entries(parameter, current_time_s, name, self.window_time)
        
    #     for obj_id, dot_product, distance, obj_position in zip(object_ids, dot_products, distances, object_positions):
            
    #         # Time to approach an object 
    #         user_object_time_xyz, user_object_time_xz = self.interaction_time_user_object(users_velocity, users_position, obj_position[0], T_Scene_Device)
            
    #         # Dictionary having the time of with the user
    #         self.object_time_interaction[obj_id] = user_object_time_xz
            
    #         # Deque for dots
    #         if obj_id in self.past_dots:
    #             self.past_dots[obj_id].append((current_time_s, dot_product))
    #         else:
    #             self.past_dots[obj_id] = deque([(current_time_s, dot_product)])
            
    #         # Deque for distances (user / objects)
    #         if obj_id in self.past_distances:
    #             self.past_distances[obj_id].append((current_time_s, distance))
    #         else:
    #             self.past_distances[obj_id] = deque([(current_time_s, distance)])
            
    #         # Counter for Visibility
    #         if obj_id in self.visibility_counter:
    #             self.visibility_counter[obj_id].append((current_time_s, self.calculate_counter(obj_id, current_time_s)))
    #         else:
    #             self.visibility_counter[obj_id] = deque([(current_time_s, 0)])
            
    #         # Counter for DOT
    #         if obj_id not in self.high_dot_counts and dot_product > self.high_dot_threshold:                  # Me: 0.9 
    #             self.high_dot_counts[obj_id] = deque([(current_time_s, 0)])
    #         elif dot_product > self.high_dot_threshold:
    #             self.high_dot_counts[obj_id].append((current_time_s, self.calculate_dot_distance_time_counters(obj_id, self.high_dot_counts, typedict='dot'))) 
            
    #         # Counter for DISTANCE
    #         if obj_id not in self.close_distance_counts and distance < self.distance_threshold:           # Me: 2 meters
    #             self.close_distance_counts[obj_id] = deque([(current_time_s, 0)])
    #         elif distance < self.distance_threshold:
    #             self.close_distance_counts[obj_id].append((current_time_s, self.calculate_dot_distance_time_counters(obj_id, self.close_distance_counts, typedict='low_distance')))  

    #     # calculate the average Dots and the average distances 
    #     for obj_id, dots in self.past_dots.items():
    #         if len(dots) > 0:
    #             total_dots = sum(dot for _, dot in dots)
    #             count_dots = len(dots)
    #             self.avg_dots[obj_id] = total_dots / count_dots
    #             self.avg_dots_list.append(self.avg_dots[obj_id])
                
    #     for obj_id, dists in self.past_distances.items():
    #         if len(dists) > 0:
    #             total_distance = sum(dist for _, dist in dists)
    #             count_distance = len(dists)
    #             self.avg_distances[obj_id] = total_distance / count_distance
    #             self.avg_distances_list.append(self.avg_distances[obj_id])
        
    #     # Return
    #     return (self.past_dots, self.past_distances,
    #             self.avg_dots, self.avg_distances, self.avg_dots_list, self.avg_distances_list, 
    #             self.visibility_counter, self.visibility_duration, 
    #             self.high_dot_counts, self.close_distance_counts, self.very_close_distance_counts, 
    #             self.object_time_interaction, self.time_to_approach_counts,
    #             )