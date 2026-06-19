
"""
Use ML model to train on the interaction history
"""
# # ## Functions to use ML methods to improve efficiency of the models ##
# def update_filters(interaction_history, context):
#     # Extract features and labels from the interaction history
#     interaction_data = []
#     for log in interaction_history:
#         interaction_data.append({
#             'features': extract_features(log, context),
#             'label': log['interaction_type']
#         })
#     # Train the model with the updated data
#     model = train_interaction_model(interaction_data)
#     return model

"""
collect data from interaction history to find tune a model 
"""
# def train_interaction_model(interaction_data):
#     X = []  # Features
#     y = []  # Labels (interaction or not)
#     for data in interaction_data:
#         X.append(data['features'])
#         y.append(data['label'])
#     model = RandomForestClassifier()
#     model.fit(X, y)
#     return model

# def predict_interaction(model, current_features):
#     return model.predict([current_features])

# def extract_features(log, context):
#     # Extract relevant features from the log and context
#     features = [
#         log['timestamp'],
#         context.get('current_room'),
#         # Add other relevant features
#     ]
#     return features

# ## Function to calculate cumulative dot products using the deque ## 

"""
Find the cumulative dot product and calulate the average 
"""
# def update_and_calculate_cumulative_dot_products(current_time_ns, visible_obj_ids, dot_products, past_dot_products):
#     # Update the deque for each object
#     for obj_id, dot_product in zip(visible_obj_ids, dot_products):
#         if obj_id in past_dot_products:
#             past_dot_products[obj_id].append((current_time_ns, dot_product))
#         else:
#             past_dot_products[obj_id] = deque([(current_time_ns, dot_product)])
    
#     # Remove entries older than 3 seconds
#     for obj_id in past_dot_products.keys():
#         while past_dot_products[obj_id] and (current_time_ns - past_dot_products[obj_id][0][0]) > 3_000_000_000:
#             past_dot_products[obj_id].popleft()
    
#     # Calculate the cumulative dot products over the last 3 seconds
#     cumulative_dot_products = {
#         obj_id: sum(dp for ts, dp in past_dot_products[obj_id])
#         for obj_id in visible_obj_ids
#     }
    
#     return cumulative_dot_products

"""
Time window variable
"""
# def time_window_variables(
#     aria_pose_start_timestamp: int,
#     current_time_ns: int,
#     time_difference: int,
#     visible_objects: List[int],
#     dot_products: List[float],
#     distances: List[float],
#     past_dots: Dict[int, deque],
#     past_distances: Dict[int, deque],
#     visibility_counter: Dict[int, deque],
#     visibility_duration: Dict[int, deque],
#     high_dot_streaks: Dict[int, int],
#     close_distance_streaks: Dict[int, int],
#     WINDOW_SIZE_NS: int
# ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, deque], Dict[int, deque], Dict[int, int], Dict[int, int]]:

#     # Calculate cutoff time
#     cutoff_time_ns = current_time_ns - WINDOW_SIZE_NS

#     # Update entries for each visible object
#     for obj_id, dot_product, distance in zip(visible_objects, dot_products, distances):
#         if obj_id in past_dots:
#             past_dots[obj_id].append((current_time_ns, dot_product))
#         else:
#             past_dots[obj_id] = deque([(current_time_ns, dot_product)])
#         # if obj_id not in past_dots:
#         #     past_dots[obj_id] = deque()
#         # past_dots[obj_id].append((current_time_ns, dot_product))
        
#         if obj_id not in past_distances:
#             past_distances[obj_id] = deque()
#         past_distances[obj_id].append((current_time_ns, distance))

#         if obj_id not in visibility_counter:
#             visibility_counter[obj_id] = deque()
#         if visibility_counter[obj_id]:
#             visibility_counter[obj_id].append((current_time_ns, visibility_counter[obj_id][-1][1] + 1))
#         else:
#             visibility_counter[obj_id].append((current_time_ns, 1))

#         if obj_id not in visibility_duration:
#             visibility_duration[obj_id] = deque()
#         if visibility_duration[obj_id]:
#             visibility_duration[obj_id].append((current_time_ns, visibility_duration[obj_id][-1][1] + time_difference))
#         else:
#             visibility_duration[obj_id].append((current_time_ns, time_difference))

#         # past_dots.setdefault(obj_id, deque()).append((current_time_ns, dot_product))
#         # past_distances.setdefault(obj_id, deque()).append((current_time_ns, distance))
#         # visibility_counter.setdefault(obj_id, deque()).append((current_time_ns, visibility_counter[obj_id][-1][1] + 1 if visibility_counter[obj_id] else 1))
#         # visibility_duration.setdefault(obj_id, deque()).append((current_time_ns, visibility_duration[obj_id][-1][1] + time_difference if visibility_duration[obj_id] else time_difference))
        
#         # Update dot streaks
#         high_dot_streaks[obj_id] = high_dot_streaks.get(obj_id, 0) + 1 if dot_product > 0.7 else 0

#         # Update distance streaks
#         close_distance_streaks[obj_id] = close_distance_streaks.get(obj_id, 0) + 1 if distance < 1.6 else 0

#         # Update dot streaks
#         high_dot_streaks[obj_id] = high_dot_streaks.get(obj_id, 0) + 1 if dot_product > 0.7 else 0

#         # Update distance streaks
#         close_distance_streaks[obj_id] = close_distance_streaks.get(obj_id, 0) + 1 if distance < 1.6 else 0

#     # Filter out old entries
#     for obj_id in list(past_dots.keys()):
#         past_dots[obj_id] = deque([(ts, dp) for ts, dp in past_dots[obj_id] if ts >= cutoff_time_ns], maxlen=len(past_dots[obj_id]))
#         past_distances[obj_id] = deque([(ts, dist) for ts, dist in past_distances[obj_id] if ts >= cutoff_time_ns], maxlen=len(past_distances[obj_id]))
#         visibility_counter[obj_id] = deque([(ts, count) for ts, count in visibility_counter[obj_id] if ts >= cutoff_time_ns], maxlen=len(visibility_counter[obj_id]))
#         visibility_duration[obj_id] = deque([(ts, duration) for ts, duration in visibility_duration[obj_id] if ts >= cutoff_time_ns], maxlen=len(visibility_duration[obj_id]))

#         if not past_dots[obj_id]:
#             del past_dots[obj_id]
#             del past_distances[obj_id]
#             del visibility_counter[obj_id]
#             del visibility_duration[obj_id]
#             del high_dot_streaks[obj_id]
#             del close_distance_streaks[obj_id]

#     avg_dots = {obj_id: sum(dp for ts, dp in dot_list) / len(dot_list) for obj_id, dot_list in past_dots.items() if len(dot_list) > 0}
#     avg_distances = {obj_id: sum(dp for ts, dp in distances) / len(distances) for obj_id, distances in past_distances.items() if len(distances) > 0}

#     return avg_dots, avg_distances, past_dots, past_distances, visibility_counter, visibility_duration, high_dot_streaks, close_distance_streaks


"""
Remove objects 
"""
# # Remove objects that haven't changed their value for the first 3 seconds
# for obj_id in list(initial_values.keys()):
#     if (current_time_ns - past_dots[obj_id][0][0]) > WINDOW_SIZE_NS:
#         if initial_values[obj_id] == (dot_products[visible_objects.index(obj_id)], distances[visible_objects.index(obj_id)]):
#             del past_dots[obj_id]
#             del visibility_counter[obj_id]
#             del visibility_duration[obj_id]
#             del high_dot_streaks[obj_id]
#             del close_distance_streaks[obj_id]
#             del initial_values[obj_id]

# return accumulated_dots, visibility_counter, visibility_duration, high_dot_streaks, close_distance_streaks\

# # Check high dot streaks

# high_dot_streaks[obj_id] = high_dot_streaks.get(obj_id, 0) + (1 if dot_product > 0.9 else 0)

# # Check close distance streaks
# close_distance_streaks[obj_id] = close_distance_streaks.get(obj_id, 0) + (1 if distance < 1.0 else 0)

# avg_dots = {obj_id: sum(dot for _, dot in dots) / len(dots) for obj_id, dots in past_dots.items() if len(dots) > 0 }
# # avg_distances = {obj_id: sum(dist for _, dist in dists) / len(dists) for obj_id, dists in past_distances.items() if len(dists) > 0}

"""
Add a temporal consistency to filer objects 
"""
# # Add temporal consistency multiplier
# seen_objects = {}
# for obj_id in filtered_obj_ids:
#     if obj_id in seen_objects:
#         seen_objects[obj_id] += 1
#     else:
#         seen_objects[obj_id] = 1

# # Apply temporal consistency multiplier
# temporal_consistency_multiplier = np.array([seen_objects[obj_id] for obj_id in filtered_obj_ids])
# temporal_consistency_mask = temporal_consistency_multiplier > 0  # Debug: Set lower threshold
# if temporal_consistency:
#     # Apply final mask for temporal consistency
#     final_filtered_obj_ids = filtered_obj_ids[temporal_consistency_mask]
#     final_filtered_obj_positions_scene = filtered_obj_positions_scene[temporal_consistency_mask]
#     final_filtered_obj_positions_cam = filtered_obj_positions_cam[temporal_consistency_mask]
#     final_filtered_dot_products = filtered_dot_products[temporal_consistency_mask]
#     final_filtered_line_distances = filtered_line_distances[temporal_consistency_mask]
#     print(f"Final filtered objects at timestamp {timestamp_ns}: {final_filtered_obj_ids}")

# # less_time_mask = np.array(time_interaction_list) < 1                                                       

# # Filter based on the combined mask
# filtered_obj_ids = filtered_obj_ids[less_time_mask]
# filtered_obj_positions_scene = filtered_obj_positions_scene[less_time_mask]
# filtered_obj_positions_cam = filtered_obj_positions_cam[less_time_mask]
# filtered_dot_products = filtered_dot_products[less_time_mask]
# filtered_line_distances = filtered_line_distances[less_time_mask]
        

"""
Find object with consistent high values 
"""
# # Update streaks for high dot and close distance
# for obj_id in filtered_obj_ids:
#     if obj_id in filtered_counter:
#         if filtered_dot_products[filtered_obj_ids == obj_id][0] > 0.7:                                          # Me: High dot value
#             high_dot_streaks[obj_id] += 1
#         else:
#             high_dot_streaks[obj_id] = 0

#         if filtered_line_distances[filtered_obj_ids == obj_id][0] < 0.6:                                        # Me: Close distance
#             close_distance_streaks[obj_id] += 1
#         else:
#             close_distance_streaks[obj_id] = 0

#  high_dot_objects = [gt_provider.get_instance_info_by_id(obj_id).name 
#                                 for obj_id, dot in zip(filtered_obj_ids, filtered_dot_products) 
#                                 if dot > 0.9]
#             close_objects = [gt_provider.get_instance_info_by_id(obj_id).name 
#                              for obj_id, dist in zip(filtered_obj_ids, filtered_line_distances) 
#                              if dist < 1]

# high_objects_duration_list = []
    # for obj_id, duration in high_objects_duration_dict.items():
    #     high_objects_duration_list.append(gt_provider.get_instance_info_by_id(obj_id).name)

"""
Calculate interaction time 
"""
#  ## Calculate time to interaction with the filtered objects (TTI)
#     user_position = aria_3d_pose_with_dt.data().transform_scene_device.translation()
#     user_velocity = aria_3d_pose_with_dt.data().device_linear_velocity                                                              # Me: normalized_velocity = normalize_velocity(user_velocity)                                                       # Me: Normalize the current velocity
#     if prev_velocity is None:                                                                                                       # Me: Initialize previous velocity for the first loop
#         user_ema_velocity = prev_velocity = user_velocity   
#     else:
#         user_ema_velocity = exponential_filter(user_velocity, prev_velocity)                                                        # Me: Apply exponential filter 
#     prev_velocity = user_ema_velocity                                                                                               # Me: Update the previous velocity

#     time_interaction_list = [time_to_interaction(user_position, 
#                                                  user_velocity, 
#                                                  obj_position, 
#                                                  obj_id
#                                                 ) 
#                              for obj_id, obj_position in zip(filtered_obj_ids, filtered_obj_positions_scene)]

# for object_name, line_distances in objects_in_vicinity.items():
#     time_interaction_list_vicinity[object_name] = time_to_interaction(user_position, 
#                                                                       user_velocity, 
#                                                                       object_name, 
#                                                                       line_distances
# #                                                                       )

"""
Include a specific object 
"""
# if include_cutting_object == True: 
#         #     include_object(4404207983027294, filtered_obj_ids, visible_obj_ids, filtered_obj_positions_scene, 
#         #                     filtered_obj_positions_cam, filtered_dot_products, filtered_line_distances, 
#         #                     visible_obj_positions_scene, visible_obj_positions_cam, dot_products, line_distances)

# indices = np.where(np.array(objects_time_approach_list) < 2)[0]
# objects_within_time = [gt_provider.get_instance_info_by_id(filtered_obj_ids[indice]).name for indice in indices]
# objects_within_time_dict = {}
# for index in indices:
#     obj_name = gt_provider.get_instance_info_by_id(filtered_obj_ids[index]).name
#     objects_within_time_dict[obj_name] = f"{objects_time_approach_list[index]:.3f}"

"""
Write to history log file 
"""
# history_log_file = os.path.join(folder,'interaction_log.csv')
 
# Write the History in the log file
    # history_log_file = os.path.join(folder, 'history_log.yaml')

"""
Debugging
"""
####                       #### 
####       DEBUGGING       ####
####                       ####

# # test my code 
# for bbox_3d in bboxes3d.values():
#     print(type(bbox_3d.transform_scene_object))
#     print(np.shape(bbox_3d.transform_scene_object.to_matrix()))

# obj_poses = np.empty(len(bboxes3d), dtype=object)
# for i, bbox_3d in enumerate(bboxes3d.values()):
#     obj_poses[i] = bbox_3d.transform_scene_object


"""
Initial try to spot the movoement of an object 
"""

# # User's position
# user_position = aria_3d_pose_with_dt.data().transform_scene_device.translation()

# # Objects specific ids
# black_ceramic_bowl_id = visible_obj_names_ids['BlackCeramicBowl']
# choppingboard_id = visible_obj_names_ids['ChoppingBoard']

# # Positions of objects (a random one and ground truth one)
# position_blackceramic_bowl = visible_obj_positions_scene[np.where(visible_obj_ids == black_ceramic_bowl_id)[0][0]][0]
# position_choppingboard = visible_obj_positions_scene[np.where(visible_obj_ids == choppingboard_id)[0][0]][0]

# # distance from user 
# distance_blackceramic_bowl = np.linalg.norm(position_blackceramic_bowl - user_position)
# distance_choppingboard = np.linalg.norm(position_choppingboard - user_ema_velocity)

# # Initialize positions 
# if position_blackceramic_bowl_before is None: 
#     position_blackceramic_bowl_before = position_blackceramic_bowl
#     position_choppingboard_before = position_choppingboard

# # Calculate movements 
# movement_black_ceramic = np.linalg.norm(position_blackceramic_bowl - position_blackceramic_bowl_before)
# movement_choppingboard = np.linalg.norm(position_choppingboard - position_choppingboard_before)

# # If movement changed significantly 
# if movement_black_ceramic > 0.1 and distance_blackceramic_bowl < 0.7:
#     counts +=1
#     print('BlackCeramicVBowl has been moved', counts)
# if movement_choppingboard > 0.1 and distance_choppingboard <0.6:
#     counts_board +=1
#     print('ChoppingBoard has been moved', counts_board)

# # Update the old values    
# position_blackceramic_bowl_before = position_blackceramic_bowl
# position_choppingboard_before = position_choppingboard

"""
Collect the lists
"""

# # Collect necessary lists
#     objects_the_user_is_looking_to = get_objects_user_looking_to()
#     objects_near_to_the_user_distance_in_meters = get_objects_near_to_user_distance()
#     objects_approachable_in_less_than_2_seconds = get_objects_approachable_in_2_seconds()

"""
LLM Reactivatiion

"""
# ###                                             #### 
# ### INTERACTION RECOGNITION & LLM REACTIVATION  #### 
# ###                                             ####

# if llm_activated == True:
#     objects_current_positions = {gt_provider.get_instance_info_by_id(obj_id).name: position for obj_id, position in zip(filtered_obj_ids, filtered_obj_positions_scene)} # Me: Initial positions of filtered objects

#     # Initialize the previous_positions
#     if objects_previous_positions is None:
#         objects_previous_positions = objects_current_positions

#     # Check if interaction occured (either movement, proximity) 
#     interacted_object = detect_user_interaction(objects_previous_positions, objects_current_positions)

#     # If interaction occured activate the llm again 
#     if interacted_object:
#         user_interact_with_object(interacted_object)
#         print('The user interacts with the object:', interacted_object)
#         # llm_activated = False

#     # Update the variable for previous positions for the next loop
#     objects_previous_positions = objects_current_positions

#     # # When interaction stops reactivate the LLM
#     # if not interacted_object & :
#     #     llm_activated = False

"""
Dictionaris from names to ids and from ids to names
"""
# obj_names_to_ids = {gt_provider.get_instance_info_by_id(obj_id).name: obj_id for obj_id in obj_ids}               # Me: Names to ids     
# obj_ids_to_names = {obj_id: gt_provider.get_instance_info_by_id(obj_id).name for obj_id in obj_ids}               # Me: Ids to names
        
"""
Prin the current time
"""
  # print(f"Current Time in seconds:", time)
        # print(f"Current Time in nanoseconds is: {current_time_ns}")
        
"""
prints after object interaction has been observed
"""
# # Check if the set of names is not empty
# if  objects_that_moved:
#         print("User is interacting with the following object:", objects_that_moved[counter])
#         print('with norm value', norm_relative_transformation_arr[indexes_of_objects_that_moved[counter]])
#         print('and distance', distance_user_objects_scene[indexes_of_objects_that_moved[counter]])

"""
store the values with lists 
"""
 # for index in indexes_ground_truth:
#     name = obj_names[index]
#     if name in value_pose:
#         value_pose[name].append(norm_relative_transformation_arr[index])
#     else: 
#         value_pose[name] = []
    
"""
Calculate the counters inside the time window function 
"""
# cutoff_time_s = current_time_s - window_time_s

# # Clean up visibility_counter
# while visibility_counter[obj_id] and visibility_counter[obj_id][0][0] < cutoff_time_s:
#     visibility_counter[obj_id].popleft()

# # every time an old item is being remover all values should be updated
# for index in range(len(visibility_counter[obj_id])):
#         timestamp, counter = visibility_counter[obj_id][index]
#         visibility_counter[obj_id][index] = (timestamp, counter - 1)

""""
Inside the removal outdated entries function
"""
# if len(dq) == 90: for debugging
#     print(f"the object {key} is visible the whole time window")
# else:
#     print(f"The object {key} is visible only some parts of the time window")

"""
Debugging the high_dot products
"""
# # debugging
# if current_time_s == 2.733 and obj_id == 4887781471251754:
# high_dot_streaks[obj_id].append((current_time_s, calculate_high_dot_low_distance(obj_id, high_dot_streaks, typedict = 'dot')))

# elif current_time_s == 3.266 and obj_id == 4887781471251754:
# high_dot_streaks[obj_id].append((current_time_s, calculate_high_dot_low_distance(obj_id, high_dot_streaks, typedict = 'dot')))

# # debugging
# elif current_time_s == 3.733 and obj_id == 4887781471251754:
# high_dot_streaks[obj_id].append((current_time_s, calculate_high_dot_low_distance(obj_id, high_dot_streaks, typedict = 'dot'))) 

# # debugging 
# elif current_time_s == 5.233 and obj_id == 4887781471251754:
# high_dot_streaks[obj_id].append((current_time_s, calculate_high_dot_low_distance(obj_id, high_dot_streaks, typedict = 'dot')))

# else:

"""
Calculate the duration
"""
# equivalent way to calculate the number of counters that an object is visible inside the time window
# if len(visibility_counter[obj_id]) > 1:
#     earliest_count = visibility_counter[obj_id][0][1]
#     lalest_count = visibility_counter[obj_id][-1][1]
#     visible_counters = lalest_count - earliest_count + 1 

"""
high dot / distance and duration thresholds
"""

## Identify objects with high dot counts
# objects_high_dot_counts = {}
# for index, (obj_id, streaks) in enumerate(filtered_high_dot_counts.items()):
#     if len(streaks) >= 30: # the streaks[-1][1] >= 45: # 45 counts means that 1.5 second of consecutive high values more than 0.9 dot value s
#         object_name  = gt_provider.get_instance_info_by_id(obj_id).name
#         objects_high_dot_counts[object_name] = f"{float(filtered_dot_products[index]):.3f}" # Me: keep only one value from the floating form
#         if  object_name not in objects_high_dot_history:
#             objects_high_dot_history.append(gt_provider.get_instance_info_by_id(obj_id).name)          
                                                    
# ## Identify objects with low distance counts
# objects_in_vicinity = {}       
# objects_time_approach_list = []                                                                                                  # Me: objects_time_approach_dict = {}
# objects_less_than_2_seconds = {}
# for index, (obj_id, streaks) in enumerate(filtered_low_distance_counts.items()):
#     if len(streaks) >= 30:
#         obj_position = filtered_obj_positions_scene[index]
#         time = time_to_interaction (user_position_scene, user_ema_velocity, obj_position, obj_id)
#         objects_in_vicinity[gt_provider.get_instance_info_by_id(obj_id).name] = f"{float(filtered_line_distances[index]):.3f}"   # Me: keep only one value from the floating form
#         objects_time_approach_list.append(time)                                                                                  # Me: objects_time_approach_dict[gt_provider.get_instance_info_by_id(obj_id).name] = f"{float(time):.2f}"
#         if time < 2:
#             objects_less_than_2_seconds[gt_provider.get_instance_info_by_id(obj_id).name] = time
    
# ## Identify objects with high duration 
# high_duration_objects_dict = get_objects_with_high_values(filtered_duration, threshold=2)       

"""
Time to interaction function for debugging
"""

# if obj_id == 4404207983027294:
#     tti = float(relative_distance / relative_speed)
#     if tti < 2:
#         logging.info(f"time for 1st interaction is: {tti:.2f} seconds")
# else:


"""
Collecting data for objects
"""
# current_objects_within_radius = object_within_radius(visible_distance_user_objects_scene, visible_obj_names, 1.5)
# print ('the length of objects close to the user 1.5 meter is', len(current_objects_within_radius))
# times.append(current_time_s)
# objects_number_with_radius.append(len(current_objects_within_radius))
# objects_within_radius.append(current_objects_within_radius)

"""
For debugging 
"""
# For debugging                                                                               # 14376694245187
# if current_time_ns == 14373328124237 or current_time_ns == 14376494276437 or current_time_ns == 14376694245187 or current_time_ns == 14384126383862:            
#     print ('stop') # 4860206504015397: 'CakeMocha_A' # 4469271893111727: 'BlackCeramicDishLarge' # 4404207983027294: 'ChoppingBoard'

"""
For debugging - Movemeent 
"""
# For debugging purposes
# for index in indexes_ground_truth:
#     name = obj_names[index]
#     if name in value_pose:
#         value_pose[name] = np.append(value_pose[name], [norm_relative_transformation_arr[index]], axis=0)
#         means[name] = np.mean(value_pose[name], axis=0)  # axis=0 to get mean for each column
#         st_dev[name] = np.std(value_pose[name], axis=0)
#         times.append(current_time_ns)
#     else:
#         value_pose[name] = np.array([norm_relative_transformation_arr[index]])
#         means[name] = np.mean(value_pose[name])
#         st_dev[name] = np.std(value_pose[name])
#         times.append(current_time_ns)

# for name, pose in value_pose.items():
#     if pose[-1] > 0.01:
#         print(f"Object with name {name} moved")

"""
Write a file with a list of objects around the user
"""
# # Optionally save output
# output = os.path.join(folder,'output.txt')
# with open(output, 'w') as file:
#     # Write the content of the list
#     file.write("Content of objects_within_radius:\n")
#     for index, sublist in enumerate(objects_within_radius, start=1):
#         file.write(f"{index}: {', '.join(sublist)}\n")
        
"""
Visulation of objects 
"""

####                                                  #### 
####                OBJECTS VISUALIZATION             ####
####                                                  ####

# if args.runrr and args.visualize_objects:
#     soon_interact_objects = [] # 6219594724780839, 4243547695770937, 6243788802362822
#     for obj_id, dot_product, distance, position_cam, position_scene, time in zip(
#         filtered_obj_ids, 
#         filtered_dot_products,
#         filtered_line_distances,https://www.youtube.com/
#         filtered_obj_positions_cam, 
#         filtered_obj_positions_scene,
#         objects_time_approach_list,
#         ):
#         instance_info = gt_provider.get_instance_info_by_id(obj_id)
        
#         # Handling the object coordinates 
#         bbox_3d = bboxes3d[obj_id]
#         aabb_coords = bbox3d_to_line_coordinates(bbox_3d.aabb)
#         obb = np.zeros(shape=(len(aabb_coords), 3))
#         for i in range(0, len(aabb_coords)):
#             aabb_pt = aabb_coords[i]
#             aabb_pt_homo = np.append(aabb_pt, [1])
#             obb_pt = (bbox_3d.transform_scene_object.to_matrix() @ aabb_pt_homo)[0:3] # Me: transformation from οbject coordinate system to scene coordinate system 
#             obb[i] = obb_pt

#         # Prints # 
#         # print(f"Object ID: {obj_id},Dot Product: {dot_product}, Distance: {distance}, Position on scene {bbox_3d.transform_scene_object.translation()}")                
#         # print(f"Position on Camera frame: {position_cam}, Position on Scene frame: {position_scene}")
#         # print("-" * 40)  # Print a line of dashes as a separator | # More sophisticated separator
        
#         # Check to which objects the user is close to
#         if time < 2:
#             soon_interact_objects.append(instance_info.name)
            
#         # Log and Visualize the object
#         args.runrr and log_object(rr, instance_info, obb)
        
#         # Log and Visualize the line from camera to object
#         line_start = p_cam_scene[0]
#         line_end = position_scene[0]
#         args.runrr and log_object_line(rr, obj_id, instance_info, line_start, line_end)
        
#         # Add to previous_obj_ids to clear in the next timestep
#         previous_obj_names.add(instance_info.name)
#         previous_obj_ids.add(obj_id)

"""
high duration objects
"""
# high_duration_objects_dict = get_objects_with_high_duration(filtered_duration, threshold=2)                                    # Me: Get the objects with high values of duration

"""
time window 
"""
#  (avg_dots,
#  avg_distances,
#  past_dots,
#  past_distances,
#  visibility_counter,
#  visibility_duration,
#  high_dot_counts,
#  close_distance_counts) = time_window(
#                                     start_time,
#                                     current_time_s,
#                                     time_diff_ns,
#                                     parameters,
#                                     visible_obj_ids,
#                                     dot_products_list,
#                                     line_distances_cam_list,
#                                     past_dots,
#                                     past_distances,
#                                     visibility_counter,
#                                     visibility_duration,
#                                     high_dot_counts,
#                                     close_distance_counts,
#                                     avg_dots, 
#                                     avg_distances,
#                                     VARIABLES_WINDOW_TIME, 
#                                     HIGH_DOT_THRESHOLD, 
#                                     LOW_DISTANCE_THRESHOLD
# )

"""
conditions to activate the LLM 
"""

# if (objects_high_dot_counts 
#     and objects_in_vicinity                                 
#     and np.any(np.array(objects_time_approach_list) < 2)
#     and objects_less_than_2_seconds_with_counts_dict        
#     and high_duration_objects_dict
#     ):

# # Print statement 
# print("the 4th criteria have been satisfied")

# append new data to the string 
# log = append_to_history_string(current_time_s, 
#                             "Living Room", 
#                             objects_high_dot_counts,
#                             objects_in_vicinity,
#                             objects_less_than_2_seconds_with_counts_dict,
#                             )

# Remove entries older than 3 seconds
# while history_log and history_log[0]['timestamp'] < round((current_time_s - HISTORY_WINDOW_TIME),3):
#     history_log.popleft() # if it was just a list could be harder to do the same thing 


"""
Prompt to LLM 
"""

# # Prompt number 1
#     prompt_constant= """
#                         The user navigates the space intending to interact with certain objects to perform a task. As is normal, before interacting with an object, 
#                         people typically look in the direction of that object or at a set of objects in its vicinity. In view of this, I have compiled a list of objects
#                         called 'objects_the_user_is_looking_to.' At each timestep, I include objects that showed consistently high values during a 3-second time window. 
#                         In other words, the user was looking at these objects for a significant duration during the last 3 seconds. The value, which indicates where the 
#                         user is looking, ranges from 0.7 to 1.0. The higher the value, the more the user is looking at the objects. Identifying objects that consistently 
#                         have very high values over many timesteps is important, as this indicates the user is focusing on them and likely intends to interact with them in
#                         the near future, even if they are not close. Therefore, these objects are strong indicators of where the user is looking and are likely potential
#                         interaction targets.
    
#                         Additionally, as the user moves through the space, they may be come closer to some objects, sometimes very close. However, this proximity does not 
#                         necessarily mean interaction, as the user might just pass by. If some of these close objects show distance consistency from the user over a time window,
#                         it indicates that the user remains near these objects during that period. Remaining around these objects suggests a high potential interaction in the near
#                         future. Objects that meet these both criteria at each timestamp (proximity and consistency) appear in the list 'objects_near_to_the_user_distance_in_meters'.
#                         If these objects consistently appear in this list across different timestamps and their distance decreases over time or remain stable, it means the user is 
#                         approaching them and shows some interest around these objects. It is important to note that an object the user will interact with has to be in this list, 
#                         especially in the most recent timestamps. It is 90 percent certain that an object the user will interact with should appear in this list at some point.

#                         Now if some of these objects that show consistency seems to have time to interaction less than 2 seconds. So now in summary if the user sees a group of 
#                         objects consistently and at some point in the sequence approach them is close also consistently that means the user might hilghy interact with these objects. 
#                      """

"""
the prompt for the LLM prediction 
"""

# def append_to_history_string(time, location, objects_high_dot_counts, objects_in_vicinity, objects_less_than_2_seconds):
# #TODO: DONE | Message add the list with high counts/duration dot products --> this will help the LLM to see which objects the user consistently used to see even if are not necessary at the current timestep
# #TODO: DONE | Add a list with high counts/duration low distance           --> this will help the LLM to see of some objects are important for
# #TODO: DONE | Add a list with high counts/duration time                   --> this will help see if the user intends to interact with an object or no

# # log_entry = f"Time: {time}, Location: {location}, High Dot Counts: {objects_high_dot_counts}, In Vicinity: {objects_in_vicinity}, Less than 2 seconds: {objects_less_than_2_seconds}\n"
# log_entry = {
# 'timestamp': time,
# 'place': location,
# 'objects_the_user_is_looking_to_value_no_measure_from_0_to_1': objects_high_dot_counts,
# 'objects_near_to_the_user_distance_in_meters': objects_in_vicinity,
# 'objects_approachable_in_less_than_2_seconds_time_in_seconds': objects_less_than_2_seconds,   
# }

# # print("Log Entry:")

# return log_entry

# def append_to_history_string_try(time, location, objects_high_dot_counts_history, objects_low_distance_counts_history, objects_less_than_2_seconds, predictions)
    
#     # log_entry = f"Time: {time}, Location: {location}, High Dot Counts: {objects_high_dot_counts}, In Vicinity: {objects_in_vicinity}, Less than 2 seconds: {objects_less_than_2_seconds}\n"
#     log_entry = {
#         'timestamp': time,
#         'place': location,
#         'objects_approachable_in_less_than_2_seconds_time_in_seconds': objects_less_than_2_seconds,
#         'objects_that_the_user_looks_consistently_intensively_in_counts_within_a_window': objects_high_dot_counts_history,
#         'objects_that_the_user_is_close_costently_in_counts_in_counts_within_a_window': objects_low_distance_counts_history,
#         'objects_that_the_user_probably_interacted_with_in_the_past': predictions, 
#     }
    
    # print("Log Entry:")
    
    # return log_entry

# def append_to_history_log(time, place, high_dot_objects, close_objects, objects_time, history_log_file='history_log.yaml'):
   
#     log_entry = {
#         'timestamp': time,
#         'place': place,
#         'objects_the_user_is_looking_The_values_are_from_0_to_1': high_dot_objects,
#         'objects_near_to_the_user_distance_in_meters': close_objects,
#         'objects_approachable_in_less_than_2_seconds_time_in_seconds': objects_time,
#     }
    
#     print(json.dumps(log_entry, indent=4))

#     with open(history_log_file, 'a') as file:
#         yaml.dump([log_entry], file, default_flow_style=False, sort_keys=False)

"""
Objectds with high dots 
"""
# for index, object_id in enumerate(filtered_obj_ids):
#     if object_id in filtered_high_dot_counts and len(filtered_high_dot_counts[object_id]) >= HIGH_DOT_COUNTERS_THRESHOLD:               # Me: The user high dot product is higher for these objects consequtively
#         object_name  = gt_provider.get_instance_info_by_id(object_id).name                                                              # Me: Object name
#         if object_name in objects_do_not_include:
#             continue
#         objects_high_dot_counts[object_name] = f"{float(filtered_dot_products[index]):.3f}"                                             # Me: Keep only one value from the floating form
#         if object_name not in objects_high_dot_history:
#             objects_high_dot_history.append(object_name)      

"""
History Log 
"""

# write the log
# log = append_to_history_string(current_time_s, 
#                         "Living Room", 
#                         filtered_names_len_high_dot_counts,
#                         filtered_names_len_low_distance_counts,
#                         filtered_names_len_time_counts,
#                         predictions_dict,
#                         )

# history_log.append(log)

# # write_to_excel(filtered_names_len_high_dot_counts, filtered_names_len_low_distance_counts, objects_less_than_2_seconds, objects_in_vicinity)

# # Convert history log to a string
# history_log_string = "\n".join([str(entry) for entry in history_log])
# history_log_string = "\n".join([str(entry) for entry in history_log])

"""
Prompts for the LLM 
"""

#  # Prompt number 1
#     prompt_constant= """
#                         The user navigates the space with the intention of interacting with specific objects to accomplish a task. We make the assumption that, before interacting with an object, 
#                         users tend to look at that object or a group of nearby ones. Based on this behavior, a list titled "observed_objects_with_high_focus_measured_in_counts" has been compiled.
#                         This list includes objects with counts representing the number of frames the user observed them intensively within a time window of a certain amount of seconds (e.g. 3 
#                         seconds). Each second consists of 30 frames. For instance, within a 3-second time window (90 frames), an object with a count of 45 indicates it was observed intensely for 
#                         45 frames (or equivelenlty 1.5 seconds) the last 3 seconds. Essentially, objects with higher counts were in user's focus for a larger portion of the time window. A higher
#                         count suggests more attention from the user, indicating a possible intention to interact with these objects, even if they are not physically close. We also collect a data
#                         which shows that last focus intensity only for objects where counts are above a certain threshold (e.g. 45 counts)

#                         As the user moves through the space, they may come close to various objects. Proximity alone doesn't necessarily indicates an interaction, as the user might just be passing
#                         by. However, tracking the duration the user stays near these objects can hint at potential future interactions. The list titled "nearby_objects_measured_in_counts" records 
#                         the duration the user remains close to objects within a specified time window similar to the way that I described for focus intensity. The values in this list are counts 
#                         (representing the number of frames), with a higher count indicating a longer time user is near an object within a time window. For example, if the time window is a 3-seconds
#                         (equivalent to 90 counts or frames), and an object has a count of 30, it means the distance from the user to that object was within a certain threshold (for instance 2 meters) 
#                         for 30 frames, or 1 second. 

#                         Identifying objects with higher counts in this list suggests a greater likelihood of interaction, even if the object has fewer counts in the "observed_objects_with_high_focus_
#                         measured_in_counts" list. If an object has more than counts than threshold counter (e.g. more than 30 counts) there is a list named "objects_names_and_latest_distance_from_the_user"
#                         which shows the current distance of the user from the object where the user was more consistently close enough.  

#                         Additionally, there is a list called 'approachable_objects_in_short_duration_measured_in_counts' which identifies objects the user can approach within a certain time 
#                         threshold, based on velocity and distance vectors. The number associated with each object represents the counts or frames during which the user can approach that object 
#                         within that threshold. These objects in this list are very close to the user and represent a subset of the "nearby_objects_measured_in_counts" list. 

#                         Finally, a list of previous predictions "previous_predictions" is provided to enhance the accuracy of future predictions and to better understand the user's overall goal.
#                     """


#     # Prompt number 2
#     prompt_change =  """
#                         Based on the provided instructions, predict the three objects the user is most likely to interact with, listed in order of highest likelihood. Use the actual
#                         names of the objects from the log and include the probability of interaction for each object. Note the sum of the 3 probabilities should be 1 according to the 
#                         definition of it. Additionally, explain the rationale behind each choice. Finally,the output should be formatted in YAML format, as the following example:

#                         The output should be plain YAML without additional formatting characters like triple quotes. Here is an example format:

#                         most_likely_objects_to_interact_with:
#                             - object: Object1
#                                 probability: 0.4
#                             - object: Object2
#                                 probability: 0.35
#                             - object: Object3
#                                 probability: 0.25

#                         rationale:
#                             - object: Object1
#                                 reason: Reason for Object1.
#                             - object: Object2
#                                 reason: Reason for Object2.
#                             - object: Object3
#                                 reason: Reason for Object3.

#                         predicted_interaction_objects:
#                             - Object1
#                             - Object2
#                             - Object3

#                         goal_of_the_user: Based on above predictions and the past predictions from the list (previous_predictions) I assume that the goal of the user is, 
#                      """


"""
Interaction Detection
"""
# def detect_user_interaction(objects_previous_positions, 
#                             objects_current_positions
#                             ):
#     """ 
#         Placeholder for actual interaction detection logic
#         This could be event-driven or polled based on application design
#     """
    
#     # Integrate different detection methods here:
#     movable_objects_interaction = detect_user_interaction_object_motion(objects_previous_positions, objects_current_positions)
#     # proximity_interaction = detect_user_interaction_proximity()
    
#     # Prioritize interactions if multiple detections occur
#     if movable_objects_interaction:
#         return movable_objects_interaction
#     # elif proximity_interaction:
#     #     return proximity_interaction
#     return None

# ## Movement 
# def has_object_moved(object_position_initial, object_position_current, movement_threshold=0.0):
#     position_difference = object_position_initial - object_position_current
#     distance = np.linalg.norm(position_difference) # np.linalg.norm except 1 argument not two
#     return distance > movement_threshold

# def detect_user_interaction_object_motion(objects_initial_positions, objects_current_positions, movement_threshold=0.0):
#     objects_moved = []
#     for object_name, initial_position in objects_initial_positions.items():
#         try:
#             current_position = objects_current_positions[object_name]
#         except KeyError:
#             print(f"KeyError: '{object_name}' not found in objects_current_positions. Skipping this object.")
#             continue
#         if object_name == 'BlackKitchenChair':
#             if has_object_moved(initial_position, current_position, movement_threshold):
#                 objects_moved.append(object_name)
#         if has_object_moved(initial_position, current_position, movement_threshold):
#                 objects_moved.append(object_name)
#     return objects_moved if objects_moved else None

# ## Distance 
# def detect_user_interaction_proximity():
#     proximity_sensor_data = is_user_consistently_near_object()
#     return handle_distance_data(proximity_sensor_data)

# def is_user_consistently_near_object(user_position, object_position, threshold_distance=0.05, consistency_duration=5):
#     distances = []
#     for timestamp in range(consistency_duration):
#         distance = np.linalg.norm(user_position[timestamp], object_position[timestamp])
#         distances.append(distance)

#     return all(d < threshold_distance for d in distances)     # Check if all distances are below the threshold

# def handle_distance_data(proximity_sensor_data):
#     for data in proximity_sensor_data:
#         if is_user_near(data):
#             return data['object_name']
#     return None

# def is_user_near(data, interaction_distance = 0.6):
#     return data['distance'] < interaction_distance

# def is_interacting(hand, obj, interaction_threshold = 0.05 ):
#     distance = np.linalg.norm(hand['coordinates'], obj['coordinates'])
#     return distance < interaction_threshold

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

"""
inlcude object for debugging 
"""
# def include_object(obj_id_object, filtered_obj_ids, visible_obj_ids, filtered_obj_positions_scene, filtered_obj_positions_cam, filtered_dot_products, filtered_line_distances, visible_obj_positions_scene, visible_obj_positions_cam, dot_products, line_distances):
#     if obj_id_object in filtered_obj_ids:
#         idx = np.where(filtered_obj_ids == obj_id_object)[0][0]  
#         # print((f"Inside the filtered_objects"))
#         # print((f"Distance of the object from the device: {filtered_line_distances[idx]}"))
#         # print((f"Dot product of the object from the device: {filtered_dot_products[idx]}"))
#         # print("")
        
#     # Check if the object ID of interest is in the visible object IDs and not already in the filtered object IDs
#     if obj_id_object in visible_obj_ids and obj_id_object not in filtered_obj_ids:
#         filtered_obj_ids = np.append(filtered_obj_ids, obj_id_object)   # Include the object ID of interest in the filtered object ID
#         idx = np.where(visible_obj_ids == obj_id_object)[0][0]          # Retrieve the index of the object ID of interest in visible object IDs
#         filtered_obj_positions_scene = np.vstack((filtered_obj_positions_scene, visible_obj_positions_scene[idx][np.newaxis, :]))
#         filtered_obj_positions_cam = np.vstack((filtered_obj_positions_cam, visible_obj_positions_cam[idx][np.newaxis, :]))
#         filtered_dot_products = np.append(filtered_dot_products, dot_products[idx])
#         filtered_line_distances =  np.append(filtered_line_distances, line_distances[idx])
#         # print((f"Inside the visible but not filtered objects"))
#         # print((f"Distance of the object from the device: {filtered_line_distances[-1]}"))
#         # print((f"Dot product of the object from the device: {filtered_dot_products[-1]}"))
#         # print("")

"""
Detection algorithm 
"""

# norm_relative_T_object, T_scene_object_before = calculate_relative_pose_difference(T_scene_object_before, visible_bboxes3d)
# indexes_ground_truth = generate_indexes(ground_truth, obj_names)

# # update the last time that object poses were checked 
# last_checked_time = current_time_s

# # Find the object indexes that satisfy your condition for potential interaction 
# indexes_activation = np.where((norm_relative_T_object) > 0.004)[0] # 4mm 

# # IF Condition to check interaction took place (apart from norm of relative transformation need to check also the distance which should be relative close)
# if indexes_activation.size > 0: # Me: # if indexes_activation.size > 0 and detection_phase is True: 

#     # use the indexes to find THE potential objects names that moved and respective distance from them:
#     objects_that_probably_moved_names = obj_names[indexes_activation]
#     distances_from_objects_that_probably_moved_names = distance_user_objects_scene[indexes_activation]

#     for i, name in enumerate(objects_that_probably_moved_names):
#         if distances_from_objects_that_probably_moved_names[i] < 1:
#             original_objects_that_moved_dict[current_time_s] = name
#             # original_objects_that_moved_list.append(name)
#             print("Objects that have been moved so far:", original_objects_that_moved_dict)
#             indexes_of_objects_that_moved.append(indexes_activation[i])
#             counter += 1 
#             detection_phase = False   # do not detect again for movement until movement has stopped 

# if indexes_activation.size == 0 and detection_phase is False:
#     detection_phase = True                

"""
Take the only poses of the visible objects 
"""
# T_scene_object = {key: value for key, (include, value) in zip(bboxes3d.keys(), zip(valid_mask, bboxes3d.values())) if include}    

"""
Refine dictionary
"""

# ## Refine the Ground Truth Values to go to the comparisons
# result_objects_that_moved_dict = refine_dict_keep_unique(original_objects_that_moved_dict, result_objects_that_moved_dict, last_value = None)   

"""
Check if interaction with objects has finished 
"""

# # TODO: norm_relative_for_specific_object and check if interaction with the object has finished
# if  objects_that_moved and detection_phase is False:
#     if norm_relative_transformation_arr[indexes_of_objects_that_moved[counter]] < 0.01 and distance_user_objects_scene[indexes_of_objects_that_moved[counter]] > 1:
#         detection_phase = True 

#         # Initialize the objects for the next round
#         objects_that_probably_moved_names = []
#         distances_from_objects_that_probably_moved_names = [] 

"""
Predict the items that the user is moving along with the user 
"""

#  objects_do_not_include = []                                                                                                                             # Me: provide a fresh list at each timestep               
#         if very_close_distance_counts:
#             for index, (object_id, values) in enumerate(very_close_distance_counts.items()):
#                 if very_close_distance_counts[object_id][-1][0] == current_time_s:                                                                              # Me: here is we check if the list for a specific object has been updated 
#                     object_name = gt_provider.get_instance_info_by_id(object_id).name                                                    
#                     if object_name not in user_object_position:                                                                          
#                         user_object_position[object_name] = deque([user_ema_position[:2]])                                                 
#                         user_object_movement[object_name] = 0
#                         user_object_movement_abs[object_name] = 0
#                     else: 
#                         user_object_position[object_name].append(user_ema_position[:2]) # only xy plane
#                         user_object_movement[object_name] += np.linalg.norm(user_object_position[object_name][-1] - user_object_position[object_name][-2])      # Me: this is the subsequent movements at each timestep 
#                         user_object_movement_abs[object_name] = np.linalg.norm(user_object_position[object_name][-1] - user_object_position[object_name][0])    # Me:  this is the linear movement from the start to current 
#                         print('user is moving', {user_object_movement[object_name]})
#                     if user_object_movement[object_name] > 1:
#                         print(f"user is moving with the object {object_name}")
#                         objects_the_user_is_moving_with[current_time_s] = object_name
#                         objects_do_not_include.append(object_name)

## Identify objects that are moving along with the user 

"""
Τwo ways of doing that: 
    - check the objects around the user less than 0.7 and if the user moved
    - check the objects that are being moved while the user is moving from one group of objects to another 
    
Below is the 1st way of doing the thing 
"""

# ###                                 ###
# ###    Objects close to the user    ###
# ###                                 ###
    
# objects_do_not_include = [] # Me: provide a fresh list at each timestep  

# # Find objects within 1 meter distance
# very_close_distance_objects = obj_names[np.where(distance_user_objects_scene < 0.85)]

# if len(very_close_distance_objects) > 0:
#     for index, name in enumerate(very_close_distance_objects):
#         # Check if the object name is not the same as the previous one
#         if name not in previous_name:
#             # Initialize tracking for new objects
#             if name not in user_object_position:
#                 user_object_position[name] = deque([user_ema_position[:2]])   # Store only recent positions
#                 user_object_movement[name] = 0
#             else:
#                 # Update position and calculate the movement
#                 start_position = user_object_position[name][0]
#                 current_position = user_ema_position[:2]                                # Only xy plane
#                 user_object_position[name].append(current_position)
#                 if name == 'ChoppingBoard':
#                     print('')
                
#                 # Calculate the distance moved in the current timestep
#                 distance_moved = np.linalg.norm(current_position - start_position)
#                 user_object_movement[name] = distance_moved

#                 # Debugging output for user movement
#                 print(f"user is moving {name}: movement = {user_object_movement[name]}")

#             # Detect if the object has moved more than 1 meter
#             if user_object_movement[name] > 0.5:
#                 if name == 'WoodenSpoon':
#                     print('')
#                 previous_name = very_close_distance_objects                             # Update previous_name
#                 user_object_movement[name] = 0                                          # Reset movement to prevent immediate re-detection
#                 print(f"user is moving with the object {name}")
#                 objects_the_user_is_moving_with[current_time_s] = name
#                 objects_do_not_include.append(name)

"""
Loop over the filtered objects to see the objects wuth low time approach 
""" 

## Identify objects with LOW TIME COUNTS
# for index, object_id in enumerate(filtered_obj_ids):
#     if object_id in filtered_time_to_approach_counts and len(filtered_time_to_approach_counts[object_id]) > TIME_COUNTERS_THRESHOLD:    # Me: the user is around this object for more than 30 frames within 90 frames
#         object_name = gt_provider.get_instance_info_by_id(object_id).name                                                               # Me: the name of the object that satisfies the criteria
#         if object_name in objects_that_is_moving:
#             continue
#         objects_less_than_2_seconds_counts[object_name] = f"{float(filtered_line_distances[index]):.3f}"                                # Me: keep only one value from the floating form
#         objects_time_approach_list.append(object_time)                                                                                  # Me: objects_time_approach_dict[gt_provider.get_instance_info_by_id(obj_id).
#         if object_time < 2:
#             objects_less_than_2_seconds[object_name] = object_time

"""
From the prompt text: 
The number associated with each object represents the counts or frames during which the user can approach that object within that threshold. 
"""


"""
Identify and activate the LLM 
"""

# if (np.any(np.array(list(filtered_names_len_high_dot_counts.values())) > HIGH_DOT_COUNTERS_THRESHOLD)
#     and np.any(np.array(list(filtered_names_len_low_distance_counts.values())) > LOW_DISTANCE_COUNTERS_THRESHOLD)
#     and objects_names_less_than_2_seconds_dict
#     and objects_high_duration
#     and not objects_less_than_2_seconds_with_counts_dict                                                            
#     ):


"""
Writing the excel files 
"""

# def write_to_excel(dict1, dict2, dict3, dict4, dict5, dict6, dict7, dataset_name, parameters_folder_name):
#     # Define paths
#     project_path = "Documents/projectaria_sandbox/projectaria_tools/projects/AriaDigitalTwinDatasetTools/object_anticipation/adt/"
#     excel_folder = os.path.join(project_path, 'utils', 'excel', dataset_name, parameters_folder_name)
#     os.makedirs(excel_folder, exist_ok=True)
    
#     excel_file = os.path.join(excel_folder, 'dictionaries.xlsx')
    
#     # Define a separator DataFrame to insert between appends
#     separator_df = pd.DataFrame([['--- APPEND SEPARATOR ---'] * len(dict1)])  # Adjust to match column length


#     # Initialize dataframes for existing data
#     if os.path.exists(excel_file):
#         with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
#             existing_sheets = writer.book.sheetnames
#             df1_existing = pd.read_excel(excel_file, sheet_name='Dot_counts') if 'Dot_counts' in existing_sheets else pd.DataFrame()
#             df2_existing = pd.read_excel(excel_file, sheet_name='Distance_counts') if 'Distance_counts' in existing_sheets else pd.DataFrame()
#             df3_existing = pd.read_excel(excel_file, sheet_name='Dot_values') if 'Dot_values' in existing_sheets else pd.DataFrame()
#             df4_existing = pd.read_excel(excel_file, sheet_name='Distance_values') if 'Distance_values' in existing_sheets else pd.DataFrame()
#             df5_existing = pd.read_excel(excel_file, sheet_name='Time_less_2') if 'Time_less_2' in existing_sheets else pd.DataFrame()
#             df6_existing = pd.read_excel(excel_file, sheet_name='Predicted') if 'Predicted' in existing_sheets else pd.DataFrame()
#             df7_existing = pd.read_excel(excel_file, sheet_name='Goal') if 'Goal' in existing_sheets else pd.DataFrame()
#             df_common_existing = pd.read_excel(excel_file, sheet_name='Common') if 'Common' in existing_sheets else pd.DataFrame()
#             df_common_at_least_two_existing = pd.read_excel(excel_file, sheet_name='Common_at_least_2') if 'Common_at_least_2' in existing_sheets else pd.DataFrame()
#             merged_df_existing = pd.read_excel(excel_file, sheet_name='Combined Lists') if 'Combined Lists' in existing_sheets else pd.DataFrame()
            
#             # Append the separator row
#             separator_df.to_excel(writer, sheet_name='Dot_counts', index=False, header=False, startrow=len(df1_existing)+1)
            
#     else:
#         df1_existing = df2_existing = df3_existing = df4_existing = df5_existing = pd.DataFrame()
#         df6_existing = df7_existing = df_common_existing = df_common_at_least_two_existing = merged_df_existing = pd.DataFrame()
    
#     # Convert the dictionaries to dataframes
#     df1 = pd.DataFrame(list(dict1.items()), columns=['Name', 'Dot - Counts'])
#     df2 = pd.DataFrame(list(dict2.items()), columns=['Name', 'Distance - Counts'])
#     df3 = pd.DataFrame(list(dict3.items()), columns=['Name', 'Dot - Value'])
#     df4 = pd.DataFrame(list(dict4.items()), columns=['Name', 'Distance - Value'])
#     df5 = pd.DataFrame(list(dict5.items()), columns=['Name', 'Time less than 2'])
#     df6 = pd.DataFrame(list(dict6.items()), columns=['Time', 'Predicted Objects'])
#     df7 = pd.DataFrame(list(dict7.items()), columns=['Time', 'Goal'])
    
#     # Merge the dataframes
#     merged_df = pd.merge(df1, df2, on='Name', how='outer')
#     merged_df = pd.merge(merged_df, df3, on='Name', how='outer')
#     merged_df = pd.merge(merged_df, df4, on='Name', how='outer')
#     merged_df = pd.merge(merged_df, df5, on='Name', how='outer')
    
#     # Find common objects in all three lists
#     common_names = set(df1['Name']).intersection(df2['Name']).intersection(df5['Name'])

#     # Create a DataFrame for common objects
#     df_common = pd.DataFrame({'Name': list(common_names)})
#     df_common['Dot'] = df_common['Name'].map(dict1)
#     df_common['Distance'] = df_common['Name'].map(dict2)
#     df_common['Time'] = df_common['Name'].map(dict5)

#     # Find objects common in at least two lists
#     at_least_common_names = set(df1['Name']).intersection(df2['Name']).union(
#                             set(df1['Name']).intersection(df5['Name'])).union(
#                             set(df2['Name']).intersection(df5['Name']))
                
#     # Create a DataFrame for common objects
#     df_common_at_least_two = pd.DataFrame({'Name': list(at_least_common_names)})
#     df_common_at_least_two['Dot'] = df_common_at_least_two['Name'].map(dict1)
#     df_common_at_least_two['Distance'] = df_common_at_least_two['Name'].map(dict2)
#     df_common_at_least_two['Time'] = df_common_at_least_two['Name'].map(dict5)

#     # Append the new data to the existing data
#     df1_combined = pd.concat([df1_existing, df1], ignore_index=True)
#     df2_combined = pd.concat([df2_existing, df2], ignore_index=True)
#     df3_combined = pd.concat([df3_existing, df3], ignore_index=True)
#     df4_combined = pd.concat([df4_existing, df4], ignore_index=True)
#     df5_combined = pd.concat([df5_existing, df5], ignore_index=True)
#     df6_combined = pd.concat([df6_existing, df6], ignore_index=True)
#     df7_combined = pd.concat([df7_existing, df7], ignore_index=True)
#     df_common_combined = pd.concat([df_common_existing, df_common], ignore_index=True)
#     df_common_at_least_two_combined = pd.concat([df_common_at_least_two_existing, df_common_at_least_two], ignore_index=True)
#     merged_df_combined = pd.concat([merged_df_existing, merged_df], ignore_index=True)

#     # Save to Excel
#     with pd.ExcelWriter(excel_file, engine='openpyxl', mode='w') as writer:
#         df1_combined.to_excel(writer, sheet_name='Dot_counts', index=False)
#         df2_combined.to_excel(writer, sheet_name='Distance_counts', index=False)
#         df3_combined.to_excel(writer, sheet_name='Dot_values', index=False)
#         df4_combined.to_excel(writer, sheet_name='Distance_values', index=False)
#         df5_combined.to_excel(writer, sheet_name='Time_less_2', index=False)
#         df6_combined.to_excel(writer, sheet_name='Predicted', index=False)
#         df7_combined.to_excel(writer, sheet_name='Goal', index=False)
        
#         # Common 
#         df_common_combined.to_excel(writer, sheet_name='Common', index=False)
#         df_common_at_least_two_combined.to_excel(writer, sheet_name='Common_at_least_2', index=False)
#         merged_df_combined.to_excel(writer, sheet_name='Combined Lists', index=False)

"""
Writing the excel files (2nd try)
"""
# def write_to_excel(dict1, dict2, dict3, dict4, dict5, dict6, dict7):
    
#     # Define paths
#     project_path = "Documents/projectaria_sandbox/projectaria_tools/projects/AriaDigitalTwinDatasetTools/object_anticipation/adt/"
#     csv_file = os.path.join(project_path,'utils','excel','dictionaries.xlsx')
#     os.makedirs(csv_file, exist_ok=True)
        
#     # Convert the dictionaries to dataframes
#     df1 = pd.DataFrame(list(dict1.items()), columns=['Name', 'Dot - Counts'])
#     df2 = pd.DataFrame(list(dict2.items()), columns=['Name', 'Distance - Counts'])
#     df3 = pd.DataFrame(list(dict3.items()), columns=['Name', 'Dot - Value'])
#     df4 = pd.DataFrame(list(dict4.items()), columns=['Name', 'Distance - Value'])
#     df5 = pd.DataFrame(list(dict5.items()), columns=['Name', 'Time less than 2'])
#     df6 = pd.DataFrame(list(dict6.items()), columns=['Time', 'Predicted Objects'])
#     df7 = pd.DataFrame(list(dict7.items()), columns=['Time', 'Goal'])
    
#     # Merge the dataframes
#     merged_df = pd.merge(df1, df2, on='Name', how='outer')
#     merged_df = pd.merge(merged_df, df3, on='Name', how='outer')
#     merged_df = pd.merge(merged_df, df4, on='Name', how='outer')
#     merged_df = pd.merge(merged_df, df5, on='Name', how='outer')
      
#     # Find common objects in all three lists
#     common_names = set(df1['Name']).intersection(df2['Name']).intersection(df5['Name'])

#     # Create a DataFrame for common objects
#     df_common = pd.DataFrame({'Name': list(common_names)})
#     df_common['Dot'] = df_common['Name'].map(dict1)
#     df_common['Distance'] = df_common['Name'].map(dict2)
#     df_common['Time'] = df_common['Name'].map(dict5)

#     # Find objects common in at least two lists
#     at_least_common_names = set(df1['Name']).intersection(df2['Name']).union(
#                             set(df1['Name']).intersection(df5['Name'])).union(
#                             set(df2['Name']).intersection(df5['Name']))
                
#     # Create a DataFrame for common objects
#     df_common_at_least_two = pd.DataFrame({'Name': list(at_least_common_names)})
#     df_common_at_least_two['Dot'] = df_common_at_least_two['Name'].map(dict1)
#     df_common_at_least_two['Distance'] = df_common_at_least_two['Name'].map(dict2)
#     df_common_at_least_two['Time'] = df_common_at_least_two['Name'].map(dict5)

#     # Save to Excel
#     with pd.ExcelWriter(csv_file) as writer:
#         df1.to_excel(writer, sheet_name='Dot_counts', index=False)
#         df2.to_excel(writer, sheet_name='Distance_counts', index=False)
#         df3.to_excel(writer, sheet_name='Dot_values', index=False)
#         df4.to_excel(writer, sheet_name='Distance_values', index=False)
#         df5.to_excel(writer, sheet_name='Time_less_2', index=False)
#         df6.to_excel(writer, sheet_name='Predicted', index=False)
#         df7.to_excel(writer, sheet_name='Goal', index=False)
        
#         # common 
#         df_common.to_excel(writer, sheet_name='Common', index=False)
#         df_common_at_least_two.to_excel(writer, sheet_name='Common_at_least_2', index=False)
#         merged_df.to_excel(writer, sheet_name='Combined Lists', index=False)

# in case in need to add the time as well 

"""
Writing the excel files (3rd try)
"""

# import pandas as pd
# from openpyxl import load_workbook
# from datetime import datetime

# def write_to_excel(filtered_names_len_high_dot_counts, filtered_names_len_low_distance_counts, objects_less_than_2_seconds, file_path='output.xlsx'):
#     # Create DataFrame for the current data
#     df_high_dot_counts = pd.DataFrame(filtered_names_len_high_dot_counts, columns=['Name', 'HighDotCount'])
#     df_low_distance_counts = pd.DataFrame(filtered_names_len_low_distance_counts, columns=['Name', 'LowDistanceCount'])
#     df_less_than_2_seconds = pd.DataFrame(objects_less_than_2_seconds, columns=['Name', 'LessThan2Seconds'])
    
#     # Add current timestamp
#     current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     df_high_dot_counts['Timestamp'] = current_time
#     df_low_distance_counts['Timestamp'] = current_time
#     df_less_than_2_seconds['Timestamp'] = current_time

#     # Append data to the Excel file
#     try:
#         # Load existing workbook
#         book = load_workbook(file_path)
        
#         with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
#             writer.book = book
#             writer.sheets = {ws.title: ws for ws in book.worksheets}
            
#             # Append to existing sheets if they exist
#             for df, sheet_name in zip([df_high_dot_counts, df_low_distance_counts, df_less_than_2_seconds], 
#                                       ['HighDotCounts', 'LowDistanceCounts', 'LessThan2Seconds']):
#                 if sheet_name in writer.sheets:
#                     # Append data to existing sheet
#                     startrow = writer.sheets[sheet_name].max_row
#                     df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False, header=False)
#                 else:
#                     # Write new sheet
#                     df.to_excel(writer, sheet_name=sheet_name, index=False)
#     except FileNotFoundError:
#         # If the file doesn't exist, create it and write data
#         with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
#             df_high_dot_counts.to_excel(writer, sheet_name='HighDotCounts', index=False)
#             df_low_distance_counts.to_excel(writer, sheet_name='LowDistanceCounts', index=False)
#             df_less_than_2_seconds.to_excel(writer, sheet_name='LessThan2Seconds', index=False)

"""
Writing the excel files (old with examples)
"""

# # Example usage within a loop
# start_time = ...  # Define your start time
# end_time = ...    # Define your end time
# while start_time < end_time:
#     # Your code to generate filtered_names_len_high_dot_counts, filtered_names_len_low_distance_counts, objects_less_than_2_seconds
#     filtered_names_len_high_dot_counts = ...
#     filtered_names_len_low_distance_counts = ...
#     objects_less_than_2_seconds = ...

#     # Call the function to write data to Excel
#     write_to_excel(filtered_names_len_high_dot_counts, filtered_names_len_low_distance_counts, objects_less_than_2_seconds)

#     # Update start_time for the next iteration
#     start_time = ...  # Update start time as per your logic


# # Define the lists
# list1 = {'Mango_A': 26, 'ChoppingBoard': 46, 'WoodenBowl': 19, 'WhiteChair': 35, 'NIT_CoffeeCan_Anon': 15, 'HeatTrivet': 15, 'WoodenFork': 38}
# list2 = {'BlackBarStool_B': 2, 'RedClock': 85}
# list3 = {'RedClock': 1.557347180232865}

"""
Parts of the visualization section / Visualize the vectos of the camera axis (what is wrong and what is right)
"""

# T_Scene_Device =aria_3d_pose_with_dt.data().transform_scene_device    
                    
# cam_x_axis_v2 = np.append(np.array([1, 0, 0]).reshape(3, 1), [1])
# cam_y_axis_v2 = np.append(np.array([0, 1, 0]).reshape(3, 1), [1])
# cam_z_axis_v2 = np.append(np.array([0, 0, 1]).reshape(3, 1), [1])

# Vectors of the camera (scene frame) THIS IS WRONG NEED TO TAKE INTO ACCOUNT THE TRANSLATION
# cam_x_axis_scene_v1 = (T_Scene_Cam.rotation() @ cam_x_axis).reshape(1,3)[0]  # reshape(1,3)[0] == flatten()
# cam_y_axis_scene_v1 = (T_Scene_Cam.rotation() @ cam_y_axis).reshape(1,3)[0]  
# cam_z_axis_scene_v1 = (T_Scene_Cam.rotation() @ cam_z_axis).reshape(1,3)[0]

# Vectors of the camera (scene frame) THIS IS CORRECT
# cam_x_axis_scene_v2 = (T_Scene_Cam.to_matrix() @ cam_x_axis_v2)[0:3] 
# cam_y_axis_scene_v2 = (T_Scene_Cam.to_matrix() @ cam_y_axis_v2)[0:3]
# cam_z_axis_scene_v2 = (T_Scene_Cam.to_matrix() @ cam_z_axis_v2)[0:3]

# # Visualize the vectors of the camera
# args.runrr and log_vector(rr, "camera_x_axis_v1", p_cam_scene[0], cam_x_axis_scene) 
# args.runrr and log_vector(rr, "camera_y_axis_v1", p_cam_scene[0], cam_y_axis_scene)
# args.runrr and log_vector(rr, "camera_z_axis_v1", p_cam_scene[0], cam_z_axis_scene) 

# # Visualize the vectors of the camera - versions
# args.runrr and log_vector(rr, "camera_x_axis_v2", p_cam_scene[0], cam_x_axis_scene_v2)
# args.runrr and log_vector(rr, "camera_y_axis_v2", p_cam_scene[0], cam_y_axis_scene_v2)
# args.runrr and log_vector(rr, "camera_z_axis_v2", p_cam_scene[0], cam_z_axis_scene_v2)

"""
Filtering for debugging
"""

# ## For Debugging
# filtered_past_dots = {gt_provider.get_instance_info_by_id(obj_id).name: past_dots[obj_id] for obj_id in filtered_obj_ids}
# filtered_past_distances = {gt_provider.get_instance_info_by_id(obj_id).name: past_distances[obj_id] for obj_id in filtered_obj_ids}
# filtered_names_ids = {gt_provider.get_instance_info_by_id(obj_id).name: obj_id for obj_id in filtered_obj_ids} 
# filtered_ids_names = {obj_id: gt_provider.get_instance_info_by_id(obj_id).name for obj_id in filtered_obj_ids}
# filtered_duration_time = {gt_provider.get_instance_info_by_id(obj_id).name: visibility_duration[obj_id][-1] for obj_id in filtered_obj_ids if obj_id in visibility_duration} 

"""
time to interaction initial coding
"""

# def interaction_time(self, user_position: np.ndarray, user_velocity: np.ndarray, object_position: np.ndarray) -> float: 
#     """Calculate the time needed for the user to approach an object"""
#     vector_user_object = object_position[:2] - user_position[:2]            # Me: take only the X, Y component
#     distance_user_object = np.linalg.norm(vector_user_object)               
#     user_velocity_xy  = np.linalg.norm(user_velocity[:2])                   # Me: take only the X, Y component because Z is very noisy
    
#     if user_velocity_xy == 0:
#         return float('inf')  # If the user is not moving, return infinity
    
#     time = float(distance_user_object / user_velocity_xy)

#     return time

# def interaction_time_full_3d(self, user_position: np.ndarray, user_velocity: np.ndarray, object_position: np.ndarray) -> float: 
    
#     """Calculate the time to approach the object considering full 3D projection of the velocity"""
    

#     # Calculate displacement vector in 3D
#     displacement_vector = object_position - user_position
#     distance = np.linalg.norm(displacement_vector)  # Magnitude of displacement vector
    
#     # Normalize the displacement vector to get direction
#     displacement_unit_vector = displacement_vector / distance

#     # Project the user's velocity onto the displacement vector
#     velocity_towards_object = np.dot(user_velocity, displacement_unit_vector)  

#     # If velocity towards the object is zero or negative (moving away), return infinite time
#     if velocity_towards_object <= 0:
#         return float('inf')
    
#     # Time to approach = distance / velocity towards the object
#     time_to_approach = distance / velocity_towards_object
    
#     return time_to_approach

# def interaction_time_xy(user_position: np.ndarray, user_velocity: np.ndarray, object_position: np.ndarray) -> float:
#     """Calculate the time to approach the object considering only the XY plane projection"""
#     # Calculate displacement vector in XY plane
#     displacement_vector_xy = object_position[:2] - user_position[:2]
#     distance_xy = np.linalg.norm(displacement_vector_xy)  # Magnitude of displacement in XY plane
    
#     # Normalize the displacement vector
#     displacement_unit_vector_xy = displacement_vector_xy / distance_xy    

#     # Project the user's velocity onto the XY displacement vector
#     velocity_towards_object_xy = np.dot(user_velocity[:2], displacement_unit_vector_xy)

#     # If velocity towards the object is zero or negative (moving away), return infinite time
#     if velocity_towards_object_xy <= 0:
#         return float('inf')
    
#     # Time to approach = distance in XY plane / velocity towards the object in XY plane
#     time_to_approach_xy = distance_xy / velocity_towards_object_xy
    
#     return time_to_approach_xy

# def interaction_time_with_rotation(user_position: np.ndarray, user_velocity: np.ndarray, object_position: np.ndarray, device_rotational_values: np.ndarray) -> float:
#     """Calculate the time to approach the object considering rotational velocity."""
#     # Calculate displacement vector in 3D
#     displacement_vector = object_position - user_position
#     distance = np.linalg.norm(displacement_vector)  # Magnitude of displacement vector
    
#     # Normalize the displacement vector to get direction
#     displacement_unit_vector = displacement_vector / distance

#     # Incorporate the rotational velocity into the velocity vector
#     # Adjust user velocity direction based on rotational velocity
#     adjusted_velocity = user_velocity + device_rotational_values  # Simplified adjustment

#     # Project the adjusted velocity onto the displacement vector
#     velocity_towards_object = np.dot(adjusted_velocity, displacement_unit_vector)

#     # If velocity towards the object is zero or negative (moving away), return infinite time
#     if velocity_towards_object <= 0:
#         return float('inf')
    
#     # Time to approach = distance / velocity towards the object
#     time_to_approach = distance / velocity_towards_object
    
#     return time_to_approach

"""
Visualization of the model
"""
# ==============================================
# Objects Visualization
# ==============================================  

# if args.runrr and args.visualize_objects:

#     for obj_id, dot_product, distance, obj_position_cam, obj_position_scene in zip(
#                                                                             filtered_obj_ids, 
#                                                                             filtered_dot_products,
#                                                                             filtered_distances,
#                                                                             filtered_obj_positions_cam, 
#                                                                             filtered_obj_positions_scene,
#                                                                             # objects_time_approach_list,
#                                                                             ):

#         # instance info 
#         instance_info = gt_provider.get_instance_info_by_id(obj_id)

#         # name
#         object_name = instance_info.name

#         # Handling the object coordinates 
#         """
#         ** bbox_3d.aabb --> represents the minimum and maximum coordinates that define the AABB --> [x_min, y_min, z_min, x_max, y_max, z_max]
#         ** bbox_3d.transform_scene_object is a transformation from οbject coordinate system to scene coordinate system (4x4 matrix)
#         ** AABB - Axis-Aligned Bounding Box ---> 16 points that form 8 pairs (edges), which define a 3D bounding box. Some of these points are repeated.
#         ** OBB  - Oriented bounding box     ---> same dimension with aabb_coords. it will store the values on the new coordinate system
#         """
#         bbox_3d = bboxes3d[obj_id]                                                     
#         aabb_coords = bbox3d_to_line_coordinates(bbox_3d.aabb)                         
#         obb = np.zeros(shape=(len(aabb_coords), 3))                                    
#         for i in range(0, len(aabb_coords)):                                           
#             aabb_pt = aabb_coords[i]                                                  
#             aabb_pt_homo = np.append(aabb_pt, [1])                                    
#             obb_pt = (bbox_3d.transform_scene_object.to_matrix() @ aabb_pt_homo)[0:3]  
#             obb[i] = obb_pt

#         # Prints # 
#         # print(f"Object name: {instance_info.name}, Dot Product: {dot_product}, Distance: {distance}, Position on scene {bbox_3d.transform_scene_object.translation()}")                
#         # print(f"Position on Camera frame: {obj_position_cam}, Position on Scene frame: {obj_position_scene}")
#         # print("-" * 40)  # Print a line of dashes as a separator | # More sophisticated separator)

#         # ==============================================
#         # VECTORS - POSITION / VELOCITY / DISTANCE / PROJECTION  
#         # ==============================================  

#         # POSITION - vectors
#         position_vector_xyz = T_Scene_Device.translation()[0]   
#         position_vector_xz = np.array([position_vector_xyz[0], 0, position_vector_xyz[2]])

#         # VELOCITY - vector in 3D
#         velocity_vector_xyz = T_Scene_Device.rotation().to_matrix() @ user_velocity_device
#         velocity_vector_xz = np.array([velocity_vector_xyz[0], 0, velocity_vector_xyz[2]])
#         velocity_vector_xyz_on_user = velocity_vector_xyz + position_vector_xyz     # End of the velocity vector (scaled by velocity)
#         velocity_vector_xz_on_user = velocity_vector_xz + position_vector_xyz       # Velocity vector in 2D 

#         # DISTANCE vector in 3D from user to object
#         displacement_vector = (obj_position_scene[0] - position_vector_xyz) 
#         displacement_unit_vector = displacement_vector / np.linalg.norm(displacement_vector)
#         displacement_vector_xz = np.array([displacement_vector[0], 0, displacement_vector[2]])
#         displacement_unit_vector_xz = np.array([displacement_unit_vector[0], 0, displacement_unit_vector[2]])

#         # DISTANCE vector in 3D from user to object ON USER
#         displacement_vector_on_user = (obj_position_scene[0] - position_vector_xyz) + position_vector_xyz
#         displacement_unit_vector_on_user = displacement_vector / np.linalg.norm(displacement_vector) + position_vector_xyz
#         displacement_vector_xz_on_user = np.array([displacement_vector[0], 0, displacement_vector[2]]) + position_vector_xyz
#         displacement_unit_vector_xz_on_user = np.array([displacement_unit_vector[0], 0, displacement_unit_vector[2]]) + position_vector_xyz

#         # PROJECTED VELOCITY vector in 3D from user to object  (Project the user's velocity onto the displacement vector)
#         projected_velocity_xyz = np.dot(velocity_vector_xyz, displacement_unit_vector) * displacement_unit_vector 
#         projected_velocity_xz = np.array([projected_velocity_xyz[0], 0, projected_velocity_xyz[2]])
#         projected_velocity_xyz_on_user = projected_velocity_xyz + position_vector_xyz 
#         projected_velocity_xz_on_user = projected_velocity_xz + position_vector_xyz

#         # ==============================================
#         # VISUALISATION ORIGIN
#         # ==============================================  

#         # # POSITION - Visualize the position vector (3D & 2D)
#         # args.runrr and log_vector(rr, "origin_position_xyz", np.array([0,0,0]), position_vector_xyz)
#         # args.runrr and log_vector(rr, "origin_position_ema_xyz", np.array([0,0,0]), user_ema_position)
#         # args.runrr and log_vector(rr, "origin_position_xz", np.array([0,0,0]), position_vector_xz)

#         # # VELOCITY - Visualize the velocity vector as a line in the GUI
#         # args.runrr and log_vector(rr, "origin_velocity_xyz", np.array([0,0,0]), velocity_vector_xyz)
#         # args.runrr and log_vector(rr, "origin_velocity_ema_xyz", np.array([0,0,0]), user_ema_velocity)
#         # args.runrr and log_vector(rr, "origin_velocity_xz", np.array([0,0,0]), velocity_vector_xz)

#         # # PROJECTED VELOCITY - Visualize the velocity vector projected towards the  
#         # args.runrr and log_vector(rr, "origin_projected_velocity_xyz", np.array([0,0,0]), projected_velocity_xyz)
#         # args.runrr and log_vector(rr, "origin_projected_velocity_xz", np.array([0,0,0]), projected_velocity_xz)

#         # # DISTANCE 3D - Visualize the position vector (3D & 2D)
#         # args.runrr and log_vector(rr, "origin_distance_vector_xyz", np.array([0,0,0]), displacement_vector)
#         # args.runrr and log_vector(rr, "origin_distance_unit_vector_xyz", np.array([0,0,0]), displacement_unit_vector)
#         # args.runrr and log_vector(rr, "origin_distance_vector_xz", np.array([0,0,0]), displacement_vector_xz)
#         # args.runrr and log_vector(rr, "origin_distance_unit_vector_xz", np.array([0,0,0]), displacement_unit_vector_xz)

#         # # OBJECT POSITION VECTOR - Visualise the position of the vector in the space 
#         # args.runrr and log_vector(rr, "origin_object_position", np.array([0,0,0]), obj_position_scene[0])
#         # args.runrr and log_vector(rr, "camera_to_object", camera_position_scene, filtered_vector_camera_objects_scene[0] + camera_position_scene)
#         # args.runrr and log_vector(rr, "origin_camera_to_object", np.array([0,0,0]), filtered_vector_camera_objects_scene[0])

#         # ==============================================
#         # VISUALISATION ON USER 
#         # ==============================================  

#         # # VELOCITY - Visualize the velocity vector as a line in the GUI
#         args.runrr and log_vector(rr, "device_velocity_xyz", position_vector_xyz, velocity_vector_xyz_on_user)
#         # args.runrr and log_vector(rr, "device_velocity_xz", position_vector_xyz, velocity_vector_xz_on_user)

#         # # PROJECTED VELOCITY - Visualize the velocity vector projected towards the  
#         # args.runrr and log_vector(rr, f"device_projected_velocity_xyz", position_vector_xyz, projected_velocity_xyz_on_user)
#         # args.runrr and log_vector(rr, f"device_projected_velocity_xz", position_vector_xyz, projected_velocity_xz_on_user)

#         # # DISTANCE 3D - Visualize the position vector (3D & 2D) - ON USER 
#         # args.runrr and log_vector(rr, f"device_distance_vector_xy", position_vector_xyz, displacement_vector_on_user)
#         # args.runrr and log_vector(rr, f"device_distance_unit_vector_xyz", position_vector_xyz, displacement_unit_vector_on_user)
#         # args.runrr and log_vector(rr, f"device_distance_vector_xz", position_vector_xyz, displacement_vector_xz_on_user)
#         # args.runrr and log_vector(rr, f"device_distance_unit_vector_xz", position_vector_xyz, displacement_unit_vector_xz_on_user)

#         # OBJECT LINE - Visualize the line from camera to object
#         # args.runrr and log_object_line(rr, instance_info, position_vector_xyz, obj_position_scene[0])
#         # args.runrr and log_object_line(rr, instance_info, camera_position_scene[0], obj_position_scene[0])

#         # OBJECT BOUNDING BOX - Visualize the object
#         args.runrr and log_object(rr, instance_info, obb)

#         # ==============================================
#         # VISUALISATION FRAMES
#         # ==============================================    

#         # # CAMERA Z-AXIS ONLY ROTATION TO SCENE
#         # cam_z_axis_rotation = T_Scene_Cam.rotation().to_matrix() @ cam_z_axis     
#         # args.runrr and log_vector(rr, "origin_camera_z_axis_rotation_only", np.array([0,0,0]), cam_z_axis_rotation)
#         # args.runrr and log_vector(rr, "origin_camera_z_axis", np.array([0,0,0]), cam_z_axis_scene)

#         # # CAMERA
#         # args.runrr and log_vector(rr, "camera_x_axis", camera_position_scene[0], cam_x_axis_scene)
#         # args.runrr and log_vector(rr, "camera_y_axis", camera_position_scene[0], cam_y_axis_scene)
#         # args.runrr and log_vector(rr, "camera_z_axis", camera_position_scene[0], cam_z_axis_scene)

#         # # DEVICE
#         # args.runrr and log_vector(rr, "device_x_axis", position_vector_xyz, device_x_axis_scene)
#         # args.runrr and log_vector(rr, "device_y_axis", position_vector_xyz, device_y_axis_scene)
#         # args.runrr and log_vector(rr, "device_z_axis", position_vector_xyz, device_z_axis_scene)

#         # # WORLD
#         # args.runrr and log_vector(rr, "world_x_axis", np.array([0,0,0]), world_x_axis)
#         # args.runrr and log_vector(rr, "world_y_axis", np.array([0,0,0]), world_y_axis)
#         # args.runrr and log_vector(rr, "world_z_axis", np.array([0,0,0]), world_z_axis)

#         # Add to previous_obj_ids to clear in the next timestep
#         previous_obj_names.add(instance_info.name)
#         previous_obj_ids.add(obj_id)

#         if gt_provider.get_instance_info_by_id(object_id).name == "ChoppingBoard":
#             print('stop')


""""
Debugging process regarding the time to approach & VISUALIZATION
"""

# DEBUGGING PURPOSES
# if object_name in ["ChoppingBoard", "KitchenKnife", "WoodenSpoon", "WoodenFork", "Donut_B", "Cereal_Anon", "DinoToy", "WoodenToothbrushHolder", "Cracker_Anon", "BlackCeramicMug"]:

#     # ==============================================
#     # 3D MOTION 
#     # ==============================================

#     # DISTANCE

#     displacement_vector_xyz = (filtered_obj_positions_scene[index][0] - user_ema_position)
#     distance_xyz = np.linalg.norm(displacement_vector_xyz)
#     displacement_unit_vector_xyz = displacement_vector_xyz / np.linalg.norm(displacement_vector_xyz)

#     # VELOCITY 
#     velocity_xyz = T_Scene_Device.rotation().to_matrix() @ user_velocity_device

#     # PROJECTED VELOCITY
#     projected_velocity_xyz = np.dot(velocity_xyz, displacement_unit_vector_xyz) * displacement_unit_vector_xyz 
#     speed_xyz = np.linalg.norm(projected_velocity_xyz)
#     time_xyz = distance_xyz / speed_xyz

#     # ==============================================
#     # PLANAR MOTION ON XZ PLANE
#     # ==============================================

#     # DISTANCE
#     displacement_vector_xz = np.array([displacement_vector_xyz[0], 0, displacement_vector_xyz[2]])
#     displacement_unit_vector_xz = displacement_vector_xz / np.linalg.norm(displacement_vector_xz)
#     distance_xz = np.linalg.norm(displacement_vector_xz)

#     # VELOCITY 
#     velocity_xz = np.array([velocity_xyz[0], 0, velocity_xyz[2]])

#     # ==============================================
#     # 2 WAYS OF CALCULATING THE TIME
#     # ==============================================

#     # 1ST WAY
#     projected_velocity_xz_v1 = np.array([projected_velocity_xyz[0], 0, projected_velocity_xyz[2]])
#     speed_xz_v1 = np.linalg.norm(projected_velocity_xz_v1)
#     time_xz_v1 = distance_xz / speed_xz_v1

#     # 2ND WAY
#     projected_velocity_xz_v2 = np.dot(velocity_xz, displacement_unit_vector_xz) * displacement_unit_vector_xz 
#     speed_xz_v2 = np.linalg.norm(projected_velocity_xz_v2)
#     time_xz_v2 = distance_xz / speed_xz_v2

#     # ==============================================
#     # PRINTS
#     # ==============================================

#     print("-" * 40)  # Print a line of dashes as a separator | # More sophisticated separator)
#     print('')
#     print(f"\t Position of {object_name} in the space is: {filtered_obj_positions_scene[index][0]}")
#     print(f"\t Position of user in the space is: {user_ema_position}")
#     print('')
#     print(f"\t Displacement from user to the {object_name} is : {displacement_vector_xyz}")
#     print(f"\t Distance is {distance_xyz}")
#     print('')
#     print(f"\t Velocity of user in the space is: {velocity_xyz}")
#     print(f"\t Projected Velocity of user in the space is: {projected_velocity_xyz}")
#     print(f"\t Speed is {speed_xyz}")
#     print(f"\t Time is {time_xyz}")
#     print('')
#     print(f"\t Displacement from user to the {object_name} in xz plane is : {displacement_vector_xz}")
#     print(f"\t Distance in 2D is {distance_xz}")
#     print('')
#     print(f"\t Velocity of user in 2D is: {velocity_xz}")  
#     print(f"\t Projected Velocity v1 of user in the space is: {projected_velocity_xz_v1}")
#     print(f"\t Speed in 2D is {speed_xz_v1}")
#     print(f"\t Time is {time_xz_v1}")
#     print('')
#     print(f"\t Projected Velocity v2 of user in the space is: {projected_velocity_xz_v2}")
#     print(f"\t Speed in 2D is {speed_xz_v2}")
#     print(f"\t Time is {time_xz_v2}")
#     print(f"\t Time as calculated using the function of STATS class", {object_time_xz})

#     if args.runrr and args.visualize_objects:

#         # ==============================================
#         # VISUALIZATION
#         # ==============================================

#         # OBJECT LINE
#         log_object_line(rr, gt_provider.get_instance_info_by_id(object_id) , user_ema_position, filtered_obj_positions_scene[index][0])

#         # OBJECT DISTANCE AND DISPLACEMENT
#         log_vector(rr, f"debugging_distance_user_{object_name}", user_ema_position, filtered_obj_positions_scene[index][0])      # line from user to object
#         log_vector(rr, f"debugging_distance_displacement_{object_name}", np.array([0,0,0]), displacement_vector_xyz)             # line from orgin with same direction and magnitude as from user to object
#         log_vector(rr, f"debugging_distance_displacement_{object_name}_xz", np.array([0,0,0]) , displacement_vector_xz)
#         log_vector(rr, f"debugging_distance_unit_displacement_{object_name}", np.array([0,0,0]) , displacement_unit_vector_xyz)
#         log_vector(rr, f"debugging_distance_unit_displacement_{object_name}_xz", np.array([0,0,0]) , displacement_unit_vector_xz)

#         # PROJECTED VELOCITY
#         log_vector(rr, "debugging_projected_velocity", np.array([0,0,0]), projected_velocity_xyz) 
#         log_vector(rr, "debugging_projected_velocity_xz_v1", np.array([0,0,0]), projected_velocity_xz_v1) 
#         log_vector(rr, "debugging_projected_velocity_xz_v2", np.array([0,0,0]), projected_velocity_xz_v2) 

#         # VELOCITY
#         log_vector(rr, "debugging_velocity", np.array([0,0,0]),  (T_Scene_Device.rotation().to_matrix() @ user_velocity_device)) # rotated velocity
#         log_vector(rr, "debugging_velocity_2d", np.array([0,0,0]), np.array([(T_Scene_Device.rotation().to_matrix() @ user_velocity_device)[0], 0, (T_Scene_Device.rotation().to_matrix() @ user_velocity_device)[2]]))
#         log_vector(rr, "debugging_device_velocity", user_ema_position,  (T_Scene_Device @ user_velocity_device)[:,0])

#         # CAMERA Z-AXIS ONLY ROTATION TO SCENE
#         cam_z_axis_rotation = (T_Scene_Cam.rotation().to_matrix() @ cam_z_axis)[:,0]
#         cam_z_axis_rotation_xz =  np.array([cam_z_axis_rotation[0], 0, cam_z_axis_rotation[2]]) 
#         log_vector(rr, "origin_camera_z_axis_rotation_only", np.array([0,0,0]), cam_z_axis_rotation)
#         log_vector(rr, "origin_camera_z_axis_rotation_only_xz", np.array([0,0,0]), cam_z_axis_rotation_xz)
#         log_vector(rr, "origin_camera_z_axis", np.array([0,0,0]), cam_z_axis_scene)

#         # CAMERA
#         log_vector(rr, "camera_x_axis", camera_position_scene[0], cam_x_axis_scene)
#         log_vector(rr, "camera_y_axis", camera_position_scene[0], cam_y_axis_scene)
#         log_vector(rr, "camera_z_axis", camera_position_scene[0], cam_z_axis_scene)

#         # WORLD
#         log_vector(rr, "world_x_axis", np.array([0,0,0]), world_x_axis)
#         log_vector(rr, "world_y_axis", np.array([0,0,0]), world_y_axis)
#         log_vector(rr, "world_z_axis", np.array([0,0,0]), world_z_axis)
                    
                    
"""
LLM activation conditions 
"""

# 1. we activate LLM only if there is object that satisfy the criteria of high dot thrshold
# 2. we activate LLM only if there is object that satisfy the criteria of the distance threshold
# 3. we activate LLM only if there is object that is approachable in less than the time threshold 
# 4. we activate LLM only if there is object that has high visibility 
# 5. we activate LLM only if there is no object that is approachable in less that 2 seconds but this remains consistent for 2 seconds (means is around this object)
            
#     if (high_dot_counts
#     and low_distance_counts
#     and less_than_2_seconds_dict
#     and high_duration
#     ):


"""
Objects with time less than threhsold and with high dot counts and low distanc
"""

# Check if object meets both high dot counts, low distance counts, and has visibility over 1 second

        # if (object_id in filtered_high_dot_counts and 
        #     object_id in filtered_low_distance_counts and 
        #     filtered_duration[object_id][-1][1] > 1):
        
"""
time to interaction
"""

# def time_to_interaction(user_position: np.ndarray , user_velocity: np.ndarray, object_position: np.ndarray):
#     """Calculate the time needed for the user to approach an object"""
#     vector_user_object = object_position[:2] - user_position[:2]
#     distance_user_object = np.linalg.norm(vector_user_object)     # Me: take only the X, Y component
#     user_velocity_xy = np.linalg.norm(user_velocity[:2])          # Me: take only the X, Y component because Z is very noisy
    
#     if user_velocity_xy == 0:
#         return float('inf')  # If the user is not moving, return infinity
    
#     time = float(distance_user_object / user_velocity_xy)
    
#     return time


"""
find the correspondances 
"""

# def adjust_predictions_with_gt_and_fn_mine(self):
            
#         """ 
#         Match LLM predictions with GT values 
        
#         Cases: 
#             1. LLM time and prediction does not correspond to any GT interaction. (FALSE POSITIVE)
#                 - Example: Two consecutive LLM times are less than current gt time which means that 1st llm time does not correspond to any GT value
                
#             2. LLM time and prediction corresponds to two GT values. 
#                 - Example: Current LLM time is less than GT time but next LLM time is bigger than next GT tine which means that current LLM time corresponds to two consecutive GT predictions
                
#             3. LLM time and prediction correspomnds to one GT value
#                 - Example: Current LLM time is less than GT time and next LLM time is less than next GT time       
#         """
        
#         # Initialize variables
#         llm_times = sorted(map(float, self.llm_predictions.keys()))
#         gt_times  = sorted(map(float, self.ground_truth.keys()))

#         # Create the iterators 
#         gt_iter = iter(gt_times)
#         current_gt_time = next(gt_iter, None)
#         next_gt_time = next(gt_iter, None)

#         # Iterate through the LLM predictions
#         for i, current_llm_time in enumerate(llm_times):

#             # check if for the first gt and subsequent gt value are less than the 1st LLM there is 
#             if current_gt_time < current_llm_time:
#                 self.Fn += 1     # there should an LLM activation before the first gt ground 
#                 current_gt_time = next_gt_time
#                 next_gt_time = next(gt_iter, None)
#                 if current_gt_time < current_llm_time:
#                     self.Fn += 1     # there should an LLM activation before the first gt ground 
#                     current_gt_time = next_gt_time
#                     next_gt_time = next(gt_iter, None)

#             # update the llm time
#             next_llm_time = llm_times[i + 1] if i + 1 < len(llm_times) else None

#             # check if two consequtive llm times are less than current gt time
#             if next_llm_time is not None and next_llm_time < current_gt_time:
#                 self.Fp += 1
#                 continue
            
#             # if next llm time is bigger than next gt timer there are two options ---> means we have one gt without correspoding llm 
#             """
#             current llm less than current gt time 
            
#             Two options: 
            
#                 A.  Next LLM time is larger than Next gt time

#                     1. either FN
#                     2. one llm for two gt 

#                     this depends on the time difference

#                 B. Next LLM time is less then Next gt time
                
#                     1. LLM corresponds to Gt 
#             """

#             matched_ground_truths =[]
            
#             #  A.  Next LLM time is larger than Next gt time
#             if current_llm_time <= current_gt_time and next_llm_time > next(gt_iter, None):
                
#                 time_diff = next(gt_iter, None) - current_gt_time
        
#                 # 1. either FN 
#                 if next_llm_time > next(gt_iter, None) and time_diff > 1: 
#                     self.Fn += 1   

#                 # 2. or prediction refers to double gt    
#                 if  next_llm_time > next(gt_iter, None) and time_diff < 1: 
#                     matched_ground_truths.append((current_gt_time, self.ground_truth[current_gt_time]))
#                     matched_ground_truths.append(( next(gt_iter, None), self.ground_truth[ next(gt_iter, None)]))
#                     current_gt_time = next_gt_time
#                     next_gt_time = next(gt_iter, None)
#                     continue
            
#             # B. Next LLM time is less then Next gt time
#             if current_llm_time <= current_gt_time and next_llm_time <= next(gt_iter, None):
#                 matched_ground_truths.append((current_gt_time, self.ground_truth[current_gt_time]))
#                 self.correspondences.append((current_llm_time, self.llm_predictions[current_llm_time], matched_ground_truths))
#                 current_gt_time = next_gt_time
#                 next_gt_time = next(gt_iter, None)
        
#         return self.correspondences

"""
handle the correspondances 
"""

# for llm_time, prediction, correspondence in self.correspondences:

#             # Case where there was no actual interaction but LLM made a prediction
#             # if correspondence == "no actual interaction":  
#             #     self.Fp += 1  # This should count as a false positive
#             #     continue  # Skip to the next correspondence

#             # Loop over the gt values that correspond to one LLM prediction
#             for gt_time, gt_object in correspondence:
                
#                 # if an LLM activation does not corresponds to "no actual interaction" --> actual prediction
#                 self.total_actual_interactions += 1
                
#                 # Flag to check if a TP is found in this correspondence
#                 # match_found = False  
                
#                 if gt_object in prediction:
#                     self.Tp += 1
#                     match_found = True
#                     continue  # Break as soon as we find a match, considering it a TP
                
#                 # If no match was found, this is a false positive
#                 elif not match_found:   
#                     self.Fp +=1

"""
calculate the TP 
"""

# def adjust_predictions_with_gt(self):
        
#         """ 
#         Match LLM predictions with GT values 
        
#         Cases: 
#             1. LLM time and prediction does not correspond to any GT value. (FALSE POSITIVE)
#                 - Example: Two consecutive LLM times are less than current gt time which means that 1st llm time does not correspond to any GT value
                
#             2. LLM time and prediction corresponds to two GT values. 
#                 - Example: Current LLM time is less than GT time but next LLM time is bigger than next GT tine which means that current LLM time corresponds to two consecutive GT predictions
                
#             3. LLM time and prediction correspomnds to one GT value
#                 - Example: Current LLM time is less than GT time and next LLM time is less than next GT time       
#         """
        
#         # Initialize variables
#         llm_times = sorted(map(float, self.llm_predictions.keys()))
#         gt_times  = sorted(map(float, self.ground_truth.keys()))

#         # Create the iterators 
#         gt_iter = iter(gt_times)
#         current_gt_time = next(gt_iter, None)
#         next_gt_time = next(gt_iter, None)

#         # Iterate through the LLM predictions
#         for i, llm_time in enumerate(llm_times):
#             if current_gt_time is None:
#                 self.correspondences.append((llm_time, self.llm_predictions[llm_time], "no actual interaction"))
#                 continue

#             next_llm_time = llm_times[i + 1] if i + 1 < len(llm_times) else None

#             if next_llm_time is not None and next_llm_time < current_gt_time:
#                 self.correspondences.append((llm_time, self.llm_predictions[llm_time], "no actual interaction"))
#                 continue

#             matched_ground_truths = []

#             if llm_time <= current_gt_time:
#                 matched_ground_truths.append((current_gt_time, self.ground_truth[current_gt_time]))

#                 if next_gt_time and next_llm_time and next_llm_time > next_gt_time:
#                     matched_ground_truths.append((next_gt_time, self.ground_truth[next_gt_time]))
#                     current_gt_time = next_gt_time
#                     next_gt_time = next(gt_iter, None)

#                 self.correspondences.append((llm_time, self.llm_predictions[llm_time], matched_ground_truths))
#                 current_gt_time = next_gt_time
#                 next_gt_time = next(gt_iter, None)

#         return self.correspondences

"""
recall 
"""
# self.recall = self.Tp / self.total_ground_truth if self.total_ground_truth else 0  # self.recall = self.Tp / (self.Tp + self.Fn) if (self.Tp + self.Fn) else 0


"""
run the multiprocess 
"""

# # ==============================================
# # Run for different parameter combinations in parallel
# # ==============================================
# if __name__ == "__main__":

#     start_time = time.time()

#     with mp.Pool(mp.cpu_count()) as pool:
#         pool.map(run_simulation, param_combinations)
    
#     # with mp.Pool(mp.cpu_count()) as pool:
#     #     chunksize = len(param_combinations) // mp.cpu_count()
#     #     pool.map(run_simulation, param_combinations, chunksize=chunksize)

#     end_time = time.time()
#     print(f"Total time taken: {end_time - start_time:.2f} seconds")


"""
plot functions
"""

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import os

# def plot_results(results, project_path):
#     sequences = list(set([result['sequence'] for result in results]))

#     # Create plots for each sequence
#     for sequence in sequences:
#         sequence_results = [r for r in results if r['sequence'] == sequence]

#         plot_folder = os.path.join(project_path, 'plots', sequence)
#         os.makedirs(plot_folder, exist_ok=True)

#         # Extract the metric values
#         overall_model_accuracies = [r['model_overall_accuracy'] for r in sequence_results]
#         precisions = [r['precision'] for r in sequence_results]
#         recalls = [r['recall'] for r in sequence_results]
#         llm_activation_sensitivities = [r['llm_activation_sensitivity'] for r in sequence_results]
#         llm_interaction_accuracies = [r['llm_interaction_accuracy'] for r in sequence_results]
#         parameter_names = [r['parameters'] for r in sequence_results]

#         # Refine parameter names (shorten if needed)
#         parameter_names_short = [param.replace("time_", "t_").replace("highdot_", "hd_").replace("dist_", "d_") for param in parameter_names]

#         # Plotting
#         plt.figure(figsize=(12, 8))

#         plt.plot(parameter_names_short, overall_model_accuracies, label='Overall Model Accuracy', marker='o', color='blue')
#         plt.plot(parameter_names_short, precisions, label='Precision', marker='x', color='orange')
#         plt.plot(parameter_names_short, recalls, label='Recall', marker='s', color='green')
#         plt.plot(parameter_names_short, llm_activation_sensitivities, label='LLM Activation Sensitivity', marker='^', color='red')
#         plt.plot(parameter_names_short, llm_interaction_accuracies, label='LLM Interaction Accuracy', marker='d', color='purple')

#         plt.title(f'Metrics for Sequence: {sequence}', fontsize=16)
#         plt.xlabel('Parameter Combination', fontsize=14)
#         plt.ylabel('Metric Values', fontsize=14)
#         plt.xticks(rotation=45, ha="right")
#         plt.grid(True)
#         plt.legend(loc='best', fontsize=12)
#         plt.tight_layout()

#         # Save the plot with a descriptive name
#         plt.savefig(os.path.join(plot_folder, 'metrics_plot.png'), format='png')
#         plt.close()

# def plot_bar_metrics(results, project_path):
#     sequences = list(set([result['sequence'] for result in results]))

#     for sequence in sequences:
#         sequence_results = [r for r in results if r['sequence'] == sequence]

#         plot_folder = os.path.join(project_path, 'plots', sequence)
#         os.makedirs(plot_folder, exist_ok=True)

#         # Extract the metric values
#         overall_model_accuracies = [r['model_overall_accuracy'] for r in sequence_results]
#         precisions = [r['precision'] for r in sequence_results]
#         recalls = [r['recall'] for r in sequence_results]
#         parameter_names = [r['parameters'] for r in sequence_results]

#         # Create a bar plot
#         index = np.arange(len(parameter_names))
#         bar_width = 0.2

#         plt.figure(figsize=(12, 8))
#         plt.bar(index, overall_model_accuracies, bar_width, label='Accuracy', color='blue')
#         plt.bar(index + bar_width, precisions, bar_width, label='Precision', color='orange')
#         plt.bar(index + 2 * bar_width, recalls, bar_width, label='Recall', color='green')

#         plt.xlabel('Parameter Combination', fontsize=14)
#         plt.ylabel('Metric Values', fontsize=14)
#         plt.xticks(index + bar_width, parameter_names, rotation=45, ha="right")
#         plt.title(f'Comparison of Metrics for Sequence: {sequence}', fontsize=16)
#         plt.legend()
#         plt.tight_layout()

#         plt.savefig(os.path.join(plot_folder, 'bar_metrics_plot.png'), format='png')
#         plt.close()
    
# def plot_precision_vs_recall(results, project_path):
#     sequences = list(set([result['sequence'] for result in results]))

#     for sequence in sequences:
#         sequence_results = [r for r in results if r['sequence'] == sequence]

#         plot_folder = os.path.join(project_path, 'plots', sequence)
#         os.makedirs(plot_folder, exist_ok=True)

#         precisions = [r['precision'] for r in sequence_results]
#         recalls = [r['recall'] for r in sequence_results]
#         accuracies = [r['model_overall_accuracy'] for r in sequence_results]
#         parameter_names = [r['parameters'] for r in sequence_results]

#         plt.figure(figsize=(10, 7))
#         scatter = plt.scatter(recalls, precisions, s=[a * 100 for a in accuracies], c=accuracies, cmap='viridis', alpha=0.7)

#         plt.title(f'Precision vs Recall for Sequence: {sequence}', fontsize=16)
#         plt.xlabel('Recall', fontsize=14)
#         plt.ylabel('Precision', fontsize=14)
#         plt.colorbar(scatter, label='Overall Accuracy')

#         for i, param in enumerate(parameter_names):
#             plt.annotate(param, (recalls[i], precisions[i]), fontsize=10)

#         plt.tight_layout()
#         plt.savefig(os.path.join(plot_folder, 'precision_vs_recall.png'), format='png')
#         plt.close()

# def plot_tp_fp_pie(results):
#     sequences = list(set([result['sequence'] for result in results]))

#     for sequence in sequences:
#         sequence_results = [r for r in results if r['sequence'] == sequence]

#         plot_folder = os.path.join(project_path, 'plots', sequence)
#         os.makedirs(plot_folder, exist_ok=True)

#         for result in sequence_results:
#             labels = ['True Positives', 'False Positives']
#             sizes = [result['Tp'], result['Fp']]
#             colors = ['green', 'red']
            
#             # Check if sizes contain NaN or invalid values
#             if any(np.isnan(sizes)) or any(np.isinf(sizes)):
#                 print(f"Skipping pie chart in {sequence} and for {result['parameters']} due to NaN or Inf values in Tp/Fp")
#                 continue

#             if sum(sizes) == 0:
#                 print(f"Skipping pie chart in {sequence} and for {result['parameters']} as both Tp and Fp are zero")
#                 continue

#             plt.figure(figsize=(6, 6))
#             plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
#             plt.title(f'TP vs FP for {result["parameters"]}', fontsize=16)
#             plt.tight_layout()

#             plt.savefig(os.path.join(plot_folder, f'tp_fp_pie_{result["parameters"]}.png'), format='png')
#             plt.close()

# def plot_combined_metrics(results):
#     sequences = list(set([result['sequence'] for result in results]))

#     for sequence in sequences:
#         sequence_results = [r for r in results if r['sequence'] == sequence]

#         plot_folder = os.path.join(project_path, 'plots', sequence)
#         os.makedirs(plot_folder, exist_ok=True)

#         overall_model_accuracies = [r['model_overall_accuracy'] for r in sequence_results]
#         recalls = [r['recall'] for r in sequence_results]
#         parameter_names = [r['parameters'] for r in sequence_results]

#         plt.figure(figsize=(10, 6))
#         plt.plot(parameter_names, overall_model_accuracies, label='Accuracy', marker='o', color='blue')
#         plt.plot(parameter_names, recalls, label='Recall', marker='x', color='green')

#         plt.title(f'Accuracy and Recall for Sequence: {sequence}', fontsize=16)
#         plt.xlabel('Parameter Combination', fontsize=14)
#         plt.ylabel('Metric Values', fontsize=14)
#         plt.xticks(rotation=45, ha="right")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()

#         plt.savefig(os.path.join(plot_folder, 'combined_metrics.png'), format='png')
#         plt.close()


"""
async retry
"""

# def retry_with_exponential_backoff(
#     initial_delay: float = 1,
#     exponential_base: float = 2,
#     jitter: bool = True,
#     max_retries: int = 10,
#     errors: tuple = (
#         APIStatusError,
#         RateLimitError,
#         Timeout,
#     ),
# ):
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             num_retries = 0
#             delay = initial_delay

#             while True:
#                 try:
#                     return func(*args, **kwargs)
#                 except errors as e:
#                     num_retries += 1
#                     if num_retries > max_retries:
#                         print(f"Maximum retries ({max_retries}) exceeded.")
#                         raise

#                     delay_with_jitter = delay * (1 + jitter * random.random())
#                     print(f"Error: {e}. Retrying in {delay_with_jitter:.2f} seconds.")
#                     time.sleep(delay_with_jitter)
#                     delay *= exponential_base
#                 except Exception as e:
#                     raise e
#         return wrapper
#     return decorator

"""
run async processes 
"""

# def process_param_combinations(params_batch, result_queue, data):
#     async def run_all_tasks():
#         # Create tasks for each parameter combination in the batch
#         tasks = [run_algorithm_async(params, result_queue, data) for params in params_batch]
#         await asyncio.gather(*tasks)

#     # Run the asyncio loop in each process
#     asyncio.run(run_all_tasks())

# def process_param_combinations(params_batch, data, rate_limiter):
#     async def run_all_tasks():
#         # Create tasks for each parameter combination in the batch
#         tasks = [run_algorithm_async(params, data, rate_limiter) for params in params_batch]
#         await asyncio.gather(*tasks)

#     # Run the asyncio loop in each process
#     asyncio.run(run_all_tasks())

# # This function runs execute_algorithm asynchronously for each parameter combination
# async def run_algorithm_async(parameters, data, rate_limiter):
#     await execute_algorithm(parameters, data, rate_limiter)


"""
retry_with_exponential_backoff
"""
# def retry_with_exponential_backoff(
#     initial_delay: float = 1,
#     exponential_base: float = 2,
#     jitter: bool = True,
#     max_retries: int = 10,
#     errors: tuple = (
#         APIStatusError,
#         RateLimitError,
#         Timeout,
#     ),
# ):
#     def decorator(func):
#         async def wrapper(*args, **kwargs):
#             num_retries = 0
#             while True:
#                 try:
#                     return func(*args, **kwargs)
#                 except errors as e:
#                     num_retries += 1
#                     if num_retries > max_retries:
#                         print(f"Maximum retries ({max_retries}) exceeded.")
#                         raise
                    
#                     delay = initial_delay * exponential_base ** (num_retries - 1)
#                     if jitter:
#                         delay *= random.uniform(0.5, 1.5) # add jitter 
#                     print(f"Rate limit exceeded. Retrying in {delay:.2f} seconds...")
#                     await asyncio.sleep(delay)
#                 except Exception as e:
#                     raise e
#         return wrapper
#     return decorator

# # @sleep_and_retry
# # @limits(calls=500, period=ONE_MINUTE) # this is used if I don't want to use the custom function. 

"""
save results
"""

# def save_results(objects_possibility_dict, rationale_dict, predictions_dict, goals_dict, project_path, sequence_path, parameter_folder_name):
#     # Define the path for saving results
#     results_folder = os.path.join(project_path, 'results', sequence_path, parameter_folder_name)
#     os.makedirs(results_folder, exist_ok=True)

#     # Combine the results into a single dictionary structure
#     results_data = {
#         "objects_possibility_dict": objects_possibility_dict,
#         "rationale_dict": rationale_dict,
#         "predictions_dict": predictions_dict,
#         "goals_dict": goals_dict
#     }

#     # Save the combined data as a JSON file for this specific parameter configuration
#     results_file_path = os.path.join(results_folder, f'results_{parameter_folder_name}.json')
#     with open(results_file_path, 'w') as result_file:
#         json.dump(results_data, result_file, indent=4)
    
#     print(f"Results saved in {results_file_path}")


"""
change of pose - object ground truth 
"""

# # Create a unique key for each movement period based on object name and the last recorded start time
# if name not in movement_time_dict:
#     # Initialize a new movement period for the object
#     unique_key = f"{name}_{current_time_s}"
#     user_object_position[unique_key] = [np.array([user_ema_position[0], 0, user_ema_position[2]])]
#     user_object_movement[unique_key] = 0
#     movement_time_dict[name] = {"start_time": current_time_s, "end_time": None}
# else:
#     # Update the existing movement period's end time if the object is already moving
#     last_entry_key = list(user_object_position.keys())[-1]  # Get the last key for the current object
#     if last_entry_key.startswith(name):
#         user_object_position[last_entry_key].append(np.array([user_ema_position[0], 0, user_ema_position[2]]))
#         movement_vector_xz = user_object_position[last_entry_key][-1] - user_object_position[last_entry_key][-2]
#         user_object_movement[last_entry_key] += np.linalg.norm(movement_vector_xz)
#         movement_time_dict[name]["end_time"] = current_time_s
#     else:
#         # Start a new movement period if the object has stopped and started again
#         unique_key = f"{name}_{current_time_s}"
#         user_object_position[unique_key] = [np.array([user_ema_position[0], 0, user_ema_position[2]])]
#         user_object_movement[unique_key] = 0
#         movement_time_dict[name] = {"start_time": current_time_s, "end_time": None}

# # Add to the original objects that moved dictionary if it's a new period
# if name not in previous_moved_names or last_entry_key != unique_key:
#     original_objects_that_moved_dict[current_time_s] = name
#     previous_moved_names.append(name)
#     indexes_of_objects_that_moved.append(indexes_activation[i])
#     print(f"\tObjects that have been moved so far: {original_objects_that_moved_dict}")
#     print(f"\tUser motion while aforementioned object is moving: {user_object_movement}")

#  for i, (name, index) in enumerate(zip(objects_that_moved_names, indexes_activation)):
                
#                 """
#                 1. loop over the objects that moved based on pose change and the list of respective indexes calculated from the object names list 
#                 2. store the distance of the user from the object
#                 3. store the position of the user and the movement of the user for the object that has been moved 
#                 4. check if in the previous step the object has been moved in order to avoid insert it in the dictionary original_objects_that_moved_dict  
#                 """
#                 # work with the ground truth 
#                 if name not in user_object_movement:
#                     user_object_position[name] = np.array([user_ema_position[0], 0, user_ema_position[2]]) 
#                     user_object_movement[name] = 0
#                 else:
#                     user_object_position[name].append(np.array([user_ema_position[0], 0, user_ema_position[2]])) # only xz plane
#                     movement_vector_xz = user_object_position[name][-1] - user_object_position[name][-2]
#                     user_object_movement[name] += np.linalg.norm(movement_vector_xz) 

#                 # fill the ground truth 
#                 if name not in previous_moved_names:
#                     original_objects_that_moved_dict[current_time_s] = name
#                     previous_moved_names = objects_that_moved_names 
#                     indexes_of_objects_that_moved.append(indexes_activation[i])
#                     print(f"\tObjects that have been moved so far: {original_objects_that_moved_dict}")
#                     print(f"\tUser motion while aforementioned object is moving: {user_object_movement}")

#                 # Track the start and end times of the object movement
#                 if name not in movement_time_dict:
#                     movement_time_dict[name] = {"start_time": current_time_s, "end_time": None} # Me: Object starts moving, record the start time
#                 else:
#                     movement_time_dict[name]["end_time"] = current_time_s                       # Me: Object is already moving, update the end time

#                 # Track the start and end times of object movement
#                 if name not in movement_time_dict_list:
#                     # Initialize with start times
#                     movement_time_dict_list[name] = {"start_times": [current_time_s], "end_times": []}
#                 else:
#                     # Append a new start time if the last recorded movement is complete
#                     if len(movement_time_dict_list[name]["end_times"]) == len(movement_time_dict_list[name]["start_times"]):
#                         movement_time_dict_list[name]["start_times"].append(current_time_s)

#         # Separate loop to handle end times for objects no longer moving
#         for name in movement_time_dict_list:
#             if name not in objects_that_moved_names:
#                 # Append the end time if the object has stopped moving and it hasn't been recorded yet
#                 if len(movement_time_dict_list[name]["end_times"]) < len(movement_time_dict_list[name]["start_times"]):
#                     movement_time_dict_list[name]["end_times"].append(current_time_s)

"""
Parameter combinations
"""

# time_thresholds = [2] # [1, 2, 3]                       
# avg_dot_threshold_highs = [0.7]                         
# avg_dot_threshold_lows = [0.2]                          
# avg_distance_threshold_highs = [3]                       
# avg_distance_threshold_lows = [1]                       
# high_dot_thresholds = [0.7, 0.8, 0.9]                 # [0.5, 0.6, 0.7, 0.8, 0.9]                 
# distance_thresholds = [1.5, 2, 2.5]                        
# high_dot_counters_threshold = [15, 30, 45, 60, 75, 90]  
# distance_counters_threshold = [15, 30, 45, 60, 75, 90]  
# variables_window_times = [3.0]                          
# minimum_time_deactivated = [2.0]                        
# maximum_time_deactivated = [5.0]                        
# user_relative_movement = [2.0]                          
# object_percentage_overlap = [0.7]           

# time_thresholds = [1.5] # [3, 4, 5, 6]
# avg_dot_threshold_highs = [0.7]
# avg_dot_threshold_lows = [0.2]
# avg_distance_threshold_highs = [3]
# avg_distance_threshold_lows = [1]
# high_dot_thresholds = [0.7, 0.8, 0.9] # [0.6, 0.7, 0.8, 0.9, 0.95]
# distance_thresholds = [1.5, 2, 2.5]   # [1, 1.5, 2, 2.5, 3, 3.5] 
# high_dot_counters_threshold = [15, 30, 45, 60, 75, 90] # [60, 75] 
# distance_counters_threshold = [15, 30, 45, 60, 75, 90] # [30] 
# variables_window_times = [3.0]
# minimum_time_deactivated = [2.0] # [1.0 ,2.0, 3.0]
# maximum_time_deactivated = [5.0] # [3.0, 5.0, 6.0] 
# user_relative_movement = [2.0] # [1.0, 2.0, 3.0]
# object_percentage_overlap = [0.7]

# time_thresholds = [1.5] # [3, 4, 5, 6]
# avg_dot_threshold_highs = [0.7]
# avg_dot_threshold_lows = [0.2]
# avg_distance_threshold_highs = [3]
# avg_distance_threshold_lows = [1]
# high_dot_thresholds = [0.9, 0.95]
# distance_thresholds = [1.5 , 3] # [1.5, 2, 2.5]
# high_dot_counters_threshold = [60, 75] #, 90] # [15, 30, 45, 
# distance_counters_threshold = [30] # [15, 30, 45] #, 60, 75, 90]
# variables_window_times = [3.0]
# minimum_time_deactivated = [1.0 ,2.0, 3.0]
# maximum_time_deactivated = [3.0, 5.0, 6.0] 
# user_relative_movement = [1.0, 2.0, 3.0]
# object_percentage_overlap = [0.7]


"""
correspondacnes algorithm
"""

# # Initialize variables
# llm_times = sorted(map(float, self.llm_predictions.keys()))
# gt_times  = sorted(map(float, self.ground_truth.keys()))

# # Create the iterator for ground truth times
# gt_iter = iter(gt_times)
# current_gt_time = next(gt_iter, None)
# next_gt_time = next(gt_iter, None)

# # Iterate through the LLM predictions
# for i, current_llm_time in enumerate(llm_times):

#     # Get the next LLM time
#     next_llm_time = llm_times[i + 1] if i + 1 < len(llm_times) else None

#     # In the case that we have more LLM predictions but no gt values --> means we don't have  
#     if current_gt_time is None: 
#         self.Fp_out +=1
#         self.total_llm_predictions +=1
#         continue

#     # In the case current gt is less than current llm and next gt is None --> means that current LLM falsely activated
#     if current_gt_time is not None and next_gt_time is None and current_gt_time < current_llm_time:
#         self.Fp_out += 1               # is Fp_out for current_llm_time or Fn for current Fn
#         self.total_llm_predictions +=1 # added 2/12
#         current_gt_time = next_gt_time
#         next_gt_time = next(gt_iter, None)
#         continue

#     # Check if the ground truth time is before the first LLM prediction
#     while current_gt_time is not None and next_gt_time is not None and current_gt_time < current_llm_time:
#         self.Fn += 1  # No LLM activation before the ground truth
#         current_gt_time = next_gt_time
#         next_gt_time = next(gt_iter, None)

#     # Check if two consecutive LLM times occur before the current ground truth time
#     if current_gt_time is not None and next_llm_time is not None and next_llm_time < current_gt_time:
#         self.Fp_out += 1  # False positive: as current LLM prediction does not correspond to any ground truth
#         self.total_llm_predictions +=1
#         continue

#     # Check if the current LLM corresponds to the current or next two ground truths
#     if current_gt_time is not None and current_llm_time <= current_gt_time:
#         time_diff = next_gt_time - current_gt_time if next_gt_time is not None else float('inf')

#         matched_ground_truths = []

#         # Option 1: next LLM time is larger than next ground truth time so 1 LLM prediction for 2 ground truths or 1gt and 1Fn
#         if next_llm_time is not None and next_gt_time is not None and next_llm_time > next_gt_time:
#             matched_ground_truths.append((current_gt_time, self.ground_truth[current_gt_time]))
#             self.total_llm_predictions +=1
#             current_gt_time = next_gt_time
#             next_gt_time = next(gt_iter, None)

#             # Either it's a missed prediction (false negative) --> for the next gt there is no LLM prediction
#             if time_diff > 1:
#                 # for current gt there is a llm
#                 self.correspondences.append((current_llm_time, self.llm_predictions[current_llm_time], matched_ground_truths))
#                 # for next gt there is not llm (as next llm > next gt)
#                 self.Fn += 1
#                 current_gt_time = next_gt_time     # added 2/12
#                 next_gt_time = next(gt_iter, None) # added 2/12

#             # Or it refers to two ground truth events
#             if time_diff <= 1:
#                 matched_ground_truths.append((current_gt_time, self.ground_truth[current_gt_time]))
#                 self.correspondences.append((current_llm_time, self.llm_predictions[current_llm_time], matched_ground_truths))
#                 self.total_llm_predictions +=1
#                 current_gt_time = next_gt_time
#                 next_gt_time = next(gt_iter, None)
    
#         # Option 2: Next LLM time is smaller than next ground truth time
#         elif next_llm_time is not None and next_gt_time is not None and next_llm_time <= next_gt_time:
#             matched_ground_truths.append((current_gt_time, self.ground_truth[current_gt_time]))
#             self.correspondences.append((current_llm_time, self.llm_predictions[current_llm_time], matched_ground_truths))
#             self.total_llm_predictions +=1
#             current_gt_time = next_gt_time
#             next_gt_time = next(gt_iter, None)

#         # Option 3: Next ground truth time is None
#         elif next_llm_time is not None and next_gt_time is None:
#             matched_ground_truths.append((current_gt_time, self.ground_truth[current_gt_time]))
#             self.correspondences.append((current_llm_time, self.llm_predictions[current_llm_time], matched_ground_truths))
#             self.total_llm_predictions +=1
#             current_gt_time = next_gt_time
#             next_gt_time = next(gt_iter, None)

# # After the LLM loop ends, handle remaining ground truth times
# while current_gt_time is not None:
#     self.Fn += 1  # False negative: LLM did not predict while ground truth remains
#     current_gt_time = next_gt_time
#     next_gt_time = next(gt_iter, None)