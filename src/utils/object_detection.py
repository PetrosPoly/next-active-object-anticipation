import numpy as np

from projectaria_tools.core.sophus import SE3

from utils.tools import exponential_filter

# Interaction detection phase # 
def calculate_relative_pose_difference(T_scene_object_before, T_scene_object):
        
        # convert to list
        T_scene_object = list(T_scene_object.values())
        
        ## initialize the poses before in case I don't know 
        if T_scene_object_before is None:
            T_scene_object_before = [None] * len(T_scene_object)  
     
        ## Initialization of lists of the objects                                                         
        T_scene_object_ema = [None] * len(T_scene_object)                                                                       # Me: It's important to specify dtype=object so that np.array be capable of holding any object
        relative_T_object = [None] * len(T_scene_object)                                                                        # Me: I would use np.array to exploit vectorization and avoid for loops but I need to use the inverse method of the objects 
        norm_relative_T_object = [None] * len(T_scene_object)                                                                   # Me: Initialization of calculating the relative pose
            
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
                vector_pose_6d = exponential_filter(vector_pose_6d, vector_pose_6d_before, alpha=0.7)                       # Apply exponential moving average (EMA) to reduce noise
                translational_vector, rotation_vector = vector_pose_6d[:3], vector_pose_6d[3:]                          # Decompose the EMA result into translational and rotational components
                T_scene_object_ema[i] = SE3.exp(translational_vector, rotation_vector)                                          # Reconstruct the SE3 object from the smoothed 6D vector
                
                ## Relative
                relative_T_object_matrix = T_scene_object_before[i].inverse().to_matrix() @ T_scene_object_ema[i].to_matrix()   # Compute the relative pose transformation matrix and convert back to SE3
                relative_T_object[i] = SE3.from_matrix(relative_T_object_matrix)
                norm_relative_T_object[i] = np.linalg.norm(relative_T_object[i].log())                                           # Optionally, calculate the norm of the relative pose transformation's logarithm
        
            else:
                
                ## Calculations 
                vector_pose_6d = T_scene_object[i].log()[0]                                                                        # Me: Calculate the 6D vector (log representation) of the current and previous SE3 transformations
                vector_pose_6d_before = T_scene_object_before[i].log()[0]
                
                ## Ema 
                vector_pose_6d = exponential_filter(vector_pose_6d, vector_pose_6d_before, alpha=0.7)                       # Apply exponential moving average (EMA) to reduce noise
                translational_vector, rotation_vector = vector_pose_6d[:3], vector_pose_6d[3:]                          # Decompose the EMA result into translational and rotational components
                T_scene_object_ema[i] = SE3.exp(translational_vector, rotation_vector)                                          # Reconstruct the SE3 object from the smoothed 6D vector
                
                ## Relative
                relative_T_object_matrix = T_scene_object_before[i].inverse().to_matrix() @ T_scene_object_ema[i].to_matrix()   # Compute the relative pose transformation matrix and convert back to SE3
                relative_T_object[i] = SE3.from_matrix(relative_T_object_matrix)
                norm_relative_T_object[i] = np.linalg.norm(relative_T_object[i].log())                                           # Optionally, calculate the norm of the relative pose transformation's logarithm                                         
                    
        # Update of the poses in the past 
        T_scene_object_before = T_scene_object_ema.copy()   
        
        # Tranform to np array
        norm_relative_T_object = np.array(norm_relative_T_object)
        return np.round(norm_relative_T_object, 8), T_scene_object_before