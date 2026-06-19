
# ==============================================
# Parameters Settting
# ==============================================

""" 
Setting the parameters to run the algorithm 
"""
# ==============================================
# Run for different parameter combinatons
# ==============================================

"""
A for Loop to run the algorithm for different parameters combinations 
""" 

            
        # ==============================================
        # Filenames / Paths  & Load of the Data
        # ==============================================

"""
        Initialise the paths and load data
"""
        
        # ==============================================
        # Initialization 
        # ==============================================

"""
        Initialise parameters 
"""
        
        # ==============================================
        # Load the Ground truth data 
        # ==============================================

"""
        Load the ground truth data
"""
        
        # ==============================================
        # Loop over all timestamps in the sequence
        # ==============================================

"""
        Loop over all the timestamps
"""
            
            # ==============================================
            # Users poses - position / velocity / movement (scene)
            # ==============================================                                                          
                                                                                    
"""
        SLAM measurements of the user
        1. user's current position 
        2. user's current timestamp velocity vector
        3. user's movement from timestamp to timestamp
        4. user's movement from start to current timestamp
"""         

            # ==============================================
            # Objects Poses
            # ==============================================      

"""
        SLAM measurements
        1. current object poses (3D vector for the position)
"""  
            
            # ==============================================
            # Visual Objects in Camera Frame
            # ==============================================  

"""
            1. Project 3D poimts of the Object to the camera frame (2D)
            2. Keep only those objects that are inside the field of view 
"""
            
            # ==============================================
            # Coordinate Transformation from Device and Camera to World
            # ==============================================  

"""
            1. Trasform x, y, z axis from Device to World Frame (Rotation Matrix)
            2. Tranform x, y, z axis from Camera to World Frame (Rotation Matrix)
"""
            
            # ==============================================
            # Dot Products - Distances
            # ==============================================  

"""
            1. Calculate the dot product betweem the camera's z-axis and vector that connects camera and ojbect
            2. z-axis, y-axis, x-axis should be expressed in the World coordinate 
            3. Calculate the distance from the user to all visible objects (objects im the field of view)
"""
            
            # ==============================================
            # Time Window - Accumulated / Average Values & Counts
            # ==============================================  
            
"""
            1. Create a sliding time window of maximum duration of 3 seconds 
            2. Gather a dictionary with the average dot value within these 3 seconds 
                - Dot Value
                - Distance 
            3. Increment dictionary count per object at each timestamp if dot > 0.9
            4. Increment dictionary count per object at each timestamp if distance < 2
            5. Increment dictionary count per object at each timestamp if object is visible 
"""
            
            # ==============================================
            # Filter Visible Objects with Average Values
            # ==============================================  
            
"""
            1. Filter again visible objects and keep those where 
                - the average dot value is higher than 0.7
                - the average distance values is less than 2
"""            
            # ==============================================
            # Keep the Important Context Information for the feasible objects
            # ==============================================  
            
"""             
             1. For visible objects that meet the conditions for average dot product and distance values
             2. Maintain a dictionary with object names and respective counts for:
                - Dot product
                - Distance
             3. Maintain a dictionary with object names and respective time to approach"
             
"""

            # ==============================================
            # 4 Criteria to enable LLM (1. High dot products duration 2. Low distance duration 3. Time to contact 4. High visibility duration)
            # ==============================================  
            
"""
            1. Initialize dictionaries and lists to be used as a condtion to fire for the LLM activation 
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
                - Close distance < 30 counts or 1 second from the 3
                - Time to approach
"""
        
            # ==============================================
            # Conditions to fire the LLM 
            # ==============================================  
            
"""
            LLM is being fired to predict only if we have the following conditions: 
            
            1. objects with high dot counts more than the threshold of high dot counts within the given time window
            2. objects with low distance counts more than the threshold low distance counrs within the given time window  
            3a.objects that is approachable in less than the time threshold and belongs to high dot counts and low distance counts dictionaries with at least one count
            3b.objects that is approachable in less than the time threshold and belongs either to (1) or to (2) 
            4. objects that has shown high visibility
            5. objects that are approachable in less than 2 seconds consistently (for several countss
            Info that is passed to LLM
"""

            # ==============================================
            # Prompt and Contextual info passed to LLM
            # ==============================================  
            
"""
            1. objects with high dot counts more than the threshold of high dot counts within the given time window
            2. objects with low distance counts more than the threshold low distance counrs within the given time window  
            3. objects belong (2) the distance value 
            4. objects belomg (2) only those that time to approach is less than 2 seconds
            5. previous predictions
            
            After firing LLM to predict the next object, should remain inactive for a period because im the opposite case 
            the previous conditions continue to be satisfied and the LLM will be activated again
"""

            # ==============================================
            # Objects Inside the radius & LLM re-activation conditions
            # ==============================================  
            
"""
            4 conditions to enable the LLM either one of the below 
            - Not earlier than 2 seconds 
            - Not later than 5 seconds 
            - user's movement significant over a threshold     ()
            - user's is not surrounding by the same of objects (overlapping)
            
            another way could be to reitinialize the above mentiioned dictionaries wihch work as a condittion to fire the LLM
"""

        # ==============================================
        # Store the predictions
        # ==============================================  
        
        # code 
        

        # ==============================================
        # Prints
        # ==============================================  
        
        # code 
        
    # if __name__ == "__main__":
    #     main()


