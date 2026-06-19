import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

from collections import deque
from datetime import datetime, timedelta

import os

# class to work with objects around the user 
class ObjectGroupAnalyzer:
    def __init__(self, change_threshold, movement_threshold, history_size=15, future_size=5):
        self.history_size = history_size
        self.future_size = future_size
        self.change_threshold = change_threshold
        self.movement_threshold = movement_threshold
        self.history = deque(maxlen=history_size)
        self.future = deque(maxlen=future_size)
        self.future_populating = False

    def add_objects(self, timestamp, objects):
        self.history.append((timestamp, set(objects)))      # adds in the deque a history of objects
        if len(self.history) >= self.history_size:          # at the beginning before we start comparison we load the data with objects that have been around the around for 15 counts (0.5 sec)
            self.future_populating = True                   # after history has been filled with values with add future values (future here refers to a more recent history only 5 counts before )
        if self.future_populating:
            self.future.append((timestamp, set(objects)))
            
    def compare_objects(self):
        
        """
        The process we follow consists of these steps:

        - We first take the set of all objects that have been near the user during the last 20 counts, excluding the most recent 5 counts.
        - Then, we take the set of objects that have been near the user during the last 5 counts.
        - We find the intersection between these two sets.
        - We then calculate the percentage change between the objects from the last 15 counts and the last 5 counts, comparing the intersection with the 15-count history.
        - If neither the history (20 counts) nor the recent history (5 counts) contains any objects, this suggests the user is moving toward a different area.
        - If the percentage of change is 100% and the recent history is empty, it indicates that the user has completely left the previous group of objects and is heading toward a new area with no objects currently close by.
        - If the percentage is 100%, the recent history is not empty, and the past history is empty, this means the user is arriving in a new area and approaching new objects.
        - Finally, if the percentage change is over 80%, we can assume the user is transitioning to a different interaction area.
        
        If any of these conditions are satisfied, the Large Language Model (LLM) is activated, receiving the contextual information. If the conditions are met, the LLM is queried to predict the next interaction.
        """
        
        # if the history and the future haven't collect the amount of data necessary for comparisons
        if len(self.history) < self.history_size or len(self.future) < self.future_size: 
            print("Collecting data...")     # waiting to collect objects 
            return False

        # Take the union of the history and the future and then the set which gives the unique values 
        history_aggregated = set().union(*(objects for _, objects in self.history))
        future_aggregated = set().union(*(objects for _, objects in self.future))
        
        # Calculate the intersection
        intersection = future_aggregated.intersection(history_aggregated)
        
        # Calculate the percentage of changed objects
        num_changed_objects = len(history_aggregated) - len(intersection)
        change_percentage = num_changed_objects / len(history_aggregated) if len(history_aggregated) > 0 else 1
        
        # The if statements for specific occasions
        if len(future_aggregated) == 0 and len(history_aggregated) == 0:
            """
            No any object around the user. The user is on the way for a different area so LLM should be active
            """
            return True 

        if change_percentage == 1 and len(future_aggregated) == 0:
            # print('the user left completely the previous group of objects')
            """
            The user left completely the previous group of objects --> initialization of the history and future
            """
            self.history = deque(maxlen=self.history_size)
            self.future = deque(maxlen=self.future_size)
            return True
    
        if change_percentage == 1 and len(history_aggregated) == 0:  
            """
            The user arrived close to a new group of objects as last 5 counts we have objects and before 20 until before 5 (history) we don't have objects 
            """
            # print('The user has arrived to a new space with objects') i
            self.history = deque(maxlen=self.history_size)
            self.future = deque(maxlen=self.future_size)
            return False
            
        return change_percentage > self.change_threshold
    
    def check_if_user_changed_area(self, result_objects, result_movement):
        if result_objects or result_movement:
            print ("User is moving to a different area. LLM can be activate again")
            return True
        else:
            print("User is on the same area, Keep LLM stable")
            return False
        
    def user_move(self, user_movement):
        return user_movement >= self.movement_threshold 
        
# Function to process the data and detect movements
def detect_movements(objects_within_radius):
    analyzer = ObjectGroupAnalyzer(history_size=30, future_size=10, change_threshold=0.7)
    
    start_timestamp = datetime.now()

    for i, objects in enumerate(objects_within_radius):
        timestamp = start_timestamp + timedelta(seconds=i / 30)  # Assuming 30 frames per second
        seconds = (timestamp - start_timestamp).total_seconds()  # Get seconds since start
        if i == 300:
            print('')
        # Update both buffers
        analyzer.add_objects(seconds, objects)

        # Always call compare_buffers to get the status
        result = analyzer.compare_objects()
        print(f"Item {i} and Timestamp {seconds:.3f}s: {result}")


# Group objects with cluster k-means 
def group_object_with_kmeans(obj_positions):
    
    # Step 1: Define Groups of Objects using K-Means
    k = 3  # Number of clusters, adjust as needed
    kmeans = KMeans(n_clusters=k).fit(obj_positions)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Step 2: Track User Position and Orientation
    # Example user position and orientation (update these with real-time data)
    user_position = np.array([1.0, 1.0, 1.0])
    user_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion (x, y, z, w)

    # Step 3: Calculate Distance to Groups
    distances = [euclidean(user_position, centroid) for centroid in centroids]

    # Step 4: Determine Approach
    def is_approaching(user_position, user_orientation, centroid, threshold=0.1):
        # Calculate direction vector from user to centroid
        direction_vector = centroid - user_position
        direction_vector /= np.linalg.norm(direction_vector)  # Normalize

        # Convert quaternion to direction vector
        qx, qy, qz, qw = user_orientation
        orientation_vector = np.array([
            2*(qx*qz + qw*qy),
            2*(qy*qz - qw*qx),
            1 - 2*(qx*qx + qy*qy)
        ])
        orientation_vector /= np.linalg.norm(orientation_vector)  # Normalize

        # Check if the user is facing the centroid within a certain threshold
        dot_product = np.dot(direction_vector, orientation_vector)
        return dot_product > threshold

    approaches = [is_approaching(user_position, user_orientation, centroid) for centroid in centroids]

    # Output the result
    for i, approaching in enumerate(approaches):
        print(f"User is approaching group {i+1}: {approaching}")


