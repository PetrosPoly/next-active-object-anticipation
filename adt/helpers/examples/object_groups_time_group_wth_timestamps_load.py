from collections import deque
from datetime import datetime, timedelta

import os

class TimeWindowAnalyzer:
    def __init__(self, history_size=15, future_size=5, change_threshold=0.8):
        self.history_size = history_size
        self.future_size = future_size
        self.change_threshold = change_threshold
        self.history = deque(maxlen=history_size)
        self.future = deque(maxlen=future_size)
        self.future_populating = False

    def add_objects(self, timestamp, objects):
        # Update the future buffer only 
        self.history.append((timestamp, set(objects)))
        if len(self.history) >= self.history_size:
            self.future_populating = True
        if self.future_populating:
            self.future.append((timestamp, set(objects)))
        # self.history.append((timestamp, set(objects)))
        # self.future.append((timestamp, set(objects)))
            
    def compare_objects(self):
        if len(self.history) < self.history_size or len(self.future) < self.future_size:
            return "Collecting data..."

        history_aggregated = set().union(*(objects for _, objects in self.history))
        future_aggregated = set().union(*(objects for _, objects in self.future))

        if len(future_aggregated) == 0 and len(history_aggregated) == 0: 
            return "User is moving is to different area"
        
        # Calculate the intersection
        intersection = future_aggregated.intersection(history_aggregated)
        
        # Calculate the percentage of changed objects
        num_changed_objects = len(history_aggregated) - len(intersection)
        change_percentage = num_changed_objects / len(history_aggregated) if len(history_aggregated) > 0 else 1
        
        if change_percentage > self.change_threshold:
            return "User has moved to a different area."
        
        return "User is in the same area."

# Function to read and parse the file content
def read_objects_from_file(file_path):
    objects_within_radius = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and ':' in line:
                _, objects = line.split(':', 1)
                objects = [obj.strip() for obj in objects.split(',') if obj.strip()]
                objects_within_radius.append(objects)
    return objects_within_radius[1:]

# Function to process the data and detect movements
def detect_movements(file_path):
    analyzer = TimeWindowAnalyzer(history_size=30, future_size=10, change_threshold=0.7)
    objects_within_radius = read_objects_from_file(file_path)
    
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

# Use the function to detect movements from the uploaded file
folder = "/Users/petrospolydorou/ETH_Thesis/coding/Actionanticipation/projectaria_sandbox/projectaria_tools/projectaria_tools/utils"
file = os.path.join(folder,'output.txt')

detect_movements(file)
