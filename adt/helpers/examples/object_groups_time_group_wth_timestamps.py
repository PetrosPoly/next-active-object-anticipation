from collections import deque
from datetime import datetime, timedelta

class TimeWindowAnalyzer:
    def __init__(self, window_size=30, min_history_size=15, change_threshold=0.7):
        self.window_size = window_size
        self.min_history_size = min_history_size
        self.change_threshold = change_threshold
        self.history = deque(maxlen=window_size)

    def add_objects(self, timestamp, objects):
        self.history.append((timestamp, set(objects)))

    def has_user_moved(self, timestamp, new_objects):
        new_set = set(new_objects)
        
        # Add the new objects to history
        self.add_objects(timestamp, new_objects)
        
        # Calculate the aggregated set of objects over the window
        aggregated_set = set().union(*(objects for _, objects in self.history))
        
        # Calculate the intersection with the new set
        intersection = new_set.intersection(aggregated_set)
        
        # Calculate the percentage of changed objects
        num_changed_objects = len(aggregated_set) - len(intersection)
        change_percentage = num_changed_objects / len(aggregated_set) if len(aggregated_set) > 0 else 1
        
        # Check the change percentage against the threshold
        if change_percentage > self.change_threshold:
            return True
        
        # Handle the case where the user is close to one object and then another close object
        if len(new_objects) <= 1 and any(len(objects) >= 3 for _, objects in self.history):
            return True
        
        return False

# Example usage with smaller representative sample
analyzer = TimeWindowAnalyzer(window_size=30, min_history_size=5, change_threshold=0.7)

# Define timestamps and content based on the provided sample data
start_timestamp = datetime.now()
timestamps_and_objects = [
    (start_timestamp + timedelta(seconds=i / 30), objects.split(', ')) for i, objects in enumerate([
        "RedClock, CoffeeCanisterLarge, WhiteUtensilTray, BlackCeramicMug",
        "RedClock, CoffeeCanisterLarge, WhiteUtensilTray, BlackCeramicMug",
        "RedClock, CoffeeCanisterLarge, WhiteUtensilTray",
        "RedClock, CoffeeCanisterLarge, WhiteUtensilTray",
        "RedClock, CoffeeCanisterLarge",
        "RedClock, CoffeeCanisterLarge",
        "RedClock",
        "RedClock",
        "RedClock",
        "ChoppingBoard",
        "ChoppingBoard, WoodenBowl",
        "ChoppingBoard, WoodenBowl",
        "Mango_A, ChoppingBoard, WoodenBowl, WoodenFork",
        "BlackCeramicBowl, Mango_A, BlackCeramicDishLarge, WoodenBowl, BlackCeramicDishSmall, WoodenFork",
        "BlackCeramicBowl, Mango_A, BlackCeramicDishLarge, WoodenBowl, BlackCeramicDishSmall, CakeMocha_A, WoodenFork"
    ])
]

# Process the data
for ts, objects in timestamps_and_objects:
    if len(analyzer.history) >= analyzer.min_history_size:
        moved = analyzer.has_user_moved(ts, objects)
        if moved:
            print(f"Timestamp {ts}: User has moved to a different area.")
        else:
            print(f"Timestamp {ts}: User is in the same area.")
    else:
        analyzer.add_objects(ts, objects)
        print(f"Timestamp {ts}: Collecting data...")