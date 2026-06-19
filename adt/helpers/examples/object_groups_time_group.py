from collections import deque

class TimeWindowAnalyzer:
    def __init__(self, window_size=3, change_threshold=0.7):
        self.window_size = window_size
        self.change_threshold = change_threshold
        self.history = deque(maxlen=window_size)

    def add_objects(self, objects):
        self.history.append(set(objects))

    def has_user_moved(self, new_objects):
        new_set = set(new_objects)
        
        if len(self.history) == 0:
            self.add_objects(new_objects)
            return False
        
        # Calculate the aggregated set of objects over the window
        aggregated_set = set().union(*self.history)
        
        # Calculate the intersection with the new set
        intersection = new_set.intersection(aggregated_set)
        
        # Calculate the percentage of changed objects
        num_changed_objects = len(aggregated_set) - len(intersection)
        change_percentage = num_changed_objects / len(aggregated_set) if len(aggregated_set) > 0 else 1
        
        # Add the new objects to history
        self.add_objects(new_objects)
        
        # Check the change percentage against the threshold
        if change_percentage > self.change_threshold:
            return True
        
        # Handle the case where the user is close to one object and then another close object
        if len(new_objects) <= 1 and any(len(hist) >= 3 for hist in self.history):
            return True
        
        return False

# Example usage
analyzer = TimeWindowAnalyzer(window_size=3, change_threshold=0.7)

# Define scenarios
scenarios = [
    (['chair', 'table', 'lamp', 'sofa', 'bookshelf'], ['plant', 'table', 'lamp', 'sofa', 'rug']),
    (['chair', 'table', 'lamp', 'sofa', 'bookshelf'], ['chair', 'table', 'lamp', 'sofa', 'rug']),
    (['sofa'], ['lamp']),
    (['sofa'], ['sofa']),
    (['chair', 'table', 'lamp', 'sofa', 'bookshelf'], ['table']),
    (['chair', 'table', 'lamp', 'sofa', 'bookshelf'], ['chair', 'table', 'lamp']),
]

# Test scenarios
for i, (previous_objects, new_objects) in enumerate(scenarios, 1):
    # Add the previous objects to initialize history for testing
    analyzer.add_objects(previous_objects)
    
    moved = analyzer.has_user_moved(new_objects)
    print(f"Scenario {i}: {'User has moved to a different area.' if moved else 'User is in the same area.'}")

