def has_user_moved(new_objects, previous_objects, change_threshold=0.7):
    """
    Determine if the user has moved to a different area based on object signatures
    and the percentage of objects that have changed.
    
    Args:
    new_objects (list): List of object names within the 1.5-meter radius at the current timestamp.
    previous_objects (list): List of object names within the 1.5-meter radius at the previous timestamp.
    change_threshold (float): The threshold for the percentage of objects that need to change.
    
    Returns:
    bool: True if the user has moved to a different area, False otherwise.
    """
    # Calculate the set of objects
    new_set = set(new_objects)
    previous_set = set(previous_objects)
    
    # Determine the intersection of both sets
    intersection = new_set.intersection(previous_set)
    
    # Calculate the percentage of changed objects
    num_changed_objects = len(previous_set) - len(intersection)
    change_percentage = num_changed_objects / len(previous_set) if len(previous_set) > 0 else 1

    # Check the change percentage against the threshold
    if change_percentage > change_threshold:
        return True
    
    # Check for significant reduction in object list size
    if len(new_objects) <= 1 and len(previous_objects) >= 3:
        return True
    
    # Handle the case where the user is close to one object and then another close object
    if len(previous_objects) <= 2 or len(new_objects) <= 2:
        # If the sets are different, consider it a move
        if new_set != previous_set:
            return True

    return False

# Define scenarios
scenarios = [
    (['chair', 'table', 'lamp', 'sofa', 'bookshelf'], ['plant', 'table', 'lamp', 'sofa', 'rug']),  # Significant Change in Large List
    (['chair', 'table', 'lamp', 'sofa', 'bookshelf'], ['chair', 'table', 'lamp', 'sofa', 'rug']),  # Minimal Change in Large List
    (['sofa'], ['lamp']),                                                                          # Small List to Different Small List
    (['sofa'], ['sofa']),                                                                          # Small List to Same Small List
    (['chair', 'table', 'lamp', 'sofa', 'bookshelf'], ['table']),                                  # Large List to Significantly Smaller List
    (['chair', 'table', 'lamp', 'sofa', 'bookshelf'], ['chair', 'table', 'lamp']),                 # Large List to Slightly Smaller List
]

# Test scenarios
for i, (previous_objects, new_objects) in enumerate(scenarios, 1):
    moved = has_user_moved(new_objects, previous_objects)
    print(f"Scenario {i}: {'User has moved to a different area.' if moved else 'User is in the same area.'}")

