import numpy as np
import pandas as pd

# Define the method: 
use = "numpy"

# the dataset 
movement_time_dict = {
    'ChoppingBoard': {'start_time': 6.432, 'end_time': 14.331},
    'WhiteChair': {'start_time': 7.965, 'end_time': 8.199},
    'BlackKitchenChair': {'start_time': 9.532, 'end_time': 9.832},
    'KitchenKnife': {'start_time': 19.73, 'end_time': 26.796},
    'WoodenSpoon': {'start_time': 20.33, 'end_time': 30.795},
    'WoodenFork': {'start_time': 34.861, 'end_time': 41.06},
    'Donut_B': {'start_time': 48.959, 'end_time': 64.756},
    'Cereal_Anon': {'start_time': 51.458, 'end_time': 58.624},
    'DinoToy': {'start_time': 71.089, 'end_time': 74.321},
    'WoodenToothbrushHolder': {'start_time': 71.955, 'end_time': 72.588},
    'Cracker_Anon': {'start_time': 73.422, 'end_time': 81.654},
    'BlackCeramicMug': {'start_time': 82.953, 'end_time': 89.852}
}

# Define the time interval 
time_increment = 1 / 30

# Generate the time values from 1 to 90 with a step of 0.033
time_values = np.arange(1, 90, time_increment)

for current_time_s in time_values:

    # Initialize an empty list to store the names of objects in motion
    objects_in_motion = []

    if use == "loop":
        # Iterate through the movement_time_dict to check which objects are in motion
        for name, times in movement_time_dict.items():
            start_time = times['start_time']
            end_time = times['end_time']
            
            if start_time <= current_time_s <= end_time:
                objects_in_motion.append(name)
        
        # `objects_in_motion` now contains the names of the objects that are in motion at the current time `s`
        print("Object that are currently in motion", objects_in_motion)
        
    if use == "numpy":
        
        # Convert the dictionary to lists
        object_names = np.array(list(movement_time_dict.keys()))
        start_times = np.array([movement_time_dict[obj]['start_time'] for obj in object_names])
        end_times = np.array([movement_time_dict[obj]['end_time'] for obj in object_names])

        # Use NumPy to find objects in motion
        objects_in_motion = object_names[(start_times <= current_time_s) & (current_time_s <= end_times)]

        # Convert to a list if needed
        objects_in_motion = objects_in_motion.tolist()
        
        # print
        print("Object that are currently in motion", objects_in_motion)

    if use == "panda":
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(movement_time_dict).T  # Transpose to make keys as rows
        df.reset_index(inplace=True)
        df.columns = ['object_name', 'start_time', 'end_time']
        
        # Filter the DataFrame to find objects in motion
        objects_in_motion = df[(df['start_time'] <= current_time_s) & (df['end_time'] >= current_time_s)]['object_name'].tolist()
        
        # print
        print("Object that are currently in motion", objects_in_motion)

        # `objects_in_motion` now contains the names of the objects that are in motion at the current time `s`