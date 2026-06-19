from collections import deque
import time
import random

# Assume these are the object IDs
instances_ids = ['obj1', 'obj2']

# Initialize deques
direction_vectors = deque(maxlen=30)  # To store direction vectors (for completeness, not used in the example)
past_dot_products = {obj_id: deque() for obj_id in instances_ids}

# Function to simulate dot product calculation and updating the deque
def update_and_calculate_cumulative_dot_products(current_time_ns, visible_obj_ids, dot_products, past_dot_products):
    # Update the deque for each object
    for obj_id, dot_product in zip(visible_obj_ids, dot_products):
        if obj_id in past_dot_products:
            past_dot_products[obj_id].append((current_time_ns, dot_product))
        else:
            past_dot_products[obj_id] = deque([(current_time_ns, dot_product)])
    
    # Remove entries older than 3 seconds
    for obj_id in past_dot_products.keys():
        while past_dot_products[obj_id] and (current_time_ns - past_dot_products[obj_id][0][0]) > 3_000_000_000:
            past_dot_products[obj_id].popleft()
    
    # Calculate the cumulative dot products over the last 3 seconds
    cumulative_dot_products = {
        obj_id: sum(dp for ts, dp in past_dot_products[obj_id])
        for obj_id in visible_obj_ids
    }
    
    return cumulative_dot_products

# Simulate updates
start_time_ns = int(time.time() * 1e9)  # Current time in nanoseconds
visible_obj_ids = ['obj1', 'obj2']

for i in range(50):
    current_time_ns = start_time_ns + i * 100_000_000  # 100 ms interval (10 Hz)
    dot_products = [random.uniform(0, 1), random.uniform(0, 1)]  # Simulated dot products for obj1 and obj2

    cumulative_dot_products = update_and_calculate_cumulative_dot_products(current_time_ns, visible_obj_ids, dot_products, past_dot_products)
    print(f"Time: {current_time_ns}, Cumulative Dot Products: {cumulative_dot_products}")

    time.sleep(0.1)  # Sleep to simulate real-time updates (100 ms interval)
