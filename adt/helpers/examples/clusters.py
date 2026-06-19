import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

# Example coordinates for 12 objects
obj_positions_scene = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 3.0, 4.0],
    [3.0, 2.0, 1.0],
    [8.0, 8.0, 8.0],
    [7.0, 8.0, 9.0],
    [8.0, 7.0, 9.0],
    [1.0, 0.0, 1.0],
    [2.0, 1.0, 1.0],
    [3.0, 1.0, 0.0],
    [8.0, 9.0, 7.0],
    [9.0, 8.0, 7.0],
    [7.0, 7.0, 7.0]
])

# Step 1: Define Groups of Objects using K-Means
k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=0).fit(obj_positions_scene)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Object Labels: ", labels)
print("Centroids: ", centroids)

# Step 2: Track User Position and Orientation
# Example user position and orientation (update these with real-time data)
user_position = np.array([2.5, 2.5, 2.5])
user_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion (x, y, z, w)

# Step 3: Calculate Distance to Groups
distances = [euclidean(user_position, centroid) for centroid in centroids]
print("Distances to Centroids: ", distances)

# Step 4: Determine Approach using Orientation
def is_approaching_using_orientation(user_position, user_orientation, centroid, threshold=0.1):
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

# Step 5: Determine Approach using Distance
# def is_approaching_using_distance(previous_distance, current_distance, threshold=0.1):
#     # Check if the distance to the centroid is decreasing
#     return current_distance < previous_distance - threshold

# # Example previous distances for the sake of demonstration (you need to store and update these in real-time)
# previous_distances = [3.5, 7.5, 5.5]  # Example previous distances to centroids

# # Calculate current distances to centroids
# current_distances = [euclidean(user_position, centroid) for centroid in centroids]

# Determine approach using orientation
approaches_by_orientation = [is_approaching_using_orientation(user_position, user_orientation, centroid) for centroid in centroids]

# Determine approach using distance
# approaches_by_distance = [is_approaching_using_distance(prev_dist, curr_dist) for prev_dist, curr_dist in zip(previous_distances, current_distances)]

# Combine both approaches (assuming both conditions need to be true)
# approaches = [approach_orientation and approach_distance for approach_orientation, approach_distance in zip(approaches_by_orientation, approaches_by_distance)]
approaches = [approach_orientation for approach_orientation in zip(approaches_by_orientation)]


# Output the result
for i, approaching in enumerate(approaches):
    print(f"User is approaching group {i+1}: {approaching}")
