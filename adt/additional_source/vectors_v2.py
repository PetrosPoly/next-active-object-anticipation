import numpy as np
import matplotlib.pyplot as plt

# Helper function to swap Y and Z components for plotting
def swap_yz(vec):
    return np.array([vec[0], vec[2], vec[1]])

# Define the desired angle between the vectors in degrees
desired_angle_degrees = 30
desired_angle_radians = np.radians(desired_angle_degrees)

# Define the length of the vector (arbitrary, set to 1 for simplicity)
vector_length = 1.0

# Define the Y-component for the object's elevation
object_elevation = 0.3  # Positive value to position the object upwards

# Define camera position at the origin for simplicity
camera_position = np.array([0.0, 0.0, 0.0])  # [X, Y, Z]

# Define the camera's Z-axis vector (pointing forward and slightly downward)
camera_z_axis = np.array([1.0, -1.0, 0.0])  # [X, Y, Z]
camera_z_axis /= np.linalg.norm(camera_z_axis)  # Normalize to unit vector

# Project the camera's Z-axis onto the XZ plane (set Y component to zero)
projection_camera_z_axis_xz = camera_z_axis.copy()
projection_camera_z_axis_xz[1] = 0  # Set Y component to 0
projection_camera_z_axis_xz /= np.linalg.norm(projection_camera_z_axis_xz)  # Normalize

# Calculate the components of the vector to the object to ensure a 30-degree angle
vx = vector_length * np.cos(desired_angle_radians)  # X component
vz = vector_length * np.sin(desired_angle_radians)  # Z component

# Define the object's position with a slight elevation in Y
object_position = np.array([vx, object_elevation, vz])  # [X, Y, Z]

# Vector from camera to object
vector_to_object = object_position - camera_position  # [X, Y, Z]

# Project the vector to the object onto the XZ plane (set Y component to zero)
projection_vector_to_object_xz = vector_to_object.copy()
projection_vector_to_object_xz[1] = 0  # Set Y component to 0
projection_vector_to_object_xz /= np.linalg.norm(projection_vector_to_object_xz)  # Normalize

# Calculate the dot product between the two projections
dot_product_xz = np.dot(projection_vector_to_object_xz, projection_camera_z_axis_xz)
cos_theta_xz = dot_product_xz  # Since vectors are normalized
theta_xz = np.arccos(cos_theta_xz)
theta_xz_degrees = np.degrees(theta_xz)

# Swap Y and Z for plotting to have Y as vertical in the plot
camera_position_plot = swap_yz(camera_position)
object_position_plot = swap_yz(object_position)
vector_to_object_plot = swap_yz(vector_to_object)
camera_z_axis_plot = swap_yz(camera_z_axis)
projection_vector_to_object_xz_plot = swap_yz(projection_vector_to_object_xz)
projection_camera_z_axis_xz_plot = swap_yz(projection_camera_z_axis_xz)

# Define transferred vectors (origin-relative)
transferred_vector_to_object_plot = swap_yz(vector_to_object)
transferred_camera_z_axis_plot = swap_yz(camera_z_axis)

# Define projections for transferred vectors
projection_transferred_vector_to_object_xz_plot = swap_yz(projection_vector_to_object_xz)
projection_transferred_camera_z_axis_xz_plot = swap_yz(projection_camera_z_axis_xz)

# Start plotting
fig = plt.figure(figsize=(14, 7))

# 1. Merged 3D Plot of Original and Transferred Vectors
ax1 = fig.add_subplot(121, projection='3d')

# Plot original positions
ax1.scatter(camera_position_plot[0], camera_position_plot[1], camera_position_plot[2],
            color='blue', label='Camera Position', s=50)
ax1.text(camera_position_plot[0], camera_position_plot[1], camera_position_plot[2],
         'Camera', color='blue')

ax1.scatter(object_position_plot[0], object_position_plot[1], object_position_plot[2],
            color='red', label='Object Position', s=50)
ax1.text(object_position_plot[0], object_position_plot[1], object_position_plot[2],
         'Object', color='red')

# Plot the vector connecting camera to object
ax1.quiver(camera_position_plot[0], camera_position_plot[1], camera_position_plot[2],
           vector_to_object_plot[0], vector_to_object_plot[1], vector_to_object_plot[2],
           color='green', label='Vector to Object', arrow_length_ratio=0.1)

# Plot the camera's Z-axis
ax1.quiver(camera_position_plot[0], camera_position_plot[1], camera_position_plot[2],
           camera_z_axis_plot[0], camera_z_axis_plot[1], camera_z_axis_plot[2],
           color='purple', label='Camera Z-axis', arrow_length_ratio=0.1)

# Plot transferred vector to object from origin
# ax1.quiver(0, 0, 0,
#            transferred_vector_to_object_plot[0], transferred_vector_to_object_plot[1], transferred_vector_to_object_plot[2],
#            color='orange', label='Transferred Vector to Object', arrow_length_ratio=0.1)

# # Plot transferred camera Z-axis from origin
# ax1.quiver(0, 0, 0,
#            transferred_camera_z_axis_plot[0], transferred_camera_z_axis_plot[1], transferred_camera_z_axis_plot[2],
#            color='cyan', label='Transferred Camera Z-axis', arrow_length_ratio=0.1)

# Plot projections of original vectors onto XZ plane
ax1.quiver(0, 0, 0,
           projection_vector_to_object_xz_plot[0], projection_vector_to_object_xz_plot[1], projection_vector_to_object_xz_plot[2],
           color='darkgreen', linestyle='dashed', label='Projection Vector to Object (XZ)', arrow_length_ratio=0.1)

ax1.quiver(0, 0, 0,
           projection_camera_z_axis_xz_plot[0], projection_camera_z_axis_xz_plot[1], projection_camera_z_axis_xz_plot[2],
           color='darkblue', linestyle='dashed', label='Projection Camera Z-axis (XZ)', arrow_length_ratio=0.1)

# # Plot projections of transferred vectors onto XZ plane
# ax1.quiver(0, 0, 0,
#            projection_transferred_vector_to_object_xz_plot[0], projection_transferred_vector_to_object_xz_plot[1], projection_transferred_vector_to_object_xz_plot[2],
#            color='olive', linestyle='dashed', label='Projection Transferred Vector (XZ)', arrow_length_ratio=0.1)

# ax1.quiver(0, 0, 0,
#            projection_transferred_camera_z_axis_xz_plot[0], projection_transferred_camera_z_axis_xz_plot[1], projection_transferred_camera_z_axis_xz_plot[2],
#            color='navy', linestyle='dashed', label='Projection Transferred Camera Z-axis (XZ)', arrow_length_ratio=0.1)

# Formatting and labels
ax1.set_xlim([-1.5, 1.5])
ax1.set_ylim([-1.5, 1.5])
ax1.set_zlim([-1.5, 1.5])
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Z-axis')
ax1.set_zlabel('Y-axis (Vertical)')
ax1.set_title('Original and Transferred Vectors with Projections')
ax1.legend(loc='upper left')

# 2. 2D Plot of Projections onto XZ Plane
ax2 = fig.add_subplot(122)

# Plot projections onto XZ plane (horizontal plane)
ax2.quiver(0, 0,
           projection_vector_to_object_xz[0], projection_vector_to_object_xz[2],
           color='darkgreen', angles='xy', scale_units='xy', scale=1,
           label='Projection Vector to Object (XZ)')

ax2.quiver(0, 0,
           projection_camera_z_axis_xz[0], projection_camera_z_axis_xz[2],
           color='darkblue', angles='xy', scale_units='xy', scale=1,
           label='Projection Camera Z-axis (XZ)')

# Plot projections of transferred vectors onto XZ plane
# ax2.quiver(0, 0,
#            projection_transferred_vector_to_object_xz_plot[0], projection_transferred_vector_to_object_xz_plot[2],
#            color='olive', angles='xy', scale_units='xy', scale=1,
#            label='Projection Transferred Vector (XZ)')

# ax2.quiver(0, 0,
#            projection_transferred_camera_z_axis_xz_plot[0], projection_transferred_camera_z_axis_xz_plot[2],
#            color='navy', angles='xy', scale_units='xy', scale=1,
#            label='Projection Transferred Camera Z-axis (XZ)')

# Annotate the angle between the vectors
ax2.text(0.0, 0.3, f'Angle: {theta_xz_degrees:.2f}°', fontsize=12)

# Formatting and labels
ax2.set_xlim([-1.5, 1.5])
ax2.set_ylim([-1.5, 1.5])
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Z-axis')
ax2.set_title('Projections onto XZ Plane')
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.show()

# Print dot product and cosine of angle in XZ plane
print(f"Dot Product (XZ Plane): {dot_product_xz:.3f}")
print(f"Cosine of Angle (XZ Plane): {cos_theta_xz:.3f}")
print(f"Angle between vectors in XZ Plane: {theta_xz_degrees:.2f} degrees")
