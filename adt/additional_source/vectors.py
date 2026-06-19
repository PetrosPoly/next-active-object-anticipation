import numpy as np
import matplotlib.pyplot as plt

# Helper function to swap Y and Z components
def swap_yz(vec):
    return np.array([vec[0], vec[2], vec[1]])

# Define the desired angle between the vectors in degrees
desired_angle_degrees = 30
desired_angle_radians = np.radians(desired_angle_degrees)

# Define the length of the vector (arbitrary, set to 1 for simplicity)
vector_length = 1.0

# Define the Y-component for the object's elevation
object_elevation = 0.3  # Positive value to position the object upwards

# Define camera position at a non-origin point
camera_position = np.array([2.0, 0.5, 1.0])  # [X, Y, Z]

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
# Adjusted based on the camera's position
object_position = camera_position + np.array([vx, object_elevation, vz])  # [X, Y, Z]

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

# Swap Y and Z for plotting to have Y as vertical
camera_position_plot = swap_yz(camera_position)
object_position_plot = swap_yz(object_position)
vector_to_object_plot = swap_yz(vector_to_object)
camera_z_axis_plot = swap_yz(camera_z_axis)
projection_vector_to_object_xz_plot = swap_yz(projection_vector_to_object_xz)
projection_camera_z_axis_xz_plot = swap_yz(projection_camera_z_axis_xz)

# Transfer vectors to the origin for the second subplot
transferred_vector_to_object = vector_to_object  # Since camera is at the origin after transfer
transferred_camera_z_axis = camera_z_axis  # Camera's Z-axis remains the same

# Swap Y and Z for transferred vectors
transferred_vector_to_object_plot = swap_yz(transferred_vector_to_object)
transferred_camera_z_axis_plot = swap_yz(transferred_camera_z_axis)

# Start plotting
fig = plt.figure(figsize=(18, 6))

# 1. 3D Plot of Original Positions and Vectors
ax1 = fig.add_subplot(131, projection='3d')

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

# Formatting and labels
ax1.set_xlim([camera_position_plot[0] - 1, camera_position_plot[0] + 2])
ax1.set_ylim([camera_position_plot[1] - 1, camera_position_plot[1] + 2])
ax1.set_zlim([camera_position_plot[2] - 1, camera_position_plot[2] + 2])
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Z-axis')
ax1.set_zlabel('Y-axis (Vertical)')
ax1.set_title('Original Positions and Vectors')
ax1.legend()

# 2. 3D Plot of Transferred Vectors and Projections
ax2 = fig.add_subplot(132, projection='3d')

# Plot transferred vectors to the origin
ax2.quiver(0, 0, 0,
           transferred_vector_to_object_plot[0], transferred_vector_to_object_plot[1], transferred_vector_to_object_plot[2],
           color='orange', label='Transferred Vector to Object', arrow_length_ratio=0.1)

ax2.quiver(0, 0, 0,
           transferred_camera_z_axis_plot[0], transferred_camera_z_axis_plot[1], transferred_camera_z_axis_plot[2],
           color='cyan', label='Transferred Camera Z-axis', arrow_length_ratio=0.1)

# Plot projections onto XZ plane
ax2.quiver(0, 0, 0,
           projection_vector_to_object_xz_plot[0], projection_vector_to_object_xz_plot[1], projection_vector_to_object_xz_plot[2],
           color='darkgreen', label='Projection of Vector to Object (XZ)', arrow_length_ratio=0.1, linestyle='dashed')

ax2.quiver(0, 0, 0,
           projection_camera_z_axis_xz_plot[0], projection_camera_z_axis_xz_plot[1], projection_camera_z_axis_xz_plot[2],
           color='darkblue', label='Projection of Camera Z-axis (XZ)', arrow_length_ratio=0.1, linestyle='dashed')

# Formatting and labels
ax2.set_xlim([-1, 2])
ax2.set_ylim([-1, 2])
ax2.set_zlim([-1, 2])
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Z-axis')
ax2.set_zlabel('Y-axis (Vertical)')
ax2.set_title('Transferred Vectors and Projections')
ax2.legend()

# Adjust view angle for better visualization
ax1.view_init(elev=20, azim=-60)
ax2.view_init(elev=20, azim=-60)

# 3. 2D Plot of Projections onto XZ Plane
ax3 = fig.add_subplot(133)

# Plot projections onto XZ plane (horizontal plane)
ax3.quiver(0, 0,
           projection_vector_to_object_xz[0], projection_vector_to_object_xz[2],
           color='darkgreen', angles='xy', scale_units='xy', scale=1,
           label='Projection of Vector to Object (XZ)')

ax3.quiver(0, 0,
           projection_camera_z_axis_xz[0], projection_camera_z_axis_xz[2],
           color='darkblue', angles='xy', scale_units='xy', scale=1,
           label='Projection of Camera Z-axis (XZ)')

# Annotate the angle between the vectors
ax3.text(0.0, 0.3, f'Angle: {theta_xz_degrees:.2f}°', fontsize=12)

# Formatting and labels
ax3.set_xlim([-1, 1.5])
ax3.set_ylim([-1, 1.5])
ax3.set_xlabel('X-axis')
ax3.set_ylabel('Z-axis')
ax3.set_title('Projections onto XZ Plane')
ax3.legend()
ax3.grid()

plt.tight_layout()
plt.savefig("vectors.png")
plt.show()

# Print dot product and cosine of angle in XZ plane
print(f"Dot Product (XZ Plane): {dot_product_xz:.3f}")
print(f"Cosine of Angle (XZ Plane): {cos_theta_xz:.3f}")
print(f"Angle between vectors in XZ Plane: {theta_xz_degrees:.2f} degrees")





# import numpy as np
# import matplotlib.pyplot as plt

# # Example vectors
# z_cam = np.array([0, 0, 1])  # Camera's z-axis
# d_cam_obj = np.array([0.5, 0, 0.86])  # Vector to object (3D)

# # Normalize vector and project onto XZ plane
# d_cam_obj_normalized = d_cam_obj / np.linalg.norm(d_cam_obj)
# z_cam_xz = np.array([z_cam[0], z_cam[2]])
# d_cam_obj_xz = np.array([d_cam_obj_normalized[0], d_cam_obj_normalized[2]])

# # Dot product in XZ plane
# dot_product = np.dot(z_cam_xz, d_cam_obj_xz)

# # Plot
# plt.figure(figsize=(8, 6))
# plt.quiver(0, 0, z_cam_xz[0], z_cam_xz[1], angles='xy', scale_units='xy', scale=1, color='blue', label="Camera Z-axis")
# plt.quiver(0, 0, d_cam_obj_xz[0], d_cam_obj_xz[1], angles='xy', scale_units='xy', scale=1, color='orange', label="Object Vector")
# plt.legend()
# plt.title(f"Dot Product in XZ Plane: {dot_product:.2f}")
# plt.grid()
# plt.xlabel("X")
# plt.ylabel("Z")
# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Define object and camera positions in space
# camera_position = np.array([-0.4, -0.5, -0.1]) 
# object_position = np.array([0.3, 0.75, 0.5])  # Object position in space
# camera_position_xz = np.array([0.1, 0.3, 0]) 
# object_position_xz = np.array([0.3, 0.6, 0])  # Camera at the origin

# # Vector from camera to object
# vector_to_object = object_position - camera_position
# vector_to_object_xz = object_position_xz - camera_position_xz

# # Start 3D plotting
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot the camera's z-axis
# ax.quiver(camera_position[0], camera_position[1], camera_position[2],
#           0, 0.5, 0, color='blue', label='Camera Z-axis', arrow_length_ratio=0.1)

# # Plot the object position
# ax.scatter(object_position[0], object_position[1], object_position[2],
#            color='red', label='Object Position', s=50)

# # Plot the vector connecting the camera to the object
# ax.quiver(camera_position[0], camera_position[1], camera_position[2],
#           vector_to_object[0], vector_to_object[1], vector_to_object[2],
#           color='green', label='Vector to Object', arrow_length_ratio=0.1)

# # Formatting and labels
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])
# ax.set_xlabel('X')
# ax.set_ylabel('Z')
# ax.set_zlabel('Y')
# ax.set_title('3D Representation of Camera Z-Axis, Object, and Connecting Vector')
# ax.legend()

# # Show plot
# plt.show()
