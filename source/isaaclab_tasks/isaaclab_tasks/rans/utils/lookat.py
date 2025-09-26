import numpy as np

def get_lookat_point_flexible(camera_position, camera_orientation, distance, rotation_order='zyx', forward_axis='-z'):
    """
    Calculates the 3D point the camera is looking at, with flexible conventions.

    Args:
        camera_position (tuple): The camera's (x, y, z) position.
        camera_orientation (tuple): The camera's (pitch, yaw, roll) orientation in degrees.
        distance (float): The distance from the camera to the look-at point.
        rotation_order (str): The order of rotation axes (e.g., 'xyz', 'zyx').
        forward_axis (str): The camera's initial forward direction ('+x', '-x', '+y', '-y', '+z', '-z').

    Returns:
        numpy.ndarray: The (x, y, z) coordinates of the look-at point.
    """
    pitch, yaw, roll = np.radians(camera_orientation)

    # 1. Define the initial forward vector based on convention
    if forward_axis == '-z':
        forward_vector = np.array([0, 0, -distance])
    elif forward_axis == '+z':
        forward_vector = np.array([0, 0, distance])
    elif forward_axis == '-y':
        forward_vector = np.array([0, -distance, 0])
    # Add other cases as needed

    # 2. Create the rotation matrices
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    R_y = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    R_z = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])

    # 3. Combine rotations based on the specified order
    if rotation_order == 'zyx':
        combined_rotation_matrix = R_z @ R_y @ R_x
    elif rotation_order == 'xyz':
        combined_rotation_matrix = R_x @ R_y @ R_z
    elif rotation_order == 'yxz':
        combined_rotation_matrix = R_y @ R_x @ R_z
    else:
        raise ValueError("Invalid rotation order.")

    # 4. Rotate the vector and calculate the look-at point
    rotated_forward_vector = combined_rotation_matrix @ forward_vector
    lookat_point = np.array(camera_position) + rotated_forward_vector

    return lookat_point

# Example usage with your values
camera_pos = (125.26,-19.13,6.85)
camera_rot = (42.41,0.0,-179.66)
distance_to_origin = np.linalg.norm(np.array(camera_pos))

# Try a different rotation order and forward axis
# This is a guess based on your initial input/output
lookat_point_guess = get_lookat_point_flexible(
    camera_pos, 
    camera_rot, 
    distance=distance_to_origin, 
    rotation_order='zyx', # This is a common alternative
    forward_axis='-z'
)

print(f"Camera Position: {camera_pos}")
print(f"Camera Orientation: {camera_rot}")
print(f"Calculated Look-at Point: {np.round(lookat_point_guess, 2)}")