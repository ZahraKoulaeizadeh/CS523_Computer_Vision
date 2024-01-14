import numpy as np
import math
import matplotlib.pyplot as plt
import cv2


def rotate_roll(camera_angle):
    camera_angle = math.radians(camera_angle)
    rotation_matrix = np.zeros((3, 3), dtype=np.float32)
    rotation_matrix[0][0] = 1
    rotation_matrix[2][1] = round(math.sin(camera_angle), 5)
    rotation_matrix[1][2] = round(-1 * math.sin(camera_angle), 5)
    rotation_matrix[1][1] = round(math.cos(camera_angle), 5)
    rotation_matrix[2][2] = round(math.cos(camera_angle), 5)

    return rotation_matrix


# def calculate_camera_matrix(camera_index):
#     camera_angle = theta[camera_index]
#     intrinsic_matrix = calculate_intrinsic_matrix()
#     rotation_matrix = rotate_roll(camera_angle)
#     camera_translation = camera_translations[camera_index]
#
#     transformation_matrix = np.concatenate((rotation_matrix, camera_translation), axis=1)
#     camera_matrix = np.matmul(intrinsic_matrix, transformation_matrix)
#
#     return camera_matrix


def calculate_intrinsic_matrix():
    intrinsic_matrix = np.zeros((3, 3))

    intrinsic_matrix[0][0] = fx
    intrinsic_matrix[1][1] = fy
    intrinsic_matrix[0][2] = cx
    intrinsic_matrix[1][2] = cy
    intrinsic_matrix[2][2] = 1

    return intrinsic_matrix


# def camera_position(camera_index):
#     camera_positions = np.dot(rotate_roll(theta[camera_index]), camera_translations[camera_index])
#     return camera_positions


# def camera_to_grid(camera_coordinates, grid_size):
#     return [int(coord + grid_size / 2) for coord in camera_coordinates]


def create_voxel_grid():
    # grid_size = int(max(abs(camera_distance) * 2.5, 500))
    grid_size = 500
    resolution = 5
    grid_size = int(grid_size / resolution)
    grid = np.ones((grid_size, grid_size, grid_size))
    return grid


def transform_to_camera_coordinates(voxel_position_, camera_index):
    homogeneous_coordinates = []
    rotated_voxel_positions = np.dot(rotate_roll(theta[camera_index]), voxel_position_)
    camera_coordinates = rotated_voxel_positions + camera_translations[camera_index]
    homogeneous_coordinates = np.vstack([camera_coordinates, np.ones((1, camera_coordinates.shape[1]))])
    return homogeneous_coordinates


def calculate_pixel_coordinates(voxel_pos, camera_index):
    pixel_coordinates_ = []
    transposed_voxel_position = np.transpose(voxel_pos)
    homogeneous_coordinates = transform_to_camera_coordinates(transposed_voxel_position, camera_index=camera_index)
    sensor_positions = np.matmul(calculate_intrinsic_matrix(), homogeneous_coordinates[:3, :])
    # Debug print
    print(f"Sensor positions: {sensor_positions}")

    # The correct perspective division is to divide by the last row of sensor_positions, not the depth (z-value)
    pixel_coordinates_ = sensor_positions[:2, :] / sensor_positions[2, :]
    # Debug print
    print(f"Pixel coordinates: {pixel_coordinates_}")
    return pixel_coordinates_


def mock_load_mask_files():
    mask_files = {
        'cam0': 'frame12_cam0_msk.jpg',
        'cam1': 'frame12_cam1_msk.jpg',
        'cam2': 'frame12_cam2_msk.jpg',
        'cam3': 'frame12_cam3_msk.jpg'
    }
    masks = {}
    for cam, filepath in mask_files.items():
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(image, 0, 256, cv2.THRESH_BINARY)
        masks[cam] = binary_mask

    return masks


def is_voxel_in_object(voxel_pixel_coords, masks):
    num_voxels = voxel_positions.shape[0]
    voxel_votes = np.zeros(num_voxels, dtype=int)

    for cam, mask in masks.items():
        cam_coords = voxel_pixel_coords[cam]

        valid_coords = (cam_coords[0] > 0) & (cam_coords[0] < mask.shape[0]) & \
                       (cam_coords[1] > 0) & (cam_coords[1] < mask.shape[1])

        valid_cam_coords = cam_coords[:, valid_coords].astype(int)
        values = mask[valid_cam_coords[0], valid_cam_coords[1]]
        print('Valid coords: ', valid_cam_coords.shape)
        is_in_object = mask[valid_cam_coords[0], valid_cam_coords[1]] > 1
        print('if any points in object', np.any(is_in_object == True))
        print('Mask object pixels', np.where(mask >= 1))
        np.add.at(voxel_votes, np.nonzero(valid_coords)[0], is_in_object)

    is_object = voxel_votes >= 3
    return is_object


def plot_voxels(object_voxels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(object_voxels[0], object_voxels[1], object_voxels[2], marker='x')
    plt.show()


def update_voxel_grid(voxel_grid, voxel_positions, is_object):
    for idx, (x, y, z) in enumerate(voxel_positions):
        if is_object[idx]:
            print('here')
            voxel_grid[x, y, z] = 2
    return voxel_grid


if __name__ == '__main__':
    focal_length_mm = 8.2
    sensor_width_mm = 11.3
    sensor_height_mm = 7.1
    sensor_width_px = 1920
    sensor_height_px = 1200
    camera_distance = 420

    fx = (focal_length_mm * sensor_width_px) / sensor_width_mm
    fy = (focal_length_mm * sensor_height_px) / sensor_height_mm

    cx = sensor_width_px / 2
    cy = sensor_height_px / 2

    camera0_translation = [[0], [0], [420]]
    camera1_translation = [[-420], [0], [0]]
    camera2_translation = [[0], [0], [-420]]
    camera3_translation = [[4200], [0], [0]]

    camera_translations = [camera0_translation, camera1_translation, camera2_translation, camera3_translation]
    theta = [10, 100, 190, 280]

    fov_x = 2 * np.arctan((sensor_width_mm / 2) / focal_length_mm)
    fov_y = 2 * np.arctan((sensor_height_mm / 2) / focal_length_mm)

    voxel_grid = create_voxel_grid()
    voxel_positions = np.argwhere(voxel_grid == 1)

    masks = mock_load_mask_files()
    print(voxel_positions.shape[0])
    voxel_pixel_coords = {}

    pixel_coordinates_cam0 = calculate_pixel_coordinates(voxel_positions, camera_index=0)
    pixel_coordinates_cam1 = calculate_pixel_coordinates(voxel_positions, camera_index=1)
    pixel_coordinates_cam2 = calculate_pixel_coordinates(voxel_positions, camera_index=2)
    pixel_coordinates_cam3 = calculate_pixel_coordinates(voxel_positions, camera_index=3)

    # for cam_index in range(4):
    #     pixel_coordinates = calculate_pixel_coordinates(voxel_positions, camera_index=cam_index)
    #     voxel_pixel_coords[f'cam{cam_index}'] = pixel_coordinates

    is_object = is_voxel_in_object(voxel_pixel_coords, masks)
    print(np.any(is_object) == True)
    update_voxel_grid(voxel_grid, voxel_positions, is_object)

    object_voxels = np.argwhere(voxel_grid == 2)
    plot_voxels(np.argwhere(object_voxels))
