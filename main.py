import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

def rotate_roll(camera_angle):
    camera_angle = math.radians(camera_angle)
    rotation_matrix = np.zeros((3, 3), dtype=np.float32)
    rotation_matrix[0][0] = 1
    rotation_matrix[2][1] = round(math.sin(camera_angle), 3)
    rotation_matrix[1][2] = round(-1 * math.sin(camera_angle), 3)
    rotation_matrix[1][1] = round(math.cos(camera_angle), 3)
    rotation_matrix[2][2] = round(math.cos(camera_angle), 3)

    return rotation_matrix


def calculate_camera_matrix(camera_index):
    camera_angle = theta[camera_index]
    intrinsic_matrix = calculate_intrinsic_matrix()
    rotation_matrix = rotate_roll(camera_angle)
    camera_translation = camera_translations[camera_index]

    transformation_matrix = np.concatenate((rotation_matrix, camera_translation), axis=1)
    camera_matrix = np.matmul(intrinsic_matrix, transformation_matrix)

    return camera_matrix


def calculate_intrinsic_matrix():
    intrinsic_matrix = np.zeros((3, 3))

    intrinsic_matrix[0][0] = fx
    intrinsic_matrix[1][1] = fy
    intrinsic_matrix[0][2] = cx
    intrinsic_matrix[1][2] = cy
    intrinsic_matrix[2][2] = 1

    return intrinsic_matrix


def camera_position(camera_index):
    camera_positions = np.matmul(rotate_roll(theta[camera_index]), camera_translations[camera_index])
    return camera_positions

def camera_to_grid(camera_coordinates, grid_size):
    return [int(coord + grid_size / 2) for coord in camera_coordinates]

def create_voxel_grid():
    grid_size = int(max(abs(camera_distance) * 2.5, 500))  # A bit arbitrary, but provides a clear visualization
    # size = 50
    resolution = 10
    grid_size = int(grid_size / resolution)

    grid = np.zeros((grid_size, grid_size, grid_size))
    return grid

def transform_to_camera_coordinates(voxel_positions, camera_index):
    voxel_positions = np.matmul(rotate_roll(theta[camera_index]), voxel_positions)
    camera_coordinates = voxel_positions - camera_position(camera_index=camera_index)
    homogeneous_coordinates = np.hstack([camera_coordinates, np.ones((camera_coordinates.shape[0], 1))])
    return homogeneous_coordinates
    # return camera_coordinates

def calculate_pixel_coordinates(camera_index):
    voxel_positions = np.argwhere(voxel_grid == 1)
    voxel_positions = np.transpose(voxel_positions)
    homogeneous_coordinates = transform_to_camera_coordinates(voxel_positions, camera_index=camera_index)
    sensor_positions = np.matmul(calculate_intrinsic_matrix(), homogeneous_coordinates).T
    pixel_coordinates = sensor_positions[:, :2] / sensor_positions[:, 2, np.newaxis]
    pixel_coordinates = pixel_coordinates.T

    return pixel_coordinates

def mock_load_mask_files():
    mask_files = {
        'cam0': 'frame6_cam0_msk.jpg',
        'cam1': 'frame6_cam1_msk.jpg',
        'cam2': 'frame6_cam2_msk.jpg',
        'cam3': 'frame6_cam3_msk.jpg'
    }
    masks = {}
    for cam, filepath in mask_files.items():
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(image, 0, 256, cv2.THRESH_BINARY)
        masks[cam] = binary_mask
    return masks


def is_voxel_in_object(voxel_pixel_coords, masks):
    count = 0
    for cam, mask in masks.items():
        x, y = voxel_pixel_coords[cam]
        if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1]:
            if mask[int(y), int(x)] == 1:
                count += 1
    return count >= len(masks) / 2

def plot_voxels(object_voxels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(grid[0], grid[1], grid[2], marker='o')
    ax.scatter(object_voxels[0], object_voxels[1], object_voxels[2], marker='x')

    plt.show()


if __name__ == '__main__':
    focal_length_mm = 8.2  # in mm
    sensor_width_mm = 11.3  # in mm
    sensor_height_mm = 7.1  # in mm
    sensor_width_px = 1920  # in pixels
    sensor_height_px = 1200  # in pixels
    camera_distance = 420  # in m

    fx = (focal_length_mm * sensor_width_px) / sensor_width_mm
    fy = (focal_length_mm * sensor_height_px) / sensor_height_mm

    cx = sensor_width_px / 2
    cy = sensor_height_px / 2

    camera0_translation = [[0], [0], [-420]]
    camera1_translation = [[0], [420], [0]]
    camera2_translation = [[0], [0], [420]]
    camera3_translation = [[0], [-420], [0]]

    camera_translations = [camera0_translation, camera1_translation, camera2_translation, camera3_translation]
    theta = [0, 90, 180, 270]
    # # The field of view (FOV) can be calculated using the focal length and sensor size
    fov_x = 2 * np.arctan((sensor_width_mm / 2) / focal_length_mm)
    fov_y = 2 * np.arctan((sensor_height_mm / 2) / focal_length_mm)

    voxel_grid = create_voxel_grid()
    # pixel_coordinates_cam0 = calculate_pixel_coordinates(camera_index=0)
    # pixel_coordinates_cam1 = calculate_pixel_coordinates(camera_index=1)
    # pixel_coordinates_cam2 = calculate_pixel_coordinates(camera_index=2)
    # pixel_coordinates_cam3 = calculate_pixel_coordinates(camera_index=3)


    # Load mask files (mocked)
    masks = mock_load_mask_files()

    # # Process each voxel
    for voxel in np.argwhere(voxel_grid == 0):
        voxel_pixel_coords = {}
        for cam_index in range(4):
            pixel_coords = calculate_pixel_coordinates(camera_index=cam_index)
            voxel_pixel_coords[f'cam{cam_index}'] = (pixel_coords[0], pixel_coords[1])

    #     # Check if the voxel is in the object
        if is_voxel_in_object(voxel_pixel_coords, masks):
            voxel_grid[voxel[0], voxel[1], voxel[2]] = 2  # Mark voxel as part of the object

    # # Plotting the voxel grid
    object_voxels = np.argwhere(voxel_grid == 2)
    plot_voxels(np.argwhere(object_voxels))







