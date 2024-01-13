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
    # grid_size = int(max(abs(camera_distance) * 2.5, 500))  # A bit arbitrary, but provides a clear visualization
    grid_size = 200
    voxel_grid = np.ones((grid_size, grid_size, grid_size))
    # camera_coordinates = [camera_position(0) / 100, camera_position(1) / 100, camera_position(2) / 100,
    #                       camera_position(3) / 100]
    #
    # for camera in camera_coordinates:
    #     x, y, z = camera_to_grid(camera, grid_size)
    #     voxel_grid[x, y, z] = 2  # Marking the camera positions
    return voxel_grid

def transform_to_camera_coordinates(voxel_positions, camera_index):
    camera_coordinates = voxel_positions - camera_position(camera_index=camera_index)
    homogeneous_coordinates = np.hstack([camera_coordinates, np.ones((camera_coordinates.shape[0], 1))])
    return homogeneous_coordinates

def calculate_pixel_coordinates(camera_index):
    voxel_positions = np.argwhere(voxel_grid == 1)
    voxel_positions = np.transpose(voxel_positions)
    homogeneous_coordinates = transform_to_camera_coordinates(voxel_positions, camera_index=camera_index)
    sensor_positions = (calculate_intrinsic_matrix() @ homogeneous_coordinates).T
    pixel_coordinates = sensor_positions[:, :2] / sensor_positions[:, 2, np.newaxis]

    return pixel_coordinates

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

    camera0_translation = [[-420], [0], [0]]
    camera1_translation = [[0], [420], [0]]
    camera2_translation = [[420], [0], [0]]
    camera3_translation = [[0], [-420], [0]]

    camera_translations = [camera0_translation, camera1_translation, camera2_translation, camera3_translation]
    theta = [0, 90, 180, 270]
    # # The field of view (FOV) can be calculated using the focal length and sensor size
    fov_x = 2 * np.arctan((sensor_width_mm / 2) / focal_length_mm)
    fov_y = 2 * np.arctan((sensor_height_mm / 2) / focal_length_mm)

    voxel_grid = create_voxel_grid()
    print(voxel_grid.shape)
    pixel_coordinates = calculate_pixel_coordinates(camera_index=0)












