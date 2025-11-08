import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn

from snaak_shredded_grasp_utils import GraspGenerator
from snaak_shredded_grasp_constants import BIN_COORDS, BIN_OFFSET
from granular_grasp_utils import CoordConverter

END_EFFECTOR_RADIUS = 0.045
INGREDIENT_NAME = "onions"

INGREDIENT2BIN_DICT = {
    "lettuce": {"pick_id": 2, "place_id": 5},
    "onions": {"pick_id": 5, "place_id": 2},
}

# Bin dimensions
BIN_WIDTH_M = 0.140
BIN_LENGTH_M = 0.240
BIN_HEIGHT = 0.065
BIN_DIMS_DICT = {
    2: {
        "width_m": BIN_WIDTH_M,
        "length_m": BIN_LENGTH_M,
        "height": BIN_HEIGHT,
        "width_pix": 189,
        "length_pix": 326,
        "cam2bin_dist_mm": 320,
    },
    5: {
        "width_m": BIN_WIDTH_M - 0.020,
        "length_m": BIN_LENGTH_M - 0.020,
        "height": BIN_HEIGHT,
        "width_pix": 130,
        "length_pix": 240,
        "cam2bin_dist_mm": 305,  # Need to measure this
    },
}

# CROP_COORDS_DICT = {
#     2: {
#         "xmin": 274,
#         "xmax": 463,
#         "ymin": 0,
#         "ymax": 326,
#     },
#     5: {  # After rotating the image anti-clockwise by 90 degrees
#         "xmin": 136,
#         "xmax": 266,
#         "ymin": 458,
#         "ymax": 698,
#     },
# }
CROP_COORDS_DICT = {
    2: {
        "xmin": 316,
        "xmax": 413,
        "ymin": 58,
        "ymax": 278,
    },
    5: {
        "xmin": 168,
        "xmax": 245,
        "ymin": 483,
        "ymax": 691,
    }
}

#####################################################################################

Z_BELOW_SURFACE_DICT = {
    "lettuce": 0.030,
    "onions": 0.020,
}
ACTION_Z_MIN = -BIN_HEIGHT
ACTION_Z_MAX = 0.02

# MEASURED EMPRICALLY
MANIPULATION_CORRECTION_DICT = {
    2: {"X": 0.0, "Y": -0.013, "Z": 0.032},
    5: {"X": 0.0, "Y": -0.010, "Z": 0.020},
}

END_EFFECTOR_RADIUS = 0.040
LEFT_BIN_IDS = [4, 5, 6]
RIGHT_BIN_IDS = [1, 2, 3]

MODEL_PATH_DICT = {
    "lettuce": "/home/snaak/Documents/manipulation_ws/src/snaak_rl_data_collection/scripts/mass_estimation_model.pth",
    "onions": "/home/snaak/Documents/manipulation_ws/src/snaak_rl_data_collection/scripts/mass_estimation_model_onions.pth",
}
ENABLE_GRANULAR_GRASP = False
DESIRED_WEIGHT = 10

class ClassicalGraspGenerator(GraspGenerator):
    def __init__(self):
        super().__init__()
        self.end_effector_radius = END_EFFECTOR_RADIUS
        self.action_z_min = ACTION_Z_MIN
        self.action_z_max = ACTION_Z_MAX
    
    def update_params(self, ingredient_name, location_id):
        self.ingredient_name = ingredient_name
        self.pick_bin_id = location_id
        self.bin_dims_dict = BIN_DIMS_DICT[self.pick_bin_id]
        self.coord_converter = CoordConverter(self.bin_dims_dict)
        self.crop_coords_dict = CROP_COORDS_DICT[self.pick_bin_id]
        self.z_below_surface = Z_BELOW_SURFACE_DICT[self.ingredient_name]
        self.cam2bin_dist_mm = self.bin_dims_dict["cam2bin_dist_mm"]
        self.manipulation_correction_dict = MANIPULATION_CORRECTION_DICT[
            self.pick_bin_id
        ]

    def sample_point_from_bin(self, depth_img):
        """
        Samples a pixel inside the given bin according to a softmax distribution
        over depth values (lower depth = higher probability).
        
        Args:
            depth_img: 2D numpy array (depth map).
            
        Returns:
            x_sampled_pix: int, x coordinate of sampled point in depth_img.
            y_sampled_pix: int, y coordinate of sampled point in depth_img.
        """

        # Translate depth map. Shift origin of camera to hover 2 cm above bin top
        depth_img = depth_img - (self.cam2bin_dist_mm - 20)

        # Clip depth values to handle reflections
        depth_img[depth_img <= 0] = (1000 * self.bin_dims_dict["height"]) + 20
        depth_img[depth_img > (1000 * self.bin_dims_dict["height"]) + 20] = (1000 * self.bin_dims_dict["height"]) + 20

        print("Depth min and max: ", np.min(depth_img), np.max(depth_img))

        depth_img = cv2.medianBlur((depth_img).astype(np.uint8), 21)
        # kernel_size = 30        
        # depth_img = cv2.blur(depth_img, (kernel_size, kernel_size))

        # cv2.imshow("Processed Depth", depth_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Take argmax over height
        anti_edge_padding = 20
        height = -depth_img.astype(np.float32)
        height_map_cropped = height[anti_edge_padding:-anti_edge_padding, anti_edge_padding:-anti_edge_padding]
        y_sampled_pix, x_sampled_pix = np.unravel_index(np.argmax(height_map_cropped), height_map_cropped.shape)
        y_sampled_pix += anti_edge_padding
        x_sampled_pix += anti_edge_padding

        # Visualize sampled point
        img_viz = depth_img.copy()
        cv2.circle(img_viz, (x_sampled_pix, y_sampled_pix), 5, (255, 0, 0), -1)
        # cv2.imshow("Blurred Depth", img_viz)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # Get processed depth value at sampled pixel
        depth_wrt_cam = depth_img[y_sampled_pix, x_sampled_pix]

        print("Sampled depth: ", depth_wrt_cam)

        depth_wrt_cam += (self.cam2bin_dist_mm - 20)

        print("Depth wrt cam (reverse translation on sampled depth): ", depth_wrt_cam)

        return x_sampled_pix, y_sampled_pix, depth_wrt_cam

    def get_pickup_point_classical(self, depth_map):
        '''
        Gets sampled dx, dy, dz from bin center based on distribution of depths
        '''
        # Clip depth map to handle reflections
        # depth_map[depth_map < self.cam2bin_dist_mm - 20] = self.cam2bin_dist_mm + 1000 * self.bin_dims_dict["height"]
        # depth_map = np.clip(depth_map, self.cam2bin_dist_mm - 20, self.cam2bin_dist_mm + 1000 * self.bin_dims_dict["height"])    
        
        # Get x,y for height point in topology
        x_pix, y_pix, depth_wrt_cam = self.sample_point_from_bin(depth_map)

        print("Depth to bin bottom: ", self.cam2bin_dist_mm + 1000 * self.bin_dims_dict["height"])
        print("Depth on sampled pt wrt cam: ", depth_wrt_cam)

        # x_pix = self.bin_dims_dict["width_pix"] // 2
        # y_pix = self.bin_dims_dict["length_pix"] // 2

        # Convert pixel to bin coordinates
        action_x, action_y = self.coord_converter.pix_xy_to_action(x_pix, y_pix)

        # Compute action_z
        action_z = (self.cam2bin_dist_mm - depth_wrt_cam) / 1000.0  # in meters
        action_z -= self.z_below_surface  # Go below surface

        return (action_x, action_y, action_z)
    
    def clip_action(self, action_x, action_y, action_z):

        """
        Clip the action to be within the bin boundaries.

        Args:
            action: tuple (x, y, z) in bin coordinate frame
        Returns:
            tuple: clipped (x, y, z) in bin coordinate frame
        """
        bin_width = self.bin_dims_dict["width_m"]
        bin_length = self.bin_dims_dict["length_m"]
        action_x = np.clip(
            action_x,
            -bin_width / 2 + self.end_effector_radius,
            bin_width / 2 - self.end_effector_radius,
        )
        action_y = np.clip(
            action_y,
            -bin_length / 2 + self.end_effector_radius,
            bin_length / 2 - self.end_effector_radius,
        )
        action_z = np.clip(action_z, self.action_z_min, self.action_z_max)

        return (action_x, action_y, action_z)
    
    def get_cropped_bin_depth(self, depth_img):
        if self.pick_bin_id in LEFT_BIN_IDS:
            # Rotate image anti-clockwise by 90 degrees
            depth_img = cv2.rotate(depth_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        crop_xmin = self.crop_coords_dict["xmin"]
        crop_xmax = self.crop_coords_dict["xmax"]
        crop_ymin = self.crop_coords_dict["ymin"]
        crop_ymax = self.crop_coords_dict["ymax"]
        cropped_depth = depth_img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        return cropped_depth

    def get_action(
            self,
            rgb_img,
            depth_img,
            bin_weight,
            ingredient_name,
            w_desired,
            location_id,
        ):

        """
        Get the action (x, y, z) for a desired weight given RGB and depth images.

        Args:
            rgb_img: RGB image (numpy array)
            depth_img: Depth image (numpy array)
            bin_weight: Weight of the bin
            ingredient_name: Name of the ingredient
            w_desired: Desired weight to grasp
            location_id: ID of the location where the ingredient is located
        Returns:
            tuple: (action_x, action_y, action_z) in bin coordinate frame
        """
        self.update_params(ingredient_name, location_id)

        cropped_depth = self.get_cropped_bin_depth(depth_img)

        action_x, action_y, action_z = self.get_pickup_point_classical(cropped_depth)

        print("Action give by classical method:", action_x, action_y, action_z)

        action_x, action_y, action_z = self.clip_action(action_x, action_y, action_z)

        print("Clipped action:", action_x, action_y, action_z)

        # action_x = 0.0
        # action_y = 0.0
        # action_z = -0.065

        # Apply manipulation correction
        action_x += self.manipulation_correction_dict["X"]
        action_y += self.manipulation_correction_dict["Y"]
        action_z += self.manipulation_correction_dict["Z"]

        return (action_x, action_y, action_z)
