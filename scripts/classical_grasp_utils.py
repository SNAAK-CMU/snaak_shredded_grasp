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

BIN_DEPTH = 0.058
END_EFFECTOR_RADIUS = 0.045
BIN_WIDTH = 0.140
BIN_LENGTH = 0.240

class ClassicalGraspGenerator(GraspGenerator):
    def __init__(self):
        super().__init__()
    
    def crop(self, image, location_id, resize=(224, 340)):
        xmin, ymin, xmax, ymax = BIN_COORDS[location_id - 1] + np.array(BIN_OFFSET)
        cropped = image[ymin:ymax, xmin:xmax]
        if location_id in [4, 5, 6]:
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if resize is not None:
            cropped = cv2.resize(cropped, resize)

        return cropped

    def sample_point_from_bin(self, image, bin_number, verbose=False):
        """
        Samples a pixel inside the given bin according to a softmax distribution
        over depth values (lower depth = higher probability).
        
        Args:
            image: 2D numpy array (depth map).
            bin_number: int, which bin to sample from (1-indexed).
            verbose: bool, visualize sampled point.
            
        Returns:
            dist: float, pixel distance from image center to sampled point.
            (x_global, y_global): coordinates of sampled point in original image.
        """
        # xmin, ymin, xmax, ymax = np.array(BIN_COORDS[bin_number-1]) + np.array(BIN_OFFSET)

        cropped = self.crop(image, bin_number) 
        cropped = cropped #+ np.array(BIN_OFFSET)
        xmin, ymin, xmax, ymax = BIN_COORDS[bin_number - 1] + np.array(BIN_OFFSET)

        cropped = cv2.GaussianBlur(cropped, (5,5), 0)
        
        cropped = cv2.medianBlur((cropped).astype(np.uint8), 7)

        disp = cv2.cvtColor((cropped / np.max(cropped) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # use negative depth so smaller depths have higher weights
        depth = -cropped.astype(np.float32)

        # softmax over inverted depth
        flat_vals = depth.flatten()
        exp_vals = np.exp(flat_vals - np.max(flat_vals))  # stability trick
        probs = exp_vals / np.sum(exp_vals)

        idx = np.argmax(probs)
        y_sampled, x_sampled = np.unravel_index(idx, depth.shape)

        # offsets from center of cropped bin
        h_crop, w_crop = cropped.shape[:2]
        dx = x_sampled - (w_crop / 2.0)   # right-positive
        dy = y_sampled - (h_crop / 2.0)   # down-positive (can flip later for robotics)

        # map back to global image coordinates
        y_global = ymin + y_sampled
        x_global = xmin + x_sampled

        if verbose:
            depth_normalized = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # Draw bin boundaries
            cv2.rectangle(depth_colored, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
            
            # Mark the selected point
            cv2.circle(depth_colored, (x_global, y_global), 8, (255, 255, 255), -1)
            cv2.circle(depth_colored, (x_global, y_global), 6, (0, 0, 0), -1)
            cv2.circle(depth_colored, (x_global, y_global), 4, (0, 255, 0), -1)
            
            cv2.imshow("Colorful Depth Map with Sampled Point", depth_colored)
            # cv2.imshow("cropped", disp)
            # cv2.waitKey(0)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()

        return (dx, dy), (x_global, y_global)    

    def get_pickup_point_classical(self, depth_map, bin_number, verbose=False):
        '''
        Gets sampled dx, dy, dz from bin center based on distribution of depths
        '''
        z0 = 0.315

        # return np.array([0.0, 0.0, 0.0])
        # sample pixel and get pixel offsets
        depth_map[depth_map < 300] = 400
        depth_map = np.clip(depth_map, 300, 400)    
        (dx, dy), (x, y) = self.sample_point_from_bin(depth_map, bin_number, verbose)

        K = np.array([[430.086, 0, 418.886],
                    [0, 429.779, 242.779],
                    [0, 0, 1]])
        fx = K[0, 0]
        fy = K[1, 1]

        z = float(depth_map[y, x]) / 1000.0  # y=row, x=col

        # robotics-friendly conversion, positive x is right, and up is positive y (in cameras frame)
        dx_m = (dx / fx) * z
        dy_m = -(dy / fy) * z

        # convert relative to center of bin, with forward being positive x and left being positive y
        if bin_number in {1, 2, 3}:
            dx_m = -dx_m
            dy_m = -dy_m
        else:
            temp = dx_m
            dx_m = dy_m
            dy_m = -temp
        
        dz_m = z0 - z - 0.025 

        return (dx_m, dy_m, dz_m)
    
    def clip_action(self, action):

        """
        Clip the action to be within the bin boundaries.

        Args:
            action: tuple (x, y, z) in bin coordinate frame
        Returns:
            tuple: clipped (x, y, z) in bin coordinate frame
        """
        x, y, z = action

        y = np.clip(x, -BIN_LENGTH/2 - END_EFFECTOR_RADIUS, BIN_LENGTH/2 - END_EFFECTOR_RADIUS)
        x = np.clip(y, -BIN_WIDTH/2 - END_EFFECTOR_RADIUS, BIN_WIDTH/2 - END_EFFECTOR_RADIUS)
        z = np.clip(z, 0.0, BIN_DEPTH)

        return (x, y, z)
    
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

        pickup_point = self.get_pickup_point_classical(depth_img, location_id, verbose=True)
        pickup_point = self.clip_action(pickup_point)
        return pickup_point