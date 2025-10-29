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

# Constants copied from data_utils.py
WINDOW_SIZE = 50  # (pixels) The model architecture depends on this!

# Bin dimensions
BIN_WIDTH_M = 0.140
BIN_LENGTH_M = 0.240
BIN_HEIGHT = 0.065
BIN_WIDTH_PIX = 189
BIN_LENGTH_PIX = 326
CAM2BIN_DIST_MM = 320

# For cropping the bin from rgb and depth image
CROP_XMIN = 274
CROP_XMAX = 463
CROP_YMIN = 0
CROP_YMAX = 326

BIN_PADDING = 50

ACTION_Z_MIN = -BIN_HEIGHT
ACTION_Z_MAX = 0.02
Z_BELOW_SURFACE = 0.030

GG_MODEL_PATH = "/home/snaak/Documents/manipulation_ws/src/snaak_shredded_grasp/models/mass_estimation_model.pth"


class CoordConverter:
    def __init__(
        self,
        bin_width_m=BIN_WIDTH_M,
        bin_length_m=BIN_LENGTH_M,
        bin_height_m=BIN_HEIGHT,
        bin_width_pix=BIN_WIDTH_PIX,
        bin_length_pix=BIN_LENGTH_PIX,
    ):
        self.bin_width_m = bin_width_m
        self.bin_length_m = bin_length_m
        self.bin_height_m = bin_height_m
        self.bin_width_pix = bin_width_pix
        self.bin_length_pix = bin_length_pix

    def m_to_pix(self, x_m, y_m):
        x_pix = int(x_m * self.bin_width_pix / self.bin_width_m)
        y_pix = int(y_m * self.bin_length_pix / self.bin_length_m)
        return x_pix, y_pix

    def pix_to_m(self, x_pix, y_pix):
        x_m = x_pix * self.bin_width_m / self.bin_width_pix
        y_m = y_pix * self.bin_length_m / self.bin_length_pix
        return x_m, y_m

    def action_xy_to_pix(self, action_x_m, action_y_m, img_w, img_h):
        x_disp_pix, y_disp_pix = self.m_to_pix(action_x_m, action_y_m)
        action_x_pix = img_w // 2 - x_disp_pix
        action_y_pix = img_h // 2 + y_disp_pix

        # Clip the action points to the image boundaries
        action_x_pix = np.clip(action_x_pix, 0, img_w - 1)
        action_y_pix = np.clip(action_y_pix, 0, img_h - 1)

        return (action_x_pix, action_y_pix)


def create_transform_rgb():
    """Create RGB transformation pipeline"""
    transform_list = [
        T.ToTensor(),
    ]
    transform = T.Compose(transform_list)
    return transform


def create_transform_depth():
    """Create depth transformation pipeline"""
    transform_list = [
        T.ToTensor(),
        T.Lambda(lambda x: x - (CAM2BIN_DIST_MM - 20)),
        T.Lambda(
            lambda x: torch.where(x < 0, torch.tensor(BIN_HEIGHT * 1000), x)
        ),  # Replace negative values with depth to bin bottom
    ]
    transform = T.Compose(transform_list)
    return transform


def get_patches_from_image_and_depth_maps(
    rgb_img, depth_img, n_patch_length=7, n_patch_width=5, bin_padding=50
):
    """
    Sample 7 points along the bin's length (x axis) and 5 points along the width (y axis),
    equally spaced. Extract centered patches around each point.
    """
    assert rgb_img.shape == (
        BIN_LENGTH_PIX,
        BIN_WIDTH_PIX,
        3,
    ), f"RGB image shape is incorrect: {rgb_img.shape}"
    assert depth_img.shape == (
        BIN_LENGTH_PIX,
        BIN_WIDTH_PIX,
    ), f"Depth image shape is incorrect: {depth_img.shape}"
    assert (
        rgb_img.shape[0:2] == depth_img.shape[0:2]
    ), "RGB and depth image shapes do not match"

    # Compute coordinates for patch centers (y, x)
    padding = bin_padding
    x_vals = np.linspace(padding, BIN_WIDTH_PIX - padding, n_patch_width)
    y_vals = np.linspace(padding, BIN_LENGTH_PIX - padding, n_patch_length)

    # Convert to int and clamp within safe valid range
    x_vals = np.round(x_vals).astype(int)
    y_vals = np.round(y_vals).astype(int)

    patches_rgb = []
    patches_depth = []
    centers = []

    for y in y_vals:
        centers_row = []
        patches_rgb_row = []
        patches_depth_row = []
        for x in x_vals:
            top = y - WINDOW_SIZE // 2
            left = x - WINDOW_SIZE // 2
            rgb_patch = rgb_img[top : top + WINDOW_SIZE, left : left + WINDOW_SIZE, :]
            depth_patch = depth_img[top : top + WINDOW_SIZE, left : left + WINDOW_SIZE]
            patches_rgb_row.append(rgb_patch)
            patches_depth_row.append(depth_patch)
            centers_row.append((x, y))
        centers.append(centers_row)
        patches_rgb.append(patches_rgb_row)
        patches_depth.append(patches_depth_row)

    return patches_rgb, patches_depth, centers


def calculate_loss(row_i, col_i, w_desired, pred_weights, lambda_neighbor=0.25):
    """Calculate loss for a specific cell considering neighbors"""
    cell_loss = abs(pred_weights[row_i][col_i] - w_desired)
    neighbor_loss = 0
    n_neighbors = 0
    for neighbor_row in [row_i - 1, row_i + 1]:
        for neighbor_col in [col_i - 1, col_i + 1]:
            # skip if neighbor is out of bounds
            if not (
                0 <= neighbor_row < pred_weights.shape[0]
                and 0 <= neighbor_col < pred_weights.shape[1]
            ):
                continue

            neighbor_loss += abs(pred_weights[neighbor_row][neighbor_col] - w_desired)
            n_neighbors += 1

    if n_neighbors > 0:
        neighbor_loss = neighbor_loss / n_neighbors

    total_loss = cell_loss + lambda_neighbor * neighbor_loss
    return total_loss


class ConvBlock(nn.Module):
    """Convolutional block with Conv2d, BatchNorm, and ReLU activation."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.block(x)


class ImageBranch(nn.Module):
    """Branch for processing either RGB or depth images."""

    def __init__(self, in_channels=3, num_blocks=5, base_channels=16):
        super(ImageBranch, self).__init__()

        channels = [in_channels] + [base_channels] * num_blocks
        self.layers = []
        for i in range(num_blocks):
            self.layers.append(ConvBlock(channels[i], channels[i + 1]))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class MassEstimationModel(nn.Module):
    """
    Neural network for estimating grasp mass from RGB and depth images.

    Architecture:
    - Dual-stream CNN processing RGB and depth images separately
    - 5 repeated ConvBlocks for each branch
    - Feature concatenation
    - Fully connected layers for regression
    """

    def __init__(
        self,
        rgb_channels=3,
        depth_channels=1,
        base_channels=16,
        fc_hidden=128,
        fc_hidden2=64,
        num_blocks=4,
    ):
        super(MassEstimationModel, self).__init__()

        # RGB image branch
        self.rgb_branch = ImageBranch(rgb_channels, num_blocks, base_channels)

        # Depth image branch
        self.depth_branch = ImageBranch(depth_channels, num_blocks, base_channels)

        # Calculate the size after conv blocks and maxpool
        # Assuming input size of 50x50, after 4 conv blocks and 1 maxpool:
        # Over 5 blocks, output size: 50 (input size) -> 25 -> 12 -> 6 -> 3 (output size)
        conv_output_size = 3 * 3 * base_channels * 2  # *2 for concatenation

        # Fully connected layers
        self.combined_branch = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        )

    def forward(self, rgb_img, depth_img):
        """
        Forward pass through the network.

        Args:
            rgb_img: RGB image tensor of shape (batch_size, 3, 150, 150)
            depth_img: Depth image tensor of shape (batch_size, 1, 150, 150)

        Returns:
            mass_estimate: Estimated grasp mass tensor of shape (batch_size, 1)
        """
        # Process RGB and depth images through separate branches
        rgb_features = self.rgb_branch(rgb_img)
        depth_features = self.depth_branch(depth_img)

        # Flatten features
        rgb_features = torch.flatten(rgb_features, start_dim=1)
        depth_features = torch.flatten(depth_features, start_dim=1)

        # Concatenate features from both branches along the feature dimension
        combined_features = torch.cat([rgb_features, depth_features], dim=1)

        # Pass through fully connected layers
        mass_estimate = self.combined_branch(combined_features)
        return mass_estimate


class GranularGraspMethod:
    """
    Neural network class for granular grasping that predicts the best x,y coordinates
    for a desired weight given RGB and depth images.
    """

    def __init__(self, model_path=None, device=None):
        """
        Initialize the neural network with a trained model.

        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on (cuda/cpu)
        """
        if model_path is None:
            model_path = GG_MODEL_PATH

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Load the model
        self.model = MassEstimationModel()
        self.model.load_state_dict(torch.load(model_path)["model_state_dict"])
        self.model.eval()
        self.model.to(self.device)

        # Initialize transforms
        self.transform_rgb = create_transform_rgb()
        self.transform_depth = create_transform_depth()

        # Initialize coordinate converter
        self.coord_converter = CoordConverter()

    def __get_best_xy_for_weight(self, rgb_img, depth_img, w_desired):
        """
        Find the best x,y coordinates for a desired weight given RGB and depth images.

        Args:
            rgb_img: RGB image (numpy array)
            depth_img: Depth image (numpy array)
            w_desired: Target weight to grasp

        Returns:
            tuple: (best_x, best_y) coordinates in meters
        """

        centers, pred_weights = self.__infer_on_bin(
            rgb_img,
            depth_img,
        )

        # Calculate a score for each point on the grid based on the desired weight
        best_x, best_y, min_loss = 0, 0, float("inf")
        losses = np.zeros((centers.shape[0], centers.shape[1])) * 1000.0
        for row_i in range(centers.shape[0]):
            for col_i in range(centers.shape[1]):
                loss = calculate_loss(row_i, col_i, w_desired, pred_weights)
                losses[row_i, col_i] = loss
                if loss < min_loss:
                    min_loss = loss
                    best_x, best_y = centers[row_i, col_i]

        return best_x, best_y

    def __infer_on_bin(self, rgb_img, depth_img):
        """
        Infer on a bin given RGB and depth images.

        Args:
            rgb_img: RGB image (numpy array)
            depth_img: Depth image (numpy array)
        """
        rgb_img = rgb_img[CROP_YMIN:CROP_YMAX, CROP_XMIN:CROP_XMAX, :]
        depth_img = depth_img[CROP_YMIN:CROP_YMAX, CROP_XMIN:CROP_XMAX]

        # Get patches from the image and depth maps
        patches_rgb, patches_depth, centers = (
            self.__get_patches_from_image_and_depth_maps(
                rgb_img, depth_img, bin_padding=BIN_PADDING
            )
        )

        # Run inference on the patches
        pred_weights = []
        for row_i in range(len(patches_rgb)):
            pred_weights_row = []
            for col_i in range(len(patches_rgb[row_i])):
                rgb_patch = patches_rgb[row_i][col_i]
                depth_patch = patches_depth[row_i][col_i]

                # Preprocess the patch
                rgb_patch = self.transform_rgb(rgb_patch)
                depth_patch = self.transform_depth(depth_patch)
                rgb_patch = rgb_patch.to(torch.float32).to(self.device)
                depth_patch = depth_patch.to(torch.float32).to(self.device)

                rgb_patch = rgb_patch.unsqueeze(0)
                depth_patch = depth_patch.unsqueeze(0)

                pred = self.model(rgb_patch, depth_patch)
                pred = pred.squeeze(-1)

                pred_weights_row.append(pred.item())
            pred_weights.append(pred_weights_row)

        centers = np.array(centers)
        pred_weights = np.array(pred_weights)

        return centers, pred_weights

    def __get_patches_from_image_and_depth_maps(
        self, rgb_img, depth_img, n_patch_length=7, n_patch_width=5, bin_padding=50
    ):
        """
        Sample n_patch_length points along the bin's length (x axis) and n_patch_width points along the width (y axis),
        equally spaced. Extract centered patches around each point.

        Images given to this function are in OpenCV format:
        - rgb_img shape: (BIN_LENGTH_PIX, BIN_WIDTH_PIX, 3), dtype=uint8, BGR order.
        - depth_img shape: (BIN_LENGTH_PIX, BIN_WIDTH_PIX), dtype=float32 or uint16.
        """

        assert rgb_img.shape == (
            BIN_LENGTH_PIX,
            BIN_WIDTH_PIX,
            3,
        ), f"RGB image shape is incorrect: {rgb_img.shape}"
        assert depth_img.shape == (
            BIN_LENGTH_PIX,
            BIN_WIDTH_PIX,
        ), f"Depth image shape is incorrect: {depth_img.shape}"
        assert (
            rgb_img.shape[0:2] == depth_img.shape[0:2]
        ), "RGB and depth image shapes do not match"

        # Compute coordinates for patch centers (y, x)
        # Length = rows (Y), Width = columns (X) in image patch
        padding = bin_padding
        x_vals = np.linspace(padding, BIN_WIDTH_PIX - padding, n_patch_width)
        y_vals = np.linspace(padding, BIN_LENGTH_PIX - padding, n_patch_length)

        # Convert to int and clamp within safe valid range
        x_vals = np.round(x_vals).astype(int)
        y_vals = np.round(y_vals).astype(int)

        patches_rgb = []
        patches_depth = []
        centers = []

        for y in y_vals:
            centers_row = []
            patches_rgb_row = []
            patches_depth_row = []
            for x in x_vals:
                top = y - WINDOW_SIZE // 2
                left = x - WINDOW_SIZE // 2
                rgb_patch = rgb_img[
                    top : top + WINDOW_SIZE, left : left + WINDOW_SIZE, :
                ]
                depth_patch = depth_img[
                    top : top + WINDOW_SIZE, left : left + WINDOW_SIZE
                ]
                patches_rgb_row.append(rgb_patch)
                patches_depth_row.append(depth_patch)
                centers_row.append((x, y))
            centers.append(centers_row)
            patches_rgb.append(patches_rgb_row)
            patches_depth.append(patches_depth_row)

        return patches_rgb, patches_depth, centers

    def __get_z_from_depth(self, x, y, depth_img):
        # TODO: Implement depth extraction logic
        cropped_depth = depth_img[CROP_YMIN:CROP_YMAX, CROP_XMIN:CROP_XMAX]  # For bin 2

        action_xpix, action_ypix = self.coord_converter.action_xy_to_pix(
            x, y, cropped_depth.shape[1], cropped_depth.shape[0]
        )

        # Get depth at action point
        depth_wrt_cam = cropped_depth[action_ypix, action_xpix]  # mm
        action_z = CAM2BIN_DIST_MM - depth_wrt_cam

        # Convert mm to m
        action_z /= 1000.0

        action_z -= Z_BELOW_SURFACE  # Go deeper than the surface of lettuce

        if not (ACTION_Z_MIN <= action_z <= ACTION_Z_MAX):
            action_z = np.clip(action_z, ACTION_Z_MIN, ACTION_Z_MAX)

        return action_z

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
        best_x_pix, best_y_pix = self.__get_best_xy_for_weight(
            rgb_img, depth_img, w_desired
        )

        action_x, action_y = self.coord_converter.pix_xy_to_action(
            best_x_pix, best_y_pix
        )
        action_z = self.__get_z_from_depth(action_x, action_y, depth_img)

        return action_x, action_y, action_z
