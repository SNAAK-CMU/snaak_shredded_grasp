#!/usr/bin/env python3

import torch
from snaak_shredded_grasp_constants import BIN_COORDS
import cv2


class GraspGenerator:
    def __init__(self):
        pass

    def get_action(
        self,
        rgb_image,
        depth_image,
        weight,
        ingredient_name,
        pickup_weight,
        location_id,
    ):
        pass

    def crop(self, image, location_id, resize=(224, 340)):
        xmin, ymin, xmax, ymax = BIN_COORDS[location_id - 1]
        cropped = image[ymin:ymax, xmin:xmax]
        if location_id in [4, 5, 6]:
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if resize is not None:
            cropped = cv2.resize(cropped, resize)

        return cropped


class DefaultGraspGenerator(GraspGenerator):
    def __init__(self):
        super().__init__()

    def get_action(
        self,
        rgb_image,
        depth_image,
        weight,
        ingredient_name,
        pickup_weight,
        location_id,
    ):
        return 0.0, 0.0, -0.04
