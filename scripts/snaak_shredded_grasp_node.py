#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from snaak_shredded_grasp.srv import GetGraspPose
import cv2
from snaak_weight_read.srv import ReadWeight
import numpy as np
import collections
from snaak_shredded_grasp_utils import DefaultGraspGenerator

GRASP_TECHNIQUE = ""

class ShreddedGraspServer(Node):
    def __init__(self):
        super().__init__('snaak_shredded_grasp')

        self._depth_subscription = self.create_subscription(
            Image, "/camera/camera/depth/image_rect_raw", self.depth_callback, 10
        )

        # Subscribe to rgb image topic
        self._rgb_subscription = self.create_subscription(
            Image, "/camera/camera/color/image_rect_raw", self.rgb_callback, 10
        )
        self.bridge = CvBridge()
        self.depth_image = None
        self.rgb_image = None

        self._service = self.create_service(GetGraspPose, 
                                            self.get_name() + "/get_grasp_pose", 
                                            self.get_grasp_pose_callback)

        self._get_weight_left_bins = self.create_client(
            ReadWeight, "snaak_weight_read/snaak_scale_bins_left/read_weight"
        )
        self._get_weight_right_bins = self.create_client(
            ReadWeight, "snaak_weight_read/snaak_scale_bins_right/read_weight"
        )
        self.depth_queue = collections.deque(maxlen=5)

        if GRASP_TECHNIQUE == "GG":
            pass
        elif GRASP_TECHNIQUE == "CLASSICAL":
            pass
        else:
            self.action_generator = DefaultGraspGenerator() # this is a dummy for testing
        
        self.get_logger().info("Shredded Grasp Server is ready to receive requests.")

        
    def get_grasp_pose_callback(self, request, response):
        try:
            location_id = request.location_id
            ingredient_name = request.ingredient_name
            pickup_weight = request.desired_pickup_weight
            self.get_logger().info("Generating grasp action...")

            if self.rgb_image is None:
                self.get_logger().error("No RGB image data received yet")
                raise Exception("No RGB image data")

            if not self.depth_queue:
                self.get_logger().error("No depth image data received yet")
                raise Exception("No depth image data")

            # use dummy weight for now
            weight = 0 # self.get_weight(location_id)

            if weight is None:
                self.get_logger().error("Failed to get weight")
                raise Exception("Failed to get weight")

            self.get_logger().info(
                f"Received request for grasp pose. Location ID: {location_id}, Ingredient: {ingredient_name}"
            )

            # TODO: this may not work
            depth_image = np.mean(self.depth_queue, axis=0).astype(np.float32)
            self.get_logger().info("Generating grasp action...")
            action = self.action_generator.get_action(self.rgb_image, 
                                                      depth_image, 
                                                      weight, 
                                                      ingredient_name, 
                                                      pickup_weight, 
                                                      location_id)
            self.get_logger().info(f"Grasp action generated: {action}")
            
            response.x = 0.0#action[0]
            response.y = 0.0#action[1]
            response.z = 0.0#action[2]
        except Exception as e:
            self.get_logger().error(f"Error processing grasp pose request: {e}")
            response.x = response.y = response.z = float("nan")
        finally:
            return response

    def depth_callback(self, msg):
        """Convert ROS Image message to OpenCV format"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
            self.depth_queue.append(self.depth_image)
            if len(self.depth_queue) > 5:
                self.depth_queue.popleft()

        except CvBridgeError as e:
            self.get_logger().error(
                f"Failed to convert depth message to OpenCV format: {e}"
            )
            self.depth_image = None

    def rgb_callback(self, msg):
        """Convert ROS Image message to OpenCV format"""
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
        except CvBridgeError as e:
            self.get_logger().error(
                f"Failed to convert image message to OpenCV format: {e}"
            )
            self.rgb_image = None

    # TODO: debug why this is hanging
    # def get_weight(self, location_id):
    #     if (location_id == 1 or location_id == 2 or location_id == 3):
    #         client = self._get_weight_right_bins
    #     else:
    #         client = self._get_weight_left_bins

    #     request = ReadWeight.Request()

    #     future = client.call_async(request)

    #     rclpy.spin_until_future_complete(self, future)

    #     # if future.result() is not None:
    #     #     return future.result().weight
    #     # else:
    #     #     self.get_logger().error("Failed to read weight")
    #     #     return None
    #     return future.result().weight.data

if __name__ == '__main__':
    rclpy.init()
    try:
        node = ShreddedGraspServer()
        rclpy.spin(node)
    except Exception as e:
        print(f"Exception in ShreddedGraspServer: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()