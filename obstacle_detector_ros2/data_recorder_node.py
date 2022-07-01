#!/usr/bin/env python3
from distutils.log import debug
from lib2to3.pytree import convert
from builtin_interfaces.msg import Duration as BuiltInDuration
from numpy import imag
from std_msgs.msg import Header
import rclpy
import rclpy.node
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rclpy
import rclpy.node

import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np
from datetime import datetime
import cv2
from pathlib import Path


class DataRecorderNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("data_recorder_node")
        self.declare_parameter("rgb_camera_topic", "rgb_image")
        self.declare_parameter("output_dir", "./data/lane_detection_data")
        self.declare_parameter("record_interval", -1)

        self.record_interval = (
            self.get_parameter("record_interval").get_parameter_value().integer_value
        )
        self.rgb_camera_topic = (
            self.get_parameter("rgb_camera_topic").get_parameter_value().string_value
        )
        self.output_dir = (
            self.get_parameter("output_dir").get_parameter_value().string_value
        )
        if Path(self.output_dir).exists() == False:
            Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        self.img_sub = self.create_subscription(
            Image,
            self.rgb_camera_topic,
            self.img_callback,
            qos_profile=rclpy.qos.qos_profile_sensor_data,
        )
        self.bridge = CvBridge()
        self.num_images_recorded = 0
        self.counter = 0
        self.get_logger().info("setup done")

    def img_callback(self, left_cam_msg):
        image = self.bridge.imgmsg_to_cv2(left_cam_msg, desired_encoding="bgr8")

        frame = self.image_resize(image=image, width=800)
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.putText(
            frame,
            text=f"Press [s] to save",
            org=(10, 20),
            fontFace=font,
            fontScale=0.5,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        frame = cv2.putText(
            frame,
            text=f"Num Images: {self.num_images_recorded}. Image Resized",
            org=(10, 40),
            fontFace=font,
            fontScale=0.5,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        cv2.imshow("image", frame)
        k = cv2.waitKey(1)
        if k == ord("s"):
            self.record_img(image=image)

        if self.record_interval > 0 and self.counter % self.record_interval == 0:
            self.record_img(image=image)
        self.counter += 1

    def record_img(self, image):
        n = datetime.now()
        filename = f"{self.output_dir}/frame_{n.timestamp()}.jpg"
        cv2.imwrite(filename, image)
        self.get_logger().info(f"Image written to {filename}")
        self.num_images_recorded += 1

    @staticmethod
    def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized


def main(args=None):
    rclpy.init()
    node = DataRecorderNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
