#!/usr/bin/env python3
from distutils.log import debug
from lib2to3.pytree import convert

from tomlkit import datetime
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
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import numpy as np
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
import tf_transformations as tr
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Transform
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Vector3
from matplotlib import cm
import numpy as np
from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import Imu
from datetime import datetime
import cv2


class TestNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("testnode")
        subs = [
            message_filters.Subscriber(
                self, Image, "/zed2i/zed_node/left/image_rect_color"
            ),
            message_filters.Subscriber(self, PointCloud2, "/livox/lidar"),
        ]
        self.ts = message_filters.ApproximateTimeSynchronizer(
            subs, 5, 0.2, allow_headerless=False  # queue size
        )
        self.ts.registerCallback(self.callback)
        self.bridge = CvBridge()
        print("setup done")

    def callback(self, left_cam_msg, center_lidar_pcl_msg):
        image = self.bridge.imgmsg_to_cv2(left_cam_msg, desired_encoding="bgr8")

        cv2.imshow("image", image)
        k = cv2.waitKey(1)
        if k == ord("s"):
            n = datetime.now()
            filename = f"/home/roar/Desktop/projects/roar-indy-ws/data/frame_{n.timestamp()}.jpg"
            cv2.imwrite(filename, image)
            print(f"Image written to {filename}")


def main(args=None):
    rclpy.init()
    node = TestNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
