#!/usr/bin/env python3

from tomlkit import datetime
import rclpy
import rclpy.node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rclpy
import rclpy.node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import Image, PointCloud2
from datetime import datetime
import cv2


class TestNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("testnode")
        subs = [
            message_filters.Subscriber(self, Image, "/zed2i/zed_node/left/image_rect_color"),
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
