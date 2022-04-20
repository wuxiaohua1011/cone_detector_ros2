"""
 Copyright (c) 2022 Ultralytics

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
 """
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
