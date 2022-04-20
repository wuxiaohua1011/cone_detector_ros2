#!/usr/bin/env python3
import rclpy
import rclpy.node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class TestVideoPlayerNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("test_video_player")
        self.declare_parameter("video_path", "NA")
        self.declare_parameter("debug", False)

        self.debug = self.get_parameter("debug").get_parameter_value().bool_value

        self.timer = self.create_timer(0.1, self.callback)
        self.video_path = self.get_parameter("video_path").get_parameter_value().string_value
        self.get_logger().info(f"{self.video_path}")
        self.cap = cv2.VideoCapture(self.video_path)
        self.img_pub = self.create_publisher(Image, "test_image", 10)
        self.bridge = CvBridge()

    def callback(self):
        ret_val, frame = self.cap.read()

        if ret_val:
            if self.debug:
                cv2.imshow("frame", frame)
                cv2.waitKey(1)
            rgb_img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            rgb_img_msg.header.frame_id = "base_link"
            self.img_pub.publish(rgb_img_msg)
        else:
            self.get_logger().info(f"Cannot open video at {self.video_path}")
            self.video_path = self.get_parameter("video_path").get_parameter_value().string_value
            self.cap = cv2.VideoCapture(self.video_path)


def main(args=None):
    rclpy.init()
    node = TestVideoPlayerNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
