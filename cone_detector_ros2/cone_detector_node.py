#!/usr/bin/env python3
import rclpy
import rclpy.node
import sys
import cv2
import numpy as np
from pathlib import Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ConeDetectorNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("cone_detector_node")


def main(args=None):
    rclpy.init()
    node = ConeDetectorNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
