#!/usr/bin/env python3
import rclpy
import rclpy.node
import sys
import cv2
import numpy as np
from pathlib import Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from ctypes import *  # convert float to uint32
from std_msgs.msg import Header
from .utils.pcd_util import pointcloud2_to_array
import time
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Vector3


class EuclideanClusteringNode(rclpy.node.Node):
    """Given point cloud, this node will seperate ground plane and obstacle

    Args:
        rclpy (_type_): rclpy node
    """

    def __init__(self):
        super().__init__("euclidean_clustering_node")
        # declare routes and get their values
        self.declare_parameter(
            "obstacle_points_topic", "/carla/ego_vehicle/center_lidar/obstacle_points"
        )
        self.declare_parameter("eps", 0.5)
        self.declare_parameter("min_points", 1)

        self.eps = self.get_parameter("eps").get_parameter_value().double_value
        self.min_points = (
            self.get_parameter("min_points").get_parameter_value().integer_value
        )
        # route definitions
        self.subscription = self.create_subscription(
            PointCloud2,
            self.get_parameter("obstacle_points_topic")
            .get_parameter_value()
            .string_value,
            self.callback,
            10,
        )
        self.subscription  # prevent unused variable warning

        self.visual_publisher = self.create_publisher(
            msg_type=Marker,
            topic=f"{self.subscription.topic}/clustered",
            qos_profile=10,
        )
        self.get_logger().info(
            f"Listening for obstacle points at {self.subscription.topic}"
        )
        self.get_logger().info(
            f"Publishing clustered obstacle points visual at {self.visual_publisher.topic}"
        )

        self.get_logger().info(f"db_cluster eps: {self.eps}")
        self.get_logger().info(f"min_points eps: {self.min_points}")

    def callback(self, msg: PointCloud2):
        points = pointcloud2_to_array(msg)
        labels = self.cluster(points=points)

        self.publish(
            labels=labels,
            points=points,
            frame_id=msg.header.frame_id,
            stamp=msg.header.stamp,
        )

    def cluster(self, points):
        points = np.array(points)
        o3d_pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(points)
        )
        labels = np.array(
            o3d_pcd.cluster_dbscan(
                eps=self.eps, min_points=self.min_points, print_progress=False
            )
        )
        return labels

    def find_avgs(self, points, labels):
        pts = []
        for i in range(0, labels.max()):
            avg_pts = np.average(points[np.where(labels == i)[0]], axis=0)
            pts.append(avg_pts)
        points = np.array(pts)
        print(np.shape(points))
        return pts

    def publish(self, labels, points, frame_id, stamp):
        max_label = labels.max()
        header: Header = Header()
        header.frame_id = frame_id
        header.stamp = stamp
        markers = Marker()
        markers.header = header
        markers.type = 6
        markers.scale = Vector3(x=float(1), y=float(1), z=float(1))

        for i in range(0, max_label):
            indicies = np.where(labels == i)[0]
            pts = points[indicies]
            avg_point = np.average(pts, axis=0)
            point = Point(
                x=float(avg_point[0]), y=float(avg_point[1]), z=float(avg_point[2])
            )
            markers.colors.append(
                ColorRGBA(
                    r=1.0,
                    g=1.0,
                    b=1.0,
                    a=1.0,
                )
            )
            markers.points.append(point)
        self.visual_publisher.publish(markers)


def main(args=None):
    rclpy.init()
    node = EuclideanClusteringNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
