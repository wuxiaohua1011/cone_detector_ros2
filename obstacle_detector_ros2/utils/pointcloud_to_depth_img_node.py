"""
 Copyright 2022 Michael Wu

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """


#!/usr/bin/env python3
from typing import List, Optional
import rclpy
import rclpy.node
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rclpy
import rclpy.node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
import cupy as cp
from image_geometry import PinholeCameraModel
from rclpy.qos import *
from .transform_utils import *
from .pcd_util import *
from pathlib import Path
import cv2
import time
from collections import deque


class PCD2DepthNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("pcd_to_depth_node")
        self.get_logger().info("initializing...")
        ### parameters
        self.declare_parameter(
            "rgb_camera_info_topic", "/carla/ego_vehicle/front_left_rgb/camera_info"
        )
        self.declare_parameter(
            "rgb_camera_topic", "/carla/ego_vehicle/front_left_rgb/image"
        )
        self.declare_parameter(
            "lidar_topics",
            [
                "/carla/ego_vehicle/left_lidar",
                "/carla/ego_vehicle/center_lidar",
                "/carla/ego_vehicle/right_lidar",
            ],
        )
        self.declare_parameter("rgb_camera_frame_id", "None")
        self.declare_parameter("num_frame_buffered", 5)

        self.num_frame_buffered = (
            self.get_parameter("num_frame_buffered").get_parameter_value().integer_value
        )
        self.default_rgb_camera_frame_id = (
            self.get_parameter("rgb_camera_frame_id").get_parameter_value().string_value
        )
        self.lidar_topics = (
            self.get_parameter("lidar_topics").get_parameter_value().string_array_value
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        profile = rclpy.qos.qos_profile_sensor_data
        self.bridge = CvBridge()
        subscribers = [
            message_filters.Subscriber(
                self,
                PointCloud2,
                topic,
                qos_profile=profile,
            )
            for topic in self.lidar_topics
        ]
        subscribers.append(
            message_filters.Subscriber(
                self,
                Image,
                self.get_parameter("rgb_camera_topic")
                .get_parameter_value()
                .string_value,
                qos_profile=profile,
            )
        )
        subscribers.append(
            message_filters.Subscriber(
                self,
                CameraInfo,
                self.get_parameter("rgb_camera_info_topic")
                .get_parameter_value()
                .string_value,
                qos_profile=profile,
            )
        )
        self.is_carla = True if "carla" in subscribers[-1].topic else False
        self.get_logger().info(
            f"Listening on topics: [{[s.topic for s in subscribers]}]"
        )

        self.ats = message_filters.ApproximateTimeSynchronizer(
            subscribers, queue_size=10, slop=5
        )
        self.ats.registerCallback(self.on_data_received)

        ### publisher
        self.depth_img_publisher = self.create_publisher(
            msg_type=Image,
            topic=f"{Path(subscribers[-2].topic).parent.as_posix()}/depth_from_pcd",
            qos_profile=profile,
        )
        self.get_logger().info(
            f"Publishing depth image to [{self.depth_img_publisher.topic}]"
        )

        # queue to buffer several frames of lidar for better detection
        self.points_queue = deque(maxlen=self.num_frame_buffered)

    def on_data_received(self, *args):
        """on data received, output depth image using lidar data and rgb camera info"""
        camera_info_msg: CameraInfo = args[-1]
        image_msg: Image = args[-2]
        lidar_msgs: List[PointCloud2] = args[0:-2]
        # for every lidar message, transform to image space
        list_of_points: List[cp.ndarray] = [
            self.lidar_to_img(
                lidar_msg=lidar_msg,
                camera_info_msg=camera_info_msg,
                image_msg=image_msg,
            )
            for lidar_msg in lidar_msgs
        ]

        # ignore points that have no data in there (maybe lidar malfunctioned, or transform is not read in)
        list_of_points = [
            points for points in list_of_points if cp.shape(points)[0] > 0
        ]
        if len(list_of_points) == 0:
            self.get_logger().info("no points detected")
            return

        # concatenate all points
        points = self.merge_points(list_of_points=list_of_points)
        self.points_queue.append(points)
        if len(self.points_queue) == self.num_frame_buffered:
            pts = np.vstack(self.points_queue)
            # publish depth image
            self.publish_depth_msg(
                points=pts, camera_info_msg=camera_info_msg, image_msg=image_msg
            )

    def publish_depth_msg(
        self, points: cp.ndarray, camera_info_msg: CameraInfo, image_msg: Image
    ):
        """Publish depth msg given points and camera info

        Args:
            points (np.ndarray): pointcloud
            camera_info_msg (CameraInfo): camera info
        """
        depth_img = self.create_depth_from_points(points, camera_info_msg)
        depth_img_msg = self.bridge.cv2_to_imgmsg(
            cp.asnumpy(depth_img), encoding="passthrough"
        )
        depth_img_msg.header.frame_id = (
            self.default_rgb_camera_frame_id
            if self.default_rgb_camera_frame_id != "None"
            else image_msg.header.frame_id
        )
        depth_img_msg.header.stamp = camera_info_msg.header.stamp
        self.depth_img_publisher.publish(depth_img_msg)

    @staticmethod
    def create_depth_from_points(points: cp.ndarray, camera_info_msg: CameraInfo):
        """create depth image from points and camera info

        Args:
            points (np.ndarray): points
            camera_info_msg (CameraInfo): camera info

        Returns:
            np.ndarray: depth image of the same size as indicated in camera info
        """
        im = cp.zeros(
            shape=(camera_info_msg.height, camera_info_msg.width), dtype=cp.float32
        )
        u_coord = points[:, 0].astype(int)
        v_coord = points[:, 1].astype(int)
        im[v_coord, u_coord] = points[:, 2]
        return im

    @staticmethod
    def merge_points(list_of_points: List[cp.ndarray]) -> cp.ndarray:
        """Given points from multiple pointclouds, merge them into one

        Args:
            list_of_points (List[np.ndarray]): list of points

        Returns:
            np.ndarray: merged pointcloud, size nx3
        """
        if len(list_of_points) == 0:
            return cp.array([])
        all_points = cp.concatenate(list_of_points, axis=0)
        # TODO: filter points that are overlapping
        return all_points

    def lidar_to_img(
        self, lidar_msg: PointCloud2, camera_info_msg: CameraInfo, image_msg: Image
    ) -> cp.ndarray:
        """Produce a list of [u,v,s] from lidar points imposed onto camera

        Args:
            lidar_msg (PointCloud2): lidar message
            camera_info_msg (CameraInfo): camera info message

        Returns:
            np.ndarray: array of shape (n,3) where
                        n is number of points in [u, v, s] where u and v are image coordinates, s denote depth at that coordinate.
        """
        camera_model = PinholeCameraModel()
        camera_model.fromCameraInfo(camera_info_msg)
        pcd_as_numpy_array = pointcloud2_to_xyz_array(
            cloud_msg=lidar_msg, remove_nans=True
        )
        point_cloud = pcd_as_numpy_array[:, :3]

        ### transform points from lidar to cam ###
        try:
            now = rclpy.time.Time()
            trans_msg = self.tf_buffer.lookup_transform(
                target_frame=image_msg.header.frame_id,
                source_frame=lidar_msg.header.frame_id,
                time=now,
            )
            lidar2cam = msg_to_se3(trans_msg)
        except TransformException as ex:
            self.get_logger().info(
                f"Could not transform {lidar_msg.header.frame_id} to {camera_info_msg.header.frame_id}: {ex}"
            )
            return []

        # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
        local_lidar_points = cp.array(point_cloud).T

        # Add an extra 1.0 at the end of each 3d point so it becomes of
        # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
        local_lidar_points = cp.r_[
            local_lidar_points, [cp.ones(local_lidar_points.shape[1])]
        ]

        ### now we tranform the points from lidar to camera ###
        sensor_points = cp.dot(lidar2cam, local_lidar_points)

        point_in_camera_coords = cp.array(
            [sensor_points[0], sensor_points[1], sensor_points[2]]
        )

        ### transform points from cam to image ###
        points_2d = (
            cp.asarray(camera_model.intrinsicMatrix().A) @ point_in_camera_coords
        )
        # in reality, lidar are going to have readings with 0 (unknown), we need to filter out those readings
        mask = points_2d[2, :] != 0
        points_2d = points_2d.T[mask].T
        # Remember to normalize the x, y values by the 3rd value.
        points_2d = cp.array(
            [
                points_2d[0, :] / points_2d[2, :],
                points_2d[1, :] / points_2d[2, :],
                points_2d[2, :],
            ]
        )
        ### extract only points within image
        """
            At this point, points_2d[0, :] contains all the x and points_2d[1, :]
            contains all the y values of our points. In order to properly
            visualize everything on a screen, the points that are out of the screen
            must be discarted, the same with points behind the camera projection plane.
        """
        points_2d = points_2d.T
        points_in_canvas_mask = (
            (points_2d[:, 0] > 0.0)
            & (points_2d[:, 0] < camera_model.width)
            & (points_2d[:, 1] > 0.0)
            & (points_2d[:, 1] < camera_model.height)
            & (points_2d[:, 2] > 0.0)
        )
        points_2d = points_2d[points_in_canvas_mask]

        # cast x, y into u, v
        # Extract the screen coords (uv) as integers.
        u_coord = points_2d[:, 0].astype(cp.int)
        v_coord = points_2d[:, 1].astype(cp.int)

        result = cp.array([u_coord, v_coord, points_2d[:, 2]]).T
        return result

    def get_transform(self, target_frame, source_frame) -> Optional[cp.ndarray]:
        """Get transform from source frame to target frame.

        If transform is not found, return None instead of throwing error.

        Args:
            target_frame (_type_): target frame
            source_frame (_type_): source frame

        Returns:
            Optional[np.ndarray]: SE3 representation of Rotation and Translation, None if transform not found
        """
        try:
            now = rclpy.time.Time()
            trans_msg = self.tf_buffer.lookup_transform(
                target_frame=target_frame, source_frame=source_frame, time=now
            )
            P = msg_to_se3(trans_msg)
        except TransformException as ex:
            self.get_logger().info(
                f"Could not transform {source_frame} to {target_frame}: {ex}"
            )
            return
        return P


def main(args=None):
    rclpy.init()
    node = PCD2DepthNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
