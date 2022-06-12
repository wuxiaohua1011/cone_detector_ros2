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
import json
from typing import List, Tuple
from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Vector3
from vision_msgs.msg import ObjectHypothesis
from builtin_interfaces.msg import Duration as BuiltInDuration
from std_msgs.msg import Header
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
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
import cupy as cp
from image_geometry import PinholeCameraModel
from rclpy.qos import *
from .utils.transform_utils import *
from .utils.pcd_util import *
from pathlib import Path
from vision_msgs.msg import (
    Detection2DArray,
    ObjectHypothesisWithPose,
)
import cv2
from .utils.cones_extractor_util import *
import time


class ObjectDistanceEstimatorNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("object_distance_estimator_node")
        self.get_logger().info("initializing...")
        ### parameters
        self.declare_parameter(
            "rgb_camera_info_topic", "/carla/ego_vehicle/front_left_rgb/camera_info"
        )
        self.declare_parameter(
            "rgb_camera_topic", "/carla/ego_vehicle/front_left_rgb/image"
        )
        self.declare_parameter("confidence_threshold", 0.6)
        self.declare_parameter("rgb_camera_frame_id", "None")
        self.declare_parameter("output_frame_id", "None")
        self.declare_parameter(
            "rgb_distance_estimator_config_path",
            "./config/rgb_distance_estimator_config.json",
        )

        self.config_path = Path(
            self.get_parameter("rgb_distance_estimator_config_path")
            .get_parameter_value()
            .string_value
        )
        self.config = json.load(self.config_path.open("r"))
        self.get_logger().info(f"Reading config from [{self.config_path}]")
        print(f"Config:\n{self.config}")
        self.output_frame_id = (
            self.get_parameter("output_frame_id").get_parameter_value().string_value
        )

        self.get_logger().info(f"output_frame_id: {self.output_frame_id}")
        if self.output_frame_id == "None":
            self.get_logger().info(
                "Publishing output to rgb_camera_frame_id or frames associated with each image"
            )
        self.default_rgb_camera_frame_id = (
            self.get_parameter("rgb_camera_frame_id").get_parameter_value().string_value
        )
        self.confidence_threshold = (
            self.get_parameter("confidence_threshold")
            .get_parameter_value()
            .double_value
        )
        self.rgb_camera_image_topic = (
            self.get_parameter("rgb_camera_topic").get_parameter_value().string_value
        )
        self.rgb_camera_info_topic = (
            self.get_parameter("rgb_camera_info_topic")
            .get_parameter_value()
            .string_value
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        profile = rclpy.qos.qos_profile_sensor_data
        self.bridge = CvBridge()
        subscribers = [
            message_filters.Subscriber(
                self, Image, self.rgb_camera_image_topic, qos_profile=profile
            ),
            message_filters.Subscriber(
                self,
                Detection2DArray,
                f"{self.rgb_camera_image_topic}/detection_2d_array",
                qos_profile=profile,
            ),
            message_filters.Subscriber(
                self, CameraInfo, self.rgb_camera_info_topic, qos_profile=profile
            ),
        ]

        self.ats = message_filters.ApproximateTimeSynchronizer(
            subscribers, queue_size=100, slop=50
        )
        self.is_carla = True if "carla" in subscribers[-1].topic else False
        self.get_logger().info(
            f"Listening on topics: [{[s.topic for s in subscribers]}]"
        )

        self.ats.registerCallback(self.on_data_received)

        ### publisher
        self.detection_3d_array_publisher = self.create_publisher(
            Detection3DArray,
            f"{Path(self.rgb_camera_info_topic).parent.as_posix()}/cone_detection",
            10,
        )
        self.detection_3d_array_visual_publisher = self.create_publisher(
            Marker,
            f"{Path(self.rgb_camera_info_topic).parent.as_posix()}/cone_detection_visual",
            10,
        )
        self.get_logger().info(
            f"Publishing 3d Array at [{self.detection_3d_array_publisher.topic}]"
        )
        self.get_logger().info(
            f"Publishing 3d Array Visual at [{self.detection_3d_array_visual_publisher.topic}]"
        )
        self.get_logger().debug("Publisher initialized")

    def on_data_received(
        self,
        rgb_img_msg: Image,
        detection_2d_array_msg: Detection2DArray,
        camera_info_msg: CameraInfo,
    ):
        """On receiving detection 2d and depth, return the coordinate of the obstacles in 3d

        Args:
            rgb_img_msg (Image): RGB image
            depth_img_msg (Image): depth image
            detection_2d_array_msg (Detection2DArray): cones bounding boxes that were found on the rgb image
            camera_info_msg (CameraInfo): camera info of the rgb image
        """
        # try:
        boxes, scores, classes = self.detection_2d_array_to_bboxes_and_scores(
            data=detection_2d_array_msg
        )
        model = PinholeCameraModel()
        model.fromCameraInfo(camera_info_msg)

        result_centers = []
        result_classes = []
        result_scores = []
        for box, score, cls in zip(boxes, scores, classes):
            # for every detection, if the confidence is above threshold, and class is within my configuration
            if score > self.confidence_threshold and cls in self.config:
                result_scores.append(score)
                minx, miny, maxx, maxy = box

                # find pixel projection using similar geometry
                # physical obj height
                obj_height = self.config[cls].get("object_height", 1)
                # pixel height
                obj_pixel_height = maxy - miny
                # find gain in my configuration table
                obj_height_gain = self.find_gain(
                    self.config[cls]["distance_gain"]["y"],
                    pixel_height=obj_pixel_height,
                )
                # find projected depth
                z = close_far_dist = (
                    model.fy() * obj_height * obj_height_gain / obj_pixel_height
                )

                obj_width = self.config[cls].get("object_width", 1)
                obj_center_hori = (maxx + minx) / 2
                is_left = model.cx() - obj_center_hori > 0

                obj_width_gain = self.find_gain(
                    self.config[cls]["distance_gain"]["x"], pixel_height=(maxx - minx)
                )

                x = left_right_dist = (
                    model.fx() * obj_width * obj_width_gain / (maxx - minx)
                )
                if is_left:
                    x *= -1
                # assume obstacle is always on the level plane as sensor
                y = up_down_dist = 0
                # result_centers.append([x, y, z])
                result_centers.append([x, y, z])
                result_classes.append(cls)
                # if cls == "cone":
                #     print("------------------------------------------------")
                #     print("object pixel height", obj_pixel_height)
                #     print("obj_height_gain", obj_height_gain)
                #     print("close_far_dist", close_far_dist)
                #     print()

                #     print("obj_pixel_width", (maxx - minx))
                #     print("obj_width_gain", obj_width_gain)
                #     print("left_right_dist", left_right_dist)
                #     print()
                #     print("***************")

        # transform frame of reference
        target_frame_id = (
            self.default_rgb_camera_frame_id
            if self.default_rgb_camera_frame_id != "None"
            else rgb_img_msg.header.frame_id
        )
        if self.output_frame_id != "None":
            current_frame = (
                self.default_rgb_camera_frame_id
                if self.default_rgb_camera_frame_id != "None"
                else rgb_img_msg.header.frame_id
            )
            target_frame_id = self.output_frame_id
            try:
                now = rclpy.time.Time()
                trans_msg = self.tf_buffer.lookup_transform(
                    target_frame=target_frame_id, source_frame=current_frame, time=now
                )
                P = msg_to_se3(trans_msg)
                result_centers = [
                    P @ cp.array([center[0], center[1], center[2], 1])
                    for center in result_centers
                ]

            except TransformException as ex:
                self.get_logger().info(
                    f"Could not transform {current_frame} to {target_frame_id}: {ex}"
                )
                return

        self.publish_detection_3d_array_visual(
            centers=result_centers,
            camera_info_msg=camera_info_msg,
            classes=result_classes,
            frame_id=target_frame_id,
        )
        self.publish_detection_3d_array(
            centers=result_centers,
            camera_info_msg=camera_info_msg,
            scores=result_scores,
            classes=result_classes,
            frame_id=target_frame_id,
        )

    @staticmethod
    def find_gain(config: dict, pixel_height: int) -> int:
        """Given a dictionary, find the gain corresponding to a pixel_height
        Assume that distances are already sorted by key
        Args:
            config (dict): config dictionary in the shape of {pixel_height:gain}
            pixel_height (int): pixel height

        Returns:
            int: gain
        """
        for k, v in config.items():
            if int(k) > pixel_height:
                return v
        return 1

    def publish_detection_3d_array_visual(
        self,
        centers,
        camera_info_msg: CameraInfo,
        classes,
        frame_id: str,
    ):
        """generate and publish markers for visualization purpose

        Args:
            centers (list): list of 3x1 points
        """
        header: Header = Header()
        header.frame_id = frame_id
        header.stamp = camera_info_msg.header.stamp
        markers = Marker()
        markers.header = header
        markers.type = 6
        if self.is_carla is False:
            markers.scale = Vector3(x=float(0.5), y=float(0.2), z=float(0.2))
        else:
            markers.scale = Vector3(x=float(1), y=float(1), z=float(1))
        markers.lifetime = BuiltInDuration(nanosec=int(5e8))
        for center, cls in zip(centers, classes):
            point = Point(x=float(center[0]), y=float(center[1]), z=float(center[2]))
            markers.colors.append(
                ColorRGBA(
                    r=self.config[cls]["r"],
                    g=self.config[cls]["g"],
                    b=self.config[cls]["b"],
                    a=1.0,
                )
            )
            markers.points.append(point)
        self.detection_3d_array_visual_publisher.publish(markers)

    def publish_detection_3d_array(
        self,
        centers,
        camera_info_msg: CameraInfo,
        scores,
        classes,
        frame_id,
    ):
        """generate and publish detection message

        Args:
             centers (list): list of 3x1 points
        """
        header: Header = Header()
        header.frame_id = frame_id
        header.stamp = camera_info_msg.header.stamp
        detection3darray = Detection3DArray(header=header)
        for i, center in enumerate(centers):
            bbox = BoundingBox3D()
            bbox.center = Pose(
                position=Point(
                    x=float(center[0]), y=float(center[1]), z=float(center[2])
                )
            )
            bbox.size = Vector3(x=float(1), y=float(1), z=float(1))
            detection3d = Detection3D(header=header, bbox=bbox)
            obs = ObjectHypothesisWithPose()
            obs.hypothesis = ObjectHypothesis(class_id=classes[i], score=float(scores[i]))

            detection3d.results.append(obs)
            detection3darray.detections.append(detection3d)
        self.detection_3d_array_publisher.publish(detection3darray)

    def detection_2d_array_to_bboxes_and_scores(
        self, data: Detection2DArray
    ) -> Tuple[list, list, list]:
        boxes = []
        scores = []
        classes = []
        for detection in data.detections:
            if len(detection.results) > 0:
                if detection.results[0].score > self.confidence_threshold:
                    score = detection.results[0].score
                    cls = detection.results[0].id
                    center = (detection.bbox.center.x, detection.bbox.center.y)
                    size_x = detection.bbox.size_x
                    size_y = detection.bbox.size_y
                    minx = int(center[0] - (size_x / 2))
                    miny = int(center[1] - (size_y / 2))
                    maxx = int(center[0] + (size_x / 2))
                    maxy = int(center[1] + (size_y / 2))
                    box = [minx, miny, maxx, maxy]
                    boxes.append(box)
                    scores.append(score)
                    classes.append(cls)
        return boxes, scores, classes


def main(args=None):
    rclpy.init()
    node = ObjectDistanceEstimatorNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
