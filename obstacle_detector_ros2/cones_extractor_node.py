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
from typing import List, Tuple
from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Vector3
from builtin_interfaces.msg import Duration as BuiltInDuration
from std_msgs.msg import Header
import rclpy
import rclpy.node
import numpy as np
from vision_msgs.msg import ObjectHypothesis
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


class ConesExtractorNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("cones_extractor_node")
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
                Image,
                f"{Path(self.rgb_camera_image_topic).parent.as_posix()}/depth_from_pcd",
                qos_profile=profile,
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
        depth_img_msg: Image,
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
        original_image = self.bridge.imgmsg_to_cv2(rgb_img_msg, desired_encoding="bgr8")
        boxes, scores, classes = self.detection_2d_array_to_bboxes_and_scores(
            data=detection_2d_array_msg
        )
        output, scores, classes = self.get_all_points(
            depth_img_msg=depth_img_msg,
            boxes=boxes,
            original_image=original_image,
            scores=scores,
            classes=classes,
        )
        if len(output) == 0:
            # nothing is detected after filtering lidar points
            return
        camera_model = PinholeCameraModel()
        camera_model.fromCameraInfo(camera_info_msg)
        K = camera_model.intrinsicMatrix().A
        output = self.img_to_cam(points=output, K=K)

        # compute the centroids for the detections in respective frames
        centers = []
        for points in output:
            meds = np.median(points, axis=1)
            centers.append([np.double(meds[0]), np.double(meds[1]), np.double(meds[2])])
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

                centers = [
                    P @ cp.array([center[0], center[1], center[2], 1])
                    for center in centers
                ]

            except TransformException as ex:
                self.get_logger().info(
                    f"Could not transform {current_frame} to {target_frame_id}: {ex}"
                )
                return
        # publish Detection3DArray and visual msg
        self.publish_detection_3d_array(
            centers=centers,
            camera_info_msg=camera_info_msg,
            image_msg=rgb_img_msg,
            scores=scores,
            classes=classes,
        )
        self.publish_detection_3d_array_visual(
            centers=centers,
            camera_info_msg=camera_info_msg,
            image_msg=rgb_img_msg,
            classes=classes,
        )
        # except Exception as e:
        #     self.get_logger().error(f"Error: {e}")

    def publish_detection_3d_array_visual(
        self, centers, camera_info_msg: CameraInfo, image_msg: Image, classes
    ):
        """generate and publish markers for visualization purpose

        Args:
            centers (list): list of 3x1 points
        """
        header: Header = Header()
        header.frame_id = (
            self.default_rgb_camera_frame_id
            if self.default_rgb_camera_frame_id != "None"
            else image_msg.header.frame_id
        )
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
            if cls == "cone":
                markers.colors.append(ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0))
            else:
                markers.colors.append(ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0))
            markers.points.append(point)
        self.detection_3d_array_visual_publisher.publish(markers)

    def publish_detection_3d_array(
        self, centers, camera_info_msg: CameraInfo, image_msg: Image, scores, classes
    ):
        """generate and publish detection message

        Args:
             centers (list): list of 3x1 points
        """
        header: Header = Header()
        header.frame_id = (
            self.default_rgb_camera_frame_id
            if self.default_rgb_camera_frame_id != "None"
            else image_msg.header.frame_id
        )
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

    def img_to_cam(self, points, K):
        """given intrinsics and a list of points, generate the points in camera coordinate

        Args:
            intrinsics (np.ndarray): 3x3 intrinsics matrix
            points (np.ndarray): 3xn array of points, in order of [u,v,s]

        Returns:
            list: list of points in camera coordinate. If there is error, return empty list
        """
        try:
            output = []
            for ps in points:
                ps = ps.T
                points_2d = cp.array(
                    [
                        ps[0, :] * ps[2, :],
                        ps[1, :] * ps[2, :],
                        ps[2, :],
                    ]
                )  # 3xn
                cam_points = cp.dot(cp.linalg.inv(cp.asarray(K)), points_2d)  # 3xn
                if self.is_carla is False:
                    cam_points = cp.array(
                        [cam_points[2], -cam_points[0], -cam_points[1]]
                    )
                output.append(cam_points)  # mx3xn, where m is number of detections
            return output
        except Exception as e:
            print(f"Error: {e}")
            return []

    def get_all_points(self, depth_img_msg, boxes, original_image, scores, classes):
        depth_img = cp.asarray(
            self.bridge.imgmsg_to_cv2(depth_img_msg, desired_encoding="passthrough")
        )

        # convert depth to points
        u_coords, v_coords = cp.where(depth_img > 0)
        depths = depth_img[u_coords, v_coords]
        points = cp.array([v_coords, u_coords, depths]).T  # Nx3
        # filter points that are in the bounding box
        filtered_points, mask = get_points_only_in_bbox(
            boxes=boxes, points=points, im=original_image, classes=classes
        )
        if len(mask) == 0:
            # nothing is detected after filtering points
            return [], [], []

        result_points = []
        for points in filtered_points:
            max_y = cp.max(points[:, 1])
            ps = points[points[:, 1] > max_y - 3]
            result_points.append(cp.asnumpy(ps))
        result_points = np.array(result_points)
        scores = np.array(scores)[mask]
        classes = np.array(classes)[mask]

        # comment only the line below to add cone_detections
        result_points, scores, classes = self.remove_cones(result_points, scores, classes)
        
        return result_points, scores, classes

    def remove_cones(self, result_points, scores, classes):
        """Filtering out cone detections ¯\_(ツ)_/¯"""
        mask = classes != 'cone'
        return result_points[mask], scores[mask], classes[mask]


    def detection_2d_array_to_bboxes_and_scores(
        self, data: Detection2DArray
    ) -> Tuple[list, list, list]:
        boxes = []
        scores = []
        classes = []
        for detection in data.detections:
            if len(detection.results) > 0:
                if detection.results[0].hypothesis.score > self.confidence_threshold:
                    score = detection.results[0].hypothesis.score
                    cls = detection.results[0].hypothesis.class_id
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
    node = ConesExtractorNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
