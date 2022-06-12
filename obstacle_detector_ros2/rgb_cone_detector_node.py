#!/usr/bin/env python3
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
from sensor_msgs.msg import Image
from vision_msgs.msg import ObjectHypothesis
from geometry_msgs.msg import Pose2D
import cupy as cp
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    BoundingBox2D,
    ObjectHypothesisWithPose,
)

from rclpy.qos import *
from .models.common import DetectMultiBackend
from .utils.plots import Annotator
from .utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
)
from .utils.torch_utils import select_device
import torch

from .utils.cone_detector_util import convert_image
import cv2
import time


class ConeDetectorNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("cone_detector_node")
        self.get_logger().info("initializing cone detector...")
        ### parameters
        self.declare_parameter("classes", ["box", "cone"])
        self.declare_parameter("model_path", "")
        self.declare_parameter("rgb_camera_topic", "test_image")
        self.declare_parameter("confidence_threshold", 0.6)
        self.declare_parameter("rgb_camera_frame_id", "None")

        self.default_rgb_camera_frame_id = (
            self.get_parameter("rgb_camera_frame_id").get_parameter_value().string_value
        )
        self.confidence_threshold = (
            self.get_parameter("confidence_threshold")
            .get_parameter_value()
            .double_value
        )
        self.bridge = CvBridge()
        self.classes = (
            self.get_parameter("classes").get_parameter_value().string_array_value
        )
        self.get_logger().info(f"Labels are: [{self.classes}]")
        ### Subscriber
        profile = rclpy.qos.qos_profile_sensor_data
        self.img_subscription = self.create_subscription(
            Image,
            self.get_parameter("rgb_camera_topic").get_parameter_value().string_value,
            self.on_img_received,
            profile,
        )
        self.get_logger().info(
            f"Listening to RGB Camera topic on [{self.img_subscription.topic}]"
        )

        ### publisher
        self.detection_2d_array_publisher = self.create_publisher(
            msg_type=Detection2DArray,
            topic=f"{self.img_subscription.topic}/detection_2d_array",
            qos_profile=self.img_subscription.qos_profile,
        )
        self.detected_img_publisher = self.create_publisher(
            msg_type=Image,
            topic=f"{self.img_subscription.topic}/bbox_img",
            qos_profile=self.img_subscription.qos_profile,
        )
        self.get_logger().info(
            f"Publishing Detection2DArray on [{self.detected_img_publisher.topic}]"
        )
        self.get_logger().info(
            f"Publishing Image with bounding box on [{self.detected_img_publisher.topic}]"
        )

        self.get_logger().debug("Publisher initialized")

        self.device = select_device(device="")
        self.weights_path = (
            self.get_parameter("model_path").get_parameter_value().string_value
        )
        self.get_logger().info(f"Loading model weights from [{self.weights_path}]")
        self.model = DetectMultiBackend(weights=self.weights_path, device=self.device)
        self.stride, self.names, self.pt = (
            self.model.stride,
            self.model.names,
            self.model.pt,
        )
        self.imgsz = (640, 640)
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        self.model.warmup(imgsz=(1 if self.pt else 1, 3, *self.imgsz))
        self.get_logger().info("Cone detector Initialized")

    def on_img_received(self, img_msg: Image):
        """On image received, detect cones, publish image with bounding boxes, publish bounding boxes

        Args:
            img_msg (Image): input image
        """
        original_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        # detect cones
        boxes, scores, classes, labeled_img = self.process_image(im=original_image)
        # publish image with bounding box for visualization
        img_with_bbox_msg = self.bridge.cv2_to_imgmsg(labeled_img)
        img_with_bbox_msg.header.frame_id = (
            self.default_rgb_camera_frame_id
            if self.default_rgb_camera_frame_id != "None"
            else img_msg.header.frame_id
        )
        img_with_bbox_msg.header.stamp = img_msg.header.stamp
        self.detected_img_publisher.publish(img_with_bbox_msg)

        # publish bounding boxes 2d
        self.publish_detection_2d_array(
            bboxes=boxes, scores=scores, classes=classes, img_msg=img_msg
        )

    def publish_detection_2d_array(
        self, bboxes: list, scores: list, classes: list, img_msg: Image
    ):
        """Given bounding boxes and scores, publish ROS detection2DArray message

        Args:
            bboxes (list): bounding boxes
            scores (list): scores
            img_msg (Image): original image message
        """
        header = Header()
        header.frame_id = (
            self.default_rgb_camera_frame_id
            if self.default_rgb_camera_frame_id != "None"
            else img_msg.header.frame_id
        )
        header.stamp = img_msg.header.stamp
        detection_2d_array_msg = Detection2DArray(header=header)
        for box, score, cls in zip(bboxes, scores, classes):
            detection_2d_msg = Detection2D(header=header)
            minx, miny, maxx, maxy = box
            detection_2d_msg.bbox = BoundingBox2D(
                center=Pose2D(
                    x=float((maxx + minx) / 2), y=float((maxy + miny) / 2), theta=0.0
                ),
                size_x=float(maxx - minx),
                size_y=float(maxy - miny),
            )
            obs = ObjectHypothesisWithPose()
            obs.hypothesis = ObjectHypothesis(class_id=cls, score=float(score))
            detection_2d_msg.results.append(obs)
            detection_2d_array_msg.detections.append(detection_2d_msg)
        self.detection_2d_array_publisher.publish(detection_2d_array_msg)

    def process_image(self, im: np.ndarray) -> list:
        """Given an image, return a list of bounding boxes for cones

        Args:
            im (np.ndarray): raw image

        Returns:
            list: list of bounding boxes
        """
        im0 = im.copy()
        im = convert_image(img0=im, img_size=self.imgsz[0], stride=self.stride)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred,
            conf_thres=0.5,
            iou_thres=0.45,
            classes=None,
            agnostic=False,
            max_det=10,
        )
        annotator = Annotator(im0, line_width=3, example=str(self.names))

        result_boxes = []
        scores = []
        classes = []
        # Process predictions
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    if conf > self.confidence_threshold:
                        c = int(cls.cpu().detach().numpy())  # integer class
                        class_name = self.classes[c]
                        conf = round(float(conf.cpu().detach().numpy()), 2)
                        minx, miny, maxx, maxy = [
                            int(t.cpu().detach().numpy()) for t in xyxy
                        ]
                        label = f"{class_name}: {conf}"
                        annotator.box_label(xyxy, label)
                        result_boxes.append([minx, miny, maxx, maxy])
                        scores.append(conf)
                        classes.append(class_name)
        return result_boxes, scores, classes, im0


def main(args=None):
    rclpy.init()
    node = ConeDetectorNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
