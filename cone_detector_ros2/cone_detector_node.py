from __future__ import division

#!/usr/bin/env python3
import rclpy
import rclpy.node
import sys
import cv2
import numpy as np
from pathlib import Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
from rclpy.qos import *
from tensorflow.compat.v1 import InteractiveSession

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


OUTPUT_WINDOW_WIDTH = 640  # Use None to use the original size of the image
DETECT_EVERY_N_SECONDS = None  # Use None to perform detection for each frame

# TUNE ME
CROP_WIDTH = CROP_HEIGHT = 600
CROP_STEP_HORIZONTAL = CROP_STEP_VERTICAL = 600 - 20  # no cone bigger than 20px
SCORE_THRESHOLD = 0.5
NON_MAX_SUPPRESSION_THRESHOLD = 0.5


import cv2
import numpy as np

_BLUE = "blue"
_ORANGE = "orange"
_YELLOW = "yellow"

_COLORS = {_BLUE: (255, 0, 0), _ORANGE: (0, 165, 255), _YELLOW: (0, 255, 255)}

_HSV_COLOR_RANGES = {
    _BLUE: (
        np.array([101, 150, 0], dtype=np.uint8),
        np.array([150, 255, 255], dtype=np.uint8),
    ),
    _ORANGE: (
        np.array([0, 50, 50], dtype=np.uint8),
        np.array([15, 255, 255], dtype=np.uint8),
    ),
    _YELLOW: (
        np.array([16, 50, 50], dtype=np.uint8),
        np.array([45, 255, 255], dtype=np.uint8),
    ),
}


def predominant_rgb_color(img, ymin, xmin, ymax, xmax):
    crop = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[ymin:ymax, xmin:xmax]
    best_color, highest_pxl_count = None, -1
    for color, r in _HSV_COLOR_RANGES.items():
        lower, upper = r
        pxl_count = np.count_nonzero(cv2.inRange(crop, lower, upper))
        if pxl_count > highest_pxl_count:
            best_color = color
            highest_pxl_count = pxl_count
    return _COLORS[best_color]


def add_rectangle_with_text(image, ymin, xmin, ymax, xmax, color, text):
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 5)
    cv2.putText(
        image,
        text,
        (int(xmin), int(ymin) - 10),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def resize_width_keeping_aspect_ratio(
    image, desired_width, interpolation=cv2.INTER_AREA
):
    (h, w) = image.shape[:2]
    r = desired_width / w
    dim = (desired_width, int(h * r))
    return cv2.resize(image, dim, interpolation=interpolation)


def load_model(path_to_frozen_graph):
    """
    Loads a TensorFlow model from a .pb file containing a frozen graph.

    Args:
        path_to_frozen_graph (str): absolute or relative path to the .pb file.

    Returns:
        tf.Graph: a TensorFlow frozen graph.

    """
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_frozen_graph, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")
    return detection_graph


def run_inference_for_batch(batch, session):
    """
    Forward propagates the batch of images in the given graph.

    Args:
        batch (ndarray): (n_images, img_height, img_width, img_channels).
        graph (tf.Graph): TensorFlow frozen graph.
        session (tf.Session): TensorFlow session.

    Returns:
        a dictionary with the following keys:
        num_detections  --  number of detections for each image.
            An ndarray of shape (n_images).
        detection_boxes --  bounding boxes (ymin, ymax, xmin, xmax) for each image.
            An ndarray of shape (n_images, max_detections, 4).
        detection_scores -- scores for each one of the previous detections.
            An ndarray of shape (n_images, max_detections)

    """
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in ["num_detections", "detection_scores", "detection_boxes"]:
        tensor_name = key + ":0"
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

    image_tensor = tf.get_default_graph().get_tensor_by_name("image_tensor:0")
    # Run inference
    output_dict = session.run(tensor_dict, feed_dict={image_tensor: batch})
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict["num_detections"] = output_dict["num_detections"].astype(np.int)
    img_height, img_width = batch.shape[1:3]
    # Transform from relative coordinates in the image (e.g., 0.42) to pixel coordinates (e.g., 542)
    output_dict["detection_boxes"] = (
        output_dict["detection_boxes"] * [img_height, img_width, img_height, img_width]
    ).astype(np.int)
    return output_dict


def extract_crops(
    img, crop_height, crop_width, step_vertical=None, step_horizontal=None
):
    """
    Extracts crops of (crop_height, crop_width) from the given image. Starting
    at (0,0) it begins taking crops horizontally and, every time a crop is taken,
    the 'xmin' start position of the crop is moved according to 'step_horizontal'.
    If some part of the crop to take is out of the bounds of the image, one last
    crop is taken with crop 'xmax' aligned with the right-most ending of the image.
    After taking all the crops in one row, the crop 'ymin' position is moved in the
     same way as before.

    Args:
        img (ndarray): image to crop.
        crop_height (int): height of the crop.
        crop_width (int): width of the crop.
        step_vertical (int): the number of pixels to move vertically before taking
            another crop. It's default value is 'crop_height'.
        step_horizontal (int): the number of pixels to move horizontally before taking
            another crop. It's default value is 'crop_width'.

    Returns:
         sequence of 2D ndarrays: each crop taken.
         sequence of tuples: (ymin, xmin, ymax, xmax) position of each crop in the
             original image.

    """

    img_height, img_width = img.shape[:2]
    crop_height = min(crop_height, img_height)
    crop_width = min(crop_width, img_width)

    # TODO: pre-allocate numpy array
    crops = []
    crops_boxes = []

    if not step_horizontal:
        step_horizontal = crop_width
    if not step_vertical:
        step_vertical = crop_height

    height_offset = 0
    last_row = False
    while not last_row:
        # If we crop 'outside' of the image, change the offset
        # so the crop finishes just at the border if it
        if img_height - height_offset < crop_height:
            height_offset = img_height - crop_height
            last_row = True
        last_column = False
        width_offset = 0
        while not last_column:
            # Same as above
            if img_width - width_offset < crop_width:
                width_offset = img_width - crop_width
                last_column = True
            ymin, ymax = height_offset, height_offset + crop_height
            xmin, xmax = width_offset, width_offset + crop_width
            a_crop = img[ymin:ymax, xmin:xmax]
            crops.append(a_crop)
            crops_boxes.append((ymin, xmin, ymax, xmax))
            width_offset += step_horizontal
        height_offset += step_vertical
    return np.stack(crops, axis=0), crops_boxes


def get_absolute_boxes(box_absolute, boxes_relative):
    """
    Given a bounding box relative to some image, and a sequence of bounding
    boxes relative to the previous one, this methods transform the coordinates
    of each of the last boxes to the same coordinate system of the former.

    For example, if the absolute bounding box is [100, 100, 400, 500] (ymin, xmin,
    ymax, xmax) and the relative one is [10, 10, 20, 30], the coordinates of the
    last one in the coordinate system of the first are [110, 410, 120, 430].

    Args:
        box_absolute (ndarray): absolute bounding box.
        boxes_relative (sequence of ndarray): relative bounding boxes.

    Returns:
        sequence of ndarray: coordinates of each of the relative boxes in the
            coordinate system of the first one.

    """
    absolute_boxes = []
    absolute_ymin, absolute_xmin, _, _ = box_absolute
    for relative_box in boxes_relative:
        absolute_boxes.append(
            relative_box + [absolute_ymin, absolute_xmin, absolute_ymin, absolute_xmin]
        )
    return absolute_boxes


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlap_thresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
        )

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


class ConeDetectorNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("cone_detector_node")
        self.declare_parameter("model_path", "")
        self.declare_parameter("image_topic", "test_image")
        self.declare_parameter("debug", False)

        self.debug = self.get_parameter("debug").get_parameter_value().bool_value

        self.bridge = CvBridge()
        self.get_logger().info(
            f"Listening to image topic: {self.get_parameter('image_topic').get_parameter_value().string_value}"
        )
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        qos_profile.durability = QoSDurabilityPolicy.VOLATILE
        self.subscription = self.create_subscription(
            Image,
            self.get_parameter("image_topic").get_parameter_value().string_value,
            self.on_image_received,
            qos_profile,
        )
        self.subscription  # prevent unused variable warning

        self.frozen_graph_path = (
            self.get_parameter("model_path").get_parameter_value().string_value
        )
        self.get_logger().info(f"File Path: {self.frozen_graph_path}")
        self.detection_graph = load_model(self.frozen_graph_path)
        self.session = InteractiveSession(graph=self.detection_graph)
        self.detected_img_pub = self.create_publisher(
            Image, "cone_detection_output", 10
        )

    def on_image_received(self, msg):
        rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        start = time.time()
        try:

            if self.debug:
                cv2.putText(
                    rgb_img,
                    f"FPS: {1 / (time.time()-start)}",
                    (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )
                cv2.imshow("rgb", rgb_img)
                cv2.waitKey(1)

            self.process_image(rgb_img)
        except Exception as e:
            self.get_logger().error(f"{e}")

    def process_image(self, frame):
        crops, crops_coordinates = extract_crops(
            frame, CROP_HEIGHT, CROP_WIDTH, CROP_STEP_VERTICAL, CROP_STEP_VERTICAL
        )

        # Uncomment this if you also uncommented the two lines before
        #  creating the TF session.
        # crops = np.array([crops[0]])
        # crops_coordinates = [crops_coordinates[0]]

        detection_dict = run_inference_for_batch(crops, self.session)
        # The detection boxes obtained are relative to each crop. Get
        # boxes relative to the original image
        # IMPORTANT! The boxes coordinates are in the following order:
        # (ymin, xmin, ymax, xmax)
        boxes = []
        for box_absolute, boxes_relative in zip(
            crops_coordinates, detection_dict["detection_boxes"]
        ):
            boxes.extend(
                get_absolute_boxes(
                    box_absolute, boxes_relative[np.any(boxes_relative, axis=1)]
                )
            )
        if boxes:
            boxes = np.vstack(boxes)

        # Remove overlapping boxes
        boxes = non_max_suppression_fast(boxes, NON_MAX_SUPPRESSION_THRESHOLD)

        # Get scores to display them on top of each detection
        boxes_scores = detection_dict["detection_scores"]
        boxes_scores = boxes_scores[np.nonzero(boxes_scores)]

        for box, score in zip(boxes, boxes_scores):
            if score > SCORE_THRESHOLD:
                ymin, xmin, ymax, xmax = box
                color_detected_rgb = predominant_rgb_color(
                    frame, ymin, xmin, ymax, xmax
                )
                text = "{:.2f}".format(score)
                add_rectangle_with_text(
                    frame, ymin, xmin, ymax, xmax, color_detected_rgb, text
                )

        if OUTPUT_WINDOW_WIDTH:
            frame = resize_width_keeping_aspect_ratio(frame, OUTPUT_WINDOW_WIDTH)

        cv2.imshow("Detection result", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init()
    node = ConeDetectorNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
