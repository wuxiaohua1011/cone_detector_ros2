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
from .utils.cv_utils import *
from .utils.operations import *

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


OUTPUT_WINDOW_WIDTH = 640  # Use None to use the original size of the image
DETECT_EVERY_N_SECONDS = None  # Use None to perform detection for each frame

# TUNE ME
CROP_WIDTH = CROP_HEIGHT = 600
CROP_STEP_HORIZONTAL = CROP_STEP_VERTICAL = 600 - 20  # no cone bigger than 20px
SCORE_THRESHOLD = 0.5
NON_MAX_SUPPRESSION_THRESHOLD = 0.5


class ConeDetectorNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("cone_detector_node")
        self.declare_parameter('model_path', "")
        self.declare_parameter('image_topic', "test_image")
        self.declare_parameter('debug', False)

        self.debug = self.get_parameter(
            'debug').get_parameter_value().bool_value

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            self.get_parameter(
                'image_topic').get_parameter_value().string_value,
            self.on_image_received,
            10)
        self.subscription  # prevent unused variable warning

        self.frozen_graph_path = self.get_parameter(
            'model_path').get_parameter_value().string_value
        self.get_logger().info(f"File Path: {self.frozen_graph_path}")
        self.detection_graph = load_model(self.frozen_graph_path)
        self.session = tf.Session(graph=self.detection_graph)

        self.detected_img_pub = self.create_publisher(
            Image, 'cone_detection_output', 10)

    def on_image_received(self, msg):
        rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        start = time.time()
        try:

            if self.debug:
                # cv2.putText(rgb_img, f"FPS: {1 / (time.time()-start)}", (10, 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.imshow("rgb", rgb_img)
                cv2.waitKey(1)

            self.process_image(rgb_img)
        except Exception as e:
            self.get_logger().error(f"{e}")

    def process_image(self, frame):
        crops, crops_coordinates = extract_crops(
            frame, CROP_HEIGHT, CROP_WIDTH,
            CROP_STEP_VERTICAL, CROP_STEP_VERTICAL)

        # Uncomment this if you also uncommented the two lines before
        #  creating the TF session.
        # crops = np.array([crops[0]])
        # crops_coordinates = [crops_coordinates[0]]

        detection_dict = run_inference_for_batch(crops, self.session)

        # # The detection boxes obtained are relative to each crop. Get
        # # boxes relative to the original image
        # # IMPORTANT! The boxes coordinates are in the following order:
        # # (ymin, xmin, ymax, xmax)
        # boxes = []
        # for box_absolute, boxes_relative in zip(
        #         crops_coordinates, detection_dict['detection_boxes']):
        #     boxes.extend(get_absolute_boxes(
        #         box_absolute,
        #         boxes_relative[np.any(boxes_relative, axis=1)]))
        # if boxes:
        #     boxes = np.vstack(boxes)

        # # Remove overlapping boxes
        # boxes = non_max_suppression_fast(
        #     boxes, NON_MAX_SUPPRESSION_THRESHOLD)

        # # Get scores to display them on top of each detection
        # boxes_scores = detection_dict['detection_scores']
        # boxes_scores = boxes_scores[np.nonzero(boxes_scores)]

        # for box, score in zip(boxes, boxes_scores):
        #     if score > SCORE_THRESHOLD:
        #         ymin, xmin, ymax, xmax = box
        #         color_detected_rgb = predominant_rgb_color(
        #             frame, ymin, xmin, ymax, xmax)
        #         text = '{:.2f}'.format(score)
        #         add_rectangle_with_text(
        #             frame, ymin, xmin, ymax, xmax,
        #             color_detected_rgb, text)

        # if OUTPUT_WINDOW_WIDTH:
        #     frame = resize_width_keeping_aspect_ratio(
        #         frame, OUTPUT_WINDOW_WIDTH)

        # cv2.imshow('Detection result', frame)
        # cv2.waitKey(1)


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
        with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
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
    for key in ['num_detections', 'detection_scores', 'detection_boxes']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph(
            ).get_tensor_by_name(tensor_name)

    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    # Run inference
    output_dict = session.run(tensor_dict, feed_dict={image_tensor: batch})
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = output_dict['num_detections'].astype(
        np.int)
    img_height, img_width = batch.shape[1:3]
    # Transform from relative coordinates in the image (e.g., 0.42) to pixel coordinates (e.g., 542)
    output_dict['detection_boxes'] = (output_dict['detection_boxes'] * [img_height, img_width,
                                                                        img_height, img_width]).astype(np.int)
    return output_dict


def main(args=None):
    rclpy.init()
    node = ConeDetectorNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
