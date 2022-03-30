#!/usr/bin/env python3
from builtin_interfaces.msg import Duration as BuiltInDuration
from numpy import imag
from std_msgs.msg import Header
import rclpy
import rclpy.node
import sys
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tensorflow.compat.v1 import InteractiveSession
import tensorflow.compat.v1 as tf
import rclpy
import rclpy.node
import sys
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import numpy as np
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
import tf_transformations as tr
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Transform
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Vector3
from matplotlib import cm
import numpy as np
from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA

tf.disable_v2_behavior()

VIRIDIS = np.array(cm.get_cmap("viridis").colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


def pose_to_pq(msg):
    """Convert a C{geometry_msgs/Pose} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.position.x, msg.position.y, msg.position.z])
    q = np.array(
        [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
    )
    return p, q


def pose_stamped_to_pq(msg):
    """Convert a C{geometry_msgs/PoseStamped} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    return pose_to_pq(msg.pose)


def transform_to_pq(msg):
    """Convert a C{geometry_msgs/Transform} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
    q = np.array([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w])
    return p, q


def transform_stamped_to_pq(msg):
    """Convert a C{geometry_msgs/TransformStamped} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    return transform_to_pq(msg.transform)


def msg_to_se3(msg):
    """Conversion from geometric ROS messages into SE(3)

    @param msg: Message to transform. Acceptable types - C{geometry_msgs/Pose}, C{geometry_msgs/PoseStamped},
    C{geometry_msgs/Transform}, or C{geometry_msgs/TransformStamped}
    @return: a 4x4 SE(3) matrix as a numpy array
    @note: Throws TypeError if we receive an incorrect type.
    """
    if isinstance(msg, Pose):
        p, q = pose_to_pq(msg)
    elif isinstance(msg, PoseStamped):
        p, q = pose_stamped_to_pq(msg)
    elif isinstance(msg, Transform):
        p, q = transform_to_pq(msg)
    elif isinstance(msg, TransformStamped):
        p, q = transform_stamped_to_pq(msg)
    else:
        raise TypeError("Invalid type for conversion to SE(3)")
    norm = np.linalg.norm(q)
    if np.abs(norm - 1.0) > 1e-3:
        raise ValueError(
            "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(
                str(q), np.linalg.norm(q)
            )
        )
    elif np.abs(norm - 1.0) > 1e-6:
        q = q / norm
    g = tr.quaternion_matrix(q)
    g[0:3, -1] = p
    return g


OUTPUT_WINDOW_WIDTH = None  # Use None to use the original size of the image
DETECT_EVERY_N_SECONDS = None  # Use None to perform detection for each frame

# TUNE ME
CROP_WIDTH = CROP_HEIGHT = 600
CROP_STEP_HORIZONTAL = CROP_STEP_VERTICAL = 600 - 20  # no cone bigger than 20px
SCORE_THRESHOLD = 0.5
NON_MAX_SUPPRESSION_THRESHOLD = 0.5


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


## The code below is "ported" from
# https://github.com/ros/common_msgs/tree/noetic-devel/sensor_msgs/src/sensor_msgs
# I'll make an official port and PR to this repo later:
# https://github.com/ros2/common_interfaces
import sys
import math
import struct
from sensor_msgs.msg import PointCloud2, PointField

_DATATYPES = {}
_DATATYPES[PointField.INT8] = ("b", 1)
_DATATYPES[PointField.UINT8] = ("B", 1)
_DATATYPES[PointField.INT16] = ("h", 2)
_DATATYPES[PointField.UINT16] = ("H", 2)
_DATATYPES[PointField.INT32] = ("i", 4)
_DATATYPES[PointField.UINT32] = ("I", 4)
_DATATYPES[PointField.FLOAT32] = ("f", 4)
_DATATYPES[PointField.FLOAT64] = ("d", 8)


def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
    """
    Read points from a L{sensor_msgs.PointCloud2} message.
    @param cloud: The point cloud to read from.
    @type  cloud: L{sensor_msgs.PointCloud2}
    @param field_names: The names of fields to read. If None, read all fields. [default: None]
    @type  field_names: iterable
    @param skip_nans: If True, then don't return any point with a NaN value.
    @type  skip_nans: bool [default: False]
    @param uvs: If specified, then only return the points at the given coordinates. [default: empty list]
    @type  uvs: iterable
    @return: Generator which yields a list of values for each point.
    @rtype:  generator
    """
    assert isinstance(cloud, PointCloud2), "cloud is not a sensor_msgs.msg.PointCloud2"
    fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    width, height, point_step, row_step, data, isnan = (
        cloud.width,
        cloud.height,
        cloud.point_step,
        cloud.row_step,
        cloud.data,
        math.isnan,
    )
    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    p = unpack_from(data, offset)
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)
                    offset += point_step


def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = ">" if is_bigendian else "<"

    offset = 0
    for field in (
        f
        for f in sorted(fields, key=lambda f: f.offset)
        if field_names is None or f.name in field_names
    ):
        if offset < field.offset:
            fmt += "x" * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print(
                "Skipping unknown PointField datatype [%d]" % field.datatype,
                file=sys.stderr,
            )
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt


class ConeDetectorNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("cone_detector_node")
        self.get_logger().info("initializing cone detector...")
        ### parameters
        self.declare_parameter("model_path", "")
        self.declare_parameter("rgb_camera_topic", "test_image")
        self.declare_parameter("debug", False)
        self.declare_parameter(
            "rgb_camera_info_topic", "/carla/ego_vehicle/front_left_rgb/camera_info"
        )
        self.declare_parameter("lidar_topic", "/carla/ego_vehicle/center_lidar")
        self.declare_parameter("lidar_frame_id", "ego_vehicle/center_lidar")
        self.declare_parameter("rgb_frame_id", "ego_vehicle/front_left_rgb")
        self.declare_parameter("output_frame_id", "")

        self.debug = self.get_parameter("debug").get_parameter_value().bool_value

        self.bridge = CvBridge()
        self.get_logger().info(
            f"Listening to image topic: {self.get_parameter('rgb_camera_topic').get_parameter_value().string_value}"
        )

        ### cone detector
        self.frozen_graph_path = (
            self.get_parameter("model_path").get_parameter_value().string_value
        )
        self.get_logger().info(f"File Path: {self.frozen_graph_path}")
        self.detection_graph = load_model(self.frozen_graph_path)
        self.get_logger().info(f"Model loaded")
        self.session = InteractiveSession(graph=self.detection_graph)
        self.get_logger().info(f"Interactive Session loaded")

        self.to_frame_rel = (
            self.get_parameter("rgb_frame_id").get_parameter_value().string_value
        )
        self.from_frame_rel = (
            self.get_parameter("lidar_frame_id").get_parameter_value().string_value
        )
        self.output_frame_id = (
            self.get_parameter("output_frame_id").get_parameter_value().string_value
        )
        if self.output_frame_id == "":
            self.get_logger().info(
                f"No output frame specified, defaulting to {self.to_frame_rel}"
            )
            # only output in sensor's point of view if no global point of view given
            self.output_frame_id = self.to_frame_rel
        print(
            self.get_parameter("rgb_camera_info_topic")
            .get_parameter_value()
            .string_value,
        )
        ### Subscriber
        self.front_left_img = message_filters.Subscriber(
            self,
            Image,
            self.get_parameter("rgb_camera_topic").get_parameter_value().string_value,
        )
        self.center_lidar = message_filters.Subscriber(
            self,
            PointCloud2,
            self.get_parameter("lidar_topic").get_parameter_value().string_value,
        )
        queue_size = 30
        self.ts = message_filters.TimeSynchronizer(
            [self.front_left_img, self.center_lidar],
            queue_size,
        )
        self.ts.registerCallback(self.callback)

        self.subscription = self.create_subscription(
            CameraInfo,
            self.get_parameter("rgb_camera_info_topic")
            .get_parameter_value()
            .string_value,
            self.camera_info_callback,
            10,
        )
        self.subscription  # prevent unused variable warning
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        ### publisher
        self.detection_3d_array_publisher = self.create_publisher(
            Detection3DArray, "cone_detection", 10
        )
        self.detection_3d_array_visual_publisher = self.create_publisher(
            MarkerArray, "cone_detection_visual", 10
        )
        self.get_logger().debug("Publisher initialized")

        ### Camera Lidar Projection variables
        self.has_received_intrinsics = False
        self.intrinsics = np.zeros(shape=(3, 3))
        self.image_w = 762
        self.image_h = 386
        self.get_logger().info("Cone detector Initialized")

    def callback(self, left_cam_msg, center_lidar_pcl_msg):
        if not self.has_received_intrinsics:
            self.get_logger().info(
                "Received Images, but not camera intrinsics, waiting..."
            )
            return
        original_image = self.bridge.imgmsg_to_cv2(
            left_cam_msg, desired_encoding="bgr8"
        )

        # project lidar onto rgb
        points_2d, lidar_camera_image = self.project_lidar_onto_camera(
            left_cam_msg=left_cam_msg, center_lidar_pcl_msg=center_lidar_pcl_msg
        )
        if points_2d is None or lidar_camera_image is None:
            return

        # run cone detection
        boxes, scores, detected_image = self.process_image(image_msg=left_cam_msg)
        if len(boxes) == 0:
            self.get_logger().info("No Traffic cone detected")
            return

        # filter points that are in the bounding box
        filtered_points = [
            self.get_points_only_in_bbox(box, points=points_2d) for box in boxes
        ]

        # change back to camera coordinate
        output = self.img_to_cam(intrinsics=self.intrinsics, points=filtered_points)
        if len(output) == 0:
            return

        # convert from cam -> output frame, if nessecary
        if self.to_frame_rel != self.output_frame_id:
            output = self.cam_to_output(
                P=self.get_cam_to_output_transform(), points=output
            )

        # compute the centroids for the detections in respective frames
        centers = [np.average(points, axis=1) for points in output]

        # publish Detection3DArray and visual msg
        self.publish_detection_3d_array(centers)
        self.publish_detection_3d_array_visual(centers)

        if self.debug:
            self.draw_filtered_points(original_image, filtered_points=filtered_points)
            cv2.imshow("lidar_projection", lidar_camera_image)
            cv2.waitKey(1)

    def camera_info_callback(self, msg):
        self.intrinsics = np.reshape(msg.k, newshape=(3, 3))
        self.image_w = msg.width
        self.image_h = msg.height
        self.has_received_intrinsics = True
        self.get_logger().debug("Camera Intrinsices updated")

    def get_lidar2cam(self):
        try:
            now = rclpy.time.Time()
            trans_msg = self.tf_buffer.lookup_transform(
                self.to_frame_rel, self.from_frame_rel, now
            )
            lidar2cam = msg_to_se3(trans_msg)
        except TransformException as ex:
            self.get_logger().info(
                f"Could not transform {self.to_frame_rel} to {self.from_frame_rel}: {ex}"
            )
            return
        return lidar2cam

    def get_cam2lidar(self):
        try:
            now = rclpy.time.Time()
            trans_msg = self.tf_buffer.lookup_transform(
                self.from_frame_rel, self.to_frame_rel, now
            )
            lidar2cam = msg_to_se3(trans_msg)
        except TransformException as ex:
            self.get_logger().info(
                f"Could not transform {self.from_frame_rel} to {self.to_frame_rel}: {ex}"
            )
            return
        return lidar2cam

    def project_lidar_onto_camera(self, left_cam_msg, center_lidar_pcl_msg):
        left_img = self.bridge.imgmsg_to_cv2(left_cam_msg)
        pcd_as_numpy_array = np.array(list(read_points(center_lidar_pcl_msg)))
        point_cloud = pcd_as_numpy_array[:, :3]
        intensity = pcd_as_numpy_array[:, 3]

        lidar2cam = self.get_lidar2cam()
        if lidar2cam is None:
            return None, None

        # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
        local_lidar_points = np.array(point_cloud).T

        # Add an extra 1.0 at the end of each 3d point so it becomes of
        # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
        local_lidar_points = np.r_[
            local_lidar_points, [np.ones(local_lidar_points.shape[1])]
        ]

        # now we tranform the points from lidar to camera
        sensor_points = np.dot(lidar2cam, local_lidar_points)
        point_in_camera_coords = np.array(
            [sensor_points[0], sensor_points[1], sensor_points[2]]
        )
        points_2d = np.dot(self.intrinsics, point_in_camera_coords)
        # Remember to normalize the x, y values by the 3rd value.
        points_2d = np.array(
            [
                points_2d[0, :] / points_2d[2, :],
                points_2d[1, :] / points_2d[2, :],
                points_2d[2, :],
            ]
        )
        # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
        # contains all the y values of our points. In order to properly
        # visualize everything on a screen, the points that are out of the screen
        # must be discarted, the same with points behind the camera projection plane.
        points_2d = points_2d.T
        intensity = intensity.T
        points_in_canvas_mask = (
            (points_2d[:, 0] > 0.0)
            & (points_2d[:, 0] < self.image_w)
            & (points_2d[:, 1] > 0.0)
            & (points_2d[:, 1] < self.image_h)
            & (points_2d[:, 2] > 0.0)
        )
        points_2d = points_2d[points_in_canvas_mask]
        intensity = intensity[points_in_canvas_mask]

        # Extract the screen coords (uv) as integers.
        u_coord = points_2d[:, 0].astype(np.int)
        v_coord = points_2d[:, 1].astype(np.int)

        # Since at the time of the creation of this script, the intensity function
        # is returning high values, these are adjusted to be nicely visualized.
        intensity = 4 * intensity - 3
        color_map = (
            np.array(
                [
                    np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
                    np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
                    np.interp(intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0,
                ]
            )
            .astype(np.int)
            .T
        )
        s = 1
        im_array = np.copy(left_img)[:, :, :3]
        # im_array[v_coord, u_coord] = color_map
        # Draw the 2d points on the image as squares of extent args.dot_extent.
        for i in range(len(points_2d)):
            # I'm not a NumPy expert and I don't know how to set bigger dots
            # without using this loop, so if anyone has a better solution,
            # make sure to update this script. Meanwhile, it's fast enough :)
            im_array[
                v_coord[i] - s : v_coord[i] + s,
                u_coord[i] - s : u_coord[i] + s,
            ] = color_map[i]
        im_array = np.array(im_array)
        return points_2d, im_array

    @staticmethod
    def cam_to_output(P, points):
        """_summary_

        Args:
            P (_type_): _description_
            points (_type_): _description_

        Returns:
            _type_: _description_
        """
        # change coordinate to output_frame
        output = [P @ np.vstack([p, np.ones(p.shape[1])]) for p in points]
        return output

    @staticmethod
    def img_to_cam(intrinsics, points):
        """given intrinsics and a list of points, generate the points in camera coordinate

        Args:
            intrinsics (np.ndarray): 3x3 intrinsics matrix
            points (np.ndarray): 3xn array of points, in order of [u,v,s]

        Returns:
            list: list of points in camera coordinate. If there is error, return empty list
        """
        try:
            output = []
            for points in points:
                points = points.T  # 3xn
                # transform points back to camera coordinate
                points_2d = np.array(
                    [
                        points[0, :] * points[2, :],
                        points[1, :] * points[2, :],
                        points[2, :],
                    ]
                )  # 3xn
                cam_points = np.dot(np.linalg.inv(intrinsics), points_2d)  # 3xn
                output.append(cam_points)  # mx3xn, where m is number of detections
            return output
        except np.linalg.LinAlgError:
            print(f"Intrinsics error!!!: {intrinsics}")
            return []
        except Exception as e:
            print(f"Error: {e}")
            return []

    def get_cam_to_output_transform(self):
        try:
            now = rclpy.time.Time()
            trans_msg = self.tf_buffer.lookup_transform(
                self.output_frame_id, self.to_frame_rel, now
            )
            P = msg_to_se3(trans_msg)
        except TransformException as ex:
            self.get_logger().info(
                f"Could not transform {self.output_frame_id} to {self.to_frame_rel}: {ex}"
            )
            return
        return P

    def publish_detection_3d_array_visual(self, centers):
        """generate and publish markers for visualization purpose

        Args:
            centers (list): list of 3x1 points
        """
        header: Header = Header()
        header.frame_id = self.output_frame_id
        header.stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        for center in centers:
            marker = Marker(
                header=header,
                type=1,
                pose=Pose(
                    position=Point(
                        x=float(center[0]), y=float(center[1]), z=float(center[2])
                    )
                ),
                scale=Vector3(x=float(1), y=float(1), z=float(3)),
                color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
                lifetime=BuiltInDuration(nanosec=int(1e8)),  # 0.1 sec
            )
            markers.markers.append(marker)
        self.detection_3d_array_visual_publisher.publish(markers)

    def publish_detection_3d_array(self, centers):
        """generate and publish detection message

        Args:
             centers (list): list of 3x1 points
        """
        header: Header = Header()
        header.frame_id = self.output_frame_id
        header.stamp = self.get_clock().now().to_msg()
        detection3darray = Detection3DArray(header=header)
        for center in centers:
            bbox = BoundingBox3D()
            bbox.center = Pose(
                position=Point(
                    x=float(center[0]), y=float(center[1]), z=float(center[2])
                )
            )
            bbox.size = Vector3(x=float(1), y=float(1), z=float(1))
            detection3d = Detection3D(header=header, bbox=bbox)
            detection3darray.detections.append(detection3d)
        self.detection_3d_array_publisher.publish(detection3darray)

    @staticmethod
    def draw_filtered_points(img, filtered_points):
        img_copy = np.copy(img)
        for points in filtered_points:
            u_coord = points[:, 0].astype(np.int)
            v_coord = points[:, 1].astype(np.int)
            s = 1

            for i in range(len(points)):
                # I'm not a NumPy expert and I don't know how to set bigger dots
                # without using this loop, so if anyone has a better solution,
                # make sure to update this script. Meanwhile, it's fast enough :)
                img_copy[
                    v_coord[i] - s : v_coord[i] + s,
                    u_coord[i] - s : u_coord[i] + s,
                ] = [0, 0, 0]
        cv2.imshow("filtered points", img_copy)

    @staticmethod
    def get_points_only_in_bbox(bbox, points):
        """
        @param
            bbox = ymin, xmin, ymax, xmax
            points: Nx3 array of points [u,v,s]
        """
        ymin, xmin, ymax, xmax = bbox

        mask = np.where(
            (points[:, 0] >= xmin)
            & (points[:, 0] <= xmax)
            & (points[:, 1] >= ymin)
            & (points[:, 1] <= ymax)
        )
        result = points[mask]
        return result

    def process_image(self, image_msg):
        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        crops, crops_coordinates = extract_crops(
            frame, CROP_HEIGHT, CROP_WIDTH, CROP_STEP_VERTICAL, CROP_STEP_VERTICAL
        )

        # # Uncomment this if you also uncommented the two lines before
        # #  creating the TF session.
        # # crops = np.array([crops[0]])
        # # crops_coordinates = [crops_coordinates[0]]

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

        return boxes, boxes_scores, frame


def main(args=None):
    rclpy.init()
    node = ConeDetectorNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
