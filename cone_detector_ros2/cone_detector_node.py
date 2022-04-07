#!/usr/bin/env python3
from distutils.log import debug
from lib2to3.pytree import convert
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
from image_geometry import PinholeCameraModel

from rclpy.qos import *
from .models.common import DetectMultiBackend
from .utils.general import (
    LOGGER,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_coords,
    strip_optimizer,
    xyxy2xywh,
)
from .utils.torch_utils import select_device
import torch
import torch.backends.cudnn as cudnn
from .utils.plots import Annotator, colors, save_one_box
from .utils.augmentations import letterbox

VIRIDIS = np.array(cm.get_cmap("viridis").colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


def convert_image(img0, img_size=640, stride=32, auto=True):
    # Padded resize
    img = letterbox(
        img0, img_size, stride=stride, auto=True, scaleFill=False, scaleup=True
    )[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    return img


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
        self.get_logger().info(f"Debug mode is set to: [{self.debug}]")

        self.bridge = CvBridge()
        self.get_logger().info(
            f"Listening to RGB Camera topic: {self.get_parameter('rgb_camera_topic').get_parameter_value().string_value}"
        )

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
        self.get_logger().info(
            f"Listening to RGB Camera Info on topic: {self.get_parameter('rgb_camera_info_topic').get_parameter_value().string_value}"
        )
        self.get_logger().info(
            f"Listening to Lidar on topic: {self.get_parameter('lidar_topic').get_parameter_value().string_value}"
        )
        self.is_carla = (
            True
            if "carla"
            in self.get_parameter("rgb_camera_topic").get_parameter_value().string_value
            else False
        )
        self.get_logger().info(f"Is Carla? {self.is_carla}")

        ### Subscriber

        profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5,
        )

        self.front_left_img = message_filters.Subscriber(
            self,
            Image,
            self.get_parameter("rgb_camera_topic").get_parameter_value().string_value,
            qos_profile=profile,
        )

        self.center_lidar = message_filters.Subscriber(
            self,
            PointCloud2,
            self.get_parameter("lidar_topic").get_parameter_value().string_value,
            qos_profile=profile,
        )
        queue_size = 100
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.front_left_img, self.center_lidar],
            queue_size=queue_size,
            slop=5,
            allow_headerless=True,
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
            Marker, "cone_detection_visual", 10
        )
        self.get_logger().debug("Publisher initialized")

        ### Camera Lidar Projection variables
        self.has_received_intrinsics = False
        self.intrinsics = np.zeros(shape=(3, 3))
        self.image_w = 762
        self.image_h = 386

        self.device = select_device(device="")
        self.weights_path = "/home/roar/Desktop/projects/roar-indy-ws/src/cone_detector_ros2/configs/best.pt"
        self.model = DetectMultiBackend(weights=self.weights_path, device=self.device)
        self.stride, self.names, self.pt = (
            self.model.stride,
            self.model.names,
            self.model.pt,
        )
        self.imgsz = (640, 640)
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        self.model.warmup(imgsz=(1 if self.pt else 1, 3, *self.imgsz))

        self.camera_model = PinholeCameraModel()
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

        boxes = self.process_image(im=original_image)
        if len(boxes) == 0 and self.debug:
            self.get_logger().info("No Traffic Cone detected")
        else:
            pass
        # project lidar onto rgb
        points_2d, lidar_camera_image = self.project_lidar_onto_camera(
            left_cam_msg=left_cam_msg, center_lidar_pcl_msg=center_lidar_pcl_msg
        )
        if points_2d is None or lidar_camera_image is None:
            return

        # filter points that are in the bounding box
        filtered_points = [
            self.get_points_only_in_bbox(box, points=points_2d) for box in boxes
        ]
        # change back to camera coordinate
        output = self.img_to_cam(points=filtered_points)
        if len(output) == 0:
            return
        # convert from cam -> output frame, if nessecary
        if self.to_frame_rel != self.output_frame_id:
            output = self.cam_to_output(
                P=self.get_cam_to_output_transform(), points=output
            )
        # compute the centroids for the detections in respective frames
        centers = []
        for points in output:
            avgs = np.average(points, axis=1)
            mins = np.min(points, axis=1)
            centers.append([avgs[0], avgs[1], 0])
        # publish Detection3DArray and visual msg
        self.publish_detection_3d_array(centers)
        self.publish_detection_3d_array_visual(centers)

        if self.debug:
            self.draw_filtered_points(original_image, filtered_points=filtered_points)
            cv2.imshow("lidar_projection", lidar_camera_image)
            cv2.waitKey(1)

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
        # Process predictions
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    minx, miny, maxx, maxy = [
                        int(t.cpu().detach().numpy()) for t in xyxy
                    ]
                    padding = int((maxy - miny) * 0.8)
                    result_boxes.append([minx, miny + padding, maxx, maxy])
        if self.debug:
            cv2.imshow("bbox", im0)
            cv2.waitKey(1)
        return result_boxes

    def camera_info_callback(self, msg):
        self.intrinsics = np.reshape(msg.k, newshape=(3, 3))
        self.image_w = msg.width
        self.image_h = msg.height
        self.has_received_intrinsics = True
        self.camera_model.fromCameraInfo(msg)
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
        # and remember that in standard image coordinate, we have x as side ways, y as up and down, z as depth
        # so we need to convert to that format.
        if self.is_carla is False:
            point_in_camera_coords = np.array(
                [
                    -point_in_camera_coords[1, :],
                    -point_in_camera_coords[2, :],
                    point_in_camera_coords[0, :],
                ]
            )
        # convert points to image coordinate
        points_2d = np.dot(self.intrinsics, point_in_camera_coords)
        # in reality, lidar are going to have readings with 0 (unknown), we need to filter out those readings
        mask = points_2d[2, :] != 0
        points_2d = points_2d.T[mask].T
        intensity = intensity[mask]
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

    def img_to_cam(self, points):
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
                points_2d = np.array(
                    [
                        ps[0, :] * ps[2, :],
                        ps[1, :] * ps[2, :],
                        ps[2, :],
                    ]
                )  # 3xn
                cam_points = np.dot(np.linalg.inv(self.intrinsics), points_2d)  # 3xn
                if self.is_carla is False:
                    cam_points = np.array(
                        [cam_points[2], -cam_points[0], -cam_points[1]]
                    )
                output.append(cam_points)  # mx3xn, where m is number of detections
            return output
        except np.linalg.LinAlgError:
            print(f"Intrinsics error!!!: {self.intrinsics}")
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
        markers = Marker()
        markers.header = header
        markers.type = 6
        if self.is_carla is False:
            markers.scale = Vector3(x=float(0.5), y=float(0.2), z=float(0.2))
        else:
            markers.scale = Vector3(x=float(1), y=float(1), z=float(3))
        markers.lifetime = BuiltInDuration(nanosec=int(2e8))
        for center in centers:
            point = Point(x=float(center[0]), y=float(center[1]), z=float(center[2]))
            markers.colors.append(ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0))
            markers.points.append(point)
        print(f"Publishing {len(markers.points)} markers")

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
        xmin, ymin, xmax, ymax = bbox

        mask = np.where(
            (points[:, 0] >= xmin)
            & (points[:, 0] <= xmax)
            & (points[:, 1] >= ymin)
            & (points[:, 1] <= ymax)
        )
        result = points[mask]
        return result


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


def main(args=None):
    rclpy.init()
    node = ConeDetectorNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
