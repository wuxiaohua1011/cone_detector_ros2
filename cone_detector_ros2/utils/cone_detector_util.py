import numpy as np
import cupy as cp
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Transform
from geometry_msgs.msg import TransformStamped
import tf_transformations as tr
from matplotlib import cm
from .augmentations import letterbox
import cv2

VIRIDIS = cp.array(cm.get_cmap("viridis").colors)
VID_RANGE = cp.linspace(0.0, 1.0, VIRIDIS.shape[0])


def convert_image(img0, img_size=640, stride=32, auto=True):
    # Padded resize
    img = letterbox(img0, img_size, stride=stride, auto=True, scaleFill=False, scaleup=True)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    return img


def get_points_only_in_bbox(boxes, points, im):
    filtered_points = [get_points_only_in_bbox_helper(box, points=points, im=im) for box in boxes]
    filtered_points = [
        p for p in filtered_points if cp.shape(p) != () and cp.shape(p)[0] > 0
    ]  # get points that have values, not empty slices
    result = []
    for points in filtered_points:
        max_y = cp.max(points[:, 1])
        ps = points[points[:, 1] > max_y - 5]
        result.append(ps)
    return result


def get_points_only_in_bbox_helper(bbox, points, im):
    """
    @param
        bbox = ymin, xmin, ymax, xmax
        points: Nx3 array of points [u,v,s]
    """
    xmin, ymin, xmax, ymax = bbox
    img = cp.copy(im)[:, :, :3]
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cp.where(
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
    )
    points_in_box = points[mask]

    # define range of orange color in HSV
    lower_orange = cp.array([0, 49, 166])
    upper_orange = cp.array([33, 255, 255])

    # uvs is any point in points_in_box
    # uvs[1] is the v-coord and uvs[0] is the u-coord
    # so for any uvs point, I check if its hsv-value is within range
    result = cp.array(
        list(
            filter(
                lambda uvs: (lower_orange <= hsv_img[int(uvs[1]), int(uvs[0])]).all()
                & (  # all hsv values of point above lower_orange
                    hsv_img[int(uvs[1]), int(uvs[0])] <= upper_orange
                ).all(),  # all hsv values of point below upper_orange
                points_in_box,
            )
        )
    )
    return result


def pose_to_pq(msg):
    """Convert a C{geometry_msgs/Pose} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = cp.array([msg.position.x, msg.position.y, msg.position.z])
    q = cp.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
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
    p = cp.array([msg.translation.x, msg.translation.y, msg.translation.z])
    q = cp.array([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w])
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
    norm = cp.linalg.norm(q)
    if cp.abs(norm - 1.0) > 1e-3:
        raise ValueError(
            "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(
                str(q), cp.linalg.norm(q)
            )
        )
    elif cp.abs(norm - 1.0) > 1e-6:
        q = q / norm
    g = tr.quaternion_matrix(q)
    g[0:3, -1] = p
    return g


def draw_filtered_points(img, filtered_points):
    img_copy = cp.copy(img)
    for points in filtered_points:
        u_coord = points[:, 0].astype(cp.int)
        v_coord = points[:, 1].astype(cp.int)
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


def cam_to_output(P, points):
    """_summary_

    Args:
        P (_type_): _description_
        points (_type_): _description_

    Returns:
        _type_: _description_
    """
    # change coordinate to output_frame
    output = [P @ cp.vstack([p, cp.ones(p.shape[1])]) for p in points]
    return output


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
