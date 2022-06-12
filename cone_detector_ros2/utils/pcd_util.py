## The code below is "ported" from
# https://github.com/ros/common_msgs/tree/noetic-devel/sensor_msgs/src/sensor_msgs
# I'll make an official port and PR to this repo later:
# https://github.com/ros2/common_interfaces
import sys
import math
import struct
from sensor_msgs.msg import PointCloud2, PointField

__docformat__ = "restructuredtext en"

import array
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import functools
import collections

_to_numpy = {}
_from_numpy = {}


def converts_to_numpy(msgtype, plural=False):
    def decorator(f):
        _to_numpy[msgtype, plural] = f
        return f

    return decorator


def converts_from_numpy(msgtype, plural=False):
    def decorator(f):
        _from_numpy[msgtype, plural] = f
        return f

    return decorator


def numpify(msg, *args, **kwargs):
    if msg is None:
        return

    conv = _to_numpy.get((msg.__class__, False))
    if not conv and isinstance(msg, collections.Sequence):
        if not msg:
            raise ValueError("Cannot determine the type of an empty Collection")
        conv = _to_numpy.get((msg[0].__class__, True))

    if not conv:
        raise ValueError(
            "Unable to convert message {} - only supports {}".format(
                msg.__class__.__name__,
                ", ".join(
                    cls.__name__ + ("[]" if pl else "") for cls, pl in _to_numpy.keys()
                ),
            )
        )

    return conv(msg, *args, **kwargs)


def msgify(msg_type, numpy_obj, *args, **kwargs):
    conv = _from_numpy.get((msg_type, kwargs.pop("plural", False)))
    if not conv:
        raise ValueError(
            "Unable to build message {} - only supports {}".format(
                msg_type.__name__,
                ", ".join(
                    cls.__name__ + ("[]" if pl else "") for cls, pl in _to_numpy.keys()
                ),
            )
        )
    return conv(numpy_obj, *args, **kwargs)


# prefix to the names of dummy fields we add to get byte alignment
# correct. this needs to not clash with any actual field names
DUMMY_FIELD_PREFIX = "__"

# mappings between PointField types and numpy types
type_mappings = [
    (PointField.INT8, np.dtype("int8")),
    (PointField.UINT8, np.dtype("uint8")),
    (PointField.INT16, np.dtype("int16")),
    (PointField.UINT16, np.dtype("uint16")),
    (PointField.INT32, np.dtype("int32")),
    (PointField.UINT32, np.dtype("uint32")),
    (PointField.FLOAT32, np.dtype("float32")),
    (PointField.FLOAT64, np.dtype("float64")),
]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

# sizes (in bytes) of PointField types
pftype_sizes = {
    PointField.INT8: 1,
    PointField.UINT8: 1,
    PointField.INT16: 2,
    PointField.UINT16: 2,
    PointField.INT32: 4,
    PointField.UINT32: 4,
    PointField.FLOAT32: 4,
    PointField.FLOAT64: 8,
}


@converts_to_numpy(PointField, plural=True)
def fields_to_dtype(fields, point_step):
    """Convert a list of PointFields to a numpy record datatype."""
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(("%s%d" % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += pftype_sizes[f.datatype] * f.count

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(("%s%d" % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1

    return np_dtype_list


@converts_from_numpy(PointField, plural=True)
def dtype_to_fields(dtype):
    """Convert a numpy record datatype into a list of PointFields."""
    fields = []
    for field_name in dtype.names:
        np_field_type, field_offset = dtype.fields[field_name]
        pf = PointField()
        pf.name = field_name
        if np_field_type.subdtype:
            item_dtype, shape = np_field_type.subdtype
            pf.count = int(np.prod(shape))
            np_field_type = item_dtype
        else:
            pf.count = 1

        pf.datatype = nptype_to_pftype[np_field_type]
        pf.offset = field_offset
        fields.append(pf)
    return fields


@converts_to_numpy(PointCloud2)
def pointcloud2_to_array(cloud_msg, squeeze=True):
    """Converts a rospy PointCloud2 message to a numpy recordarray
    Reshapes the returned array to have shape (height, width), even if the
    height is 1.
    The reason for using np.frombuffer rather than struct.unpack is
    speed... especially for large point clouds, this will be <much> faster.
    """
    # construct a numpy record type equivalent to the point type of this cloud
    dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

    # parse the cloud into an array
    cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)

    # remove the dummy fields that were added
    cloud_arr = cloud_arr[
        [
            fname
            for fname, _type in dtype_list
            if not (fname[: len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)
        ]
    ]

    if squeeze and cloud_msg.height == 1:
        return np.reshape(cloud_arr, (cloud_msg.width,))
    else:
        return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))


@converts_from_numpy(PointCloud2)
def array_to_pointcloud2(cloud_arr, stamp=None, frame_id=None):
    """Converts a numpy record array to a sensor_msgs.msg.PointCloud2."""
    # make it 2d (even if height will be 1)
    cloud_arr = np.atleast_2d(cloud_arr)

    cloud_msg = PointCloud2()

    if stamp is not None:
        cloud_msg.header.stamp = stamp
    if frame_id is not None:
        cloud_msg.header.frame_id = frame_id
    cloud_msg.height = cloud_arr.shape[0]
    cloud_msg.width = cloud_arr.shape[1]
    cloud_msg.fields = dtype_to_fields(cloud_arr.dtype)
    cloud_msg.is_bigendian = False  # assumption
    cloud_msg.point_step = cloud_arr.dtype.itemsize
    cloud_msg.row_step = cloud_msg.point_step * cloud_arr.shape[1]
    cloud_msg.is_dense = all(
        [np.isfinite(cloud_arr[fname]).all() for fname in cloud_arr.dtype.names]
    )

    # The PointCloud2.data setter will create an array.array object for you if you don't
    # provide it one directly. This causes very slow performance because it iterates
    # over each byte in python.
    # Here we create an array.array object using a memoryview, limiting copying and
    # increasing performance.
    memory_view = memoryview(cloud_arr)
    if memory_view.nbytes > 0:
        array_bytes = memory_view.cast("B")
    else:
        # Casting raises a TypeError if the array has no elements
        array_bytes = b""
    as_array = array.array("B")
    as_array.frombytes(array_bytes)
    cloud_msg.data = as_array
    return cloud_msg


def merge_rgb_fields(cloud_arr):
    """Takes an array with named np.uint8 fields 'r', 'g', and 'b', and returns
    an array in which they have been merged into a single np.float32 'rgb'
    field. The first byte of this field is the 'r' uint8, the second is the
    'g', uint8, and the third is the 'b' uint8.
    This is the way that pcl likes to handle RGB colors for some reason.
    """
    r = np.asarray(cloud_arr["r"], dtype=np.uint32)
    g = np.asarray(cloud_arr["g"], dtype=np.uint32)
    b = np.asarray(cloud_arr["b"], dtype=np.uint32)
    rgb_arr = np.array((r << 16) | (g << 8) | (b << 0), dtype=np.uint32)

    # not sure if there is a better way to do this. i'm changing the type of
    # the array from uint32 to float32, but i don't want any conversion to take
    # place -jdb
    rgb_arr.dtype = np.float32

    # create a new array, without r, g, and b, but with rgb float32 field
    new_dtype = []
    for field_name in cloud_arr.dtype.names:
        field_type, field_offset = cloud_arr.dtype.fields[field_name]
        if field_name not in ("r", "g", "b"):
            new_dtype.append((field_name, field_type))
    new_dtype.append(("rgb", np.float32))
    new_cloud_arr = np.zeros(cloud_arr.shape, new_dtype)

    # fill in the new array
    for field_name in new_cloud_arr.dtype.names:
        if field_name == "rgb":
            new_cloud_arr[field_name] = rgb_arr
        else:
            new_cloud_arr[field_name] = cloud_arr[field_name]

    return new_cloud_arr


def split_rgb_field(cloud_arr):
    """Takes an array with a named 'rgb' float32 field, and returns an array in
    which this has been split into 3 uint 8 fields: 'r', 'g', and 'b'.
    (pcl stores rgb in packed 32 bit floats)
    """
    rgb_arr = cloud_arr["rgb"].copy()
    rgb_arr.dtype = np.uint32
    r = np.asarray((rgb_arr >> 16) & 255, dtype=np.uint8)
    g = np.asarray((rgb_arr >> 8) & 255, dtype=np.uint8)
    b = np.asarray(rgb_arr & 255, dtype=np.uint8)

    # create a new array, without rgb, but with r, g, and b fields
    new_dtype = []
    for field_name in cloud_arr.dtype.names:
        field_type, field_offset = cloud_arr.dtype.fields[field_name]
        if not field_name == "rgb":
            new_dtype.append((field_name, field_type))
    new_dtype.append(("r", np.uint8))
    new_dtype.append(("g", np.uint8))
    new_dtype.append(("b", np.uint8))
    new_cloud_arr = np.zeros(cloud_arr.shape, new_dtype)

    # fill in the new array
    for field_name in new_cloud_arr.dtype.names:
        if field_name == "r":
            new_cloud_arr[field_name] = r
        elif field_name == "g":
            new_cloud_arr[field_name] = g
        elif field_name == "b":
            new_cloud_arr[field_name] = b
        else:
            new_cloud_arr[field_name] = cloud_arr[field_name]
    return new_cloud_arr


def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    """Pulls out x, y, and z columns from the cloud recordarray, and returns
    a 3xN matrix.
    """
    # remove crap points
    if remove_nans:
        mask = (
            np.isfinite(cloud_array["x"])
            & np.isfinite(cloud_array["y"])
            & np.isfinite(cloud_array["z"])
        )
        cloud_array = cloud_array[mask]

    # pull out x, y, and z values
    points = np.zeros(cloud_array.shape + (3,), dtype=dtype)
    points[..., 0] = cloud_array["x"]
    points[..., 1] = cloud_array["y"]
    points[..., 2] = cloud_array["z"]

    return points


def pointcloud2_to_xyz_array(cloud_msg, remove_nans=True):
    return get_xyz_points(pointcloud2_to_array(cloud_msg), remove_nans=remove_nans)


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
