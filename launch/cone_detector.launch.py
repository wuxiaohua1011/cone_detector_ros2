from asyncio import base_subprocess
from email.mime import base

from sqlalchemy import true
from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
import launch_ros
from pathlib import Path
import launch


def generate_launch_description():
    base_path = os.path.realpath(get_package_share_directory("cone_detector_ros2"))
    model_path = (Path(base_path) / "configs" / "best.pt").as_posix()

    pointcloud_to_depth_img_node = Node(
        package="cone_detector_ros2",
        executable="pointcloud_to_depth_img_node",
        name="pointcloud_to_depth_img_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "rgb_camera_topic": launch.substitutions.LaunchConfiguration(
                    "rgb_camera_topic"
                ),
                "rgb_camera_info_topic": launch.substitutions.LaunchConfiguration(
                    "rgb_camera_info_topic"
                ),
                "lidar_topics": launch.substitutions.LaunchConfiguration(
                    "lidar_topics"
                ),
                "rgb_camera_frame_id": launch.substitutions.LaunchConfiguration(
                    "rgb_camera_frame_id"
                ),
                "num_frame_buffered": launch.substitutions.LaunchConfiguration(
                    "num_frame_buffered"
                ),
            },
        ],
    )
    cones_extractor_node = Node(
        package="cone_detector_ros2",
        executable="cones_extractor_node",
        name="cones_extractor_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "rgb_camera_info_topic": launch.substitutions.LaunchConfiguration(
                    "rgb_camera_info_topic"
                ),
                "rgb_camera_topic": launch.substitutions.LaunchConfiguration(
                    "rgb_camera_topic"
                ),
                "rgb_camera_frame_id": launch.substitutions.LaunchConfiguration(
                    "rgb_camera_frame_id"
                ),
                "output_frame_id": launch.substitutions.LaunchConfiguration(
                    "output_frame_id"
                ),
            },
        ],
    )
    rgb_cone_detector_node = Node(
        package="cone_detector_ros2",
        executable="rgb_cone_detector_node",
        name="rgb_cone_detector_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {"model_path": launch.substitutions.LaunchConfiguration("model_path")},
            {
                "rgb_camera_topic": launch.substitutions.LaunchConfiguration(
                    "rgb_camera_topic"
                )
            },
            {"classes": launch.substitutions.LaunchConfiguration("classes")},
            {
                "confidence_threshold": launch.substitutions.LaunchConfiguration(
                    "confidence_threshold"
                )
            },
            {
                "rgb_camera_frame_id": launch.substitutions.LaunchConfiguration(
                    "rgb_camera_frame_id"
                ),
            },
        ],
    )

    return LaunchDescription(
        [
            launch.actions.DeclareLaunchArgument(
                name="model_path", default_value=model_path
            ),
            launch.actions.DeclareLaunchArgument(
                name="rgb_camera_topic",
                default_value="/carla/ego_vehicle/front_left_rgb/image",
                description="image topic to subscribe to",
            ),
            launch.actions.DeclareLaunchArgument(
                name="rgb_camera_info_topic",
                default_value="/carla/ego_vehicle/front_left_rgb/camera_info",
            ),
            launch.actions.DeclareLaunchArgument(
                name="lidar_topics",
                default_value="[/carla/ego_vehicle/left_lidar,/carla/ego_vehicle/center_lidar,/carla/ego_vehicle/right_lidar]",
            ),
            launch.actions.DeclareLaunchArgument(
                name="confidence_threshold", default_value="0.65"
            ),
            launch.actions.DeclareLaunchArgument(
                name="classes", default_value="['box','cone']"
            ),
            launch.actions.DeclareLaunchArgument(
                name="rgb_camera_frame_id", default_value="None"
            ),
            launch.actions.DeclareLaunchArgument(
                name="num_frame_buffered", default_value="10"
            ),
            launch.actions.DeclareLaunchArgument(
                name="output_frame_id", default_value="None"
            ),
            rgb_cone_detector_node,
            pointcloud_to_depth_img_node,
            cones_extractor_node,
        ]
    )
