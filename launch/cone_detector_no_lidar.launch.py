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
    config_path = (
        Path(base_path) / "configs" / "rgb_distance_estimator_config.json"
    ).as_posix()
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

    object_distance_estimator_node = Node(
        package="cone_detector_ros2",
        executable="object_distance_estimator_node",
        name="object_distance_estimator_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "rgb_camera_topic": launch.substitutions.LaunchConfiguration(
                    "rgb_camera_topic"
                )
            },
            {
                "rgb_camera_info_topic": launch.substitutions.LaunchConfiguration(
                    "rgb_camera_info_topic"
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
            {
                "output_frame_id": launch.substitutions.LaunchConfiguration(
                    "output_frame_id"
                )
            },
            {
                "rgb_distance_estimator_config_path": launch.substitutions.LaunchConfiguration(
                    "rgb_distance_estimator_config_path"
                )
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
                name="confidence_threshold", default_value="0.65"
            ),
            launch.actions.DeclareLaunchArgument(
                name="classes", default_value="['box','cone']"
            ),
            launch.actions.DeclareLaunchArgument(
                name="rgb_camera_frame_id", default_value="None"
            ),
            launch.actions.DeclareLaunchArgument(
                name="output_frame_id", default_value="ego_vehicle"
            ),
            launch.actions.DeclareLaunchArgument(
                name="rgb_distance_estimator_config_path", default_value=config_path
            ),
            launch.actions.DeclareLaunchArgument(
                name="rgb_camera_info_topic",
                default_value="/carla/ego_vehicle/front_left_rgb/camera_info",
            ),
            rgb_cone_detector_node,
            object_distance_estimator_node,
        ]
    )
