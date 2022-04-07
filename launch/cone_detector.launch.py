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
    model_path = (Path(base_path) / "configs" / "frozen_inference_graph.pb").as_posix()

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
            launch.actions.DeclareLaunchArgument(name="debug", default_value="false"),
            launch.actions.DeclareLaunchArgument(
                name="rgb_camera_info_topic",
                default_value="/carla/ego_vehicle/front_left_rgb/camera_info",
            ),
            launch.actions.DeclareLaunchArgument(
                name="lidar_topic",
                default_value="/carla/ego_vehicle/center_lidar",
            ),
            launch.actions.DeclareLaunchArgument(
                name="lidar_frame_id", default_value="ego_vehicle/center_lidar"
            ),
            launch.actions.DeclareLaunchArgument(
                name="rgb_frame_id", default_value="ego_vehicle/front_left_rgb"
            ),
            launch.actions.DeclareLaunchArgument(
                name="output_frame_id", default_value="ego_vehicle"
            ),
            Node(
                package="cone_detector_ros2",
                executable="cone_detector_node",
                name="conde_detector_node",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "model_path": launch.substitutions.LaunchConfiguration(
                            "model_path"
                        )
                    },
                    {
                        "rgb_camera_topic": launch.substitutions.LaunchConfiguration(
                            "rgb_camera_topic"
                        )
                    },
                    {
                        "lidar_topic": launch.substitutions.LaunchConfiguration(
                            "lidar_topic"
                        )
                    },
                    {"debug": launch.substitutions.LaunchConfiguration("debug")},
                    {
                        "rgb_camera_info_topic": launch.substitutions.LaunchConfiguration(
                            "rgb_camera_info_topic"
                        )
                    },
                    {
                        "lidar_frame_id": launch.substitutions.LaunchConfiguration(
                            "lidar_frame_id"
                        )
                    },
                    {
                        "rgb_frame_id": launch.substitutions.LaunchConfiguration(
                            "rgb_frame_id"
                        )
                    },
                    {
                        "output_frame_id": launch.substitutions.LaunchConfiguration(
                            "output_frame_id"
                        )
                    },
                ],
            ),
        ]
    )
