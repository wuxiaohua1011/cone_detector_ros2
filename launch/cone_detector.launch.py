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
                name="image_topic",
                default_value="/camera/front_left/image",
                description="image topic to subscribe to",
            ),
            Node(
                package="cone_detector_ros2",
                executable="cone_detector_node",
                name="conde_detector_node",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {"model_path": model_path},
                    {
                        "image_topic": launch.substitutions.LaunchConfiguration(
                            "image_topic"
                        )
                    },
                    {"debug": True},
                ],
            ),
        ]
    )
