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
    base_path = os.path.realpath(
        get_package_share_directory("cone_detector_ros2")
    )  # also tried without realpath
    rviz_path = base_path + "/configs/carla_rviz.rviz"
    return LaunchDescription(
        [
            launch.actions.IncludeLaunchDescription(
                launch.launch_description_sources.PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("roar_carla_ros2"),
                        "roar_carla_no_rviz.launch.py",
                    )
                )
            ),
            launch.actions.IncludeLaunchDescription(
                launch.launch_description_sources.PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("cone_detector_ros2"),
                        "cone_detector.launch.py",
                    )
                ),
                launch_arguments={"debug": "true"}.items(),
            ),
            launch_ros.actions.Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_path],
            ),
        ]
    )
