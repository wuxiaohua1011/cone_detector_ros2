from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    base_path = os.path.realpath(get_package_share_directory("cone_detector_ros2"))
    model_path = (Path(base_path) / "configs" / "frozen_inference_graph.pb").as_posix()
    video_path = (Path(base_path) / "configs" / "sample_video.mp4").as_posix()

    return LaunchDescription(
        [
            Node(
                package="cone_detector_ros2",
                executable="cone_detector_node",
                name="conde_detector_node",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {"model_path": model_path},
                    {"image_topic": "test_image"},
                    {"debug": True},
                ],
            ),
            Node(
                package="cone_detector_ros2",
                executable="video_player_node",
                name="video_player_node",
                output="screen",
                emulate_tty=True,
                parameters=[{"video_path": video_path}, {"debug": False}],
            ),
        ]
    )
