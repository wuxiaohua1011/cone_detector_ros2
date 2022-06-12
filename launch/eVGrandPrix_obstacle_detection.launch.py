import launch
import os
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    cone_detector_node = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            [
                get_package_share_directory("cone_detector_ros2"),
                "/launch/cone_detector.launch.py",
            ]
        ),
        launch_arguments={
            "rgb_camera_topic": "/zed2_front_left/left/image_rect_color",
            "rgb_camera_info_topic": "/zed2_front_left/left/camera_info",
            "lidar_topics": "[/livox/lidar_3WEDH8A001R6541]",
            "confidence_threshold": "0.5",
            "output_frame_id": "base_link",
            "rgb_camera_frame_id": "zed2_front_left_left_camera_frame",
        }.items(),
    )
    return launch.LaunchDescription([cone_detector_node])
