import launch
import os
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

base_path = os.path.realpath(get_package_share_directory("cone_detector_ros2"))
rgb_distance_estimator_config_path = (
    Path(base_path) / "configs" / "rgb_distance_estimator_zed_config.json"
).as_posix()


def generate_launch_description():
    cone_detector_node = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            [
                get_package_share_directory("cone_detector_ros2"),
                "/launch/cone_detector_no_lidar.launch.py",
            ]
        ),
        launch_arguments={
            "rgb_camera_topic": "/zed2i/zed_node/left/image_rect_color",
            "rgb_camera_info_topic": "/zed2i/zed_node/left/camera_info",
            "confidence_threshold": "0.1",
            "rgb_camera_frame_id": "zed2i_left_camera_optical_frame",
            "output_frame_id": "base_link",
            "rgb_distance_estimator_config_path": rgb_distance_estimator_config_path,
        }.items(),
    )
    return launch.LaunchDescription([cone_detector_node])
