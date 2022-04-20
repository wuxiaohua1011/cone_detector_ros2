import launch
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    return launch.LaunchDescription(
        [
            launch.actions.IncludeLaunchDescription(
                launch.launch_description_sources.PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("cone_detector_ros2"),
                        "cone_detector.launch.py",
                    )
                ),
                launch_arguments={
                    "rgb_camera_topic": "/zed2i/zed_node/left/image_rect_color",
                    "rgb_camera_info_topic": "/zed2i/zed_node/left/camera_info",
                    "lidar_frame_id": "lidar_link",
                    "rgb_frame_id": "camera_link",
                    "output_frame_id": "camera_link",
                    "lidar_topic": "/livox/lidar",
                    "debug": "True",
                }.items(),
            )
        ]
    )
