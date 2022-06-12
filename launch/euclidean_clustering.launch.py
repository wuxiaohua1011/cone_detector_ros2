from launch import LaunchDescription
from launch_ros.actions import Node
import launch


def generate_launch_description():
    euclidean_cluster_node = Node(
        package="obstacle_detector_ros2",
        executable="euclidean_clustering_node",
        name="euclidean_clustering_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "obstacle_points_topic": launch.substitutions.LaunchConfiguration(
                    "obstacle_points_topic"
                ),
                "eps": launch.substitutions.LaunchConfiguration("eps"),
                "min_points": launch.substitutions.LaunchConfiguration("min_points"),
            },
        ],
    )

    return LaunchDescription(
        [
            launch.actions.DeclareLaunchArgument(
                name="obstacle_points_topic",
                default_value="/carla/ego_vehicle/center_lidar/obstacle_points",
            ),
            launch.actions.DeclareLaunchArgument(
                name="eps",
                default_value="0.1",
            ),
            launch.actions.DeclareLaunchArgument(
                name="min_points",
                default_value="1",
            ),
            euclidean_cluster_node,
        ]
    )
