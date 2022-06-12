from setuptools import setup, find_packages
import os
from glob import glob

package_name = "obstacle_detector_ros2"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["configs", "launch", "test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join(os.path.join("share", package_name), "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join(os.path.join("share", package_name), "configs"),
            glob("configs/*"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="roar",
    maintainer_email="wuxiaohua1011@berkeley.edu",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "rgb_cone_detector_node = obstacle_detector_ros2.rgb_cone_detector_node:main",
            "pointcloud_to_depth_img_node = obstacle_detector_ros2.pointcloud_to_depth_img_node:main",
            "cones_extractor_node = obstacle_detector_ros2.cones_extractor_node:main",
            "data_recorder_node=obstacle_detector_ros2.data_recorder_node:main",
            "object_distance_estimator_node=obstacle_detector_ros2.object_distance_estimator_node:main",
            "dummy_3d_publisher_node=obstacle_detector_ros2.dummy_3d_publisher_node:main",
            "euclidean_clustering_node=obstacle_detector_ros2.euclidean_clustering_node:main",
        ],
    },
)
