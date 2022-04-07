from setuptools import setup
import os
from glob import glob

package_name = "cone_detector_ros2"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name), glob("launch/*.launch.py")),
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
            "video_player_node = cone_detector_ros2.video_player_node:main",
            "cone_detector_node = cone_detector_ros2.cone_detector_node:main",
            "test_node = cone_detector_ros2.test_node:main",
        ],
    },
)
