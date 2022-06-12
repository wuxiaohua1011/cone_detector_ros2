import rclpy
import rclpy.node
from std_msgs.msg import Header
from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D, ObjectHypothesisWithPose
from vision_msgs.msg import ObjectHypothesis
from geometry_msgs.msg import Point, Pose, Vector3
import json 
from pathlib import Path
from pprint import pprint
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration as BuiltInDuration
from std_msgs.msg import ColorRGBA


class Dummy3DPublisher(rclpy.node.Node):
    def __init__(self):
        super().__init__("dummy_3d_publisher_node")
        self.get_logger().info("initializing...")   

        # parameters
        self.declare_parameter("dt", 0.2)
        # self.declare_parameter("num_obs", 1)
        self.declare_parameter("config_file_path", "None")
        self.declare_parameter("target_frame", "map")
        self.dt = self.get_parameter("dt").get_parameter_value().double_value   
        # frame
        self.frame = self.get_parameter("target_frame").get_parameter_value().string_value
        self.get_logger().info(f"Publishing in {self.frame} frame")

        self.config_file_path:Path = Path(self.get_parameter("config_file_path").get_parameter_value().string_value)
        assert self.config_file_path.exists(), f"{self.config_file_path} does not exist"
        self.get_logger().info(f"Reading config path: {self.config_file_path }")
        self.data = json.load(self.config_file_path.open('r'))
        self.get_logger().info(f"Will be publishing boxes at [{self.frame}] frame")

        # publisher
        self.detection_3d_array_publisher = self.create_publisher(
            Detection3DArray,
            "/dummy_cone_detection",
            10,
        )
        self.detection_3d_array_visual_publisher = self.create_publisher(
            Marker,
            "/dummy_cone_detection_visual",
            10,
        )

        self.get_logger().info(
            f"Publishing 3d Array at [{self.detection_3d_array_publisher.topic}]"
        )
        self.get_logger().info(
            f"Visualizing Array at [{self.detection_3d_array_visual_publisher.topic}]"
        )
        
        self.create_timer(self.dt, self.timer_callback)
        self.get_logger().debug("Publisher initialized")

    def timer_callback(self):
        self.publish_dummy_3d_detections()
        self.publish_detection_3d_array_visual()

    def publish_dummy_3d_detections(self):
        """
        Publishes dummy 3d detections for simulation purposes
        """

        header = Header()
        header.frame_id = self.frame
        header.stamp = self.get_clock().now().to_msg()

        detection3darray = Detection3DArray(header=header)
        for _, diction in self.data.items():
            bbox = BoundingBox3D()
            bbox.center = Pose(position=Point(x=float(diction["x"]), y=float(diction["y"]), z=float(diction["z"])))
            bbox.size = Vector3(x=float(1), y=float(1), z=float(1))
            detection3d = Detection3D(header=header, bbox=bbox)
            obs = ObjectHypothesisWithPose()    
            obs.hypothesis = ObjectHypothesis(class_id=diction["class_id"], score=diction["confidence"])
            detection3d.results.append(obs)
            detection3darray.detections.append(detection3d)
        self.detection_3d_array_publisher.publish(detection3darray)

    def publish_detection_3d_array_visual(self):

        header: Header = Header()
        header.frame_id = self.frame
        header.stamp = self.get_clock().now().to_msg()
        markers = Marker()
        markers.header = header
        markers.type = 6
        markers.scale = Vector3(x=float(1), y=float(1), z=float(1))
        markers.lifetime = BuiltInDuration(nanosec=int(5e8))
        for _, diction in self.data.items():
            point = Point(x=float(diction["x"]), y=float(diction["y"]), z=float(diction["z"]))
            markers.colors.append(ColorRGBA(r=float(diction.get("r", 1.0)), 
                                            g=float(diction.get("g", 1.0)), 
                                            b=float(diction.get("b", 1.0)), 
                                            a=float(diction.get("a", 1.0))))
            markers.points.append(point)
        self.detection_3d_array_visual_publisher.publish(markers)

def main(args=None):
    rclpy.init()
    node = Dummy3DPublisher()
    rclpy.spin(node)


if __name__ == "__main__":
    main()



