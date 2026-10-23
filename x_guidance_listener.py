#!/usr/bin/env python3
"""Print X-target guidance JSON messages from the ROS 2 topic."""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class GuidanceListener(Node):
    def __init__(self):
        super().__init__("x_target_guidance_listener")
        self.declare_parameter("topic", "/x_target/guidance")
        topic = self.get_parameter("topic").value
        self.subscription = self.create_subscription(
            String, topic, self.on_message, 10
        )
        self.get_logger().info(f"Listening on {topic}")

    def on_message(self, msg):
        self.get_logger().info(msg.data)


def main():
    rclpy.init()
    node = GuidanceListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
