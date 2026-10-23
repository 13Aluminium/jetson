#!/usr/bin/env python3
"""Publish changing sample X guidance messages without camera or hardware."""
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimulatedGuidancePublisher(Node):
    def __init__(self):
        super().__init__("x_target_guidance_simulator")
        self.declare_parameter("topic", "/x_target/guidance")
        self.publisher = self.create_publisher(
            String, self.get_parameter("topic").value, 10
        )
        self.samples = [
            {"target_detected": True, "horizontal_offset_px": -120,
             "vertical_offset_px": 40, "confidence": 0.91,
             "centered": False, "guidance": "<< LEFT (120px) | V:OK"},
            {"target_detected": True, "horizontal_offset_px": 0,
             "vertical_offset_px": 0, "confidence": 0.96,
             "centered": True, "guidance": "CENTERED"},
            {"target_detected": False, "horizontal_offset_px": None,
             "vertical_offset_px": None, "confidence": 0.0,
             "centered": False, "guidance": "NO_TARGET"},
        ]
        self.index = 0
        self.create_timer(1.0, self.publish_sample)

    def publish_sample(self):
        msg = String()
        msg.data = json.dumps(self.samples[self.index])
        self.publisher.publish(msg)
        self.get_logger().info(msg.data)
        self.index = (self.index + 1) % len(self.samples)


def main():
    rclpy.init()
    node = SimulatedGuidancePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
