#!/usr/bin/env python3
"""Run x_detect_guide.py and publish its guidance over ROS 2.

The original camera, YOLO, OpenCV window, and terminal display remain in use.
This adapter only adds a ROS 2 JSON publisher.
"""
import argparse
import time
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import x_detect_guide as vision


class GuidancePublisher(Node):
    """Publish the latest guidance state at 10 Hz."""

    def __init__(self, topic):
        super().__init__("x_target_guidance_publisher")
        self.publisher = self.create_publisher(String, topic, 10)
        self._latest = self._not_detected()
        self._last_detection = 0.0
        self.create_timer(0.1, self._publish_latest)

    @staticmethod
    def _not_detected():
        return {
            "target_detected": False,
            "horizontal_offset_px": None,
            "vertical_offset_px": None,
            "confidence": 0.0,
            "centered": False,
            "guidance": "NO_TARGET",
        }

    def update_detection(self, dx, dy, confidence, centered, guidance):
        self._latest = {
            "target_detected": True,
            "horizontal_offset_px": int(dx),
            "vertical_offset_px": int(dy),
            "confidence": float(confidence),
            "centered": bool(centered),
            "guidance": str(guidance),
        }
        self._last_detection = time.monotonic()

    def _publish_latest(self):
        # Clear the state promptly if the camera loop stops seeing the X.
        if time.monotonic() - self._last_detection > 0.5:
            self._latest = self._not_detected()
        msg = String()
        msg.data = json.dumps(self._latest)
        self.publisher.publish(msg)


def make_parser():
    parser = argparse.ArgumentParser(
        description="X Target Detector + Centering Guide with ROS 2 publishing"
    )
    parser.add_argument("--weights", default="best_22.pt")
    parser.add_argument("--mode", choices=["4k", "1080p"], default="1080p")
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--deadzone", type=int, default=50)
    parser.add_argument("--ros-topic", default="/x_target/guidance")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--headless", action="store_true")
    modes.add_argument("--snapshot", action="store_true")
    return parser


def main():
    args = make_parser().parse_args()
    rclpy.init()
    node = GuidancePublisher(args.ros_topic)

    # Hook into the existing guidance calculation and center finders. Keeping
    # the original run_live/run_headless functions preserves camera handling,
    # visualization, snapshots, and terminal output.
    original_compute = vision.compute_guidance
    original_bbox_center = vision.find_x_center_bbox
    original_refined_center = vision.find_x_center_refined
    last_confidence = {"value": 0.0}

    def record_bbox_center(box):
        last_confidence["value"] = float(box.conf[0])
        return original_bbox_center(box)

    def record_refined_center(frame, box):
        last_confidence["value"] = float(box.conf[0])
        return original_refined_center(frame, box)

    def publish_guidance(*call_args, **call_kwargs):
        result = original_compute(*call_args, **call_kwargs)
        direction, dx, dy, centered = result
        node.update_detection(dx, dy, last_confidence["value"], centered, direction)
        rclpy.spin_once(node, timeout_sec=0.0)
        return result

    vision.find_x_center_bbox = record_bbox_center
    vision.find_x_center_refined = record_refined_center
    vision.compute_guidance = publish_guidance

    try:
        if args.headless:
            vision.run_headless(args)
        elif args.snapshot:
            vision.run_snapshot(args)
        else:
            vision.run_live(args)
    finally:
        vision.find_x_center_bbox = original_bbox_center
        vision.find_x_center_refined = original_refined_center
        vision.compute_guidance = original_compute
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
