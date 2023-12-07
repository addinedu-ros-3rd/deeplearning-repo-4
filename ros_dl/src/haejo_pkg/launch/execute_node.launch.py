from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                namespace = "camera", package = "usb_cam",
                executable = "usb_cam_node_exe", output = "screen"),
            Node(
                namespace = "detect_phone", package = "haejo_pkg",
                executable = "deep_learning", output = "screen"),
        ]
    )