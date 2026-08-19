# ALOHA YOLO Perception (`aloha_yolo_pickup`)

ROS 2 perception package for Mobile ALOHA that runs YOLOv8 object detection, computes 3D depth back-projection, and publishes object poses into the `world` frame.

## Quick Start

1. Launch hardware cameras:
   ```bash
   ros2 launch aloha aloha_bringup.launch.py robot:=aloha_stationary use_cameras:=true
   ```
2. Publish the static frame transform:
   ```bash
   ros2 run tf2_ros static_transform_publisher 0.0 0.0 1.0 0.0 0.0 0.0 world cam_high_color_optical_frame
   ```
3. Run the YOLO detection node:
   ```bash
   ros2 run aloha_yolo_pickup yolo_detection_node
   ```
4. View the 2D visualizer:
   ```bash
   ros2 run rqt_image_view rqt_image_view
   ```
