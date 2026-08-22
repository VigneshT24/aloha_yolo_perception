# ALOHA YOLO Perception (`aloha_yolo_pickup`)

ROS 2 perception package for Mobile ALOHA that runs YOLOv8 object detection, computes 3D depth back-projection, and publishes object poses into the `world` frame.

## Important Prerequisites (System Reset)

To ensure a clean environment and prevent "Device Busy" or shared memory errors, **always run this reset sequence before starting the pipeline** (especially if a previous node crashed).

1. **Clear ROS 2 Shared Memory & Restart Daemon:**
   ```bash
   sudo rm -rf /dev/shm/*
   ros2 daemon stop
   ros2 daemon start

2. **Hard-Reset USB Controllers & Verify Devices:**
   ```bash
   sudo bash -c '
   echo -n 0000:00:14.0 > /sys/bus/pci/drivers/xhci_hcd/unbind
   sleep 5
   echo -n 0000:00:14.0 > /sys/bus/pci/drivers/xhci_hcd/bind
   '
   sleep 10
   
   # Sanity check to confirm the USB devices are visible again
   lsusb | grep 0403
   ls -l /dev/ttyDXL_*
   ```

## Quick Start

**Note**: You must run each of the following steps in a separate, newly opened terminal window.

1. Make sure you source all the terminals that you are going to run the following commands in using:
   ```bash
   source /opt/ros/humble/setup.bash
   source ~/interbotix_ws/install/setup.bash
   ```

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
   ros2 launch aloha aloha_bringup.launch.py robot:=aloha_stationary use_cameras:=true
   ```
4. View the 2D visualizer:
   ```bash
   ros2 launch aloha aloha_bringup.launch.py robot:=aloha_stationary use_cameras:=true
   ```
1. Lastly, activate the mobile ALOHA arms
   ```bash
   cd ~/interbotix_ws/src/aloha/scripts/
   python3 dual_side_teleop.py
   ```

## After You Change The Code:

```bash
cd ~/ros2_ws
rm -rf build/aloha_yolo_pickup install/aloha_yolo_pickup
colcon build --packages-select aloha_yolo_pickup
source install/setup.bash
```
