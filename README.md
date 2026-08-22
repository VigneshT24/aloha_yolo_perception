# ALOHA YOLO Perception (`aloha_yolo_pickup`)

ROS 2 perception package for Mobile ALOHA that runs YOLOv8 object detection, computes 3D depth back-projection, and publishes object poses into the `world` frame.

## Pipeline Introduction
### 1. Artificial Intelligence & Vision
*   **YOLOv8 (Medium):** A state-of-the-art, single-stage object detection model. We are running it in "stateless" mode, meaning it treats every frame independently to extract 2D bounding boxes and class labels (like "bottle" or "cup") without getting confused by the four different RealSense camera angles.
*   **OpenCV & CvBridge:** We use OpenCV for the visual bounding box overlays, and CvBridge to translate the raw ROS 2 image messages into standard NumPy arrays that YOLO and OpenCV can process.

### 2. 3D Spatial Mathematics & Tracking
*   **Depth Back-Projection:** This is the math that bridges 2D and 3D. By taking the center pixel of the YOLO bounding box and checking the aligned depth map, we use the camera's intrinsic matrix (focal length and optical center) to calculate the raw $X, Y, Z$ coordinates in meters.
*   **Principal Component Analysis (PCA):** If the depth map shows enough points for an object, the code uses PCA to find the longest axis of the object. This determines the primary angle, which is then converted into a 3D quaternion so the robot knows exactly how to orient its gripper.
*   **3D Euclidean Distance Matching:** Instead of tracking pixels on a 2D screen, we calculate the absolute distance between objects in the 3D `world` space. If a new detection pops up within 15 cm of an existing track, the system mathematically groups them together as the same physical object.
*   **Exponential Moving Average (EMA):** This is the temporal smoothing filter. By blending the newest, jittery depth reading with the historical average of that object's position, it acts as a low-pass filter to provide a rock-solid, stationary 3D coordinate for the robot arm to target.

### 3. Middleware & System Architecture
*   **TF2 (Transform Framework):** The ROS 2 library handling the complex matrix multiplications required to translate a coordinate from a specific camera's optical frame (e.g., `cam_left_wrist`) into the unified, absolute `world` frame.
*   **Asynchronous Threading:** Because four camera streams are hitting the system concurrently, we utilize Python's `threading.Lock()` and a ROS 2 `MultiThreadedExecutor`. This prevents the camera callbacks from crashing into each other while trying to access the YOLO model on your RTX 3050 simultaneously.

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

1. First, clone this repo:
   ```bash
   git clone https://github.com/VigneshT24/aloha_yolo_perception.git
   ```

2. Make sure you source all the terminals that you are going to run the following commands in using:
   ```bash
   source /opt/ros/humble/setup.bash
   source ~/interbotix_ws/install/setup.bash
   ```

3. Launch hardware cameras:
   ```bash
   ros2 launch aloha aloha_bringup.launch.py robot:=aloha_stationary use_cameras:=true
   ```
4. Publish the static frame transform:
   ```bash
   ros2 run tf2_ros static_transform_publisher 0.0 0.0 1.0 0.0 0.0 0.0 world cam_high_color_optical_frame
   ```
5. Run the YOLO detection node:
   ```bash
   ros2 launch aloha aloha_bringup.launch.py robot:=aloha_stationary use_cameras:=true
   ```
6. View the 2D visualizer:
   ```bash
   ros2 launch aloha aloha_bringup.launch.py robot:=aloha_stationary use_cameras:=true
   ```
7. Lastly, activate the mobile ALOHA arms
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
