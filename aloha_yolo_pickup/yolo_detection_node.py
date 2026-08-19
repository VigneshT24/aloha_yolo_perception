"""YOLO object detection node for Mobile ALOHA"""

import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
)
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Quaternion
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers transforms


def _reliable_qos(depth: int) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def _best_effort_qos(depth: int) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [qx, qy, qz, qw]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


class YoloDetectionNode(Node):
    """Runs YOLOv8 on camera images and publishes 3D object poses."""

    # Optical frame per camera key
    _OPTICAL_FRAME = {
        'cam_high': 'cam_high_color_optical_frame',
        'cam_low': 'cam_low_color_optical_frame',
        'cam_left_wrist': 'cam_left_wrist_color_optical_frame',
        'cam_right_wrist': 'cam_right_wrist_color_optical_frame',
    }

    def __init__(self):
        super().__init__('yolo_detection_node')

        self.declare_parameter('model', 'yolov8n.pt')
        self.declare_parameter('confidence', 0.5)
        self.declare_parameter('target_classes', [])
        self.declare_parameter('use_best_camera', True)

        model_path = self.get_parameter('model').value
        self._confidence = self.get_parameter('confidence').value
        self._target_classes = self.get_parameter('target_classes').value
        self._use_best_camera = self.get_parameter('use_best_camera').value

        from ultralytics import YOLO
        self._model = YOLO(model_path)
        self._bridge = CvBridge()

        self._cbg = ReentrantCallbackGroup()
        self._lock = threading.Lock()

        # Per-camera caches: {cam: latest_msg}
        self._rgb: dict[str, Image] = {}
        self._depth: dict[str, Image] = {}
        self._cam_info: dict[str, CameraInfo] = {}

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Publishers
        qos = _reliable_qos(5)
        self._pub_det = self.create_publisher(
            Detection2DArray, '/yolo_detections', qos)
        self._pub_poses = self.create_publisher(
            PoseArray, '/detected_object_poses', qos)
        self._pub_grasp = self.create_publisher(
            PoseStamped, '/grasp_pose', qos)
        self._pub_vis = self.create_publisher(
            Image, '/yolo_visualization', qos)

        # subscribe dynamically to all 4 RealSense camera streams
        # self._camera_names = ['cam_top', 'cam_bottom', 'wrist_left', 'wrist_right']
        self._camera_names = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
        rqos = _reliable_qos(10)
        beqos = _best_effort_qos(5)

        for cam in self._camera_names:
            #use best effort for the moving wrist cameras, reliable for the static ones
            sub_qos = beqos if 'wrist' in cam else rqos
            
            self.create_subscription(
                Image, f'/{cam}/camera/color/image_raw',
                lambda m, c=cam: self._rgb_cb(m, c), sub_qos, callback_group=self._cbg)
            
            self.create_subscription(
                Image, f'/{cam}/camera/aligned_depth_to_color/image_raw',
                lambda m, c=cam: self._depth_cb(m, c), sub_qos, callback_group=self._cbg)
            
            self.create_subscription(
                CameraInfo, f'/{cam}/camera/color/camera_info',
                lambda m, c=cam: self._info_cb(m, c), sub_qos, callback_group=self._cbg)

    def _rgb_cb(self, msg: Image, cam: str) -> None:
        with self._lock:
            self._rgb[cam] = msg
        self._run_inference(cam)

    def _depth_cb(self, msg: Image, cam: str) -> None:
        with self._lock:
            self._depth[cam] = msg

    def _info_cb(self, msg: CameraInfo, cam: str) -> None:
        with self._lock:
            if cam not in self._cam_info:
                self._cam_info[cam] = msg

    def _run_inference(self, cam: str) -> None:
        """Run YOLOv8 on the latest RGB from `cam`, publish results."""
        with self._lock:
            rgb_msg = self._rgb.get(cam)
            depth_msg = self._depth.get(cam)
            info_msg = self._cam_info.get(cam)

        if rgb_msg is None:
            return
        if info_msg is None:
            self.get_logger().debug(f'[{cam}] No camera_info yet, skipping')
            return

        try:
            bgr = self._bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        except Exception as exc:
            self.get_logger().warn(f'[{cam}] cv_bridge rgb error: {exc}')
            return

        depth_img = None
        if depth_msg is not None:
            try:
                depth_img = self._bridge.imgmsg_to_cv2(
                    depth_msg, desired_encoding='passthrough')
            except Exception as exc:
                self.get_logger().debug(f'[{cam}] cv_bridge depth error: {exc}')

        # Run YOLO
        results = self._model(bgr, verbose=False)
        K = np.array(info_msg.k).reshape(3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        optical_frame = self._OPTICAL_FRAME.get(cam, f'{cam}_color_optical_frame')
        det_array = Detection2DArray()
        det_array.header.stamp = rgb_msg.header.stamp
        det_array.header.frame_id = 'world'

        poses_3d: list[tuple[Pose, float]] = []  # (pose_in_base, confidence)

        vis_img = bgr.copy()

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self._confidence:
                    continue
                cls_id = int(box.cls[0])
                class_name = self._model.names.get(cls_id, str(cls_id))
                if (self._target_classes
                        and class_name not in self._target_classes):
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                px = int((x1 + x2) / 2)
                py = int((y1 + y2) / 2)

                # Build Detection2D
                det = Detection2D()
                det.header = det_array.header
                det.bbox.center.position.x = (x1 + x2) / 2
                det.bbox.center.position.y = (y1 + y2) / 2
                det.bbox.size_x = x2 - x1
                det.bbox.size_y = y2 - y1
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = class_name
                hyp.hypothesis.score = conf
                det.results.append(hyp)
                det_array.detections.append(det)

                # 3D back-projection
                pose_cam = self._back_project(
                    px, py, x1, y1, x2, y2,
                    depth_img, fx, fy, cx, cy)
                if pose_cam is None:
                    continue

                # Transform to world
                try:
                    tf = self._tf_buffer.lookup_transform(
                        'world', optical_frame,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.2))
                except Exception as exc:
                    self.get_logger().warn(
                        f'[{cam}] TF {optical_frame}→world: {exc}',
                        throttle_duration_sec=5.0)
                    continue

                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = optical_frame
                pose_stamped.header.stamp = rgb_msg.header.stamp
                pose_stamped.pose = pose_cam
                try:
                    pose_base = tf2_geometry_msgs.do_transform_pose(
                        pose_cam, tf)
                except Exception as exc:
                    self.get_logger().warn(f'[{cam}] Transform error: {exc}')
                    continue

                poses_3d.append((pose_base, conf))

                # Visualization
                cv2.rectangle(vis_img, (int(x1), int(y1)),
                              (int(x2), int(y2)), (0, 255, 0), 2)
                label = f'{cam}:{class_name} {conf:.2f}'
                cv2.putText(vis_img, label, (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Publish detection array
        self._pub_det.publish(det_array)

        # Publish pose array
        if poses_3d:
            pa = PoseArray()
            pa.header.stamp = rgb_msg.header.stamp
            pa.header.frame_id = 'world'
            pa.poses = [p for p, _ in poses_3d]
            self._pub_poses.publish(pa)

            # Best grasp = highest confidence
            best_pose, best_conf = max(poses_3d, key=lambda x: x[1])
            gp = PoseStamped()
            gp.header.stamp = rgb_msg.header.stamp
            gp.header.frame_id = 'world'
            gp.pose = best_pose
            self._pub_grasp.publish(gp)

        # Publish visualization
        try:
            vis_msg = self._bridge.cv2_to_imgmsg(vis_img, 'bgr8')
            vis_msg.header = rgb_msg.header
            self._pub_vis.publish(vis_msg)
        except Exception as exc:
            self.get_logger().debug(f'[{cam}] vis publish error: {exc}')

    def _back_project(self, px, py, x1, y1, x2, y2,
                      depth_img, fx, fy, cx, cy) -> 'Pose | None':
        """Back-project bbox center to 3D.  Returns Pose in camera optical frame."""
        if depth_img is None:
            return None

        h, w = depth_img.shape[:2]
        px_clamped = max(0, min(px, w - 1))
        py_clamped = max(0, min(py, h - 1))

        # Try centroid of depth segment inside bbox first
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(w, int(x2)), min(h, int(y2))
        region = depth_img[y1i:y2i, x1i:x2i].astype(np.float32)
        valid = region[(region > 0) & np.isfinite(region)]

        if len(valid) > 5:
            raw_depth = float(np.median(valid))
        else:
            raw_depth = float(depth_img[py_clamped, px_clamped])

        if raw_depth <= 0 or not np.isfinite(raw_depth):
            return None

        # Convert mm → m if depth looks like millimetres
        depth_m = raw_depth / 1000.0 if raw_depth > 100.0 else raw_depth

        # Segment centroid for XY
        ys_idx, xs_idx = np.where((region > 0) & np.isfinite(region))
        if len(xs_idx) > 5:
            xs_3d = (xs_idx + x1i - cx) * depth_m / fx
            ys_3d = (ys_idx + y1i - cy) * depth_m / fy
            zs_3d = np.full_like(xs_3d, depth_m)
            pts = np.column_stack([xs_3d, ys_3d, zs_3d])

            centroid = pts.mean(axis=0)
            obj_x, obj_y, obj_z = centroid

            # PCA for orientation
            if len(pts) >= 3:
                cov = np.cov(pts[:, :2].T)
                eigvals, eigvecs = np.linalg.eigh(cov)
                major = eigvecs[:, -1]  # eigenvector of largest eigenvalue
                angle = float(np.arctan2(major[1], major[0]))
                # Top-down grasp: rotate around Z by angle
                half = angle / 2.0
                quat = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
            else:
                quat = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            obj_x = (px - cx) * depth_m / fx
            obj_y = (py - cy) * depth_m / fy
            obj_z = depth_m
            quat = np.array([0.0, 0.0, 0.0, 1.0])

        pose = Pose()
        pose.position.x = float(obj_x)
        pose.position.y = float(obj_y)
        pose.position.z = float(obj_z)
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])
        return pose


def main(args=None):
    """Entry point — spin with MultiThreadedExecutor (4 threads)."""
    rclpy.init(args=args)
    node = YoloDetectionNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()