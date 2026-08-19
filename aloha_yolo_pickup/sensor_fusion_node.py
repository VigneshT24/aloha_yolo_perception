"""Sensor fusion node for Mobile ALOHA."""

import struct
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
)
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
import tf2_ros

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

def _extract_xyz_rgb(cloud_msg: PointCloud2) -> np.ndarray:
    """Return Nx6 float32 array [x,y,z,r,g,b] from a PointCloud2.

    Handles both XYZ-only and XYZRGB clouds.  RGB packed as float32 is
    unpacked to separate uint8 channels; if no RGB field is present the
    channel defaults to white (255).
    """
    fields = {f.name: f for f in cloud_msg.fields}
    point_step = cloud_msg.point_step
    data = bytes(cloud_msg.data)
    n_points = cloud_msg.width * cloud_msg.height

    x_off = fields['x'].offset
    y_off = fields['y'].offset
    z_off = fields['z'].offset

    xs = np.frombuffer(data, dtype=np.float32,
                       count=n_points, offset=x_off)[::point_step // 4]
    ys = np.frombuffer(data, dtype=np.float32,
                       count=n_points, offset=y_off)[::point_step // 4]
    zs = np.frombuffer(data, dtype=np.float32,
                       count=n_points, offset=z_off)[::point_step // 4]

    # Rebuild by iterating rows to respect point_step correctly
    xs = np.array([
        struct.unpack_from('f', data, i * point_step + x_off)[0]
        for i in range(n_points)
    ], dtype=np.float32)
    ys = np.array([
        struct.unpack_from('f', data, i * point_step + y_off)[0]
        for i in range(n_points)
    ], dtype=np.float32)
    zs = np.array([
        struct.unpack_from('f', data, i * point_step + z_off)[0]
        for i in range(n_points)
    ], dtype=np.float32)

    if 'rgb' in fields:
        rgb_off = fields['rgb'].offset
        rgb_packed = np.array([
            struct.unpack_from('I', data, i * point_step + rgb_off)[0]
            for i in range(n_points)
        ], dtype=np.uint32)
        rs = ((rgb_packed >> 16) & 0xFF).astype(np.float32)
        gs = ((rgb_packed >> 8) & 0xFF).astype(np.float32)
        bs = (rgb_packed & 0xFF).astype(np.float32)
    else:
        rs = np.full(n_points, 255.0)
        gs = np.full(n_points, 255.0)
        bs = np.full(n_points, 255.0)

    points = np.column_stack([xs, ys, zs, rs, gs, bs])
    # Remove NaN/Inf rows
    mask = np.isfinite(points[:, :3]).all(axis=1)
    return points[mask]

def _voxel_downsample(points: np.ndarray, leaf: float) -> np.ndarray:
    """Voxel grid downsampling.  Returns one representative point per voxel."""
    if len(points) == 0:
        return points
    coords = points[:, :3]
    voxel_idx = np.floor(coords / leaf).astype(np.int64)
    # Encode 3D voxel index to 1D key
    mins = voxel_idx.min(axis=0)
    voxel_idx -= mins
    dims = voxel_idx.max(axis=0) + 1
    keys = (voxel_idx[:, 0] * dims[1] * dims[2]
            + voxel_idx[:, 1] * dims[2]
            + voxel_idx[:, 2])
    order = np.argsort(keys)
    keys_sorted = keys[order]
    unique_keys, first_idx = np.unique(keys_sorted, return_index=True)
    representative = order[first_idx]
    return points[representative]

def _make_xyzrgb_cloud(points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """Pack Nx6 [x,y,z,r,g,b] array into a PointCloud2 message."""
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = len(points)
    msg.is_dense = True
    msg.is_bigendian = False

    fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
    ]
    msg.fields = fields
    msg.point_step = 16
    msg.row_step = 16 * len(points)

    buf = bytearray(msg.point_step * len(points))
    for i, (x, y, z, r, g, b) in enumerate(points):
        rgb = (int(r) << 16) | (int(g) << 8) | int(b)
        struct.pack_into('fffI', buf, i * 16, x, y, z, rgb)
    msg.data = bytes(buf)
    return msg

def _apply_transform(points_xyz: np.ndarray, t) -> np.ndarray:
    """Apply a TransformStamped to an Nx3 XYZ array."""
    tr = t.transform.translation
    rot = t.transform.rotation
    # Quaternion to rotation matrix (Hamilton convention)
    qx, qy, qz, qw = rot.x, rot.y, rot.z, rot.w
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [  2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),   2*(qy*qz - qx*qw)],
        [  2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)
    translated = points_xyz @ R.T + np.array([tr.x, tr.y, tr.z])
    return translated.astype(np.float32)

class SensorFusionNode(Node):
    """Merges three camera point clouds into a single fused cloud in base_link."""

    # Depth clip limits per camera key
    _DEPTH_CLIP = {
        'cam_high':     (0.20, 3.5),
        'cam_low':  (0.20, 3.5),
        'cam_left_wrist':  (0.05, 1.5),
        'cam_right_wrist': (0.05, 1.5),
    }

    # Workspace bounding box in base_link (m)
    _WS = (-1.5, 1.5, -1.5, 1.5, 0.0, 2.0)  # xmin,xmax,ymin,ymax,zmin,zmax

    def __init__(self):
        super().__init__('sensor_fusion_node')

        self._cbg = ReentrantCallbackGroup()
        self._lock = threading.Lock()

        # _clouds[key] = {'points': np.ndarray, 'stamp': float}
        self._clouds: dict = {}

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Publishers
        self._pub_fused = self.create_publisher(
            PointCloud2, '/fused/points', _reliable_qos(5))
        self._pub_status = self.create_publisher(
            String, '/fused/status', _reliable_qos(5))

        # subscriptions for the 4 RealSense cameras
        # camera_names = ['cam_top', 'cam_bottom', 'wrist_left', 'wrist_right']
        camera_names = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
        rqos = _reliable_qos(10)
        beqos = _best_effort_qos(5)

        for cam in camera_names:
            sub_qos = beqos if 'wrist' in cam else rqos
            self.create_subscription(
                PointCloud2, f'/{cam}/camera/depth/color/points',
                lambda m, c=cam: self._cloud_cb(m, c),
                sub_qos, callback_group=self._cbg)

        # Fusion timer at 10 Hz
        self.create_timer(0.1, self._fuse, callback_group=self._cbg)

        self.get_logger().info('Sensor fusion node ready')

    def _cloud_cb(self, msg: PointCloud2, key: str) -> None:
        """Receive a raw camera cloud, extract and clip, store for fusion."""
        try:
            pts = _extract_xyz_rgb(msg)
        except Exception as exc:
            self.get_logger().warn(
                f'[{key}] Failed to parse cloud: {exc}', throttle_duration_sec=5.0)
            return

        if len(pts) == 0:
            return

        # Depth clipping in camera frame (z axis = depth)
        zmin, zmax = self._DEPTH_CLIP[key]
        mask = (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
        pts = pts[mask]

        with self._lock:
            self._clouds[key] = {'points': pts, 'stamp': time.monotonic(),
                                 'frame': msg.header.frame_id}

    def _fuse(self) -> None:
        """Collect all valid clouds, transform to base_link, fuse, publish."""
        now_mono = time.monotonic()
        stale_timeout = 8.0
        target_frame = 'world'
        voxel_leaf = 0.01

        all_parts: list[np.ndarray] = []
        live: list[str] = []
        missing: list[str] = []

        with self._lock:
            snapshot = {k: v.copy() for k, v in self._clouds.items()}

        for key in ('cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'):
            entry = snapshot.get(key)
            if entry is None or (now_mono - entry['stamp']) > stale_timeout:
                missing.append(key)
                continue

            pts = entry['points']
            src_frame = entry['frame']
            if not src_frame:
                missing.append(key)
                continue

            # TF lookup
            try:
                tf = self._tf_buffer.lookup_transform(
                    target_frame, src_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.2))
            except Exception as exc:
                self.get_logger().warn(
                    f'[{key}] TF lookup {src_frame}→{target_frame} failed: {exc}',
                    throttle_duration_sec=5.0)
                missing.append(key)
                continue

            xyz_transformed = _apply_transform(pts[:, :3].astype(np.float64), tf)
            rgb = pts[:, 3:6]
            transformed = np.column_stack([xyz_transformed, rgb])
            all_parts.append(transformed)
            live.append(key)

        # Publish status regardless of cloud availability
        status_msg = String()
        status_msg.data = f'live={live} missing={missing}'
        self._pub_status.publish(status_msg)

        if not all_parts:
            return

        # Concatenate
        fused = np.vstack(all_parts)

        # Workspace bounding box filter
        xmin, xmax, ymin, ymax, zmin, zmax = self._WS
        mask = (
            (fused[:, 0] >= xmin) & (fused[:, 0] <= xmax) &
            (fused[:, 1] >= ymin) & (fused[:, 1] <= ymax) &
            (fused[:, 2] >= zmin) & (fused[:, 2] <= zmax)
        )
        fused = fused[mask]

        if len(fused) == 0:
            return

        # Voxel downsample
        fused = _voxel_downsample(fused, voxel_leaf)

        # Statistical outlier removal (disabled by default — uncomment to enable)
        # from sklearn.neighbors import NearestNeighbors
        # if len(fused) > 10:
        #     nbrs = NearestNeighbors(n_neighbors=10).fit(fused[:, :3])
        #     dists, _ = nbrs.kneighbors(fused[:, :3])
        #     mean_dist = dists[:, 1:].mean(axis=1)
        #     threshold = mean_dist.mean() + 2.0 * mean_dist.std()
        #     fused = fused[mean_dist < threshold]

        stamp = self.get_clock().now().to_msg()
        cloud_msg = _make_xyzrgb_cloud(fused, target_frame, stamp)
        self._pub_fused.publish(cloud_msg)

        self.get_logger().info(
            f'Fused {len(fused)} pts from {live}',
            throttle_duration_sec=2.0)


def main(args=None):
    """Entry point — spin with MultiThreadedExecutor (4 threads)."""
    rclpy.init(args=args)
    node = SensorFusionNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()