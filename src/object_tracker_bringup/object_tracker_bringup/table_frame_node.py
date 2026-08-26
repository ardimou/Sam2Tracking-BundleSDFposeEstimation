"""Estimate a stable table frame and transform raw object poses into it."""
from collections import OrderedDict
import json
import math
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
import tf2_ros

from object_tracker_common import topics
from object_tracker_msgs.msg import (
    InstanceMask, InstanceSupervisorStatus, InstanceTablePose,
    InstanceTrackerStatus, MaskState)


class TableFrameNode(Node):

    def __init__(self):
        super().__init__('table_frame_node')
        self._declare_parameters()
        self._validate_parameters()
        self._enabled = bool(self.get_parameter('use_table_frame').value)
        self._bridge = CvBridge()
        self._depth_frames = OrderedDict()
        self._masks_by_stamp = OrderedDict()
        self._mask_arrival_times = {}
        self._pending_stamp = None
        self._T_camera_table = None
        self._estimation_requested = True
        self._last_estimation_attempt = 0.0
        self._previous_table_z = None
        self._camera_frame = str(self.get_parameter('camera_frame_id').value)
        self._active_ids = set()
        self._initial_table_rotations = {}
        self._initial_vertical_object = {}
        self._initial_tilts = {}
        self._T_raw_object_viz = {}

        self._static_tf = tf2_ros.StaticTransformBroadcaster(self)
        self._object_tf = tf2_ros.TransformBroadcaster(self)
        self._table_pose_pub = self.create_publisher(
            InstanceTablePose, topics.INSTANCE_TABLE_POSE_TOPIC,
            topics.reliable_qos(depth=32))
        self._raw_table_pose_pub = self.create_publisher(
            InstanceTablePose, topics.INSTANCE_TABLE_RAW_POSE_TOPIC,
            topics.reliable_qos(depth=32))
        self._primary_pose_pub = self.create_publisher(
            PoseStamped, topics.TABLE_POSE_TOPIC, topics.latest_only_qos())
        self._status_pub = self.create_publisher(
            String, topics.TABLE_FRAME_STATUS_TOPIC, topics.latched_qos())
        self.create_service(
            Trigger, str(self.get_parameter('table_reestimate_service').value),
            self._on_reestimate)

        if self._enabled:
            self.create_subscription(
                Image, str(self.get_parameter('depth_topic').value),
                self._on_depth, topics.latest_only_qos())
            self.create_subscription(
                CameraInfo, str(self.get_parameter('camera_info_topic').value),
                self._on_camera_info, topics.latest_only_qos())
            self.create_subscription(
                InstanceMask, str(self.get_parameter('instance_mask_topic').value),
                self._on_mask, topics.reliable_qos(depth=32))
            self.create_subscription(
                InstanceTrackerStatus,
                str(self.get_parameter('camera_instance_pose_topic').value),
                self._on_camera_pose, topics.reliable_qos(depth=32))
            self.create_subscription(
                InstanceSupervisorStatus, topics.INSTANCE_TRACKING_STATE_TOPIC,
                self._on_instance_state, topics.reliable_qos(depth=32))
            self.create_timer(0.05, self._try_estimate)
            self._publish_status({'state': 'waiting_for_masked_depth'})
            self.get_logger().info(
                'table_frame_node enabled: waiting for registered depth and '
                'the first SAM instance batch.')
        else:
            self._publish_status({'state': 'disabled'})
            self.get_logger().info(
                'table_frame_node disabled; existing camera-relative TF remains active.')

    def _declare_parameters(self):
        defaults = {
            'use_table_frame': True,
            'table_frame_id': topics.TABLE_FRAME,
            'camera_frame_id': topics.CAMERA_OPTICAL_FRAME,
            'depth_topic': topics.ADAPTER_DEPTH_TOPIC,
            'camera_info_topic': topics.ADAPTER_INFO_TOPIC,
            'instance_mask_topic': topics.INSTANCE_MASK_TOPIC,
            'camera_instance_pose_topic': topics.INSTANCE_TRACKER_STATUS_TOPIC,
            'table_reestimate_service': '/table_frame/reestimate',
            'table_ransac_distance_threshold_m': 0.015,
            'table_minimum_inlier_ratio': 0.30,
            'table_minimum_inlier_count': 400,
            'table_ransac_iterations': 80,
            'table_maximum_ransac_points': 5000,
            'table_estimation_retry_cooldown_sec': 2.0,
            'table_point_stride': 4,
            'table_minimum_depth_m': 0.25,
            'table_maximum_depth_m': 3.0,
            'table_mask_collection_delay_sec': 0.10,
            'table_mask_sync_tolerance_sec': 0.08,
            'table_estimate_once': True,
            'table_expected_up_camera_x': 0.0,
            'table_expected_up_camera_y': 0.0,
            'table_expected_up_camera_z': -1.0,
            'table_minimum_up_alignment': 0.50,
            'table_horizontal_reference_x': 1.0,
            'table_horizontal_reference_y': 0.0,
            'table_horizontal_reference_z': 0.0,
            'table_log_pose_matrices': False,
            'use_semantic_object_viz_frame': True,
            'semantic_object_frame_suffix': '_viz',
            'semantic_align_z_to_table': True,
            'semantic_x_source': 'project_raw_x',
            'publish_raw_object_frame': True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _validate_parameters(self):
        if float(self.get_parameter(
                'table_ransac_distance_threshold_m').value) <= 0.0:
            raise ValueError('table_ransac_distance_threshold_m must be positive')
        ratio = float(self.get_parameter('table_minimum_inlier_ratio').value)
        if not 0.0 < ratio <= 1.0:
            raise ValueError('table_minimum_inlier_ratio must be in (0, 1]')
        if int(self.get_parameter('table_point_stride').value) < 1:
            raise ValueError('table_point_stride must be positive')
        if int(self.get_parameter('table_ransac_iterations').value) < 1:
            raise ValueError('table_ransac_iterations must be positive')
        if int(self.get_parameter('table_maximum_ransac_points').value) < 3:
            raise ValueError('table_maximum_ransac_points must be at least 3')
        if float(self.get_parameter(
                'table_estimation_retry_cooldown_sec').value) < 0.0:
            raise ValueError(
                'table_estimation_retry_cooldown_sec must be non-negative')
        if not str(self.get_parameter('table_frame_id').value).strip():
            raise ValueError('table_frame_id must not be empty')
        if str(self.get_parameter('semantic_x_source').value) not in (
                'project_raw_x',):
            raise ValueError(
                'semantic_x_source currently supports only project_raw_x')
        suffix = str(self.get_parameter('semantic_object_frame_suffix').value)
        if not suffix or suffix == '_raw':
            raise ValueError(
                'semantic_object_frame_suffix must be non-empty and differ '
                'from the raw-frame suffix')

    @staticmethod
    def _stamp_key(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def _trim(cache, maximum=90):
        while len(cache) > maximum:
            cache.popitem(last=False)

    def _on_camera_info(self, msg):
        if not self._estimation_requested:
            return
        K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        if K[0, 0] > 0.0 and K[1, 1] > 0.0:
            self._K = K
            if msg.header.frame_id:
                self._camera_frame = msg.header.frame_id

    def _on_depth(self, msg):
        if not self._estimation_requested:
            return
        try:
            depth = self._bridge.imgmsg_to_cv2(
                msg, desired_encoding='32FC1').copy()
        except Exception as exc:
            self.get_logger().warn(f'Cannot decode table depth image: {exc}')
            return
        key = self._stamp_key(msg.header.stamp)
        self._depth_frames[key] = (depth, msg.header)
        self._depth_frames.move_to_end(key)
        self._trim(self._depth_frames)
        if msg.header.frame_id:
            self._camera_frame = msg.header.frame_id

    def _on_mask(self, msg):
        if not self._estimation_requested:
            return
        if (int(msg.state) in (MaskState.LOST, MaskState.SWITCHED) or
                not msg.mask.height or not msg.mask.width):
            return
        try:
            mask = self._bridge.imgmsg_to_cv2(
                msg.mask, desired_encoding='mono8') > 0
        except Exception as exc:
            self.get_logger().warn(f'Cannot decode table exclusion mask: {exc}')
            return
        key = self._stamp_key(msg.header.stamp)
        if key in self._masks_by_stamp:
            self._masks_by_stamp[key] |= mask
        else:
            self._masks_by_stamp[key] = mask.copy()
            self._mask_arrival_times[key] = time.monotonic()
        self._masks_by_stamp.move_to_end(key)
        self._trim(self._masks_by_stamp, 30)
        while len(self._mask_arrival_times) > 30:
            self._mask_arrival_times.pop(next(iter(self._mask_arrival_times)))
        if self._T_camera_table is None:
            self._pending_stamp = key

    def _nearest_depth(self, mask_stamp):
        if mask_stamp in self._depth_frames:
            return self._depth_frames[mask_stamp]
        if not self._depth_frames:
            return None
        nearest = min(self._depth_frames, key=lambda value: abs(value - mask_stamp))
        tolerance_ns = int(float(self.get_parameter(
            'table_mask_sync_tolerance_sec').value) * 1_000_000_000)
        return (self._depth_frames[nearest]
                if abs(nearest - mask_stamp) <= tolerance_ns else None)

    def _try_estimate(self):
        if not self._enabled or not self._estimation_requested:
            return
        if self._pending_stamp is None or not hasattr(self, '_K'):
            return
        arrival = self._mask_arrival_times.get(self._pending_stamp, 0.0)
        if time.monotonic() - arrival < float(self.get_parameter(
                'table_mask_collection_delay_sec').value):
            return
        now = time.monotonic()
        cooldown = float(self.get_parameter(
            'table_estimation_retry_cooldown_sec').value)
        if now - self._last_estimation_attempt < cooldown:
            return
        depth_entry = self._nearest_depth(self._pending_stamp)
        mask = self._masks_by_stamp.get(self._pending_stamp)
        if depth_entry is None or mask is None:
            return
        depth, header = depth_entry
        if mask.shape != depth.shape:
            self.get_logger().warn(
                'Table estimation rejected: mask/depth dimensions differ.',
                throttle_duration_sec=2.0)
            self._pending_stamp = None
            return
        self._last_estimation_attempt = now
        result = self._estimate_plane(depth, mask, self._K)
        self._pending_stamp = None
        if result is None:
            return
        T_camera_table, diagnostics = result
        new_z = T_camera_table[:3, 2]
        if self._previous_table_z is not None and np.dot(
                self._previous_table_z, new_z) < 0.0:
            self.get_logger().warn(
                'A refreshed RANSAC table normal attempted to flip sign; '
                'the estimate was rejected.')
            return
        self._T_camera_table = T_camera_table
        self._estimation_requested = False
        self._previous_table_z = new_z.copy()
        self._initial_table_rotations.clear()
        self._initial_vertical_object.clear()
        self._initial_tilts.clear()
        self._broadcast_table_transform(header)
        self._depth_frames.clear()
        self._masks_by_stamp.clear()
        self._mask_arrival_times.clear()
        self._publish_status({
            'state': 'ready',
            'estimate_once': bool(self.get_parameter('table_estimate_once').value),
            **diagnostics})
        self.get_logger().info(
            'table_frame initialized: '
            f'inliers={diagnostics["inlier_count"]} '
            f'ratio={diagnostics["inlier_ratio"]:.3f} '
            f'normal={diagnostics["normal_camera"]} '
            f'det={diagnostics["rotation_determinant"]:.6f}')

    def _estimate_plane(self, depth, exclusion_mask, K):
        stride = int(self.get_parameter('table_point_stride').value)
        sampled = np.asarray(depth, dtype=np.float64)[::stride, ::stride]
        excluded = np.asarray(exclusion_mask, dtype=bool)[::stride, ::stride]
        zmin = float(self.get_parameter('table_minimum_depth_m').value)
        zmax = float(self.get_parameter('table_maximum_depth_m').value)
        valid = (~excluded & np.isfinite(sampled) &
                 (sampled >= zmin) & (sampled <= zmax))
        rows, cols = np.nonzero(valid)
        minimum_count = int(self.get_parameter('table_minimum_inlier_count').value)
        if len(rows) < max(3, minimum_count):
            self._plane_failure(
                f'only {len(rows)} valid points remain after mask exclusion')
            return None
        rows_full = rows.astype(np.float64) * stride
        cols_full = cols.astype(np.float64) * stride
        z = sampled[rows, cols]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        points = np.column_stack((
            (cols_full - cx) * z / fx,
            (rows_full - cy) * z / fy,
            z,
        ))
        maximum_points = int(self.get_parameter(
            'table_maximum_ransac_points').value)
        if maximum_points > 0 and len(points) > maximum_points:
            sample_ids = np.linspace(
                0, len(points) - 1, maximum_points, dtype=np.int64)
            points = points[sample_ids]
        threshold = float(self.get_parameter(
            'table_ransac_distance_threshold_m').value)
        iterations = int(self.get_parameter('table_ransac_iterations').value)
        rng = np.random.default_rng(7)
        best_indices = None
        best_count = 0
        for _ in range(iterations):
            sample_ids = rng.choice(len(points), size=3, replace=False)
            a, b, c = points[sample_ids]
            normal = np.cross(b - a, c - a)
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-9:
                continue
            normal /= norm
            offset = -float(np.dot(normal, a))
            inliers = np.abs(points @ normal + offset) <= threshold
            count = int(np.count_nonzero(inliers))
            if count > best_count:
                best_count = count
                best_indices = inliers
        if best_indices is None or best_count < minimum_count:
            self._plane_failure(f'RANSAC found only {best_count} inliers')
            return None
        ratio = best_count / float(len(points))
        minimum_ratio = float(self.get_parameter(
            'table_minimum_inlier_ratio').value)
        if ratio < minimum_ratio:
            self._plane_failure(
                f'RANSAC inlier ratio {ratio:.3f} is below {minimum_ratio:.3f}')
            return None

        inlier_points = points[best_indices]
        centroid = np.mean(inlier_points, axis=0)
        _, _, vt = np.linalg.svd(inlier_points - centroid, full_matrices=False)
        normal = vt[-1]
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        offset = -float(np.dot(normal, centroid))
        expected_up = self._parameter_vector('table_expected_up_camera')
        expected_up /= max(float(np.linalg.norm(expected_up)), 1e-12)
        if np.dot(normal, expected_up) < 0.0:
            normal *= -1.0
            offset *= -1.0
        alignment = float(np.dot(normal, expected_up))
        minimum_alignment = float(self.get_parameter(
            'table_minimum_up_alignment').value)
        if alignment < minimum_alignment:
            self._plane_failure(
                f'normal/up alignment {alignment:.3f} is below '
                f'{minimum_alignment:.3f}; likely fitted a wall')
            return None

        reference = self._parameter_vector('table_horizontal_reference')
        x_axis = reference - normal * float(np.dot(reference, normal))
        if np.linalg.norm(x_axis) < 1e-6:
            for fallback in (np.array([1.0, 0.0, 0.0]),
                             np.array([0.0, 1.0, 0.0])):
                x_axis = fallback - normal * float(np.dot(fallback, normal))
                if np.linalg.norm(x_axis) >= 1e-6:
                    break
        x_axis /= max(float(np.linalg.norm(x_axis)), 1e-12)
        y_axis = np.cross(normal, x_axis)
        y_axis /= max(float(np.linalg.norm(y_axis)), 1e-12)
        x_axis = np.cross(y_axis, normal)
        x_axis /= max(float(np.linalg.norm(x_axis)), 1e-12)
        rotation = np.column_stack((x_axis, y_axis, normal))
        determinant = float(np.linalg.det(rotation))
        if determinant < 0.0:
            y_axis *= -1.0
            rotation = np.column_stack((x_axis, y_axis, normal))
            determinant = float(np.linalg.det(rotation))
        if not np.isfinite(rotation).all() or abs(determinant - 1.0) > 1e-3:
            self._plane_failure(
                f'generated table rotation is improper (det={determinant:.6f})')
            return None

        if abs(normal[2]) > 1e-6:
            ray_distance = -offset / normal[2]
            origin = np.array([0.0, 0.0, ray_distance], dtype=np.float64)
        else:
            origin = -offset * normal
        if not np.isfinite(origin).all() or origin[2] <= 0.0:
            origin = centroid
        T_camera_table = np.eye(4, dtype=np.float64)
        T_camera_table[:3, :3] = rotation
        T_camera_table[:3, 3] = origin
        diagnostics = {
            'inlier_count': best_count,
            'candidate_count': int(len(points)),
            'inlier_ratio': ratio,
            'normal_camera': [float(value) for value in normal],
            'up_alignment': alignment,
            'rotation_determinant': determinant,
            'origin_camera': [float(value) for value in origin],
        }
        return T_camera_table, diagnostics

    def _parameter_vector(self, prefix):
        return np.asarray([
            float(self.get_parameter(f'{prefix}_x').value),
            float(self.get_parameter(f'{prefix}_y').value),
            float(self.get_parameter(f'{prefix}_z').value),
        ], dtype=np.float64)

    def _plane_failure(self, reason):
        self._publish_status({'state': 'invalid', 'reason': reason})
        self.get_logger().warn(
            f'Table-plane estimation invalid: {reason}',
            throttle_duration_sec=2.0)

    def _broadcast_table_transform(self, source_header):
        transform = TransformStamped()
        transform.header.stamp = source_header.stamp
        transform.header.frame_id = self._camera_frame
        transform.child_frame_id = str(self.get_parameter('table_frame_id').value)
        transform.transform.translation.x = float(self._T_camera_table[0, 3])
        transform.transform.translation.y = float(self._T_camera_table[1, 3])
        transform.transform.translation.z = float(self._T_camera_table[2, 3])
        transform.transform.rotation = self._matrix_quaternion(
            self._T_camera_table[:3, :3])
        self._static_tf.sendTransform(transform)

    def _on_camera_pose(self, msg):
        if self._T_camera_table is None or not msg.status.tracking_success:
            return
        header = msg.status.header
        if header.frame_id and header.frame_id != self._camera_frame:
            self.get_logger().warn(
                f'Cannot compose table pose: tracker frame {header.frame_id} '
                f'differs from table camera frame {self._camera_frame}.',
                throttle_duration_sec=2.0)
            return
        object_id = int(msg.object_id)
        T_camera_object = self._pose_matrix(msg.status.t_camera_object.pose)
        T_table_camera = np.linalg.inv(self._T_camera_table)
        T_table_object_raw = T_table_camera @ T_camera_object
        rotation = T_table_object_raw[:3, :3]
        if object_id not in self._initial_table_rotations:
            self._initial_table_rotations[object_id] = rotation.copy()
            vertical_object = rotation.T @ np.array([0.0, 0.0, 1.0])
            self._initial_vertical_object[object_id] = vertical_object
            self._initial_tilts[object_id] = self._vertical_tilt(
                rotation, vertical_object)
            self._T_raw_object_viz[object_id] = (
                self._initialize_semantic_transform(object_id, rotation))
        T_raw_object_viz = self._T_raw_object_viz.get(
            object_id, np.eye(4, dtype=np.float64))
        T_table_object_viz = T_table_object_raw @ T_raw_object_viz
        initial_rotation = self._initial_table_rotations[object_id]
        relative_rotation = initial_rotation.T @ rotation
        acos_argument = float(np.clip(
            (np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
        rotation_from_initial = math.degrees(math.acos(acos_argument))
        tilt = self._vertical_tilt(
            rotation, self._initial_vertical_object[object_id])

        raw_output = self._table_pose_message(
            header.stamp, object_id, msg.label, T_table_object_raw,
            rotation_from_initial, tilt)
        if bool(self.get_parameter('publish_raw_object_frame').value):
            self._raw_table_pose_pub.publish(raw_output)

        output = self._table_pose_message(
            header.stamp, object_id, msg.label, T_table_object_viz,
            rotation_from_initial, tilt)
        self._table_pose_pub.publish(output)
        self._active_ids.add(object_id)

        self._broadcast_raw_and_semantic_tf(
            header.stamp, object_id, T_camera_object, T_raw_object_viz)
        self._broadcast_object_tf(output)
        if object_id == min(self._active_ids):
            primary = PoseStamped()
            primary.header = output.header
            primary.pose = output.pose
            self._primary_pose_pub.publish(primary)
            self._broadcast_primary_tf(output)

        if bool(self.get_parameter('table_log_pose_matrices').value):
            self.get_logger().info(
                'table pose composition\n'
                f'raw T_camera_object=\n{np.array2string(T_camera_object, precision=4)}\n'
                f'raw T_table_object=\n'
                f'{np.array2string(T_table_object_raw, precision=4)}\n'
                f'semantic T_table_object_viz=\n'
                f'{np.array2string(T_table_object_viz, precision=4)}\n'
                f'rotation_from_initial_deg={rotation_from_initial:.2f} '
                f'vertical_tilt_deg={tilt:.2f}',
                throttle_duration_sec=2.0)
        else:
            self.get_logger().debug(
                f'object {object_id} raw T_camera_object='
                f'{T_camera_object.tolist()} transformed T_table_object_raw='
                f'{T_table_object_raw.tolist()} semantic T_table_object_viz='
                f'{T_table_object_viz.tolist()}')
        self.get_logger().info(
            f'object {object_id} table diagnostics: '
            f'rotation_from_initial={rotation_from_initial:.1f}deg '
            f'vertical_tilt={tilt:.1f}deg '
            f'initial_tilt={self._initial_tilts[object_id]:.1f}deg',
            throttle_duration_sec=2.0)

    def _initialize_semantic_transform(self, object_id, R_table_raw):
        """Construct the fixed T_raw_viz without changing tracker output."""
        transform = np.eye(4, dtype=np.float64)
        if not bool(self.get_parameter(
                'use_semantic_object_viz_frame').value):
            return transform

        z_table = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if bool(self.get_parameter('semantic_align_z_to_table').value):
            z_raw = np.asarray(R_table_raw, dtype=np.float64).T @ z_table
        else:
            z_raw = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        z_raw /= max(float(np.linalg.norm(z_raw)), 1e-12)

        x_raw = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_raw -= float(np.dot(x_raw, z_raw)) * z_raw
        if float(np.linalg.norm(x_raw)) < 1e-6:
            x_raw = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            x_raw -= float(np.dot(x_raw, z_raw)) * z_raw
        if float(np.linalg.norm(x_raw)) < 1e-6:
            reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(float(np.dot(reference, z_raw))) > 0.9:
                reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            x_raw = np.cross(reference, z_raw)
        x_raw /= max(float(np.linalg.norm(x_raw)), 1e-12)
        y_raw = np.cross(z_raw, x_raw)
        y_raw /= max(float(np.linalg.norm(y_raw)), 1e-12)
        x_raw = np.cross(y_raw, z_raw)
        x_raw /= max(float(np.linalg.norm(x_raw)), 1e-12)
        R_raw_viz = np.column_stack((x_raw, y_raw, z_raw))
        determinant = float(np.linalg.det(R_raw_viz))
        if determinant < 0.0:
            y_raw *= -1.0
            x_raw = np.cross(y_raw, z_raw)
            x_raw /= max(float(np.linalg.norm(x_raw)), 1e-12)
            R_raw_viz = np.column_stack((x_raw, y_raw, z_raw))
            determinant = float(np.linalg.det(R_raw_viz))
        transform[:3, :3] = R_raw_viz
        quaternion = self._matrix_quaternion(R_raw_viz)
        self.get_logger().info(
            f'object {object_id} semantic frame initialized once:\n'
            f'initial R_table_raw=\n'
            f'{np.array2string(R_table_raw, precision=5)}\n'
            f'z_table={z_table.tolist()}\n'
            f'z_table expressed in raw={z_raw.tolist()}\n'
            f'viz axes in raw: X={x_raw.tolist()} Y={y_raw.tolist()} '
            f'Z={z_raw.tolist()}\n'
            f'det(R_raw_viz)={determinant:.8f}\n'
            f'q_raw_viz=({quaternion.x:.6f}, {quaternion.y:.6f}, '
            f'{quaternion.z:.6f}, {quaternion.w:.6f})')
        return transform

    def _table_pose_message(self, stamp, object_id, label, transform,
                            rotation_from_initial, tilt):
        output = InstanceTablePose()
        output.header.stamp = stamp
        output.header.frame_id = str(self.get_parameter('table_frame_id').value)
        output.object_id = object_id
        output.label = label
        output.pose = self._matrix_pose(transform)
        output.rotation_from_initial_deg = float(rotation_from_initial)
        output.vertical_tilt_deg = float(tilt)
        output.initial_vertical_tilt_deg = float(self._initial_tilts[object_id])
        return output

    def _broadcast_raw_and_semantic_tf(self, stamp, object_id,
                                       T_camera_object, T_raw_object_viz):
        raw_frame = f'tracked_object_{object_id}_raw'
        suffix = str(self.get_parameter('semantic_object_frame_suffix').value)
        viz_frame = f'tracked_object_{object_id}{suffix}'
        semantic = TransformStamped()
        semantic.header.stamp = stamp
        if bool(self.get_parameter('publish_raw_object_frame').value):
            semantic.header.frame_id = raw_frame
            semantic_matrix = T_raw_object_viz
        else:
            semantic.header.frame_id = str(
                self.get_parameter('table_frame_id').value)
            T_table_camera = np.linalg.inv(self._T_camera_table)
            semantic_matrix = (
                T_table_camera @ T_camera_object @ T_raw_object_viz)
        semantic.child_frame_id = viz_frame
        semantic_pose = self._matrix_pose(semantic_matrix)
        semantic.transform.translation.x = semantic_pose.position.x
        semantic.transform.translation.y = semantic_pose.position.y
        semantic.transform.translation.z = semantic_pose.position.z
        semantic.transform.rotation = semantic_pose.orientation
        self._object_tf.sendTransform(semantic)

    def _broadcast_object_tf(self, msg):
        transform = TransformStamped()
        transform.header = msg.header
        transform.child_frame_id = f'tracked_object_{msg.object_id}'
        transform.transform.translation.x = msg.pose.position.x
        transform.transform.translation.y = msg.pose.position.y
        transform.transform.translation.z = msg.pose.position.z
        transform.transform.rotation = msg.pose.orientation
        self._object_tf.sendTransform(transform)

    def _broadcast_primary_tf(self, msg):
        transforms = []
        for child_frame in (
                topics.TRACKED_OBJECT_FRAME,
                topics.TRACKED_OBJECT_FRAME + str(self.get_parameter(
                    'semantic_object_frame_suffix').value)):
            transform = TransformStamped()
            transform.header = msg.header
            transform.child_frame_id = child_frame
            transform.transform.translation.x = msg.pose.position.x
            transform.transform.translation.y = msg.pose.position.y
            transform.transform.translation.z = msg.pose.position.z
            transform.transform.rotation = msg.pose.orientation
            transforms.append(transform)
        self._object_tf.sendTransform(transforms)

    def _on_instance_state(self, msg):
        if int(msg.status.state) == 5:  # supervisor LOST
            object_id = int(msg.object_id)
            self._active_ids.discard(object_id)
            self._initial_table_rotations.pop(object_id, None)
            self._initial_vertical_object.pop(object_id, None)
            self._initial_tilts.pop(object_id, None)
            self._T_raw_object_viz.pop(object_id, None)

    def _on_reestimate(self, request, response):
        if not self._enabled:
            response.success = False
            response.message = 'table-frame feature is disabled'
            return response
        if bool(self.get_parameter('table_estimate_once').value):
            response.success = False
            response.message = (
                'table_estimate_once is true; set it false to permit an '
                'explicit /table_frame/reestimate request')
            return response
        self._previous_table_z = (None if self._T_camera_table is None else
                                  self._T_camera_table[:3, 2].copy())
        self._T_camera_table = None
        self._estimation_requested = True
        self._last_estimation_attempt = 0.0
        self._initial_table_rotations.clear()
        self._initial_vertical_object.clear()
        self._initial_tilts.clear()
        self._T_raw_object_viz.clear()
        if self._masks_by_stamp:
            self._pending_stamp = next(reversed(self._masks_by_stamp))
            self._mask_arrival_times[self._pending_stamp] = 0.0
        self._publish_status({'state': 'reestimate_requested'})
        response.success = True
        response.message = 'table plane will be re-estimated from the next synchronized mask/depth frame'
        return response

    def _publish_status(self, value):
        self._status_pub.publish(String(data=json.dumps(value)))

    @staticmethod
    def _pose_matrix(pose):
        x, y, z, w = (float(pose.orientation.x), float(pose.orientation.y),
                      float(pose.orientation.z), float(pose.orientation.w))
        norm = max(math.sqrt(x*x + y*y + z*z + w*w), 1e-12)
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        rotation = np.asarray([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return matrix

    @classmethod
    def _matrix_pose(cls, matrix):
        from geometry_msgs.msg import Pose
        pose = Pose()
        pose.position.x = float(matrix[0, 3])
        pose.position.y = float(matrix[1, 3])
        pose.position.z = float(matrix[2, 3])
        pose.orientation = cls._matrix_quaternion(matrix[:3, :3])
        return pose

    @staticmethod
    def _matrix_quaternion(rotation):
        from geometry_msgs.msg import Quaternion
        matrix = np.asarray(rotation, dtype=np.float64)
        trace = float(np.trace(matrix))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (matrix[2, 1] - matrix[1, 2]) / s
            y = (matrix[0, 2] - matrix[2, 0]) / s
            z = (matrix[1, 0] - matrix[0, 1]) / s
        else:
            index = int(np.argmax(np.diag(matrix)))
            if index == 0:
                s = math.sqrt(max(1e-12, 1.0 + matrix[0, 0] -
                                  matrix[1, 1] - matrix[2, 2])) * 2.0
                x = 0.25 * s
                y = (matrix[0, 1] + matrix[1, 0]) / s
                z = (matrix[0, 2] + matrix[2, 0]) / s
                w = (matrix[2, 1] - matrix[1, 2]) / s
            elif index == 1:
                s = math.sqrt(max(1e-12, 1.0 + matrix[1, 1] -
                                  matrix[0, 0] - matrix[2, 2])) * 2.0
                x = (matrix[0, 1] + matrix[1, 0]) / s
                y = 0.25 * s
                z = (matrix[1, 2] + matrix[2, 1]) / s
                w = (matrix[0, 2] - matrix[2, 0]) / s
            else:
                s = math.sqrt(max(1e-12, 1.0 + matrix[2, 2] -
                                  matrix[0, 0] - matrix[1, 1])) * 2.0
                x = (matrix[0, 2] + matrix[2, 0]) / s
                y = (matrix[1, 2] + matrix[2, 1]) / s
                z = 0.25 * s
                w = (matrix[1, 0] - matrix[0, 1]) / s
        norm = max(math.sqrt(x*x + y*y + z*z + w*w), 1e-12)
        return Quaternion(x=x/norm, y=y/norm, z=z/norm, w=w/norm)

    @staticmethod
    def _vertical_tilt(rotation, vertical_object):
        z_table = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        vertical_now = np.asarray(rotation) @ np.asarray(vertical_object)
        denominator = float(np.linalg.norm(vertical_now) * np.linalg.norm(z_table))
        if denominator <= 1e-12:
            return float('nan')
        cosine = float(np.clip(
            np.dot(vertical_now, z_table) / denominator, -1.0, 1.0))
        return math.degrees(math.acos(cosine))


def main(args=None):
    rclpy.init(args=args)
    node = TableFrameNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
