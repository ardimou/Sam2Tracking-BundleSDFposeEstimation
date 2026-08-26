"""Small Qt operator GUI backed entirely by ROS topics and services."""
from collections import deque
import json
import math
import sys
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from python_qt_binding.QtCore import Qt, QTimer
from python_qt_binding.QtGui import QImage, QPixmap
from python_qt_binding.QtWidgets import (
    QApplication, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger

from object_tracker_common import topics
from object_tracker_msgs.msg import (
    GroundingResult, InstanceSupervisorStatus, InstanceTablePose,
    InstanceTrackerStatus, MaskState, SupervisorStatus)


TRACKING_STATES = ['IDLE', 'GROUNDING', 'INITIALIZING', 'TRACKING',
                   'UNCERTAIN', 'LOST', 'REINITIALIZING']
MASK_STATES = ['INITIALIZING', 'TRACKING', 'UNCERTAIN', 'LOST', 'SWITCHED']


class OperatorGuiNode(Node):
    def __init__(self):
        super().__init__('object_tracker_gui')
        self.declare_parameter('initial_query', '')
        self.declare_parameter('debug_metrics_enabled', False)
        self.declare_parameter('use_table_frame', True)

        self.query_pub = self.create_publisher(
            String, topics.QUERY_TOPIC, topics.reliable_qos())
        self._pending_query = None
        self._have_annotated_image = False
        self._have_visualization_image = False
        self._grounding_available = False
        self._session_start_ns = 0
        self._latest_raw_bgr = None
        self._initial_pose_rotations = {}
        self._relative_rotation_degrees = {}
        self._vertical_tilt_degrees = {}
        self._initial_tilt_degrees = {}
        self._tracking_name = 'IDLE'
        self._tracking_confidence = 0.0
        self._last_pose_processing_ms = None
        self._cloud_receipt_times = deque(maxlen=60)
        self.create_timer(0.2, self._publish_pending_query)
        self.create_subscription(
            Image, topics.ADAPTER_RGB_TOPIC, self._on_rgb,
            topics.latest_only_qos())
        self.create_subscription(
            Image, topics.VIZ_IMAGE_TOPIC, self._on_visualization_image,
            topics.latest_only_qos())
        self.create_subscription(
            GroundingResult, topics.GROUNDING_RESULT_TOPIC, self._on_grounding,
            topics.reliable_qos())
        self.create_subscription(
            SupervisorStatus, topics.TRACKING_STATE_TOPIC, self._on_tracking,
            topics.reliable_qos())
        self.create_subscription(
            MaskState, topics.MASK_STATE_TOPIC, self._on_mask,
            topics.reliable_qos())
        self.create_subscription(
            String, topics.GROUNDING_STATUS_TOPIC, self._on_grounding_status,
            topics.latched_qos())
        self.create_subscription(
            InstanceTrackerStatus, topics.INSTANCE_POSE_TOPIC,
            self._on_instance_pose, topics.reliable_qos(depth=1))
        if bool(self.get_parameter('use_table_frame').value):
            self.create_subscription(
                InstanceTablePose, topics.INSTANCE_TABLE_POSE_TOPIC,
                self._on_table_pose, topics.reliable_qos(depth=1))
        self.create_subscription(
            InstanceSupervisorStatus, topics.INSTANCE_TRACKING_STATE_TOPIC,
            self._on_instance_tracking_state, topics.reliable_qos(depth=10))
        self.create_subscription(
            PointCloud2, topics.UNIFIED_CLOUD_TOPIC, self._on_unified_cloud,
            topics.reliable_qos(depth=1))

        self.reset_client = self.create_client(Trigger, topics.RESET_SERVICE)
        self.bridge = CvBridge()
        self.window = OperatorWindow(
            self, str(self.get_parameter('initial_query').value))
        self.create_timer(0.2, self._update_runtime_metrics)

    def detect(self, query):
        query = query.strip()
        if not query:
            self.window.set_notice('Enter an object description first.', error=True)
            return
        self._pending_query = query
        self._have_annotated_image = False
        self._have_visualization_image = False
        self._session_start_ns = self.get_clock().now().nanoseconds
        self._initial_pose_rotations.clear()
        self._relative_rotation_degrees.clear()
        self._vertical_tilt_degrees.clear()
        self._initial_tilt_degrees.clear()
        self._last_pose_processing_ms = None
        self._cloud_receipt_times.clear()
        self.window.cloud_window.clear_session()
        self.window.clear_session_status()
        if self._latest_raw_bgr is not None:
            self.window.set_image(self._latest_raw_bgr.copy())
        self._publish_pending_query()

    def _publish_pending_query(self):
        if not self._pending_query:
            return
        if not self._grounding_available:
            self.window.set_notice(
                'Waiting for GroundingDINO to initialize its ROS interfaces.')
            return
        query = self._pending_query
        self._pending_query = None
        self.query_pub.publish(String(data=query))
        self.window.set_notice(f'Detection requested for "{query}".')

    def reset(self):
        if not self.reset_client.service_is_ready():
            self.window.set_notice('Reset service is not available.', error=True)
            return
        future = self.reset_client.call_async(Trigger.Request())
        future.add_done_callback(lambda done: self._service_result(done, 'Reset'))

    def retry(self, query):
        """Reset BundleSDF first, then begin a completely fresh query."""
        query = query.strip()
        if not query:
            self.window.set_notice('Enter an object description first.', error=True)
            return
        if not self.reset_client.service_is_ready():
            self.window.set_notice(
                'Cannot retry: tracker reset service is not available.',
                error=True)
            return
        self.window.set_notice('Resetting tracker before retry …')
        future = self.reset_client.call_async(Trigger.Request())
        future.add_done_callback(
            lambda done, retry_query=query:
            self._retry_after_reset(done, retry_query))

    def _retry_after_reset(self, future, query):
        try:
            response = future.result()
            if not bool(getattr(response, 'success', False)):
                self.window.set_notice(
                    f'Retry reset failed: {getattr(response, "message", "")}',
                    error=True)
                return
            self.detect(query)
        except Exception as exc:
            self.window.set_notice(f'Retry reset failed: {exc}', error=True)

    def _service_result(self, future, action):
        try:
            response = future.result()
            accepted = getattr(response, 'accepted', getattr(response, 'success', False))
            message = getattr(response, 'message', '')
            self.window.set_notice(
                f'{action}: {message or ("accepted" if accepted else "rejected")}',
                error=not accepted)
        except Exception as exc:
            self.window.set_notice(f'{action} failed: {exc}', error=True)

    def _on_visualization_image(self, msg):
        stamp = (int(msg.header.stamp.sec) * 1_000_000_000 +
                 int(msg.header.stamp.nanosec))
        if stamp < self._session_start_ns:
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self._have_visualization_image = True
            self._have_annotated_image = True
            self.window.set_image(bgr)
        except Exception as exc:
            self.window.set_notice(
                f'Could not display pose visualization: {exc}', error=True)

    def _on_rgb(self, msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self._latest_raw_bgr = bgr.copy()
            if self._have_annotated_image:
                return
            self.window.set_image(bgr)
        except Exception as exc:
            self.window.set_notice(f'Could not display RGB image: {exc}', error=True)

    def _on_grounding_status(self, msg):
        status = msg.data
        self._grounding_available = True
        if status.startswith('loading:'):
            self.window.set_notice(f'GroundingDINO is loading ({status[8:]}).')
        elif status.startswith('ready:'):
            self.window.set_notice(f'GroundingDINO is ready ({status[6:]}).')
        elif status.startswith('error:'):
            self.window.set_notice(f'Grounding backend error: {status}', error=True)

    def _on_grounding(self, msg):
        self.window.set_grounding(msg)

    def _on_tracking(self, msg):
        name = TRACKING_STATES[msg.state] if msg.state < len(TRACKING_STATES) else str(msg.state)
        self._tracking_name = name
        self._tracking_confidence = float(msg.confidence)
        self._refresh_tracking_text()
        self.window.tracking_value.setStyleSheet(
            'color: #4ade80; font-weight: 700;' if name == 'TRACKING' else
            'color: #fb7185; font-weight: 700;' if name == 'LOST' else
            'color: #fbbf24; font-weight: 700;')
        if bool(self.get_parameter('debug_metrics_enabled').value):
            failure_text = msg.failure_reason or '—'
        elif name == 'LOST':
            failure_text = 'Tracking lost'
        elif name == 'UNCERTAIN':
            failure_text = 'Pose update uncertain'
        else:
            failure_text = '—'
        self.window.failure_value.setText(failure_text)

    def _refresh_tracking_text(self):
        text = f'{self._tracking_name} ({self._tracking_confidence:.2f})'
        if self._last_pose_processing_ms is not None:
            text += f'  |  pose {self._last_pose_processing_ms:.0f} ms'
        self.window.tracking_value.setText(text)

    def _on_mask(self, msg):
        name = MASK_STATES[msg.state] if msg.state < len(MASK_STATES) else str(msg.state)
        self.window.mask_value.setText(f'{name} ({msg.confidence:.2f})')

    @staticmethod
    def _quaternion_rotation_matrix(q):
        x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
        norm = math.sqrt(x*x + y*y + z*z + w*w)
        if norm <= 1e-12:
            return None
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        return np.asarray([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

    @staticmethod
    def _event_rate(receipt_times):
        cutoff = time.monotonic() - 5.0
        recent = [value for value in receipt_times if value >= cutoff]
        if len(recent) < 2:
            return 0.0
        elapsed = float(recent[-1] - recent[0])
        return (len(recent) - 1) / elapsed if elapsed > 1e-6 else 0.0

    def _on_instance_pose(self, msg):
        tracker = msg.status
        stamp = (int(tracker.header.stamp.sec) * 1_000_000_000 +
                 int(tracker.header.stamp.nanosec))
        if stamp < self._session_start_ns:
            return
        if not tracker.tracking_success:
            return
        object_id = int(msg.object_id)
        self.window.cloud_window.set_pose(
            object_id, tracker.t_camera_object.pose, msg.label)
        self._last_pose_processing_ms = float(tracker.processing_time_ms)
        self._refresh_tracking_text()
        if bool(self.get_parameter('use_table_frame').value):
            return
        rotation = self._quaternion_rotation_matrix(
            tracker.t_camera_object.pose.orientation)
        if rotation is not None:
            initial = self._initial_pose_rotations.setdefault(
                object_id, rotation.copy())
            relative = initial.T @ rotation
            acos_argument = float(np.clip(
                (np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
            self._relative_rotation_degrees[object_id] = math.degrees(
                math.acos(acos_argument))
            orientation_summary = ' · '.join(
                f'ID {instance_id}: {angle:.1f}°'
                for instance_id, angle in sorted(
                    self._relative_rotation_degrees.items()))
            self.window.pose_value.setText(orientation_summary)
    def _on_table_pose(self, msg: InstanceTablePose):
        stamp = (int(msg.header.stamp.sec) * 1_000_000_000 +
                 int(msg.header.stamp.nanosec))
        if stamp < self._session_start_ns:
            return
        object_id = int(msg.object_id)
        self._relative_rotation_degrees[object_id] = float(
            msg.rotation_from_initial_deg)
        self._vertical_tilt_degrees[object_id] = float(msg.vertical_tilt_deg)
        self._initial_tilt_degrees[object_id] = float(
            msg.initial_vertical_tilt_deg)
        summaries = []
        for instance_id in sorted(self._relative_rotation_degrees):
            rotation = self._relative_rotation_degrees[instance_id]
            tilt = self._vertical_tilt_degrees.get(instance_id, float('nan'))
            initial_tilt = self._initial_tilt_degrees.get(instance_id, 0.0)
            summaries.append(
                f'ID {instance_id}: rotation {rotation:.1f}° · '
                f'tilt {tilt:.1f}° (Δ{tilt - initial_tilt:+.1f}°)')
        self.window.pose_value.setText('  |  '.join(summaries))

    def _on_instance_tracking_state(self, msg):
        if int(msg.status.state) == 5:  # supervisor LOST
            object_id = int(msg.object_id)
            self.window.cloud_window.remove_pose(object_id)
            self._relative_rotation_degrees.pop(object_id, None)
            self._vertical_tilt_degrees.pop(object_id, None)
            self._initial_tilt_degrees.pop(object_id, None)
            orientation_summary = '  |  '.join(
                f'ID {instance_id}: rotation {angle:.1f}° · '
                f'tilt {self._vertical_tilt_degrees.get(instance_id, float("nan")):.1f}°'
                for instance_id, angle in sorted(
                    self._relative_rotation_degrees.items()))
            self.window.pose_value.setText(orientation_summary or '—')

    @staticmethod
    def _cloud_arrays(msg):
        count = int(msg.width) * int(msg.height)
        if count == 0:
            return (np.empty((0, 3), dtype=np.float32),
                    np.empty((0, 3), dtype=np.uint8))
        offsets = {field.name: int(field.offset) for field in msg.fields}
        if not all(name in offsets for name in ('x', 'y', 'z')):
            raise ValueError('PointCloud2 has no XYZ fields')
        xyz = [np.ndarray(
            shape=(count,), dtype='<f4', buffer=msg.data,
            offset=offsets[name], strides=(int(msg.point_step),)).copy()
            for name in ('x', 'y', 'z')]
        points = np.column_stack(xyz)
        if 'rgb' in offsets:
            packed = np.ndarray(
                shape=(count,), dtype='<u4', buffer=msg.data,
                offset=offsets['rgb'], strides=(int(msg.point_step),)).copy()
            colors = np.column_stack((
                (packed >> 16) & 255, (packed >> 8) & 255, packed & 255,
            )).astype(np.uint8)
        else:
            colors = np.full((count, 3), 220, dtype=np.uint8)
        valid = np.isfinite(points).all(axis=1)
        return points[valid], colors[valid]

    def _on_unified_cloud(self, msg):
        stamp = (int(msg.header.stamp.sec) * 1_000_000_000 +
                 int(msg.header.stamp.nanosec))
        if stamp < self._session_start_ns:
            return
        try:
            points, colors = self._cloud_arrays(msg)
            self.window.cloud_window.set_cloud(points, colors)
            self._cloud_receipt_times.append(time.monotonic())
        except Exception as exc:
            self.window.set_notice(
                f'Could not visualize unified cloud: {exc}', error=True)

    def _update_runtime_metrics(self):
        cloud_rate = self._event_rate(self._cloud_receipt_times)
        self.window.cloud_rate_value.setText(
            f'{cloud_rate:.1f} Hz' if self._cloud_receipt_times else '—')


class PointCloudWindow(QMainWindow):
    """Rotating unified scene cloud with all tracked 6DoF frames."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Unified tracked-object cloud')
        self.resize(800, 600)
        self.setStyleSheet(
            'QMainWindow { background: #0b1120; } '
            'QLabel { background: #111827; color: #cbd5e1; '
            'border: 1px solid #263449; }')
        self.label = QLabel('Waiting for /object/unified_cloud …')
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(
            'background: #111827; color: #cbd5e1; '
            'border: 1px solid #263449; padding: 6px;')
        self.setCentralWidget(self.label)
        self._points = np.empty((0, 3), dtype=np.float32)
        self._colors = np.empty((0, 3), dtype=np.uint8)
        self._poses = {}
        self._angle = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._render)
        self._timer.start(100)

    def set_cloud(self, points, colors):
        if len(points) > 40000:
            colors_array = np.asarray(colors, dtype=np.uint8)
            neutral_scene = (np.all(
                colors_array == colors_array[:, :1], axis=1) &
                (colors_array[:, 0] <= 100))
            blue_scene = np.all(
                colors_array == np.array([70, 110, 180], dtype=np.uint8),
                axis=1)
            scene_mask = neutral_scene | blue_scene
            object_indices = np.flatnonzero(~scene_mask)
            scene_indices = np.flatnonzero(scene_mask)
            scene_budget = max(0, 40000 - len(object_indices))
            if len(scene_indices) > scene_budget:
                if scene_budget:
                    sample = np.linspace(
                        0, len(scene_indices) - 1, scene_budget,
                        dtype=np.int64)
                    scene_indices = scene_indices[sample]
                else:
                    scene_indices = scene_indices[:0]
            indices = np.concatenate((scene_indices, object_indices))
            if len(indices) > 40000:
                sample = np.linspace(
                    0, len(indices) - 1, 40000, dtype=np.int64)
                indices = indices[sample]
            points, colors = points[indices], colors_array[indices]
        self._points = np.asarray(points, dtype=np.float32)
        self._colors = np.asarray(colors, dtype=np.uint8)
        if self.isVisible():
            self._render()

    def set_pose(self, object_id, pose, label):
        x, y, z, w = (float(pose.orientation.x), float(pose.orientation.y),
                      float(pose.orientation.z), float(pose.orientation.w))
        norm = max(math.sqrt(x*x + y*y + z*z + w*w), 1e-12)
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        rotation = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ], dtype=np.float32)
        translation = np.array(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=np.float32)
        self._poses[int(object_id)] = (rotation, translation, label)

    def remove_pose(self, object_id):
        self._poses.pop(int(object_id), None)

    def clear_session(self):
        self._points = np.empty((0, 3), dtype=np.float32)
        self._colors = np.empty((0, 3), dtype=np.uint8)
        self._poses.clear()
        self.label.clear()
        self.label.setText('Waiting for the new tracking session …')

    def _render(self):
        if not self.isVisible() or not len(self._points):
            return
        width = max(320, self.label.width())
        height = max(240, self.label.height())
        center = np.median(self._points, axis=0)
        points = self._points - center
        self._angle += 0.015
        c, s = math.cos(self._angle), math.sin(self._angle)
        yaw = np.array([[c, 0.0, s], [0.0, 1.0, 0.0],
                        [-s, 0.0, c]], dtype=np.float32)
        cp, sp = math.cos(-0.35), math.sin(-0.35)
        pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp],
                          [0.0, sp, cp]], dtype=np.float32)
        view_rotation = pitch @ yaw
        view = points @ view_rotation.T
        radius = float(np.percentile(np.linalg.norm(view[:, :2], axis=1), 98))
        scale = 0.42 * min(width, height) / max(radius, 1e-6)
        u = np.rint(view[:, 0] * scale + width * 0.5).astype(np.int32)
        v = np.rint(-view[:, 1] * scale + height * 0.5).astype(np.int32)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        order = np.argsort(view[:, 2])
        order = order[inside[order]]
        canvas = np.full((height, width, 3), 32, dtype=np.uint8)
        canvas[v[order], u[order]] = self._colors[order, ::-1]
        canvas = cv2.dilate(canvas, np.ones((2, 2), dtype=np.uint8))
        axis_colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
        for object_id, (rotation, translation, label) in self._poses.items():
            length = 0.12
            axes = np.vstack((
                translation,
                translation + rotation[:, 0] * length,
                translation + rotation[:, 1] * length,
                translation + rotation[:, 2] * length,
            ))
            axes_view = (axes - center) @ view_rotation.T
            au = np.rint(axes_view[:, 0] * scale + width * 0.5).astype(int)
            av = np.rint(-axes_view[:, 1] * scale + height * 0.5).astype(int)
            origin = (int(au[0]), int(av[0]))
            for index, axis_color in enumerate(axis_colors, start=1):
                cv2.arrowedLine(
                    canvas, origin, (int(au[index]), int(av[index])),
                    axis_color, 4, cv2.LINE_AA, tipLength=0.25)
            cv2.putText(canvas, f'ID {object_id}',
                        (origin[0] + 5, origin[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                        cv2.LINE_AA)
        cv2.putText(canvas, f'{len(self._points)} unified points', (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1,
                    cv2.LINE_AA)
        rgb = np.ascontiguousarray(canvas[:, :, ::-1])
        image = QImage(rgb.data, width, height, width * 3, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(image.copy()))

    def show_cloud(self):
        self.show()
        self.raise_()
        self.activateWindow()
        if not len(self._points):
            self.label.setText('Waiting for /object/unified_cloud …')
        else:
            self._render()


class OperatorWindow(QMainWindow):
    def __init__(self, node, initial_query):
        super().__init__()
        self.node = node
        self._pixmap = None
        self.cloud_window = PointCloudWindow()
        self.setWindowTitle('Object Tracker')
        self.resize(1040, 780)
        self.setStyleSheet("""
            QMainWindow { background: #0b1120; }
            QWidget#root { background: #0b1120; color: #e5e7eb; }
            QLabel { color: #dbe5f3; font-size: 13px; }
            QLabel#title { color: #f8fafc; font-size: 22px;
                           font-weight: 700; }
            QLabel#subtitle, QLabel#muted { color: #8291a8; font-size: 12px; }
            QLabel#cameraView { background: #080d18; color: #8291a8;
                                border: 1px solid #263449;
                                border-radius: 8px; padding: 4px; }
            QLineEdit { background: #111827; color: #f8fafc;
                        border: 1px solid #334155; border-radius: 6px;
                        padding: 9px 11px; selection-background-color: #2563eb; }
            QLineEdit:focus { border: 1px solid #60a5fa; }
            QPushButton { background: #1e293b; color: #e2e8f0;
                          border: 1px solid #334155; border-radius: 6px;
                          padding: 8px 16px; font-weight: 600; }
            QPushButton:hover { background: #334155; }
            QPushButton:pressed { background: #172033; }
            QPushButton#primary { background: #2563eb; color: white;
                                  border-color: #3b82f6; }
            QPushButton#primary:hover { background: #3b82f6; }
            QPushButton#retry { color: #fde68a; border-color: #92400e; }
            QGroupBox { background: #111827; color: #cbd5e1;
                        border: 1px solid #263449; border-radius: 8px;
                        margin-top: 12px; padding: 14px 10px 10px 10px;
                        font-weight: 700; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px;
                               padding: 0 6px; }
            QLabel#statusName { color: #8291a8; font-size: 12px; }
            QLabel#statusValue { color: #f1f5f9; font-family: monospace; }
            QLabel#notice { background: #10251d; color: #86efac;
                            border: 1px solid #166534; border-radius: 6px;
                            padding: 8px 10px; }
        """)

        root = QWidget()
        root.setObjectName('root')
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel('Object Tracker')
        title.setObjectName('title')
        subtitle = QLabel(
            'Open-vocabulary detection · instance masks · relative 6DoF pose')
        subtitle.setObjectName('subtitle')
        layout.addWidget(title)
        layout.addWidget(subtitle)

        query_row = QHBoxLayout()
        target_label = QLabel('TARGET')
        target_label.setObjectName('muted')
        query_row.addWidget(target_label)
        self.query_edit = QLineEdit(initial_query)
        self.query_edit.setPlaceholderText('e.g. the red mug')
        self.query_edit.returnPressed.connect(self._detect)
        query_row.addWidget(self.query_edit, 1)
        detect = QPushButton('Detect')
        detect.setObjectName('primary')
        detect.clicked.connect(self._detect)
        query_row.addWidget(detect)
        retry = QPushButton('Retry')
        retry.setObjectName('retry')
        retry.clicked.connect(self._retry)
        query_row.addWidget(retry)
        layout.addLayout(query_row)

        self.image_label = QLabel('Waiting for /object/grounding_image …')
        self.image_label.setObjectName('cameraView')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 360)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet('background: #202124; color: #dddddd;')
        layout.addWidget(self.image_label, 1)

        status_box = QGroupBox('Pipeline status')
        status = QGridLayout(status_box)
        status.setHorizontalSpacing(18)
        status.setVerticalSpacing(7)
        status.setColumnStretch(1, 1)
        self.detection_value = QLabel('—')
        self.confidence_value = QLabel('—')
        self.tracking_value = QLabel('—')
        self.mask_value = QLabel('—')
        self.pose_value = QLabel('—')
        self.cloud_rate_value = QLabel('—')
        self.failure_value = QLabel('—')
        self.failure_value.setWordWrap(True)
        status_rows = [
                ('Detection', self.detection_value),
                ('Tracking', self.tracking_value),
                ('Mask', self.mask_value),
                ('Relative orientation from initialization', self.pose_value),
                ('Cloud updates', self.cloud_rate_value),
                ('Failure reason', self.failure_value),
        ]
        for row, (name, value) in enumerate(status_rows):
            name_label = QLabel(name.upper())
            name_label.setObjectName('statusName')
            value.setObjectName('statusValue')
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            status.addWidget(name_label, row, 0)
            status.addWidget(value, row, 1)
        pose_note = QLabel(
            'Orientation is measured in table_frame relative to the first '
            'valid object pose—not semantic object axes. X red · Y green · '
            'Z blue; symmetric objects may have ambiguous rotation.')
        pose_note.setObjectName('muted')
        pose_note.setWordWrap(True)
        status.addWidget(pose_note, len(status_rows), 0, 1, 2)
        layout.addWidget(status_box)

        controls = QHBoxLayout()
        reset = QPushButton('Reset tracker')
        reset.clicked.connect(node.reset)
        controls.addWidget(reset)
        show_cloud = QPushButton('Show unified cloud')
        show_cloud.clicked.connect(self.cloud_window.show_cloud)
        controls.addWidget(show_cloud)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.notice = QLabel('Ready.')
        self.notice.setObjectName('notice')
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)
        self.setCentralWidget(root)

    def _detect(self):
        self.node.detect(self.query_edit.text())

    def _retry(self):
        self.node.retry(self.query_edit.text())

    def clear_session_status(self):
        for value in (
                self.detection_value, self.confidence_value,
                self.tracking_value, self.mask_value,
                self.pose_value,
                self.cloud_rate_value,
                self.failure_value):
            value.setText('—')

    def set_notice(self, text, error=False):
        self.notice.setText(text)
        self.notice.setStyleSheet(
            'background: #2b151b; color: #fda4af; border: 1px solid #9f1239; '
            'border-radius: 6px; padding: 8px 10px;' if error else
            'background: #10251d; color: #86efac; border: 1px solid #166534; '
            'border-radius: 6px; padding: 8px 10px;')

    def set_grounding(self, msg):
        if msg.success:
            self.detection_value.setText(msg.object_description or msg.query)
            self.confidence_value.setText(f'{msg.confidence:.3f}')
            self.set_notice('Detection completed.')
        else:
            self.detection_value.setText('No detection')
            self.confidence_value.setText('—')
            self.set_notice(f'Detection failed: {msg.failure_reason}', error=True)

    def set_image(self, bgr):
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(image.copy())
        self._fit_image()

    def _fit_image(self):
        if self._pixmap is not None:
            self.image_label.setPixmap(self._pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_image()


def main(args=None):
    rclpy.init(args=args)
    app = QApplication(sys.argv)
    node = OperatorGuiNode()
    timer = QTimer()
    timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0.0))
    timer.start(20)
    node.window.show()
    try:
        exit_code = app.exec_()
    finally:
        timer.stop()
        node.destroy_node()
        rclpy.try_shutdown()
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
