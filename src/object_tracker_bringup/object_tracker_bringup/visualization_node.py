"""Publish the mask overlay and RViz object-frame markers."""
from collections import OrderedDict
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, String, Float32
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge

from object_tracker_msgs.msg import GroundingResult as GroundingResultMsg
from object_tracker_msgs.msg import (
    InstanceMask, InstanceTablePose, InstanceTrackerStatus, MaskState)
from object_tracker_msgs.msg import SupervisorStatus as SupervisorStatusMsg
from object_tracker_msgs.msg import TrackerStatus as TrackerStatusMsg

from object_tracker_common import topics

_STATE_NAMES = ['IDLE', 'GROUNDING', 'INITIALIZING', 'TRACKING', 'UNCERTAIN', 'LOST',
                'REINITIALIZING']
_STATE_COLORS = {
    'TRACKING': (0, 200, 0), 'UNCERTAIN': (0, 165, 255), 'LOST': (0, 0, 255),
}


class VisualizationNode(Node):

    def __init__(self):
        super().__init__('visualization_node')

        self.bridge = CvBridge()
        self._grounding_boxes = []
        self._show_grounding_box = False
        self._grounding_box_received_at = 0.0
        self._grounding_displayed_for_session = False
        self._last_query = ''
        self._last_state = 0
        self._last_confidence = 0.0
        self._last_latency_ms = 0.0
        self._instance_masks = {}
        self._rgb_grays = OrderedDict()
        self._mask_flow = cv2.DISOpticalFlow_create(
            cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self._last_mask_stamp = -1
        self._camera_matrix = None
        self._instance_poses = {}
        self._camera_instance_poses = {}
        self._reference_wireframes = {}
        self._screen_pose_axes = {}
        self._session_start_ns = 0
        self.declare_parameter('pose_axis_length_m', 0.06)
        self.declare_parameter('use_table_frame', True)
        self.declare_parameter('table_frame_id', topics.TABLE_FRAME)
        self.declare_parameter('semantic_object_frame_suffix', '_viz')
        self.declare_parameter('pose_axis_length_px', 44)
        self.declare_parameter('show_pose_axes_in_image', False)
        self.declare_parameter('grounding_box_display_sec', 2.0)
        self.declare_parameter('show_global_tracking_overlay', False)
        self.declare_parameter('show_camera_optical_frame', False)
        self.declare_parameter('pose_axis_shaft_diameter_m', 0.012)
        self.declare_parameter('pose_axis_head_diameter_m', 0.024)
        self.declare_parameter('pose_axis_head_length_m', 0.030)
        self.declare_parameter('pose_axis_label_height_m', 0.018)
        self.declare_parameter('reference_wireframe_line_width_m', 0.004)
        self.declare_parameter('reference_wireframe_min_extent_m', 0.025)
        self.declare_parameter('reference_wireframe_max_extent_m', 0.30)
        self.declare_parameter('show_reference_wireframe', False)
        self.declare_parameter('camera_axis_length_m', 0.15)
        self.declare_parameter('pose_display_timeout_sec', 0.0)

        self.create_subscription(Image, topics.ADAPTER_RGB_TOPIC, self._on_rgb,
                                  topics.latest_only_qos())
        self.create_subscription(
            InstanceMask, topics.INSTANCE_MASK_TOPIC,
            self._on_instance_mask, topics.reliable_qos(depth=10))
        self.create_subscription(GroundingResultMsg, topics.GROUNDING_RESULT_TOPIC,
                                  self._on_grounding, topics.reliable_qos())
        self.create_subscription(
            String, topics.QUERY_TOPIC, self._on_query, topics.reliable_qos())
        self.create_subscription(SupervisorStatusMsg, topics.TRACKING_STATE_TOPIC,
                                  self._on_supervisor_state, topics.reliable_qos())
        self.create_subscription(TrackerStatusMsg, topics.TRACKER_STATUS_TOPIC,
                                  self._on_diag, topics.latest_only_qos())
        self.create_subscription(
            CameraInfo, topics.ADAPTER_INFO_TOPIC, self._on_camera_info,
            topics.latest_only_qos())
        self.create_subscription(
            InstanceTrackerStatus, topics.INSTANCE_POSE_TOPIC,
            self._on_camera_instance_pose, topics.reliable_qos(depth=32))
        self.create_subscription(
            InstanceTrackerStatus, topics.INSTANCE_TRACKER_STATUS_TOPIC,
            self._on_camera_instance_pose, topics.reliable_qos(depth=32))
        if bool(self.get_parameter('use_table_frame').value):
            self.create_subscription(
                InstanceTablePose, topics.INSTANCE_TABLE_POSE_TOPIC,
                self._on_table_instance_pose, topics.reliable_qos(depth=32))
        self._img_pub = self.create_publisher(Image, topics.VIZ_IMAGE_TOPIC,
                                               topics.latest_only_qos())
        self._marker_pub = self.create_publisher(
            MarkerArray, topics.VIZ_MARKERS_TOPIC,
            topics.reliable_qos(depth=10))

        self.get_logger().info('visualization_node up.')

    def _on_instance_mask(self, msg: InstanceMask):
        object_id = int(msg.object_id)
        stamp = (int(msg.header.stamp.sec) * 1_000_000_000 +
                 int(msg.header.stamp.nanosec))
        if stamp < self._session_start_ns:
            return
        self._last_mask_stamp = max(self._last_mask_stamp, stamp)
        if (object_id <= 0 or int(msg.state) in
                (MaskState.LOST, MaskState.SWITCHED) or
                not msg.mask.height or not msg.mask.width):
            self._instance_masks.pop(object_id, None)
            self._instance_poses.pop(object_id, None)
            self._camera_instance_poses.pop(object_id, None)
            self._reference_wireframes.pop(object_id, None)
            self._screen_pose_axes.pop(object_id, None)
            return
        mask = self.bridge.imgmsg_to_cv2(
            msg.mask, desired_encoding='mono8').copy()
        source_gray = self._rgb_grays.get(stamp)
        if source_gray is None and self._rgb_grays:
            nearest = min(
                self._rgb_grays, key=lambda candidate: abs(candidate - stamp))
            source_gray = self._rgb_grays[nearest]
        self._instance_masks[object_id] = {
            'mask': mask,
            'label': msg.label,
            'confidence': float(msg.confidence),
            'state': int(msg.state),
            'gray': source_gray,
            'stamp': stamp,
        }

    def _on_grounding(self, msg: GroundingResultMsg):
        self._last_query = msg.query
        if (msg.success and len(msg.bounding_box) == 4 and
                not self._grounding_displayed_for_session):
            self._grounding_boxes = [(
                int(msg.object_id),
                msg.object_description or msg.query,
                list(msg.bounding_box), float(msg.confidence))]
            self._grounding_boxes.extend(
                (int(candidate.object_id),
                 candidate.description or msg.query,
                 list(candidate.bounding_box), float(candidate.confidence))
                for candidate in msg.alternative_candidates
                if len(candidate.bounding_box) == 4)
            self._show_grounding_box = True
            self._grounding_box_received_at = time.monotonic()
            self._grounding_displayed_for_session = True

    def _on_query(self, msg: String):
        self._session_start_ns = self.get_clock().now().nanoseconds
        self._last_query = msg.data
        self._last_state = 1  # GROUNDING
        self._last_confidence = 0.0
        self._last_latency_ms = 0.0
        self._grounding_boxes.clear()
        self._show_grounding_box = False
        self._grounding_box_received_at = 0.0
        self._grounding_displayed_for_session = False
        self._instance_masks.clear()
        self._instance_poses.clear()
        self._camera_instance_poses.clear()
        self._reference_wireframes.clear()
        self._screen_pose_axes.clear()
        self._rgb_grays.clear()
        self._last_mask_stamp = -1
        marker_array = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)
        self._marker_pub.publish(marker_array)

    def _on_supervisor_state(self, msg: SupervisorStatusMsg):
        self._last_state = msg.state
        self._last_confidence = msg.confidence

    def _on_diag(self, msg: TrackerStatusMsg):
        self._last_latency_ms = msg.processing_time_ms

    def _on_camera_info(self, msg: CameraInfo):
        self._camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)

    def _on_camera_instance_pose(self, msg: InstanceTrackerStatus):
        tracker = msg.status
        pose_stamp = (int(tracker.header.stamp.sec) * 1_000_000_000 +
                      int(tracker.header.stamp.nanosec))
        if pose_stamp < self._session_start_ns:
            return
        object_id = int(msg.object_id)
        if object_id <= 0:
            return
        if not tracker.tracking_success:
            if tracker.reconstruction_status.startswith('tracking_lost:'):
                self._instance_poses.pop(object_id, None)
                self._camera_instance_poses.pop(object_id, None)
                self._screen_pose_axes.pop(object_id, None)
            return
        camera_value = (
            tracker.t_camera_object, msg.label, float(tracker.tracking_quality),
            time.monotonic())
        self._camera_instance_poses[object_id] = camera_value
        if not bool(self.get_parameter('use_table_frame').value):
            self._instance_poses[object_id] = camera_value
        if (bool(self.get_parameter('show_reference_wireframe').value) and
                object_id not in self._reference_wireframes):
            vertices = self._make_reference_wireframe(
                object_id, tracker.t_camera_object)
            if vertices is not None:
                self._reference_wireframes[object_id] = vertices

    def _on_table_instance_pose(self, msg: InstanceTablePose):
        stamp = (int(msg.header.stamp.sec) * 1_000_000_000 +
                 int(msg.header.stamp.nanosec))
        if stamp < self._session_start_ns or int(msg.object_id) <= 0:
            return
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose
        self._instance_poses[int(msg.object_id)] = (
            pose, msg.label, 1.0, time.monotonic())

    def _on_rgb(self, msg: Image):
        raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        stamp = (int(msg.header.stamp.sec) * 1_000_000_000 +
                 int(msg.header.stamp.nanosec))
        self._rgb_grays[stamp] = gray
        self._rgb_grays.move_to_end(stamp)
        while len(self._rgb_grays) > 90:
            self._rgb_grays.popitem(last=False)
        flows = {}
        for context in self._instance_masks.values():
            previous_gray = context['gray']
            previous_stamp = context['stamp']
            if previous_gray is None or previous_stamp >= stamp:
                continue
            if previous_stamp not in flows:
                flows[previous_stamp] = self._mask_flow.calc(
                    previous_gray, gray, None)
            context['mask'] = self._forward_warp_mask(
                context['mask'], flows[previous_stamp])
            context['gray'] = gray
            context['stamp'] = stamp
        img = raw.copy()
        state_name = _STATE_NAMES[self._last_state] if self._last_state < len(_STATE_NAMES) else '?'
        color = _STATE_COLORS.get(state_name, (200, 200, 200))

        overlay = img.copy()
        for object_id, context in sorted(self._instance_masks.items()):
            mask = context['mask']
            label = context['label']
            confidence = context['confidence']
            mask_state = context['state']
            if mask.shape != img.shape[:2]:
                continue
            selected = mask > 0
            instance_color = self._instance_color(object_id)
            overlay[selected] = instance_color
            contours, _ = cv2.findContours(
                selected.astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, instance_color, 2)
            rows, cols = np.nonzero(selected)
            if len(rows):
                mask_names = (
                    'INITIALIZING', 'TRACKING', 'UNCERTAIN', 'LOST',
                    'SWITCHED')
                name = (mask_names[mask_state]
                        if 0 <= mask_state < len(mask_names)
                        else str(mask_state))
                compact_status = f'ID {object_id} | {name}'
                text_origin = (int(cols.min()), max(18, int(rows.min()) - 6))
                cv2.putText(
                    img, compact_status, text_origin,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 4,
                    cv2.LINE_AA)
                cv2.putText(
                    img, compact_status, text_origin,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, instance_color, 2,
                    cv2.LINE_AA)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

        box_duration = float(
            self.get_parameter('grounding_box_display_sec').value)
        show_boxes = (
            self._show_grounding_box and self._grounding_boxes and
            (box_duration <= 0.0 or
             time.monotonic() - self._grounding_box_received_at <=
             box_duration))
        if show_boxes:
            for object_id, label, box, confidence in self._grounding_boxes:
                x0, y0, x1, y1 = [int(round(v)) for v in box]
                box_color = self._instance_color(object_id)
                cv2.rectangle(img, (x0, y0), (x1, y1), box_color, 2)
                cv2.putText(
                    img, f'ID {object_id} {label}: {confidence:.2f}',
                    (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, box_color, 2, cv2.LINE_AA)
        elif self._show_grounding_box:
            self._show_grounding_box = False

        if bool(self.get_parameter('show_pose_axes_in_image').value):
            self._draw_pose_axes(img)

        if bool(self.get_parameter('show_global_tracking_overlay').value):
            summary = f'{state_name} | pose {self._last_latency_ms:.0f} ms'
            cv2.putText(img, summary, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, summary, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 2, cv2.LINE_AA)

        out = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        out.header = msg.header
        self._img_pub.publish(out)

        self._publish_markers(msg.header, state_name, color)

    @staticmethod
    def _instance_color(object_id):
        seed = max(1, int(object_id))
        return ((37 * seed) % 200 + 55,
                (97 * seed) % 200 + 55,
                (157 * seed) % 200 + 55)

    @staticmethod
    def _forward_warp_mask(mask, flow):
        rows, cols = np.nonzero(mask > 0)
        warped = np.zeros(mask.shape, dtype=np.uint8)
        if not len(rows):
            return warped
        destinations = np.rint(
            np.column_stack((cols, rows)) + flow[rows, cols]).astype(np.int32)
        valid = ((destinations[:, 0] >= 0) &
                 (destinations[:, 0] < mask.shape[1]) &
                 (destinations[:, 1] >= 0) &
                 (destinations[:, 1] < mask.shape[0]))
        destinations = destinations[valid]
        warped[destinations[:, 1], destinations[:, 0]] = 255
        return cv2.morphologyEx(
            warped, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    @staticmethod
    def _rotation_matrix(quaternion):
        x, y, z, w = (float(quaternion.x), float(quaternion.y),
                      float(quaternion.z), float(quaternion.w))
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        if norm <= 1e-12:
            return None
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

    def _fresh_poses(self):
        timeout = float(self.get_parameter('pose_display_timeout_sec').value)
        now = time.monotonic()
        return [
            (object_id, value)
            for object_id, value in self._instance_poses.items()
            if (object_id in self._instance_masks and
                (timeout <= 0.0 or now - value[3] <= timeout))
        ]

    def _fresh_camera_poses(self):
        timeout = float(self.get_parameter('pose_display_timeout_sec').value)
        now = time.monotonic()
        return [
            (object_id, value)
            for object_id, value in self._camera_instance_poses.items()
            if (object_id in self._instance_masks and
                (timeout <= 0.0 or now - value[3] <= timeout))
        ]

    def _pose_points(self, pose):
        R = self._rotation_matrix(pose.pose.orientation)
        if R is None:
            return None
        origin = np.array([
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z],
            dtype=np.float64)
        length = float(self.get_parameter('pose_axis_length_m').value)
        return np.vstack((origin, origin + R[:, 0] * length,
                          origin + R[:, 1] * length,
                          origin + R[:, 2] * length))

    def _make_reference_wireframe(self, object_id, pose):
        """Create a fixed, mask-sized box in the initial object frame.

        This box is a visual reference only. Its dimensions are estimated once
        from the initial mask's projected width/height and initial pose depth;
        no geometry or orientation is recomputed on later frames.
        """
        if self._camera_matrix is None or object_id not in self._instance_masks:
            return None
        mask = np.asarray(self._instance_masks[object_id]['mask']) > 0
        rows, cols = np.nonzero(mask)
        if not len(rows):
            return None
        fx, fy = float(self._camera_matrix[0, 0]), float(self._camera_matrix[1, 1])
        depth = float(pose.pose.position.z)
        if fx <= 0.0 or fy <= 0.0 or depth <= 0.02:
            return None
        minimum = float(self.get_parameter('reference_wireframe_min_extent_m').value)
        maximum = float(self.get_parameter('reference_wireframe_max_extent_m').value)
        width = np.clip((float(cols.max() - cols.min() + 1) * depth) / fx,
                        minimum, maximum)
        height = np.clip((float(rows.max() - rows.min() + 1) * depth) / fy,
                         minimum, maximum)
        thickness = np.clip(0.55 * min(width, height), minimum, maximum)
        hx, hy, hz = 0.5 * width, 0.5 * thickness, 0.5 * height
        return np.asarray([
            [-hx, -hy, -hz], [hx, -hy, -hz],
            [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy, hz], [hx, -hy, hz],
            [hx, hy, hz], [-hx, hy, hz],
        ], dtype=np.float64)

    def _draw_pose_axes(self, image):
        if self._camera_matrix is None:
            return
        K = self._camera_matrix
        colors = ((0, 0, 255), (0, 255, 0))
        for object_id, (pose, label, confidence, _) in self._fresh_camera_poses():
            points = self._pose_points(pose)
            if points is None or np.any(points[:, 2] <= 0.02):
                continue
            pixels = np.column_stack((
                K[0, 0] * points[:, 0] / points[:, 2] + K[0, 2],
                K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2],
            ))
            origin_xy = pixels[0]
            x_projected = pixels[1] - origin_xy
            y_projected = pixels[2] - origin_xy
            x_norm = float(np.linalg.norm(x_projected))
            y_norm = float(np.linalg.norm(y_projected))
            if x_norm > 1e-6:
                x_direction = x_projected / x_norm
                y_direction = np.array(
                    [-x_direction[1], x_direction[0]], dtype=np.float64)
                if np.dot(y_direction, y_projected) < 0.0:
                    y_direction *= -1.0
            elif y_norm > 1e-6:
                y_direction = y_projected / y_norm
                x_direction = np.array(
                    [y_direction[1], -y_direction[0]], dtype=np.float64)
            else:
                continue
            previous_directions = self._screen_pose_axes.get(object_id)
            if previous_directions is not None:
                if np.dot(x_direction, previous_directions[0]) < 0.0:
                    x_direction *= -1.0
                if np.dot(y_direction, previous_directions[1]) < 0.0:
                    y_direction *= -1.0
            self._screen_pose_axes[object_id] = (
                x_direction.copy(), y_direction.copy())
            pixel_length = max(
                8, int(self.get_parameter('pose_axis_length_px').value))
            endpoints = (
                origin_xy + x_direction * pixel_length,
                origin_xy + y_direction * pixel_length,
            )
            origin = tuple(np.rint(origin_xy).astype(int))
            for endpoint, axis_color in zip(endpoints, colors):
                endpoint = tuple(np.rint(endpoint).astype(int))
                cv2.arrowedLine(image, origin, endpoint, axis_color, 4,
                                cv2.LINE_AA, tipLength=0.25)

    def _publish_markers(self, header, state_name, color_bgr):
        arr = MarkerArray()
        axis_colors = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                       (0.0, 0.0, 1.0))
        if bool(self.get_parameter('show_camera_optical_frame').value):
            length = float(
                self.get_parameter('camera_axis_length_m').value)
            origin = Point(x=0.0, y=0.0, z=0.0)
            endpoints = (
                Point(x=length, y=0.0, z=0.0),
                Point(x=0.0, y=length, z=0.0),
                Point(x=0.0, y=0.0, z=length),
            )
            for axis, (endpoint, axis_color) in enumerate(zip(
                    endpoints, axis_colors)):
                arrow = Marker()
                arrow.header = header
                arrow.ns = 'camera_optical_frame_axes'
                arrow.id = axis
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.pose.orientation.w = 1.0
                arrow.points = [origin, endpoint]
                arrow.scale.x = 0.008
                arrow.scale.y = 0.016
                arrow.scale.z = 0.024
                arrow.color.r, arrow.color.g, arrow.color.b = axis_color
                arrow.color.a = 1.0
                arrow.lifetime.sec = 1
                arr.markers.append(arrow)
            camera_text = Marker()
            camera_text.header = header
            camera_text.ns = 'camera_optical_frame_text'
            camera_text.id = 0
            camera_text.type = Marker.TEXT_VIEW_FACING
            camera_text.action = Marker.ADD
            camera_text.pose.position.z = length + 0.04
            camera_text.pose.orientation.w = 1.0
            camera_text.scale.z = 0.025
            camera_text.color.r = camera_text.color.g = 1.0
            camera_text.color.b = 0.0
            camera_text.color.a = 1.0
            camera_text.text = header.frame_id or topics.CAMERA_OPTICAL_FRAME
            camera_text.lifetime.sec = 1
            arr.markers.append(camera_text)
        for object_id, (pose, label, confidence, _) in self._fresh_poses():
            semantic_frame = bool(self.get_parameter('use_table_frame').value)
            if semantic_frame:
                marker_header = Header()
                marker_header.stamp = pose.header.stamp
                marker_header.frame_id = (
                    f'tracked_object_{object_id}' + str(self.get_parameter(
                        'semantic_object_frame_suffix').value))
                length = float(self.get_parameter('pose_axis_length_m').value)
                points = np.asarray([
                    [0.0, 0.0, 0.0], [length, 0.0, 0.0],
                    [0.0, length, 0.0], [0.0, 0.0, length],
                ], dtype=np.float64)
            else:
                points = self._pose_points(pose)
                if points is None:
                    continue
                marker_header = pose.header
            for axis, axis_color in enumerate(axis_colors):
                arrow = Marker()
                arrow.header = marker_header
                arrow.ns = f'object_{object_id}_pose_axes'
                arrow.id = object_id * 10 + axis
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.pose.orientation.w = 1.0
                arrow.points = [Point(x=float(points[0, 0]),
                                      y=float(points[0, 1]),
                                      z=float(points[0, 2])),
                                Point(x=float(points[axis + 1, 0]),
                                      y=float(points[axis + 1, 1]),
                                      z=float(points[axis + 1, 2]))]
                arrow.scale.x = float(self.get_parameter(
                    'pose_axis_shaft_diameter_m').value)
                arrow.scale.y = float(self.get_parameter(
                    'pose_axis_head_diameter_m').value)
                arrow.scale.z = float(self.get_parameter(
                    'pose_axis_head_length_m').value)
                arrow.color.r, arrow.color.g, arrow.color.b = axis_color
                arrow.color.a = 1.0
                arrow.lifetime.sec = 1
                arr.markers.append(arrow)

                axis_label = Marker()
                axis_label.header = marker_header
                axis_label.ns = f'object_{object_id}_axis_labels'
                axis_label.id = object_id * 10 + axis
                axis_label.type = Marker.TEXT_VIEW_FACING
                axis_label.action = Marker.ADD
                axis_label.pose.position = Point(
                    x=float(points[axis + 1, 0]),
                    y=float(points[axis + 1, 1]),
                    z=float(points[axis + 1, 2]))
                axis_label.pose.orientation.w = 1.0
                axis_label.scale.z = float(self.get_parameter(
                    'pose_axis_label_height_m').value)
                axis_label.color.r, axis_label.color.g, axis_label.color.b = axis_color
                axis_label.color.a = 1.0
                axis_label.text = ('+X', '+Y', '+Z')[axis]
                axis_label.lifetime.sec = 1
                arr.markers.append(axis_label)

            local_vertices = (self._reference_wireframes.get(object_id)
                              if bool(self.get_parameter(
                                  'show_reference_wireframe').value) else None)
            if local_vertices is not None:
                if semantic_frame:
                    vertices = local_vertices
                else:
                    rotation = self._rotation_matrix(pose.pose.orientation)
                    if rotation is None:
                        continue
                    translation = np.asarray([
                        pose.pose.position.x, pose.pose.position.y,
                        pose.pose.position.z], dtype=np.float64)
                    vertices = local_vertices @ rotation.T + translation
                edge_indices = (
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                )
                wireframe = Marker()
                wireframe.header = marker_header
                wireframe.ns = f'object_{object_id}_reference_wireframe'
                wireframe.id = object_id
                wireframe.type = Marker.LINE_LIST
                wireframe.action = Marker.ADD
                wireframe.pose.orientation.w = 1.0
                wireframe.scale.x = float(self.get_parameter(
                    'reference_wireframe_line_width_m').value)
                wireframe.color.r = 0.35
                wireframe.color.g = 0.85
                wireframe.color.b = 1.0
                wireframe.color.a = 0.9
                for start, end in edge_indices:
                    wireframe.points.extend((
                        Point(x=float(vertices[start, 0]),
                              y=float(vertices[start, 1]),
                              z=float(vertices[start, 2])),
                        Point(x=float(vertices[end, 0]),
                              y=float(vertices[end, 1]),
                              z=float(vertices[end, 2])),
                    ))
                wireframe.lifetime.sec = 1
                arr.markers.append(wireframe)

        self._marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
