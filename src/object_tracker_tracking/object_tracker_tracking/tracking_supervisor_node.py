"""Validate BundleTrack poses against SAM2 state and motion limits."""
from enum import IntEnum

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
import tf2_ros
from geometry_msgs.msg import TransformStamped

from object_tracker_msgs.msg import TrackerStatus as TrackerStatusMsg
from object_tracker_msgs.msg import MaskState as MaskStateMsg
from object_tracker_msgs.msg import SupervisorStatus as SupervisorStatusMsg
from object_tracker_msgs.msg import (InstanceMask, InstanceTrackerStatus,
                                     InstanceSupervisorStatus)
from object_tracker_common import topics


class State(IntEnum):
    IDLE = 0
    GROUNDING = 1
    INITIALIZING = 2
    TRACKING = 3
    UNCERTAIN = 4
    LOST = 5
    REINITIALIZING = 6


class TrackingSupervisorNode(Node):

    def __init__(self):
        super().__init__('tracking_supervisor_node')

        self.declare_parameter('max_position_jump_m', 0.15)   # per processed frame
        self.declare_parameter('max_latency_ms', 250.0)
        self.declare_parameter('uncertain_frames_before_lost', 10)
        self.declare_parameter('instance_subscription_depth', 32)
        self.declare_parameter('use_table_frame', False)

        self._max_jump = self.get_parameter('max_position_jump_m').value
        self._max_latency = self.get_parameter('max_latency_ms').value
        self._uncertain_limit = self.get_parameter('uncertain_frames_before_lost').value
        self._instance_depth = int(
            self.get_parameter('instance_subscription_depth').value)

        self._state = State.IDLE
        self._instance_states = {}
        self._session_start_ns = 0

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.create_subscription(InstanceMask, topics.INSTANCE_MASK_TOPIC,
                                 self._on_instance_mask, topics.reliable_qos())
        self.create_subscription(InstanceTrackerStatus, topics.INSTANCE_TRACKER_STATUS_TOPIC,
                                 self._on_instance_tracker,
                                 topics.reliable_qos(depth=self._instance_depth))
        self.create_subscription(
            String, topics.QUERY_TOPIC, self._on_query, topics.reliable_qos())

        self._pose_pub = self.create_publisher(
            TrackerStatusMsg, topics.POSE_TOPIC, topics.latest_only_qos())
        self._state_pub = self.create_publisher(
            SupervisorStatusMsg, topics.TRACKING_STATE_TOPIC, topics.reliable_qos())
        self._conf_pub = self.create_publisher(
            Float32, topics.CONFIDENCE_TOPIC, topics.latest_only_qos())
        self._reason_pub = self.create_publisher(
            String, topics.FAILURE_REASON_TOPIC, topics.latest_only_qos())
        self._instance_pose_pub = self.create_publisher(
            InstanceTrackerStatus, topics.INSTANCE_POSE_TOPIC,
            topics.reliable_qos(depth=self._instance_depth))
        self._instance_state_pub = self.create_publisher(
            InstanceSupervisorStatus, topics.INSTANCE_TRACKING_STATE_TOPIC,
            topics.reliable_qos())

        self.get_logger().info('tracking_supervisor_node up.')

    def _on_instance_mask(self, msg: InstanceMask):
        if self._stamp_ns(msg.header) < self._session_start_ns:
            return
        context = self._instance_states.setdefault(int(msg.object_id), {
            'label': msg.label, 'mask_confidence': 0.0,
            'mask_state': MaskStateMsg.LOST, 'last_position': None,
            'uncertain_streak': 0, 'state': State.IDLE})
        context['label'] = msg.label
        context['mask_confidence'] = float(msg.confidence)
        context['mask_state'] = int(msg.state)
        if msg.state == MaskStateMsg.INITIALIZING:
            context['state'] = State.INITIALIZING
        elif msg.state == MaskStateMsg.UNCERTAIN:
            context['state'] = State.UNCERTAIN
        elif msg.state in (MaskStateMsg.LOST, MaskStateMsg.SWITCHED):
            context['state'] = State.LOST
        else:
            return
        reason = f'mask_state={int(msg.state)}'
        self._publish_instance_state(
            int(msg.object_id), context, msg.header, reason)
        if int(msg.object_id) == min(self._instance_states):
            self._publish_primary_state(context, msg.header, reason)

    def _on_instance_tracker(self, msg: InstanceTrackerStatus):
        if self._stamp_ns(msg.status.header) < self._session_start_ns:
            return
        object_id = int(msg.object_id)
        context = self._instance_states.setdefault(object_id, {
            'label': msg.label, 'mask_confidence': 0.0,
            'mask_state': MaskStateMsg.LOST, 'last_position': None,
            'uncertain_streak': 0, 'state': State.IDLE})
        context['label'] = msg.label
        tracker = msg.status
        reason = self._instance_failure_reason(context, tracker)
        if reason is None:
            context['uncertain_streak'] = 0
            context['state'] = State.TRACKING
            self._instance_pose_pub.publish(msg)
            self._broadcast_instance_trusted(object_id, tracker)
        else:
            context['uncertain_streak'] += 1
            context['state'] = (State.LOST if context['uncertain_streak'] >= self._uncertain_limit
                                else State.UNCERTAIN)
        self._publish_instance_state(object_id, context, tracker.header, reason or '')
        if object_id == min(self._instance_states):
            self._publish_primary(context, tracker, reason or '')

    @staticmethod
    def _stamp_ns(header):
        return (int(header.stamp.sec) * 1_000_000_000 +
                int(header.stamp.nanosec))

    def _on_query(self, msg):
        self._session_start_ns = self.get_clock().now().nanoseconds
        self._instance_states.clear()
        self._state = State.GROUNDING

    def _instance_failure_reason(self, context, tracker):
        if context['mask_state'] != MaskStateMsg.TRACKING:
            return f'mask_state={context["mask_state"]}'
        if not tracker.tracking_success:
            detail = tracker.reconstruction_status or 'unknown'
            return f'pose_track_failed:{detail}'
        if tracker.processing_time_ms > self._max_latency:
            return f'latency_{tracker.processing_time_ms:.0f}ms'
        pos = tracker.t_camera_object.pose.position
        current = np.array([pos.x, pos.y, pos.z])
        previous = context['last_position']
        if previous is not None:
            jump = float(np.linalg.norm(current - previous))
            if jump > self._max_jump:
                return f'pose_jump_{jump:.3f}m'
        context['last_position'] = current
        return None

    def _publish_instance_state(self, object_id, context, header, reason):
        status = SupervisorStatusMsg()
        status.header = header
        status.state = int(context['state'])
        status.confidence = float(context['mask_confidence'])
        status.failure_reason = reason
        output = InstanceSupervisorStatus()
        output.object_id = object_id
        output.label = context['label']
        output.status = status
        self._instance_state_pub.publish(output)

    def _publish_primary(self, context, tracker, reason):
        self._state = context['state']
        if self._state == State.TRACKING:
            self._pose_pub.publish(tracker)
            if not bool(self.get_parameter('use_table_frame').value):
                transform = TransformStamped()
                transform.header = tracker.header
                transform.child_frame_id = topics.TRACKED_OBJECT_FRAME
                pose = tracker.t_camera_object.pose
                transform.transform.translation.x = pose.position.x
                transform.transform.translation.y = pose.position.y
                transform.transform.translation.z = pose.position.z
                transform.transform.rotation = pose.orientation
                self.tf_broadcaster.sendTransform(transform)
        self._publish_primary_state(context, tracker.header, reason)

    def _publish_primary_state(self, context, header, reason):
        self._state = context['state']
        status = SupervisorStatusMsg()
        status.header = header
        status.state = int(self._state)
        status.confidence = float(context['mask_confidence'])
        status.failure_reason = reason
        self._state_pub.publish(status)
        self._conf_pub.publish(Float32(data=float(context['mask_confidence'])))
        self._reason_pub.publish(String(data=reason))

    def _broadcast_instance_trusted(self, object_id, tracker):
        if bool(self.get_parameter('use_table_frame').value):
            return
        transform = TransformStamped()
        transform.header = tracker.header
        transform.child_frame_id = f'tracked_object_{object_id}'
        pose = tracker.t_camera_object.pose
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = pose.position.z
        transform.transform.rotation = pose.orientation
        self.tf_broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = TrackingSupervisorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
