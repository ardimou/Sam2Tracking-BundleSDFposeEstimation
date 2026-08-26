"""Synchronize Xtion RGB and registered depth, publishing depth in metres."""
import copy

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from object_tracker_common import topics


class RGBDAdapterNode(Node):

    def __init__(self):
        super().__init__('rgbd_adapter_node')

        self.declare_parameter('rgb_topic', topics.XTION_RGB_TOPIC)
        self.declare_parameter('depth_topic', topics.XTION_DEPTH_TOPIC)
        self.declare_parameter('info_topic', topics.XTION_INFO_TOPIC)
        self.declare_parameter('sync_slop_sec', 0.015)
        self.declare_parameter('max_rgb_depth_delta_sec', 0.015)
        self.declare_parameter('max_valid_depth_m', 6.0)
        self.declare_parameter('min_alignment_valid_ratio', 0.05)
        self.declare_parameter('depth_unit_scale', 0.0)

        self.bridge = CvBridge()
        self._max_valid_depth = self.get_parameter('max_valid_depth_m').value
        self._min_align_ratio = self.get_parameter('min_alignment_valid_ratio').value
        self._depth_unit_scale = float(self.get_parameter('depth_unit_scale').value)
        self._max_rgb_depth_delta = float(
            self.get_parameter('max_rgb_depth_delta_sec').value)
        self._latest_info = None

        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        info_topic = self.get_parameter('info_topic').value
        slop = self.get_parameter('sync_slop_sec').value

        self._rgb_sub = message_filters.Subscriber(self, Image, rgb_topic)
        self._depth_sub = message_filters.Subscriber(self, Image, depth_topic)

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub],
            queue_size=5,
            slop=slop,
        )
        self._sync.registerCallback(self._on_synced)
        self._info_sub = self.create_subscription(
            CameraInfo, info_topic, self._on_camera_info,
            topics.latest_only_qos(depth=1))

        out_qos = topics.reliable_qos(depth=1)
        self._rgb_pub = self.create_publisher(Image, topics.ADAPTER_RGB_TOPIC, out_qos)
        self._depth_pub = self.create_publisher(Image, topics.ADAPTER_DEPTH_TOPIC, out_qos)
        self._info_pub = self.create_publisher(CameraInfo, topics.ADAPTER_INFO_TOPIC, out_qos)

        self.latest_frame = None  # dict: timestamp, rgb, depth, intrinsics, frame_id

        self._dropped_alignment = 0
        self._dropped_timing = 0
        self._received = 0

        self.get_logger().info(
            f'rgbd_adapter_node up. rgb={rgb_topic} depth={depth_topic} info={info_topic}'
        )

    @staticmethod
    def _stamp_ns(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _on_camera_info(self, msg: CameraInfo):
        self._latest_info = msg

    def _on_synced(self, rgb_msg: Image, depth_msg: Image):
        self._received += 1

        delta_sec = abs(
            self._stamp_ns(rgb_msg.header.stamp) -
            self._stamp_ns(depth_msg.header.stamp)) * 1e-9
        delta_ms = delta_sec * 1000.0
        if delta_sec > self._max_rgb_depth_delta:
            self._dropped_timing += 1
            self.get_logger().warn(
                f'RGB-depth source delta {delta_ms:.1f} ms exceeds '
                f'{self._max_rgb_depth_delta * 1000.0:.1f} ms; dropping pair '
                f'({self._dropped_timing} timing drops).',
                throttle_duration_sec=1.0)
            return
        self.get_logger().info(
            f'RGB-depth source delta: {delta_ms:.1f} ms',
            throttle_duration_sec=2.0)

        info_msg = self._latest_info
        if info_msg is None:
            self.get_logger().warn(
                'Waiting for camera_info before publishing RGB-D frames.',
                throttle_duration_sec=2.0)
            return

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        depth_m = self._to_metres(depth_raw, depth_msg.encoding)

        alignment_ok, alignment_reason = self._validate_alignment(rgb, depth_m)
        if not alignment_ok:
            self._dropped_alignment += 1
            if self._dropped_alignment % 30 == 1:
                self.get_logger().warn(
                    f'RGB/depth alignment check failed '
                    f'({alignment_reason}; {self._dropped_alignment} drops so far) '
                    f'- dropping frame.'
                )
            return

        stamp = rgb_msg.header.stamp
        frame_id = rgb_msg.header.frame_id or topics.CAMERA_OPTICAL_FRAME

        depth_out = self.bridge.cv2_to_imgmsg(depth_m, encoding='32FC1')
        depth_out.header.stamp = stamp
        depth_out.header.frame_id = frame_id

        rgb_out = copy.deepcopy(rgb_msg)
        rgb_out.header.frame_id = frame_id

        info_out = copy.deepcopy(info_msg)
        info_out.header.stamp = stamp
        info_out.header.frame_id = frame_id

        self._rgb_pub.publish(rgb_out)
        self._depth_pub.publish(depth_out)
        self._info_pub.publish(info_out)

        self.latest_frame = {
            'timestamp': stamp,
            'rgb': rgb,
            'depth': depth_m,
            'intrinsics': np.array(info_msg.k, dtype=np.float64).reshape(3, 3),
            'camera_frame': frame_id,
            'rgb_depth_delta_ms': delta_ms,
        }

    def _to_metres(self, depth_raw: np.ndarray, encoding: str) -> np.ndarray:
        if self._depth_unit_scale > 0.0:
            scale = self._depth_unit_scale
        elif encoding.upper() in ('16UC1', 'MONO16') or depth_raw.dtype == np.uint16:
            scale = 0.001
        elif encoding.upper() == '32FC1' or depth_raw.dtype in (np.float32, np.float64):
            scale = 1.0
        else:
            self.get_logger().warn(
                f'Unknown depth encoding {encoding!r}; assuming metres.')
            scale = 1.0
        depth_m = depth_raw.astype(np.float32) * scale
        sensor_valid = depth_m[np.isfinite(depth_m) & (depth_m > 0.0)]
        if sensor_valid.size:
            self._last_depth_stats = (
                f'encoding={encoding}, sensor_valid={sensor_valid.size / depth_m.size:.3f}, '
                f'min={float(np.min(sensor_valid)):.2f}m, '
                f'median={float(np.median(sensor_valid)):.2f}m, '
                f'max={float(np.max(sensor_valid)):.2f}m')
        else:
            self._last_depth_stats = f'encoding={encoding}, sensor_valid=0.000'
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[depth_m > self._max_valid_depth] = 0.0
        depth_m[depth_m < 0] = 0.0
        return depth_m

    def _validate_alignment(self, rgb: np.ndarray, depth_m: np.ndarray):
        """Cheap alignment/sanity check, not a full calibration check.

        Confirms resolutions match (registered depth should share the RGB
        frame's resolution) and that a plausible fraction of pixels have
        valid depth. This catches driver misconfiguration (e.g. depth
        registration turned off) rather than subtle miscalibration.
        """
        if rgb.shape[0] != depth_m.shape[0] or rgb.shape[1] != depth_m.shape[1]:
            return False, (f'resolution mismatch: RGB={rgb.shape[1]}x{rgb.shape[0]}, '
                           f'depth={depth_m.shape[1]}x{depth_m.shape[0]}')
        valid_ratio = float(np.count_nonzero(depth_m > 0)) / depth_m.size
        if valid_ratio < self._min_align_ratio:
            return False, (f'valid depth ratio={valid_ratio:.3f}, '
                           f'minimum={self._min_align_ratio:.3f}; '
                           f'{getattr(self, "_last_depth_stats", "no depth statistics")}')
        return True, 'ok'


def main(args=None):
    rclpy.init(args=args)
    node = RGBDAdapterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
