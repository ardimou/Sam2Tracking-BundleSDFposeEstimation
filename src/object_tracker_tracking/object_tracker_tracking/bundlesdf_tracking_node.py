"""Track each SAM2 instance with BundleTrack and publish pose and cloud data."""
import json
import os
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import String
from std_srvs.srv import Trigger
import tf2_ros
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge

from object_tracker_msgs.msg import TrackerStatus as TrackerStatusMsg
from object_tracker_msgs.msg import MaskState as MaskStateMsg
from object_tracker_msgs.msg import InstanceMask, InstanceTrackerStatus
from object_tracker_msgs.srv import ExportMesh, SaveModel

from object_tracker_common import topics
from object_tracker_tracking.preprocessing import (
    RGBDPreprocessor, masked_depth_statistics)
from object_tracker_tracking.bundlesdf_wrapper import BundleSDFWrapper


class BundleSDFTrackingNode(Node):

    def __init__(self):
        super().__init__('bundlesdf_tracking_node')

        self.declare_parameter('backend', 'inprocess')
        self.declare_parameter('bundlesdf_repo_path', '')
        self.declare_parameter('bundlesdf_session_root', '/tmp/bundlesdf_sessions')
        self.declare_parameter('neural_reconstruction_enabled', False)
        self.declare_parameter('start_nerf_keyframes', 5)
        self.declare_parameter('nerf_sync_max_delay_keyframes', 10)
        self.declare_parameter('nerf_training_steps', 100)
        self.declare_parameter('nerf_mesh_resolution_m', 0.01)
        self.declare_parameter('bundlesdf_debug_level', 0)
        self.declare_parameter('depth_percentile', 100.0)
        self.declare_parameter('depth_zfar_m', 3.0)
        self.declare_parameter('depth_erode_diff_m', 0.01)
        self.declare_parameter('depth_erode_ratio', 1.0)
        self.declare_parameter('depth_edge_normal_threshold_deg', 3.0)
        self.declare_parameter('min_match_with_ref', 5)
        self.declare_parameter('min_match_after_ransac', 5)
        self.declare_parameter('ransac_inlier_dist_m', 0.01)
        self.declare_parameter('ransac_max_iterations', 800)
        self.declare_parameter('ransac_max_translation_neighbor_m', 0.10)
        self.declare_parameter('ransac_max_rotation_neighbor_deg', 45.0)
        self.declare_parameter('feature_correspondence_resize', 320)
        self.declare_parameter('bundle_outer_iterations', 4)
        self.declare_parameter('bundle_inner_iterations', 3)
        self.declare_parameter('bundle_window_size', 4)
        self.declare_parameter('bundle_max_ba_frames', 4)
        self.declare_parameter('max_consecutive_tracking_failures', 5)
        self.declare_parameter('minimum_masked_depth_points', 100)
        self.declare_parameter('minimum_masked_depth_ratio', 0.01)
        self.declare_parameter('maximum_depth_mask_delta_sec', 0.05)
        self.declare_parameter('target_width', 0)      # 0 = keep native resolution
        self.declare_parameter('target_height', 0)
        self.declare_parameter('max_instances', 10)
        self.declare_parameter('frame_cache_size', 60)
        self.declare_parameter('reconstruction_publish_rate_hz', 0.5)
        self.declare_parameter('reconstruction_max_points', 50000)
        self.declare_parameter('unified_cloud_publish_rate_hz', 2.0)
        self.declare_parameter('maximum_unified_cloud_points', 75000)
        self.declare_parameter('nearby_cloud_pixel_stride', 4)
        self.declare_parameter('object_cloud_pixel_stride', 1)
        self.declare_parameter('nearby_cloud_gray_level', 90)
        self.declare_parameter('nearby_cloud_color_r', 70)
        self.declare_parameter('nearby_cloud_color_g', 110)
        self.declare_parameter('nearby_cloud_color_b', 180)
        self.declare_parameter('object_cloud_color_mode', 'instance')
        self.declare_parameter('pose_translation_source', 'sam_xtion')
        self.declare_parameter('translation_mask_minimum_pixels', 25)
        self.declare_parameter('translation_depth_mask_erosion_px', 2)
        self.declare_parameter('translation_minimum_depth_pixels', 10)
        self.declare_parameter('translation_minimum_depth_m', 0.1)
        self.declare_parameter('translation_maximum_depth_m', 3.0)
        self.declare_parameter('instance_status_qos_depth', 32)
        self.declare_parameter('instance_mask_subscription_depth', 10)
        translation_source = str(
            self.get_parameter('pose_translation_source').value).lower()
        if translation_source not in ('bundlesdf', 'sam_xtion'):
            raise ValueError(
                'pose_translation_source must be bundlesdf or sam_xtion')
        feature_size = int(
            self.get_parameter('feature_correspondence_resize').value)
        if feature_size < 64 or feature_size % 8:
            raise ValueError(
                'feature_correspondence_resize must be >= 64 and divisible by 8')
        for parameter_name in (
                'ransac_max_iterations', 'bundle_outer_iterations',
                'bundle_inner_iterations'):
            if int(self.get_parameter(parameter_name).value) < 1:
                raise ValueError(f'{parameter_name} must be positive')
        if float(self.get_parameter(
                'maximum_depth_mask_delta_sec').value) < 0.0:
            raise ValueError('maximum_depth_mask_delta_sec must be non-negative')

        topic_defaults = {
            'rgb_topic': topics.ADAPTER_RGB_TOPIC,
            'depth_topic': topics.ADAPTER_DEPTH_TOPIC,
            'camera_info_topic': topics.ADAPTER_INFO_TOPIC,
            'mask_topic': topics.MASK_TOPIC,
            'mask_state_topic': topics.MASK_STATE_TOPIC,
            'instance_mask_topic': topics.INSTANCE_MASK_TOPIC,
            'pose_raw_topic': topics.POSE_RAW_TOPIC,
            'tracker_status_topic': topics.TRACKER_STATUS_TOPIC,
            'tracker_diagnostics_topic': topics.TRACKER_DIAGNOSTICS_TOPIC,
            'instance_tracker_status_topic': topics.INSTANCE_TRACKER_STATUS_TOPIC,
            'model_cloud_topic': topics.MODEL_CLOUD_TOPIC,
            'model_mesh_topic': topics.MODEL_MESH_TOPIC,
            'reconstruction_status_topic': topics.RECONSTRUCTION_STATUS_TOPIC,
            'unified_cloud_topic': topics.UNIFIED_CLOUD_TOPIC,
            'reset_service': topics.RESET_SERVICE,
            'export_mesh_service': topics.EXPORT_MESH_SERVICE,
            'save_model_service': topics.SAVE_MODEL_SERVICE,
        }
        for name, default in topic_defaults.items():
            self.declare_parameter(name, default)
        self._endpoint = {
            name: str(self.get_parameter(name).value) for name in topic_defaults
        }

        tw = self.get_parameter('target_width').value
        th = self.get_parameter('target_height').value
        target_size = (tw, th) if tw and th else None

        self.pre = RGBDPreprocessor(target_size=target_size)
        self._wrapper_config = {
            'backend': self.get_parameter('backend').value,
            'bundlesdf_repo_path': self.get_parameter('bundlesdf_repo_path').value,
            'bundlesdf_session_root': self.get_parameter('bundlesdf_session_root').value,
            'neural_reconstruction_enabled': self.get_parameter(
                'neural_reconstruction_enabled').value,
            'start_nerf_keyframes': self.get_parameter('start_nerf_keyframes').value,
            'nerf_sync_max_delay_keyframes': self.get_parameter(
                'nerf_sync_max_delay_keyframes').value,
            'nerf_training_steps': self.get_parameter('nerf_training_steps').value,
            'nerf_mesh_resolution_m': self.get_parameter(
                'nerf_mesh_resolution_m').value,
            'bundlesdf_debug_level': self.get_parameter('bundlesdf_debug_level').value,
            'depth_percentile': self.get_parameter('depth_percentile').value,
            'depth_zfar_m': self.get_parameter('depth_zfar_m').value,
            'depth_erode_diff_m': self.get_parameter('depth_erode_diff_m').value,
            'depth_erode_ratio': self.get_parameter('depth_erode_ratio').value,
            'depth_edge_normal_threshold_deg': self.get_parameter(
                'depth_edge_normal_threshold_deg').value,
            'min_match_with_ref': self.get_parameter('min_match_with_ref').value,
            'min_match_after_ransac': self.get_parameter('min_match_after_ransac').value,
            'ransac_inlier_dist_m': self.get_parameter('ransac_inlier_dist_m').value,
            'ransac_max_iterations': self.get_parameter(
                'ransac_max_iterations').value,
            'ransac_max_translation_neighbor_m': self.get_parameter(
                'ransac_max_translation_neighbor_m').value,
            'ransac_max_rotation_neighbor_deg': self.get_parameter(
                'ransac_max_rotation_neighbor_deg').value,
            'feature_correspondence_resize': self.get_parameter(
                'feature_correspondence_resize').value,
            'bundle_outer_iterations': self.get_parameter(
                'bundle_outer_iterations').value,
            'bundle_inner_iterations': self.get_parameter(
                'bundle_inner_iterations').value,
            'bundle_window_size': self.get_parameter(
                'bundle_window_size').value,
            'bundle_max_ba_frames': self.get_parameter(
                'bundle_max_ba_frames').value,
        }
        self.wrapper = None
        self._instances = {}
        self._instance_cursor = 0
        self._rgb_frames = OrderedDict()
        self._depth_frames = OrderedDict()
        self._info_frames = OrderedDict()
        self._frame_cache_size = int(self.get_parameter('frame_cache_size').value)
        self._sensor_callback_group = MutuallyExclusiveCallbackGroup()
        self._control_callback_group = MutuallyExclusiveCallbackGroup()
        self._output_callback_group = MutuallyExclusiveCallbackGroup()
        self._sensor_lock = threading.RLock()
        self._bundle_lock = threading.RLock()
        self._reset_requested = threading.Event()

        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self._latest_rgb = None
        self._latest_depth = None
        self._latest_info = None
        self._latest_mask = None
        self._mask_state = MaskStateMsg.LOST
        self._camera_frame = topics.CAMERA_OPTICAL_FRAME

        self._track_active = False

        self.create_subscription(
            Image, self._endpoint['rgb_topic'], self._on_rgb,
            topics.latest_only_qos(),
            callback_group=self._sensor_callback_group)
        self.create_subscription(
            Image, self._endpoint['depth_topic'], self._on_depth,
            topics.latest_only_qos(),
            callback_group=self._sensor_callback_group)
        self.create_subscription(
            CameraInfo, self._endpoint['camera_info_topic'], self._on_info,
            topics.latest_only_qos(),
            callback_group=self._sensor_callback_group)
        self.create_subscription(
            Image, self._endpoint['mask_topic'], self._on_mask, topics.latest_only_qos())
        self.create_subscription(
            MaskStateMsg, self._endpoint['mask_state_topic'], self._on_mask_state,
            topics.reliable_qos())
        self.create_subscription(InstanceMask, self._endpoint['instance_mask_topic'],
                                 self._on_instance_mask, topics.reliable_qos(
                                     depth=int(self.get_parameter(
                                         'instance_mask_subscription_depth').value)))

        self._pose_pub = self.create_publisher(
            TrackerStatusMsg, self._endpoint['pose_raw_topic'], topics.latest_only_qos())
        self._status_pub = self.create_publisher(
            TrackerStatusMsg, self._endpoint['tracker_status_topic'], topics.latest_only_qos())
        self._diag_pub = self.create_publisher(
            String, self._endpoint['tracker_diagnostics_topic'], topics.latest_only_qos())
        self._instance_status_pub = self.create_publisher(
            InstanceTrackerStatus, self._endpoint['instance_tracker_status_topic'],
            topics.reliable_qos(depth=int(self.get_parameter(
                'instance_status_qos_depth').value)))
        self._cloud_pub = self.create_publisher(
            PointCloud2, self._endpoint['model_cloud_topic'], topics.reliable_qos(depth=1))
        self._mesh_pub = self.create_publisher(
            String, self._endpoint['model_mesh_topic'], topics.reliable_qos())
        self._reconstruction_status_pub = self.create_publisher(
            String, self._endpoint['reconstruction_status_topic'], topics.reliable_qos())
        self._unified_cloud_pub = self.create_publisher(
            PointCloud2, self._endpoint['unified_cloud_topic'],
            topics.reliable_qos(depth=1))

        self.create_service(
            Trigger, self._endpoint['reset_service'], self._on_reset,
            callback_group=self._control_callback_group)
        self.create_service(
            ExportMesh, self._endpoint['export_mesh_service'], self._on_export_mesh)
        self.create_service(
            SaveModel, self._endpoint['save_model_service'], self._on_save_model)

        self.declare_parameter('process_rate_hz', 15.0)
        rate = self.get_parameter('process_rate_hz').value
        self.create_timer(1.0 / rate, self._process_latest)
        reconstruction_rate = float(
            self.get_parameter('reconstruction_publish_rate_hz').value)
        if (bool(self.get_parameter('neural_reconstruction_enabled').value) and
                reconstruction_rate > 0.0):
            self.create_timer(1.0 / reconstruction_rate,
                              self._publish_reconstruction_preview)
        unified_rate = float(
            self.get_parameter('unified_cloud_publish_rate_hz').value)
        if unified_rate > 0.0:
            self.create_timer(
                1.0 / unified_rate, self._publish_unified_cloud,
                callback_group=self._output_callback_group)

        self.get_logger().info(
            'bundlesdf_tracking_node up: '
            f"rgb={self._endpoint['rgb_topic']} depth={self._endpoint['depth_topic']} "
            f"instances={self._endpoint['instance_mask_topic']} "
            f"poses={self._endpoint['instance_tracker_status_topic']} "
            f"orientation=bundlesdf translation={translation_source}")
        self.get_logger().info(
            'BundleSDF online reconstruction profile: '
            f"neural_enabled="
            f"{self._wrapper_config['neural_reconstruction_enabled']} "
            f"start_keyframes={self._wrapper_config['start_nerf_keyframes']} "
            f"training_steps={self._wrapper_config['nerf_training_steps']} "
            f"sync_max_delay_keyframes="
            f"{self._wrapper_config['nerf_sync_max_delay_keyframes']} "
            f"mesh_resolution_m={self._wrapper_config['nerf_mesh_resolution_m']}")
        self.get_logger().info(
            'BundleTrack live pose profile (effective per object): '
            f"LoFTR={self._wrapper_config['feature_correspondence_resize']}x"
            f"{self._wrapper_config['feature_correspondence_resize']} "
            f"bundle_iterations={self._wrapper_config['bundle_outer_iterations']}x"
            f"{self._wrapper_config['bundle_inner_iterations']} "
            f"ransac_max_iterations={self._wrapper_config['ransac_max_iterations']} "
            f"window={self._wrapper_config['bundle_window_size']} "
            f"max_BA_frames={self._wrapper_config['bundle_max_ba_frames']}")

    def _on_rgb(self, msg: Image):
        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self._sensor_lock:
            self._latest_rgb = rgb
            self._camera_frame = msg.header.frame_id or self._camera_frame
            self._latest_stamp = msg.header.stamp
            self._rgb_frames[self._stamp_key(msg.header.stamp)] = (
                rgb, msg.header.stamp)
            self._trim_cache(self._rgb_frames)

    def _on_depth(self, msg: Image):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        with self._sensor_lock:
            self._latest_depth = depth
            self._depth_frames[self._stamp_key(msg.header.stamp)] = depth
            self._trim_cache(self._depth_frames)

    @staticmethod
    def _stamp_key(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _trim_cache(self, cache):
        while len(cache) > self._frame_cache_size:
            cache.popitem(last=False)

    def _on_info(self, msg: CameraInfo):
        intrinsics = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        with self._sensor_lock:
            self._latest_info = intrinsics
            self._info_frames[self._stamp_key(msg.header.stamp)] = intrinsics
            self._trim_cache(self._info_frames)

    def _on_mask(self, msg: Image):
        self._latest_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

    def _on_mask_state(self, msg: MaskStateMsg):
        self._mask_state = msg.state

    def _on_instance_mask(self, msg: InstanceMask):
        object_id = int(msg.object_id)
        if object_id <= 0:
            self.get_logger().warn('Ignoring instance mask with invalid object_id=0.')
            return
        if object_id not in self._instances:
            if len(self._instances) >= int(self.get_parameter('max_instances').value):
                self.get_logger().warn('Ignoring instance: max_instances reached.')
                return
            self._instances[object_id] = {
                'label': msg.label,
                'wrapper': None,
                'active': False,
                'mask': None,
                'mask_state': msg.state,
                'mask_stamp': msg.header.stamp,
                'last_frame_key': None,
                'needs_reset': False,
                'consecutive_failures': 0,
                'tracking_attempts': 0,
                'tracking_successes': 0,
                'tracking_rejections': 0,
                'last_processing_ms': 0.0,
                'mean_processing_ms': 0.0,
                'last_depth_mask_delta_ms': 0.0,
                'mask_frames': OrderedDict(),
                'model_points': np.empty((0, 3), dtype=np.float32),
                'model_colors': np.empty((0, 3), dtype=np.uint8),
                'last_pose': None,
            }
        context = self._instances[object_id]
        context['label'] = msg.label
        context['mask_state'] = msg.state
        context['mask_stamp'] = msg.header.stamp
        if msg.mask.height and msg.mask.width:
            context['mask'] = self.bridge.imgmsg_to_cv2(msg.mask, desired_encoding='mono8')
            frame_key = self._stamp_key(msg.header.stamp)
            context['mask_frames'][frame_key] = context['mask'].copy()
            self._trim_cache(context['mask_frames'])

    def _on_reset(self, request, response):
        self._reset_requested.set()
        try:
            with self._bundle_lock:
                if self.wrapper is not None:
                    self.wrapper.reset()
                    self.wrapper = None
                self._track_active = False
                for context in self._instances.values():
                    if context['wrapper'] is not None:
                        context['wrapper'].reset()
                self._instances.clear()
                self._instance_cursor = 0
                response.success = True
                response.message = 'BundleSDF wrapper reset.'
        finally:
            self._reset_requested.clear()
        return response

    def _has_usable_masked_depth(self, inp, object_id):
        valid_points, mask_points, ratio = masked_depth_statistics(inp)
        enough = (
            valid_points >= int(self.get_parameter(
                'minimum_masked_depth_points').value) and
            ratio >= float(self.get_parameter(
                'minimum_masked_depth_ratio').value))
        if not enough:
            self.get_logger().warn(
                f'Waiting for usable raw depth for object {object_id}: '
                f'valid={valid_points}/{mask_points} ({ratio:.3f}), '
                f"required={self.get_parameter('minimum_masked_depth_points').value} "
                f"and ratio>={self.get_parameter('minimum_masked_depth_ratio').value:.3f}. "
                'Frame was not sent to BundleSDF.',
                throttle_duration_sec=5.0)
        return enough

    def _process_latest(self):
        if self._reset_requested.is_set():
            return
        if not self._bundle_lock.acquire(blocking=False):
            return
        try:
            if self._reset_requested.is_set():
                return
            self._process_latest_locked()
        finally:
            self._bundle_lock.release()

    def _process_latest_locked(self):
        if self._instances:
            self._process_instances()
            return
        if (self._latest_rgb is None or self._latest_depth is None or
                self._latest_info is None or self._latest_mask is None):
            return
        if self._mask_state in (MaskStateMsg.LOST, MaskStateMsg.SWITCHED):
            return  # nothing trustworthy to initialize/track from

        t0 = time.time()
        inp = self.pre.process(
            self._latest_rgb, self._latest_depth, self._latest_mask,
            self._latest_info, self._latest_stamp)

        if not self._has_usable_masked_depth(inp, 0):
            self._publish_diagnostics(0.0, 'insufficient_masked_depth')
            return

        if self.wrapper is None:
            self.wrapper = BundleSDFWrapper({**self._wrapper_config, 'object_id': 0})

        if not self._track_active:
            ok = self.wrapper.initialize(inp)
            self._track_active = ok
            if not ok:
                self._publish_diagnostics(0.0, 'initialize_failed')
                return
            T = self.wrapper.last_pose
            result_success = T is not None
            quality, recon_status = (1.0 if result_success else 0.0,
                                     ('neural_accumulating' if bool(
                                         self._wrapper_config[
                                             'neural_reconstruction_enabled'])
                                      else 'external_online_fusion')
                                     if result_success else 'initialize_failed')
        else:
            result = self.wrapper.track(inp)
            result_success, T = result.tracking_success, result.T_camera_object
            quality, recon_status = result.tracking_quality, result.reconstruction_status
            if not result_success:
                self._track_active = False

        if result_success and T is not None:
            T = self._compose_output_pose(T, inp)
            if T is None:
                result_success = False
                recon_status = 'sam_xtion_translation_unavailable'
                self.get_logger().warn(
                    'BundleSDF rotation is available, but the SAM/Xtion '
                    'translation could not be composed; suppressing pose.',
                    throttle_duration_sec=2.0)
            else:
                recon_status = (
                    f'{recon_status};orientation=bundlesdf;'
                    f'translation={self.get_parameter("pose_translation_source").value}')
        dt_ms = (time.time() - t0) * 1000.0
        self._publish_status(T, result_success, quality, dt_ms, recon_status)
        if result_success and T is not None:
            self._broadcast_tf(T, self._latest_stamp)
        self._publish_diagnostics(dt_ms, 'ok' if result_success else 'track_failed')

    def _process_instances(self):
        if (self._latest_rgb is None or self._latest_info is None or
                self._latest_depth is None):
            return
        object_ids = sorted(self._instances)
        if not object_ids:
            return
        self._instance_cursor %= len(object_ids)
        object_id = object_ids[self._instance_cursor]
        self._instance_cursor = (self._instance_cursor + 1) % len(object_ids)
        for object_id, context in [(object_id, self._instances[object_id])]:
            if context['mask'] is None or context['mask_state'] in (
                    MaskStateMsg.LOST, MaskStateMsg.SWITCHED):
                continue
            frame_key = self._stamp_key(context['mask_stamp'])
            depth_key, depth = self._nearest_depth_frame(frame_key)
            mask = context['mask']
            if depth_key is not None:
                context['last_depth_mask_delta_ms'] = (
                    abs(depth_key - frame_key) / 1_000_000.0)
            if frame_key == context['last_frame_key']:
                continue
            if context['needs_reset']:
                status = self._make_status(
                    None, False, 0.0, 0.0, 'reset_required',
                    context['mask_stamp'])
                self._publish_instance_status(object_id, context, status)
                context['last_frame_key'] = frame_key
                continue
            with self._sensor_lock:
                rgb_entry = self._rgb_frames.get(frame_key)
            if rgb_entry is None or depth is None:
                if depth is None:
                    self.get_logger().warn(
                        f'No registered depth near object {object_id} mask '
                        f'(tolerance={self.get_parameter("maximum_depth_mask_delta_sec").value:.3f}s).',
                        throttle_duration_sec=2.0)
                continue
            rgb, stamp = rgb_entry
            t0 = time.time()
            with self._sensor_lock:
                intrinsics = self._info_frames.get(
                    frame_key, self._latest_info)
            inp = self.pre.process(rgb, depth, mask, intrinsics, stamp)
            if not self._has_usable_masked_depth(inp, object_id):
                context['last_frame_key'] = frame_key
                status = self._make_status(
                    None, False, 0.0, 0.0, 'insufficient_masked_depth', stamp)
                self._publish_instance_status(object_id, context, status)
                continue
            wrapper = context['wrapper']
            if wrapper is None:
                try:
                    wrapper = BundleSDFWrapper({
                        **self._wrapper_config, 'object_id': object_id})
                    context['wrapper'] = wrapper
                except Exception as exc:
                    context['last_frame_key'] = frame_key
                    context['needs_reset'] = True
                    self.get_logger().error(
                        f'Could not start BundleSDF for object {object_id}: {exc}')
                    status = self._make_status(
                        None, False, 0.0, 0.0, 'backend_start_failed', stamp)
                    self._publish_instance_status(object_id, context, status)
                    continue
            context['tracking_attempts'] += 1
            try:
                if not context['active']:
                    success = wrapper.initialize(inp)
                    context['active'] = success
                    T = wrapper.last_pose if success else None
                    quality = 1.0 if success else 0.0
                    reconstruction = (
                        ('neural_accumulating' if bool(
                            self._wrapper_config[
                                'neural_reconstruction_enabled'])
                         else 'external_online_fusion')
                        if success else 'initialize_failed')
                    context['needs_reset'] = not success
                    context['consecutive_failures'] = 0 if success else 1
                else:
                    result = wrapper.track(inp)
                    success = result.tracking_success
                    T = result.T_camera_object
                    quality = result.tracking_quality
                    reconstruction = result.reconstruction_status
                    if success:
                        context['consecutive_failures'] = 0
                    else:
                        context['consecutive_failures'] += 1
                        failure_limit = int(self.get_parameter(
                            'max_consecutive_tracking_failures').value)
                        if context['consecutive_failures'] >= failure_limit:
                            context['active'] = False
                            context['needs_reset'] = True
            except Exception as exc:
                success, T, quality = False, None, 0.0
                reconstruction = 'backend_exception_reset_required'
                context['active'] = False
                context['needs_reset'] = True
                context['consecutive_failures'] += 1
                self.get_logger().error(
                    f'BundleSDF object {object_id} failed: {exc}')
            context['last_frame_key'] = frame_key
            if success and T is not None:
                T = self._compose_output_pose(T, inp)
                if T is None:
                    success = False
                    reconstruction = 'sam_xtion_translation_unavailable'
                    self.get_logger().warn(
                        f'BundleSDF rotation is available for object '
                        f'{object_id}, but the SAM/Xtion translation could '
                        'not be composed; suppressing pose.',
                        throttle_duration_sec=2.0)
                else:
                    reconstruction = (
                        f'{reconstruction};orientation=bundlesdf;'
                        f'translation={self.get_parameter("pose_translation_source").value}')
            if success and T is not None:
                context['last_pose'] = np.asarray(T, dtype=np.float64).copy()
                self._update_instance_model(context, inp, T)
            elapsed = (time.time() - t0) * 1000.0
            context['last_processing_ms'] = elapsed
            attempts = int(context['tracking_attempts'])
            previous_mean = float(context['mean_processing_ms'])
            context['mean_processing_ms'] = (
                previous_mean + (elapsed - previous_mean) / max(attempts, 1))
            if success:
                context['tracking_successes'] += 1
            else:
                context['tracking_rejections'] += 1
            status = self._make_status(
                T, success, quality, elapsed, reconstruction, stamp)
            self._publish_instance_status(object_id, context, status)
            self._broadcast_instance_tf(object_id, T, success, stamp)
        self._publish_instance_diagnostics()

    def _compose_output_pose(self, bundlesdf_pose, inp):
        """Combine BundleTrack rotation with SAM/Xtion translation."""
        pose = np.asarray(bundlesdf_pose, dtype=np.float64).copy()
        U, _, Vt = np.linalg.svd(pose[:3, :3])
        rotation = U @ Vt
        if np.linalg.det(rotation) < 0.0:
            U[:, -1] *= -1.0
            rotation = U @ Vt
        pose[:3, :3] = rotation
        if str(self.get_parameter(
                'pose_translation_source').value).lower() == 'bundlesdf':
            return pose

        mask = np.asarray(inp.mask) > 0
        depth = np.asarray(inp.depth, dtype=np.float64)
        if mask.shape != depth.shape:
            return None
        rows, cols = np.nonzero(mask)
        if len(rows) < int(self.get_parameter(
                'translation_mask_minimum_pixels').value):
            return None

        depth_mask = mask.astype(np.uint8)
        erosion = max(0, int(self.get_parameter(
            'translation_depth_mask_erosion_px').value))
        if erosion > 0:
            size = erosion * 2 + 1
            eroded = cv2.erode(
                depth_mask, np.ones((size, size), dtype=np.uint8))
            if np.count_nonzero(eroded) >= int(self.get_parameter(
                    'translation_minimum_depth_pixels').value):
                depth_mask = eroded
        samples = depth[depth_mask > 0]
        samples = samples[
            np.isfinite(samples) &
            (samples >= float(self.get_parameter(
                'translation_minimum_depth_m').value)) &
            (samples <= float(self.get_parameter(
                'translation_maximum_depth_m').value))]
        if len(samples) < int(self.get_parameter(
                'translation_minimum_depth_pixels').value):
            return None

        z = float(np.median(samples))
        mean_u = float(np.mean(cols))
        mean_v = float(np.mean(rows))
        K = np.asarray(inp.intrinsics, dtype=np.float64)
        pose[:3, 3] = (
            (mean_u - K[0, 2]) * z / K[0, 0],
            (mean_v - K[1, 2]) * z / K[1, 1],
            z,
        )
        return pose

    def _update_instance_model(self, context, inp, pose):
        """Store a lightweight latest Xtion surface in object coordinates."""
        valid = ((np.asarray(inp.mask) > 0) & np.isfinite(inp.depth) &
                 (inp.depth > 0.0))
        stride = max(1, int(self.get_parameter(
            'object_cloud_pixel_stride').value))
        sampled = np.zeros(valid.shape, dtype=bool)
        sampled[::stride, ::stride] = True
        valid &= sampled
        rows, cols = np.nonzero(valid)
        if not len(rows):
            return
        z = inp.depth[rows, cols].astype(np.float64)
        K = inp.intrinsics
        camera_points = np.column_stack((
            (cols - K[0, 2]) * z / K[0, 0],
            (rows - K[1, 2]) * z / K[1, 1], z))
        inverse = np.linalg.inv(np.asarray(pose, dtype=np.float64))
        context['model_points'] = (
            camera_points @ inverse[:3, :3].T + inverse[:3, 3]
        ).astype(np.float32)
        context['model_colors'] = np.ascontiguousarray(
            inp.rgb[rows, cols][:, ::-1], dtype=np.uint8)

    def _publish_unified_cloud(self):
        """Publish gray live Xtion context plus every learned object model."""
        with self._sensor_lock:
            if (self._latest_depth is None or self._latest_rgb is None or
                    self._latest_info is None or
                    not hasattr(self, '_latest_stamp')):
                return
            depth = np.asarray(self._latest_depth, dtype=np.float32)
            K = np.asarray(self._latest_info, dtype=np.float64)
            stamp = self._latest_stamp
            camera_frame = self._camera_frame
        try:
            contexts = list(self._instances.items())
        except RuntimeError:
            return
        excluded = np.zeros(depth.shape, dtype=bool)
        for _, context in contexts:
            mask = context.get('mask')
            if mask is not None and mask.shape == excluded.shape:
                excluded |= mask > 0
        stride = max(1, int(
            self.get_parameter('nearby_cloud_pixel_stride').value))
        sampled = np.zeros(depth.shape, dtype=bool)
        sampled[::stride, ::stride] = True
        valid = (sampled & ~excluded & np.isfinite(depth) &
                 (depth > 0.1) & (depth < 3.0))
        rows, cols = np.nonzero(valid)
        z = depth[rows, cols].astype(np.float64)
        scene_points = np.column_stack((
            (cols - K[0, 2]) * z / K[0, 0],
            (rows - K[1, 2]) * z / K[1, 1], z)).astype(np.float32)
        scene_color = np.array([
            np.clip(int(self.get_parameter('nearby_cloud_color_r').value), 0, 255),
            np.clip(int(self.get_parameter('nearby_cloud_color_g').value), 0, 255),
            np.clip(int(self.get_parameter('nearby_cloud_color_b').value), 0, 255),
        ], dtype=np.uint8)
        scene_colors = np.tile(scene_color, (len(scene_points), 1))

        object_points = []
        object_colors = []
        color_mode = str(self.get_parameter(
            'object_cloud_color_mode').value).lower()
        for object_id, context in contexts:
            pose = context.get('last_pose')
            model = context.get('model_points')
            if pose is None or model is None or not len(model):
                continue
            object_points.append((
                model @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32))
            if color_mode == 'source_rgb':
                object_colors.append(context['model_colors'])
            else:
                object_colors.append(np.tile(
                    self._instance_cloud_color(object_id),
                    (len(model), 1)))
        object_count = sum(len(part) for part in object_points)
        limit = int(self.get_parameter(
            'maximum_unified_cloud_points').value)
        if limit > 0 and object_count > limit:
            combined_points = np.concatenate(object_points, axis=0)
            combined_colors = np.concatenate(object_colors, axis=0)
            indices = np.linspace(
                0, len(combined_points) - 1, limit, dtype=np.int64)
            object_points = [combined_points[indices]]
            object_colors = [combined_colors[indices]]
            object_count = limit
            scene_points = scene_points[:0]
            scene_colors = scene_colors[:0]
        elif limit > 0 and len(scene_points) + object_count > limit:
            scene_budget = max(0, limit - object_count)
            if scene_budget < len(scene_points):
                indices = (np.linspace(
                    0, len(scene_points) - 1, scene_budget, dtype=np.int64)
                           if scene_budget else np.empty(0, dtype=np.int64))
                scene_points = scene_points[indices]
                scene_colors = scene_colors[indices]
        points = np.concatenate([scene_points] + object_points, axis=0)
        colors = np.concatenate([scene_colors] + object_colors, axis=0)
        self._unified_cloud_pub.publish(self._rgb_cloud_message(
            points, colors, camera_frame, stamp))

    @staticmethod
    def _instance_cloud_color(object_id):
        """Bright RGB palette, visually distinct from the blue scene cloud."""
        palette = np.asarray([
            [251, 146, 60],   # orange
            [74, 222, 128],   # green
            [244, 114, 182],  # pink
            [250, 204, 21],   # yellow
            [192, 132, 252],  # purple
            [34, 211, 238],   # cyan
        ], dtype=np.uint8)
        return palette[(max(1, int(object_id)) - 1) % len(palette)]

    @staticmethod
    def _rgb_cloud_message(points, colors, frame_id, stamp):
        packed = ((colors[:, 0].astype(np.uint32) << 16) |
                  (colors[:, 1].astype(np.uint32) << 8) |
                  colors[:, 2].astype(np.uint32))
        data = np.empty(len(points), dtype=[
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4')])
        if len(points):
            data['x'], data['y'], data['z'] = points.T
        data['rgb'] = packed
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * msg.width
        msg.is_dense = True
        msg.data = data.tobytes()
        return msg

    def _publish_instance_status(self, object_id, context, status):
        instance_status = InstanceTrackerStatus()
        instance_status.object_id = object_id
        instance_status.label = context['label']
        instance_status.status = status
        self._instance_status_pub.publish(instance_status)
        if self._instances and object_id == min(self._instances):
            self._pose_pub.publish(status)
            self._status_pub.publish(status)

    def _make_status(self, T, success, quality, dt_ms, recon_status, stamp=None):
        msg = TrackerStatusMsg()
        msg.header.stamp = stamp or self.get_clock().now().to_msg()
        msg.header.frame_id = self._camera_frame
        msg.tracking_success = bool(success)
        msg.tracking_quality = float(quality)
        msg.processing_time_ms = float(dt_ms)
        msg.reconstruction_status = recon_status
        if T is not None:
            msg.t_camera_object.header = msg.header
            msg.t_camera_object.pose = BundleSDFWrapper.to_ros_pose(T)
        return msg

    def _broadcast_instance_tf(self, object_id, T, success, stamp=None):
        if not success or T is None:
            return
        t = TransformStamped()
        t.header.stamp = stamp or self.get_clock().now().to_msg()
        t.header.frame_id = self._camera_frame
        t.child_frame_id = f'tracked_object_{object_id}_raw'
        pose = BundleSDFWrapper.to_ros_pose(T)
        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z
        t.transform.rotation = pose.orientation
        self.tf_broadcaster.sendTransform(t)

    def _publish_instance_diagnostics(self):
        states = {
            str(object_id): {
                'label': context['label'],
                'active': context['active'],
                'needs_reset': context['needs_reset'],
                'tracking_attempts': context['tracking_attempts'],
                'tracking_successes': context['tracking_successes'],
                'tracking_rejections': context['tracking_rejections'],
                'success_ratio': (
                    context['tracking_successes'] /
                    max(context['tracking_attempts'], 1)),
                'last_processing_ms': context['last_processing_ms'],
                'mean_processing_ms': context['mean_processing_ms'],
                'depth_mask_delta_ms': context['last_depth_mask_delta_ms'],
                'frames_processed': (context['wrapper'].num_frames_processed
                                     if context['wrapper'] is not None else 0),
            }
            for object_id, context in self._instances.items()
        }
        self._diag_pub.publish(String(data=json.dumps({'instances': states})))

    def _nearest_depth_frame(self, target_key):
        with self._sensor_lock:
            if not self._depth_frames:
                return None, None
            nearest_key = min(
                self._depth_frames, key=lambda key: abs(key - target_key))
            maximum_delta_ns = int(float(self.get_parameter(
                'maximum_depth_mask_delta_sec').value) * 1_000_000_000)
            if abs(nearest_key - target_key) > maximum_delta_ns:
                return None, None
            return nearest_key, self._depth_frames[nearest_key]

    def _publish_status(self, T, success, quality, dt_ms, recon_status):
        msg = self._make_status(
            T, success, quality, dt_ms, recon_status, self._latest_stamp)
        self._pose_pub.publish(msg)
        self._status_pub.publish(msg)

    def _broadcast_tf(self, T: np.ndarray, stamp=None):
        t = TransformStamped()
        t.header.stamp = stamp or self.get_clock().now().to_msg()
        t.header.frame_id = self._camera_frame
        t.child_frame_id = topics.TRACKED_OBJECT_FRAME + '_raw'
        pose = BundleSDFWrapper.to_ros_pose(T)
        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z
        t.transform.rotation = pose.orientation
        self.tf_broadcaster.sendTransform(t)

    def _publish_diagnostics(self, dt_ms: float, note: str):
        diag = {
            'processing_time_ms': dt_ms,
            'frames_processed': self.wrapper.num_frames_processed if self.wrapper else 0,
            'note': note,
        }
        self._diag_pub.publish(String(data=json.dumps(diag)))

    def _primary_wrapper(self):
        if self._instances:
            for object_id in sorted(self._instances):
                wrapper = self._instances[object_id].get('wrapper')
                if wrapper is not None:
                    return object_id, wrapper
        return 0, self.wrapper

    def _publish_reconstruction_preview(self):
        object_id, wrapper = self._primary_wrapper()
        if wrapper is None:
            return
        try:
            mesh = wrapper.get_reconstruction()
            if mesh is None:
                progress = wrapper.get_reconstruction_progress()
                progress['object_id'] = object_id
                self._reconstruction_status_pub.publish(
                    String(data=json.dumps(progress)))
                return
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            vertices = vertices[np.isfinite(vertices).all(axis=1)]
            limit = int(self.get_parameter('reconstruction_max_points').value)
            if limit > 0 and len(vertices) > limit:
                indices = np.linspace(0, len(vertices) - 1, limit, dtype=np.int64)
                vertices = vertices[indices]
            cloud = self._vertices_to_cloud(vertices, object_id)
            self._cloud_pub.publish(cloud)
            faces = len(getattr(mesh, 'faces', ()))
            status = {
                'state': 'available', 'object_id': object_id,
                'num_vertices': int(len(mesh.vertices)), 'num_faces': int(faces),
            }
            self._reconstruction_status_pub.publish(
                String(data=json.dumps(status)))
        except Exception as exc:
            self.get_logger().warn(
                f'Could not publish reconstruction preview: {exc}',
                throttle_duration_sec=5.0)

    def _vertices_to_cloud(self, vertices, object_id):
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = (f'tracked_object_{object_id}_raw'
                               if object_id else topics.TRACKED_OBJECT_FRAME + '_raw')
        msg.height = 1
        msg.width = int(len(vertices))
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = np.ascontiguousarray(vertices, dtype='<f4').tobytes()
        return msg

    def _on_export_mesh(self, request, response):
        object_id, wrapper = self._primary_wrapper()
        if not request.path.strip():
            response.message = 'An export path is required.'
            return response
        path = os.path.abspath(os.path.expanduser(request.path))
        if wrapper is None:
            response.message = 'No initialized BundleSDF instance.'
            return response
        if not path or path == '/':
            response.message = 'Refusing an empty or root export path.'
            return response
        try:
            parent = os.path.dirname(path)
            os.makedirs(parent, exist_ok=True)
            response.success = bool(wrapper.export_mesh(path))
            mesh = wrapper.get_reconstruction() if response.success else None
            response.num_vertices = int(len(mesh.vertices)) if mesh is not None else 0
            response.num_faces = int(len(getattr(mesh, 'faces', ()))) if mesh is not None else 0
            response.message = (f'Exported object {object_id} mesh to {path}'
                                if response.success else 'Reconstruction is not ready yet.')
            if response.success:
                self._mesh_pub.publish(String(data=path))
        except Exception as exc:
            response.success = False
            response.message = f'Mesh export failed: {exc}'
        return response

    def _on_save_model(self, request, response):
        if not request.path.strip():
            response.message = 'A model directory is required.'
            return response
        directory = os.path.abspath(os.path.expanduser(request.path))
        if not directory or directory == '/':
            response.message = 'Refusing an empty or root model directory.'
            return response
        object_id, wrapper = self._primary_wrapper()
        if wrapper is None:
            response.message = 'No initialized BundleSDF instance.'
            return response
        try:
            os.makedirs(directory, exist_ok=True)
            mesh_path = os.path.join(directory, f'object_{object_id}_mesh.ply')
            response.success = bool(wrapper.export_mesh(mesh_path))
            response.message = (f'Saved model mesh to {mesh_path}'
                                if response.success else 'Reconstruction is not ready yet.')
            if response.success:
                self._mesh_pub.publish(String(data=mesh_path))
        except Exception as exc:
            response.success = False
            response.message = f'Model save failed: {exc}'
        return response

    def destroy_node(self):
        if self.wrapper is not None:
            self.wrapper.reset()
            self.wrapper = None
        for context in self._instances.values():
            if context.get('wrapper') is not None:
                context['wrapper'].reset()
        self._instances.clear()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BundleSDFTrackingNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
