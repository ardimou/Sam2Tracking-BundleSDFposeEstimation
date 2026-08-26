"""Initialize SAM2 from grounded boxes and propagate instance masks."""
from collections import OrderedDict
import sys
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

from object_tracker_msgs.msg import GroundingResult as GroundingResultMsg
from object_tracker_msgs.msg import MaskState as MaskStateMsg
from object_tracker_msgs.msg import InstanceMask
from object_tracker_common import topics
from object_tracker_perception.backends import segmentation_backend as seg


class SegmentationNode(Node):

    def __init__(self):
        super().__init__('segmentation_node')
        self._declare_parameters()
        self._validate_parameters()

        self.bridge = CvBridge()
        self._backend = None
        self._backend_load_failed = False
        self._active = False
        self._instances = {}
        self._last_propagation_time = 0.0
        self._frame_cache = OrderedDict()

        self._mask_pub = self.create_publisher(
            Image, self._param('mask_topic'), topics.reliable_qos(depth=1))
        self._conf_pub = self.create_publisher(
            Float32, self._param('confidence_topic'), topics.latest_only_qos())
        self._state_pub = self.create_publisher(
            MaskStateMsg, self._param('state_topic'), topics.reliable_qos())
        self._status_pub = self.create_publisher(
            String, self._param('status_topic'), topics.latched_qos())
        self._instance_mask_pub = self.create_publisher(
            InstanceMask, self._param('instance_mask_topic'), topics.reliable_qos())
        self._image_pub = self.create_publisher(
            Image, self._param('annotated_image_topic'), topics.reliable_qos(depth=1))

        self.create_subscription(
            Image, self._param('rgb_topic'), self._on_rgb, topics.latest_only_qos())
        self.create_subscription(
            GroundingResultMsg, self._param('grounding_result_topic'),
            self._on_grounding, topics.latched_qos())
        self.create_subscription(
            String, topics.QUERY_TOPIC, self._on_query,
            topics.reliable_qos())

        self._backend_name = self._param('backend')
        if bool(self._param('lazy_load_backend')):
            self._publish_status(f'waiting_for_grounding:{self._backend_name}')
        else:
            self._load_backend()
        self.get_logger().info(
            f'segmentation_node up (backend={self._backend_name}, '
            f'propagation_mode={self._param("propagation_mode")}, '
            f'lazy_load={bool(self._param("lazy_load_backend"))}, '
            f'python={sys.executable})')

    def _load_backend(self):
        if self._backend is not None:
            return True
        if self._backend_load_failed:
            return False
        self._publish_status(f'loading:{self._backend_name}')
        try:
            self._backend = self._make_backend(self._backend_name)
        except Exception as exc:
            self._backend_load_failed = True
            self.get_logger().error(
                f'Could not load segmentation backend with {sys.executable}: {exc}')
            self._publish_status(f'error:model_load_failed:{type(exc).__name__}')
            return False
        self._publish_status(f'ready:{self._backend_name}')
        return True

    def _declare_parameters(self):
        defaults = {
            'backend': 'sam2',
            'lazy_load_backend': False,
            'model_id': 'facebook/sam2.1-hiera-tiny',
            'device': 'auto',
            'propagation_mode': 'video_memory',
            'propagation_rate_hz': 5.0,
            'confidence_tracking_threshold': 0.75,
            'confidence_uncertain_threshold': 0.5,
            'max_centroid_jump_frac': 0.25,
            'num_interior_points': 2,
            'frame_cache_size': 180,
            'offload_video_to_cpu': True,
            'offload_state_to_cpu': False,
            'retained_raw_frames': 2,
            'sam2_show_progress': False,
            'sam2_fill_hole_area': 0,
            'rgb_topic': topics.ADAPTER_RGB_TOPIC,
            'grounding_result_topic': topics.GROUNDING_RESULT_TOPIC,
            'mask_topic': topics.MASK_TOPIC,
            'confidence_topic': topics.MASK_CONFIDENCE_TOPIC,
            'state_topic': topics.MASK_STATE_TOPIC,
            'status_topic': topics.SEGMENTATION_STATUS_TOPIC,
            'instance_mask_topic': topics.INSTANCE_MASK_TOPIC,
            'annotated_image_topic': topics.SEGMENTATION_IMAGE_TOPIC,
            'publish_annotated_image': True,
            'mask_overlay_alpha': 0.45,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _param(self, name):
        return self.get_parameter(name).value

    def _validate_parameters(self):
        if self._param('backend') != 'sam2':
            raise ValueError('backend must be sam2')
        if self._param('propagation_mode') != 'video_memory':
            raise ValueError('propagation_mode must be video_memory')
        if float(self._param('propagation_rate_hz')) <= 0.0:
            raise ValueError('propagation_rate_hz must be positive')
        uncertain = float(self._param('confidence_uncertain_threshold'))
        tracking = float(self._param('confidence_tracking_threshold'))
        if not 0.0 <= uncertain <= tracking <= 1.0:
            raise ValueError('confidence thresholds must satisfy 0 <= uncertain <= tracking <= 1')
        if not 0.0 <= float(self._param('max_centroid_jump_frac')) <= 1.0:
            raise ValueError('max_centroid_jump_frac must be in [0, 1]')
        if int(self._param('num_interior_points')) < 0:
            raise ValueError('num_interior_points must be non-negative')
        if int(self._param('frame_cache_size')) < 1:
            raise ValueError('frame_cache_size must be positive')
        if int(self._param('sam2_fill_hole_area')) < 0:
            raise ValueError('sam2_fill_hole_area must be non-negative')
        if not 0.0 <= float(self._param('mask_overlay_alpha')) <= 1.0:
            raise ValueError('mask_overlay_alpha must be in [0, 1]')
        for name in ('rgb_topic', 'grounding_result_topic', 'mask_topic',
                     'confidence_topic', 'state_topic', 'status_topic',
                     'instance_mask_topic', 'annotated_image_topic'):
            if not str(self._param(name)).strip():
                raise ValueError(f'{name} must not be empty')

    def _make_backend(self, name):
        device = str(self._param('device')).strip()
        return seg.Sam2VideoBackend(
            model_id=str(self._param('model_id')),
            device=None if device.lower() == 'auto' else device,
            confidence_tracking_threshold=float(
                self._param('confidence_tracking_threshold')),
            confidence_uncertain_threshold=float(
                self._param('confidence_uncertain_threshold')),
            offload_video_to_cpu=bool(self._param('offload_video_to_cpu')),
            offload_state_to_cpu=bool(self._param('offload_state_to_cpu')),
            retained_raw_frames=int(self._param('retained_raw_frames')),
            show_progress=bool(self._param('sam2_show_progress')),
            fill_hole_area=int(self._param('sam2_fill_hole_area')))

    @staticmethod
    def _stamp_key(header):
        return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)

    def _on_rgb(self, msg: Image):
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Could not convert RGB image: {exc}')
            return
        self._frame_cache[self._stamp_key(msg.header)] = (rgb, msg.header)
        while len(self._frame_cache) > int(self._param('frame_cache_size')):
            self._frame_cache.popitem(last=False)

        if not self._instances or self._param('propagation_mode') == 'initial_only':
            return
        now = time.monotonic()
        if now - self._last_propagation_time < 1.0 / float(self._param('propagation_rate_hz')):
            return
        self._last_propagation_time = now
        if self._param('propagation_mode') == 'video_memory':
            self._propagate_video(rgb, msg.header)
            return
        primary_id = min(self._instances)
        overlay_outputs = {}
        for object_id, context in list(self._instances.items()):
            try:
                self._backend.import_tracking_state(context['state'])
                result = self._backend.propagate(rgb)
                context['state'] = self._backend.export_tracking_state()
            except Exception as exc:
                self.get_logger().error(
                    f'SAM2 propagation failed for object {object_id}: {exc}')
                result = seg.MaskResult(None, 0.0, seg.MaskState.LOST)
            self._publish_instance(
                object_id, context['label'], result, msg.header, rgb.shape,
                compatibility=(object_id == primary_id))
            overlay_outputs[object_id] = (context['label'], result)
            if result.state in (seg.MaskState.LOST, seg.MaskState.SWITCHED):
                del self._instances[object_id]
        self._publish_annotated_image(rgb, msg.header, overlay_outputs)

    def _on_grounding(self, msg: GroundingResultMsg):
        if not msg.success or len(msg.bounding_box) != 4:
            self.get_logger().warn('Ignoring failed/empty grounding result.')
            return

        if self._backend is None and not self._load_backend():
            self.get_logger().error('Segmentation requested but model is unavailable.')
            self._publish_status('error:model_not_available')
            return
        if msg.source_image.height and msg.source_image.width:
            try:
                rgb = self.bridge.imgmsg_to_cv2(
                    msg.source_image, desired_encoding='bgr8')
                header = msg.source_image.header
            except Exception as exc:
                self.get_logger().error(
                    f'Could not decode grounding source RGB: {exc}')
                self._publish_status('failed:grounding_source_decode')
                return
        else:
            frame = self._frame_cache.get(self._stamp_key(msg.header))
            if frame is None:
                self.get_logger().warn(
                    'Grounding source RGB is no longer in the frame cache.')
                self._publish_status('failed:grounding_frame_not_cached')
                return
            rgb, header = frame
        detections = [(msg.object_id, msg.object_description or msg.query,
                       list(msg.bounding_box))]
        detections.extend((candidate.object_id, candidate.description,
                           list(candidate.bounding_box))
                          for candidate in msg.alternative_candidates)
        self._instances.clear()
        self._publish_status(f'inferencing:{len(detections)}_initial_masks')
        if self._param('propagation_mode') == 'video_memory':
            self._initialize_video(rgb, header, detections)
            return
        overlay_outputs = {}
        for index, (object_id, label, box) in enumerate(detections):
            try:
                result = self._backend.init_from_box(rgb, box)
            except Exception as exc:
                self.get_logger().error(
                    f'SAM2 initialization failed for object {object_id}: {exc}')
                result = seg.MaskResult(None, 0.0, seg.MaskState.LOST)
            result = seg.validate_mask_result(result, rgb.shape)
            if result.mask is not None and result.state != seg.MaskState.LOST:
                self._instances[int(object_id)] = {
                    'label': label, 'state': self._backend.export_tracking_state()}
            self._publish_instance(
                int(object_id), label, result, header, rgb.shape,
                compatibility=(index == 0))
            overlay_outputs[int(object_id)] = (label, result)
        self._publish_annotated_image(rgb, header, overlay_outputs)
        self._active = bool(self._instances)
        self._last_propagation_time = time.monotonic()

    def _on_query(self, msg: String):
        self.reset()

    def _initialize_video(self, rgb, header, detections):
        try:
            outputs = self._backend.initialize(rgb, detections)
        except Exception as exc:
            self.get_logger().error(f'SAM2 video initialization failed: {exc}')
            self._publish_status(f'error:video_initialization_failed:{type(exc).__name__}')
            return
        for index, (object_id, label, _) in enumerate(detections):
            result_label, result = outputs.get(
                int(object_id), (label, seg.MaskResult(None, 0.0, seg.MaskState.LOST)))
            if result.mask is not None:
                self._instances[int(object_id)] = {'label': result_label or label}
            self._publish_instance(
                int(object_id), result_label or label, result, header, rgb.shape,
                compatibility=(index == 0))
        self._publish_annotated_image(rgb, header, outputs)
        self._active = bool(self._instances)
        self._last_propagation_time = time.monotonic()

    def _propagate_video(self, rgb, header):
        labels = {object_id: context['label']
                  for object_id, context in self._instances.items()}
        try:
            outputs = self._backend.propagate(rgb, labels)
        except Exception as exc:
            self.get_logger().error(f'SAM2 video propagation failed: {exc}')
            self._publish_status(f'error:video_propagation_failed:{type(exc).__name__}')
            return
        primary_id = min(self._instances) if self._instances else None
        for object_id, (label, result) in outputs.items():
            self._publish_instance(
                object_id, label, result, header, rgb.shape,
                compatibility=(object_id == primary_id))
            if result.state in (seg.MaskState.LOST, seg.MaskState.SWITCHED):
                self._instances.pop(object_id, None)
        self._publish_annotated_image(rgb, header, outputs)
        self._active = bool(self._instances)

    def _publish_instance(self, object_id, label, result, header, image_shape,
                          compatibility=False):
        result = seg.validate_mask_result(result, image_shape)
        instance = InstanceMask()
        instance.header = header
        instance.object_id = object_id
        instance.label = label
        instance.confidence = result.confidence
        instance.state = int(result.state)
        if result.mask is not None:
            instance.mask = self.bridge.cv2_to_imgmsg(result.mask * 255, encoding='mono8')
            instance.mask.header = header
        self._instance_mask_pub.publish(instance)
        if compatibility:
            self._publish(result, header, image_shape)

    def _publish(self, result, header, image_shape):
        result = seg.validate_mask_result(result, image_shape)
        if result.mask is not None:
            mask_msg = self.bridge.cv2_to_imgmsg(result.mask * 255, encoding='mono8')
            mask_msg.header = header
            self._mask_pub.publish(mask_msg)
        self._conf_pub.publish(Float32(data=result.confidence))
        state_msg = MaskStateMsg()
        state_msg.header = header
        state_msg.state = int(result.state)
        state_msg.confidence = result.confidence
        self._state_pub.publish(state_msg)
        if result.state in (seg.MaskState.LOST, seg.MaskState.SWITCHED):
            self._active = False
            self._publish_status(f'failed:{result.state.name.lower()}')
        else:
            self._publish_status(f'active:{result.state.name.lower()}')

    def _publish_annotated_image(self, rgb, header, outputs):
        """Publish one composite RGB image containing every instance mask."""
        if not bool(self._param('publish_annotated_image')):
            return
        annotated = rgb.copy()
        alpha = float(self._param('mask_overlay_alpha'))
        for object_id in sorted(outputs):
            label, result = outputs[object_id]
            result = seg.validate_mask_result(result, rgb.shape)
            if result.mask is None:
                continue
            mask = np.asarray(result.mask, dtype=bool)
            color = np.array(self._instance_color(object_id), dtype=np.float32)
            pixels = annotated[mask].astype(np.float32)
            annotated[mask] = np.clip(
                (1.0 - alpha) * pixels + alpha * color, 0, 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, tuple(int(v) for v in color), 2)
            ys, xs = np.nonzero(mask)
            if xs.size:
                origin = (int(xs.min()), max(20, int(ys.min()) - 7))
                text = (f'ID {object_id} {label}: {result.confidence:.2f} '
                        f'[{result.state.name}]')
                cv2.putText(annotated, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, tuple(int(v) for v in color), 2, cv2.LINE_AA)
        image_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        image_msg.header = header
        self._image_pub.publish(image_msg)

    @staticmethod
    def _instance_color(object_id):
        seed = max(1, int(object_id))
        return ((37 * seed) % 200 + 55,
                (97 * seed) % 200 + 55,
                (157 * seed) % 200 + 55)

    def _publish_status(self, value):
        self._status_pub.publish(String(data=value))

    def reset(self):
        if self._backend is not None:
            self._backend.reset()
        self._active = False
        self._instances.clear()
        self._publish_status('idle')


def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
