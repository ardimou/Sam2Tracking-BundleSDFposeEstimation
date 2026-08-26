"""Run GroundingDINO once for each object query."""
import sys

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from object_tracker_msgs.msg import GroundingResult as GroundingResultMsg
from object_tracker_msgs.msg import GroundingCandidate as GroundingCandidateMsg

from object_tracker_common import topics
from object_tracker_perception.backends import grounding_backend


class GroundingNode(Node):

    def __init__(self):
        super().__init__('grounding_node')

        self.declare_parameter('backend', 'grounding_dino')
        self.declare_parameter('lazy_load_backend', False)
        self.declare_parameter('initial_query', '')
        self.declare_parameter('model_id', 'IDEA-Research/grounding-dino-tiny')
        self.declare_parameter('device', 'auto')
        self.declare_parameter('box_threshold', 0.35)
        self.declare_parameter('text_threshold', 0.25)
        self.declare_parameter('max_alternative_candidates', 3)
        self.declare_parameter('max_instances', 3)
        self.declare_parameter('rgb_topic', topics.ADAPTER_RGB_TOPIC)
        self.declare_parameter('query_topic', topics.QUERY_TOPIC)
        self.declare_parameter('result_topic', topics.GROUNDING_RESULT_TOPIC)
        self.declare_parameter('annotated_image_topic', topics.GROUNDING_IMAGE_TOPIC)
        self.declare_parameter('publish_annotated_image', True)
        self.declare_parameter('status_topic', topics.GROUNDING_STATUS_TOPIC)

        self._validate_configuration()

        self.bridge = CvBridge()
        self._backend_name = str(self.get_parameter('backend').value)
        self._backend_load_failed = False
        self._latest_rgb = None
        self._latest_rgb_msg = None
        self._current_query = str(self.get_parameter('initial_query').value).strip()
        self._pending_query = self._current_query or None
        self._pending_timer = None
        self._next_object_id = 1

        self.create_subscription(
            Image, str(self.get_parameter('rgb_topic').value), self._on_rgb,
            topics.latest_only_qos())
        self.create_subscription(
            String, str(self.get_parameter('query_topic').value), self._on_query,
            topics.reliable_qos())

        self._result_pub = self.create_publisher(
            GroundingResultMsg, str(self.get_parameter('result_topic').value),
            topics.latched_qos())
        self._image_pub = self.create_publisher(
            Image, str(self.get_parameter('annotated_image_topic').value),
            topics.reliable_qos(depth=1))
        self._status_pub = self.create_publisher(
            String, str(self.get_parameter('status_topic').value), topics.latched_qos())

        self._backend = None
        if bool(self.get_parameter('lazy_load_backend').value):
            self._publish_status(f'waiting_for_query:{self._backend_name}')
        else:
            self._load_backend()
        self.add_on_set_parameters_callback(self._on_parameters_changed)
        self.get_logger().info(
            f'grounding_node up (backend={self._backend_name}, '
            f'lazy_load={bool(self.get_parameter("lazy_load_backend").value)}, '
            f'python={sys.executable})')
        if self._pending_query:
            self.get_logger().info(
                f'Initial query queued for first RGB frame: "{self._current_query}"')

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
                f'Could not load grounding backend with {sys.executable}: {exc}')
            self._publish_status(f'error:model_load_failed:{type(exc).__name__}')
            return False
        self._publish_status(f'ready:{self._backend_name}')
        return True

    def _validate_configuration(self):
        for name in ('box_threshold', 'text_threshold'):
            value = float(self.get_parameter(name).value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must be in [0, 1], got {value}')
        if int(self.get_parameter('max_alternative_candidates').value) < 0:
            raise ValueError('max_alternative_candidates must be non-negative')
        if int(self.get_parameter('max_instances').value) < 1:
            raise ValueError('max_instances must be positive')
        for name in ('rgb_topic', 'query_topic', 'result_topic',
                     'annotated_image_topic', 'status_topic'):
            if not str(self.get_parameter(name).value).strip():
                raise ValueError(f'{name} must not be empty')

    def _publish_status(self, status: str):
        self._status_pub.publish(String(data=status))

    def _on_parameters_changed(self, parameters):
        restart_parameters = {
            'backend', 'model_id', 'device', 'box_threshold', 'text_threshold',
            'max_alternative_candidates', 'max_instances',
            'rgb_topic', 'query_topic', 'result_topic', 'annotated_image_topic',
            'status_topic'}
        for parameter in parameters:
            if parameter.name in restart_parameters:
                return SetParametersResult(
                    successful=False, reason=f'{parameter.name} requires a node restart')
        for parameter in parameters:
            if parameter.name == 'initial_query':
                query = str(parameter.value).strip()
                if not query:
                    self._current_query = ''
                    self._pending_query = None
                    continue
                self._current_query = query
                self._pending_query = query
                if self._latest_rgb is not None:
                    if self._pending_timer is None:
                        self._pending_timer = self.create_timer(0.001, self._run_pending_once)
        return SetParametersResult(successful=True)

    def _run_pending_once(self):
        if self._pending_timer is not None:
            self._pending_timer.cancel()
            self.destroy_timer(self._pending_timer)
            self._pending_timer = None
        if self._pending_query and self._latest_rgb is not None:
            query = self._pending_query
            self._pending_query = None
            self._run_grounding(query)

    def _make_backend(self, name: str):
        if name == 'grounding_dino':
            device = str(self.get_parameter('device').value).strip()
            return grounding_backend.get_backend(
                name,
                model_id=str(self.get_parameter('model_id').value),
                device=None if device.lower() == 'auto' else device,
                box_threshold=float(self.get_parameter('box_threshold').value),
                text_threshold=float(self.get_parameter('text_threshold').value),
                max_alternative_candidates=int(
                    self.get_parameter('max_alternative_candidates').value),
            )
        raise ValueError(f'Unknown grounding backend "{name}".')

    def _on_rgb(self, msg: Image):
        try:
            first_frame = self._latest_rgb is None
            self._latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self._latest_rgb_msg = msg
        except Exception as exc:
            self.get_logger().error(f'Could not convert RGB image: {exc}')
            return

        if first_frame:
            self.get_logger().info(
                f'First RGB received on {self.get_parameter("rgb_topic").value}: '
                f'{msg.width}x{msg.height} encoding={msg.encoding}')

        if self._pending_query:
            query = self._pending_query
            self._pending_query = None
            self._run_grounding(query)

    def _on_query(self, msg: String):
        query = msg.data.strip()
        if not query:
            self.get_logger().warn('Ignoring empty grounding query.')
            return
        self._next_object_id = 1
        self._current_query = query
        if self._latest_rgb is None:
            self._pending_query = query
            self.get_logger().info(f'Queued grounding query until RGB arrives: "{query}"')
            self._publish_failure(query, 'waiting_for_rgb_frame')
        else:
            self._run_grounding(query)

    def _publish_failure(self, query: str, reason: str):
        out = GroundingResultMsg()
        out.query = query
        out.success = False
        out.failure_reason = reason
        self._result_pub.publish(out)

    def _run_grounding(self, query: str):
        if self._latest_rgb is None:
            self.get_logger().warn('Grounding requested but no RGB frame received yet.')
            self._pending_query = query
            self._publish_failure(query, 'waiting_for_rgb_frame')
            return
        if self._backend is None and not self._load_backend():
            self._publish_failure(query, 'model_not_available')
            return

        source_rgb = self._latest_rgb.copy()
        source_msg = self._latest_rgb_msg
        try:
            self._publish_status(f'inferencing:{query}')
            result = self._backend.ground(source_rgb, query)
            result = grounding_backend.validate_result(
                result, source_rgb.shape[1], source_rgb.shape[0])
            if result.success:
                self._assign_instance_ids(result)
        except Exception as exc:
            self.get_logger().error(f'Grounding inference failed: {exc}')
            result = grounding_backend.GroundingResult(
                query=query, success=False, failure_reason='backend_inference_error')

        out = GroundingResultMsg()
        out.header = source_msg.header
        out.source_image = source_msg
        out.query = result.query
        out.object_id = int(getattr(result, 'object_id', 0))
        out.success = result.success
        out.object_description = result.object_description
        out.confidence = result.confidence
        out.failure_reason = result.failure_reason
        if result.bounding_box is not None:
            out.bounding_box = [float(v) for v in result.bounding_box]
        out.alternative_candidates = [
            GroundingCandidateMsg(
                object_id=int(c.object_id), description=c.description,
                bounding_box=[float(v) for v in c.bounding_box],
                confidence=c.confidence)
            for c in result.alternative_candidates
        ]
        self._result_pub.publish(out)
        self._publish_annotated_image(source_rgb, source_msg, result)
        self._publish_status('ready' if result.success else f'failed:{result.failure_reason}')
        self.get_logger().info(
            f'Grounded "{query}" -> success={result.success} '
            f'instances={1 + len(result.alternative_candidates) if result.success else 0} '
            f'conf={result.confidence:.2f}')

    def _assign_instance_ids(self, result):
        """Assign IDs once for this detection set; no cross-query re-identification."""
        result.object_id = self._next_object_id
        self._next_object_id += 1
        limit = max(1, int(self.get_parameter('max_instances').value))
        result.alternative_candidates = result.alternative_candidates[:limit - 1]
        for candidate in result.alternative_candidates:
            candidate.object_id = self._next_object_id
            self._next_object_id += 1

    def _publish_annotated_image(self, source_rgb, source_msg, result):
        if not bool(self.get_parameter('publish_annotated_image').value):
            return
        annotated = source_rgb.copy()
        if result.success and result.bounding_box is not None:
            detections = [(result.object_id, result.object_description or result.query,
                           result.bounding_box, result.confidence)]
            detections.extend((candidate.object_id, candidate.description,
                               candidate.bounding_box, candidate.confidence)
                              for candidate in result.alternative_candidates)
            for index, (object_id, description, box, confidence) in enumerate(detections):
                color = ((37 * index) % 200 + 55, (97 * index) % 200 + 55,
                         (157 * index) % 200 + 55)
                self._draw_detection(
                    annotated, box, f'ID {object_id} {description}: {confidence:.2f}', color)
        image_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        image_msg.header = source_msg.header
        self._image_pub.publish(image_msg)

    @staticmethod
    def _draw_detection(image, box, label, color):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        p1 = (max(0, min(w - 1, int(round(x1)))),
              max(0, min(h - 1, int(round(y1)))))
        p2 = (max(0, min(w - 1, int(round(x2)))),
              max(0, min(h - 1, int(round(y2)))))
        cv2.rectangle(image, p1, p2, color, 2)
        cv2.putText(image, label, (p1[0], max(20, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    def reground_current_target(self):
        """Called by recovery flows to re-locate the same query."""
        if self._current_query:
            self._run_grounding(self._current_query)


def main(args=None):
    rclpy.init(args=args)
    node = GroundingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
