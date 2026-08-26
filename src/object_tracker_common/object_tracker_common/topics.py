"""Shared topic names, frame names, and QoS profiles."""
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,
                       QoSDurabilityPolicy)


def latest_only_qos(depth: int = 1) -> QoSProfile:
    """Best-effort QoS for latest-frame data."""
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def reliable_qos(depth: int = 10) -> QoSProfile:
    """Reliable QoS for state and result topics."""
    return QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def latched_qos(depth: int = 1) -> QoSProfile:
    """Reliable transient-local QoS for the latest node state."""
    return QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )


XTION_RGB_TOPIC = '/camera/rgb/image_raw'
XTION_DEPTH_TOPIC = '/camera/depth_registered/image_raw'
XTION_INFO_TOPIC = '/camera/rgb/camera_info'

ADAPTER_RGB_TOPIC = '/rgbd/rgb'
ADAPTER_DEPTH_TOPIC = '/rgbd/depth_m'
ADAPTER_INFO_TOPIC = '/rgbd/camera_info'
ADAPTER_FRAME_TOPIC = '/rgbd/synced'

QUERY_TOPIC = '/object/query'
GROUNDING_RESULT_TOPIC = '/object/grounding_result'
GROUNDING_IMAGE_TOPIC = '/object/grounding_image'
GROUNDING_STATUS_TOPIC = '/object/grounding_status'

MASK_TOPIC = '/object/mask'
MASK_CONFIDENCE_TOPIC = '/object/mask_confidence'
MASK_STATE_TOPIC = '/object/mask_state'
SEGMENTATION_STATUS_TOPIC = '/object/segmentation_status'
SEGMENTATION_IMAGE_TOPIC = '/object/segmentation_image'
INSTANCE_MASK_TOPIC = '/objects/masks'

UNIFIED_CLOUD_TOPIC = '/object/unified_cloud'

POSE_RAW_TOPIC = '/object/pose_raw'
RECONSTRUCTION_PREVIEW_TOPIC = '/object/reconstruction'
TRACKER_STATUS_TOPIC = '/object/tracker_status'
TRACKER_DIAGNOSTICS_TOPIC = '/object/tracker_diagnostics'
INSTANCE_TRACKER_STATUS_TOPIC = '/objects/tracker_status'
INSTANCE_POSE_TOPIC = '/objects/poses'
INSTANCE_TABLE_POSE_TOPIC = '/objects/table_poses'
INSTANCE_TABLE_RAW_POSE_TOPIC = '/objects/table_raw_poses'
TABLE_POSE_TOPIC = '/object/pose_table'
TABLE_FRAME_STATUS_TOPIC = '/table_frame/status'
INSTANCE_TRACKING_STATE_TOPIC = '/objects/tracking_states'

POSE_TOPIC = '/object/pose'
TRACKING_STATE_TOPIC = '/object/tracking_state'
CONFIDENCE_TOPIC = '/object/confidence'
FAILURE_REASON_TOPIC = '/object/failure_reason'

MODEL_CLOUD_TOPIC = '/object/model_cloud'
MODEL_MESH_TOPIC = '/object/model_mesh'
RECONSTRUCTION_STATUS_TOPIC = '/object/reconstruction_status'

SAVE_MODEL_SERVICE = '/object/save_model'
EXPORT_MESH_SERVICE = '/object/export_mesh'
RESET_SERVICE = '/object/reset'

VIZ_IMAGE_TOPIC = '/object/viz/image'
VIZ_MARKERS_TOPIC = '/object/viz/markers'
CAMERA_OPTICAL_FRAME = 'camera_rgb_optical_frame'
TABLE_FRAME = 'table_frame'
TRACKED_OBJECT_FRAME = 'tracked_object'
