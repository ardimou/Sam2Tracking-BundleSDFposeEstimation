"""SAM2 video-memory segmentation backend."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional
from contextlib import nullcontext
from collections import OrderedDict
import numpy as np


class MaskState(IntEnum):
    INITIALIZING = 0
    TRACKING = 1
    UNCERTAIN = 2
    LOST = 3
    SWITCHED = 4


@dataclass
class MaskResult:
    mask: Optional[np.ndarray]   # HxW bool/uint8, None if no mask
    confidence: float
    state: MaskState


def validate_mask_result(result: MaskResult, image_shape) -> MaskResult:
    """Normalize and validate a backend result before ROS publication."""
    confidence = float(np.clip(result.confidence, 0.0, 1.0))
    if result.mask is None:
        return MaskResult(None, confidence, result.state)
    mask = np.asarray(result.mask)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2 or tuple(mask.shape) != tuple(image_shape[:2]):
        return MaskResult(None, 0.0, MaskState.LOST)
    mask = np.ascontiguousarray(mask > 0, dtype=np.uint8)
    if not np.any(mask):
        return MaskResult(None, 0.0, MaskState.LOST)
    return MaskResult(mask, confidence, result.state)


class SegmentationBackend(ABC):

    @abstractmethod
    def init_from_box(self, rgb: np.ndarray, box: List[float]) -> MaskResult:
        """(Re)initialize tracking from a VLM bounding box on `rgb`."""
        raise NotImplementedError

    @abstractmethod
    def propagate(self, rgb: np.ndarray) -> MaskResult:
        """Advance the mask to a new frame. Called once per processed
        frame while a track is active."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    def export_tracking_state(self):
        """Return lightweight per-instance state; model weights stay shared."""
        raise NotImplementedError

    def import_tracking_state(self, state) -> None:
        raise NotImplementedError



class Sam2VideoBackend:
    """Append live ROS frames to one multi-object SAM2 video state."""

    def __init__(self, model_id='facebook/sam2.1-hiera-tiny', device=None,
                 confidence_tracking_threshold=0.75,
                 confidence_uncertain_threshold=0.5,
                 offload_video_to_cpu=True, offload_state_to_cpu=False,
                 retained_raw_frames=2, show_progress=False,
                 fill_hole_area=0):
        self._configure_local_sam2_import()
        import torch
        import sam2.sam2_video_predictor as video_predictor_module
        from sam2.sam2_video_predictor import SAM2VideoPredictor

        self._torch = torch
        self._device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self._predictor = SAM2VideoPredictor.from_pretrained(
            model_id, device=str(self._device))
        if not show_progress:
            video_predictor_module.tqdm = (
                lambda iterable, *args, **kwargs: iterable)
        self._predictor.fill_hole_area = max(0, int(fill_hole_area))
        self._tracking_thr = confidence_tracking_threshold
        self._uncertain_thr = confidence_uncertain_threshold
        self._offload_video = offload_video_to_cpu
        self._offload_state = offload_state_to_cpu
        self._retained_raw_frames = max(1, int(retained_raw_frames))
        self._state = None

    @staticmethod
    def _configure_local_sam2_import():
        """Prefer the checked-out SAM2 package when running from this workspace."""
        import sys
        from pathlib import Path

        candidates = [Path.cwd() / 'sam2']
        candidates.extend(parent / 'sam2' for parent in Path(__file__).resolve().parents)
        for candidate in candidates:
            if (candidate / 'sam2' / 'sam2_video_predictor.py').is_file():
                path = str(candidate)
                if path not in sys.path:
                    sys.path.insert(0, path)
                loaded = sys.modules.get('sam2')
                if loaded is not None and getattr(loaded, '__file__', None) is None:
                    del sys.modules['sam2']
                return

    def initialize(self, bgr, detections):
        """Initialize all `(object_id, label, box)` detections on frame zero."""
        self.reset()
        autocast = (self._torch.autocast('cuda', dtype=self._torch.bfloat16)
                    if self._device.type == 'cuda' else nullcontext())
        with self._torch.inference_mode(), autocast:
            self._state = self._new_state(bgr)
            labels = {int(object_id): label for object_id, label, _ in detections}
            object_ids, logits = [], None
            for object_id, label, box in detections:
                _, object_ids, logits = self._predictor.add_new_points_or_box(
                    self._state, frame_idx=0, obj_id=int(object_id),
                    box=np.asarray(box, dtype=np.float32))
            self._predictor.propagate_in_video_preflight(self._state)
        return self._decode(object_ids, logits, labels) if logits is not None else {}

    def propagate(self, bgr, labels):
        if self._state is None:
            return {}
        frame_idx = self._append_frame(bgr)
        autocast = (self._torch.autocast('cuda', dtype=self._torch.bfloat16)
                    if self._device.type == 'cuda' else nullcontext())
        with self._torch.inference_mode(), autocast:
            iterator = self._predictor.propagate_in_video(
                self._state, start_frame_idx=frame_idx, max_frame_num_to_track=1)
            _, object_ids, logits = next(iterator)
        self._release_old_raw_frames(frame_idx)
        return self._decode(object_ids, logits, labels)

    def reset(self):
        self._state = None
        if hasattr(self, '_torch') and self._device.type == 'cuda':
            self._torch.cuda.empty_cache()

    def _new_state(self, bgr):
        frame = self._prepare_frame(bgr)
        storage_device = self._torch.device('cpu') if self._offload_state else self._device
        state = {
            'images': [frame], 'num_frames': 1,
            'offload_video_to_cpu': self._offload_video,
            'offload_state_to_cpu': self._offload_state,
            'video_height': int(bgr.shape[0]), 'video_width': int(bgr.shape[1]),
            'device': self._device, 'storage_device': storage_device,
            'point_inputs_per_obj': {}, 'mask_inputs_per_obj': {},
            'cached_features': {}, 'constants': {},
            'obj_id_to_idx': OrderedDict(), 'obj_idx_to_id': OrderedDict(),
            'obj_ids': [], 'output_dict_per_obj': {},
            'temp_output_dict_per_obj': {}, 'frames_tracked_per_obj': {},
            'tracking_has_started': False,
        }
        self._predictor._get_image_feature(state, frame_idx=0, batch_size=1)
        return state

    def _append_frame(self, bgr):
        frame_idx = self._state['num_frames']
        if (int(bgr.shape[0]) != self._state['video_height'] or
                int(bgr.shape[1]) != self._state['video_width']):
            raise ValueError('SAM2 video frame dimensions changed during tracking')
        self._state['images'].append(self._prepare_frame(bgr))
        self._state['num_frames'] += 1
        return frame_idx

    def _prepare_frame(self, bgr):
        import cv2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self._predictor.image_size, self._predictor.image_size),
                         interpolation=cv2.INTER_LINEAR)
        tensor = self._torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        mean = self._torch.tensor((0.485, 0.456, 0.406))[:, None, None]
        std = self._torch.tensor((0.229, 0.224, 0.225))[:, None, None]
        tensor = (tensor - mean) / std
        return tensor if self._offload_video else tensor.to(self._device)

    def _release_old_raw_frames(self, current_idx):
        cutoff = current_idx - self._retained_raw_frames
        for index in range(max(0, cutoff)):
            frame = self._state['images'][index]
            if frame is not None:
                self._state['images'][index] = None

    def _decode(self, object_ids, logits, labels):
        outputs = {}
        for index, object_id in enumerate(object_ids):
            score_map = self._torch.sigmoid(logits[index, 0]).detach().float().cpu().numpy()
            mask = score_map > 0.5
            confidence = float(score_map[mask].mean()) if np.any(mask) else 0.0
            if confidence >= self._tracking_thr:
                state = MaskState.TRACKING
            elif confidence >= self._uncertain_thr:
                state = MaskState.UNCERTAIN
            else:
                state = MaskState.LOST
            outputs[int(object_id)] = (
                labels.get(int(object_id), ''),
                validate_mask_result(MaskResult(mask, confidence, state), mask.shape))
        return outputs
