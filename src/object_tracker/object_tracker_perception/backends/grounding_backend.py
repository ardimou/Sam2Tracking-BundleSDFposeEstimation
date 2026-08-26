"""GroundingDINO inference and structured detection results."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import inspect
from typing import List, Optional
import numpy as np


@dataclass
class GroundingCandidate:
    description: str
    bounding_box: List[float]   # [x_min, y_min, x_max, y_max] pixels
    confidence: float
    object_id: int = 0


@dataclass
class GroundingResult:
    query: str
    success: bool
    object_description: str = ''
    bounding_box: Optional[List[float]] = None
    confidence: float = 0.0
    alternative_candidates: List[GroundingCandidate] = field(default_factory=list)
    failure_reason: str = ''
    object_id: int = 0


class GroundingBackend(ABC):

    @abstractmethod
    def ground(self, rgb: np.ndarray, query: str) -> GroundingResult:
        """Locate `query` in `rgb` (HxWx3, BGR uint8). Must return
        structured data - never free-form text - per the project spec."""
        raise NotImplementedError


class GroundingDINOBackend(GroundingBackend):
    """Hugging Face GroundingDINO backend."""

    def __init__(self, model_id: str = 'IDEA-Research/grounding-dino-tiny',
                 device: Optional[str] = None,
                 box_threshold: float = 0.35, text_threshold: float = 0.25,
                 max_alternative_candidates: int = 3):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self._torch = torch
        self._device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self._device)
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._max_alt = max_alternative_candidates

    def ground(self, rgb: np.ndarray, query: str) -> GroundingResult:
        from PIL import Image

        image = Image.fromarray(rgb[:, :, ::-1])  # BGR (our convention) -> RGB for PIL
        text = self._format_query(query)

        inputs = self._processor(images=image, text=text, return_tensors='pt').to(self._device)
        with self._torch.no_grad():
            outputs = self._model(**inputs)

        post_process = self._processor.post_process_grounded_object_detection
        threshold_name = ('threshold' if 'threshold' in
                          inspect.signature(post_process).parameters
                          else 'box_threshold')
        post_process_kwargs = {
            threshold_name: self._box_threshold,
            'text_threshold': self._text_threshold,
            'target_sizes': [image.size[::-1]],
        }
        results = post_process(
            outputs, inputs.input_ids, **post_process_kwargs)[0]

        boxes = results['boxes'].detach().cpu().numpy()
        scores = results['scores'].detach().cpu().numpy()
        labels = results.get('text_labels', results.get('labels', [query] * len(boxes)))

        if len(boxes) == 0:
            return GroundingResult(
                query=query, success=False,
                failure_reason='no_detection_above_threshold')

        order = np.argsort(-scores)
        boxes, scores = boxes[order], scores[order]
        labels = [labels[i] for i in order] if not isinstance(labels, str) else labels

        best_box = [float(v) for v in boxes[0]]
        alternatives = [
            GroundingCandidate(
                object_id=0, description=str(labels[i]) if i < len(labels) else query,
                bounding_box=[float(v) for v in boxes[i]],
                confidence=float(scores[i]))
            for i in range(1, min(len(boxes), self._max_alt + 1))
        ]

        return GroundingResult(
            query=query, success=True,
            object_description=str(labels[0]) if len(labels) else query,
            bounding_box=best_box, confidence=float(scores[0]),
            alternative_candidates=alternatives,
        )

    @staticmethod
    def _format_query(query: str) -> str:
        q = query.strip().lower()
        if not q.endswith('.'):
            q += '.'
        return q


def validate_result(result: GroundingResult, width: int, height: int) -> GroundingResult:
    """Validate and clamp backend output before it crosses the ROS boundary."""
    if not result.success:
        return result
    if result.bounding_box is None or len(result.bounding_box) != 4:
        return GroundingResult(query=result.query, success=False,
                               failure_reason='invalid_bounding_box')
    box = np.asarray(result.bounding_box, dtype=np.float64)
    if not np.all(np.isfinite(box)):
        return GroundingResult(query=result.query, success=False,
                               failure_reason='invalid_bounding_box')
    x1, y1, x2, y2 = box.tolist()
    x1, x2 = np.clip([x1, x2], 0.0, max(0.0, width - 1.0))
    y1, y2 = np.clip([y1, y2], 0.0, max(0.0, height - 1.0))
    if x2 <= x1 or y2 <= y1:
        return GroundingResult(query=result.query, success=False,
                               failure_reason='invalid_bounding_box')
    result.bounding_box = [float(x1), float(y1), float(x2), float(y2)]
    result.confidence = float(np.clip(result.confidence, 0.0, 1.0))
    valid_alternatives = []
    for candidate in result.alternative_candidates:
        candidate_result = validate_result(
            GroundingResult(query=result.query, success=True,
                            bounding_box=candidate.bounding_box,
                            confidence=candidate.confidence), width, height)
        if candidate_result.success:
            valid_alternatives.append(GroundingCandidate(
                description=candidate.description,
                bounding_box=candidate_result.bounding_box,
                confidence=candidate_result.confidence))
    result.alternative_candidates = valid_alternatives
    return result


def get_backend(name: str = 'grounding_dino', **kwargs) -> GroundingBackend:
    if name == 'grounding_dino':
        return GroundingDINOBackend(**kwargs)
    raise ValueError(f'Unknown grounding backend "{name}".')
