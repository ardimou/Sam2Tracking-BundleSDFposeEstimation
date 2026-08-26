"""RGB-D and mask preprocessing for BundleSDF."""
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class BundleSDFInput:
    rgb: np.ndarray          # HxWx3 uint8, contiguous, BGR (upstream OpenCV order)
    depth: np.ndarray        # HxW float32, metres, contiguous
    mask: np.ndarray         # HxW uint8 {0,1}, contiguous
    intrinsics: np.ndarray   # 3x3 float64, scaled to match rgb/depth size
    timestamp: object        # builtin_interfaces/Time or float seconds


def masked_depth_statistics(inp: BundleSDFInput):
    """Return valid masked depth count, mask count, and their ratio."""
    mask = np.asarray(inp.mask) > 0
    depth = np.asarray(inp.depth)
    mask_points = int(np.count_nonzero(mask))
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    valid_points = int(np.count_nonzero(valid))
    return valid_points, mask_points, valid_points / max(mask_points, 1)


class RGBDPreprocessor:

    def __init__(self, target_size=None, min_depth_m: float = 0.1,
                 max_depth_m: float = 4.0):
        """
        target_size: (width, height) to resize to, or None to keep native
            Xtion resolution (640x480). BundleSDF's NeRF training cost
            scales with resolution, so downsizing is often worth it even
            at some pose-accuracy cost - tune per scene.
        min_depth_m / max_depth_m: BundleSDF-relevant working volume;
            anything outside this range is treated as invalid, separately
            from the adapter's own (wider) sensor-range truncation.
        """
        self.target_size = target_size
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m

    def process(self, rgb_bgr: np.ndarray, depth_m: np.ndarray,
                mask: np.ndarray, intrinsics: np.ndarray,
                timestamp) -> BundleSDFInput:
        h0, w0 = rgb_bgr.shape[:2]

        rgb = rgb_bgr.copy()
        depth = depth_m.astype(np.float32, copy=True)
        m = self._as_binary_mask(mask, (h0, w0))

        invalid = (~np.isfinite(depth)) | (depth < self.min_depth_m) | (depth > self.max_depth_m)
        depth[invalid] = 0.0

        K = intrinsics.astype(np.float64).copy()

        if self.target_size is not None and self.target_size != (w0, h0):
            rgb, depth, m, K = self._resize_all(rgb, depth, m, K, self.target_size)

        rgb = np.ascontiguousarray(rgb)
        depth = np.ascontiguousarray(depth)
        m = np.ascontiguousarray(m)

        return BundleSDFInput(rgb=rgb, depth=depth, mask=m, intrinsics=K,
                               timestamp=timestamp)

    def _as_binary_mask(self, mask: np.ndarray, shape) -> np.ndarray:
        if mask is None:
            return np.zeros(shape, dtype=np.uint8)
        m = mask
        if m.shape[:2] != shape:
            m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        return (m > 0).astype(np.uint8)

    def _resize_all(self, rgb, depth, mask, K, target_size):
        tw, th = target_size
        h0, w0 = rgb.shape[:2]
        sx, sy = tw / w0, th / h0

        rgb_r = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
        depth_r = cv2.resize(depth, (tw, th), interpolation=cv2.INTER_NEAREST)
        mask_r = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)

        K_r = K.copy()
        K_r[0, 0] *= sx  # fx
        K_r[1, 1] *= sy  # fy
        K_r[0, 2] *= sx  # cx
        K_r[1, 2] *= sy  # cy

        return rgb_r, depth_r, (mask_r > 0).astype(np.uint8), K_r
