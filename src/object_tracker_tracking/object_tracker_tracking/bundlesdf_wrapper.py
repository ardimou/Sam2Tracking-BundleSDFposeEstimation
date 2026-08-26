"""In-process adapter between ROS/OpenCV arrays and BundleSDF."""
from dataclasses import dataclass
from typing import Optional
import os
import sys
import tempfile
import time

import numpy as np

from object_tracker_tracking.preprocessing import BundleSDFInput

@dataclass
class TrackResult:
    T_camera_object: Optional[np.ndarray]   # 4x4, camera frame, or None on failure
    tracking_success: bool
    tracking_quality: float                  # 0-1
    processing_time_ms: float
    reconstruction_status: str               # e.g. "accumulating" | "paused" | "idle"


class _Estimator:
    """Common interface the wrapper drives, whichever backend is active."""

    def initialize(self, rgb, depth, mask, K) -> Optional[np.ndarray]:
        raise NotImplementedError

    def track(self, rgb, depth, mask, K) -> TrackResult:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def export_mesh(self, path: str) -> bool:
        raise NotImplementedError

    def get_reconstruction(self):
        raise NotImplementedError

    def get_reconstruction_progress(self) -> dict:
        return {'state': 'unavailable'}


class InProcessBundleSDFEstimator(_Estimator):
    """Live adapter pinned to the checked-out NVlabs BundleSDF API."""

    def __init__(self, config: dict):
        repo = config.get('bundlesdf_repo_path') or os.environ.get('BUNDLESDF_ROOT')
        if not repo:
            raise RuntimeError('bundlesdf_repo_path (or BUNDLESDF_ROOT) is required.')
        repo = os.path.abspath(repo)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        try:
            from bundlesdf import BundleSdf, my_cpp
            from ruamel.yaml import YAML
        except ImportError as e:
            raise RuntimeError(
                f'BundleSDF could not be imported from {repo}: '
                f'{type(e).__name__}: {e}. Run docker/build_bundle.sh from the host; '
                'that script compiles the native extensions in the GPU container.'
            ) from e
        self._failed_status = my_cpp.Frame.FAIL
        self._session_root = config.get(
            'bundlesdf_session_root', '/tmp/bundlesdf_sessions')
        object_id = int(config.get('object_id', 0))
        os.makedirs(self._session_root, exist_ok=True)
        self._session_dir = tempfile.mkdtemp(
            prefix=f'object_{object_id}_', dir=self._session_root)
        yaml = YAML()

        with open(os.path.join(repo, 'BundleTrack', 'config_ho3d.yml')) as stream:
            track_cfg = yaml.load(stream)
        track_cfg['SPDLOG'] = int(config.get('bundlesdf_debug_level', 0))
        track_cfg['debug_dir'] = self._session_dir + '/'
        track_cfg['depth_processing']['percentile'] = float(
            config.get('depth_percentile', 100.0))
        track_cfg['depth_processing']['zfar'] = float(
            config.get('depth_zfar_m', 3.0))
        track_cfg['depth_processing']['erode']['diff'] = float(
            config.get('depth_erode_diff_m', 0.01))
        track_cfg['depth_processing']['erode']['ratio'] = float(
            config.get('depth_erode_ratio', 1.0))
        track_cfg['depth_processing']['edge_normal_thres'] = float(
            config.get('depth_edge_normal_threshold_deg', 3.0))
        track_cfg['feature_corres']['min_match_with_ref'] = int(
            config.get('min_match_with_ref', 5))
        track_cfg['feature_corres']['resize'] = int(
            config.get('feature_correspondence_resize', 400))
        track_cfg['ransac']['min_match_after_ransac'] = int(
            config.get('min_match_after_ransac', 5))
        track_cfg['ransac']['inlier_dist'] = float(
            config.get('ransac_inlier_dist_m', 0.01))
        track_cfg['ransac']['max_iter'] = int(
            config.get('ransac_max_iterations', 2000))
        track_cfg['ransac']['max_trans_neighbor'] = float(
            config.get('ransac_max_translation_neighbor_m', 0.10))
        track_cfg['ransac']['max_rot_deg_neighbor'] = float(
            config.get('ransac_max_rotation_neighbor_deg', 45.0))
        track_cfg['bundle']['num_iter_outter'] = int(
            config.get('bundle_outer_iterations', 7))
        track_cfg['bundle']['num_iter_inner'] = int(
            config.get('bundle_inner_iterations', 5))
        track_cfg['bundle']['window_size'] = int(
            config.get('bundle_window_size', 4))
        track_cfg['bundle']['max_BA_frames'] = int(
            config.get('bundle_max_ba_frames', 4))
        track_path = os.path.join(self._session_dir, 'config_bundletrack.yml')
        with open(track_path, 'w') as stream:
            yaml.dump(track_cfg, stream)

        with open(os.path.join(repo, 'config.yml')) as stream:
            nerf_cfg = yaml.load(stream)
        self._neural_reconstruction_enabled = bool(
            config.get('neural_reconstruction_enabled', False))
        nerf_cfg['continual'] = True
        nerf_cfg['datadir'] = os.path.join(self._session_dir, 'nerf_online')
        nerf_cfg['save_dir'] = nerf_cfg['datadir']
        nerf_cfg['far'] = track_cfg['depth_processing']['zfar']
        nerf_cfg['sync_max_delay'] = int(
            config.get('nerf_sync_max_delay_keyframes', 10))
        nerf_cfg['n_step'] = int(config.get('nerf_training_steps', 100))
        nerf_cfg['mesh_resolution'] = float(
            config.get('nerf_mesh_resolution_m', 0.01))
        nerf_path = os.path.join(self._session_dir, 'config_nerf.yml')
        with open(nerf_path, 'w') as stream:
            yaml.dump(nerf_cfg, stream)

        self._est = BundleSdf(
            cfg_track_dir=track_path,
            cfg_nerf_dir=nerf_path,
            start_nerf_keyframes=int(config.get('start_nerf_keyframes', 5)),
            enable_nerf=self._neural_reconstruction_enabled,
            use_gui=False,
        )
        self._frame_idx = 0
        self._initialized = False

    def initialize(self, rgb, depth, mask, K):
        self._frame_idx = 0
        pose = self._call(rgb, depth, mask, K)
        self._initialized = pose is not None
        self._frame_idx += 1
        return pose

    def track(self, rgb, depth, mask, K) -> TrackResult:
        t0 = time.time()
        if not self._initialized:
            return TrackResult(None, False, 0.0, 0.0, 'idle')
        pose = self._call(rgb, depth, mask, K)
        self._frame_idx += 1
        dt_ms = (time.time() - t0) * 1000.0
        success = pose is not None
        reconstruction = ('neural_accumulating'
                          if self._neural_reconstruction_enabled
                          else 'external_online_fusion')
        return TrackResult(pose, success, 1.0 if success else 0.0, dt_ms,
                           reconstruction if success else 'track_failed')

    def reset(self) -> None:
        if self._est is not None:
            self._est.on_finish()
            self._est = None
        self._initialized = False
        self._frame_idx = 0

    def export_mesh(self, path: str) -> bool:
        mesh = self.get_reconstruction()
        if mesh is None:
            return False
        mesh.export(path)
        return True

    def get_reconstruction(self):
        if (not self._neural_reconstruction_enabled or
                not self._initialized or self._est is None):
            return None
        if self._est.mesh is not None:
            return self._est.mesh
        with self._est.lock:
            return self._est.p_dict.get('mesh')

    def get_reconstruction_progress(self) -> dict:
        if self._est is None:
            return {'state': 'idle', 'keyframes': 0, 'required_keyframes': 0,
                    'nerf_frames': 0, 'worker_alive': False}
        if not self._neural_reconstruction_enabled:
            return {'state': 'disabled_using_online_tsdf', 'keyframes': 0,
                    'required_keyframes': 0, 'nerf_frames': 0,
                    'worker_alive': False}
        keyframes = len(self._est.bundler._keyframes)
        required = int(self._est.start_nerf_keyframes)
        with self._est.lock:
            running = bool(self._est.p_dict.get('running', False))
            nerf_frames = int(self._est.p_dict.get('nerf_num_frames', 0))
            mesh_ready = 'mesh' in self._est.p_dict
        worker_alive = bool(
            self._est.p_nerf is not None and self._est.p_nerf.is_alive())
        if mesh_ready:
            state = 'available'
        elif not worker_alive:
            state = 'worker_stopped'
        elif keyframes < required:
            state = 'waiting_for_keyframes'
        elif running:
            state = 'training'
        else:
            state = 'queued'
        return {
            'state': state,
            'keyframes': keyframes,
            'required_keyframes': required,
            'nerf_frames': nerf_frames,
            'worker_alive': worker_alive,
        }

    def _call(self, rgb, depth, mask, K):
        if self._est is None:
            return None
        binary_mask = np.ascontiguousarray(mask > 0, dtype=np.uint8)
        self._est.run(
            np.ascontiguousarray(rgb, dtype=np.uint8),
            np.ascontiguousarray(depth, dtype=np.float32),
            np.ascontiguousarray(K, dtype=np.float64),
            f'{self._frame_idx:06d}', mask=binary_mask,
            pose_in_model=np.eye(4, dtype=np.float64),
        )
        frame = self._est.bundler._newframe
        if frame is None or frame._status == self._failed_status:
            return None
        return np.linalg.inv(np.asarray(frame._pose_in_model, dtype=np.float64))


class BundleSDFWrapper:
    """Stable interface to the in-process BundleSDF estimator."""

    def __init__(self, config: Optional[dict] = None):
        self._config = config or {}
        backend = self._config.get('backend', 'inprocess')
        self._estimator = self._make_estimator(backend)
        self._num_frames_processed = 0
        self._last_pose = None

    def _make_estimator(self, backend: str) -> _Estimator:
        if backend == 'inprocess':
            return InProcessBundleSDFEstimator(self._config)
        raise ValueError(f'Unknown BundleSDF backend "{backend}".')

    def initialize(self, inp: BundleSDFInput) -> bool:
        self._num_frames_processed = 0
        T = self._estimator.initialize(inp.rgb, inp.depth, inp.mask, inp.intrinsics)
        self._last_pose = T
        self._num_frames_processed = 1
        return T is not None

    def track(self, inp: BundleSDFInput) -> TrackResult:
        result = self._estimator.track(inp.rgb, inp.depth, inp.mask, inp.intrinsics)
        self._last_pose = result.T_camera_object
        if result.tracking_success:
            self._num_frames_processed += 1
        return result

    def reset(self) -> None:
        self._estimator.reset()
        self._num_frames_processed = 0
        self._last_pose = None

    def export_mesh(self, path: str) -> bool:
        return self._estimator.export_mesh(path)

    def get_reconstruction(self):
        return self._estimator.get_reconstruction()

    def get_reconstruction_progress(self) -> dict:
        return self._estimator.get_reconstruction_progress()


    @property
    def num_frames_processed(self) -> int:
        return self._num_frames_processed

    @property
    def last_pose(self):
        return self._last_pose

    @staticmethod
    def to_ros_pose(T_camera_object: np.ndarray):
        """Convert a 4x4 camera-frame object pose (OpenCV convention) into
        a geometry_msgs Pose. Kept as a static helper here (not in the
        tracking node) so offline tooling can reuse the exact same
        conversion the live node uses.
        """
        from geometry_msgs.msg import Pose
        from scipy.spatial.transform import Rotation

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = T_camera_object[:3, 3]
        quat = Rotation.from_matrix(T_camera_object[:3, :3]).as_quat()  # x,y,z,w
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
        return pose
