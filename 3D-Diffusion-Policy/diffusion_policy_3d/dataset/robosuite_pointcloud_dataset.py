"""robosuite_pointcloud_dataset.py (TFDS/RLDS backend)

This replaces the original zarr-based RobosuitePointcloudDataset.

It reads RLDS (TFDS) exported per-UUID directories like:
  /mnt/project/simvla/data/deploy-data/elephant-6w/graspsim_train/<UUID>/dataset_info.json

Output format (sequence sample):
  {
    "obs": {
      "point_cloud": (horizon, K, 3) float32,
      "agent_pos":   (horizon, 8)    float32,   # eef_position(3) + eef_orientation(4) + gripper_proprio(1)
    },
    "action":        (horizon, 8)    float32,   # command_eef_position(3) + command_eef_orientation(4) + gripper_action(1)
  }

Notes
- "online" here means: depth bytes -> depth float32 -> point cloud happens on-the-fly in __getitem__.
- FPS is O(N*K) and can dominate training time. Use downsample_method='random' to first run through.
- TensorFlow + PyTorch DataLoader(num_workers>0) can be unstable on some systems (fork). If you hang/crash,
  set dataloader.num_workers=0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import copy
import os

import numpy as np
import torch

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


# ---------------------------
# TF imports (lazy-safe)
# ---------------------------
try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
except Exception:  # pragma: no cover
    tf = None
    tfds = None


@dataclass(frozen=True)
class EpisodeRef:
    """A reference to one trajectory inside one TFDS builder directory."""
    builder_dir: str
    traj_idx: int
    length: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class EpisodeData:
    depth_seq: np.ndarray                 # (T,) dtype=object (bytes)
    eef_position: np.ndarray              # (T, 3) float32, eef_position
    eef_orientation: np.ndarray           # (T, 4) float32, eef_orientation
    command_eef_position: np.ndarray      # (T, 3) float32, command_eef_position
    command_eef_orientation: np.ndarray   # (T, 4) float32, command_eef_orientation
    gripper_proprio: np.ndarray           # (T, 1) float32
    gripper_action: np.ndarray            # (T, 1) float32

    @property
    def T(self) -> int:
        return int(self.eef_position.shape[0])

    @property
    def action(self) -> np.ndarray:
        # (T,8) command_eef_position(3) + command_eef_orientation(4) + gripper_action(1)
        return np.concatenate([
            self.command_eef_position,
            self.command_eef_orientation,
            self.gripper_action
        ], axis=-1).astype(np.float32)

    @property
    def eef_pos_orientation(self) -> np.ndarray:
        # (T,7) eef_position(3) + eef_orientation(4)
        return np.concatenate([self.eef_position, self.eef_orientation], axis=-1).astype(np.float32)

    @property
    def agent_pos(self) -> np.ndarray:
        # (T,8) eef_position(3) + eef_orientation(4) + gripper_proprio(1)
        return np.concatenate([self.eef_pos_orientation, self.gripper_proprio], axis=-1).astype(np.float32)


def _disable_tf_gpu_if_possible():
    """Prevent TF from grabbing GPU memory (PyTorch will use GPU)."""
    if tf is None:
        return
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


def _list_builder_dirs(rlds_root: str) -> List[str]:
    rlds_root = os.path.expanduser(rlds_root)
    if not os.path.isdir(rlds_root):
        raise FileNotFoundError(f"rlds_root not found: {rlds_root}")

    # rlds_root may itself be a TFDS dir
    if os.path.exists(os.path.join(rlds_root, "dataset_info.json")):
        return [rlds_root]

    subdirs = [os.path.join(rlds_root, d) for d in os.listdir(rlds_root)]
    dirs = [
        d for d in subdirs
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "dataset_info.json"))
    ]
    dirs.sort()
    if len(dirs) == 0:
        raise FileNotFoundError(f"No TFDS dirs (dataset_info.json) under: {rlds_root}")
    return dirs


def _safe_eval_env_config(env_config_bytes: bytes) -> dict:
    """environment_config is a python-literal string that uses array(...)."""
    s = env_config_bytes.decode("utf-8")
    return eval(s, {"array": np.array, "__builtins__": {}})


def _pick_split(builder, prefer: str) -> str:
    splits = list(builder.info.splits.keys())
    if prefer in splits:
        return prefer
    return splits[0]


def _read_one_trajectory_tf(
    builder_dir: str,
    traj_idx: int,
    split_prefer: str,
    read_config: Optional["tfds.ReadConfig"],
):
    """Read one trajectory (episode) from a TFDS directory by index."""
    builder = tfds.builder_from_directory(builder_dir)
    split = _pick_split(builder, split_prefer)
    ds = builder.as_dataset(
        split=split,
        shuffle_files=False,
        read_config=read_config,
        decoders={"steps": tfds.decode.SkipDecoding()},
    )
    for i, tr in enumerate(ds):
        if i == traj_idx:
            return tr
    raise IndexError(f"traj_idx={traj_idx} out of range for {builder_dir}")


def _decode_depth(depth_bytes: bytes) -> np.ndarray:
    """Decode RLDS depth0 bytes -> (H,W) float32."""
    t = tf.io.decode_image(depth_bytes, channels=4, expand_animations=False, dtype=tf.uint8)
    d = tf.bitcast(t, tf.float32).numpy()
    return np.squeeze(d).astype(np.float32)


def _get_uv_grid(H: int, W: int, cache: dict) -> Tuple[np.ndarray, np.ndarray]:
    key = (H, W)
    if key not in cache:
        u = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
        v = np.arange(H, dtype=np.float32)[:, None].repeat(W, axis=1)
        cache[key] = (u, v)
    return cache[key]


def _depth_to_points(
    depth_hw: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    uv_cache: dict,
) -> np.ndarray:
    H, W = depth_hw.shape
    u, v = _get_uv_grid(H, W, uv_cache)

    d = depth_hw
    m = np.isfinite(d) & (d > 0)
    if not np.any(m):
        return np.zeros((0, 3), dtype=np.float32)

    dd = d[m]
    uu = u[m]
    vv = v[m]

    # keep your sign convention
    X = dd
    Y = -(uu - cx) * dd / fx
    Z = -(vv - cy) * dd / fy
    return np.stack([X, Y, Z], axis=-1).astype(np.float32)


def _crop_workspace(points_xyz: np.ndarray, bounds: Tuple[float, float, float, float, float, float]) -> np.ndarray:
    if points_xyz.shape[0] == 0:
        return points_xyz
    x0, x1, y0, y1, z0, z1 = bounds
    X, Y, Z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    m = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1) & (Z >= z0) & (Z <= z1)
    return points_xyz[m]


def _downsample_random(points_xyz: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    N = points_xyz.shape[0]
    if N == 0:
        return np.zeros((K, 3), dtype=np.float32)
    if N >= K:
        idx = rng.choice(N, size=K, replace=False)
    else:
        idx = rng.choice(N, size=K, replace=True)
    return points_xyz[idx].astype(np.float32)


def _downsample_fps(points_xyz: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """Naive FPS on CPU (slow)."""
    N = points_xyz.shape[0]
    if N == 0:
        return np.zeros((K, 3), dtype=np.float32)
    if N <= K:
        idx = rng.integers(0, N, size=(K,))
        return points_xyz[idx].astype(np.float32)

    centroids = np.empty((K,), dtype=np.int64)
    farthest = int(rng.integers(0, N))
    dist = np.full((N,), np.inf, dtype=np.float32)

    for i in range(K):
        centroids[i] = farthest
        c = points_xyz[farthest][None, :]
        d = np.sum((points_xyz - c) ** 2, axis=1)
        dist = np.minimum(dist, d)
        farthest = int(np.argmax(dist))

    return points_xyz[centroids].astype(np.float32)


def _pad_window(seq: np.ndarray, start: int, horizon: int) -> np.ndarray:
    """Pad by repeating boundary values (same behavior as clamp indexing)."""
    T = int(seq.shape[0])
    idxs = np.arange(start, start + horizon)
    idxs = np.clip(idxs, 0, T - 1)
    return seq[idxs]


class RobosuitePointcloudDataset(BaseDataset):
    def __init__(
        self,
        rlds_root: str,
        horizon: int = 4,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = 100, #fake

        split_prefer: str = "train",
        # keys
        depth_key: str = "depth0",
        camera_index: int = 0,
        eef_position_key: str = "eef_position",
        eef_orientation_key: str = "eef_orientation",
        command_eef_position_key: str = "command_eef_position",
        command_eef_orientation_key: str = "command_eef_orientation",
        gripper_proprio_key: str = "gripper_proprio",
        gripper_action_key: str = "gripper_action",
        # point cloud
        n_points: int = 1024,
        workspace_bounds: Tuple[float, float, float, float, float, float] = (0.05, 3.0, -0.8, 1.0, -0.25, 2.0),
        downsample_method: str = "fps",  # 'fps' | 'random'
        # DP3 uses only first n_obs_steps; we still output horizon-length obs
        n_obs_steps: int = 2,
        # caching
        cache_size: int = 2,
        # TFDS read config
        interleave_cycle_length: int = 48,
    ):
        super().__init__()

        if tf is None or tfds is None:
            raise ImportError("tensorflow + tensorflow_datasets are required for TFDS RLDS reading.")

        _disable_tf_gpu_if_possible()

        self.rlds_root = os.path.expanduser(rlds_root)
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.seed = int(seed)
        self.val_ratio = float(val_ratio)
        self.max_train_episodes = max_train_episodes
        self.split_prefer = str(split_prefer)

        self.depth_key = depth_key
        self.camera_index = int(camera_index)
        self.eef_position_key = eef_position_key
        self.eef_orientation_key = eef_orientation_key
        self.command_eef_position_key = command_eef_position_key
        self.command_eef_orientation_key = command_eef_orientation_key
        self.gripper_proprio_key = gripper_proprio_key
        self.gripper_action_key = gripper_action_key

        self.n_points = int(n_points)
        self.workspace_bounds = workspace_bounds
        self.downsample_method = str(downsample_method)
        self.n_obs_steps = int(n_obs_steps)
        self.cache_size = int(cache_size)

        self.read_config = tfds.ReadConfig(
            skip_prefetch=True,
            num_parallel_calls_for_interleave_files=tf.data.AUTOTUNE,
            interleave_cycle_length=int(interleave_cycle_length),
        )

        # uv grid cache for depth backprojection
        self._uv_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

        # episode data LRU cache
        self._episode_cache: Dict[Tuple[str, int], EpisodeData] = {}
        self._episode_cache_order: List[Tuple[str, int]] = []

        # discover TFDS dirs
        self.builder_dirs = _list_builder_dirs(self.rlds_root)

        # scan episodes (lightweight: do NOT tfds.as_numpy(steps) here)
        self.episodes: List[EpisodeRef] = []
        total_episodes_added = 0
        count=0

        for bdir in self.builder_dirs:
            builder = tfds.builder_from_directory(bdir)
            split = _pick_split(builder, self.split_prefer)
            ds = builder.as_dataset(
                split=split,
                shuffle_files=False,
                read_config=self.read_config,
                decoders={"steps": tfds.decode.SkipDecoding()},
            )
            print(f"{count}th builder_dir: {bdir}, split: {split}")
            count+=1
            for traj_idx, tr in enumerate(ds):
                jp = tr["steps"]["action"][self.eef_position_key]
                T = jp.shape[0]
                if T is None:
                    T = int(tf.shape(jp)[0].numpy())
                else:
                    T = int(T)

                env_cfg = _safe_eval_env_config(tr["environment_config"].numpy())
                cam = env_cfg["camera_info"][self.camera_index]
                fx, fy, cx, cy = map(float, (cam["fx"], cam["fy"], cam["cx"], cam["cy"]))

                self.episodes.append(EpisodeRef(
                    builder_dir=bdir,
                    traj_idx=int(traj_idx),
                    length=T,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                ))

                total_episodes_added += 1
                print(f"{traj_idx}: Discovered episode: builder_dir={bdir}, traj_idx={traj_idx}, length={T}")

                # Early termination if we've reached the max_train_episodes limit
                if self.max_train_episodes is not None and total_episodes_added >= self.max_train_episodes:
                    print(f"Early termination: Reached max_train_episodes limit ({self.max_train_episodes}) during dataset initialization.")
                    break
            print(f"out {count} times")
            # Break outer loop as well if we've reached the limit
            if self.max_train_episodes is not None and total_episodes_added >= self.max_train_episodes:
                break

        if len(self.episodes) == 0:
            raise RuntimeError(f"No episodes found under {self.rlds_root}")

        # ---------------------------
        # train/val split WITHOUT sampler.py (no zarr, no numba)
        # ---------------------------
        n = len(self.episodes)

        if self.val_ratio is None or self.val_ratio <= 0:
            self.val_mask = np.zeros((n,), dtype=bool)
            self.train_mask = np.ones((n,), dtype=bool)
        else:
            rng = np.random.default_rng(self.seed)
            idx = np.arange(n)
            rng.shuffle(idx)
            n_val = int(round(n * float(self.val_ratio)))
            val_ids = set(idx[:n_val].tolist())
            self.val_mask = np.array([i in val_ids for i in range(n)], dtype=bool)
            self.train_mask = ~self.val_mask

        # optional downsample train episodes
        if self.max_train_episodes is not None:
            train_ids = np.nonzero(self.train_mask)[0]
            if len(train_ids) > int(self.max_train_episodes):
                rng = np.random.default_rng(self.seed)
                keep = rng.choice(train_ids, size=int(self.max_train_episodes), replace=False)
                m = np.zeros((n,), dtype=bool)
                m[keep] = True
                self.train_mask = m

        # build active episode list + prefix sums for indexing
        self._build_active_index(mask=self.train_mask)

    # ---------------------------
    # indexing
    # ---------------------------

    def _num_windows_for_episode(self, T: int) -> int:
        L = self.horizon
        pb = min(max(self.pad_before, 0), L - 1)
        pa = min(max(self.pad_after, 0), L - 1)
        min_start = -pb
        max_start = T - L + pa
        n = max(0, max_start - min_start + 1)
        return int(n)

    def _build_active_index(self, mask: np.ndarray):
        self.active_episode_ids: List[int] = [i for i, m in enumerate(mask) if bool(m)]
        self.active_windows: List[int] = []
        for epi_id in self.active_episode_ids:
            T = self.episodes[epi_id].length
            self.active_windows.append(self._num_windows_for_episode(T))

        # prefix sums for binary search
        self.prefix = np.zeros((len(self.active_windows) + 1,), dtype=np.int64)
        np.cumsum(np.array(self.active_windows, dtype=np.int64), out=self.prefix[1:])

    def __len__(self) -> int:
        return int(self.prefix[-1])

    def _locate(self, idx: int) -> Tuple[int, int, int]:
        """Map global idx -> (episode_id, local_window_idx, window_start)."""
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        ep_pos = int(np.searchsorted(self.prefix, idx, side="right") - 1)
        local = int(idx - self.prefix[ep_pos])
        epi_id = int(self.active_episode_ids[ep_pos])

        pb = min(max(self.pad_before, 0), self.horizon - 1)
        start = local - pb
        return epi_id, local, start

    # ---------------------------
    # dataset API
    # ---------------------------

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set._episode_cache = {}
        val_set._episode_cache_order = []
        val_set._uv_cache = {}

        # validation uses val_mask (can be empty if val_ratio<=0)
        val_set._build_active_index(mask=val_set.val_mask)
        return val_set

    def _cache_put(self, key: Tuple[str, int], value: EpisodeData):
        self._episode_cache[key] = value
        self._episode_cache_order.append(key)
        if len(self._episode_cache_order) > self.cache_size:
            old = self._episode_cache_order.pop(0)
            self._episode_cache.pop(old, None)

    def _load_episode_data(self, epi: EpisodeRef) -> EpisodeData:
        key = (epi.builder_dir, epi.traj_idx)
        if key in self._episode_cache:
            return self._episode_cache[key]

        tr = _read_one_trajectory_tf(
            builder_dir=epi.builder_dir,
            traj_idx=epi.traj_idx,
            split_prefer=self.split_prefer,
            read_config=self.read_config,
        )

        steps = tfds.as_numpy(tr["steps"])
        obs = steps["observation"]
        act = steps["action"]

        depth_seq = obs[self.depth_key]
        eef_position = act[self.eef_position_key].astype(np.float32)
        eef_orientation = act[self.eef_orientation_key].astype(np.float32)
        cmd_eef_position = act[self.command_eef_position_key].astype(np.float32)
        cmd_eef_orientation = act[self.command_eef_orientation_key].astype(np.float32)
        gripper_proprio = act[self.gripper_proprio_key].astype(np.float32)
        gripper_action = act[self.gripper_action_key].astype(np.float32)

        if gripper_proprio.ndim == 1:
            gripper_proprio = gripper_proprio[:, None]
        if gripper_action.ndim == 1:
            gripper_action = gripper_action[:, None]

        data = EpisodeData(
            depth_seq=depth_seq,
            eef_position=eef_position,
            eef_orientation=eef_orientation,
            command_eef_position=cmd_eef_position,
            command_eef_orientation=cmd_eef_orientation,
            gripper_proprio=gripper_proprio,
            gripper_action=gripper_action,
        )
        self._cache_put(key, data)
        return data

    def _downsample(self, points_xyz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.downsample_method == "random":
            return _downsample_random(points_xyz, self.n_points, rng)
        elif self.downsample_method == "fps":
            return _downsample_fps(points_xyz, self.n_points, rng)
        else:
            raise ValueError(f"downsample_method must be 'random' or 'fps', got {self.downsample_method}")

    def get_normalizer(self, mode: str = "limits", max_fit_episodes: int = 64, **kwargs):
        """Fit normalizer on agent_pos/action only; point_cloud uses identity."""
        rng = np.random.default_rng(self.seed)
        epi_ids = np.nonzero(self.train_mask)[0]
        if len(epi_ids) == 0:
            epi_ids = np.arange(len(self.episodes))
        if len(epi_ids) > max_fit_episodes:
            epi_ids = rng.choice(epi_ids, size=max_fit_episodes, replace=False)

        agent_list = []
        action_list = []
        for epi_id in epi_ids:
            epi = self.episodes[int(epi_id)]
            ed = self._load_episode_data(epi)
            agent_list.append(ed.agent_pos)
            action_list.append(ed.action)

        agent = np.concatenate(agent_list, axis=0)
        action = np.concatenate(action_list, axis=0)

        normalizer = LinearNormalizer()
        normalizer.fit(
            data={
                "agent_pos": agent,
                "action": action,
            },
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )

        # point cloud: identity (avoid scanning huge point clouds)
        normalizer["point_cloud"] = SingleFieldLinearNormalizer.create_identity()
        return normalizer
    
#data loading:
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        epi_id, _local, start = self._locate(idx)
        epi = self.episodes[epi_id]
        ed = self._load_episode_data(epi)

        rng = np.random.default_rng(self.seed + int(idx))

        agent_pos_win = _pad_window(ed.agent_pos, start, self.horizon)  # (H,8)
        action_win = _pad_window(ed.action, start, self.horizon)        # (H,8)

        pcd_seq = np.zeros((self.horizon, self.n_points, 3), dtype=np.float32)

        # only compute first n_obs_steps, then repeat last
        To = min(self.n_obs_steps, self.horizon)
        for i in range(To):
            src_t = int(np.clip(start + i, 0, ed.T - 1))
            depth_hw = _decode_depth(ed.depth_seq[src_t])
            pts = _depth_to_points(depth_hw, epi.fx, epi.fy, epi.cx, epi.cy, self._uv_cache)
            pts = _crop_workspace(pts, self.workspace_bounds)
            pcd_seq[i] = self._downsample(pts, rng)

        if To < self.horizon:
            pcd_seq[To:] = pcd_seq[max(To - 1, 0)]

        data = {
            "obs": {
                "point_cloud": pcd_seq,
                "agent_pos": agent_pos_win,
            },
            "action": action_win,
        }
        return dict_apply(data, torch.from_numpy)
