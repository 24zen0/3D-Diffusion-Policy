# FIXME: import urchin before others, otherwise segfault, unknown reason
from urchin import URDF, Collision, Sphere, Geometry  # type: ignore  # must stay first (parity with vla_inference.py)

import os  # stdlib for env + paths
import transforms3d as t3d  
# Optional VSCode/pydev remote debugging; mirrors vla_inference entry.
if 'DEBUG_PORT' in os.environ:  # optional remote debugging hook
    import debugpy  # debugger entry
    debugpy.listen(int(os.environ['DEBUG_PORT']))  # bind debugger port
    print('waiting for debugger to attach...')  # notify operator
    debugpy.wait_for_client()  # block until debugger connected

import argparse  # CLI parsing (identical flags to vla_inference.py)
arg_parser = argparse.ArgumentParser()  # build parser instance
# Network / batching knobs kept for protocol compatibility.
arg_parser.add_argument("--port", type=str, required=True)  # ZMQ service port
arg_parser.add_argument("--batching-delay", type=float, default=80)  # ms before forcing batch
arg_parser.add_argument("--batch-size", type=int, default=1)  # max reqs per batch
arg_parser.add_argument("--dataset-statistics", type=str, required=True)  # reserved flag (unused here)
arg_parser.add_argument("--path", type=str, required=True)  # DP3 checkpoint path
arg_parser.add_argument("--openloop", action="store_true")  # open-loop toggle (unused)
arg_parser.add_argument("--top_k", type=int, default=None)  # optional top-k filter (unused)
arg_parser.add_argument("--compile", action="store_true")  # unused; kept for API parity
arg_parser.add_argument("--pc-break-on-empty", action="store_true",
                        help="Raise when point cloud has too few points after cropping.")
arg_parser.add_argument("--pc-min-points", type=int, default=8,
                        help="Minimum acceptable point count when pc-break-on-empty is set.")
arg_parser.add_argument("--pc-verbose", action="store_true",
                        help="Print detailed depth/point statistics for every frame.")
arg_parser.add_argument("--viz-left-window", action="store_true",
                        help="Open a window with left RGB, depth colormap, and live point cloud.")
arg_parser.add_argument("--viz-max-fps", type=float, default=15.0,
                        help="Cap visualization refresh rate to avoid blocking inference.")

import PIL  # image decoding
import io  # bytes buffer
from typing import List, Dict, Tuple, Optional  # type hints
import zmq  # networking
import pickle  # serialization
import time  # timing utilities
import numpy as np  # numerical ops
from tqdm import tqdm  # progress meter
import torch  # ML backend
torch.autograd.set_grad_enabled(False)  # force inference-only mode

import hydra  # config instantiation
import dill  # pickle support for torch.load
import sys
import os
from dataclasses import dataclass
sys.path.append('/mnt/home/zengyitao/DP3baseline/3D-Diffusion-Policy/3D-Diffusion-Policy')

from diffusion_policy_3d.policy.dp3 import DP3  # DP3 policy

# -------------------------------------------------------------------------------------- #
# Visualization helper (lazy imports so missing GUI deps don't crash headless runs)
# -------------------------------------------------------------------------------------- #

class LeftCamVisualizer:
    """Lightweight RGB/depth/point cloud previewer for debugging."""

    def __init__(self, enabled: bool, max_fps: float = 15.0, point_size: float = 2.0):
        self.enabled = enabled
        self.max_fps = max_fps
        self.point_size = point_size
        self.cv2 = None
        self.o3d = None
        self.o3d_vis = None
        self.pcd = None
        self._last_ts = 0.0
        if not enabled:
            return
        try:
            import cv2  # type: ignore
            self.cv2 = cv2
        except Exception as e:  # pragma: no cover
            print(f"[dp3][viz] OpenCV not available, RGB/depth windows disabled: {e}")
        try:
            import open3d as o3d  # type: ignore
            self.o3d = o3d
            self.o3d_vis = o3d.visualization.Visualizer()
            self.o3d_vis.create_window("dp3-left pointcloud", width=640, height=480, visible=True)
            self.pcd = o3d.geometry.PointCloud()
            self.o3d_vis.add_geometry(self.pcd)
            render_opt = self.o3d_vis.get_render_option()
            render_opt.point_size = self.point_size
        except Exception as e:
            print(f"[dp3][viz] Open3D not available, point cloud window disabled: {e}")
            self.o3d = None
            self.o3d_vis = None

    def _throttle(self) -> bool:
        """Return True if we should skip this frame to honor FPS cap."""
        now = time.time()
        if self.max_fps <= 0:
            return False
        min_dt = 1.0 / self.max_fps
        if now - self._last_ts < min_dt:
            return True
        self._last_ts = now
        return False

    def update(self,
               rgb: Optional[np.ndarray],
               depth: Optional[np.ndarray],
               points_xyz: Optional[np.ndarray]):
        if not self.enabled or self._throttle():
            return

        if self.cv2 is not None:
            try:
                if rgb is not None:
                    rgb_to_show = rgb
                    if rgb_to_show.ndim == 3 and rgb_to_show.shape[2] == 3:
                        rgb_to_show = rgb_to_show[:, :, ::-1].copy()  # RGB -> BGR
                    self.cv2.imshow("dp3-left-rgb", rgb_to_show)
                if depth is not None:
                    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                    if d.ndim == 3:
                        d = d.squeeze()
                    if d.size > 0:
                        positive = d[d > 0]
                        scale_ref = np.percentile(positive, 99.0) if positive.size else 1.0
                        if scale_ref <= 0:
                            scale_ref = 1.0
                        depth_norm = np.clip(d / scale_ref, 0, 1)
                        depth_vis = (depth_norm * 255).astype(np.uint8)
                        depth_color = self.cv2.applyColorMap(depth_vis, self.cv2.COLORMAP_TURBO)
                    else:
                        depth_color = np.zeros((*d.shape, 3), dtype=np.uint8)
                    self.cv2.imshow("dp3-left-depth", depth_color)
                if rgb is not None or depth is not None:
                    self.cv2.waitKey(1)
            except Exception as e:
                print(f"[dp3][viz] disabling OpenCV windows due to error: {e}")
                self.cv2 = None

        if self.o3d_vis is not None and points_xyz is not None:
            try:
                pts = points_xyz
                if pts.ndim == 2 and pts.shape[0] > 0:
                    self.pcd.points = self.o3d.utility.Vector3dVector(pts.astype(np.float64))
                else:
                    self.pcd.points = self.o3d.utility.Vector3dVector(np.zeros((1, 3)))
                self.o3d_vis.update_geometry(self.pcd)
                self.o3d_vis.poll_events()
                self.o3d_vis.update_renderer()
            except Exception as e:
                print(f"[dp3][viz] disabling Open3D window due to error: {e}")
                self.o3d_vis = None

# -------------------------------------------------------------------------------------- #
# Point cloud helpers (subset of robosuite_pointcloud_dataset: _crop_workspace, sampling)
# -------------------------------------------------------------------------------------- #

@dataclass
class PcDebugConfig:
    verbose: bool = False
    break_on_empty: bool = False
    min_points: int = 8
    env_id: Optional[int] = None
    frame_idx: Optional[int] = None  # within trajectory

def ensure_agent_pos_dim8(prop: np.ndarray) -> np.ndarray:
    """
    Accepts:
      - 8D: [x,y,z,qw,qx,qy,qz,grip] -> return as-is
      - 7D: [x,y,z,roll,pitch,yaw,grip] -> convert rpy -> quat, output 8D
    """
    prop = np.asarray(prop, dtype=np.float32).reshape(-1)
    if prop.shape[0] == 8:
        return prop.astype(np.float32)

    if prop.shape[0] == 7:
        x, y, z, roll, pitch, yaw, grip = prop.tolist()
        # transforms3d: euler2quat returns (w, x, y, z) by default for given axes
        qw, qx, qy, qz = t3d.euler.euler2quat(roll, pitch, yaw, axes='sxyz')
        return np.array([x, y, z, qw, qx, qy, qz, grip], dtype=np.float32)

    raise ValueError(f"agent_pos must be 7 (xyz+rpy+grip) or 8 (xyz+quat+grip), got {prop.shape[0]}")

def _get_uv_grid(H: int, W: int, cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]):
    """Cached pixel grid to avoid reallocating per frame."""
    key = (H, W)
    if key not in cache:
        u = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
        v = np.arange(H, dtype=np.float32)[:, None].repeat(W, axis=1)
        cache[key] = (u, v)
    return cache[key]

def _depth_to_points(depth_hw: np.ndarray, fx: float, fy: float, cx: float, cy: float, uv_cache: dict) -> np.ndarray:
    """Back-project depth map into XYZ camera frame."""
    H, W = depth_hw.shape
    u, v = _get_uv_grid(H, W, uv_cache)
    m = np.isfinite(depth_hw) & (depth_hw > 0)
    if not np.any(m):
        return np.zeros((0, 3), dtype=np.float32)
    d = depth_hw[m]
    uu = u[m]
    vv = v[m]
    X = d
    Y = -(uu - cx) * d / fx
    Z = -(vv - cy) * d / fy
    return np.stack([X, Y, Z], axis=-1).astype(np.float32)

def _crop_workspace(points_xyz: np.ndarray, bounds: Tuple[float, float, float, float, float, float]) -> np.ndarray:
    """Axis-aligned crop to match training workspace."""
    if points_xyz.shape[0] == 0:
        return points_xyz
    x0, x1, y0, y1, z0, z1 = bounds
    X, Y, Z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    m = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1) & (Z >= z0) & (Z <= z1)
    return points_xyz[m]

def _downsample_random(points_xyz: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random sampling with replacement when N < K."""
    N = points_xyz.shape[0]
    if N == 0:
        return np.zeros((K, 3), dtype=np.float32)
    if N >= K:
        idx = rng.choice(N, size=K, replace=False)
    else:
        idx = rng.choice(N, size=K, replace=True)
    return points_xyz[idx].astype(np.float32)

def depth_to_pointcloud(depth_hw: np.ndarray,
                        bounds: Tuple[float, float, float, float, float, float],
                        n_points: int,
                        uv_cache: dict,
                        rng: np.random.Generator,
                        debug_cfg: Optional[PcDebugConfig] = None) -> np.ndarray:
    """
    Convert depth (H,W) -> point cloud, then crop & downsample.
    Intrinsics: follow robosuite_pointcloud_dataset assumption with fx=fy=W, cx=cy=W/2
    to avoid changing the vla_inference CLI. If your camera intrinsics differ,
    adjust here or encode real intrinsics inside depth frames before sending.
    """
    debug_cfg = debug_cfg or PcDebugConfig()
    H, W = depth_hw.shape
    fx = fy = float(W)
    cx = float(W) / 2.0
    cy = float(H) / 2.0
    valid_mask = np.isfinite(depth_hw) & (depth_hw > 0)
    valid_px = int(valid_mask.sum())
    valid_ratio = valid_px / float(depth_hw.size)
    depth_min = float(np.nanmin(depth_hw[valid_mask])) if valid_px > 0 else None
    depth_max = float(np.nanmax(depth_hw[valid_mask])) if valid_px > 0 else None

    pts = _depth_to_points(depth_hw, fx, fy, cx, cy, uv_cache)
    pts_cropped = _crop_workspace(pts, bounds)

    tag = f"env={debug_cfg.env_id}, frame={debug_cfg.frame_idx}" if (debug_cfg.env_id is not None) else ""
    print(f"[dp3][pc] {tag} raw_pts={pts.shape[0]} cropped_pts={pts_cropped.shape[0]} "
          f"valid_px={valid_px} ({valid_ratio*100:.2f}%) depth_min={depth_min} depth_max={depth_max}")
    if debug_cfg.verbose:
        print(f"[dp3][pc] {tag} H={H} W={W} bounds={bounds}")

    if debug_cfg.break_on_empty and pts_cropped.shape[0] < debug_cfg.min_points:
        raise RuntimeError(
            f"[dp3][pc] insufficient points after crop: {pts_cropped.shape[0]} < {debug_cfg.min_points} | "
            f"valid_px={valid_px} depth_min={depth_min} depth_max={depth_max} H={H} W={W} env={debug_cfg.env_id}")

    pts_down = _downsample_random(pts_cropped, n_points, rng)
    if debug_cfg.verbose:
        print(f"[dp3] after sample N={pts_down.shape[0]} (target {n_points})")
    return pts_down

# ------------------------------ DP3 helpers -------------------------------- #

def load_policy(ckpt_path: str, device: torch.device) -> DP3:
    """Instantiate DP3 + optional EMA weights from checkpoint."""
    payload = torch.load(open(ckpt_path, "rb"), map_location=device, pickle_module=dill)
    cfg = payload["cfg"]
    model: DP3 = hydra.utils.instantiate(cfg.policy)
    ema_model = None
    if cfg.training.use_ema:
        import copy
        try:
            ema_model = copy.deepcopy(model)
        except Exception:
            ema_model = hydra.utils.instantiate(cfg.policy)
    model.to(device)
    if ema_model is not None:
        ema_model.to(device)

    model.load_state_dict(payload["state_dicts"]["model"])
    if cfg.training.use_ema and ema_model is not None:
        ema_model.load_state_dict(payload["state_dicts"]["ema_model"])
        policy = ema_model
    else:
        policy = model

    policy.num_inference_steps = 4  # mirror eval_real_ros responsiveness
    print(f"[dp3] policy loaded from {ckpt_path} | device={device} | n_obs_steps={policy.n_obs_steps} | n_action_steps={policy.n_action_steps}")
    return policy

def build_obs(sample: dict,
              n_obs_steps: int,
              n_points: int,
              bounds: Tuple[float, float, float, float, float, float],
              uv_cache: dict,
              rng: np.random.Generator,
              debug_cfg: Optional[PcDebugConfig] = None,
              visualizer: Optional[LeftCamVisualizer] = None):
    """Select recent frames, build point cloud + proprio windows for DP3."""
    depth_seq = sample.get('depth_array', [])
    proprio_seq = sample.get('proprio_array', [])
    if len(depth_seq) == 0:
        depth_seq = [np.zeros((1, 1), dtype=np.float32)]
    if len(proprio_seq) == 0:
        proprio_seq = [np.zeros((7,), dtype=np.float32)]

    # mimic vla_inference temporal slicing: take most recent frames
    idxs = list(range(-(n_obs_steps), 0))
    idxs = [max(i, -len(depth_seq)) for i in idxs]
    idxs = [len(depth_seq) + i for i in idxs]

    pcs = []
    props = []
    for i in idxs:
        depth = depth_seq[i]
        if depth.ndim == 3:
            depth = depth.squeeze()
        frame_debug_cfg = None
        if debug_cfg is not None:
            frame_debug_cfg = PcDebugConfig(
                verbose=debug_cfg.verbose,
                break_on_empty=debug_cfg.break_on_empty,
                min_points=debug_cfg.min_points,
                env_id=debug_cfg.env_id,
                frame_idx=i,
            )
        pc = depth_to_pointcloud(depth, bounds, n_points, uv_cache, rng, debug_cfg=frame_debug_cfg)
        pcs.append(pc)

        prop = proprio_seq[min(i, len(proprio_seq) - 1)]
        prop = ensure_agent_pos_dim8(prop)
        props.append(prop)

        if visualizer is not None and i == idxs[-1]:
            rgb_seq = sample.get('image_array') or sample.get('image_wrist_array') or []
            rgb_frame = None
            if isinstance(rgb_seq, list) and len(rgb_seq) > 0:
                rgb_frame = rgb_seq[min(i, len(rgb_seq) - 1)]
            visualizer.update(rgb_frame, depth, pc)

    point_cloud = np.stack(pcs, axis=0)
    agent_pos = np.stack(props, axis=0)
    return point_cloud, agent_pos

def batch_process(dp3_model: DP3,
                  batch: List[dict],
                  device: torch.device,
                  n_points: int,
                  bounds: Tuple[float, float, float, float, float, float],
                  uv_cache: dict,
                  debug_cfg: Optional[PcDebugConfig] = None,
                  visualizer: Optional[LeftCamVisualizer] = None):
    """Decode batch payloads, run DP3, return action arrays."""
    print(f"[dp3] batch_process start | batch_size={len(batch)}")
    rng = np.random.default_rng()
    pcs = []
    props = []
    env_ids = []
    for sample in batch:
        sample_debug_cfg = None
        if debug_cfg is not None:
            sample_debug_cfg = PcDebugConfig(
                verbose=debug_cfg.verbose,
                break_on_empty=debug_cfg.break_on_empty,
                min_points=debug_cfg.min_points,
                env_id=sample.get('env_id'),
            )
        pc, prop = build_obs(sample, dp3_model.n_obs_steps, n_points, bounds, uv_cache, rng,
                             debug_cfg=sample_debug_cfg, visualizer=visualizer)
        pcs.append(pc)
        props.append(prop)
        env_ids.append(sample.get('env_id'))

    obs = {
        "point_cloud": torch.from_numpy(np.stack(pcs, axis=0)).to(device),
        "agent_pos": torch.from_numpy(np.stack(props, axis=0)).to(device),
    }

    with torch.no_grad():
        action_dict = dp3_model.predict_action(obs)
    actions = action_dict["action"].cpu().numpy()
    print(f"[dp3] batch_process done | actions_shape={actions.shape}")

    ret = []
    for env_id, act in zip(env_ids, actions):
        ret.append({
            'result': act,
            'env_id': env_id,
            'debug': {},  # keep shape consistent with vla_inference return
        })
    return ret

def decompress_batch(batch: List[dict]):
    """
    Byte decode matches vla_inference: optional image_array / image_wrist_array / depth_array.
    Images are ignored by DP3 but retained for API parity.
    """
    for sample in batch:
        if sample.get('compressed', False):
            for key in ['image_array', 'image_wrist_array']:
                if key in sample:
                    decompressed_image_array = []
                    for compressed_image in sample[key]:
                        decompressed_image_array.append(np.array(PIL.Image.open(io.BytesIO(compressed_image))))
                    sample[key] = decompressed_image_array
            if 'depth_array' in sample:
                decompressed_depth_array = []
                for compressed_depth in sample['depth_array']:
                    decompressed_depth_array.append(
                        np.array(PIL.Image.open(io.BytesIO(compressed_depth))).view(np.float32).squeeze(axis=-1)
                    )
                sample['depth_array'] = decompressed_depth_array
            sample['compressed'] = False

def warmup(dp3_model: DP3,
           device: torch.device,
           n_points,
           bounds,
           uv_cache,
           debug_cfg: Optional[PcDebugConfig] = None,
           visualizer: Optional[LeftCamVisualizer] = None):
    """Run a few dry passes to stabilize CUDA graphs/kernels."""
    SAMPLES = [
        {
            'text': 'dummy',
            'image_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'image_wrist_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'depth_array': [np.zeros((256, 256), dtype=np.float32)],
            'proprio_array': [np.array([0,0,0, 1,0,0,0, 0], dtype=np.float32)] * 4,
            'traj_metadata': None,
            'env_id': 1,
        },
    ]
    NUM_TESTS = 3
    for i in tqdm(range(NUM_TESTS)):
        # Warmup should never abort on empty clouds; disable break_on_empty during warmup.
        warmup_debug_cfg = None
        if debug_cfg is not None:
            warmup_debug_cfg = PcDebugConfig(
                verbose=debug_cfg.verbose,
                break_on_empty=False,
                min_points=debug_cfg.min_points,
            )
        _ = batch_process(dp3_model, [SAMPLES[0]], device, n_points, bounds, uv_cache,
                          debug_cfg=warmup_debug_cfg, visualizer=visualizer)
        print(f"[dp3] warmup {i+1}/{NUM_TESTS} complete")

def main():
    args = arg_parser.parse_args()
    dp3_model = load_policy(args.path, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    device = next(dp3_model.parameters()).device

    bounds = (0.05, 3.0, -0.8, 1.0, -0.25, 2.0)  # default workspace crop
    uv_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    pc_debug_cfg = PcDebugConfig(
        verbose=args.pc_verbose,
        break_on_empty=args.pc_break_on_empty,
        min_points=args.pc_min_points,
    )
    visualizer = LeftCamVisualizer(enabled=args.viz_left_window, max_fps=args.viz_max_fps)

    warmup(dp3_model, device, n_points=1024, bounds=bounds, uv_cache=uv_cache,
           debug_cfg=pc_debug_cfg, visualizer=visualizer)
    print("[dp3] warmup finished, entering serve loop")

    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    print("ZMQ socket type:", socket.getsockopt(zmq.TYPE))
    socket.bind(f"tcp://*:{args.port}")
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    requests = []  # queued (client_id, payload)
    first_arrive_time = None  # timestamp of first queued item

    print('start serving')
    while True:
        current_time = time.time() * 1000  # ms
        if (len(requests) >= args.batch_size or
            (first_arrive_time is not None and
             current_time - first_arrive_time > args.batching_delay and
             len(requests) > 0)):
            data_num = min(args.batch_size, len(requests))  # slice batch size
            client_ids, data_batch = zip(*requests[:data_num])

            decompress_batch(list(data_batch))
            print(f"[dp3] dequeued {data_num} requests, queue_left={len(requests)-data_num}")

            tbegin = time.time()
            results = batch_process(dp3_model, list(data_batch), device, n_points=1024, bounds=bounds, uv_cache=uv_cache)
            tend = time.time()
            print(f'finished {len(requests)} requests in {tend - tbegin:.3f}s')

             # Send responses back to each requester.
            # vla_client expects delta_actions: shape (T, 7)
            #   [dx, dy, dz, droll, dpitch, dyaw, gripper]
            # and requires gripper quantized:
            #   assert action[6] in [-1., 1., 0.]

            GRIPPER_THR = 0.3  # threshold (deadzone). Tune 0.2~0.4 if needed.

            for client_id, result in zip(client_ids, results):
                act_seq = np.asarray(result["result"], dtype=np.float32)

                # DP3 returns (T, 8): [dx,dy,dz, qw,qx,qy,qz, gripper_raw]
                # Convert to (T, 7): [dx,dy,dz, droll,dpitch,dyaw, gripper_quant]
                if act_seq.ndim == 1:
                    act_seq = act_seq[None, :]  # (1, D)

                if act_seq.ndim != 2 or act_seq.shape[1] != 8:
                    raise RuntimeError(f"[dp3] unexpected action shape: {act_seq.shape}, expected (T,8)")

                T = act_seq.shape[0]
                out = np.zeros((T, 7), dtype=np.float32)

                # pos delta
                out[:, 0:3] = act_seq[:, 0:3]

                # quat -> euler (delta rotation)
                # DP3 quat order is (w,x,y,z). transforms3d expects the same for quat2mat.
                for t in range(T):
                    qw, qx, qy, qz = act_seq[t, 3:7].tolist()
                    R = t3d.quaternions.quat2mat([qw, qx, qy, qz])
                    # axes must match your earlier euler2quat(..., axes='sxyz')
                    roll, pitch, yaw = t3d.euler.mat2euler(R, axes='sxyz')
                    out[t, 3:6] = np.array([roll, pitch, yaw], dtype=np.float32)

                # gripper quantize: use DP3 gripper at index 7 -> put into out[:,6]
                g = act_seq[:, 7]
                out[:, 6] = np.where(g > GRIPPER_THR, 1.0,
                            np.where(g < -GRIPPER_THR, -1.0, 0.0)).astype(np.float32)

                socket.send_multipart([
                    client_id,
                    b'',
                    pickle.dumps({
                        "info": "success",
                        "env_id": result["env_id"],
                        "result": out,          # <-- IMPORTANT: send (T,7)
                        "debug": result["debug"],
                    })
                ])


            requests = requests[data_num:]  # drop processed batch
            if len(requests) == 0:
                first_arrive_time = None

        # try getting new sample
        try:
            client_id, empty, data = socket.recv_multipart(zmq.DONTWAIT)
            if len(requests) == 0:
                first_arrive_time = time.time() * 1000

            data = pickle.loads(data)
            requests.append((client_id, data))
            print(f"[dp3] received new request | queue_size={len(requests)}")
        except zmq.Again:
            pass


if __name__ == "__main__":  # entrypoint guard
    main()  # start server
