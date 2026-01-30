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

import PIL  # image decoding
import io  # bytes buffer
from typing import List, Dict, Tuple  # type hints
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
sys.path.append('/mnt/home/zengyitao/DP3baseline/3D-Diffusion-Policy/3D-Diffusion-Policy')

from diffusion_policy_3d.policy.dp3 import DP3  # DP3 policy

# -------------------------------------------------------------------------------------- #
# Point cloud helpers (subset of robosuite_pointcloud_dataset: _crop_workspace, sampling)
# -------------------------------------------------------------------------------------- #

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

def _discretize_gripper(val: float) -> float:
    """Quantize continuous gripper logits into {-1,0,1} expected by grasp_mode."""
    if val < -0.5:
        return -1.0
    if val > 0.5:
        return 1.0
    return 0.0

def _normalize_quat(q: np.ndarray) -> np.ndarray:
    """Safe quaternion normalization; returns identity when norm is tiny."""
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q)
    if n < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q / n

def abs_actions_to_delta(actions_abs: np.ndarray, start_pose: np.ndarray) -> np.ndarray:
    """
    Convert absolute actions (xyz + quat + gripper) into deltas relative to running state.
    Output shape matches grasp_mode expectation: (dx, dy, dz, droll, dpitch, dyaw, grip).
    """
    actions_abs = np.asarray(actions_abs, dtype=np.float32)
    if actions_abs.ndim == 1:
        actions_abs = actions_abs[None, :]

    # normalize start pose to quat form
    curr_pose = ensure_agent_pos_dim8(start_pose)
    curr_pos = curr_pose[:3].astype(np.float32)
    curr_quat = _normalize_quat(curr_pose[3:7])

    deltas = []
    for step in actions_abs:
        if step.shape[-1] == 8:
            tgt_pos = step[:3].astype(np.float32)
            tgt_quat = _normalize_quat(step[3:7])
            grip_raw = float(step[7])
        elif step.shape[-1] == 7:
            tgt_pos = step[:3].astype(np.float32)
            tgt_quat = t3d.quaternions.mat2quat(t3d.euler.euler2mat(*step[3:6])).astype(np.float32)
            tgt_quat = _normalize_quat(tgt_quat)
            grip_raw = float(step[6])
        else:
            raise ValueError(f"Unsupported action dimension {step.shape[-1]} (expected 7 or 8)")

        delta_pos = tgt_pos - curr_pos
        delta_R = t3d.quaternions.quat2mat(tgt_quat) @ t3d.quaternions.quat2mat(curr_quat).T
        delta_euler = t3d.euler.mat2euler(delta_R)
        grip = _discretize_gripper(grip_raw)

        deltas.append(np.array([delta_pos[0], delta_pos[1], delta_pos[2],
                                delta_euler[0], delta_euler[1], delta_euler[2],
                                grip], dtype=np.float32))

        curr_pos = tgt_pos
        curr_quat = tgt_quat

    return np.stack(deltas, axis=0)

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
                        rng: np.random.Generator) -> np.ndarray:
    """
    Convert depth (H,W) -> point cloud, then crop & downsample.
    Intrinsics: follow robosuite_pointcloud_dataset assumption with fx=fy=W, cx=cy=W/2
    to avoid changing the vla_inference CLI. If your camera intrinsics differ,
    adjust here or encode real intrinsics inside depth frames before sending.
    """
    H, W = depth_hw.shape
    fx = fy = float(W)
    cx = float(W) / 2.0
    cy = float(H) / 2.0
    pts = _depth_to_points(depth_hw, fx, fy, cx, cy, uv_cache)
    # Debug: monitor point count before/after cropping to catch bad depth or bounds.
    print(f"[dp3] depth->points raw N={pts.shape[0]} H={H} W={W}")
    pts = _crop_workspace(pts, bounds)
    print(f"[dp3] after crop N={pts.shape[0]} bounds={bounds}")
    pts = _downsample_random(pts, n_points, rng)
    print(f"[dp3] after sample N={pts.shape[0]} (target {n_points})")
    return pts

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
              rng: np.random.Generator):
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
        pc = depth_to_pointcloud(depth, bounds, n_points, uv_cache, rng)
        pcs.append(pc)

        prop = proprio_seq[min(i, len(proprio_seq) - 1)]
        prop = ensure_agent_pos_dim8(prop)
        props.append(prop)

    point_cloud = np.stack(pcs, axis=0)
    agent_pos = np.stack(props, axis=0)
    return point_cloud, agent_pos

def batch_process(dp3_model: DP3,
                  batch: List[dict],
                  device: torch.device,
                  n_points: int,
                  bounds: Tuple[float, float, float, float, float, float],
                  uv_cache: dict):
    """Decode batch payloads, run DP3, return action arrays."""
    print(f"[dp3] batch_process start | batch_size={len(batch)}")
    rng = np.random.default_rng()
    pcs = []
    props = []
    env_ids = []
    start_poses = []
    for sample in batch:
        pc, prop = build_obs(sample, dp3_model.n_obs_steps, n_points, bounds, uv_cache, rng)
        pcs.append(pc)
        props.append(prop)
        env_ids.append(sample.get('env_id'))
        proprio_seq = sample.get('proprio_array', [])
        start_pose = proprio_seq[-1] if isinstance(proprio_seq, list) and len(proprio_seq) > 0 else np.zeros((8,), dtype=np.float32)
        start_pose = ensure_agent_pos_dim8(start_pose)
        start_poses.append(start_pose)

    obs = {
        "point_cloud": torch.from_numpy(np.stack(pcs, axis=0)).to(device),
        "agent_pos": torch.from_numpy(np.stack(props, axis=0)).to(device),
    }

    with torch.no_grad():
        action_dict = dp3_model.predict_action(obs)
    actions = action_dict["action"].cpu().numpy()
    print(f"[dp3] batch_process done | actions_shape={actions.shape}")

    ret = []
    for env_id, act, start_pose in zip(env_ids, actions, start_poses):
        delta_act = abs_actions_to_delta(act, start_pose)
        print(f"[dp3] abs->delta converted | env={env_id} abs_shape={act.shape} delta_shape={delta_act.shape}")
        ret.append({
            'result': delta_act,
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

def warmup(dp3_model: DP3, device: torch.device, n_points, bounds, uv_cache):
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
        _ = batch_process(dp3_model, [SAMPLES[0]], device, n_points, bounds, uv_cache)
        print(f"[dp3] warmup {i+1}/{NUM_TESTS} complete")

def main():
    args = arg_parser.parse_args()
    dp3_model = load_policy(args.path, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    device = next(dp3_model.parameters()).device

    bounds = (0.05, 3.0, -0.8, 1.0, -0.25, 2.0)  # default workspace crop
    uv_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    warmup(dp3_model, device, n_points=1024, bounds=bounds, uv_cache=uv_cache)
    print("[dp3] warmup finished, entering serve loop")

    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(f"tcp://*:{args.port}")

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
            for client_id, result in zip(client_ids, results):
                socket.send_multipart([
                    client_id,
                    b'',
                    pickle.dumps({
                        'info': 'success',
                        'env_id': result['env_id'],
                        'result': result['result'],
                        'debug': result['debug'],
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

