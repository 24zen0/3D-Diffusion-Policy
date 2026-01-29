"""
Run diffusion-policy inference against a real robot over ROS.

Quick start:
    (robodiff)$ python eval_real_ros(use_this).py -i <ckpt_path> -f 5 --s 16

Safety:
    - Keep the hardware E-stop within reach.
    - Use "S" in the OpenCV window to stop evaluation and regain manual control.
"""

import sys
import time
import math
import copy
from typing import Optional

import click
import cv2
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

# --------------------------------------------------------------------------- #
# Local environment injection (kept to mirror existing deployment layout)
# --------------------------------------------------------------------------- #
# NOTE: These sys.path entries point to the operator’s existing virtualenv,
# controller scripts, and point-cloud utilities. Keep them aligned with your deployment.
# Keep sys.path edits near imports so dependency provenance is obvious.
sys.path.append("/home/dc/mambaforge/envs/robodiff/lib/python3.9/site-packages")
sys.path.append("/home/dc/Desktop/dp_ycw/follow_control/follow1/src/arm_control/scripts/KUKA-Controller")
sys.path.append("/home/dc/Desktop/ARX_dp3_training/src/codebase_of_dp3/3D-Diffusion-Policy/diffusion_policy_3d")
sys.path.append("/home/dc/Desktop/ARX_dp3_training/src/codebase_of_dp3/3D-Diffusion-Policy")
sys.path.append("/home/dc/Desktop/ARX_dp3_training/src/depth-to-pointcloud-for-dp3-deploymentBranch/Sustech_pcGenerate")

# Imports below rely on the path injections above.
from Cloud_Process import preprocess_point_cloud
from Convert_PointCloud import PointCloudGenerator
from arm_control.msg import JointControl, JointInformation
from common.cv2_util import get_image_transform
from common.precise_sleep import precise_wait
from common.pytorch_util import dict_apply
from cv_bridge import CvBridge
from diffusion_policy_3d.policy.dp3 import DP3
from message_filters import ApproximateTimeSynchronizer, Subscriber
from multiprocessing.managers import SharedMemoryManager
from real_world.real_inference_util import get_real_obs_dict
from sensor_msgs.msg import Image
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
import rospy

# --------------------------------------------------------------------------- #
# Globals & constants
# --------------------------------------------------------------------------- #
# CvBridge converts ROS Image messages into NumPy/OpenCV matrices.
bridge = CvBridge()
# Shared memory ring buffer reference populated in main(), consumed in callback().
obs_ring_buffer: Optional[SharedMemoryRingBuffer] = None

# Default checkpoint used when CLI flag is omitted.
DEFAULT_CKPT = (
    "/home/dc/Desktop/ARX_dp3_training/src/codebase_of_dp3/3D-Diffusion-Policy/"
    "data/outputs/lemon_plate-dp3-0903_seed0/checkpoints/latest.ckpt"
)
# Sensor timing and safety windows; keep aligned with camera publishing rate.
VIDEO_CAPTURE_FPS = 30
MAX_OBS_BUFFER_SIZE = 30
DEPTH_TIMEOUT = 10.0
START_DELAY = 1.0
FRAME_LATENCY = 1 / 30

# Register eval resolver so Hydra config snippets in checkpoints still parse.
OmegaConf.register_new_resolver("eval", eval, replace=True)
# More legible printing for debugging arrays in logs.
np.set_printoptions(suppress=True)


# CLI entrypoint built with click for easy flag defaults.
@click.command()
@click.option(
    "--input_path",
    "-i",
    help="Path to checkpoint",
    default=DEFAULT_CKPT,
)
# Control loop frequency defaults conservative for safety on hardware.
@click.option(
    "--frequency",
    "-f",
    default=5,
    type=int,
    help="Control frequency (Hz) for publishing actions.",
)
# Number of future joint setpoints predicted per inference call.
@click.option(
    "--steps_per_inference",
    "-s",
    default=16,
    type=int,
    help="Action horizon per policy inference.",
)
def main(input_path: str, frequency: int, steps_per_inference: int) -> None:
    """
    Run the online inference loop.

    Args:
        input_path: Filesystem path to a trained DP3 checkpoint.
        frequency: Control loop frequency (Hz) for sending joint commands.
        steps_per_inference: Number of predicted action steps per policy call.
    """
    global obs_ring_buffer

    # ------------------------------------------------------------------ #
    # Timing & observation layout
    # ------------------------------------------------------------------ #
    # Control period in seconds derived from chosen frequency.
    dt = 1 / frequency
    # Defines observation keys and shapes expected by the policy/normalizer.
    # The `type` tags are informational; shapes drive encoder construction.
    shape_meta = {
        "obs": {
            "point_cloud": {"type": "point_cloud", "shape": (1024, 3)},
            "agent_pos": {"type": "low_dimx", "shape": (7,)},
        }
    }

    # ------------------------------------------------------------------ #
    # Load checkpoint and build policy
    # ------------------------------------------------------------------ #
    # Instantiate the diffusion policy and move weights to GPU.
    policy = load_policy(input_path)

    # Shared memory buffer used by ROS callbacks to stream observations
    # Number of observation steps the policy needs for conditioning.
    n_obs_steps = policy.n_obs_steps
    # Multiprocessing manager backs the zero-copy shared memory ring buffer.
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    # Pre-allocate zero-copy buffers for depth → point-cloud + joints stream
    obs_ring_buffer = create_ring_buffer(shm_manager)

    # ------------------------------------------------------------------ #
    # ROS wiring: subscribers, publishers, and synchronizer
    # ------------------------------------------------------------------ #
    # Initialize this node before creating pubs/subs.
    rospy.init_node("eval_real_ros")
    # Subscribe to joint states and depth; publish joint commands
    agent_pos_sub = Subscriber("joint_information", JointInformation)
    depth_sub = Subscriber("/depth_camera", Image)
    control_robot = rospy.Publisher("joint_control", JointControl, queue_size=10)
    # Synchronize depth and joint topics with slight tolerance.
    # queue_size big enough to absorb network jitter; slop=0.3s tolerates mild skew.
    ats = ApproximateTimeSynchronizer([agent_pos_sub, depth_sub], queue_size=40, slop=0.3)
    ats.registerCallback(callback)
    # Fixed-rate loop for publishing predicted commands.
    rate = rospy.Rate(frequency)

    # ------------------------------------------------------------------ #
    # Start episode
    # ------------------------------------------------------------------ #
    # Reset any policy buffers (e.g., EMA schedulers) before stepping.
    policy.reset()
    # Add a small delay so ROS streams stabilize before first inference.
    eval_t_start = time.time() + START_DELAY
    t_start = time.monotonic() + START_DELAY
    precise_wait(eval_t_start - FRAME_LATENCY, time_func=time.time)
    print("Started!")
    iter_idx = 0

    # Buffer reused between iterations to reduce allocations.
    last_data = None
    # Message object reused to avoid repeated allocations when publishing.
    right_control = JointControl()

    # ------------------------------------------------------------------ #
    # Inference loop
    # ------------------------------------------------------------------ #
    # Continues until ROS shutdown signal (Ctrl+C or shutdown request).
    while not rospy.is_shutdown():
        print("=== Inference cycle start ===")
        test_t_start = time.perf_counter()
        # Target wall-clock when the last action of this batch should land.
        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt
        print(f"t_cycle_end: {t_cycle_end:.6f} (monotonic)")

        # -------------------------------------------------------------- #
        # Acquire synchronized observations from the buffer
        # -------------------------------------------------------------- #
        # Compute how many raw samples correspond to the policy's obs window.
        k = math.ceil(n_obs_steps * (VIDEO_CAPTURE_FPS / frequency))
        print(f"Need k={k} synchronized samples (n_obs_steps={n_obs_steps}, fps={VIDEO_CAPTURE_FPS}, freq={frequency})")

        # Block until enough depth+joint tuples are available.
        wait_for_buffer(k)

        # Pull the latest k synced samples from shared memory
        # out=last_data reuses existing buffers to reduce allocations.
        last_data = obs_ring_buffer.get_last_k(k=k, out=last_data)
        print(f"Fetched initial batch, buffer count={obs_ring_buffer.count}")
        # Use the newest timestamp as anchor for alignment.
        last_timestamp = last_data["timestamp"][-1]
        # Build target timestamps spaced by control dt, newest first.
        obs_align_timestamps = last_timestamp - (np.arange(n_obs_steps)[::-1] * dt)
        print(f"Last timestamp: {last_timestamp:.6f} | Align targets: {obs_align_timestamps}")

        t0 = time.perf_counter()
        obs_dict = {}
        this_timestamps = last_data["timestamp"]
        this_idxs = []
        # For each desired time, find the nearest sample at or before it.
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps <= t)[0]
            this_idx = is_before_idxs[-1] if len(is_before_idxs) > 0 else 0
            this_idxs.append(this_idx)
        # Slice every key consistently to maintain alignment.
        for key in last_data.keys():
            obs_dict[key] = last_data[key][this_idxs]

        obs_timestamps = obs_dict["timestamp"]
        print(f"Aligned obs_timestamps: {obs_timestamps}")
        print(f"Obs shapes: point_cloud={obs_dict['point_cloud'].shape}, agent_pos={obs_dict['agent_pos'].shape}")
        print(f"Observation gather time: {(time.perf_counter()-t0)*1000:.1f}ms")

        # -------------------------------------------------------------- #
        # Policy inference
        # -------------------------------------------------------------- #
        t1 = time.perf_counter()
        # Disable autograd for speed; inference only.
        with torch.no_grad():
            # Convert raw ROS-friendly dict into policy-friendly np arrays (pads, normalizes).
            obs_dict_np = get_real_obs_dict(env_obs=obs_dict, shape_meta=shape_meta)

            for key, value in obs_dict_np.items():
                # Guard against python lists sneaking through and breaking torch.from_numpy.
                if not isinstance(value, np.ndarray):
                    print(f"ERROR: {key} is not a NumPy array! Type: {type(value)}")
                    obs_dict_np[key] = np.array(value)

            # Move to CUDA and add batch dim
            obs_dict_torch = dict_apply(
                obs_dict_np,
                lambda x: torch.from_numpy(x).unsqueeze(0).to(torch.device("cuda:0")),
            )

            # Policy expects point_cloud without the obs-step dimension.
            obs_dict_torch["point_cloud"] = obs_dict_torch["point_cloud"].squeeze(1)
            # Cast joints to float32 to match network weights.
            obs_dict_torch["agent_pos"] = (
                obs_dict_torch["agent_pos"].squeeze(1).to(dtype=torch.float32)
            )

            print(
                "Torch obs dtypes/shapes:",
                {k: (v.dtype, tuple(v.shape)) for k, v in obs_dict_torch.items()},
            )

            # Run diffusion policy to get a horizon of joint targets.
            action_dict = policy.predict_action(obs_dict_torch)
            print(f"Inference latency: {(time.perf_counter()-t1)*1000:.1f}ms")

        # Bring torch tensors back to CPU numpy for ROS publication.
        np_action_dict = dict_apply(
            action_dict, lambda x: x.detach().to("cpu").numpy()
        )
        # Remove batch dimension; result now shape (T, joints).
        action = np_action_dict["action"].squeeze(0)
        print(f"Raw action shape: {action.shape}, dtype: {action.dtype}, min/max: {action.min():.4f}/{action.max():.4f}")

        # -------------------------------------------------------------- #
        # Execute actions (joint positions) through ROS publisher
        # -------------------------------------------------------------- #
        # Only keep first steps_per_inference rows to match control horizon.
        action = action[:steps_per_inference, :]
        # Timestamp each setpoint relative to last observation.
        action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]
        print(f"Trimmed action shape: {action.shape}")
        print(f"obs_timestamps: {obs_timestamps}")
        print(f"Action target timestamps: {action_timestamps}")

        print("Execute Action!")
        print(f"Action preview (first 3 rows):\n{action[:3]}")
        for item in action:
            t3 = time.perf_counter()
            # Publish each predicted joint position step-by-step
            right_control.joint_pos = item
            control_robot.publish(right_control)
            # Sleep to honor commanded control frequency.
            rate.sleep()
            print(f"Execute latency: {(time.perf_counter()-t3)*1000:.1f}ms | cmd={item}")

        # Sleep remainder of cycle to maintain requested frequency.
        precise_wait(t_cycle_end - FRAME_LATENCY)
        iter_idx += steps_per_inference

        # Monitor achieved loop rate to catch overloads.
        print(
            f"Inference Actual frequency {steps_per_inference/(time.perf_counter() - test_t_start)}"
        )
        print("=== Inference cycle end ===\n")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_policy(ckpt_path: str) -> DP3:
    """
    Load a DP3 policy checkpoint, handling EMA weights when present.
    """
    # Force CUDA device to align with training environment.
    device = torch.device("cuda:0")
    # dill is required because configs may include lambdas/resolvers.
    payload = torch.load(open(ckpt_path, "rb"), map_location="cuda:0", pickle_module=dill)
    cfg = payload["cfg"]

    # Instantiate architecture defined in checkpoint config.
    model: DP3 = hydra.utils.instantiate(cfg.policy)
    ema_model: Optional[DP3] = None

    if cfg.training.use_ema:
        try:
            ema_model = copy.deepcopy(model)
        except Exception:
            # Minkowski engine cannot be deep-copied; rebuild instead.
            ema_model = hydra.utils.instantiate(cfg.policy)

    # Move base and EMA models onto GPU memory.
    model.to(device)
    if ema_model is not None:
        ema_model.to(device)

    # Load trained weights; prefer EMA parameters when available.
    model.load_state_dict(payload["state_dicts"]["model"])
    if cfg.training.use_ema and ema_model is not None:
        ema_model.load_state_dict(payload["state_dicts"]["ema_model"])
        policy = ema_model
    else:
        policy = model

    # Use a short diffusion denoising schedule to cut latency.
    policy.num_inference_steps = 4
    print("Policy loaded successfully.")
    print(f"N_Action_Steps: {policy.n_action_steps}")
    return policy


def create_ring_buffer(shm_manager: SharedMemoryManager) -> SharedMemoryRingBuffer:
    """
    Configure shared memory buffers for streaming point clouds and joint states.
    """
    # Allocate sample arrays to define buffer shapes/dtypes.
    examples = {
        "point_cloud": np.empty(shape=(1024, 3), dtype=np.float32),
        "agent_pos": np.empty(shape=(7,), dtype=np.float64),
        "timestamp": 0.0,
    }
    return SharedMemoryRingBuffer.create_from_examples(
        shm_manager=shm_manager,
        examples=examples,
        # Cap history to avoid unbounded RAM if consumers stall.
        get_max_k=MAX_OBS_BUFFER_SIZE,
        # Time budget prevents blocking producer thread.
        get_time_budget=0.2,
        # Hint desired insert frequency for internal pacing.
        put_desired_frequency=VIDEO_CAPTURE_FPS,
    )


def wait_for_buffer(k: int) -> None:
    """
    Block until at least k observations are available in the ring buffer.
    Shuts the node down if no data arrive before DEPTH_TIMEOUT seconds.
    """
    print("等待ROS数据...")
    start_time = time.time()
    while not rospy.is_shutdown():
        current_count = obs_ring_buffer.count if obs_ring_buffer else 0
        print(f"缓冲区数据: {current_count}/{k}")

        if current_count >= k:
            # Enough samples collected; proceed to inference.
            print("缓冲区已准备好")
            return

        if time.time() - start_time > DEPTH_TIMEOUT:
            print(f"超时：{DEPTH_TIMEOUT}秒内未收到足够数据")
            print("请检查以下话题是否有数据:")
            print("1. /joint_information")
            print("2. /depth_camera")
            rospy.signal_shutdown("Missing required ROS topics")
            return

        # Avoid tight loop; yield to ROS scheduler.
        rospy.sleep(0.2)


def callback(agent_pos, depth_msg):
    """
    ROS callback that converts synchronized joint state + depth image
    into a fused observation and pushes it into the shared ring buffer.
    """
    global obs_ring_buffer
    try:
        # Convert depth image to NumPy in meters
        if depth_msg.encoding == "16UC1":
            depth_cv2 = bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            # 16-bit depth arrives in millimeters; cast to float meters.
            depth_cv2 = depth_cv2.astype(np.float32)
        elif depth_msg.encoding == "32FC1":
            depth_cv2 = bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        else:
            rospy.logerr(f"Unsupported depth encoding: {depth_msg.encoding}")
            return

        # Ensure memory is contiguous for downstream point cloud generator.
        depth_cv2 = np.ascontiguousarray(depth_cv2)

        # Generate point cloud from depth
        t_pc = time.perf_counter()
        point_cloud_data = depth_to_pointcloud(depth_cv2)
        print(f"Generate PC: {(time.perf_counter()-t_pc)*1000:.1f}ms")

        if point_cloud_data is None:
            rospy.logwarn("Point cloud generation returned None; skipping sample.")
            return

        # Package observation
        obs_data = {
            # Incoming ROS JointInformation already contains joint_pos list.
            "agent_pos": np.array(agent_pos.joint_pos, dtype=np.float64),
            # Downsampled point cloud fed directly to PointNet encoder.
            "point_cloud": point_cloud_data.astype(np.float32),
        }

        # Non-blocking push so callback stays lightweight.
        obs_ring_buffer.put(obs_data, wait=False)
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"Callback error: {exc}")
        import traceback

        traceback.print_exc()


def transform(data, video_capture_resolution=(640, 480), obs_image_resolution=(160, 120)):
    """
    Resize and convert BGR images to RGB for downstream consumption.
    """
    # Build deterministic resize + color-space transform (camera res -> model res).
    color_tf = get_image_transform(
        input_res=video_capture_resolution,
        output_res=obs_image_resolution,
        bgr_to_rgb=True,
    )
    return color_tf(data)


def depth_to_pointcloud(depth_image_cv2):
    """
    Convert a depth image (meters) to a downsampled point cloud suitable for DP3.
    """
    pc_generator = PointCloudGenerator(
        img_size=depth_image_cv2.shape[1]  # Use image width for calibration
    )

    if np.all(depth_image_cv2 == 0):
        # Early exit if sensor is unplugged or publishing invalid frames.
        print("警告: 深度图全为零值")
        return None

    # Generate dense point cloud then crop/downsample inside helper.
    points = pc_generator.generateCroppedPointCloud(depth_data=depth_image_cv2)
    if not isinstance(points, np.ndarray) or points.size == 0:
        raise ValueError("生成的点云为空")

    # Uniformly sample to 1024 points expected by the model.
    sampled_points = preprocess_point_cloud(points)
    return sampled_points


if __name__ == "__main__":
    main()
