
"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -f <frequency> --s <steps_per_inference>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

import sys

sys.path.append("/home/dc/mambaforge/envs/robodiff/lib/python3.9/site-packages")
sys.path.append("/home/dc/Desktop/dp_ycw/follow_control/follow1/src/arm_control/scripts/KUKA-Controller")
sys.path.append("/home/dc/Desktop/ARX_dp3_training/src/codebase_of_dp3/3D-Diffusion-Policy/diffusion_policy_3d")
sys.path.append("/home/dc/Desktop/ARX_dp3_training/src/codebase_of_dp3/3D-Diffusion-Policy")
#for depth to pointcloud
sys.path.append('/home/dc/Desktop/ARX_dp3_training/src/depth-to-pointcloud-for-dp3-deploymentBranch/Sustech_pcGenerate')
from Cloud_Process import farthest_point_sampling,preprocess_point_cloud
from Convert_PointCloud import PointCloudGenerator


import time
import math
import copy
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
from omegaconf import OmegaConf,DictConfig
import scipy.spatial.transform as st


from common.precise_sleep import precise_wait
from real_world.real_inference_util import get_real_obs_resolution, get_real_obs_dict

from common.pytorch_util import dict_apply
from workspace.base_workspace import BaseWorkspace
from common.cv2_util import get_image_transform

from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from multiprocessing.managers import SharedMemoryManager
import rospy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from arm_control.msg import JointInformation
from arm_control.msg import JointControl
from arm_control.msg import PosCmd
from sensor_msgs.msg import Image
from threading import Lock
from cv_bridge import CvBridge,CvBridgeError
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
import visualizer
import open3d as o3d



bridge = CvBridge()


OmegaConf.register_new_resolver("eval", eval, replace=True)
np.set_printoptions(suppress=True)

@click.command()
@click.option(
    "--input_path",
    "-i",
    help="Path to checkpoint",
    default="/home/dc/Desktop/ARX_dp3_training/src/codebase_of_dp3/3D-Diffusion-Policy/data/outputs/lemon_plate-dp3-0903_seed0/checkpoints/latest.ckpt"
)
@click.option(
    "--frequency",
    "-f",
    default=5,
    type=int,
    help="control frequency",
)
@click.option(
    "--steps_per_inference",
    "-s",
    default=16,
    type=int,
    help="Action horizon for inference.",
)

# @profile
def main(
    input_path,
    frequency,
    steps_per_inference,
):
    global obs_ring_buffer

    dt = 1 / frequency
    video_capture_fps = 30
    max_obs_buffer_size = 30
    shape_meta = {
        "obs":{
            "point_cloud":{
            "type": "point_cloud",
            "shape": (1024, 3),
            },
        
            "agent_pos":{
                "type": "low_dimx",
                "shape": (7,),
            }
        }
    }

    # load checkpoint and policy
    ckpt_path = input_path
    device = torch.device("cuda:0")
    payload = torch.load(open(ckpt_path, "rb"), map_location="cuda:0", pickle_module=dill)
    cfg = payload["cfg"]
    model: DP3 = hydra.utils.instantiate(cfg.policy)
    ema_model: DP3 = None
    if cfg.training.use_ema:
        try:
            ema_model = copy.deepcopy(model)
        except: # minkowski engine could not be copied. recreate it
            ema_model = hydra.utils.instantiate(cfg.policy)
    model.to(device)
    if ema_model is not None:
        ema_model.to(device)

    if cfg.training.use_ema:
        ema = hydra.utils.instantiate(cfg.ema,model=ema_model)

    model.load_state_dict(payload['state_dicts']['model'])
    if cfg.training.use_ema:
        ema_model.load_state_dict(payload['state_dicts']['ema_model'])
    
    if cfg.training.use_ema:
        policy = ema_model
    else:
        policy = model

    policy.num_inference_steps = 4
    print("Policy loaded successfully.")
    print(f"N_Action_Steps:{policy.n_action_steps}")

    n_obs_steps = policy.n_obs_steps
    shm_manager = SharedMemoryManager()
    shm_manager.start()

    examples = dict()
    point_cloud_shape_from_cfg = (1024,3)
    examples["point_cloud"] = np.empty(shape=point_cloud_shape_from_cfg, dtype=np.float32)
    examples["agent_pos"] = np.empty(shape=(7,), dtype=np.float64)
    examples["timestamp"] = 0.0
    obs_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
        shm_manager=shm_manager,
        examples=examples,
        get_max_k=max_obs_buffer_size,
        get_time_budget=0.2,
        put_desired_frequency=video_capture_fps,
    )

    # ros config
    rospy.init_node("eval_real_ros")
    agent_pos = Subscriber("joint_information", JointInformation)
    depth = Subscriber("/depth_camera", Image)                             
    control_robot2 = rospy.Publisher("joint_control", JointControl, queue_size=10)
    ats = ApproximateTimeSynchronizer(
        [agent_pos, depth], queue_size=40, slop=0.3
    )
    ats.registerCallback(callback)
    rate = rospy.Rate(frequency)

    # data
    last_data = None
    right_control = JointControl()
    

    # start episode

    # print("等待缓冲区填充...")
    # print("活动话题列表:")
    # print(rospy.get_published_topics())
    # while not rospy.is_shutdown():
        
    #     current_count = obs_ring_buffer.count
    #     # print(current_count)
    #     if current_count >= 6:
    #         break
    #     # print(f"当前数据量: {current_count}/{6}，等待中...")
    # time.sleep(0.1)
    policy.reset()
    start_delay = 1.0
    eval_t_start = time.time() + start_delay
    t_start = time.monotonic() + start_delay
    frame_latency = 1/30
    precise_wait(eval_t_start - frame_latency, time_func=time.time)
    print("Started!")
    iter_idx = 0
    start_time = time.time()

    # inference loop
    while not rospy.is_shutdown():
        print("enter while2")
        test_t_start = time.perf_counter()
        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt
        print(f"t_cycle_end:{t_cycle_end}")

        # get observation
        k = math.ceil(n_obs_steps * (video_capture_fps / frequency))
        print(f"需要 k={k} 个数据点")

        # 等待缓冲区填充
        print("等待ROS数据...")
        timeout = 10.0
        start_time = time.time()
        while not rospy.is_shutdown():
            current_count = obs_ring_buffer.count
            print(f"缓冲区数据: {current_count}/{k}")
            
            if current_count >= k:
                print("缓冲区已准备好")
                break
                
            if time.time() - start_time > timeout:
                print(f"超时：{timeout}秒内未收到足够数据")
                print("请检查以下话题是否有数据:")
                print("1. /joint_information")
                print("2. /depth_camera")
                return
                
            rospy.sleep(0.2)

        # 获取初始数据
        last_data = obs_ring_buffer.get_last_k(k=k, out=last_data)
        print(f"成功获取初始数据，count={obs_ring_buffer.count}")
        last_timestamp = last_data["timestamp"][-1]
        obs_align_timestamps = last_timestamp - (np.arange(n_obs_steps)[::-1] * dt)

        t0 = time.perf_counter()
        obs_dict = dict()
        this_timestamps = last_data["timestamp"]
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps <= t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)
        for key in last_data.keys():
            obs_dict[key] = last_data[key][this_idxs]

        obs_timestamps = obs_dict["timestamp"]
        print(f"obs_timestamps1:{obs_timestamps}")
        print("Got Observation!")
        print(f"Observation: {(time.perf_counter()-t0)*1000:.1f}ms")


        # run inference
        t1 = time.perf_counter()
        with torch.no_grad():
            obs_dict_np = get_real_obs_dict(
                env_obs=obs_dict, shape_meta=shape_meta)

            for key, value in obs_dict_np.items():
                if not isinstance(value, np.ndarray):
                    print(f"ERROR: {key} is not a NumPy array! Type: {type(value)}")
                    # 强制转换（临时修复）
                    obs_dict_np[key] = np.array(value)

            obs_dict = dict_apply(obs_dict_np,
                lambda x: torch.from_numpy(x).unsqueeze(0).to(torch.device("cuda:0")))

            obs_dict["point_cloud"] = obs_dict["point_cloud"].squeeze(1)  # 形状变为 [1, 1024, 3]

            # 修正 agent_pos 维度
            obs_dict["agent_pos"] = obs_dict["agent_pos"].squeeze(1)  # 形状变为 [2, 7]
            obs_dict["agent_pos"] = obs_dict["agent_pos"].to(dtype=torch.float32)
            # 示例输出: {'point_cloud': (1024, 3), 'agent_pos': (7,)}
            action_dict = policy.predict_action(obs_dict)
            print(f"Inference: {(time.perf_counter()-t1)*1000:.1f}ms")

        np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
        action = np_action_dict['action'].squeeze(0)

        # preprocess action
        t2 = time.perf_counter()
        action = action[:steps_per_inference, :]
        print(f"obs_timestamps:{obs_timestamps}")
        # print(f"Pre Action:{action}")
        action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]
        print(f"timestamps:{action_timestamps}")

        action_exec_latency = 0.01
        curr_time = time.time()
        is_new = action_timestamps > (curr_time + action_exec_latency)
        # print(f"is_new:{is_new}")

        # if np.sum(is_new) == 0:
        #     print("1111111111")
        #     action = action[[-1]]
        #     next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
        #     action_timestamp = eval_t_start + (next_step_idx) * dt
        #     print("Over budget", action_timestamp - curr_time)
        #     action_timestamps = np.array([action_timestamp])
        # else:
        #     print("22222222222")
        #     action = action[is_new]
        #     action_timestamps = action_timestamps[is_new]
        # print(f"Preprocess: {(time.perf_counter()-t2)*1000:.1f}ms")
        # execute actions
        print("Execute Action!")
        print(f"Action:{action}")
        for item in action:
            t3 = time.perf_counter()
            right_control.joint_pos = item
            control_robot2.publish(right_control)
            rate.sleep()
            print(f"Execute: {(time.perf_counter()-t3)*1000:.1f}ms")
            
        
        precise_wait(t_cycle_end - frame_latency)
        iter_idx += steps_per_inference
        

        print(f"Inference Actual frequency {steps_per_inference/(time.perf_counter() - test_t_start)}")


def debug_callback(msg):
    print(f"收到 {msg._type} 消息! 长度: {len(str(msg))}")

global success_count , fail_count
success_count = 0
fail_count = 0

def callback(agent_pos, depth_msg):
    global obs_ring_buffer
    global success_count , fail_count
    try:
        # 转换深度图
        if depth_msg.encoding == '16UC1':
            depth_cv2 = bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            # 转换为米单位的float32
            depth_cv2 = depth_cv2.astype(np.float32) 
        elif depth_msg.encoding == '32FC1':
            depth_cv2 = bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        else:
            rospy.logerr(f"不支持的深度图格式: {depth_msg.encoding}")
            return
        
        # 确保内存连续
        depth_cv2 = np.ascontiguousarray(depth_cv2)
        
        # 检查深度图属性
        # print(f"深度图尺寸: {depth_cv2.shape}, 类型: {depth_cv2.dtype}, 范围: {np.min(depth_cv2)}-{np.max(depth_cv2)}米")
        
        # 生成点云
        t_pc = time.perf_counter()
        point_cloud_data = depth_to_pointcloud(depth_cv2)
        print(f"Generate PC: {(time.perf_counter()-t_pc)*1000:.1f}ms")
        # print(f"点云尺寸: {point_cloud_data.shape}")
        
        # 准备观测数据
        obs_data = {
            "agent_pos": np.array(agent_pos.joint_pos, dtype=np.float64),
            "point_cloud": point_cloud_data.astype(np.float32),
        }
        
        # 放入缓冲区
        obs_ring_buffer.put(obs_data, wait=False)
        # if success:
        #     print(f"成功写入缓冲区! 当前计数: {obs_ring_buffer.count}")
        #     success_count = success_count + 1   
        #     print(f"Success:{success_count}")
        # else:
        #     print("缓冲区写入失败 (可能已满或关闭)")
        #     fail_count = fail_count + 1   
        #     print(f"Fail:{fail_count}")
            
    except Exception as e:
        print(f"回调函数错误: {e}")
        import traceback
        traceback.print_exc()


def transform(data, video_capture_resolution=(640, 480), obs_image_resolution=(160, 120)):
    color_tf = get_image_transform(
                input_res=video_capture_resolution,
                output_res=obs_image_resolution,
                # obs output rgb
                bgr_to_rgb=True,
            )
    
    tf_data = color_tf(data)
    return tf_data


def depth_to_pointcloud(depth_image_cv2):
    # print(f"深度数据形状: {depth_image_cv2.shape}, 数据类型: {depth_image_cv2.dtype}")
    
    # 1. 初始化点云生成器
    pc_generator = PointCloudGenerator(
        img_size=depth_image_cv2.shape[1]  # 使用深度图高度
    )
    # print("点云生成器初始化完成")
    
    # 检查深度图有效性
    # print(f"深度图范围: {np.min(depth_image_cv2)} - {np.max(depth_image_cv2)}")
    if np.all(depth_image_cv2 == 0):
        print("警告: 深度图全为零值")
        return None
    
    # 2. 生成点云
    # print("正在生成点云...")
    points = pc_generator.generateCroppedPointCloud(depth_data=depth_image_cv2)
    # print(f"生成点云形状: {points.shape if isinstance(points, np.ndarray) else '无效'}")
    
    if not isinstance(points, np.ndarray) or points.size == 0:
        raise ValueError("生成的点云为空")
    
    # 3. 预处理点云（裁剪+FPS）
    sampled_points = preprocess_point_cloud(points)
    # print(f"处理后点云形状: {sampled_points.shape}")

    # 4. 可视化（实时窗口）
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sampled_points[:, :3])  # 仅取XYZ坐标
    
    # 创建坐标系（红色-X，绿色-Y，蓝色-Z）
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05,  # 坐标系轴长度
        origin=[0, 0, 0]  # 坐标系原点
    )
    
    # 定义工作空间边界框（根据你的实际范围调整）
    WORK_SPACE = [
        [-0.14, 0.1],  # X轴范围
        [-0.03, 0.2],  # Y轴范围
        [0, 0.1]       # Z轴范围
    ]
    
    # # 创建边界框
    # min_bound = np.array([WORK_SPACE[0][0], WORK_SPACE[1][0], WORK_SPACE[2][0]])
    # max_bound = np.array([WORK_SPACE[0][1], WORK_SPACE[1][1], WORK_SPACE[2][1]])
    # bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    # bbox.color = (0, 1, 0)  # 绿色边界框
    
    # # 创建可视化窗口
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name="PointCloud with Workspace")
    
    # # 添加几何体
    # vis.add_geometry(pcd)
    # # vis.add_geometry(coord_frame)
    # vis.add_geometry(bbox)
    
    # # 设置背景色和点大小
    # render_opt = vis.get_render_option()
    # render_opt.background_color = np.array([0.5, 0.5, 0.5])  # 灰色背景
    # render_opt.point_size = 3.0  # 点云大小
    
    # # 运行可视化（阻塞式，窗口关闭后继续执行）
    # vis.run()
    # vis.destroy_window()
    
    return sampled_points

if __name__ == "__main__":
    main()
