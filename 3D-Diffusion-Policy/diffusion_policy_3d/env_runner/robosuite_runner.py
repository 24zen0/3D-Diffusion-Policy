import wandb
import numpy as np
import torch
import collections
import tqdm
from termcolor import cprint
from diffusion_policy_3d.env import RobosuiteEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util


class RobosuiteRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 task_name='Lift',
                 robot_name='Panda',
                 gripper_name='PandaGripper',
                 num_points=1024,
                 mask_robot=True,
                 use_point_crop=True,
                 n_train=20,
                 max_steps=75,
                 n_obs_steps=2,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 log_video=False,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        self.log_video = log_video
        
        steps_per_render = max(10 // fps, 1)
        if log_video:
            def env_fn(is_test=True):
                return MultiStepWrapper(
                    SimpleVideoRecordingWrapper(RobosuiteEnv(
                        task_name=task_name,
                        robot_name=robot_name,
                        gripper_name=gripper_name,
                        num_points=num_points,
                        mask_robot=mask_robot,
                        use_point_crop=use_point_crop,
                    )),
                    n_obs_steps=n_obs_steps,
                    n_action_steps=n_action_steps,
                    max_episode_steps=max_steps,
                    reward_agg_method='sum',
                )
        else:
            def env_fn(is_test=True):
                return MultiStepWrapper(
                    RobosuiteEnv(
                        task_name=task_name,
                        robot_name=robot_name,
                        gripper_name=gripper_name,
                        num_points=num_points,
                        mask_robot=mask_robot,
                        use_point_crop=use_point_crop,
                    ),
                    n_obs_steps=n_obs_steps,
                    n_action_steps=n_action_steps,
                    max_episode_steps=max_steps,
                    reward_agg_method='sum',
                )

        self.env_train = env_fn(is_test=False)
        self.episode_train = n_train

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_train = logger_util.LargestKRecorder(K=3)
        self.logger_util_train10 = logger_util.LargestKRecorder(K=5)

        
    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env_train = self.env_train

        all_returns_train = []
        all_success_rates_train = []


        ##############################
        # train env loop
        for episode_id in tqdm.tqdm(range(self.episode_train), desc=f"Robosuite {self.task_name} Train Env",leave=False, mininterval=self.tqdm_interval_sec):
            # start rollout
            obs = env_train.reset()

            policy.reset()

            done = False
            reward_sum = 0.
            i = 0
            # print(f"Max Steps:{self.max_steps}")
            # print("实际 horizon:", env_train.env.horizon)
            for step_id in range(self.max_steps):
                # i = i + 1
                # create obs dict
                np_obs_dict = dict(obs)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # run policy
                with torch.no_grad():
                    # add batch dim to match. (1,2,3,84,84)
                    # and multiply by 255, align with all envs
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)[..., :3]
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)


                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)

                # step env
                obs, reward, done, info = env_train.step(action)
                reward_sum += reward
                done = np.all(done)
                success = env_train.is_success()
                done = done or success

                if done:
                    # 检查终止原因
                    if info.get("success", False):
                        print("Episode 成功终止：任务完成！")
                    else:
                        print("Episode 因最大步数(horizon)耗尽终止")
                        print(f"Step:{step_id}")
                        # print(info)
                        # print(action)
                        # env_train.render()
                    break

            all_returns_train.append(reward_sum)
            all_success_rates_train.append(env_train.is_success())

       

        SR_mean_train = np.mean(all_success_rates_train)
        returns_mean_train = np.mean(all_returns_train)

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        log_data
        log_data['mean_success_rates_train'] = SR_mean_train
        log_data['mean_returns_train'] = returns_mean_train

        log_data['test_mean_score'] = SR_mean_train

        self.logger_util_train.record(SR_mean_train)
        self.logger_util_train10.record(SR_mean_train)

        log_data['SR_train_L3'] = self.logger_util_train.average_of_largest_K()
        log_data['SR_train_L5'] = self.logger_util_train10.average_of_largest_K()
        

        cprint( f"Mean SR train: {SR_mean_train:.3f}", 'green')
        
        if self.log_video:
            # visualize sim
            videos_train = env_train.env.get_video()

            if len(videos_train.shape) == 5:
                videos_train = videos_train[:, 0]
            sim_video_train = wandb.Video(videos_train, fps=self.fps, format="mp4")
            log_data[f'sim_video_train'] = sim_video_train

            # clear out video buffer
            _ = env_train.reset()
            videos_train = None
            del env_train

        return log_data
