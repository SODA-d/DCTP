import numpy as np
import gymnasium
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Dict
from collections import deque
from follower.planning import ResettablePlanner, PlannerConfig
from copy import deepcopy
import math
from direction_cpp import Helloworld as hw
import time
class PreprocessorConfig(PlannerConfig):
    network_input_radius: int = 5
    intrinsic_target_reward: float = 0.01


def follower_preprocessor(env, algo_config):
    env = wrap_preprocessors(env, algo_config.training_config.preprocessing)
    return env


def wrap_preprocessors(env, config: PreprocessorConfig, auto_reset=False):
    env = FollowerWrapper(env=env, config=config)
    env = CutObservationWrapper(env, target_observation_radius=config.network_input_radius)
    env = ConcatPositionalFeatures(env)

    if auto_reset:
        env = AutoResetWrapper(env)
    return env


class FollowerWrapper(ObservationWrapper):

    def __init__(self, env, config: PreprocessorConfig):
        super().__init__(env)
        self.num_agents=env.get_num_agents()
        self._cfg: PreprocessorConfig = config
        self.re_plan = ResettablePlanner(self._cfg,self.num_agents)
        self.prev_goals = None # 存储上一时刻的子目标
        self.intrinsic_reward = None
        self.previous_step_distance = None
        self.comparison_result = None
    @staticmethod
    def get_relative_xy(x, y, tx, ty, obs_radius):#计算临时目标距离智能体的相对坐标
        dx, dy = x - tx, y - ty
        if dx > obs_radius or dx < -obs_radius or dy > obs_radius or dy < -obs_radius:
            return None, None
        return obs_radius - dx, obs_radius - dy
    
    def observation(self, observations):
        obstacles = self.grid.get_obstacles(ignore_borders=True).astype(int).tolist()
        agents_xy = self.get_global_agents_xy()
        targets_xy = self.get_global_targets_xy()
        distances = [math.sqrt((target[0] - agent[0]) ** 2 + (target[1] - agent[1]) ** 2) for agent, target in zip(agents_xy, targets_xy)]
        if self.previous_step_distance is not None:
            self.comparison_result = [-1 if prev < dist else 1 if prev > dist else 0 for prev, dist in zip(self.previous_step_distance, distances)]
   
        # for agent_idx, obs_1 in enumerate(observations):
        #     communicate_target = []
        #     for i,obs_2 in enumerate(observations):
        #         target = deepcopy(obs_2['visibled_targets'][agent_idx])
        #         if not np.allclose(target, [0.0, 0.0]) and not np.allclose(target,np.array(agents_xy[agent_idx], dtype=float)):
        #             communicate_target.append(target)
        #     communicate_target = self.remove_duplicate_arrays(communicate_target)
        #     median_point_xy = [0,0]
        #     for j,communicate_target_i in enumerate(communicate_target):
        #         self.mark_direction(agents_xy[agent_idx],tuple(communicate_target_i),obs_1['agents'],median_point_xy)
        #         a_x = median_point_xy[0]
        #         b_y = median_point_xy[1]
        #         a_x,b_y = self.find_nearest_passable(obstacles,a_x-5,b_y-5)
        #         a_x = a_x+5
        #         b_y = b_y+5
        #         obs_1['edge_target_xy'] = (a_x-agents_xy[agent_idx][0]+obs_1['xy'][0],b_y-agents_xy[agent_idx][1]+obs_1['xy'][1])

        #     if obs_1['edge_target_xy'] == (0,0):
        #         obs_1['edge_target_xy'] = obs_1['xy']
        
        hw.process_observations(observations,agents_xy,obstacles)
        # print("执行成功")
        # Update cost penalties based on the current observations, independently for each agent.
        #基于当前观察结果（observations），为每个智能体独立更新代价惩罚（cost penalties）
        self.re_plan.update(observations,observations[1]['isReward_on_goal'])
        # temp_rewards = self.rewards[0]
        # self.rewards[0] = self.rewards[1]
        # self.rewards[1] = temp_rewards
        # Retrieve the shortest path to the global target for each agent.
        #为每个智能体获取到达全局目标的最短路径
        paths = self.re_plan.get_path()

        new_goals = []  # Initialize a list to store new goals for each agent.  初始化一个列表，用于存储每个智能体的新目标
        intrinsic_rewards = [0.0] * self.num_agents  # Initialize a list to store intrinsic rewards for each agent. 初始化一个列表，用于存储每个智能体的内在奖励（intrinsic rewards）。

        # Iterate through agents and their respective paths. 遍历每个智能体及其对应的路径。
        for k, path in enumerate(paths):
            obs = observations[k]
            # Check if there is no valid path available. 检查是否没有有效的路径可用。
            if path is None or len(path) == 0: 
                new_goals.append(obs['target_xy'])  # Use the target position as a new goal.将目标位置作为新的目标。
                path = []
            else:#如果路径不为空
                # Check if the agent reached their subgoal from its previous step 检查智能体是否从上一步到达了它的子目标。
                subgoal_achieved = self.prev_goals and obs['xy'] == self.prev_goals[k]#返回值为1则智能体已经到达了子目标
                # Assign an intrinsic reward if conditions are met, otherwise set it to 0.如果智能体k到达子目标，则为智能体分配内在奖励，否则将奖励设置为 0。
                # intrinsic_rewards[k] = self._cfg.intrinsic_target_reward if subgoal_achieved else 0.0
                intrinsic_rewards[k] = 0.3 if subgoal_achieved else 0.0
                #如果智能体离动态目标更近了，则给奖励在加1
                if self.comparison_result is not None:
                    if self.comparison_result[k] == 1 or obs['isReward_on_goal'][k]:
                        intrinsic_rewards[k] = intrinsic_rewards[k] + 0.7
                        if obs['isReward_on_goal'][k]:
                            intrinsic_rewards[k] = intrinsic_rewards[k] + 0.3
                    elif self.comparison_result[k] == -1 and obs['isReward_on_goal'][k] == 0:
                        intrinsic_rewards[k] = intrinsic_rewards[k] - 0.7
                    elif self.comparison_result[k] == 0:
                        intrinsic_rewards[k] = intrinsic_rewards[k] - 0.5
                # # Select a new target point.将智能体的新目标存入new_goals[k]中
                if observations[k]['agents'][4,5] == 1 and  observations[k]['agents'][3,5] == 1:
                    intrinsic_rewards[k] = intrinsic_rewards[k] - 2
                if observations[k]['agents'][6,5] == 1 and  observations[k]['agents'][7,5] == 1:
                    intrinsic_rewards[k] = intrinsic_rewards[k] - 2
                if observations[k]['agents'][5,4] == 1 and  observations[k]['agents'][5,3] == 1:
                    intrinsic_rewards[k] = intrinsic_rewards[k] - 2
                if observations[k]['agents'][5,6] == 1 and  observations[k]['agents'][5,7] == 1:
                    intrinsic_rewards[k] = intrinsic_rewards[k] - 2
                new_goals.append(path[1])

            # Set obstacle values to -1.0 in the observation.
            obs['obstacles'][obs['obstacles'] > 0] *= -1

            # Adding path to the observation, setting path values to +1.0. 将路径添加到局部观察中，并将路径上的值设置为 +1.0。
            r = obs['obstacles'].shape[0] // 2   #计算观察空间的半径（r），即障碍物矩阵的中心。
            for idx, (gx, gy) in enumerate(path):
                x, y = self.get_relative_xy(*obs['xy'], gx, gy, r)
                if x is not None and y is not None:
                    obs['obstacles'][x, y] = 1.0#不停循环
                else:
                    break
            # print(obs['obstacles'])
        # Update the previous goals and intrinsic rewards for the next step.
        self.prev_goals = new_goals#将上面算法计算的新目标存下来下次移动做参考
        self.intrinsic_reward = intrinsic_rewards#将上一步的奖励保存下来，为下次移动做参考
        self.previous_step_distance = distances

        return observations#代码看到这

    def get_intrinsic_rewards(self, reward):
        for agent_idx, r in enumerate(reward):
            reward[agent_idx] = self.intrinsic_reward[agent_idx]
        return reward

    def step(self, action):
        observation, reward, done, tr, info = self.env.step(action)
        return self.observation(observation), self.get_intrinsic_rewards(reward), done, tr, info

    def reset_state(self):
        self.re_plan.reset_states()
        # print(self.get_global_obstacles())
        self.re_plan._agent.add_grid_obstacles(self.get_global_obstacles(), self.get_global_agents_xy())

        self.prev_goals = None
        self.intrinsic_reward = None

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self.reset_state()
        return self.observation(observations), infos


class CutObservationWrapper(ObservationWrapper): # 裁剪智能体的观察空间
    def __init__(self, env, target_observation_radius):
        super().__init__(env)
        self._target_obs_radius = target_observation_radius # 目标观察半径
        self._initial_obs_radius = self.env.observation_space['obstacles'].shape[0] // 2 # 初始环境中的原始观测半径

        for key, value in self.observation_space.items():
            d = self._initial_obs_radius * 2 + 1 # 原始观测范围的大小
            if value.shape == (d, d):
                r = self._target_obs_radius
                self.observation_space[key] = Box(0.0, 1.0, shape=(r * 2 + 1, r * 2 + 1)) # 缩小观测空间，仅保留目标半径 r 的区域

    def observation(self, observations):# 对每个智能体的观察进行裁剪，只保留目标半径 r 的区域
        tr = self._target_obs_radius
        ir = self._initial_obs_radius
        d = ir * 2 + 1

        for obs in observations:
            for key, value in obs.items():
                if hasattr(value, 'shape') and value.shape == (d, d):
                    obs[key] = value[ir - tr:ir + tr + 1, ir - tr:ir + tr + 1]

        return observations


class ConcatPositionalFeatures(ObservationWrapper): # 将智能体的位置信息与环境信息进行拼接

    def __init__(self, env):
        super().__init__(env)
        self.to_concat = []

        observation_space = Dict()
        full_size = self.env.observation_space['obstacles'].shape[0] # 观测空间大小

        for key, value in self.observation_space.items():
            if value.shape == (full_size, full_size):
            # if value.shape == (full_size, full_size) and key != 'other_target_xy':
                self.to_concat.append(key)
            else:
                observation_space[key] = value

        obs_shape = (len(self.to_concat), full_size, full_size)
        observation_space['obs'] = Box(0.0, 1.0, shape=obs_shape,dtype=np.float32)
        self.to_concat.sort(key=self.key_comparator)
        self.observation_space = observation_space
    def mark_direction(self,agents_xy, target_xy, obs1, obs_edge_target_xy, obs_xy,median_point_xy):
    # 解析坐标
        agent_x, agent_y = agents_xy
        target_x, target_y = target_xy
    
    # 计算目标在网格中的相对坐标
        dx_abs = target_x - agent_x
        dy_abs = target_y - agent_y
        grid_x = 5 + dx_abs
        grid_y = 5 + dy_abs
    
    # 计算相对于网格中心(5,5)的偏移量
        dx = grid_x - 5
        dy = grid_y - 5
    
    # 处理重合情况
        if dx == 0 and dy == 0:
            return 
    
    # 计算边缘交点
        if dx == 0:  # 垂直方向
            edge_y = 10 if dy > 0 else 0
            edge_x = 5
        elif dy == 0:  # 水平方向
            edge_x = 10 if dx > 0 else 0
            edge_y = 5
        else:
        # 计算步长
            tx = (10.0-5)/dx if dx > 0 else (0.0-5)/dx
            ty = (10.0-5)/dy if dy > 0 else (0.0-5)/dy
        
        # 取有效最小步长
            t = min(tx, ty) if (tx > 0 and ty > 0) else max(tx, ty)
        
        # 计算交点坐标
            edge_x = round(5 + dx * t)
            edge_y = round(5 + dy * t)
        
        # 确保在边界
            edge_x = max(0, min(10, edge_x))
            edge_y = max(0, min(10, edge_y))
    # 标记最外圈方向
        obs1[edge_x][edge_y] = 2    
        obs_edge_target_xy[0] = edge_x-5+obs_xy[0]
        obs_edge_target_xy[1] = edge_y-5+obs_xy[1]
        median_point_xy.append(obs_edge_target_xy)

    def remove_duplicate_arrays(self, lst, atol=1e-8):
        unique_arrays = []
        for arr in lst:
            if not any(np.allclose(arr, unique, atol=atol) for unique in unique_arrays):
                unique_arrays.append(arr)
        return unique_arrays
    
    def observation(self, observations):
        for agent_idx, obs in enumerate(observations):
            main_obs = np.concatenate([obs[key][None] for key in self.to_concat])
            for key in self.to_concat:
                del obs[key]

            for key in obs:
                obs[key] = np.array(obs[key], dtype=np.float32)
            observations[agent_idx]['obs'] = main_obs.astype(np.float32)
            
            # communicate_target = [item['visibled_targets'][agent_idx] for item in observations if not np.allclose(item['visibled_targets'][agent_idx], [0.0,0.0])]
            # communicate_target = self.remove_duplicate_arrays(communicate_target)
            # median_point_xy = []
            # for i,communicate_target_i in enumerate(communicate_target):
            #     self.mark_direction(agents_xy[agent_idx],tuple(communicate_target_i),obs['obs'][1],obs['edge_target_xy'],obs['xy'],median_point_xy) 
            # if len(median_point_xy) > 0:
            #     obs['edge_target_xy'] = np.round(np.median(median_point_xy, axis=0))

        return observations

    @staticmethod
    def key_comparator(x):
        if x == 'obstacles':
            return '0_' + x
        elif 'agents' in x:
            return '1_' + x
        return '2_' + x


class AutoResetWrapper(gymnasium.Wrapper): # 用于所有智能体都完成任务或环境被截断时，自动重置环境
    def step(self, action):
        observations, rewards, terminated, truncated, infos = self.env.step(action)
        if all(terminated) or all(truncated):
            observations, _ = self.env.reset()
        return observations, rewards, terminated, truncated, infos
