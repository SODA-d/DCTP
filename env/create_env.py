import time

import numpy as np
from pogema import AnimationConfig, AnimationMonitor

from pogema import pogema_v0

from pogema.wrappers.metrics import AgentsDensityWrapper

from follower.training_config import Environment

import gymnasium
import re
from copy import deepcopy
from pogema import GridConfig

from env.custom_maps import MAPS_REGISTRY
from follower.preprocessing import wrap_preprocessors, PreprocessorConfig


class ProvideGlobalObstacles(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def get_global_obstacles(self):
        return self.grid.get_obstacles().astype(int).tolist()
    
    def get_global_targets_absolute_grid(self):
        return self.grid.get_global_targets_absolute_grid()

    def get_global_agents_xy(self):
        return self.grid.get_agents_xy()
    
    def get_global_targets_xy(self):
        return self.grid.get_targets_xy()
    
    def get_targets_xy_relative(self):
        return self.grid.get_targets_xy_relative()
    
    def get_state(self):
        return self.grid.get_state()
    
    def get_agents_xy_relative(self):
        return self.grid.get_agents_xy_relative()
    
    def get_grid_config(self):
         return self.grid.get_grid_config()


def create_env_base(config: Environment):
    env = pogema_v0(grid_config=config.grid_config)
    env = ProvideGlobalObstacles(env)#给环境添加方法：1、获取地图上所有智能体的位置（返回[(),()....]）2、获取地图上所有障碍物的位置，以二维数组的形式返回
    # env = AgentsDensityWrapper(env)
    if config.use_maps:
        env = MultiMapWrapper(env)
    if config.with_animation:
        env = AnimationMonitor(env, AnimationConfig(directory='renders', egocentric_idx=None))

    # adding runtime metrics
    env = RuntimeMetricWrapper(env)

    return env


class RuntimeMetricWrapper(gymnasium.Wrapper):#运行时间计算
    def __init__(self, env):
        super().__init__(env)
        self._start_time = None # 记录开始时间
        self._env_step_time = None # 记录环境step时间总和
           
    def step(self, actions): # 执行环境一步，并记录时间
        env_step_start = time.monotonic() # 存储开始时间
        observations, rewards, terminated, truncated, infos = self.env.step(actions) # 执行环境动作，terminated表示代理是否到达终点，truncated表示代理是否超出最大步数
        env_step_end = time.monotonic() # 存储结束时间
        self._env_step_time += env_step_end - env_step_start # 累加执行环境动作的时间
        if all(terminated) or all(truncated): # 如果所有智能体都到达终点或者超出最大步数
            final_time = time.monotonic() - self._start_time - self._env_step_time # 最终总和时间要排除step方法计算的时间
            if 'metrics' not in infos[0]:
                infos[0]['metrics'] = {}
            infos[0]['metrics'].update(runtime=final_time) # 把计算出的运行时间存入 infos
        return observations, rewards, terminated, truncated, infos

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs) # 这里的reset会调用MultiMapWrapper的reset
        self._start_time = time.monotonic()
        self._env_step_time = 0.0
        return obs


class MultiMapWrapper(gymnasium.Wrapper): # MultiMapWrapper 仅负责选择地图，环境初始化的实际过程仍由 self.env 负责。
    def __init__(self, env):
        super().__init__(env)
        self._configs = []
        self._rnd = np.random.default_rng(self.grid_config.seed) # 给定随机数生成一个随机数
        pattern = self.grid_config.map_name

        if pattern:
            for map_name in sorted(MAPS_REGISTRY):#遍历所有地图选择符合正则表达式地图
                if re.match(pattern, map_name):
                    cfg = deepcopy(self.grid_config)
                    cfg.map = MAPS_REGISTRY[map_name]
                    cfg.map_name = map_name
                    cfg = GridConfig(**cfg.dict())
                    self._configs.append(cfg)#将所有匹配的地图及其配置加入_configs，便于后续随机选择地图
                    # print(cfg.map_name)
            if not self._configs:#如果没有匹配的地图则抛出错误
                raise KeyError(f"No map matching: {pattern}")

    def reset(self, seed=None, **kwargs):
        self._rnd = np.random.default_rng(seed)
        if self._configs is not None and len(self._configs) >= 1:
            map_idx = self._rnd.integers(0, len(self._configs)) # 随机选择地图索引
            cfg = deepcopy(self._configs[map_idx])  # 复制选中的地图配置
            self.env.unwrapped.grid_config = cfg# 更新环境的地图配置
            self.env.unwrapped.grid_config.seed = seed # 更新环境的随机种子
        return self.env.reset(seed=seed, **kwargs)


def main():
    env = create_env_base(config=Environment())
    env = wrap_preprocessors(env, config=PreprocessorConfig())
    env.reset()
    env.render()


if __name__ == '__main__':
    main()
