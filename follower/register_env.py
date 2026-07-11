import numpy as np
from sample_factory.utils.typing import Env
from sample_factory.envs.env_utils import register_env

from env.create_env import create_env_base
from follower.training_config import Experiment

import gymnasium
from follower.training_config import Environment
from follower.preprocessing import PreprocessorConfig, wrap_preprocessors


def create_env(environment_cfg: Environment, preprocessing_cfg: PreprocessorConfig):
    env = create_env_base(environment_cfg) # 创建环境所需的基础，具体看这几个类：ProvideGlobalObstacles、MultiMapWrapper、RuntimeMetricWrapper
    env = wrap_preprocessors(env, config=preprocessing_cfg, auto_reset=True)
    return env


class MultiEnv(gymnasium.Wrapper): # 创建多个环境
    def __init__(self, env_cfg: Environment, preprocessing_cfg: PreprocessorConfig): # 创建实例时初始化__init__创建环境
        if env_cfg.target_num_agents is None:
            self.envs = [create_env(env_cfg, preprocessing_cfg)]
        else:
            assert env_cfg.target_num_agents % env_cfg.grid_config.num_agents == 0, \
                "Target num follower must be divisible by num agents"
            num_envs = env_cfg.target_num_agents // env_cfg.grid_config.num_agents
            self.envs = [create_env(env_cfg, preprocessing_cfg) for _ in range(num_envs)]

        super().__init__(self.envs[0])

    def step(self, actions):
        obs, rewards, dones, truncated, infos = [], [], [], [], []
        last_agents = 0
        for env in self.envs:
            env_num_agents = env.get_num_agents()
            action = actions[last_agents: last_agents + env_num_agents]
            last_agents = last_agents + env_num_agents
            o, r, d, t, i = env.step(action)
            obs += o
            rewards += r
            dones += d
            truncated += t
            infos += i
        return obs, rewards, dones, truncated, infos

    def reset(self, seed, **kwargs):
        obs = []
        for idx, env in enumerate(self.envs):
            inner_seed = seed + idx
            o, _ = env.reset(seed=inner_seed, **kwargs)
            obs += o
        return obs, {}

    def sample_actions(self):
        actions = []
        for env in self.envs:
            actions += list(env.sample_actions())
        return np.array(actions)

    @property
    def num_agents(self):
        return sum([env.get_num_agents() for env in self.envs])

    def render(self):
        for env in self.envs:
            env.render()


def make_env(full_env_name, cfg=None, env_config=None, render_mode=None):
    p_config = Experiment(**vars(cfg))
    environment_config = p_config.environment  # 环境配置
    preprocessing_config = p_config.preprocessing # 预处理配置 use_static_cost，use_dynamic_cost，reset_dynamic_cost，network_input_radius，intrinsic_target_reward
    # todo make this code simpler

    if environment_config.agent_bins is not None and environment_config.target_num_agents is not None:
        if environment_config.env_id is None:
            num_agents = environment_config.agent_bins[0]#设定智能体数量为128
        else:
            num_agents = environment_config.agent_bins[environment_config.env_id % len(environment_config.agent_bins)] #设定智能体数量至少为128。env_id参数可能是有多个env，给每个env分配一个id号？
        environment_config.grid_config.num_agents = num_agents #更新环境智能体数量

        return MultiEnv(environment_config, preprocessing_config)
    return create_env(environment_config, preprocessing_config)


class CustomEnv:
    def make_env(self, env_name, cfg, env_config, render_mode) -> Env:
        return make_env(env_name, cfg, env_config, render_mode)


def register_pogema_envs(env_name):
    env_factory = CustomEnv()
    register_env(env_name, env_factory.make_env)


def register_custom_components(env_name):
    register_pogema_envs(env_name)
