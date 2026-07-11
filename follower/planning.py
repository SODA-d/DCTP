from pogema import GridConfig

# noinspection PyUnresolvedReferences
import cppimport.import_hook
# noinspection PyUnresolvedReferences
from follower_cpp.planner import planner
# try:
#     from static_planning import planner
# except Exception as e:
#     from follower.static_planning import planner


from pydantic import BaseModel

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class PlannerConfig(BaseModel):#静态成本，动态成本。因为地图会变化，所以动态成本需要更新。配置文件。
    use_static_cost: bool = True
    use_dynamic_cost: bool = True
    reset_dynamic_cost: bool = True


class Planner:#启发式路径规划实现,计算惩罚矩阵也在这个类里面
    def __init__(self, cfg: PlannerConfig,num_agents):
        self.planner = None
        self.obstacles = None
        self.starts = None
        self.cfg = cfg

    def add_grid_obstacles(self, obstacles, starts):
        self.obstacles = obstacles # 存储的是全局障碍物 75*75 多了观测半径5
        self.starts = starts#静态路径的起点，也就是智能体的起始位置 128个智能体的位置,基于75x75网格的绝对路径
        self.planner = None#障碍物改变后，路径规划器也要更新

    def update(self, obs,isReward_on_goal): # obs包含128个智能体的自己的观测空间，空间大小为11x11
        
        num_agents = len(obs)#通过包含所有智能体观测空间的长度获取智能体的数量
        obs_radius = len(obs[0]['obstacles']) // 2#获取观测空间的半径
            
        if self.planner is None:#如果规划器为空，则调用c++写的规划器代码为每一个智能体规划路径，使用planner数组存储每一个智能体的路径
            self.planner = [planner(self.obstacles, self.cfg.use_static_cost, self.cfg.use_dynamic_cost, self.cfg.reset_dynamic_cost) for _ in range(num_agents)]
            for i, p in enumerate(self.planner):
                p.set_abs_start(self.starts[i])#starts存储了每一个智能体的起点，为每个智能体设置起始点
            if self.cfg.use_static_cost:
                pen_calc = planner(self.obstacles, self.cfg.use_static_cost, self.cfg.use_dynamic_cost, self.cfg.reset_dynamic_cost)
                penalties = pen_calc.precompute_penalty_matrix(obs_radius)#计算惩罚矩阵
                for p in self.planner:
                    p.set_penalties(penalties)#将惩罚矩阵应用到所有智能体的路径规划。

        for k in range(num_agents): 
            if obs[k]['xy'] == obs[k]['target_xy']:#如果智能体的当前位置等于目标位置，则跳过当前智能体的路径规划
                continue
            obs[k]['agents'][obs_radius][obs_radius] = 0#暂时移除智能体的位置，避免路径规划时将其视为障碍物

            if abs(obs[k]['target_xy'][0]-obs[k]['xy'][0]) > 5 or abs(obs[k]['target_xy'][0]-obs[k]['xy'][0]) > 5:
                self.planner[k].update_occupations(obs[k]['agents'], (obs[k]['xy'][0] - obs_radius, obs[k]['xy'][1] - obs_radius), obs[k]['edge_target_xy'],isReward_on_goal[k])#更新k号智能体的动态障碍物信息
                obs[k]['agents'][obs_radius][obs_radius] = 1#恢复智能体位置
                self.planner[k].update_path(obs[k]['xy'],obs[k]['edge_target_xy'])#更新k号智能体的路径规划
            else:
                self.planner[k].update_occupations(obs[k]['agents'], (obs[k]['xy'][0] - obs_radius, obs[k]['xy'][1] - obs_radius), obs[k]['target_xy'],isReward_on_goal[k])#更新k号智能体的动态障碍物信息
                obs[k]['agents'][obs_radius][obs_radius] = 1#恢复智能体位置
                self.planner[k].update_path(obs[k]['xy'],obs[k]['target_xy'])
        # # print(f"Elapsed time: {elapsed_time} seconds")

    def get_path(self):
        results = []
        for idx in range(len(self.planner)):
            results.append(self.planner[idx].get_path())
        return results


class ResettablePlanner:#agent发生变化（即地图发生变化时）调用路径规划器规划一条全局路径
    def __init__(self, cfg: PlannerConfig,num_agents):
        self._cfg = cfg
        self._agent = None
        self.num_agents = num_agents
    def update(self, observations,isReward_on_goal):#地图发生变化时，可以调用这个函数更新路径规划器的状态
        return self._agent.update(observations,isReward_on_goal)

    def get_path(self):
        return self._agent.get_path()

    def reset_states(self, ):
        self._agent = Planner(self._cfg,self.num_agents)

def main():
    from pogema import pogema_v0, GridConfig
    import gymnasium
    grid = """
        #..........
        ...........
        ...........
        ...........
        ...........
        #..........
        ...........
        ..#........
        ...........
        ...........
        ...........
            """

# 定义地图大小和智能体数量
    grid_config = GridConfig(
        map=grid,
        num_agents=3,  # 智能体数量
        size=10,       # 地图大小
        targets_xy=[(1,1), (2,2), (3,3)],
        agents_xy=[(4,2),(5,3),(9,4)],
        obs_radius=2,
        observation_type='POMAPF',
        on_target='restart',
        persistent=True ,
        collision_system='block_both'
    )
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

        def get_targets_xy_relative(self):
            return self.grid.get_targets_xy_relative()
    
        def get_grid_config(self):
            return self.grid.get_grid_config()
    class PreprocessorConfig(PlannerConfig):
        network_input_radius: int = 5
        intrinsic_target_reward: float = 0.01
        use_static_cost: bool = True
        use_dynamic_cost: bool = True
        reset_dynamic_cost: bool = True
# 初始化环境    
    env = pogema_v0(grid_config=grid_config)
    obs, info = env.reset()
    env=ProvideGlobalObstacles(env)
    env.render()
    # obs, reward, terminated, truncated, info  =env.step([3,3,3])
    # print(env.get_global_targets_xy())
    # print(env.get_global_targets_absolute_grid())
    plan = Planner(PreprocessorConfig())
    plan.add_grid_obstacles(env.get_global_obstacles(),env.get_global_agents_xy())
    plan.update(obs,[(3,2),(3,2),(3,2)])
    a=plan.get_path()
    print(a)
if __name__ == '__main__':
    main()