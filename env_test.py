# -*- coding: utf-8 -*-
# @Time : 2022/5/17 上午10:44
# @Author :  wangshulei
# @FileName: env_test.py
# @Software: PyCharm
from RL_algorithm_package.s_maddpg.mpe.environment import MultiAgentEnv
import RL_algorithm_package.s_maddpg.mpe.scenarios as scenarios
from RL_algorithm_package.s_maddpg.utils import space_n_to_shape_n
import numpy as np


class mpe_env:
    def __init__(self,
                 mpe_env_name,
                 seed):
        self.mpe_env_name = mpe_env_name
        self.seed = seed
        self.mpe_env = self.env_init()

    def env_init(self):
        scenario = scenarios.load(self.mpe_env_name + '.py').Scenario()
        # create world
        world = scenario.make_world()
        # create multiagent environment
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            area_callback=scenario.calculate_area, info_callback=None,
                            shared_viewer=True)
        env.seed(self.seed)
        return env

    def get_space(self):
        obs_shape_n = space_n_to_shape_n(self.mpe_env.observation_space)
        act_shape_n = space_n_to_shape_n(self.mpe_env.action_space)
        return obs_shape_n[0][0], act_shape_n[0][0]

    def get_agent_number(self):
        return self.mpe_env.n


if __name__ == "__main__":
    world = mpe_env('simple_search', 65535)
    world.mpe_env.reset()
    world.mpe_env.render()
    for n_games in range(15):
        action_n = [np.array([0, 0, 1, 1, 0]), np.array([0, 0, 0, 1, 0]), np.array([0, 0, 1, 0, 0])]
        new_obs_n, reward_n, done_n, info_n = world.mpe_env.step(action_n)
        world.mpe_env.render()
    print("over")
